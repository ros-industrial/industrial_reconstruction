#include "image_buffer.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace industrial_reconstruction {

ImageBuffer::ImageBuffer(size_t max_size) 
    : max_size_(max_size), compression_enabled_(false) {
}

void ImageBuffer::push(const cv::Mat& depth_image, const cv::Mat& color_image, 
                      std::chrono::time_point<std::chrono::high_resolution_clock> timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Remove oldest items if buffer is full
    while (buffer_.size() >= max_size_) {
        buffer_.pop();
    }
    
    ImageData data(depth_image, color_image, timestamp);
    
    if (compression_enabled_) {
        compress_image(data.depth_image);
        compress_image(data.color_image);
    }
    
    buffer_.push(std::move(data));
    condition_.notify_one();
}

bool ImageBuffer::pop(ImageData& data, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                           [this] { return !buffer_.empty(); })) {
        data = std::move(buffer_.front());
        buffer_.pop();
        
        if (compression_enabled_) {
            decompress_image(data.depth_image);
            decompress_image(data.color_image);
        }
        
        return true;
    }
    return false;
}

bool ImageBuffer::try_pop(ImageData& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (buffer_.empty()) {
        return false;
    }
    
    data = std::move(buffer_.front());
    buffer_.pop();
    
    if (compression_enabled_) {
        decompress_image(data.depth_image);
        decompress_image(data.color_image);
    }
    
    return true;
}

void ImageBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<ImageData> empty;
    buffer_.swap(empty);
}

size_t ImageBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
}

bool ImageBuffer::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty();
}

bool ImageBuffer::full() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size() >= max_size_;
}

void ImageBuffer::set_max_size(size_t max_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_size_ = max_size;
    
    // Remove excess items if new size is smaller
    while (buffer_.size() > max_size_) {
        buffer_.pop();
    }
}

void ImageBuffer::enable_compression(bool enable) {
    std::lock_guard<std::mutex> lock(mutex_);
    compression_enabled_ = enable;
}

void ImageBuffer::compress_image(cv::Mat& image) {
    if (image.type() == CV_16UC1) {
        // For depth images, use lossless compression
        std::vector<uchar> buffer;
        cv::imencode(".png", image, buffer);
        image = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    } else {
        // For color images, use JPEG compression
        std::vector<uchar> buffer;
        cv::imencode(".jpg", image, buffer, {cv::IMWRITE_JPEG_QUALITY, 95});
        image = cv::imdecode(buffer, cv::IMREAD_COLOR);
    }
}

void ImageBuffer::decompress_image(cv::Mat& image) {
    // Images are already decompressed when loaded from buffer
    // This is a placeholder for any additional decompression logic
}

} // namespace industrial_reconstruction

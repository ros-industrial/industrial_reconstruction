#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace industrial_reconstruction {

struct ImageData {
    cv::Mat depth_image;
    cv::Mat color_image;
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    
    ImageData() = default;
    ImageData(const cv::Mat& depth, const cv::Mat& color, 
              std::chrono::time_point<std::chrono::high_resolution_clock> ts)
        : depth_image(depth.clone()), color_image(color.clone()), timestamp(ts) {}
};

class ImageBuffer {
public:
    ImageBuffer(size_t max_size = 100);
    ~ImageBuffer() = default;
    
    // Thread-safe operations
    void push(const cv::Mat& depth_image, const cv::Mat& color_image, 
              std::chrono::time_point<std::chrono::high_resolution_clock> timestamp);
    bool pop(ImageData& data, int timeout_ms = 100);
    bool try_pop(ImageData& data);
    
    // Buffer management
    void clear();
    size_t size() const;
    bool empty() const;
    bool full() const;
    
    // Memory optimization
    void set_max_size(size_t max_size);
    void enable_compression(bool enable);
    
private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::queue<ImageData> buffer_;
    size_t max_size_;
    bool compression_enabled_;
    
    void compress_image(cv::Mat& image);
    void decompress_image(cv::Mat& image);
};

} // namespace industrial_reconstruction

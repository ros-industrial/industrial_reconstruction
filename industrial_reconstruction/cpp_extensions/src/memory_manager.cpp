#include "memory_manager.h"
#include <iostream>
#include <algorithm>

namespace industrial_reconstruction {

// Global memory manager implementation
std::unique_ptr<ImageMemoryManager> GlobalMemoryManager::instance_ = nullptr;
std::mutex GlobalMemoryManager::instance_mutex_;

ImageMemoryManager& GlobalMemoryManager::get_image_manager() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::make_unique<ImageMemoryManager>();
    }
    return *instance_;
}

void GlobalMemoryManager::initialize() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::make_unique<ImageMemoryManager>();
    }
}

void GlobalMemoryManager::cleanup() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    instance_.reset();
}

// ImageMemoryManager implementation
ImageMemoryManager::ImageMemoryManager() 
    : allocated_memory_(0), pool_memory_(0) {
}

cv::Mat ImageMemoryManager::allocate_image(int rows, int cols, int type) {
    cv::Mat image(rows, cols, type);
    size_t size = calculate_image_size(rows, cols, type);
    
    std::lock_guard<std::mutex> lock(mutex_);
    allocated_memory_ += size;
    track_allocation(image.data, size);
    
    return image;
}

void ImageMemoryManager::deallocate_image(cv::Mat& image) {
    if (image.empty()) return;
    
    size_t size = calculate_image_size(image.rows, image.cols, image.type());
    
    std::lock_guard<std::mutex> lock(mutex_);
    track_deallocation(image.data);
    allocated_memory_ -= size;
    
    image.release();
}

cv::Mat ImageMemoryManager::get_temp_depth_image() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (size_t i = 0; i < temp_buffer_used_.size(); ++i) {
        if (!temp_buffer_used_[i]) {
            temp_buffer_used_[i] = true;
            return temp_depth_buffers_[i].clone();
        }
    }
    
    // If no free buffer, create a new one
    cv::Mat new_buffer = cv::Mat::zeros(480, 640, CV_16UC1);
    temp_depth_buffers_.push_back(new_buffer);
    temp_buffer_used_.push_back(true);
    
    return new_buffer.clone();
}

cv::Mat ImageMemoryManager::get_temp_color_image() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (size_t i = 0; i < temp_color_buffers_.size(); ++i) {
        if (!temp_buffer_used_[i]) {
            temp_buffer_used_[i] = true;
            return temp_color_buffers_[i].clone();
        }
    }
    
    // If no free buffer, create a new one
    cv::Mat new_buffer = cv::Mat::zeros(480, 640, CV_8UC3);
    temp_color_buffers_.push_back(new_buffer);
    temp_buffer_used_.push_back(true);
    
    return new_buffer.clone();
}

void ImageMemoryManager::return_temp_image(cv::Mat& image) {
    if (image.empty()) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find the corresponding buffer and mark it as unused
    for (size_t i = 0; i < temp_depth_buffers_.size(); ++i) {
        if (image.data == temp_depth_buffers_[i].data) {
            temp_buffer_used_[i] = false;
            break;
        }
    }
    
    for (size_t i = 0; i < temp_color_buffers_.size(); ++i) {
        if (image.data == temp_color_buffers_[i].data) {
            temp_buffer_used_[i] = false;
            break;
        }
    }
    
    image.release();
}

size_t ImageMemoryManager::get_allocated_memory() const {
    return allocated_memory_.load();
}

size_t ImageMemoryManager::get_pool_memory() const {
    return pool_memory_.load();
}

void ImageMemoryManager::print_memory_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::cout << "Memory Statistics:" << std::endl;
    std::cout << "  Allocated Memory: " << allocated_memory_.load() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Pool Memory: " << pool_memory_.load() / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Depth Image Pool Size: " << depth_image_pool_.size() << std::endl;
    std::cout << "  Color Image Pool Size: " << color_image_pool_.size() << std::endl;
    std::cout << "  Temp Buffers: " << temp_depth_buffers_.size() + temp_color_buffers_.size() << std::endl;
}

void ImageMemoryManager::preallocate_buffers(size_t num_buffers, int rows, int cols) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    temp_depth_buffers_.reserve(num_buffers);
    temp_color_buffers_.reserve(num_buffers);
    temp_buffer_used_.reserve(num_buffers);
    
    for (size_t i = 0; i < num_buffers; ++i) {
        temp_depth_buffers_.emplace_back(rows, cols, CV_16UC1);
        temp_color_buffers_.emplace_back(rows, cols, CV_8UC3);
        temp_buffer_used_.push_back(false);
        
        pool_memory_ += calculate_image_size(rows, cols, CV_16UC1);
        pool_memory_ += calculate_image_size(rows, cols, CV_8UC3);
    }
}

void ImageMemoryManager::clear_unused_buffers() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Remove unused temporary buffers
    for (size_t i = temp_buffer_used_.size(); i > 0; --i) {
        if (!temp_buffer_used_[i - 1]) {
            size_t depth_size = calculate_image_size(
                temp_depth_buffers_[i - 1].rows, 
                temp_depth_buffers_[i - 1].cols, 
                temp_depth_buffers_[i - 1].type());
            size_t color_size = calculate_image_size(
                temp_color_buffers_[i - 1].rows, 
                temp_color_buffers_[i - 1].cols, 
                temp_color_buffers_[i - 1].type());
            
            pool_memory_ -= (depth_size + color_size);
            
            temp_depth_buffers_.erase(temp_depth_buffers_.begin() + i - 1);
            temp_color_buffers_.erase(temp_color_buffers_.begin() + i - 1);
            temp_buffer_used_.erase(temp_buffer_used_.begin() + i - 1);
        }
    }
}

size_t ImageMemoryManager::calculate_image_size(int rows, int cols, int type) const {
    int elem_size = CV_ELEM_SIZE(type);
    return static_cast<size_t>(rows * cols * elem_size);
}

void ImageMemoryManager::track_allocation(void* ptr, size_t size) {
    memory_map_[ptr] = size;
}

void ImageMemoryManager::track_deallocation(void* ptr) {
    memory_map_.erase(ptr);
}

// ScopedImageBuffer implementation
ScopedImageBuffer::ScopedImageBuffer(ImageMemoryManager& manager, int rows, int cols, int type)
    : manager_(manager), owns_image_(true) {
    image_ = manager_.allocate_image(rows, cols, type);
}

ScopedImageBuffer::~ScopedImageBuffer() {
    if (owns_image_) {
        manager_.deallocate_image(image_);
    }
}

ScopedImageBuffer::ScopedImageBuffer(ScopedImageBuffer&& other) noexcept
    : manager_(other.manager_), image_(std::move(other.image_)), owns_image_(other.owns_image_) {
    other.owns_image_ = false;
}

ScopedImageBuffer& ScopedImageBuffer::operator=(ScopedImageBuffer&& other) noexcept {
    if (this != &other) {
        if (owns_image_) {
            manager_.deallocate_image(image_);
        }
        
        // Don't copy the manager reference, just move the image and ownership
        image_ = std::move(other.image_);
        owns_image_ = other.owns_image_;
        other.owns_image_ = false;
    }
    return *this;
}

} // namespace industrial_reconstruction

#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace industrial_reconstruction {

// Memory pool for frequently allocated objects
template<typename T>
class ObjectPool {
public:
    ObjectPool(size_t initial_size = 10) : pool_size_(initial_size) {
        for (size_t i = 0; i < initial_size; ++i) {
            pool_.push_back(std::make_unique<T>());
        }
    }
    
    std::unique_ptr<T> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.empty()) {
            return std::make_unique<T>();
        }
        auto obj = std::move(pool_.back());
        pool_.pop_back();
        return obj;
    }
    
    void release(std::unique_ptr<T> obj) {
        if (!obj) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.size() < pool_size_) {
            pool_.push_back(std::move(obj));
        }
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }
    
private:
    std::vector<std::unique_ptr<T>> pool_;
    mutable std::mutex mutex_;
    size_t pool_size_;
};

// Specialized memory manager for image processing
class ImageMemoryManager {
public:
    ImageMemoryManager();
    ~ImageMemoryManager() = default;
    
    // Image allocation with memory pooling
    cv::Mat allocate_image(int rows, int cols, int type);
    void deallocate_image(cv::Mat& image);
    
    // Pre-allocated image buffers
    cv::Mat get_temp_depth_image();
    cv::Mat get_temp_color_image();
    void return_temp_image(cv::Mat& image);
    
    // Memory statistics
    size_t get_allocated_memory() const;
    size_t get_pool_memory() const;
    void print_memory_stats() const;
    
    // Memory optimization
    void preallocate_buffers(size_t num_buffers, int rows, int cols);
    void clear_unused_buffers();
    
private:
    mutable std::mutex mutex_;
    std::atomic<size_t> allocated_memory_;
    std::atomic<size_t> pool_memory_;
    
    // Pools for different image types
    ObjectPool<cv::Mat> depth_image_pool_;
    ObjectPool<cv::Mat> color_image_pool_;
    
    // Pre-allocated temporary buffers
    std::vector<cv::Mat> temp_depth_buffers_;
    std::vector<cv::Mat> temp_color_buffers_;
    std::vector<bool> temp_buffer_used_;
    
    // Memory tracking
    std::unordered_map<void*, size_t> memory_map_;
    
    size_t calculate_image_size(int rows, int cols, int type) const;
    void track_allocation(void* ptr, size_t size);
    void track_deallocation(void* ptr);
};

// RAII wrapper for automatic memory management
class ScopedImageBuffer {
public:
    ScopedImageBuffer(ImageMemoryManager& manager, int rows, int cols, int type);
    ~ScopedImageBuffer();
    
    cv::Mat& get() { return image_; }
    const cv::Mat& get() const { return image_; }
    
    // Disable copy, allow move
    ScopedImageBuffer(const ScopedImageBuffer&) = delete;
    ScopedImageBuffer& operator=(const ScopedImageBuffer&) = delete;
    ScopedImageBuffer(ScopedImageBuffer&& other) noexcept;
    ScopedImageBuffer& operator=(ScopedImageBuffer&& other) noexcept;
    
private:
    ImageMemoryManager& manager_;
    cv::Mat image_;
    bool owns_image_;
};

// Global memory manager instance
class GlobalMemoryManager {
public:
    static ImageMemoryManager& get_image_manager();
    static void initialize();
    static void cleanup();
    
private:
    static std::unique_ptr<ImageMemoryManager> instance_;
    static std::mutex instance_mutex_;
};

} // namespace industrial_reconstruction

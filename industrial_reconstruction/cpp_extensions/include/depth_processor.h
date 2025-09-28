#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace industrial_reconstruction {

class DepthProcessor {
public:
    DepthProcessor();
    ~DepthProcessor() = default;
    
    // Main processing pipeline
    cv::Mat processDepthImage(const cv::Mat& raw_depth, const cv::Mat& color_image = cv::Mat());
    
    // Individual processing steps
    cv::Mat bilateralFilter(const cv::Mat& depth, int d = 5, double sigma_color = 50.0, double sigma_space = 50.0);
    cv::Mat medianFilter(const cv::Mat& depth, int kernel_size = 5);
    cv::Mat morphologicalFilter(const cv::Mat& depth, int kernel_size = 3);
    cv::Mat edgePreservingFilter(const cv::Mat& depth, int flags = 1, float sigma_s = 50.0f, float sigma_r = 0.4f);
    cv::Mat temporalFilter(const cv::Mat& current_depth, const cv::Mat& previous_depth, float alpha = 0.1f);
    
    // Noise reduction
    cv::Mat removeOutliers(const cv::Mat& depth, int min_neighbors = 8, double max_distance = 0.1);
    cv::Mat fillHoles(const cv::Mat& depth, int max_hole_size = 10);
    
    // Quality assessment
    double calculateDepthQuality(const cv::Mat& depth);
    bool isDepthValid(const cv::Mat& depth, double min_quality_threshold = 0.7);
    
    // Configuration
    void setBilateralParams(int d, double sigma_color, double sigma_space);
    void setMedianKernelSize(int size);
    void setMorphologicalKernelSize(int size);
    void setTemporalAlpha(float alpha);
    void setQualityThreshold(double threshold);
    
    // Advanced processing
    cv::Mat adaptiveProcessing(const cv::Mat& depth, const cv::Mat& color = cv::Mat());
    cv::Mat multiScaleProcessing(const cv::Mat& depth, int num_scales = 3);
    
private:
    // Processing parameters
    int bilateral_d_;
    double bilateral_sigma_color_;
    double bilateral_sigma_space_;
    int median_kernel_size_;
    int morphological_kernel_size_;
    float temporal_alpha_;
    double quality_threshold_;
    
    // Temporal filtering state
    cv::Mat previous_depth_;
    bool has_previous_depth_;
    
    // Helper functions
    cv::Mat createGaussianKernel(int size, double sigma);
    cv::Mat applyGaussianBlur(const cv::Mat& input, int kernel_size, double sigma);
    double calculateLocalVariance(const cv::Mat& depth, int x, int y, int window_size = 5);
    bool isEdgePixel(const cv::Mat& depth, int x, int y, double threshold = 0.05);
};

} // namespace industrial_reconstruction

#include "depth_processor.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <algorithm>
#include <cmath>

namespace industrial_reconstruction {

DepthProcessor::DepthProcessor() 
    : bilateral_d_(5), bilateral_sigma_color_(50.0), bilateral_sigma_space_(50.0),
      median_kernel_size_(5), morphological_kernel_size_(3), temporal_alpha_(0.1f),
      quality_threshold_(0.7), has_previous_depth_(false) {
}

cv::Mat DepthProcessor::processDepthImage(const cv::Mat& raw_depth, const cv::Mat& color_image) {
    if (raw_depth.empty()) {
        return cv::Mat();
    }
    
    cv::Mat processed_depth = raw_depth.clone();
    
    // Step 1: Remove outliers
    processed_depth = removeOutliers(processed_depth);
    
    // Step 2: Fill holes
    processed_depth = fillHoles(processed_depth);
    
    // Step 3: Apply bilateral filtering for edge preservation
    processed_depth = bilateralFilter(processed_depth);
    
    // Step 4: Apply median filtering for noise reduction
    processed_depth = medianFilter(processed_depth);
    
    // Step 5: Morphological operations for cleanup
    processed_depth = morphologicalFilter(processed_depth);
    
    // Step 6: Temporal filtering if previous frame available
    if (has_previous_depth_) {
        processed_depth = temporalFilter(processed_depth, previous_depth_);
    }
    
    // Update previous depth for next frame
    previous_depth_ = processed_depth.clone();
    has_previous_depth_ = true;
    
    return processed_depth;
}

cv::Mat DepthProcessor::bilateralFilter(const cv::Mat& depth, int d, double sigma_color, double sigma_space) {
    cv::Mat filtered;
    cv::bilateralFilter(depth, filtered, d, sigma_color, sigma_space);
    return filtered;
}

cv::Mat DepthProcessor::medianFilter(const cv::Mat& depth, int kernel_size) {
    cv::Mat filtered;
    cv::medianBlur(depth, filtered, kernel_size);
    return filtered;
}

cv::Mat DepthProcessor::morphologicalFilter(const cv::Mat& depth, int kernel_size) {
    cv::Mat filtered = depth.clone();
    
    // Create morphological kernel
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                             cv::Size(kernel_size, kernel_size));
    
    // Apply opening (erosion followed by dilation) to remove noise
    cv::morphologyEx(filtered, filtered, cv::MORPH_OPEN, kernel);
    
    // Apply closing (dilation followed by erosion) to fill small holes
    cv::morphologyEx(filtered, filtered, cv::MORPH_CLOSE, kernel);
    
    return filtered;
}

cv::Mat DepthProcessor::edgePreservingFilter(const cv::Mat& depth, int flags, float sigma_s, float sigma_r) {
    cv::Mat filtered;
    cv::edgePreservingFilter(depth, filtered, flags, sigma_s, sigma_r);
    return filtered;
}

cv::Mat DepthProcessor::temporalFilter(const cv::Mat& current_depth, const cv::Mat& previous_depth, float alpha) {
    if (previous_depth.empty() || current_depth.size() != previous_depth.size()) {
        return current_depth.clone();
    }
    
    cv::Mat filtered;
    cv::addWeighted(current_depth, alpha, previous_depth, 1.0f - alpha, 0.0, filtered);
    return filtered;
}

cv::Mat DepthProcessor::removeOutliers(const cv::Mat& depth, int min_neighbors, double max_distance) {
    cv::Mat filtered = depth.clone();
    cv::Mat mask = cv::Mat::zeros(depth.size(), CV_8UC1);
    
    // Create mask for valid depth values
    cv::Mat valid_mask = (depth > 0);
    
    // Apply statistical outlier removal
    for (int y = 1; y < depth.rows - 1; ++y) {
        for (int x = 1; x < depth.cols - 1; ++x) {
            if (valid_mask.at<uchar>(y, x)) {
                float center_depth = depth.at<float>(y, x);
                int valid_neighbors = 0;
                double total_distance = 0.0;
                
                // Check 8-neighborhood
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (valid_mask.at<uchar>(ny, nx)) {
                            float neighbor_depth = depth.at<float>(ny, nx);
                            double distance = std::abs(center_depth - neighbor_depth);
                            
                            if (distance < max_distance) {
                                valid_neighbors++;
                                total_distance += distance;
                            }
                        }
                    }
                }
                
                // Mark as outlier if not enough valid neighbors
                if (valid_neighbors < min_neighbors) {
                    mask.at<uchar>(y, x) = 255;
                }
            }
        }
    }
    
    // Remove outliers by setting to 0
    filtered.setTo(0, mask);
    
    return filtered;
}

cv::Mat DepthProcessor::fillHoles(const cv::Mat& depth, int max_hole_size) {
    cv::Mat filled = depth.clone();
    cv::Mat mask = (depth == 0);
    
    // Use morphological closing to fill small holes
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, 
                                             cv::Size(max_hole_size, max_hole_size));
    cv::morphologyEx(filled, filled, cv::MORPH_CLOSE, kernel);
    
    // For larger holes, use inpainting
    if (cv::countNonZero(mask) > 0) {
        cv::inpaint(filled, mask, filled, 3, cv::INPAINT_TELEA);
    }
    
    return filled;
}

double DepthProcessor::calculateDepthQuality(const cv::Mat& depth) {
    if (depth.empty()) {
        return 0.0;
    }
    
    cv::Mat valid_mask = (depth > 0);
    int total_pixels = depth.rows * depth.cols;
    int valid_pixels = cv::countNonZero(valid_mask);
    
    if (valid_pixels == 0) {
        return 0.0;
    }
    
    double coverage_ratio = static_cast<double>(valid_pixels) / total_pixels;
    
    // Calculate local variance as a measure of smoothness
    cv::Mat variance_map;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(depth, variance_map, cv::MORPH_GRADIENT, kernel);
    
    cv::Scalar mean_variance = cv::mean(variance_map, valid_mask);
    double smoothness_score = 1.0 / (1.0 + mean_variance[0] / 1000.0); // Normalize
    
    // Combine coverage and smoothness
    double quality_score = 0.7 * coverage_ratio + 0.3 * smoothness_score;
    
    return std::min(1.0, std::max(0.0, quality_score));
}

bool DepthProcessor::isDepthValid(const cv::Mat& depth, double min_quality_threshold) {
    double quality = calculateDepthQuality(depth);
    return quality >= min_quality_threshold;
}

void DepthProcessor::setBilateralParams(int d, double sigma_color, double sigma_space) {
    bilateral_d_ = d;
    bilateral_sigma_color_ = sigma_color;
    bilateral_sigma_space_ = sigma_space;
}

void DepthProcessor::setMedianKernelSize(int size) {
    median_kernel_size_ = size;
}

void DepthProcessor::setMorphologicalKernelSize(int size) {
    morphological_kernel_size_ = size;
}

void DepthProcessor::setTemporalAlpha(float alpha) {
    temporal_alpha_ = alpha;
}

void DepthProcessor::setQualityThreshold(double threshold) {
    quality_threshold_ = threshold;
}

cv::Mat DepthProcessor::adaptiveProcessing(const cv::Mat& depth, const cv::Mat& color) {
    cv::Mat processed = depth.clone();
    
    // Calculate local quality metrics
    cv::Mat quality_map = cv::Mat::zeros(depth.size(), CV_32F);
    
    for (int y = 0; y < depth.rows; ++y) {
        for (int x = 0; x < depth.cols; ++x) {
            double local_quality = calculateLocalVariance(depth, x, y);
            quality_map.at<float>(y, x) = static_cast<float>(local_quality);
        }
    }
    
    // Apply adaptive filtering based on local quality
    cv::Mat high_quality_mask = (quality_map < 0.1);
    cv::Mat low_quality_mask = (quality_map > 0.3);
    
    // Light filtering for high-quality regions
    cv::Mat light_filtered = bilateralFilter(depth, 3, 25.0, 25.0);
    light_filtered.copyTo(processed, high_quality_mask);
    
    // Heavy filtering for low-quality regions
    cv::Mat heavy_filtered = bilateralFilter(depth, 9, 75.0, 75.0);
    heavy_filtered = medianFilter(heavy_filtered, 7);
    heavy_filtered.copyTo(processed, low_quality_mask);
    
    return processed;
}

cv::Mat DepthProcessor::multiScaleProcessing(const cv::Mat& depth, int num_scales) {
    cv::Mat result = depth.clone();
    
    for (int scale = 1; scale <= num_scales; ++scale) {
        int kernel_size = 2 * scale + 1;
        cv::Mat scaled_result = bilateralFilter(result, kernel_size, 
                                              50.0 * scale, 50.0 * scale);
        
        // Blend with previous result
        double alpha = 1.0 / num_scales;
        cv::addWeighted(result, 1.0 - alpha, scaled_result, alpha, 0.0, result);
    }
    
    return result;
}

cv::Mat DepthProcessor::createGaussianKernel(int size, double sigma) {
    cv::Mat kernel = cv::getGaussianKernel(size, sigma, CV_32F);
    return kernel * kernel.t();
}

cv::Mat DepthProcessor::applyGaussianBlur(const cv::Mat& input, int kernel_size, double sigma) {
    cv::Mat output;
    cv::GaussianBlur(input, output, cv::Size(kernel_size, kernel_size), sigma);
    return output;
}

double DepthProcessor::calculateLocalVariance(const cv::Mat& depth, int x, int y, int window_size) {
    int half_window = window_size / 2;
    int start_x = std::max(0, x - half_window);
    int end_x = std::min(depth.cols - 1, x + half_window);
    int start_y = std::max(0, y - half_window);
    int end_y = std::min(depth.rows - 1, y + half_window);
    
    std::vector<float> values;
    for (int py = start_y; py <= end_y; ++py) {
        for (int px = start_x; px <= end_x; ++px) {
            float value = depth.at<float>(py, px);
            if (value > 0) {
                values.push_back(value);
            }
        }
    }
    
    if (values.size() < 3) {
        return 1.0; // High variance for insufficient data
    }
    
    // Calculate variance
    double mean = 0.0;
    for (float value : values) {
        mean += value;
    }
    mean /= values.size();
    
    double variance = 0.0;
    for (float value : values) {
        variance += (value - mean) * (value - mean);
    }
    variance /= values.size();
    
    return variance;
}

bool DepthProcessor::isEdgePixel(const cv::Mat& depth, int x, int y, double threshold) {
    if (x <= 0 || x >= depth.cols - 1 || y <= 0 || y >= depth.rows - 1) {
        return false;
    }
    
    float center = depth.at<float>(y, x);
    if (center <= 0) return false;
    
    // Check horizontal gradient
    float left = depth.at<float>(y, x - 1);
    float right = depth.at<float>(y, x + 1);
    if (left > 0 && right > 0) {
        double h_gradient = std::abs(center - left) + std::abs(center - right);
        if (h_gradient > threshold) return true;
    }
    
    // Check vertical gradient
    float top = depth.at<float>(y - 1, x);
    float bottom = depth.at<float>(y + 1, x);
    if (top > 0 && bottom > 0) {
        double v_gradient = std::abs(center - top) + std::abs(center - bottom);
        if (v_gradient > threshold) return true;
    }
    
    return false;
}

} // namespace industrial_reconstruction

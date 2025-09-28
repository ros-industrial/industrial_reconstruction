#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <map>

namespace industrial_reconstruction {

struct ReconstructionParams {
    double voxel_length;
    double sdf_trunc;
    double depth_scale;
    double depth_trunc;
    double translation_threshold;
    double rotation_threshold;
    bool use_adaptive_params;
    bool enable_depth_preprocessing;
    bool enable_mesh_postprocessing;
    int mesh_smoothing_iterations;
    double mesh_smoothing_lambda;
    double quality_threshold;
};

struct SceneAnalysis {
    double scene_complexity;      // 0-1, based on depth variance and edge density
    double motion_velocity;       // Camera motion speed
    double depth_quality;         // Average depth image quality
    double coverage_ratio;        // Percentage of valid depth pixels
    double edge_density;          // Number of edges per unit area
    double noise_level;           // Estimated noise level in depth data
    std::string scene_type;       // "indoor", "outdoor", "industrial", "textured"
};

class AdaptiveReconstructor {
public:
    AdaptiveReconstructor();
    ~AdaptiveReconstructor() = default;
    
    // Main reconstruction pipeline
    ReconstructionParams optimizeParameters(const cv::Mat& depth_image, 
                                          const cv::Mat& color_image,
                                          const SceneAnalysis& scene_analysis);
    
    // Scene analysis
    SceneAnalysis analyzeScene(const cv::Mat& depth_image, 
                             const cv::Mat& color_image,
                             const cv::Mat& previous_depth = cv::Mat());
    
    // Parameter optimization
    ReconstructionParams optimizeForQuality(const SceneAnalysis& analysis);
    ReconstructionParams optimizeForSpeed(const SceneAnalysis& analysis);
    ReconstructionParams optimizeForAccuracy(const SceneAnalysis& analysis);
    
    // Adaptive parameter adjustment
    void updateParametersBasedOnFeedback(const ReconstructionParams& current_params,
                                        double quality_feedback,
                                        const SceneAnalysis& analysis);
    
    // Quality prediction
    double predictMeshQuality(const ReconstructionParams& params, 
                            const SceneAnalysis& analysis);
    double estimateProcessingTime(const ReconstructionParams& params,
                                const SceneAnalysis& analysis);
    
    // Configuration
    void setQualityMode(const std::string& mode); // "speed", "quality", "balanced"
    void setSceneType(const std::string& type);
    void setTargetQuality(double quality);
    void setMaxProcessingTime(double time_seconds);
    
    // Learning and adaptation
    void recordReconstructionResult(const ReconstructionParams& params,
                                  const SceneAnalysis& analysis,
                                  double actual_quality,
                                  double processing_time);
    void updateParameterDatabase();
    
private:
    // Configuration
    std::string quality_mode_;
    std::string scene_type_;
    double target_quality_;
    double max_processing_time_;
    
    // Parameter database for different scenarios
    std::map<std::string, ReconstructionParams> parameter_database_;
    std::vector<std::tuple<ReconstructionParams, SceneAnalysis, double, double>> result_history_;
    
    // Helper functions
    double calculateSceneComplexity(const cv::Mat& depth_image, const cv::Mat& color_image);
    double calculateMotionVelocity(const cv::Mat& current_depth, const cv::Mat& previous_depth);
    double calculateDepthQuality(const cv::Mat& depth_image);
    double calculateCoverageRatio(const cv::Mat& depth_image);
    double calculateEdgeDensity(const cv::Mat& color_image);
    double estimateNoiseLevel(const cv::Mat& depth_image);
    std::string classifySceneType(const cv::Mat& depth_image, const cv::Mat& color_image);
    
    // Parameter optimization algorithms
    ReconstructionParams geneticAlgorithmOptimization(const SceneAnalysis& analysis);
    ReconstructionParams gradientDescentOptimization(const SceneAnalysis& analysis);
    ReconstructionParams ruleBasedOptimization(const SceneAnalysis& analysis);
    
    // Quality metrics
    double calculateParameterQuality(const ReconstructionParams& params);
    double calculateParameterEfficiency(const ReconstructionParams& params);
    double calculateParameterRobustness(const ReconstructionParams& params);
    
    // Learning algorithms
    void updateParameterWeights(const std::vector<std::tuple<ReconstructionParams, SceneAnalysis, double, double>>& results);
    ReconstructionParams interpolateParameters(const SceneAnalysis& analysis);
    void pruneParameterDatabase();
};

} // namespace industrial_reconstruction

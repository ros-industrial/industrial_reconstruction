# Enhanced TSDF parameters for improved mesh quality
industrial_reconstruction_msgs/TSDFVolumeParams base_params

# Advanced depth preprocessing parameters
bool enable_depth_preprocessing
float32 bilateral_d
float32 bilateral_sigma_color
float32 bilateral_sigma_space
float32 median_kernel_size
float32 morphological_kernel_size
float32 temporal_alpha
float32 outlier_std_threshold
float32 max_hole_size

# Advanced mesh post-processing parameters
bool enable_mesh_postprocessing
int32 smoothing_iterations
float32 smoothing_lambda
float32 decimation_ratio
float32 target_edge_length
float32 hole_filling_diameter
float32 curvature_threshold
float32 quality_threshold

# Adaptive quality control
bool enable_adaptive_params
string quality_mode  # "speed", "quality", "balanced", "accuracy"
float32 target_quality_score
float32 max_processing_time
string scene_type    # "indoor", "outdoor", "industrial", "textured"

# Advanced filtering options
bool enable_bilateral_smoothing
bool enable_curvature_smoothing
bool enable_laplacian_smoothing
bool enable_topology_optimization
bool enable_hole_filling
bool enable_outlier_removal

####################################################################
# See Open3d function create_from_color_and_depth for more details #
####################################################################
# The ratio to scale depth values. The depth values will first be scaled and then truncated.
float32 depth_scale # Recommended default: 1000.0
# Depth values larger than depth_trunc gets truncated to 0. The depth values will first be scaled and then truncated.
float32 depth_trunc # Recommended default: 3.0
# Whether to convert RGB image to intensity image.
bool convert_rgb_to_intensity # Recommended default: True

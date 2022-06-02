##################################################
# See Open3d ScalableTSDFVolume for more details #
##################################################
# Means a single voxel size in meters for TSDF volume. Example: 4.0m / 512.0 = 0.0078125m
# Lowering this value makes a high-resolution TSDF volume, but the
# integration result can be susceptible to depth noise
float32 voxel_length # Recommended default: 0.01
# Specifies the truncation value for the signed distance function (SDF)
float32 sdf_trunc # Recommended default: 0.04
geometry_msgs/Vector3 min_box_values
geometry_msgs/Vector3 max_box_values
########################################################
# See Open3d GlobalOptimizationOption for more details #
########################################################
# Identifies which distance value is used for finding neighboring points when making information matrix
float32 max_correspondence_distance # Recommended default: 0.01
# If the line_process weight < edge_prune_threshold it is pruned
float32 edge_prune_threshold # Recommended default: 0.25
# [0, 1] Balancing parameter to decide which one is more reliable
# 0 - try to unchange odometry edges
# 1 - try to utilize loop-closure
float32 preference_loop_closure # Recommended default: 0.1

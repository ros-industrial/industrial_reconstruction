#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <memory>
#include <mutex>

namespace industrial_reconstruction {

struct TransformData {
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation;
    double timestamp;
    
    TransformData() : translation(Eigen::Vector3d::Zero()), 
                     rotation(Eigen::Quaterniond::Identity()), timestamp(0.0) {}
    
    TransformData(const Eigen::Vector3d& trans, const Eigen::Quaterniond& rot, double ts)
        : translation(trans), rotation(rot), timestamp(ts) {}
};

class PoseCalculator {
public:
    PoseCalculator();
    ~PoseCalculator() = default;
    
    // Transform operations
    Eigen::Matrix4d create_transformation_matrix(const Eigen::Vector3d& translation, 
                                                const Eigen::Quaterniond& rotation) const;
    Eigen::Matrix4d invert_transformation(const Eigen::Matrix4d& transform) const;
    
    // Distance calculations
    double calculate_translation_distance(const Eigen::Vector3d& pos1, 
                                        const Eigen::Vector3d& pos2) const;
    double calculate_rotation_distance(const Eigen::Quaterniond& rot1, 
                                     const Eigen::Quaterniond& rot2) const;
    
    // Pose filtering and smoothing
    bool should_process_pose(const TransformData& current_pose, 
                           const TransformData& previous_pose,
                           double translation_threshold, 
                           double rotation_threshold) const;
    
    // Interpolation
    TransformData interpolate_poses(const TransformData& pose1, 
                                  const TransformData& pose2, 
                                  double alpha) const;
    
    // Batch processing
    std::vector<Eigen::Matrix4d> process_pose_sequence(
        const std::vector<TransformData>& poses,
        double translation_threshold,
        double rotation_threshold) const;
    
    // Memory-efficient pose storage
    void add_pose(const TransformData& pose);
    bool get_pose_at_index(size_t index, TransformData& pose) const;
    size_t get_pose_count() const;
    void clear_poses();
    
private:
    std::vector<TransformData> pose_history_;
    mutable std::mutex pose_mutex_;
    
    // Helper functions
    Eigen::Quaterniond normalize_quaternion(const Eigen::Quaterniond& q) const;
    double quaternion_angle_distance(const Eigen::Quaterniond& q1, 
                                   const Eigen::Quaterniond& q2) const;
};

} // namespace industrial_reconstruction

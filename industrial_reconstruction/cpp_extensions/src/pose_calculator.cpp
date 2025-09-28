#include "pose_calculator.h"
#include <mutex>
#include <algorithm>
#include <cmath>

namespace industrial_reconstruction {

PoseCalculator::PoseCalculator() {
}

Eigen::Matrix4d PoseCalculator::create_transformation_matrix(
    const Eigen::Vector3d& translation, 
    const Eigen::Quaterniond& rotation) const {
    
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = rotation.normalized().toRotationMatrix();
    transform.block<3, 1>(0, 3) = translation;
    return transform;
}

Eigen::Matrix4d PoseCalculator::invert_transformation(const Eigen::Matrix4d& transform) const {
    Eigen::Matrix4d inverse = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
    Eigen::Vector3d translation = transform.block<3, 1>(0, 3);
    
    inverse.block<3, 3>(0, 0) = rotation.transpose();
    inverse.block<3, 1>(0, 3) = -rotation.transpose() * translation;
    
    return inverse;
}

double PoseCalculator::calculate_translation_distance(
    const Eigen::Vector3d& pos1, 
    const Eigen::Vector3d& pos2) const {
    return (pos1 - pos2).norm();
}

double PoseCalculator::calculate_rotation_distance(
    const Eigen::Quaterniond& rot1, 
    const Eigen::Quaterniond& rot2) const {
    return quaternion_angle_distance(rot1, rot2);
}

bool PoseCalculator::should_process_pose(
    const TransformData& current_pose, 
    const TransformData& previous_pose,
    double translation_threshold, 
    double rotation_threshold) const {
    
    double trans_dist = calculate_translation_distance(
        current_pose.translation, previous_pose.translation);
    double rot_dist = calculate_rotation_distance(
        current_pose.rotation, previous_pose.rotation);
    
    return (trans_dist >= translation_threshold) || (rot_dist >= rotation_threshold);
}

TransformData PoseCalculator::interpolate_poses(
    const TransformData& pose1, 
    const TransformData& pose2, 
    double alpha) const {
    
    // Clamp alpha to [0, 1]
    alpha = std::max(0.0, std::min(1.0, alpha));
    
    // Interpolate translation
    Eigen::Vector3d interp_translation = 
        (1.0 - alpha) * pose1.translation + alpha * pose2.translation;
    
    // Interpolate rotation using spherical linear interpolation
    Eigen::Quaterniond interp_rotation = pose1.rotation.slerp(alpha, pose2.rotation);
    
    // Interpolate timestamp
    double interp_timestamp = (1.0 - alpha) * pose1.timestamp + alpha * pose2.timestamp;
    
    return TransformData(interp_translation, interp_rotation, interp_timestamp);
}

std::vector<Eigen::Matrix4d> PoseCalculator::process_pose_sequence(
    const std::vector<TransformData>& poses,
    double translation_threshold,
    double rotation_threshold) const {
    
    std::vector<Eigen::Matrix4d> filtered_transforms;
    
    if (poses.empty()) {
        return filtered_transforms;
    }
    
    // Always include the first pose
    filtered_transforms.push_back(create_transformation_matrix(
        poses[0].translation, poses[0].rotation));
    
    TransformData last_accepted_pose = poses[0];
    
    for (size_t i = 1; i < poses.size(); ++i) {
        if (should_process_pose(poses[i], last_accepted_pose, 
                              translation_threshold, rotation_threshold)) {
            filtered_transforms.push_back(create_transformation_matrix(
                poses[i].translation, poses[i].rotation));
            last_accepted_pose = poses[i];
        }
    }
    
    return filtered_transforms;
}

void PoseCalculator::add_pose(const TransformData& pose) {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    pose_history_.push_back(pose);
}

bool PoseCalculator::get_pose_at_index(size_t index, TransformData& pose) const {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    
    if (index >= pose_history_.size()) {
        return false;
    }
    
    pose = pose_history_[index];
    return true;
}

size_t PoseCalculator::get_pose_count() const {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    return pose_history_.size();
}

void PoseCalculator::clear_poses() {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    pose_history_.clear();
}

Eigen::Quaterniond PoseCalculator::normalize_quaternion(const Eigen::Quaterniond& q) const {
    return q.normalized();
}

double PoseCalculator::quaternion_angle_distance(
    const Eigen::Quaterniond& q1, 
    const Eigen::Quaterniond& q2) const {
    
    Eigen::Quaterniond q1_norm = normalize_quaternion(q1);
    Eigen::Quaterniond q2_norm = normalize_quaternion(q2);
    
    // Calculate the angle between quaternions
    double dot_product = std::abs(q1_norm.dot(q2_norm));
    dot_product = std::min(1.0, dot_product); // Clamp to avoid numerical errors
    
    return 2.0 * std::acos(dot_product);
}

} // namespace industrial_reconstruction

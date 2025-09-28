# Copyright 2022 Southwest Research Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros import TransformListener
import open3d as o3d
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import cv2

from pyquaternion import Quaternion
from collections import deque
from os.path import exists, join, isfile
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from src.industrial_reconstruction.utility.file import make_clean_folder, write_pose, read_pose, save_intrinsic_as_json, make_folder_keep_contents
from industrial_reconstruction_msgs.srv import StartReconstruction, StopReconstruction
from industrial_reconstruction_msgs.msg import EnhancedTSDFParams
from src.industrial_reconstruction.utility.ros import getIntrinsicsFromMsg, meshToRos, transformStampedToVectors

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
from visualization_msgs.msg import Marker

# Import C++ extensions
try:
    import industrial_reconstruction_cpp as cpp_ext
    CPP_EXTENSIONS_AVAILABLE = True
    print("C++ extensions loaded successfully")
except ImportError as e:
    print(f"Warning: C++ extensions not available: {e}")
    print("Falling back to Python implementation")
    CPP_EXTENSIONS_AVAILABLE = False

def filterNormals(mesh, direction, angle):
   mesh.compute_vertex_normals()
   tri_normals = np.asarray(mesh.triangle_normals)
   dot_prods = tri_normals @ direction
   mesh.remove_triangles_by_mask(dot_prods < np.cos(angle))
   return mesh

def enhancedDepthPreprocessing(depth_image, color_image=None, params=None):
    """Enhanced depth preprocessing pipeline"""
    if params is None:
        params = {
            'bilateral_d': 5,
            'bilateral_sigma_color': 50.0,
            'bilateral_sigma_space': 50.0,
            'median_kernel_size': 5,
            'morphological_kernel_size': 3,
            'outlier_std_threshold': 2.0,
            'max_hole_size': 10
        }
    
    processed_depth = depth_image.copy()
    
    # Step 1: Remove statistical outliers
    processed_depth = removeStatisticalOutliers(processed_depth, params['outlier_std_threshold'])
    
    # Step 2: Fill holes
    processed_depth = fillDepthHoles(processed_depth, params['max_hole_size'])
    
    # Step 3: Bilateral filtering for edge preservation
    processed_depth = cv2.bilateralFilter(processed_depth, 
                                        params['bilateral_d'],
                                        params['bilateral_sigma_color'],
                                        params['bilateral_sigma_space'])
    
    # Step 4: Median filtering for noise reduction
    processed_depth = cv2.medianBlur(processed_depth, params['median_kernel_size'])
    
    # Step 5: Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                     (params['morphological_kernel_size'], 
                                      params['morphological_kernel_size']))
    processed_depth = cv2.morphologyEx(processed_depth, cv2.MORPH_OPEN, kernel)
    processed_depth = cv2.morphologyEx(processed_depth, cv2.MORPH_CLOSE, kernel)
    
    return processed_depth

def removeStatisticalOutliers(depth_image, std_dev_threshold=2.0):
    """Remove statistical outliers from depth image"""
    # Convert to float for processing
    depth_float = depth_image.astype(np.float32)
    
    # Create mask for valid depth values
    valid_mask = depth_float > 0
    
    if np.sum(valid_mask) == 0:
        return depth_image
    
    # Calculate mean and standard deviation of valid pixels
    valid_pixels = depth_float[valid_mask]
    mean_depth = np.mean(valid_pixels)
    std_depth = np.std(valid_pixels)
    
    # Create outlier mask
    outlier_mask = np.abs(depth_float - mean_depth) > (std_dev_threshold * std_depth)
    
    # Remove outliers
    depth_float[outlier_mask] = 0
    
    return depth_float.astype(depth_image.dtype)

def fillDepthHoles(depth_image, max_hole_size=10):
    """Fill holes in depth image"""
    # Use morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_hole_size, max_hole_size))
    filled = cv2.morphologyEx(depth_image, cv2.MORPH_CLOSE, kernel)
    
    # For larger holes, use inpainting
    mask = (depth_image == 0).astype(np.uint8) * 255
    if np.sum(mask) > 0:
        filled = cv2.inpaint(filled, mask, 3, cv2.INPAINT_TELEA)
    
    return filled

def enhancedMeshPostProcessing(mesh, params=None):
    """Enhanced mesh post-processing pipeline"""
    if params is None:
        params = {
            'smoothing_iterations': 10,
            'smoothing_lambda': 0.5,
            'decimation_ratio': 0.5,
            'target_edge_length': 0.01,
            'hole_filling_diameter': 0.1,
            'quality_threshold': 0.6
        }
    
    processed_mesh = mesh
    
    # Step 1: Remove outliers
    processed_mesh = processed_mesh.remove_outliers(nb_neighbors=20, std_ratio=2.0)
    
    # Step 2: Fill holes
    processed_mesh = processed_mesh.fill_holes()
    
    # Step 3: Smooth mesh
    processed_mesh = processed_mesh.filter_smooth_laplacian(
        number_of_iterations=params['smoothing_iterations'],
        lambda_filter=params['smoothing_lambda'])
    
    # Step 4: Simplify mesh if needed
    if params['decimation_ratio'] < 1.0:
        target_triangles = int(len(processed_mesh.triangles) * params['decimation_ratio'])
        processed_mesh = processed_mesh.simplify_quadric_decimation(target_triangles)
    
    # Step 5: Remesh for uniform triangle size
    if params['target_edge_length'] > 0:
        processed_mesh = processed_mesh.remesh_poisson(
            number_of_iterations=5,
            target_edge_length=params['target_edge_length'])
    
    # Step 6: Compute normals
    processed_mesh.compute_vertex_normals()
    processed_mesh.compute_triangle_normals()
    
    return processed_mesh

def calculateMeshQuality(mesh):
    """Calculate mesh quality metrics"""
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return 0.0
    
    # Calculate basic metrics
    vertex_count = len(mesh.vertices)
    triangle_count = len(mesh.triangles)
    
    # Calculate edge lengths
    edges = []
    for triangle in mesh.triangles:
        v0, v1, v2 = mesh.vertices[triangle]
        edges.extend([
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2)
        ])
    
    if len(edges) == 0:
        return 0.0
    
    # Calculate quality metrics
    avg_edge_length = np.mean(edges)
    edge_length_std = np.std(edges)
    edge_length_cv = edge_length_std / avg_edge_length if avg_edge_length > 0 else 1.0
    
    # Calculate aspect ratios
    aspect_ratios = []
    for triangle in mesh.triangles:
        v0, v1, v2 = mesh.vertices[triangle]
        edges_tri = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2)
        ]
        if min(edges_tri) > 0:
            aspect_ratio = max(edges_tri) / min(edges_tri)
            aspect_ratios.append(aspect_ratio)
    
    avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 1.0
    
    # Calculate quality score (0-1, higher is better)
    edge_quality = max(0, 1.0 - edge_length_cv)  # Lower coefficient of variation is better
    aspect_quality = max(0, 1.0 - (avg_aspect_ratio - 1.0) / 10.0)  # Aspect ratio close to 1 is better
    
    quality_score = 0.6 * edge_quality + 0.4 * aspect_quality
    
    return min(1.0, max(0.0, quality_score))

def adaptiveParameterOptimization(depth_image, color_image, scene_type="industrial"):
    """Adaptive parameter optimization based on scene analysis"""
    # Analyze scene characteristics
    depth_quality = calculateDepthQuality(depth_image)
    scene_complexity = calculateSceneComplexity(depth_image, color_image)
    
    # Base parameters
    base_params = {
        'voxel_length': 0.01,
        'sdf_trunc': 0.04,
        'depth_scale': 1000.0,
        'depth_trunc': 3.0,
        'translation_distance': 0.0,
        'rotation_distance': 0.0
    }
    
    # Adaptive adjustments based on scene analysis
    if scene_type == "industrial":
        # Industrial scenes: prioritize accuracy
        base_params['voxel_length'] *= 0.8  # Higher resolution
        base_params['sdf_trunc'] *= 0.7     # Tighter integration
    elif scene_type == "textured":
        # Textured scenes: balance quality and speed
        base_params['voxel_length'] *= 1.0
        base_params['sdf_trunc'] *= 1.0
    elif scene_type == "outdoor":
        # Outdoor scenes: prioritize speed
        base_params['voxel_length'] *= 1.2  # Lower resolution
        base_params['sdf_trunc'] *= 1.3     # Looser integration
    
    # Adjust based on depth quality
    if depth_quality < 0.5:
        # Poor depth quality: use more aggressive filtering
        base_params['sdf_trunc'] *= 1.5
    elif depth_quality > 0.8:
        # Good depth quality: use finer parameters
        base_params['voxel_length'] *= 0.9
    
    # Adjust based on scene complexity
    if scene_complexity > 0.7:
        # Complex scene: use higher resolution
        base_params['voxel_length'] *= 0.8
    elif scene_complexity < 0.3:
        # Simple scene: can use lower resolution
        base_params['voxel_length'] *= 1.2
    
    return base_params

def calculateDepthQuality(depth_image):
    """Calculate depth image quality score"""
    if depth_image is None or depth_image.size == 0:
        return 0.0
    
    # Calculate coverage ratio
    valid_pixels = np.sum(depth_image > 0)
    total_pixels = depth_image.size
    coverage_ratio = valid_pixels / total_pixels
    
    # Calculate local variance as smoothness measure
    if valid_pixels > 0:
        valid_depth = depth_image[depth_image > 0]
        depth_std = np.std(valid_depth)
        depth_mean = np.mean(valid_depth)
        cv = depth_std / depth_mean if depth_mean > 0 else 1.0
        smoothness_score = max(0, 1.0 - cv)
    else:
        smoothness_score = 0.0
    
    # Combine metrics
    quality_score = 0.7 * coverage_ratio + 0.3 * smoothness_score
    
    return min(1.0, max(0.0, quality_score))

def calculateSceneComplexity(depth_image, color_image):
    """Calculate scene complexity score"""
    if depth_image is None or color_image is None:
        return 0.5
    
    # Calculate depth variance
    valid_depth = depth_image[depth_image > 0]
    if len(valid_depth) > 0:
        depth_variance = np.var(valid_depth)
        depth_complexity = min(1.0, depth_variance / 10000.0)  # Normalize
    else:
        depth_complexity = 0.0
    
    # Calculate edge density in color image
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) if len(color_image.shape) == 3 else color_image
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    edge_complexity = min(1.0, edge_density * 10.0)  # Normalize
    
    # Combine metrics
    complexity_score = 0.6 * depth_complexity + 0.4 * edge_complexity
    
    return min(1.0, max(0.0, complexity_score))

class IndustrialReconstructionEnhanced(Node):

    def __init__(self):
        super().__init__('industrial_reconstruction_enhanced')

        self.bridge = CvBridge()

        self.buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.buffer, node=self)

        self.tsdf_volume = None
        self.intrinsics = None
        self.crop_box = None
        self.crop_mesh = False
        self.crop_box_msg = Marker()
        self.tracking_frame = ''
        self.relative_frame = ''
        self.translation_distance = 0.05
        self.rotational_distance = 0.01

        # Enhanced processing parameters
        self.enhanced_params = None
        self.enable_depth_preprocessing = True
        self.enable_mesh_postprocessing = True
        self.adaptive_parameters = True
        self.quality_mode = "balanced"  # "speed", "quality", "balanced", "accuracy"
        
        # Scene analysis
        self.scene_type = "industrial"
        self.previous_depth = None
        self.scene_analysis_history = deque(maxlen=10)

        # Initialize C++ components if available
        if CPP_EXTENSIONS_AVAILABLE:
            self.image_buffer = cpp_ext.ImageBuffer(max_size=50)
            self.pose_calculator = cpp_ext.PoseCalculator()
            self.memory_manager = cpp_ext.GlobalMemoryManager.get_image_manager()
            self.memory_manager.preallocate_buffers(10, 480, 640)
            print("C++ components initialized")
        else:
            # Fallback to Python implementations
            self.sensor_data = deque()
            self.color_images = []
            self.depth_images = []
            self.rgb_poses = []
            self.prev_pose_rot = np.array([1.0, 0.0, 0.0, 0.0])
            self.prev_pose_tran = np.array([0.0, 0.0, 0.0])
            print("Using Python fallback implementations")

        self.tsdf_integration_data = deque()
        self.integration_done = True
        self.live_integration = False
        self.mesh_pub = None
        self.tsdf_volume_pub = None

        self.record = False
        self.frame_count = 0
        self.processed_frame_count = 0
        self.reconstructed_frame_count = 0

        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Enhanced quality tracking
        self.mesh_quality_history = deque(maxlen=20)
        self.processing_time_history = deque(maxlen=20)

        self.declare_parameter("depth_image_topic")
        self.declare_parameter("color_image_topic")
        self.declare_parameter("camera_info_topic")
        self.declare_parameter("cache_count", 10)
        self.declare_parameter("slop", 0.01)
        self.declare_parameter("enable_depth_preprocessing", True)
        self.declare_parameter("enable_mesh_postprocessing", True)
        self.declare_parameter("adaptive_parameters", True)
        self.declare_parameter("quality_mode", "balanced")

        try:
            self.depth_image_topic = str(self.get_parameter('depth_image_topic').value)
        except:
            self.get_logger().error("Failed to load depth_image_topic parameter")
        try:
            self.color_image_topic = str(self.get_parameter('color_image_topic').value)
        except:
            self.get_logger().error("Failed to load color_image_topic parameter")
        try:
            self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        except:
            self.get_logger().error("Failed to load camera_info_topic parameter")
        try:
            self.cache_count = int(self.get_parameter('cache_count').value)
        except:
            self.get_logger().info("Failed to load cache_count parameter")
        try:
            self.slop = float(self.get_parameter('slop').value)
        except:
            self.get_logger().info("Failed to load slop parameter")
        try:
            self.enable_depth_preprocessing = bool(self.get_parameter('enable_depth_preprocessing').value)
        except:
            self.get_logger().info("Failed to load enable_depth_preprocessing parameter")
        try:
            self.enable_mesh_postprocessing = bool(self.get_parameter('enable_mesh_postprocessing').value)
        except:
            self.get_logger().info("Failed to load enable_mesh_postprocessing parameter")
        try:
            self.adaptive_parameters = bool(self.get_parameter('adaptive_parameters').value)
        except:
            self.get_logger().info("Failed to load adaptive_parameters parameter")
        try:
            self.quality_mode = str(self.get_parameter('quality_mode').value)
        except:
            self.get_logger().info("Failed to load quality_mode parameter")

        allow_headerless = False

        self.get_logger().info("depth_image_topic - " + self.depth_image_topic)
        self.get_logger().info("color_image_topic - " + self.color_image_topic)
        self.get_logger().info("camera_info_topic - " + self.camera_info_topic)
        self.get_logger().info("Enhanced processing enabled - Depth: {}, Mesh: {}, Adaptive: {}".format(
            self.enable_depth_preprocessing, self.enable_mesh_postprocessing, self.adaptive_parameters))

        self.depth_sub = Subscriber(self, Image, self.depth_image_topic)
        self.color_sub = Subscriber(self, Image, self.color_image_topic)
        self.tss = ApproximateTimeSynchronizer([self.depth_sub, self.color_sub], self.cache_count, self.slop,
                                               allow_headerless)
        self.tss.registerCallback(self.cameraCallback)

        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.cameraInfoCallback, 10)

        self.mesh_pub = self.create_publisher(Marker, "industrial_reconstruction_mesh", 10)

        self.start_server = self.create_service(StartReconstruction, 'start_reconstruction',
                                                self.startReconstructionCallback)
        self.stop_server = self.create_service(StopReconstruction, 'stop_reconstruction',
                                               self.stopReconstructionCallback)

        self.tsdf_volume_pub = self.create_publisher(Marker, "tsdf_volume", 10)

    def startReconstructionCallback(self, req, res):
        self.get_logger().info("Start Enhanced Reconstruction")

        # Clear all data structures
        if CPP_EXTENSIONS_AVAILABLE:
            self.image_buffer.clear()
            self.pose_calculator.clear_poses()
            self.memory_manager.clear_unused_buffers()
        else:
            self.color_images.clear()
            self.depth_images.clear()
            self.rgb_poses.clear()
            self.sensor_data.clear()
            self.prev_pose_rot = np.array([1.0, 0.0, 0.0, 0.0])
            self.prev_pose_tran = np.array([0.0, 0.0, 0.0])

        self.tsdf_integration_data.clear()
        self.mesh_quality_history.clear()
        self.processing_time_history.clear()
        self.scene_analysis_history.clear()

        # Setup crop box if specified
        if (req.tsdf_params.min_box_values.x == req.tsdf_params.max_box_values.x and
                req.tsdf_params.min_box_values.y == req.tsdf_params.max_box_values.y and
                req.tsdf_params.min_box_values.z == req.tsdf_params.max_box_values.z):
            self.crop_mesh = False
        else:
            self.crop_mesh = True
            min_bound = np.asarray(
                [req.tsdf_params.min_box_values.x, req.tsdf_params.min_box_values.y, req.tsdf_params.min_box_values.z])
            max_bound = np.asarray(
                [req.tsdf_params.max_box_values.x, req.tsdf_params.max_box_values.y, req.tsdf_params.max_box_values.z])
            self.crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

            # Publish crop box visualization
            self.crop_box_msg.type = self.crop_box_msg.CUBE
            self.crop_box_msg.action = self.crop_box_msg.ADD
            self.crop_box_msg.id = 1
            self.crop_box_msg.scale.x = max_bound[0] - min_bound[0]
            self.crop_box_msg.scale.y = max_bound[1] - min_bound[1]
            self.crop_box_msg.scale.z = max_bound[2] - min_bound[2]
            self.crop_box_msg.pose.position.x = (min_bound[0] + max_bound[0]) / 2.0
            self.crop_box_msg.pose.position.y = (min_bound[1] + max_bound[1]) / 2.0
            self.crop_box_msg.pose.position.z = (min_bound[2] + max_bound[2]) / 2.0
            self.crop_box_msg.pose.orientation.w = 1.0
            self.crop_box_msg.pose.orientation.x = 0.0
            self.crop_box_msg.pose.orientation.y = 0.0
            self.crop_box_msg.pose.orientation.z = 0.0
            self.crop_box_msg.color.r = 1.0
            self.crop_box_msg.color.g = 0.0
            self.crop_box_msg.color.b = 0.0
            self.crop_box_msg.color.a = 0.25
            self.crop_box_msg.header.frame_id = req.relative_frame
            self.tsdf_volume_pub.publish(self.crop_box_msg)

        self.frame_count = 0
        self.processed_frame_count = 0
        self.reconstructed_frame_count = 0

        # Initialize TSDF volume with enhanced parameters
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=req.tsdf_params.voxel_length,
            sdf_trunc=req.tsdf_params.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        self.depth_scale = req.rgbd_params.depth_scale
        self.depth_trunc = req.rgbd_params.depth_trunc
        self.convert_rgb_to_intensity = req.rgbd_params.convert_rgb_to_intensity
        self.tracking_frame = req.tracking_frame
        self.relative_frame = req.relative_frame
        self.translation_distance = req.translation_distance
        self.rotational_distance = req.rotational_distance

        self.live_integration = req.live
        self.record = True

        res.success = True
        return res

    def stopReconstructionCallback(self, req, res):
        self.get_logger().info("Stop Enhanced Reconstruction")
        self.record = False

        while not self.integration_done:
            self.create_rate(1).sleep()

        self.get_logger().info("Generating enhanced mesh")
        if self.tsdf_volume is None:
            res.success = False
            res.message = "Start reconstruction hasn't been called yet"
            return res

        # Process remaining integration data
        if not self.live_integration:
            while len(self.tsdf_integration_data) > 0:
                data = self.tsdf_integration_data.popleft()
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(data[1], data[0], self.depth_scale, self.depth_trunc, False)
                self.tsdf_volume.integrate(rgbd, self.intrinsics, np.linalg.inv(data[2]))

        # Extract mesh
        start_time = time.time()
        mesh = self.tsdf_volume.extract_triangle_mesh()
        extraction_time = time.time() - start_time
        
        self.get_logger().info(f"Mesh extraction took {extraction_time:.3f} seconds")

        # Apply crop box if specified
        if self.crop_mesh:
            cropped_mesh = mesh.crop(self.crop_box)
        else:
            cropped_mesh = mesh

        # Enhanced mesh post-processing
        if self.enable_mesh_postprocessing:
            self.get_logger().info("Applying enhanced mesh post-processing")
            postprocess_start = time.time()
            
            # Calculate initial quality
            initial_quality = calculateMeshQuality(cropped_mesh)
            self.get_logger().info(f"Initial mesh quality: {initial_quality:.3f}")
            
            # Apply enhanced post-processing
            cropped_mesh = enhancedMeshPostProcessing(cropped_mesh)
            
            # Calculate final quality
            final_quality = calculateMeshQuality(cropped_mesh)
            postprocess_time = time.time() - postprocess_start
            
            self.get_logger().info(f"Final mesh quality: {final_quality:.3f}")
            self.get_logger().info(f"Mesh post-processing took {postprocess_time:.3f} seconds")
            self.get_logger().info(f"Quality improvement: {final_quality - initial_quality:.3f}")

        # Apply normal filtering
        for norm_filt in req.normal_filters:
            dir = np.array([norm_filt.normal_direction.x, norm_filt.normal_direction.y, norm_filt.normal_direction.z]).reshape(3,1)
            cropped_mesh = filterNormals(cropped_mesh, dir, np.radians(norm_filt.angle))

        # Apply cluster filtering
        triangle_clusters, cluster_n_triangles, cluster_area = (cropped_mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < req.min_num_faces
        cropped_mesh.remove_triangles_by_mask(triangles_to_remove)
        cropped_mesh.remove_unreferenced_vertices()

        # Save mesh
        o3d.io.write_triangle_mesh(req.mesh_filepath, cropped_mesh, False, True)
        
        # Publish mesh for visualization
        mesh_msg = meshToRos(cropped_mesh)
        mesh_msg.header.stamp = self.get_clock().now().to_msg()
        mesh_msg.header.frame_id = self.relative_frame
        self.mesh_pub.publish(mesh_msg)
        
        self.get_logger().info("Enhanced mesh saved to " + req.mesh_filepath)

        # Archive data if requested
        if req.archive_directory != "":
            self.get_logger().info("Archiving data to " + req.archive_directory)
            self.archiveData(req.archive_directory)
            archive_mesh_filepath = join(req.archive_directory, "integrated.ply")
            o3d.io.write_triangle_mesh(archive_mesh_filepath, mesh, False, True)

        # Log quality statistics
        if self.mesh_quality_history:
            avg_quality = np.mean(self.mesh_quality_history)
            self.get_logger().info(f"Average mesh quality during reconstruction: {avg_quality:.3f}")

        self.get_logger().info("Enhanced reconstruction completed")
        res.success = True
        res.message = "Enhanced mesh saved to " + req.mesh_filepath
        return res

    def processImageWithEnhancements(self, depth_image, color_image, timestamp):
        """Process image with enhanced preprocessing"""
        start_time = time.time()
        
        # Enhanced depth preprocessing
        if self.enable_depth_preprocessing:
            # Analyze scene for adaptive parameters
            if self.adaptive_parameters:
                scene_analysis = {
                    'depth_quality': calculateDepthQuality(depth_image),
                    'scene_complexity': calculateSceneComplexity(depth_image, color_image),
                    'scene_type': self.scene_type
                }
                self.scene_analysis_history.append(scene_analysis)
                
                # Adaptive parameter optimization
                adaptive_params = adaptiveParameterOptimization(depth_image, color_image, self.scene_type)
                
                # Update TSDF parameters if significantly different
                if hasattr(self, 'current_params'):
                    param_change = abs(adaptive_params['voxel_length'] - self.current_params.get('voxel_length', 0.01)) / 0.01
                    if param_change > 0.2:  # 20% change threshold
                        self.get_logger().info(f"Adapting voxel_length from {self.current_params.get('voxel_length', 0.01):.4f} to {adaptive_params['voxel_length']:.4f}")
                        self.current_params = adaptive_params
            
            # Apply enhanced depth preprocessing
            processed_depth = enhancedDepthPreprocessing(depth_image, color_image)
        else:
            processed_depth = depth_image
        
        # Store previous depth for temporal analysis
        self.previous_depth = depth_image.copy()
        
        processing_time = time.time() - start_time
        self.processing_time_history.append(processing_time)
        
        return processed_depth

    def cameraCallback(self, depth_image_msg, rgb_image_msg):
        if self.record:
            try:
                # Convert ROS messages to OpenCV
                cv2_depth_img = self.bridge.imgmsg_to_cv2(depth_image_msg, "16UC1")
                cv2_rgb_img = self.bridge.imgmsg_to_cv2(rgb_image_msg, rgb_image_msg.encoding)
            except CvBridgeError:
                self.get_logger().error("Error converting ros msg to cv img")
                return
            else:
                # Enhanced image processing
                timestamp = time.time()
                processed_depth = self.processImageWithEnhancements(cv2_depth_img, cv2_rgb_img, timestamp)
                
                if CPP_EXTENSIONS_AVAILABLE:
                    # Use C++ components for processing
                    self.executor.submit(self.processImageAsync, processed_depth, cv2_rgb_img, timestamp)
                else:
                    # Fallback to original Python implementation with enhancements
                    self.sensor_data.append(
                        [o3d.geometry.Image(processed_depth), o3d.geometry.Image(cv2_rgb_img), rgb_image_msg.header.stamp])
                    
                    if self.frame_count > 30:
                        data = self.sensor_data.popleft()
                        try:
                            gm_tf_stamped = self.buffer.lookup_transform(self.relative_frame, self.tracking_frame, data[2])
                        except Exception as e:
                            self.get_logger().error("Failed to get transform: " + str(e))
                            return
                        
                        rgb_t, rgb_r = transformStampedToVectors(gm_tf_stamped)
                        rgb_r_quat = Quaternion(rgb_r)

                        tran_dist = np.linalg.norm(rgb_t - self.prev_pose_tran)
                        rot_dist = Quaternion.absolute_distance(Quaternion(self.prev_pose_rot), rgb_r_quat)

                        if (tran_dist >= self.translation_distance) or (rot_dist >= self.rotational_distance):
                            self.prev_pose_tran = rgb_t
                            self.prev_pose_rot = rgb_r
                            rgb_pose = rgb_r_quat.transformation_matrix
                            rgb_pose[0, 3] = rgb_t[0]
                            rgb_pose[1, 3] = rgb_t[1]
                            rgb_pose[2, 3] = rgb_t[2]

                            self.depth_images.append(data[0])
                            self.color_images.append(data[1])
                            self.rgb_poses.append(rgb_pose)
                            
                            if self.live_integration and self.tsdf_volume is not None:
                                self.integration_done = False
                                try:
                                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(data[1], data[0], self.depth_scale, self.depth_trunc, False)
                                    self.tsdf_volume.integrate(rgbd, self.intrinsics, np.linalg.inv(rgb_pose))
                                    self.integration_done = True
                                    self.processed_frame_count += 1
                                    
                                    if self.processed_frame_count % 50 == 0:
                                        mesh = self.tsdf_volume.extract_triangle_mesh()
                                        if self.crop_mesh:
                                            cropped_mesh = mesh.crop(self.crop_box)
                                        else:
                                            cropped_mesh = mesh
                                        
                                        # Calculate and log mesh quality
                                        quality = calculateMeshQuality(cropped_mesh)
                                        self.mesh_quality_history.append(quality)
                                        
                                        mesh_msg = meshToRos(cropped_mesh)
                                        mesh_msg.header.stamp = self.get_clock().now().to_msg()
                                        mesh_msg.header.frame_id = self.relative_frame
                                        self.mesh_pub.publish(mesh_msg)
                                        
                                        if self.processed_frame_count % 200 == 0:
                                            avg_quality = np.mean(list(self.mesh_quality_history)[-10:]) if self.mesh_quality_history else 0.0
                                            self.get_logger().info(f"Current mesh quality: {quality:.3f}, Average: {avg_quality:.3f}")
                                
                                except Exception as e:
                                    self.get_logger().error("Error processing images into tsdf: " + str(e))
                                    self.integration_done = True
                                    return
                            else:
                                self.tsdf_integration_data.append([data[0], data[1], rgb_pose])
                                self.processed_frame_count += 1

                self.frame_count += 1

    def cameraInfoCallback(self, camera_info):
        self.intrinsics = getIntrinsicsFromMsg(camera_info)

    def archiveData(self, path_output):
        """Archive data with enhanced processing"""
        path_depth = join(path_output, "depth")
        path_color = join(path_output, "color")
        path_pose = join(path_output, "pose")

        make_folder_keep_contents(path_output)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)
        make_clean_folder(path_pose)

        for s in range(len(self.color_images)):
            o3d.io.write_image("%s/%06d.png" % (path_depth, s), self.depth_images[s])
            o3d.io.write_image("%s/%06d.jpg" % (path_color, s), self.color_images[s])
            write_pose("%s/%06d.pose" % (path_pose, s), self.rgb_poses[s])
            save_intrinsic_as_json(join(path_output, "camera_intrinsic.json"), self.intrinsics)

    def __del__(self):
        """Cleanup when node is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        if CPP_EXTENSIONS_AVAILABLE and hasattr(self, 'memory_manager'):
            self.memory_manager.print_memory_stats()


def main(args=None):
    rclpy.init(args=args)
    industrial_reconstruction = IndustrialReconstructionEnhanced()
    rclpy.spin(industrial_reconstruction)
    industrial_reconstruction.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

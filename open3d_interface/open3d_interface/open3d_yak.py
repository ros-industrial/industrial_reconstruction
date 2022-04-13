# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros import TransformListener
import open3d as o3d
import numpy as np

from ament_index_python.packages import get_package_share_directory
pkg_share_dir = get_package_share_directory('open3d_interface')
print(pkg_share_dir)
sys.path.append(pkg_share_dir)
sys.path.append(pkg_share_dir + "/open3d_interface")

from pyquaternion import Quaternion
from collections import deque
from os.path import exists, join, isfile
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from utility.file import make_clean_folder, write_pose, read_pose, save_intrinsic_as_json, make_folder_keep_contents
from open3d_interface_msgs.srv import StartYakReconstruction, StopYakReconstruction
from utility.ros import getIntrinsicsFromMsg, meshToRos, transformStampedToVectors

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
from visualization_msgs.msg import Marker



class Open3dYak(Node):

    def __init__(self):
        super().__init__('open3d_yak')

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
        self.translation_distance = 0.05  # 5cm
        self.rotational_distance = 0.01  # Quaternion Distance

        ####################################################################
        # See Open3d function create_from_color_and_depth for more details #
        ####################################################################
        # The ratio to scale depth values. The depth values will first be scaled and then truncated.
        self.depth_scale = 1000.0
        # Depth values larger than depth_trunc gets truncated to 0. The depth values will first be scaled and then truncated.
        self.depth_trunc = 3.0
        # Whether to convert RGB image to intensity image.
        self.convert_rgb_to_intensity = False

        # Used to store the data used for constructing TSDF
        self.sensor_data = deque()
        self.color_images = []
        self.depth_images = []
        self.rgb_poses = []
        self.prev_pose_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self.prev_pose_tran = np.array([0.0, 0.0, 0.0])

        self.tsdf_integration_data = deque()
        self.integration_done = True
        self.live_integration = False
        self.mesh_pub = None
        self.tsdf_volume_pub = None

        self.record = False
        self.frame_count = 0
        self.processed_frame_count = 0
        self.reconstructed_frame_count = 0

        self.declare_parameter("depth_image_topic")
        self.declare_parameter("color_image_topic")
        self.declare_parameter("camera_info_topic")
        self.declare_parameter("cache_count", 10)
        self.declare_parameter("slop", 0.01)

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
        allow_headerless = False

        self.get_logger().info("depth_image_topic - " + self.depth_image_topic)
        self.get_logger().info("color_image_topic - " + self.color_image_topic)
        self.get_logger().info("camera_info_topic - " + self.camera_info_topic)

        self.depth_sub = Subscriber(self, Image, self.depth_image_topic)
        self.color_sub = Subscriber(self, Image, self.color_image_topic)
        self.tss = ApproximateTimeSynchronizer([self.depth_sub, self.color_sub], self.cache_count, self.slop,
                                               allow_headerless)
        self.tss.registerCallback(self.cameraCallback)

        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.cameraInfoCallback, 10)

        self.mesh_pub = self.create_publisher(Marker, "open3d_mesh", 10)

        self.start_server = self.create_service(StartYakReconstruction, 'start_reconstruction',
                                                self.startYakReconstructionCallback)
        self.stop_server = self.create_service(StopYakReconstruction, 'stop_reconstruction',
                                               self.stopYakReconstructionCallback)

        self.tsdf_volume_pub = self.create_publisher(Marker, "tsdf_volume", 10)

    def archiveData(self, path_output):
        path_depth = join(path_output, "depth")
        path_color = join(path_output, "color")
        path_pose = join(path_output, "pose")

        make_folder_keep_contents(path_output)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)
        make_clean_folder(path_pose)

        for s in range(len(self.color_images)):
            # Save your OpenCV2 image as a jpeg
            o3d.io.write_image("%s/%06d.png" % (path_depth, s), self.depth_images[s])
            o3d.io.write_image("%s/%06d.jpg" % (path_color, s), self.color_images[s])
            write_pose("%s/%06d.pose" % (path_pose, s), self.rgb_poses[s])
            save_intrinsic_as_json(join(path_output, "camera_intrinsic.json"), self.intrinsics)


    def startYakReconstructionCallback(self, req, res):
        self.get_logger().info(" Start Reconstruction")

        self.color_images.clear()
        self.depth_images.clear()
        self.rgb_poses.clear()
        self.sensor_data.clear()
        self.tsdf_integration_data.clear()
        self.prev_pose_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self.prev_pose_tran = np.array([0.0, 0.0, 0.0])

        if (req.tsdf_params.min_box_values.x == req.tsdf_params.max_box_values.x and
                req.tsdf_params.min_box_values.y == req.tsdf_params.max_box_values.y and
                req.tsdf_params.min_box_values.z == req.tsdf_params.max_box_values.z):
            crop_mesh = False
        else:
            crop_mesh = True
            min_bound = np.asarray(
                [req.tsdf_params.min_box_values.x, req.tsdf_params.min_box_values.y, req.tsdf_params.min_box_values.z])
            max_bound = np.asarray(
                [req.tsdf_params.max_box_values.x, req.tsdf_params.max_box_values.y, req.tsdf_params.max_box_values.z])
            crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

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

    def stopYakReconstructionCallback(self, req, res):
        self.get_logger().info("Stop Reconstruction")
        self.record = False

        while not self.integration_done:
            self.create_rate(1).sleep()

        print("Generating mesh")
        if self.tsdf_volume is None:
            res.success = False
            return res
        if not self.live_integration:
            while len(self.tsdf_integration_data) > 0:
                data = self.tsdf_integration_data.popleft()
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(data[1], data[0], self.depth_scale, self.depth_trunc,
                                                                          False)
                self.tsdf_volume.integrate(rgbd, self.intrinsics, np.linalg.inv(data[2]))
        mesh = self.tsdf_volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        if self.crop_mesh:
            cropped_mesh = mesh.crop(self.crop_box)
        else:
            cropped_mesh = mesh

        mesh_filepath = join(req.results_directory, "integrated.ply")
        o3d.io.write_triangle_mesh(mesh_filepath, cropped_mesh, False, True)
        mesh_msg = meshToRos(cropped_mesh)
        mesh_msg.header.stamp = self.get_clock().now().to_msg()
        mesh_msg.header.frame_id = self.relative_frame
        self.mesh_pub.publish(mesh_msg)
        print("Mesh Generated")

        if (req.archive_directory != ""):
            self.get_logger().info("Archiving data to " + req.archive_directory)
            self.archiveData(req.archive_directory)
            archive_mesh_filepath = join(req.archive_directory, "integrated.ply")
            o3d.io.write_triangle_mesh(archive_mesh_filepath, mesh, False, True)

        self.get_logger().info("DONE")
        res.success = True
        res.mesh_filepath = mesh_filepath
        return res

    def cameraCallback(self, depth_image_msg, rgb_image_msg):
        if self.record:
            try:
                # Convert your ROS Image message to OpenCV2
                # TODO: Generalize image type
                cv2_depth_img = self.bridge.imgmsg_to_cv2(depth_image_msg, "16UC1")
                cv2_rgb_img = self.bridge.imgmsg_to_cv2(rgb_image_msg, rgb_image_msg.encoding)
            except CvBridgeError:
                self.get_logger().error("Error converting ros msg to cv img")
                return
            else:
                self.sensor_data.append(
                    [o3d.geometry.Image(cv2_depth_img), o3d.geometry.Image(cv2_rgb_img), rgb_image_msg.header.stamp])
                if (self.frame_count > 30):
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

                    # TODO: Testing if this is a good practice, min jump to accept data
                    if (tran_dist > self.translation_distance) or (rot_dist > self.rotational_distance):
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
                            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(data[1], data[0], self.depth_scale,
                                                                                      self.depth_trunc, False)
                            self.tsdf_volume.integrate(rgbd, self.intrinsics, np.linalg.inv(rgb_pose))
                            self.integration_done = True
                            if self.processed_frame_count % 50 == 0:
                                mesh = self.tsdf_volume.extract_triangle_mesh()
                                if self.crop_mesh:
                                    cropped_mesh = mesh.crop(self.crop_box)
                                else:
                                    cropped_mesh = mesh
                                mesh_msg = meshToRos(cropped_mesh)
                                mesh_msg.header.stamp = self.get_clock().now().to_msg()
                                mesh_msg.header.frame_id = self.relative_frame
                                self.mesh_pub.publish(mesh_msg)
                        else:
                            self.tsdf_integration_data.append([data[0], data[1], rgb_pose])
                        self.processed_frame_count += 1

                self.frame_count += 1

    def cameraInfoCallback(self, camera_info):
        self.intrinsics = getIntrinsicsFromMsg(camera_info)


def main(args=None):
    rclpy.init(args=args)
    open3d_yak = Open3dYak()
    rclpy.spin(open3d_yak)
    open3d_yak.destroy_node()
    rclpy.shutdown()

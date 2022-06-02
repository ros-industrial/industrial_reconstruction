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

import rclpy
from rclpy.node import Node
import open3d as o3d
from os import listdir
from os.path import join
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge

def read_pose(filename):
    f = open(filename, "r")
    content = f.readlines()
    return np.array(list(map(float, (''.join(content[0:4])).strip().split()))).reshape((4, 4))

class ArchivePlayer(Node):

    def __init__(self):
        super().__init__('industrial_reconstruction_archive_player')

        self.declare_parameter("depth_image_topic")
        self.declare_parameter("color_image_topic")
        self.declare_parameter("camera_info_topic")
        self.declare_parameter("relative_frame")
        self.declare_parameter("tracking_frame")
        self.declare_parameter("image_directory")
        self.declare_parameter("pub_rate")

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
            self.pose_rel_frame = str(self.get_parameter('relative_frame').value)
        except:
            self.get_logger().error("Failed to load relative_frame parameter")
        try:
            self.pose_track_frame = str(self.get_parameter('tracking_frame').value)
        except:
            self.get_logger().error("Failed to load tracking_frame parameter")
        try:
            self.img_dir = str(self.get_parameter('image_directory').value)
        except:
            self.get_logger().error("Failed to load image_directory parameter")
        try:
            self.pub_rate = int(self.get_parameter('pub_rate').value)
        except:
            self.get_logger().error('Failed to load pub_rate parameter')

        for parameter in self._parameters.values():
            print(parameter.name, ":", parameter.value)

        self.depth_pub = self.create_publisher(Image, self.depth_image_topic, 10)
        self.rgb_pub = self.create_publisher(Image, self.color_image_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)
        self.publisher_tf = self.create_publisher(TFMessage, 'tf', 10)

        self.start_server = self.create_service(Trigger, 'start_publishing', self.startPublishingCallback)
        self.stop_server = self.create_service(Trigger, 'stop_publishing', self.stopPublishingCallback)
        self.restart_server = self.create_service(Trigger, 'restart_publishing', self.restartPublishingCallback)

        self.time = self.create_timer(1.0/float(self.pub_rate), self.timerCallback)

        camera_intrinsic_fp = join(self.img_dir, 'camera_intrinsic.json')
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(camera_intrinsic_fp)

        self.camera_intrinsic_msg = CameraInfo()
        self.camera_intrinsic_msg.header.stamp = self.get_clock().now().to_msg()
        self.camera_intrinsic_msg.header.frame_id = self.pose_track_frame
        self.camera_intrinsic_msg.height = intrinsic.height
        self.camera_intrinsic_msg.width = intrinsic.width
        self.camera_intrinsic_msg.k = [intrinsic.intrinsic_matrix[0][0], 0.0, intrinsic.intrinsic_matrix[0][2],
                                       0.0, intrinsic.intrinsic_matrix[1][1], intrinsic.intrinsic_matrix[1][2],
                                       0.0, 0.0, intrinsic.intrinsic_matrix[2][2]]

        self.get_logger().info("depth_image_topic - " + self.depth_image_topic)
        self.get_logger().info("color_image_topic - " + self.color_image_topic)
        self.get_logger().info("camera_info_topic - " + self.camera_info_topic)

        self.current_index = 0
        self.publishing = False

        self.color_dir = join(self.img_dir, 'color')
        self.depth_dir = join(self.img_dir, 'depth')
        self.pose_dir = join(self.img_dir, 'pose')
        self.num_imgs = len(listdir(self.color_dir))

        self.tfmsg = TFMessage()
        self.tfmsg.transforms.append(TransformStamped())

        self.bridge = CvBridge()


    def startPublishingCallback(self, req, res):
        global publishing
        self.publishing = True

        return res


    def stopPublishingCallback(self, req, res):
        global publishing
        self.publishing = False

        return res


    def restartPublishingCallback(self, req, res):
        global publishing, current_index
        current_index = 0

        return res

    def timerCallback(self):
        if self.publishing:
            self.current_index += 1
            if self.current_index >= self.num_imgs:
                self.current_index = 0
            color_index_string = f"{self.current_index:06d}" + ".jpg"
            depth_index_string = f"{self.current_index:06d}" + ".png"
            pose_index_string = f"{self.current_index:06d}" + ".pose"
            color_img_fp = join(self.color_dir, color_index_string)
            depth_img_fp = join(self.depth_dir, depth_index_string)
            pose_fp = join(self.pose_dir, pose_index_string)
            color_img = o3d.io.read_image(color_img_fp)
            depth_img = o3d.io.read_image(depth_img_fp)
            pose = read_pose(pose_fp)
            np_pose = np.asarray(pose)
            rotation = R.from_matrix(np_pose[0:3, 0:3])
            quat = rotation.as_quat()
            curr_time = self.get_clock().now().to_msg()
            self.tfmsg.transforms[0].header.stamp = curr_time
            self.tfmsg.transforms[0].header.frame_id = self.pose_rel_frame
            self.tfmsg.transforms[0].child_frame_id = self.pose_track_frame
            self.tfmsg.transforms[0].transform.translation.x = np_pose[0, 3]
            self.tfmsg.transforms[0].transform.translation.y = np_pose[1, 3]
            self.tfmsg.transforms[0].transform.translation.z = np_pose[2, 3]
            self.tfmsg.transforms[0].transform.rotation.w = quat[3]
            self.tfmsg.transforms[0].transform.rotation.x = quat[0]
            self.tfmsg.transforms[0].transform.rotation.y = quat[1]
            self.tfmsg.transforms[0].transform.rotation.z = quat[2]
            self.publisher_tf.publish(self.tfmsg)

            self.camera_intrinsic_msg.header.stamp = curr_time
            self.camera_info_pub.publish(self.camera_intrinsic_msg)
            image_message_color = self.bridge.cv2_to_imgmsg(np.asarray(color_img), encoding='rgb8')
            image_message_color.header.stamp = curr_time
            image_message_color.header.frame_id = self.pose_track_frame
            image_message_depth = self.bridge.cv2_to_imgmsg(np.asarray(depth_img), encoding='16UC1')
            image_message_depth.header.stamp = curr_time
            image_message_depth.header.frame_id = self.pose_track_frame
            self.rgb_pub.publish(image_message_color)
            self.depth_pub.publish(image_message_depth)


def main(args=None):

    rclpy.init(args=args)
    ir_arcihve_player = ArchivePlayer()
    rclpy.spin(ir_arcihve_player)
    ir_arcihve_player.destroy_node()
    rclpy.shutdown()

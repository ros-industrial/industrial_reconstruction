# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
import rospkg
import rospy
import tf
import open3d as o3d
import numpy as np
from collections import deque
from os import makedirs, listdir
from os.path import exists, join, isfile
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from open3d_interface.utility.file import make_clean_folder, write_pose, read_pose, save_intrinsic_as_json
from open3d_interface.srv import StartYakReconstruction, StopYakReconstruction
from open3d_interface.srv import StartYakReconstructionResponse, StopYakReconstructionResponse
from open3d_interface.utility.ros import getIntrinsicsFromMsg

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

bridge = CvBridge()

tf_listener = None

tsdf_volume = None
intrinsics = None
tracking_frame = ''
relative_frame = ''

####################################################################
# See Open3d function create_from_color_and_depth for more details #
####################################################################
# The ratio to scale depth values. The depth values will first be scaled and then truncated.
depth_scale = 1000.0
# Depth values larger than depth_trunc gets truncated to 0. The depth values will first be scaled and then truncated.
depth_trunc = 3.0
# Whether to convert RGB image to intensity image.
convert_rgb_to_intensity = False

# Used to store the data used for constructing TSDF
sensor_data = deque()
color_images = []
depth_images = []
rgb_poses = []

record = False
frame_count = 0
processed_frame_count = 0

def archiveData(path_output):
  global depth_images, color_images, rgb_poses, intrinsics
  path_depth = join(path_output, "depth")
  path_color = join(path_output, "color")
  path_pose = join(path_output, "pose")

  make_clean_folder(path_output)
  make_clean_folder(path_depth)
  make_clean_folder(path_color)
  make_clean_folder(path_pose)

  for s in range(len(color_images)):
    # Save your OpenCV2 image as a jpeg
    o3d.io.write_image("%s/%06d.png" % (path_depth, s), depth_images[s])
    o3d.io.write_image("%s/%06d.jpg" % (path_color, s), color_images[s])
    write_pose("%s/%06d.pose" % (path_pose, s), rgb_poses[s])
    save_intrinsic_as_json(join(path_output, "camera_intrinsic.json"), intrinsics)


def startYakReconstructionCallback(req):
  global record, frame_count, processed_frame_count, relative_frame, tracking_frame
  global color_images, depth_images, rgb_poses, sensor_data, tsdf_volume
  global depth_scale, depth_trunc, convert_rgb_to_intensity
  rospy.loginfo(rospy.get_caller_id() + ": Start Reconstruction")

  color_images.clear()
  depth_images.clear()
  rgb_poses.clear()
  sensor_data.clear()

  frame_count = 0
  processed_frame_count = 0

  tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
      voxel_length=req.tsdf_params.voxel_length,
      sdf_trunc=req.tsdf_params.sdf_trunc,
      color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

  depth_scale = req.rgbd_params.depth_scale
  depth_trunc = req.rgbd_params.depth_trunc
  convert_rgb_to_intensity = req.rgbd_params.convert_rgb_to_intensity
  tracking_frame = req.tracking_frame
  relative_frame = req.relative_frame


  record = True
  return StartYakReconstructionResponse(True)

def stopYakReconstructionCallback(req):
  global record, tsdf_volume, depth_images, color_images, rgb_poses, depth_scale, depth_trunc, intrinsics

  rospy.loginfo(rospy.get_caller_id() + ": Stop Reconstruction")
  record = False

  for s in range(len(color_images)):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_images[s], depth_images[s], depth_scale, depth_trunc, False)
    tsdf_volume.integrate(rgbd, intrinsics, np.linalg.inv(rgb_poses[s]))

  mesh = tsdf_volume.extract_triangle_mesh()
  mesh.compute_vertex_normals()

  if (req.archive_directory != ""):
    rospy.loginfo(rospy.get_caller_id() + ": Archiving data to " + req.archive_directory)
    archiveData(req.archive_directory)
    mesh_name = join(req.archive_directory, "integrated.ply")
    o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)

  return StopYakReconstructionResponse(True,'')


def cameraCallback(depth_image_msg, rgb_image_msg):
  global frame_count, processed_frame_count, record, tracking_frame, relative_frame, tf_listener
  global color_images, depth_images, rgb_poses, intrinsics

  if record:
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_depth_img = bridge.imgmsg_to_cv2(depth_image_msg, "16UC1")
        cv2_rgb_img = bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
    except CvBridgeError:
        print(e)
    else:
        # Get camera intrinsic from camera info
        if frame_count == 0:
          camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
          intrinsics = getIntrinsicsFromMsg(camera_info)

        sensor_data.append([o3d.geometry.Image(cv2_depth_img), o3d.geometry.Image(cv2_rgb_img), rgb_image_msg.header.stamp])
        if (frame_count > 30):
          data = sensor_data.popleft()
          (rgb_t,rgb_r) = tf_listener.lookupTransform(relative_frame, tracking_frame, data[2])
          rgb_pose = tf.transformations.quaternion_matrix(rgb_r)
          rgb_pose[0,3] = rgb_t[0]
          rgb_pose[1,3] = rgb_t[1]
          rgb_pose[2,3] = rgb_t[2]

          depth_images.append(data[0])
          color_images.append(data[1])
          rgb_poses.append(rgb_pose)

          processed_frame_count += 1

        frame_count += 1


def main():
  global camera_info_topic, tf_listener, tracking_frame, world_frame

  rospy.init_node('open3d_tsdf_rgb_recorder', anonymous=True)

  # Create TF listener
  tf_listener = tf.TransformListener()

  # Get parameters
  depth_image_topic = rospy.get_param('~depth_image_topic')
  color_image_topic = rospy.get_param('~color_image_topic')
  camera_info_topic = rospy.get_param('~camera_info_topic')
  cache_count = rospy.get_param('~cache_count', 10)
  slop = rospy.get_param('~slop', 0.01) # The delay (in seconds) with which messages can be synchronized.
  allow_headerless = False #allow storing headerless messages with current ROS time instead of timestamp

  rospy.loginfo(rospy.get_caller_id() + ": depth_image_topic - " + depth_image_topic)
  rospy.loginfo(rospy.get_caller_id() + ": color_image_topic - " + color_image_topic)
  rospy.loginfo(rospy.get_caller_id() + ": camera_info_topic - " + camera_info_topic)

  depth_sub = Subscriber(depth_image_topic, Image)
  color_sub = Subscriber(color_image_topic, Image)
  tss = ApproximateTimeSynchronizer([depth_sub, color_sub], cache_count, slop, allow_headerless)
  tss.registerCallback(cameraCallback)

  start_server = rospy.Service('start_reconstruction', StartYakReconstruction, startYakReconstructionCallback)
  stop_server = rospy.Service('stop_reconstruction', StopYakReconstruction, stopYakReconstructionCallback)

  rospy.spin()

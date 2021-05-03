# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
import rospkg
import rospy
import tf
import open3d as o3d
import numpy as np
from os import makedirs, listdir
from os.path import exists, join, isfile
from sensor_msgs.msg import Image, CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber
from open3d_interface.utility.file import make_clean_folder, write_pose, read_pose
from open3d_interface.srv import StartRecording, StopRecording, ReconstructTSDFSurface
from open3d_interface.srv import StartRecordingResponse, StopRecordingResponse, ReconstructTSDFSurfaceResponse
from open3d_interface.utility.ros import save_camera_info_intrinsic_as_json

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

bridge = CvBridge()

tf_listener = None
path_output = "/tmp"
path_depth = join(path_output, "depth")
path_color = join(path_output, "color")
path_pose = join(path_output, "pose")

camera_info_topic = '/camera/rgb/camera_info'
tracking_frame = ''
world_frame = ''

record = False
frame_count = 0

def startRecordingCallback(req):
  global record, path_output, path_depth, path_color, path_pose, frame_count

  rospy.loginfo(rospy.get_caller_id() + "Start Recording")

  path_output = req.directory
  path_depth = join(path_output, "depth")
  path_color = join(path_output, "color")
  path_pose = join(path_output, "pose")

  make_clean_folder(path_output)
  make_clean_folder(path_depth)
  make_clean_folder(path_color)
  make_clean_folder(path_pose)

  frame_count = 0
  record = True

  return StartRecordingResponse(True)

def stopRecordingCallback(req):
  global record
  rospy.loginfo(rospy.get_caller_id() + "Stop Recording")
  record = False

  return StopRecordingResponse(True)


def register_known_rgbd_pair(source_rgbd_image, target_rgbd_image, odo_init, intrinsic, req):
    option = o3d.pipelines.odometry.OdometryOption()
    option.max_depth_diff = req.max_depth_diff
    option.min_depth = req.min_depth
    option.max_depth = req.max_depth

    [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                              source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                              o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                              option)

    return [success, trans, info]

def createPoseGraph(pose_array, rgbd_array, intrinsic, req):
  pose_graph = o3d.pipelines.registration.PoseGraph()
  pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose_array[0]))
  for s in range(len(rgbd_array)):
      for t in range(s + 1, len(rgbd_array)):
          # odometry
          odo_init = np.linalg.inv(pose_array[s]).dot(pose_array[t])

          # TODO: Need to understand what info is because we may be able to just use the pose information directly
          if t == s + 1:
              print("RGBD odometry matching between frame : %d and %d" % (s, t))
              [success, trans, info] = register_known_rgbd_pair(rgbd_array[s], rgbd_array[t], odo_init, intrinsic, req)
              trans_odometry = pose_array[t]
              pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry))
              pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=False))

          # TODO: Should also leverage the pose information to determine if a keyframe
          # keyframe loop closure
#          if s % req.n_keyframes_per_n_frame == 0  and t % req.n_keyframes_per_n_frame == 0:
          if s % 5 == 0  and t % 5 == 0:
              print("RGBD keyframes matching between frame : %d and %d" % (s, t))
              [success, trans, info] = register_known_rgbd_pair(rgbd_array[s], rgbd_array[t], odo_init, intrinsic, req)
              if success:
                  pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=True))

  return pose_graph

def reconstructTSDFSurfaceCallback(req):
  # Get number images
  pose_dir = join(req.path_dataset, "pose")
  num_frames = len([name for name in listdir(pose_dir) if isfile(join(pose_dir, name))])
  print("Number of frames captured: {:d}".format(num_frames))

  # Get intrinsic data
  if req.path_intrinsic:
      intrinsic = o3d.io.read_pinhole_camera_intrinsic(req.path_intrinsic)
  else:
      intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

  # Load data
  color_array = []
  depth_array = []
  rgbd_array = []
  rgbd_color_array = []
  pose_array = []
  for i in range(num_frames):
      color = o3d.io.read_image("{:s}/color/{:06d}.jpg".format(req.path_dataset,i))
      depth = o3d.io.read_image("{:s}/depth/{:06d}.png".format(req.path_dataset,i))
      pose = read_pose("{:s}/pose/{:06d}.pose".format(req.path_dataset,i))
      color_array.append(color)
      depth_array.append(depth)
      pose_array.append(pose)

      # Used for building pose graph
      rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, req.depth_scale, req.depth_trunc, req.convert_rgb_to_intensity)
      rgbd_array.append(rgbd)

      # Used for scene integration
      rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, req.depth_scale, req.depth_trunc, False)
      rgbd_color_array.append(rgbd)

  # Create integration volume
  volume = o3d.pipelines.integration.ScalableTSDFVolume(
      voxel_length=req.voxel_length,
      sdf_trunc=req.sdf_trunc,
      color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

  if req.global_optimization:
    # Create initial pose graph
    pose_graph = createPoseGraph(pose_array, rgbd_array, intrinsic, req)

    # Optimize pose graph
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=req.max_correspondence_distance,
        edge_prune_threshold=req.edge_prune_threshold,
        preference_loop_closure=req.preference_loop_closure,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)

    # Integrate scene
    for s in range(len(pose_graph.nodes)):
      volume.integrate(rgbd_color_array[s], intrinsic, np.linalg.inv(pose_graph.nodes[s].pose))
  else:
    # Integrate scene
    for s in range(num_frames):
      volume.integrate(rgbd_color_array[s], intrinsic, np.linalg.inv(pose_array[s]))

  # Extract Mesh and save to file
  mesh = volume.extract_triangle_mesh()
  mesh.compute_vertex_normals()
  if req.debug_mode:
      o3d.visualization.draw_geometries([mesh])

  mesh_name = join(req.path_dataset, "integrated.ply")
  o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)

  return ReconstructTSDFSurfaceResponse(True, mesh_name)


def cameraCallback(depth_image_msg, rgb_image_msg):
  global frame_count, record, path_output, path_depth, path_color, path_pose, tracking_frame, world_frame

  if record:
    rospy.loginfo(rospy.get_caller_id() + "I heard depth image")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_depth_img = bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")
        cv2_rgb_img = bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
        (rgb_t,rgb_r) = tf_listener.lookupTransform(world_frame, tracking_frame, rgb_image_msg.header.stamp)
    except CvBridgeError:
        print(e)
    else:
        # Get camera intrinsic from camera info
        if frame_count == 0:
            camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
            save_camera_info_intrinsic_as_json(join(path_output, "camera_intrinsic.json"), camera_info)

        # Save your OpenCV2 image as a jpeg
        cv2.imwrite("%s/%06d.png" % (path_depth, frame_count), cv2_depth_img)
        cv2.imwrite("%s/%06d.jpg" % (path_color, frame_count), cv2_rgb_img)

        rgb_pose = tf.transformations.quaternion_matrix(rgb_r)
        rgb_pose[0,3] = rgb_t[0]
        rgb_pose[1,3] = rgb_t[1]
        rgb_pose[2,3] = rgb_t[2]

        write_pose("%s/%06d.pose" % (path_pose, frame_count), rgb_pose)
        frame_count += 1

def main():
  global camera_info_topic, tf_listener, tracking_frame, world_frame

  rospy.init_node('open3d_tsdf_rgb_recorder', anonymous=True)

  # Create TF listener
  tf_listener = tf.TransformListener()

  # TODO: Make these ros parameters
  depth_image_topic = '/camera/depth_registered/image'
  rgb_image_topic = '/camera/rgb/image_raw'
  camera_info_topic = '/camera/rgb/camera_info'
  tracking_frame = 'camera_rgb_optical_frame'
  world_frame = 'camera_link'
  cache_count = 10
  slop = 0.01 # The delay (in seconds) with which messages can be synchronized.
  allow_headerless = False #allow storing headerless messages with current ROS time instead of timestamp

  depth_sub = Subscriber(depth_image_topic, Image)
  rgb_sub = Subscriber(rgb_image_topic, Image)
  tss = ApproximateTimeSynchronizer([depth_sub, rgb_sub], cache_count, slop, allow_headerless)
  tss.registerCallback(cameraCallback)

  start_server = rospy.Service('start_recording', StartRecording, startRecordingCallback)
  stop_server = rospy.Service('stop_recording', StopRecording, stopRecordingCallback)
  stop_server = rospy.Service('reconstruct', ReconstructTSDFSurface, reconstructTSDFSurfaceCallback)

  rospy.spin()

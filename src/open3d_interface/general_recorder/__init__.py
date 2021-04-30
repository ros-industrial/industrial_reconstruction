# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
import rospkg
import rospy
import time
import datetime
from os import makedirs
from os.path import exists, join
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

from open3d_interface.reconstruction.initialize_config import initialize_config
from open3d_interface.utility.file import check_folder_structure, make_clean_folder

from open3d_interface.srv import StartRecording, StopRecording, ReconstructSurface
from open3d_interface.srv import StartRecordingResponse, StopRecordingResponse, ReconstructSurfaceResponse

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

bridge = CvBridge()

path_output = "/tmp"
path_depth = join(path_output, "depth")
path_color = join(path_output, "color")

record = False
frame_count = 0

def startRecording(req):
  global record, path_output, path_depth, path_color, frame_count

  rospy.loginfo(rospy.get_caller_id() + "Start Recording")

  path_output = req.directory
  path_depth = join(path_output, "depth")
  path_color = join(path_output, "color")
  path_bag = join(path_output, "realsense.bag")

  make_clean_folder(path_output)
  make_clean_folder(path_depth)
  make_clean_folder(path_color)

  frame_count = 0
  record = True

  return StartRecordingResponse(True)

def stopRecording(req):
  global record
  rospy.loginfo(rospy.get_caller_id() + "Stop Recording")
  record = False
  return StopRecordingResponse(True)

def reconstruct(req):
  rospy.loginfo(rospy.get_caller_id() + "Reconstruct Surface")

  if not req.make and not req.register and not req.refine and not req.integrate:
     return ReconstructSurfaceResponse(False)

  config = {}
  if not req.use_default_settings:
    config["name"] = req.name
    config["max_depth"] = req.max_depth
    config["voxel_size"] = req.voxel_size
    config["max_depth_diff"] = req.max_depth_diff
    config["preference_loop_closure_odometry"] = req.preference_loop_closure_odometry
    config["preference_loop_closure_registration"] = req.preference_loop_closure_registration
    config["tsdf_cubic_size"] = req.tsdf_cubic_size
    config["global_registration"] = req.global_registration
    config["python_multi_threading"] = req.python_multi_threading

  config["path_dataset"] = req.path_dataset
  config["path_intrinsic"] = req.path_intrinsic
  config["debug_mode"] = req.debug_mode

  # check folder structure
  initialize_config(config)
  check_folder_structure(config["path_dataset"])
  assert config is not None

  print("====================================")
  print("Configuration")
  print("====================================")
  for key, val in config.items():
      print("%40s : %s" % (key, str(val)))

  times = [0, 0, 0, 0]
  if req.make:
      start_time = time.time()
      import open3d_interface.reconstruction.make_fragments as make_fragments
      make_fragments.run(config)
      times[0] = time.time() - start_time
  if req.register:
      start_time = time.time()
      import open3d_interface.reconstruction.register_fragments as register_fragments
      register_fragments.run(config)
      times[1] = time.time() - start_time
  if req.refine:
      start_time = time.time()
      import open3d_interface.reconstruction.refine_registration as refine_registration
      refine_registration.run(config)
      times[2] = time.time() - start_time
  if req.integrate:
      start_time = time.time()
      import open3d_interface.reconstruction.integrate_scene as integrate_scene
      integrate_scene.run(config)
      times[3] = time.time() - start_time

  print("====================================")
  print("Elapsed time (in h:m:s)")
  print("====================================")
  print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
  print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
  print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
  print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
  print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
  sys.stdout.flush()

  return ReconstructSurfaceResponse(True, path_output + "/scene/integrated.ply")

def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)

def cameraCallback(depth_image_msg, rgb_image_msg):
  global frame_count, record, path_output, path_depth, path_color

  if record:
    rospy.loginfo(rospy.get_caller_id() + "I heard depth image")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_depth_img = bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")
        cv2_rgb_img = bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
    except CvBridgeError:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
#        if frame_count == 0:
#            save_intrinsic_as_json(join(path_output, "camera_intrinsic.json"), color_frame)
        cv2.imwrite("%s/%06d.png" % (path_depth, frame_count), cv2_depth_img)
        cv2.imwrite("%s/%06d.jpg" % (path_color, frame_count), cv2_rgb_img)
        print("Saved color + depth image %06d" % frame_count)
        frame_count += 1

def main():
  rospy.init_node('open3d_recorder', anonymous=True)

  # TODO: Make these ros parameters
  depth_image_topic = '/camera/depth_registered/image'
  rgb_image_topic = '/camera/rgb/image_raw'
  cache_count = 10
  slop = 0.1 # The delay (in seconds) with which messages can be synchronized.
  allow_headerless = False #allow storing headerless messages with current ROS time instead of timestamp

  depth_sub = Subscriber(depth_image_topic, Image)
  rgb_sub = Subscriber(rgb_image_topic, Image)
  tss = ApproximateTimeSynchronizer([depth_sub, rgb_sub], cache_count, slop, allow_headerless)
  tss.registerCallback(cameraCallback)

  start_server = rospy.Service('start_recording', StartRecording, startRecording)
  stop_server = rospy.Service('stop_recording', StopRecording, stopRecording)
  stop_server = rospy.Service('reconstruct', ReconstructSurface, reconstruct)

  rospy.spin()

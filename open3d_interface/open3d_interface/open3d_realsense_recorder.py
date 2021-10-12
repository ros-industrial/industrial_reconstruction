# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/sensors/realsense_recorder.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import numpy as np
import cv2
import shutil
import open3d as o3d
import rospkg
import rospy

from os import makedirs
from os.path import exists, join
from enum import IntEnum
from open3d_interface.utility.file import make_clean_folder
from open3d_interface.srv import StartRecording, StopRecording
from open3d_interface.srv import StartRecordingResponse, StopRecordingResponse
from open3d_interface.utility.ros import reconstructCallback

path_output = "/tmp"
path_depth = join(path_output, "depth")
path_color = join(path_output, "color")
path_bag = join(path_output, "realsense.bag")

record_imgs = True
record_rosbag = False
record = False

pipeline = None
frame_count = 0

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

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


def startRecordingCallback(req):
  global record, record_rosbag, pipeline, frame_count
  global path_output, path_depth, path_color, path_bag

  rospy.loginfo(rospy.get_caller_id() + "Start Recording")

  path_output = req.directory
  path_depth = join(path_output, "depth")
  path_color = join(path_output, "color")
  path_bag = join(path_output, "realsense.bag")

  make_clean_folder(path_output)
  make_clean_folder(path_depth)
  make_clean_folder(path_color)

  frame_count = 0

  # Create a pipeline
  if pipeline is None:
    pipeline = rs.pipeline()

  #Create a config and configure the pipeline to stream
  #  different resolutions of color and depth streams
  config = rs.config()

  if record_imgs or record_rosbag:
      # note: using 640 x 480 depth resolution produces smooth depth boundaries
      #       using rs.format.bgr8 for color image format for OpenCV based image visualization
      config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
      config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
      if record_rosbag:
          config.enable_record_to_file(path_bag)

  # Start streaming
  profile = pipeline.start(config)
  depth_sensor = profile.get_device().first_depth_sensor()

  # Using preset HighAccuracy for recording
  if record_rosbag or record_imgs:
      depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

  # Getting the depth sensor's depth scale (see rs-align example for explanation)
  depth_scale = depth_sensor.get_depth_scale()

  # We will not display the background of objects more than
  #  clipping_distance_in_meters meters away
  clipping_distance_in_meters = 3  # 3 meter
  clipping_distance = clipping_distance_in_meters / depth_scale

  # Create an align object
  # rs.align allows us to perform alignment of depth frames to others frames
  # The "align_to" is the stream type to which we plan to align depth frames.
  align_to = rs.stream.color
  align = rs.align(align_to)

  record = True
  return StartRecordingResponse(True)

def stopRecordingCallback(req):
  global record
  rospy.loginfo(rospy.get_caller_id() + "Stop Recording")
  record = False
  return StopRecordingResponse(True)

def main():
  global pipeline, output_folder, path_depth, path_color, frame_count, record, clipping_distance

  rospy.init_node('open3d_realsense_recorder', anonymous=True)

  start_server = rospy.Service('start_recording', StartRecording, startRecordingCallback)
  stop_server = rospy.Service('stop_recording', StopRecording, stopRecordingCallback)
  stop_server = rospy.Service('reconstruct', ReconstructSurface, reconstructCallback)

  # Streaming loop
  try:
      while True:
          if rospy.is_shutdown():
              break

          if not record:
            continue

          # Get frameset of color and depth
          frames = pipeline.wait_for_frames()

          # Align the depth frame to color frame
          aligned_frames = align.process(frames)

          # Get aligned frames
          aligned_depth_frame = aligned_frames.get_depth_frame()
          color_frame = aligned_frames.get_color_frame()

          # Validate that both frames are valid
          if not aligned_depth_frame or not color_frame:
              continue

          depth_image = np.asanyarray(aligned_depth_frame.get_data())
          color_image = np.asanyarray(color_frame.get_data())

          if record_imgs:
              if frame_count == 0:
                  save_intrinsic_as_json(join(path_output, "camera_intrinsic.json"), color_frame)
              cv2.imwrite("%s/%06d.png" % (path_depth, frame_count), depth_image)
              cv2.imwrite("%s/%06d.jpg" % (path_color, frame_count), color_image)
              print("Saved color + depth image %06d" % frame_count)
              frame_count += 1

  finally:
      if pipeline is not None:
        pipeline.stop()

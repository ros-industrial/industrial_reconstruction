cmake_minimum_required(VERSION 2.8.3)
project(open3d_interface)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

catkin_python_setup()

add_message_files(
  DIRECTORY msg
  FILES
    ComputeRGBDOdometryParams.msg
    GlobalOptimizationParams.msg
    RGBDImageParams.msg
    TSDFVolumeParams.msg
)

add_service_files(
  DIRECTORY srv
  FILES
    StartRecording.srv
    StopRecording.srv
    RunReconstructionSystem.srv
    RunTSDFReconstruction.srv
    StartYakReconstruction.srv
    StopYakReconstruction.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs)

include_directories(include ${catkin_INCLUDE_DIRS})

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  PROGRAMS
    scripts/open3d_general_recorder
    scripts/open3d_realsense_recorder
    scripts/open3d_reconstruction
    scripts/open3d_yak
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

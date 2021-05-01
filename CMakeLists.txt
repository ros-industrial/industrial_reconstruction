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

add_service_files(
  DIRECTORY srv
  FILES
    StartRecording.srv
    StopRecording.srv
    ReconstructSurface.srv
    ReconstructTSDFSurface.srv
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

install(PROGRAMS scripts/open3d_general_recorder DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(PROGRAMS scripts/open3d_realsense_recorder DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

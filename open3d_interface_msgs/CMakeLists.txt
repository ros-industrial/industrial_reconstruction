cmake_minimum_required(VERSION 3.5)
project(open3d_interface_msgs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(msg_files
    "msg/RGBDImageParams.msg"
    "msg/TSDFVolumeParams.msg"
    "msg/NormalFilterParams.msg")

set(srv_files
    "srv/StartReconstruction.srv"
    "srv/StopReconstruction.srv")

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs)


ament_export_dependencies(rosidl_default_runtime)

ament_package()

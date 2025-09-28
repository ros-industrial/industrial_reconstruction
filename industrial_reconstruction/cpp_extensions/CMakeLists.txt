cmake_minimum_required(VERSION 3.16)
project(industrial_reconstruction_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files
set(SOURCES
    src/image_buffer.cpp
    src/pose_calculator.cpp
    src/memory_manager.cpp
    src/bindings.cpp
)

# Create the Python module
pybind11_add_module(industrial_reconstruction_cpp ${SOURCES})

# Link libraries
target_link_libraries(industrial_reconstruction_cpp PRIVATE 
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Compiler flags
target_compile_definitions(industrial_reconstruction_cpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Set output directory
set_target_properties(industrial_reconstruction_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../industrial_reconstruction
)

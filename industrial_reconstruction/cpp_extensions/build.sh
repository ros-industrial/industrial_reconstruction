#!/bin/bash

# Build script for industrial reconstruction C++ extensions

set -e

echo "Building Industrial Reconstruction C++ Extensions..."

# Check if required packages are installed
echo "Checking dependencies..."

# Check for pybind11
python3 -c "import pybind11" 2>/dev/null || {
    echo "Error: pybind11 not found. Install with: pip3 install pybind11"
    exit 1
}

# Check for OpenCV
python3 -c "import cv2" 2>/dev/null || {
    echo "Error: OpenCV not found. Install with: pip3 install opencv-python"
    exit 1
}

# Check for Eigen3
if [ ! -d "/usr/include/eigen3" ] && [ ! -d "/usr/local/include/eigen3" ]; then
    echo "Warning: Eigen3 not found in standard locations. Make sure it's installed."
fi

# Create build directory
mkdir -p build
cd build

# Build using CMake
echo "Building with CMake..."
cmake ..
make -j$(nproc)

# Also build with setuptools as backup
echo "Building with setuptools..."
cd ..
python3 setup.py build_ext --inplace

echo "Build completed successfully!"
echo "The C++ extensions are now available in the industrial_reconstruction package."

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os
import sys

# Get the include directory for pybind11
pybind11_include = pybind11.get_include()

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "industrial_reconstruction_cpp",
        [
            "src/image_buffer.cpp",
            "src/pose_calculator.cpp", 
            "src/memory_manager.cpp",
            "src/bindings.cpp",
        ],
        include_dirs=[
            pybind11_include,
            "include",
            "/usr/include/opencv4",
            "/usr/include/eigen3",
        ],
        libraries=["opencv_core", "opencv_imgproc", "opencv_imgcodecs"],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="industrial_reconstruction_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)

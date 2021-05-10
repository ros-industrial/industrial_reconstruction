#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args  = generate_distutils_setup(
    packages=['open3d_interface','open3d_interface.yak','open3d_interface.general_recorder','open3d_interface.reconstruction','open3d_interface.utility'],
    package_dir={'': 'src'}
)

setup(**setup_args )

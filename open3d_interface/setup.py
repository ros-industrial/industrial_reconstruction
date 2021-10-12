from setuptools import setup
from setuptools import find_packages
import os
from glob import glob

package_name = 'open3d_interface'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    (os.path.join('share', package_name, 'open3d_interface', 'reconstruction'), glob('src/open3d_interface/reconstruction/*.py')),
    (os.path.join('share', package_name, 'open3d_interface', 'utility'), glob('src/open3d_interface/utility/*.py'))],
    install_requires=[
        'setuptools',
        'launch',
        'launch_ros'
    ],
    zip_safe=True,
    author='Tyler Marr',
    author_email='TODO',
    maintainer='Tyler Marr',
    maintainer_email='TODO',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Package containing first attempt at making python package.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'open3d_reconstruction = open3d_interface.open3d_reconstruction:main',
            'open3d_general_recorder = open3d_interface.open3d_general_recorder:main',
            'open3d_realsense_recorder = open3d_interface.open3d_realsense_recorder:main',
            'open3d_yak = open3d_interface.open3d_yak:main'
        ],
    },
)

from setuptools import setup
from setuptools import find_packages
import os
from glob import glob

package_name = 'open3d_interface'

setup(
    name=package_name,
    version='0.0.1',
    packages=['open3d_interface', 'src/open3d_interface/utility'],
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.*')),
    (os.path.join('share', package_name, 'config'), glob('config/*')),
    (os.path.join('share', package_name, 'open3d_interface', 'utility'), glob('src/open3d_interface/utility/*.py'))],
    install_requires=[
        'setuptools',
        'launch',
        'launch_ros'
    ],
    zip_safe=True,
    author='Tyler Marr',
    author_email='tyler.marr@swri.org',
    maintainer='Tyler Marr',
    maintainer_email='tyler.marr@swri.org',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='ROS wrapper for TSDF function in Open3d',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'open3d_archive_player = open3d_interface.open3d_archive_player:main',
            'open3d_reconstruction = open3d_interface.open3d_reconstruction:main'
        ],
    },
)

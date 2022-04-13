# open3d_interface

## Example Usage

Launch reconstruction node
```
ros2 launch open3d_interface yak.launch depth_image_topic:=/camera/depth_image/raw color_image_topic:=/camera/color_image/raw camera_info_topic:=/camera/camera_info
```

Call service to start reconstruction
```
ros2 service call /start_reconstruction open3d_interface_msgs/srv/StartYakReconstruction "tracking_frame: 'camera'
relative_frame: 'world'
translation_distance: 0.0
rotational_distance: 0.0
live: true
tsdf_params:
  voxel_length: 0.02
  sdf_trunc: 0.04
  min_box_values: {x: 0.05, y: 0.25, z: 0.1}
  max_box_values: {x: 7.0, y: 3.0, z: 1.2}
rgbd_params: {depth_scale: 1000.0, depth_trunc: 0.75, convert_rgb_to_intensity: false}"
```

Call service to stop reconstruction
```
ros2 service call /stop_reconstruction open3d_interface_msgs/srv/StopYakReconstruction "archive_directory: '/home/ros-industrial/open3d_archive/archive'
results_directory: '/home/ros-industrial/open3d_archive/results'"
```

All of the raw color and depth images along with all of the associated tf data and camera info will be stored in the `archive_directory`. A mesh file, `results_mesh.ply` will be saved in the results_directory.

Note: if you leave the archive directory field blank then it will not save any of the images or tf data which can save time when calling stop reconstruction. 
___
### Parameters Explained
**tracking_frame:** Camera tf frame where the image and depth image are relative to

**relative_frame:** Base tf frame that the TSDF mesh will be generated relative to

**translation_distance:** Distance the tracking_frame must travel relative to the relative_frame before another image is allowed to be added to the volume (typically works best with 0 and hasn't yet been thoroughly tested with greater values)

**rotational_distance:** Rotational distance the tracking_frame must rotate relative to the relative_frame before another image is allowed to be added to the volume (typically works best with 0 and hasn't yet been thoroughly tested with greater values)

**live:** Whether or not the TSDF integration is performed while receiving images. Setting this to `true` allows for live visualization of the mesh being generated, but may cause the system to miss some frames

**tsdf_params:** Parameters related to the TSDF mesh generated

 - **voxel_length:** Controls the size of triangles created (Note: this is not in meters, in units associated with the camera)
  
 - **sdf_trunc:** Controls how connected to make adjacent points. A higher value will connected more disparate data, a low value requires that points be very close to be connected
  
 - **min_box_values:** The minimum values of the TSDF volume bounding box used to crop the TSDF mesh that is generated (Leaving all box values as 0 will lead to no bounding box being used)
  
 - **max_box_values:** The maximum values of the TSDF volume bounding box used to crop the TSDF mesh that is generated
  
**rgbd_params:** Parameters relating specifically to the camera being used

 - **depth_scale:** Scale of the data. Set to 1000 to get the output data in meters if the camera's default distance scale is in milimeters
  
 - **depth_trunc:** The distance at which data beyond is clipped and not used. Example: 1.0 would lead to anything greater than 1.0 meters away from the camera would be ignored
  
 - **convert_rgb_to_intensity:** Allows for using float type intensity if using grayscale as well. Usually set this to `false` unless you have a specific reason to do otherwise
  
For more info see the [Open3D documentation on RGBD Integration](http://www.open3d.org/docs/0.12.0/tutorial/pipelines/rgbd_integration.html).

---
## Running on archived data

Launch script to bringup main reconstruction node and archive player
```
ros2 launch open3d_interface sim_from_archive.launch.xml image_directory:=</path/to/archived/data> rviz:=true
```

Call service to start publishing data by parsing through archived camera info, poses, color images, and depth images
```
ros2 service call /start_publishing std_srvs/srv/Trigger
```

Call service to start reconstruction
```
ros2 service call /start_reconstruction open3d_interface_msgs/srv/StartYakReconstruction "tracking_frame: 'sim_camera'
relative_frame: 'world'
translation_distance: 0.0
rotational_distance: 0.0
live: true
tsdf_params:
  voxel_length: 0.02
  sdf_trunc: 0.04
  min_box_values: {x: 0.05, y: 0.25, z: 0.1}
  max_box_values: {x: 7.0, y: 3.0, z: 1.2}
rgbd_params: {depth_scale: 1000.0, depth_trunc: 0.75, convert_rgb_to_intensity: false}"
```

Call service to stop reconstruction
```
ros2 service call /stop_reconstruction open3d_interface_msgs/srv/StopYakReconstruction "archive_directory: '/home/ros-industrial/open3d_archive/new_archive'
results_directory: '/home/ros-industrial/open3d_archive/results'"
```

Call service to stop publishing data
```
ros2 service call /stop_publishing std_srvs/srv/Trigger
```

Optionally restart back at the beginning of the data set (The node will automatically cycle back to the first data point and continue forever without intervention)
```
ros2 service call /restart_publishing std_srvs/srv/Trigger
```
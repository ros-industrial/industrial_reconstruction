# Industrial Reconstruction

A utility to create meshes using an RGB-D camera feed given known positions and parameters of the camera. This package makes heavy use of the python library [Open3D](https://github.com/isl-org/Open3D).

<p align="center">
<img src="https://user-images.githubusercontent.com/41449746/171745032-c915a431-0dbd-462d-9020-768ad63ff0f0.GIF" />
</p>

## Setup

1. Install Open3D
    ```
    pip3 install open3d
    ```

2. Clone this repository into your workspace
3. Build the package as a normal ROS2 package

## Example Usage

Launch reconstruction node
```
ros2 launch industrial_reconstruction reconstruction.launch.xml depth_image_topic:=/camera/depth_image/raw color_image_topic:=/camera/color_image/raw camera_info_topic:=/camera/camera_info
```

Call service to start reconstruction
```
ros2 service call /start_reconstruction industrial_reconstruction_msgs/srv/StartReconstruction "tracking_frame: 'camera'
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
ros2 service call /stop_reconstruction industrial_reconstruction_msgs/srv/StopReconstruction "archive_directory: '/home/ros-industrial/industrial_reconstruction_archive/archive'
mesh_filepath: '/home/ros-industrial/industrial_reconstruction_archive/results_mesh.ply'
normal_filters: [{ normal_direction: {x: 0.0, y: 0.0, z: 1.0}, angle: 90}]
min_num_faces: 1000"
```

___
## Parameters Explained

### StartReconstruction

- **tracking_frame:** Camera tf frame where the image and depth image are relative to

- **relative_frame:** Base tf frame that the TSDF mesh will be generated relative to

- **translation_distance:** Distance the tracking_frame must travel relative to the relative_frame before another image is allowed to be added to the volume (typically works best with 0 and hasn't yet been thoroughly tested with greater values)

- **rotational_distance:** Rotational distance the tracking_frame must rotate relative to the relative_frame before another image is allowed to be added to the volume (typically works best with 0 and hasn't yet been thoroughly tested with greater values)

- **live:** Whether or not the TSDF integration is performed while receiving images. Setting this to `true` allows for live visualization of the mesh being generated, but may cause the system to miss some frames

- **tsdf_params:** Parameters related to the TSDF mesh generated

    - **sdf_trunc:** The distance (in meters) beyond which the signed distance field is clipped (i.e., points in the volume more than this distance away from the implicit surface have a signed distance of 1.0).
      This value should be at least 2 times *smaller* than the smallest distance between distinct features in the reconstruction volume.

    - **voxel_length:** Length of the voxels (in meters) used in the TSDF volume.
      This value should be at least 2-4 times smaller than `sdf_trunc`.
      Smaller values lead to greater resolution in the volume and resulting mesh but at the cost of greater memory consumption and processing compute time.

     - **min_box_values:** The 3D position (meters) of the minimum corner of the bounding box used to crop the TSDF mesh that is generated. Note: if the volume of the crop box is close to 0, the crop box will not be used.

     - **max_box_values:** The 3D position (meters) of the maximum corner of the bounding box used to crop the TSDF mesh that is generated. Note: if the volume of the crop box is close to 0, the crop box will not be used.

- **rgbd_params:** Parameters relating specifically to the camera being used

    - **depth_scale:** Scale factor for the points in the depth image.
      Many cameras report depth values in units of millimeters, so set this value to 1000 in this case to convert the depth values to meters.

     - **depth_trunc:** The distance (in meters) beyond which depth data is clipped.
       Note: this distance is relative to the camera frame, not the reconstruction frame, and is intended to quickly filter depth points that are far from the camera. 

     - **convert_rgb_to_intensity:** Allows for using float type intensity if using grayscale as well. Usually set this to `false` unless you have a specific reason to do otherwise

For more info see the [Open3D documentation on RGBD Integration](http://www.open3d.org/docs/0.12.0/tutorial/pipelines/rgbd_integration.html).

#### Example

Consider reconstructing a pipe with 100 mm diameter and 500 mm length.
The minimum distance between features on the pipe is roughly the diameter of the pipe, so `sdf_trunc < 100 mm / 2`, so a value of 0.025 m would be appropriate.
The value of `voxel_length` should be something less than 2-4 times smaller than `sdf_trunc`, so a value of 0.005 m would be appropriate.
If the pipe is being scanned at a distance of 1 m away, then a reasonable value of `depth_trunc` would be 1.5 m.
If the pipe axis aligns with the z-axis of `relative_frame` and the center of one of its end faces sits near the origin of `relative_frame`, reasonable values for the crop box would be:
    - `min_box_value = [-0.075, -0.075, -0.05]` (0.5 times the cylinder radius in the radial directions and a little below the bottom face in the axial direction)
    - `max_box_value = [0.075, 0.075, 0.550]` (0.5 times the cylinder radius in the radial directions and 1.1 times the length of the pipe in the axial direction)

### StopReconstruction

- **archive_directory:** (optional) Where to store all the captured color images, depth imaegs, poses, and camera info, this can take a while to write all of these files and it will skip this step if you leave this blank

- **mesh_filepath:** Where to save the resulting ply file, include file extension in name.

- **normal_filters:** (optional) A vector of filters applied based off triangle normals, an empty vector will apply no normal filtering

    - **normal_direction:** The desired direction of face normals to keep

     - **angle:** The angle, in degrees, relative to the normal direction to keep. Ex: normal direction of [0,0,1] with an angle of 30 degrees will only keep faces within 30 degrees of straight vertical

- **min_num_faces:** (optional) The minimum number of connected faces required to be included in the exported mesh, setting this to 0 or leaving blank will result in no filtering applied

---
## Running on archived data

Launch script to bringup main reconstruction node and archive player
```
ros2 launch industrial_reconstruction sim_from_archive.launch.xml image_directory:=</path/to/archived/data> rviz:=true
```

Call service to start publishing data by parsing through archived camera info, poses, color images, and depth images
```
ros2 service call /start_publishing std_srvs/srv/Trigger
```

Call service to start reconstruction
```
ros2 service call /start_reconstruction industrial_reconstruction_msgs/srv/StartReconstruction "tracking_frame: 'sim_camera'
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
ros2 service call /stop_reconstruction industrial_reconstruction_msgs/srv/StopReconstruction "archive_directory: '/home/ros-industrial/industrial_reconstruction_archive/archive'
mesh_filepath: '/home/ros-industrial/industrial_reconstruction_archive/results_mesh.ply'
normal_filters: [{ normal_direction: {x: 0.0, y: 0.0, z: 1.0}, angle: 90}]
min_num_faces: 1000"
```

Call service to stop publishing data
```
ros2 service call /stop_publishing std_srvs/srv/Trigger
```

Optionally restart back at the beginning of the data set (The node will automatically cycle back to the first data point and continue forever without intervention)
```
ros2 service call /restart_publishing std_srvs/srv/Trigger
```

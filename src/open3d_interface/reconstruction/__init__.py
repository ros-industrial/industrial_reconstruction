import sys
import rospkg
import rospy
import time
import datetime
import open3d as o3d
import numpy as np

from os import makedirs, listdir
from os.path import join, isfile
from open3d_interface.reconstruction.initialize_config import initialize_config
from open3d_interface.utility.file import check_folder_structure, make_clean_folder, write_pose, read_pose
from open3d_interface.srv import RunReconstructionSystem, RunReconstructionSystemResponse
from open3d_interface.srv import RunTSDFReconstruction, RunTSDFReconstructionResponse

def register_known_rgbd_pair(source_rgbd_image, target_rgbd_image, odo_init, intrinsic, req):
    option = o3d.pipelines.odometry.OdometryOption()
    option.max_depth_diff = req.max_depth_diff
    option.min_depth = req.min_depth
    option.max_depth = req.max_depth

    [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                              source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                              o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                              option)

    return [success, trans, info]

def createPoseGraph(pose_array, rgbd_array, intrinsic, req):
  pose_graph = o3d.pipelines.registration.PoseGraph()
  pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(pose_array[0]))
  for s in range(len(rgbd_array)):
      for t in range(s + 1, len(rgbd_array)):
          # odometry
          odo_init = np.linalg.inv(pose_array[s]).dot(pose_array[t])

          # TODO: Need to understand what info is because we may be able to just use the pose information directly
          if t == s + 1:
              print("RGBD odometry matching between frame : %d and %d" % (s, t))
              [success, trans, info] = register_known_rgbd_pair(rgbd_array[s], rgbd_array[t], odo_init, intrinsic, req)
              trans_odometry = pose_array[t]
              pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry))
              pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=False))

          # TODO: Should also leverage the pose information to determine if a keyframe
          # keyframe loop closure
#          if s % req.n_keyframes_per_n_frame == 0  and t % req.n_keyframes_per_n_frame == 0:
          if s % 5 == 0  and t % 5 == 0:
              print("RGBD keyframes matching between frame : %d and %d" % (s, t))
              [success, trans, info] = register_known_rgbd_pair(rgbd_array[s], rgbd_array[t], odo_init, intrinsic, req)
              if success:
                  pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, trans, info, uncertain=True))

  return pose_graph

def runTSDFReconstructionCallback(req):

  # Get number images
  pose_dir = join(req.path_dataset, "pose")
  num_frames = len([name for name in listdir(pose_dir) if isfile(join(pose_dir, name))])
  print("Number of frames captured: {:d}".format(num_frames))

  # Get intrinsic data
  if req.path_intrinsic:
      intrinsic = o3d.io.read_pinhole_camera_intrinsic(req.path_intrinsic)
  else:
      intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

  # Load data
  color_array = []
  depth_array = []
  rgbd_array = []
  rgbd_color_array = []
  pose_array = []
  for i in range(num_frames):
      color = o3d.io.read_image("{:s}/color/{:06d}.jpg".format(req.path_dataset,i))
      depth = o3d.io.read_image("{:s}/depth/{:06d}.png".format(req.path_dataset,i))
      pose = read_pose("{:s}/pose/{:06d}.pose".format(req.path_dataset,i))
      color_array.append(color)
      depth_array.append(depth)
      pose_array.append(pose)

      # Used for building pose graph
      rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, req.rgbd_params.depth_scale, req.rgbd_params.depth_trunc, req.rgbd_params.convert_rgb_to_intensity)
      rgbd_array.append(rgbd)

      # Used for scene integration
      rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, req.rgbd_params.depth_scale, req.rgbd_params.depth_trunc, False)
      rgbd_color_array.append(rgbd)

  # Create integration volume
  volume = o3d.pipelines.integration.ScalableTSDFVolume(
      voxel_length=req.tsdf_params.voxel_length,
      sdf_trunc=req.tsdf_params.sdf_trunc,
      color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

  if req.global_optimization:
    # Create initial pose graph
    pose_graph = createPoseGraph(pose_array, rgbd_array, intrinsic, req)

    # Optimize pose graph
    method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=req.max_correspondence_distance,
        edge_prune_threshold=req.edge_prune_threshold,
        preference_loop_closure=req.preference_loop_closure,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)

    # Integrate scene
    for s in range(len(pose_graph.nodes)):
      volume.integrate(rgbd_color_array[s], intrinsic, np.linalg.inv(pose_graph.nodes[s].pose))
  else:
    # Integrate scene
    for s in range(num_frames):
      print("Integrating frames: {:d} of {:d}".format(s, num_frames))
      volume.integrate(rgbd_color_array[s], intrinsic, np.linalg.inv(pose_array[s]))

  # Extract Mesh and save to file
  mesh = volume.extract_triangle_mesh()
  mesh.compute_vertex_normals()
  if req.debug_mode:
      o3d.visualization.draw_geometries([mesh])

  mesh_name = join(req.path_dataset, "integrated.ply")
  o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)

  return RunTSDFReconstructionResponse(True, mesh_name)

def runReconstructionSystemCallback(req):
  rospy.loginfo(rospy.get_caller_id() + "Reconstructing Surface")

  if not req.make and not req.register and not req.refine and not req.integrate:
     return ReconstructSurfaceResponse(False)

  config = {}
  if not req.use_default_settings:
    config["name"] = req.name
    config["max_depth"] = req.max_depth
    config["voxel_size"] = req.voxel_size
    config["max_depth_diff"] = req.max_depth_diff
    config["preference_loop_closure_odometry"] = req.preference_loop_closure_odometry
    config["preference_loop_closure_registration"] = req.preference_loop_closure_registration
    config["tsdf_cubic_size"] = req.tsdf_cubic_size
    config["global_registration"] = req.global_registration
    config["python_multi_threading"] = req.python_multi_threading

  config["path_dataset"] = req.path_dataset
  config["path_intrinsic"] = req.path_intrinsic
  config["has_tracking"] = req.has_tracking
  config["debug_mode"] = req.debug_mode

  # check folder structure
  initialize_config(config)
  check_folder_structure(config["path_dataset"], req.has_tracking)
  assert config is not None

  print("====================================")
  print("Configuration")
  print("====================================")
  for key, val in config.items():
      print("%40s : %s" % (key, str(val)))

  times = [0, 0, 0, 0]
  if req.make:
      start_time = time.time()
      import open3d_interface.reconstruction.make_fragments as make_fragments
      make_fragments.run(config)
      times[0] = time.time() - start_time
  if req.register:
      start_time = time.time()
      import open3d_interface.reconstruction.register_fragments as register_fragments
      register_fragments.run(config)
      times[1] = time.time() - start_time
  if req.refine:
      start_time = time.time()
      import open3d_interface.reconstruction.refine_registration as refine_registration
      refine_registration.run(config)
      times[2] = time.time() - start_time
  if req.integrate:
      start_time = time.time()
      import open3d_interface.reconstruction.integrate_scene as integrate_scene
      integrate_scene.run(config)
      times[3] = time.time() - start_time

  print("====================================")
  print("Elapsed time (in h:m:s)")
  print("====================================")
  print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
  print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
  print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
  print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
  print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
  sys.stdout.flush()

  return RunReconstructionSystemResponse(True, req.path_dataset + "/scene/integrated.ply")

def main():
  rospy.init_node('open3d_reconstruction', anonymous=True)

  reconstruction_system_server = rospy.Service('run_reconstruction_system', RunReconstructionSystem, runReconstructionSystemCallback)
  tsdf_reconstruction_server = rospy.Service('run_tsdf_reconstruction', RunTSDFReconstruction, runTSDFReconstructionCallback)

  rospy.spin()

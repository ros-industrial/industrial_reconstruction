# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
import rospkg
import rospy
import time
import datetime
import json

from open3d_interface.reconstruction.initialize_config import initialize_config
from open3d_interface.utility.file import check_folder_structure
from open3d_interface.srv import ReconstructSurface
from open3d_interface.srv import ReconstructSurfaceResponse

def save_camera_info_intrinsic_as_json(filename, camera_info_msg):
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    camera_info_msg.width,
                'height':
                    camera_info_msg.height,
                'intrinsic_matrix': [
                    camera_info_msg.K[0], 0, 0, 0, camera_info_msg.K[4], 0, camera_info_msg.K[2],
                    camera_info_msg.K[5], 1
                ]
            },
            outfile,
            indent=4)

def reconstructSystemCallback(req):
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
  config["debug_mode"] = req.debug_mode

  # check folder structure
  initialize_config(config)
  check_folder_structure(config["path_dataset"])
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

  return ReconstructSurfaceResponse(True, req.path_dataset + "/scene/integrated.ply")

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import json
import open3d as o3d

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


def getIntrinsicsFromMsg(camera_info_msg):
  return o3d.camera.PinholeCameraIntrinsic(camera_info_msg.width, camera_info_msg.height, camera_info_msg.K[0], camera_info_msg.K[4], camera_info_msg.K[2], camera_info_msg.K[5])


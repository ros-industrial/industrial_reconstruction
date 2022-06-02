# Copyright 2022 Southwest Research Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os import listdir, makedirs
from os.path import exists, isfile, join, splitext, dirname, basename
import shutil
import re
from warnings import warn
import json
import open3d as o3d


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


def add_if_exists(path_dataset, folder_names):
    for folder_name in folder_names:
        if exists(join(path_dataset, folder_name)):
            path = join(path_dataset, folder_name)
    return path


def get_rgbd_folders(path_dataset):
    path_color = add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
    path_depth = join(path_dataset, "depth/")
    path_pose = join(path_dataset, "pose/")
    return path_color, path_depth, path_pose


def get_rgbd_file_lists(path_dataset, has_tracking):
    path_color, path_depth, path_pose = get_rgbd_folders(path_dataset)
    color_files = get_file_list(path_color, ".jpg") + \
            get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    if has_tracking:
      pose_files = get_file_list(path_pose, ".pose")
    else:
      pose_files = []
    return color_files, depth_files, pose_files


def make_clean_folder(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)
    else:
        shutil.rmtree(path_folder)
        makedirs(path_folder)

def make_folder_keep_contents(path_folder):
    if not exists(path_folder):
        makedirs(path_folder)

def check_folder_structure(path_dataset, has_tracking):
    if isfile(path_dataset) and path_dataset.endswith(".bag"):
        return
    path_color, path_depth, path_pose = get_rgbd_folders(path_dataset)
    assert exists(path_depth), "Path %s is not exist!" % path_depth
    assert exists(path_color), "Path %s is not exist!" % path_color
    if has_tracking:
      assert exists(path_pose), "Path %s is not exist!" % path_color


def write_pose(filename, pose):
    with open(filename, 'w') as f:
        f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
            pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3]))
        f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
            pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3]))
        f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
            pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3]))
        f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
            pose[3, 0], pose[3, 1], pose[3, 2], pose[3, 3]))


def read_pose(filename):
    import numpy as np

    f = open(filename, "r")
    content = f.readlines()
    return np.array(list(map(float, (''.join(content[0:4])).strip().split()))).reshape((4, 4))


def write_poses_to_log(filename, poses):
    with open(filename, 'w') as f:
        for i, pose in enumerate(poses):
            f.write('{} {} {}\n'.format(i, i, i + 1))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3]))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3]))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3]))
            f.write('{0:.8f} {1:.8f} {2:.8f} {3:.8f}\n'.format(
                pose[3, 0], pose[3, 1], pose[3, 2], pose[3, 3]))


def read_poses_from_log(traj_log):
    import numpy as np

    trans_arr = []
    with open(traj_log) as f:
        content = f.readlines()

        # Load .log file.
        for i in range(0, len(content), 5):
            # format %d (src) %d (tgt) %f (fitness)
            data = list(map(float, content[i].strip().split(' ')))
            ids = (int(data[0]), int(data[1]))
            fitness = data[2]

            # format %f x 16
            T_gt = np.array(
                list(map(float, (''.join(
                    content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

            trans_arr.append(T_gt)

    return trans_arr


def extract_rgbd_frames(rgbd_video_file):
    """
    Extract color and aligned depth frames and intrinsic calibration from an
    RGBD video file (currently only RealSense bag files supported). Folder
    structure is:
        <directory of rgbd_video_file/<rgbd_video_file name without extension>/
            {depth/00000.jpg,color/00000.png,intrinsic.json}
    """
    frames_folder = join(dirname(rgbd_video_file),
                         basename(splitext(rgbd_video_file)[0]))
    path_intrinsic = join(frames_folder, "intrinsic.json")
    if isfile(path_intrinsic):
        warn(f"Skipping frame extraction for {rgbd_video_file} since files are"
             " present.")
    else:
        rgbd_video = o3d.t.io.RGBDVideoReader.create(rgbd_video_file)
        rgbd_video.save_frames(frames_folder)
    with open(path_intrinsic) as intr_file:
        intr = json.load(intr_file)
    depth_scale = intr["depth_scale"]
    return frames_folder, path_intrinsic, depth_scale

def save_intrinsic_as_json(filename, intrinsics):
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.intrinsic_matrix[0,0], 0, 0, 0, intrinsics.intrinsic_matrix[1,1], 0, intrinsics.intrinsic_matrix[0,2],
                    intrinsics.intrinsic_matrix[1,2], 1
                ]
            },
            outfile,
            indent=4)

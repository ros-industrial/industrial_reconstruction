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

import sys
import os
import time
import subprocess
from pathlib import Path
from collections import deque
from os.path import exists, join, isfile
from shlex import quote as sh_quote

import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros import TransformListener
import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
import json
import gc
import cv2

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from message_filters import ApproximateTimeSynchronizer, Subscriber

from src.industrial_reconstruction.utility.file import (
    make_clean_folder,
    write_pose,
    read_pose,
    save_intrinsic_as_json,
    make_folder_keep_contents,
)
from industrial_reconstruction_msgs.srv import StartReconstruction, StopReconstruction
from std_srvs.srv import SetBool
from src.industrial_reconstruction.utility.ros import (
    getIntrinsicsFromMsg,
    meshToRos,
    transformStampedToVectors,
)

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError


def filterNormals(mesh, direction, angle):
    mesh.compute_vertex_normals()
    tri_normals = np.asarray(mesh.triangle_normals)
    dot_prods = tri_normals @ direction
    mask = (dot_prods.ravel() < np.cos(angle))  # ensure 1-D mask
    mesh.remove_triangles_by_mask(mask)
    return mesh


class IndustrialReconstruction(Node):
    def __init__(self):
        super().__init__("industrial_reconstruction")

        self.bridge = CvBridge()

        self.buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.buffer, node=self)

        self.tsdf_volume = None
        self.intrinsics = None
        self.crop_box = None
        self.crop_mesh = False
        self.crop_box_msg = Marker()
        self.tracking_frame = ""
        self.relative_frame = ""
        self.translation_distance = 0.05  # 5cm
        self.rotational_distance = 0.01  # Quaternion Distance

        ####################################################################
        # See Open3d function create_from_color_and_depth for more details #
        ####################################################################
        # The ratio to scale depth values. The depth values will first be scaled and then truncated.
        self.depth_scale = 1000.0
        # Depth values larger than depth_trunc gets truncated to 0. The depth values will first be scaled and then truncated.
        self.depth_trunc = 3.0
        # Whether to convert RGB image to intensity image.
        self.convert_rgb_to_intensity = False

        # Used to store the data used for constructing TSDF
        self.sensor_data = deque()
        self.color_images = []
        self.depth_images = []
        self.rgb_poses = []
        self.prev_pose_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self.prev_pose_tran = np.array([0.0, 0.0, 0.0])

        self.tsdf_integration_data = deque()
        self.integration_done = True
        self.live_integration = False
        self.mesh_pub = None
        self.tsdf_volume_pub = None

        self.record = False
        self.frame_count = 0
        self.processed_frame_count = 0
        self.reconstructed_frame_count = 0

        # ---------------- Existing parameters ----------------
        self.declare_parameter("depth_image_topic")
        self.declare_parameter("color_image_topic")
        self.declare_parameter("camera_info_topic")
        self.declare_parameter("cache_count", 10)
        self.declare_parameter("slop", 0.01)
        self.declare_parameter("camera_name", "camera")  # Camera namespace for controlling streams

        # Manual external editor hook
        self.declare_parameter("enable_external_edit", False)
        self.declare_parameter("editor_cmd", "meshlab")  # or "meshlab.mesa"
        self.declare_parameter("external_edit_timeout_sec", 0)  # 0 = no timeout

        # ---------------- Automatic research pipeline parameters ----------------
        self.declare_parameter("auto_filter_strategy", "off")  # off|script|implicit_filter
        self.declare_parameter(
            "auto_filter_cmd", ""
        )  # e.g. "python /path/pcdnf_infer.py --in {in} --out {out}"
        self.declare_parameter("auto_filter_timeout_sec", 600)  # 0 = no timeout

        self.declare_parameter("auto_normals", "open3d")  # open3d|script|off
        self.declare_parameter(
            "hae_cmd", ""
        )  # e.g. "python /path/hae_normals.py --in {in} --out {out}"

        self.declare_parameter("auto_meshing_method", "poisson")  # poisson|bpa|none
        self.declare_parameter("poisson_depth", 9)
        self.declare_parameter("bpa_radii", [0.005, 0.01, 0.02])  # meters list

        self.declare_parameter("normal_radius", 0.02)
        self.declare_parameter("normal_max_nn", 50)
        self.declare_parameter("orient_k", 50)

        # ---- Depth edge rejection params ----
        self.declare_parameter("depth_edge_filter", True)          # enable/disable
        self.declare_parameter("depth_edge_threshold", 0.008)      # meters-per-pixel gradient
        self.declare_parameter("depth_edge_dilate", 1)             # pixels to dilate (0 = off)
        self.declare_parameter("depth_gradient_ksize", 3)          # 3 or 5 (Sobel kernel)

        # Read manual edit parameters
        self.enable_external_edit = bool(self.get_parameter("enable_external_edit").value)
        self.editor_cmd = str(self.get_parameter("editor_cmd").value)
        self.external_edit_timeout_sec = int(self.get_parameter("external_edit_timeout_sec").value)

        # Read auto pipeline parameters
        self.auto_filter_strategy = str(self.get_parameter("auto_filter_strategy").value).lower()
        self.auto_filter_cmd = str(self.get_parameter("auto_filter_cmd").value)
        self.auto_filter_timeout_sec = int(self.get_parameter("auto_filter_timeout_sec").value)

        self.auto_normals = str(self.get_parameter("auto_normals").value).lower()
        self.hae_cmd = str(self.get_parameter("hae_cmd").value)

        self.auto_meshing_method = str(self.get_parameter("auto_meshing_method").value).lower()
        self.poisson_depth = int(self.get_parameter("poisson_depth").value)
        self.bpa_radii = [float(r) for r in self.get_parameter("bpa_radii").value]

        self.normal_radius = float(self.get_parameter("normal_radius").value)
        self.normal_max_nn = int(self.get_parameter("normal_max_nn").value)
        self.orient_k = int(self.get_parameter("orient_k").value)

        # Read core topics/sync params
        try:
            self.depth_image_topic = str(self.get_parameter("depth_image_topic").value)
        except Exception:
            self.get_logger().error("Failed to load depth_image_topic parameter")
        try:
            self.color_image_topic = str(self.get_parameter("color_image_topic").value)
        except Exception:
            self.get_logger().error("Failed to load color_image_topic parameter")
        try:
            self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        except Exception:
            self.get_logger().error("Failed to load camera_info_topic parameter")
        try:
            self.cache_count = int(self.get_parameter("cache_count").value)
        except Exception:
            self.get_logger().info("Failed to load cache_count parameter")
        try:
            self.slop = float(self.get_parameter("slop").value)
        except Exception:
            self.get_logger().info("Failed to load slop parameter")
        allow_headerless = False

        self.get_logger().info("depth_image_topic - " + self.depth_image_topic)
        self.get_logger().info("color_image_topic - " + self.color_image_topic)
        self.get_logger().info("camera_info_topic - " + self.camera_info_topic)

        self.depth_sub = Subscriber(self, Image, self.depth_image_topic)
        self.color_sub = Subscriber(self, Image, self.color_image_topic)
        self.tss = ApproximateTimeSynchronizer(
            [self.depth_sub, self.color_sub], self.cache_count, self.slop, allow_headerless
        )
        self.tss.registerCallback(self.cameraCallback)

        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.cameraInfoCallback, 10)

        self.mesh_pub = self.create_publisher(Marker, "industrial_reconstruction_mesh", 10)

        self.start_server = self.create_service(
            StartReconstruction, "start_reconstruction", self.startReconstructionCallback
        )
        self.stop_server = self.create_service(
            StopReconstruction, "stop_reconstruction", self.stopReconstructionCallback
        )

        self.tsdf_volume_pub = self.create_publisher(Marker, "tsdf_volume", 10)

        # Read depth edge params
        self.depth_edge_filter = bool(self.get_parameter("depth_edge_filter").value)
        self.depth_edge_threshold = float(self.get_parameter("depth_edge_threshold").value)
        self.depth_edge_dilate = int(self.get_parameter("depth_edge_dilate").value)
        self.depth_gradient_ksize = int(self.get_parameter("depth_gradient_ksize").value)

        # Camera stream control using toggle_color and toggle_depth services
        self.camera_name = str(self.get_parameter("camera_name").value)
        self.toggle_color_client = self.create_client(
            SetBool, f"/{self.camera_name}/toggle_color"
        )
        self.toggle_depth_client = self.create_client(
            SetBool, f"/{self.camera_name}/toggle_depth"
        )
        
        # Wait for camera services to be available and disable streams initially
        self.get_logger().info(f"Waiting for camera services /{self.camera_name}/toggle_color and /{self.camera_name}/toggle_depth...")
        color_ready = self.toggle_color_client.wait_for_service(timeout_sec=10.0)
        depth_ready = self.toggle_depth_client.wait_for_service(timeout_sec=10.0)
        
        if color_ready and depth_ready:
            self.get_logger().info("Camera services available, disabling streams initially")
            self._control_camera_streams(False)
        else:
            self.get_logger().warn(f"Camera services not available. Stream control will be disabled.")

    # ===================== Manual editor launcher =====================
    def _launch_editor_and_wait(self, pointcloud_path: Path, target_mesh_path: Path, timeout_sec: int) -> bool:
        """
        Launch the editor (MeshLab) with a sanitized Qt environment and wait
        until the user saves a mesh at target_mesh_path.
        """
        env = os.environ.copy()

        # If DISPLAY is missing, GUI won’t work. Bail early.
        if os.name != "nt" and "DISPLAY" not in env and env.get("WAYLAND_DISPLAY") is None:
            self.get_logger().warn("[External Edit] No X/Wayland display found; GUI editor will fail.")
            return False

        # Strip problematic variables commonly set by OpenCV/ROS stacks
        for var in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "LD_PRELOAD"):
            env.pop(var, None)

        # Point Qt to the system plugin dir explicitly (Ubuntu/Qt5 path shown; adjust if Qt6)
        qt5_plugins = "/usr/lib/x86_64-linux-gnu/qt5/plugins"
        if os.path.isdir(qt5_plugins):
            env["QT_PLUGIN_PATH"] = qt5_plugins

        # Force XCB (often works even under Wayland via XWayland)
        env.setdefault("QT_QPA_PLATFORM", "xcb")

        self.get_logger().info(f"[External Edit] Launching '{self.editor_cmd}' with {pointcloud_path}")
        try:
            proc = subprocess.Popen(
                [self.editor_cmd, str(pointcloud_path)],
                env=env,
                start_new_session=True,
                close_fds=True,
            )
        except Exception as e:
            self.get_logger().error(f"[External Edit] Failed to launch editor: {e}")
            return False

        self.get_logger().info(
            f"[External Edit] Please save your final mesh to: {target_mesh_path}\n"
            "              (PLY/OBJ/STL supported; PLY recommended)"
        )

        start_time = time.time()
        start_mtime = target_mesh_path.stat().st_mtime if target_mesh_path.exists() else 0.0

        while True:
            # Mesh saved/updated?
            if target_mesh_path.exists() and target_mesh_path.stat().st_mtime > start_mtime:
                time.sleep(0.5)  # grace period to finish writes
                return True

            # Timeout?
            if timeout_sec > 0 and (time.time() - start_time) > timeout_sec:
                self.get_logger().warn(
                    f"[External Edit] Timeout ({timeout_sec}s) waiting for mesh save. Falling back."
                )
                return False

            # Editor closed without saving
            if proc.poll() is not None and not target_mesh_path.exists():
                self.get_logger().warn("[External Edit] Editor closed and no mesh detected. Falling back.")
                return False

            time.sleep(0.5)

    # ===================== Utility: archive sensor data =====================
    def archiveData(self, path_output):
        path_depth = join(path_output, "depth")
        path_color = join(path_output, "color")
        path_pose = join(path_output, "pose")

        make_folder_keep_contents(path_output)
        make_clean_folder(path_depth)
        make_clean_folder(path_color)
        make_clean_folder(path_pose)

        for s in range(len(self.color_images)):
            # Save your OpenCV2 image as a jpeg
            o3d.io.write_image("%s/%06d.png" % (path_depth, s), self.depth_images[s])
            o3d.io.write_image("%s/%06d.jpg" % (path_color, s), self.color_images[s])
            write_pose("%s/%06d.pose" % (path_pose, s), self.rgb_poses[s])
            save_intrinsic_as_json(join(path_output, "camera_intrinsic.json"), self.intrinsics)

    # ===================== Depth edge suppression =====================
    def _suppress_depth_edges(self, depth_u: np.ndarray) -> np.ndarray:
        """
        Zero-out depth at strong depth discontinuities to avoid TSDF 'skins'.
        Accepts 16UC1 (mm) or float32 (m). Returns same dtype as input.
        Uses Sobel gradient on depth-in-meters; masks pixels with |∇z| > threshold.
        """
        if not self.depth_edge_filter:
            return depth_u

        # Convert to meters float for gradient
        if depth_u.dtype == np.uint16:
            depth_m = depth_u.astype(np.float32) / float(self.depth_scale)
            convert_back = "u16"
        else:
            depth_m = depth_u.astype(np.float32)
            convert_back = "f32"

        # Compute gradient magnitude |∇z| in m/pixel
        k = 3 if self.depth_gradient_ksize not in (3, 5) else self.depth_gradient_ksize
        gx = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=k)
        gy = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=k)
        gradmag = cv2.magnitude(gx, gy)

        # Mask where gradient is large
        thr = max(0.0, float(self.depth_edge_threshold))
        mask = gradmag > thr

        # Optional: expand the mask a bit to remove grazing-edge pixels
        if self.depth_edge_dilate > 0:
            r = int(self.depth_edge_dilate)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Zero out those pixels (0 = invalid depth for Open3D)
        if convert_back == "u16":
            out = depth_u.copy()
            out[mask] = 0
            return out
        else:
            out = depth_m.copy()
            out[mask] = 0.0
            return out

    # ===================== Camera stream control =====================
    def _control_camera_streams(self, enable: bool):
        """Enable or disable camera streams via toggle_color and toggle_depth services"""
        if not self.toggle_color_client.service_is_ready() or not self.toggle_depth_client.service_is_ready():
            self.get_logger().warn(f"Camera services not ready, cannot {'enable' if enable else 'disable'} streams")
            return False
        
        request = SetBool.Request()
        request.data = enable
        
        success = True
        
        # Toggle color stream
        try:
            future_color = self.toggle_color_client.call_async(request)
            timeout = 5.0
            start_time = time.time()
            while not future_color.done() and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
            
            if future_color.done():
                response = future_color.result()
                if response.success:
                    self.get_logger().info(f"Color stream {'enabled' if enable else 'disabled'} successfully")
                else:
                    self.get_logger().error(f"Failed to {'enable' if enable else 'disable'} color stream: {response.message}")
                    success = False
            else:
                self.get_logger().warn(f"Timeout waiting for color stream toggle response")
                success = False
        except Exception as e:
            self.get_logger().error(f"Exception calling color stream toggle: {e}")
            success = False
        
        # Toggle depth stream
        try:
            future_depth = self.toggle_depth_client.call_async(request)
            timeout = 5.0
            start_time = time.time()
            while not future_depth.done() and (time.time() - start_time) < timeout:
                rclpy.spin_once(self, timeout_sec=0.1)
            
            if future_depth.done():
                response = future_depth.result()
                if response.success:
                    self.get_logger().info(f"Depth stream {'enabled' if enable else 'disabled'} successfully")
                else:
                    self.get_logger().error(f"Failed to {'enable' if enable else 'disable'} depth stream: {response.message}")
                    success = False
            else:
                self.get_logger().warn(f"Timeout waiting for depth stream toggle response")
                success = False
        except Exception as e:
            self.get_logger().error(f"Exception calling depth stream toggle: {e}")
            success = False
        
        return success

    # ===================== Start reconstruction =====================
    def startReconstructionCallback(self, req, res):
        try:
            self.get_logger().info(" Start Reconstruction")

            self.color_images.clear()
            self.depth_images.clear()
            self.rgb_poses.clear()
            self.sensor_data.clear()
            self.tsdf_integration_data.clear()
            self.prev_pose_rot = np.array([1.0, 0.0, 0.0, 0.0])
            self.prev_pose_tran = np.array([0.0, 0.0, 0.0])

            if (
                req.tsdf_params.min_box_values.x == req.tsdf_params.max_box_values.x
                and req.tsdf_params.min_box_values.y == req.tsdf_params.max_box_values.y
                and req.tsdf_params.min_box_values.z == req.tsdf_params.max_box_values.z
            ):
                self.crop_mesh = False
            else:
                self.crop_mesh = True
                min_bound = np.asarray(
                    [req.tsdf_params.min_box_values.x, req.tsdf_params.min_box_values.y, req.tsdf_params.min_box_values.z]
                )
                max_bound = np.asarray(
                    [req.tsdf_params.max_box_values.x, req.tsdf_params.max_box_values.y, req.tsdf_params.max_box_values.z]
                )
                self.crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

                self.crop_box_msg.type = self.crop_box_msg.CUBE
                self.crop_box_msg.action = self.crop_box_msg.ADD
                self.crop_box_msg.id = 1
                self.crop_box_msg.scale.x = max_bound[0] - min_bound[0]
                self.crop_box_msg.scale.y = max_bound[1] - min_bound[1]
                self.crop_box_msg.scale.z = max_bound[2] - min_bound[2]
                self.crop_box_msg.pose.position.x = (min_bound[0] + max_bound[0]) / 2.0
                self.crop_box_msg.pose.position.y = (min_bound[1] + max_bound[1]) / 2.0
                self.crop_box_msg.pose.position.z = (min_bound[2] + max_bound[2]) / 2.0
                self.crop_box_msg.pose.orientation.w = 1.0
                self.crop_box_msg.pose.orientation.x = 0.0
                self.crop_box_msg.pose.orientation.y = 0.0
                self.crop_box_msg.pose.orientation.z = 0.0
                self.crop_box_msg.color.r = 1.0
                self.crop_box_msg.color.g = 0.0
                self.crop_box_msg.color.b = 0.0
                self.crop_box_msg.color.a = 0.25
                self.crop_box_msg.header.frame_id = req.relative_frame

                self.tsdf_volume_pub.publish(self.crop_box_msg)

            self.frame_count = 0
            self.processed_frame_count = 0
            self.reconstructed_frame_count = 0

            self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=req.tsdf_params.voxel_length,
                sdf_trunc=req.tsdf_params.sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )

            self.depth_scale = req.rgbd_params.depth_scale
            self.depth_trunc = req.rgbd_params.depth_trunc
            self.convert_rgb_to_intensity = req.rgbd_params.convert_rgb_to_intensity
            self.tracking_frame = req.tracking_frame
            self.relative_frame = req.relative_frame
            self.translation_distance = req.translation_distance
            self.rotational_distance = req.rotational_distance

            self.live_integration = req.live
            self.record = True

            # Enable camera streams when starting reconstruction
            self._control_camera_streams(True)

            res.success = True
            return res
        except Exception as e:
            self.get_logger().error(f"startReconstruction failed: {e}")
            res.success = False
            res.message = str(e)
            return res

    # ===================== Helpers: Auto pipeline =====================
    def _run_cmd(self, cmd: str, timeout: int = 0) -> bool:
        """Run a shell command string; returns True on exit code 0."""
        self.get_logger().info(f"[Auto] Running: {cmd}")
        try:
            if timeout and timeout > 0:
                subprocess.check_call(cmd, shell=True, timeout=timeout)
            else:
                subprocess.check_call(cmd, shell=True)
            return True
        except subprocess.TimeoutExpired:
            self.get_logger().error(f"[Auto] Command timed out after {timeout}s")
            return False
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"[Auto] Command failed (exit {e.returncode})")
            return False
        except Exception as e:
            self.get_logger().error(f"[Auto] Command exception: {e}")
            return False

    def _auto_filter_pointcloud(self, strategy, in_pcd: Path, out_pcd: Path, out_mesh: Path):
        """
        Returns (mesh_path, pcd_path). Only one will be non-None.
        - strategy 'script': expects self.auto_filter_cmd with {in},{out} producing a filtered PCD (PLY).
        - strategy 'implicit_filter': expects self.auto_filter_cmd to output a mesh directly to {out}.
        """
        strategy = (strategy or "off").lower()
        if strategy == "off":
            return (None, None)

        if not self.auto_filter_cmd:
            self.get_logger().warn("[Auto] auto_filter_cmd is empty; skipping auto filter.")
            return (None, None)

        # Safe mapping for format_map (avoids using 'in'/'out' as Python keywords)
        map_script_pcd = {"in": sh_quote(str(in_pcd)), "out": sh_quote(str(out_pcd))}
        map_script_mesh = {"in": sh_quote(str(in_pcd)), "out": sh_quote(str(out_mesh))}

        if strategy == "script":
            cmd = self.auto_filter_cmd.format_map(map_script_pcd)
            ok = self._run_cmd(cmd, timeout=self.auto_filter_timeout_sec)
            if not ok or (not out_pcd.exists() or out_pcd.stat().st_size == 0):
                self.get_logger().error("[Auto] Filter script did not produce a valid point cloud.")
                return (None, None)
            return (None, out_pcd)

        if strategy == "implicit_filter":
            cmd = self.auto_filter_cmd.format_map(map_script_mesh)
            ok = self._run_cmd(cmd, timeout=self.auto_filter_timeout_sec)
            if not ok or (not out_mesh.exists() or out_mesh.stat().st_size == 0):
                self.get_logger().error("[Auto] Implicit filter did not produce a valid mesh.")
                return (None, None)
            return (out_mesh, None)

        self.get_logger().warn(f"[Auto] Unknown auto_filter_strategy '{strategy}'.")
        return (None, None)

    def _auto_orient_normals(self, in_pcd: Path, out_pcd: Path):
        """
        If self.auto_normals == 'script', run self.hae_cmd to output oriented normals PLY.
        If 'open3d', estimate + orient in-process and write out.
        Returns path to oriented PLY, or None on failure.
        """
        method = (self.auto_normals or "open3d").lower()

        if method == "off":
            return in_pcd

        if method == "script":
            if not self.hae_cmd:
                self.get_logger().error("[Auto] auto_normals=script but hae_cmd is empty.")
                return None
            map_normals = {"in": sh_quote(str(in_pcd)), "out": sh_quote(str(out_pcd))}
            cmd = self.hae_cmd.format_map(map_normals)
            ok = self._run_cmd(cmd, timeout=self.auto_filter_timeout_sec)
            if not ok or (not out_pcd.exists() or out_pcd.stat().st_size == 0):
                self.get_logger().error("[Auto] Normal-orientation script failed.")
                return None
            return out_pcd

        # Default: Open3D
        try:
            pcd = o3d.io.read_point_cloud(str(in_pcd))
            if pcd.is_empty():
                self.get_logger().error("[Auto] Open3D normals: empty input cloud.")
                return None
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.normal_radius, max_nn=self.normal_max_nn
                )
            )
            # Orient consistently (works reasonably well without camera pose)
            pcd.orient_normals_consistent_tangent_plane(self.orient_k)
            o3d.io.write_point_cloud(str(out_pcd), pcd, write_ascii=False, compressed=False)
            return out_pcd
        except Exception as e:
            self.get_logger().error(f"[Auto] Open3D normals failed: {e}")
            return None

    def _mesh_from_point_cloud(self, pcd_path: Path, method: str):
        try:
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            if pcd.is_empty():
                self.get_logger().error("[Auto] Meshing: point cloud is empty.")
                return None

            method = (method or "poisson").lower()
            if method == "none":
                self.get_logger().info("[Auto] Meshing method is 'none'; skipping.")
                return None

            if method == "bpa":
                radii = o3d.utility.DoubleVector(self.bpa_radii)
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
            else:
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=self.poisson_depth
                )
                # Optional: remove spurious far vertices by cropping to the pcd bounds
                bbox = pcd.get_axis_aligned_bounding_box()
                mesh = mesh.crop(bbox)

            mesh.compute_vertex_normals()
            return mesh
        except Exception as e:
            self.get_logger().error(f"[Auto] Meshing failed: {e}")
            return None

    def _postprocess_mesh(self, mesh: o3d.geometry.TriangleMesh, req) -> o3d.geometry.TriangleMesh:
        if self.crop_mesh:
            mesh = mesh.crop(self.crop_box)

        for norm_filt in req.normal_filters:
            dir = np.array(
                [norm_filt.normal_direction.x, norm_filt.normal_direction.y, norm_filt.normal_direction.z]
            ).reshape(3, 1)
            mesh = filterNormals(mesh, dir, np.radians(norm_filt.angle))

        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < req.min_num_faces
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        return mesh

    # ===================== Stop reconstruction (mesh generation) =====================
    def stopReconstructionCallback(self, req, res):
        try:
            self.get_logger().info("Stop Reconstruction")
            self.record = False

            # Disable camera streams when stopping reconstruction
            self._control_camera_streams(False)

            while not self.integration_done:
                time.sleep(1.0)

            self.get_logger().info("Generating mesh")
            if self.tsdf_volume is None:
                res.success = False
                res.message = "Start reconstruction hasn't been called yet"
                return res

            # If not live, integrate any queued frames first
            if not self.live_integration:
                while len(self.tsdf_integration_data) > 0:
                    data = self.tsdf_integration_data.popleft()
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        data[1], data[0], self.depth_scale, self.depth_trunc, self.convert_rgb_to_intensity
                    )
                    self.tsdf_volume.integrate(rgbd, self.intrinsics, np.linalg.inv(data[2]))

            # Path setup
            target_mesh = Path(req.mesh_filepath).expanduser().resolve()
            target_mesh.parent.mkdir(parents=True, exist_ok=True)

            # Extract a dense point cloud once (used by manual and auto paths)
            try:
                full_pcd = self.tsdf_volume.extract_point_cloud()
            except Exception as e:
                self.get_logger().error(f"[Auto] Failed to extract point cloud from TSDF: {e}")
                full_pcd = None

            edit_cloud_path = target_mesh.parent / "raw_cloud_for_edit.ply"
            if full_pcd is not None:
                try:
                    o3d.io.write_point_cloud(str(edit_cloud_path), full_pcd, write_ascii=False, compressed=False)
                    self.get_logger().info(f"[Auto] Base point cloud saved: {edit_cloud_path}")
                except Exception as e:
                    self.get_logger().error(f"[Auto] Failed to write base point cloud: {e}")

            mesh_to_publish = None

            # ---- (1) Manual edit path, if enabled ----
            if self.enable_external_edit and edit_cloud_path.exists():
                self.get_logger().info("[External Edit] Enabled: launching editor.")
                ok = self._launch_editor_and_wait(
                    edit_cloud_path, target_mesh, timeout_sec=self.external_edit_timeout_sec
                )
                if ok:
                    try:
                        user_mesh = o3d.io.read_triangle_mesh(str(target_mesh))
                        if not user_mesh.is_empty():
                            user_mesh.compute_vertex_normals()
                            # Treat user mesh as the final authority (no post-process)
                            mesh_to_publish = user_mesh
                            self.get_logger().info(f"[External Edit] Using user-provided mesh: {target_mesh}")
                    except Exception as e:
                        self.get_logger().warn(
                            f"[External Edit] Failed to load user mesh ({e}); will try auto path or fallback."
                        )

            # ---- (2) Automatic research-based path ----
            if mesh_to_publish is None and self.auto_filter_strategy != "off" and edit_cloud_path.exists():
                self.get_logger().info(f"[Auto] Running auto_filter_strategy='{self.auto_filter_strategy}'")
                filtered_cloud_path = target_mesh.parent / "filtered_cloud.ply"
                auto_mesh_path = target_mesh.parent / "auto_filtered_mesh.ply"

                # 2a) Filter (may output mesh OR a filtered cloud)
                produced_mesh, produced_pcd = self._auto_filter_pointcloud(
                    self.auto_filter_strategy, edit_cloud_path, filtered_cloud_path, auto_mesh_path
                )

                # 2b) If we got a mesh directly, post-process and use it
                if produced_mesh is not None and produced_mesh.exists():
                    try:
                        mm = o3d.io.read_triangle_mesh(str(produced_mesh))
                        if not mm.is_empty():
                            mm.compute_vertex_normals()
                            mm = self._postprocess_mesh(mm, req)
                            mesh_to_publish = mm
                            self.get_logger().info(f"[Auto] Using implicit-filter mesh: {produced_mesh}")
                    except Exception as e:
                        self.get_logger().warn(f"[Auto] Failed to load implicit-filter mesh ({e}).")

                # 2c) Otherwise, we should have a filtered point cloud → orient normals → mesh
                if mesh_to_publish is None:
                    pcd_for_meshing = produced_pcd if (produced_pcd and produced_pcd.exists()) else edit_cloud_path
                    oriented_path = target_mesh.parent / "oriented_cloud.ply"

                    # Oriented normals
                    oriented = self._auto_orient_normals(pcd_for_meshing, oriented_path)
                    if oriented is None:
                        self.get_logger().warn("[Auto] Orientation failed; will try meshing as-is.")
                        oriented = pcd_for_meshing

                    # Meshing from point cloud
                    if self.auto_meshing_method != "none":
                        mm = self._mesh_from_point_cloud(oriented, self.auto_meshing_method)
                        if mm is not None:
                            mm = self._postprocess_mesh(mm, req)
                            mesh_to_publish = mm
                            self.get_logger().info(f"[Auto] Meshing complete via {self.auto_meshing_method}.")

            # ---- (3) Original TSDF→mesh fallback ----
            if mesh_to_publish is None:
                mesh = self.tsdf_volume.extract_triangle_mesh()
                mesh.compute_vertex_normals()
                mesh_to_publish = self._postprocess_mesh(mesh, req)
                self.get_logger().info("[Auto] Fallback to original TSDF mesh.")

            # Save whichever mesh we’re using to req.mesh_filepath
            o3d.io.write_triangle_mesh(
                str(target_mesh),
                mesh_to_publish,
                write_ascii=False,
                compressed=True,
                write_vertex_normals=True,
            )
            mesh_msg = meshToRos(mesh_to_publish)
            mesh_msg.header.stamp = self.get_clock().now().to_msg()
            mesh_msg.header.frame_id = self.relative_frame
            self.mesh_pub.publish(mesh_msg)
            self.get_logger().info("Mesh Saved to " + str(target_mesh))

            # Archive if requested
            if req.archive_directory != "":
                # Create a per-scan subfolder: MM-DD-YYYY-HH-MM-SS (local time)
                ts = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
                archive_root = Path(req.archive_directory).expanduser().resolve()
                scan_dir = (archive_root / ts)
                scan_dir.mkdir(parents=True, exist_ok=True)

                self.get_logger().info(f"Archiving data to {scan_dir}")

                # Save frame data (depth/color/pose + camera_intrinsic.json) under the scan folder
                self.archiveData(str(scan_dir))

                # Save a raw integrated mesh (direct from TSDF, before post-processing)
                try:
                    raw_mesh = self.tsdf_volume.extract_triangle_mesh()
                    raw_mesh.compute_vertex_normals()
                    o3d.io.write_triangle_mesh(
                        str(scan_dir / "integrated_raw_tsdf.ply"),
                        raw_mesh,
                        write_ascii=False,
                        compressed=True,
                        write_vertex_normals=True,
                    )
                except Exception as e:
                    self.get_logger().warn(f"Failed to archive raw integrated mesh: {e}")

                # Also save the final mesh we just published into the archive folder
                try:
                    o3d.io.write_triangle_mesh(
                        str(scan_dir / "final_mesh.ply"),
                        mesh_to_publish,
                        write_ascii=False,
                        compressed=True,
                        write_vertex_normals=True,
                    )
                except Exception as e:
                    self.get_logger().warn(f"Failed to archive final mesh: {e}")

                # Store some useful metadata about this scan
                try:
                    meta = {
                        "timestamp": ts,
                        "target_mesh": str(target_mesh),
                        "relative_frame": self.relative_frame,
                        "tracking_frame": self.tracking_frame,
                        "depth_scale": self.depth_scale,
                        "depth_trunc": self.depth_trunc,
                        "convert_rgb_to_intensity": self.convert_rgb_to_intensity,
                        "translation_distance": self.translation_distance,
                        "rotational_distance": self.rotational_distance,
                        "live_integration": self.live_integration,
                        "frames_total": len(self.color_images),
                        "frames_processed": self.processed_frame_count,
                        "auto_filter_strategy": self.auto_filter_strategy,
                        "auto_meshing_method": self.auto_meshing_method,
                        "poisson_depth": self.poisson_depth,
                        "bpa_radii": self.bpa_radii,
                    }
                    with open(scan_dir / "metadata.json", "w") as f:
                        json.dump(meta, f, indent=2)
                except Exception as e:
                    self.get_logger().warn(f"Failed to write metadata.json: {e}")

                # Cleanup big per-scan caches now that archiving is done
                try:
                    del full_pcd
                except NameError:
                    pass
                try:
                    del mesh_to_publish
                except NameError:
                    pass

                self.color_images.clear()
                self.depth_images.clear()
                self.rgb_poses.clear()
                self.sensor_data.clear()
                self.tsdf_integration_data.clear()

                self.tsdf_volume = None
                self.crop_box = None
                self.crop_mesh = False
                gc.collect()

                res.success = True
                res.message = f"Mesh Saved to {target_mesh}"
                return res
            else:
                # No archive; return success now
                res.success = True
                res.message = f"Mesh Saved to {target_mesh}"
                return res

        except Exception as e:
            self.get_logger().error(f"stopReconstruction failed: {e}")
            res.success = False
            res.message = str(e)
            return res

    # ===================== Frame callbacks =====================
    def cameraCallback(self, depth_image_msg, rgb_image_msg):
        if self.record:
            try:
                # Convert ROS Image messages to OpenCV2
                depth_u16 = self.bridge.imgmsg_to_cv2(depth_image_msg, "16UC1")
                color_img = self.bridge.imgmsg_to_cv2(rgb_image_msg, rgb_image_msg.encoding)

                if depth_u16.size == 0:
                    return

                # --- Depth denoise with bilateral on float32 meters ---
                depth_f32 = depth_u16.astype(np.float32) / float(self.depth_scale)

                # Preserve invalids (zeros)
                invalid_mask = (depth_u16 == 0)

                # Bilateral filter (units: meters for sigmaColor, pixels for sigmaSpace)
                # Good defaults: d=5, sigmaColor=0.02m (2 cm), sigmaSpace=7 px
                depth_f32 = cv2.bilateralFilter(depth_f32, d=5, sigmaColor=0.02, sigmaSpace=7)

                # Restore invalids
                depth_f32[invalid_mask] = 0.0

                # Edge suppression on float32 meters
                depth_f32 = self._suppress_depth_edges(depth_f32)

                # Convert back to 16-bit depth units for Open3D (0 stays 0)
                depth_u16 = np.clip(
                    np.round(depth_f32 * float(self.depth_scale)),
                    0, np.iinfo(np.uint16).max
                ).astype(np.uint16)

                o3d_depth = o3d.geometry.Image(depth_u16)
                o3d_color = o3d.geometry.Image(color_img)

            except CvBridgeError:
                self.get_logger().error("Error converting ros msg to cv img")
                return
            else:
                self.sensor_data.append(
                    [o3d_depth, o3d_color, rgb_image_msg.header.stamp]
                )
                if self.frame_count > 30:
                    data = self.sensor_data.popleft()
                    try:
                        gm_tf_stamped = self.buffer.lookup_transform(
                            self.relative_frame, self.tracking_frame, data[2]
                        )
                    except Exception as e:
                        self.get_logger().error("Failed to get transform: " + str(e))
                        return
                    rgb_t, rgb_r = transformStampedToVectors(gm_tf_stamped)
                    rgb_r_quat = Quaternion(rgb_r)

                    tran_dist = np.linalg.norm(rgb_t - self.prev_pose_tran)
                    rot_dist = Quaternion.absolute_distance(Quaternion(self.prev_pose_rot), rgb_r_quat)

                    # Min jump to accept data
                    if (tran_dist >= self.translation_distance) or (rot_dist >= self.rotational_distance):
                        self.prev_pose_tran = rgb_t
                        self.prev_pose_rot = rgb_r
                        rgb_pose = rgb_r_quat.transformation_matrix
                        rgb_pose[0, 3] = rgb_t[0]
                        rgb_pose[1, 3] = rgb_t[1]
                        rgb_pose[2, 3] = rgb_t[2]

                        self.depth_images.append(data[0])
                        self.color_images.append(data[1])
                        self.rgb_poses.append(rgb_pose)
                        if self.live_integration and self.tsdf_volume is not None:
                            self.integration_done = False
                            try:
                                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                    data[1], data[0], self.depth_scale, self.depth_trunc, self.convert_rgb_to_intensity
                                )
                                self.tsdf_volume.integrate(rgbd, self.intrinsics, np.linalg.inv(rgb_pose))
                                self.integration_done = True
                                self.processed_frame_count += 1
                                if self.processed_frame_count % 50 == 0:
                                    mesh = self.tsdf_volume.extract_triangle_mesh()
                                    if self.crop_mesh:
                                        cropped_mesh = mesh.crop(self.crop_box)
                                    else:
                                        cropped_mesh = mesh
                                    mesh_msg = meshToRos(cropped_mesh)
                                    mesh_msg.header.stamp = self.get_clock().now().to_msg()
                                    mesh_msg.header.frame_id = self.relative_frame
                                    self.mesh_pub.publish(mesh_msg)
                            except Exception as e:
                                self.get_logger().error("Error processing images into tsdf: " + str(e))
                                self.integration_done = True
                                return
                        else:
                            self.tsdf_integration_data.append([data[0], data[1], rgb_pose])
                            self.processed_frame_count += 1

                self.frame_count += 1

    def cameraInfoCallback(self, camera_info):
        self.intrinsics = getIntrinsicsFromMsg(camera_info)


def main(args=None):
    rclpy.init(args=args)
    industrial_reconstruction = IndustrialReconstruction()
    rclpy.spin(industrial_reconstruction)
    industrial_reconstruction.destroy_node()
    rclpy.shutdown()

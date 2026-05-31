#!/usr/bin/env python3
"""
Phase 1 Environment Mapper for TurtleBot 4 Phase 2 bridge.

Run this node while doing Phase 1 SLAM + RViz + teleoperation.
It collects rich environment data that Phase 2 can consume:

1. Red cube sighting positions — camera-based HSV red detection during teleop.
   If the cube is glimpsed, its global coordinates are saved.
   Phase 2 can navigate directly to the cube instead of blindly searching.

2. Trajectory breadcrumbs — the robot's driven path (safe corridor hint).

3. Final SLAM map — subscribes to the live OccupancyGrid from slam_toolbox
   and saves it on shutdown as .pgm + .yaml for Phase 2 to load.

4. LiDAR free/blocked cell memory — coarse occupancy overlay.

5. RViz real-time visualisation — publishes:
   - phase1_traj          (nav_msgs/Path)
   - phase1_cube_markers  (visualization_msgs/MarkerArray)
   - phase1_map           (nav_msgs/OccupancyGrid, relayed)

6. Bridge output on shutdown: phase1_env_data.json
   Phase 2 can load this file to accelerate SEARCHING and RETURNING.

Usage:
  Terminal 1: ros2 launch turtlebot4_navigation slam.launch.py namespace:=/T13
  Terminal 2: ros2 launch turtlebot4_viz view_robot.launch.py namespace:=/T13
  Terminal 3: teleop
  Terminal 4: ros2 run tb4_sensor_reader phase1_env_mapper --ros-args -p namespace:=/T13
"""

import json
import math
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
# Default settings
# ============================================================
NAMESPACE = "/T13"
NODE_NAME = "phase1_env_mapper"

OUTPUT_BASENAME = "phase1_env_data"

# ----- Trajectory -----
BREADCRUMB_SPACING_M = 0.15    # record a waypoint every N metres
MAX_TRAJ_POINTS = 2000

# ----- LiDAR free/blocked grid -----
GRID_RESOLUTION = 0.05          # 5 cm cells
GRID_DEFAULT_SIZE = 400         # 20 m × 20 m fallback
LIDAR_MAX_USEFUL = 6.0
LIDAR_RAY_STRIDE = 6

# Front direction correction (observed: -pi/2 rad = robot forward)
FRONT_ANGLE_RAD = -math.pi / 2.0

# ----- Red cube detection (same HSV as map_frame_avoidance.py) -----
RED_LOW1 = np.array([0, 150, 110])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 150, 110])
RED_HIGH2 = np.array([180, 255, 255])

MIN_RED_PIXELS = 5800
MIN_BOX_WIDTH_PX = 8
MIN_BOX_HEIGHT_PX = 8
RED_ASPECT_MIN = 0.5
RED_ASPECT_MAX = 2.0
RED_CUBE_SIZE_M = 0.06

OAK_RGB_HFOV_DEG = 66.0
OAK_RGB_VFOV_DEG = 54.0

# Debounce between cube sightings (seconds) so one frame doesn't spam
CUBE_SIGHTING_COOLDOWN = 2.0
CUBE_CONFIRM_FRAMES = 3

# ----- Autosave -----
AUTOSAVE_INTERVAL_SEC = 8.0


def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bresenham(x0, y0, x1, y1):
    """Integer grid line iterator."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Phase1EnvMapper(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        # ---- Parameters ----
        self.declare_parameter("namespace", NAMESPACE)
        self.declare_parameter("output_basename", OUTPUT_BASENAME)
        self.declare_parameter("output_dir", str(Path.cwd()))

        self.ns = str(self.get_parameter("namespace").value).rstrip("/")
        self.output_basename = str(self.get_parameter("output_basename").value)
        self.output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ---- Bridge / CV ----
        self.bridge = CvBridge()

        # ---- State ----
        self.origin_set = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.have_odom = False

        # ---- Trajectory ----
        self.traj_points = []             # list of (x, y, yaw, t) in odom frame
        self.last_breadcrumb = None       # (x, y)

        # ---- LiDAR free/blocked grid ----
        self.grid_origin_x = 0.0
        self.grid_origin_y = 0.0
        self.grid_w = GRID_DEFAULT_SIZE
        self.grid_h = GRID_DEFAULT_SIZE
        self.grid_res = GRID_RESOLUTION
        self.grid_set = False
        self.free_cells = set()
        self.blocked_cells = set()

        # ---- SLAM map ----
        self.latest_map: OccupancyGrid | None = None

        # ---- Red cube sightings ----
        self.cube_sightings = []          # list of dicts
        self.last_sighting_time = 0.0
        self.cube_seen_count = 0

        # ---- Latest sensor data ----
        self.latest_frame = None
        self.latest_scan: LaserScan | None = None

        # ---- Timers ----
        self.start_time = time.time()
        self.last_save_time = 0.0

        # ========================================================
        # Topics: resolve namespace
        # ========================================================
        odom_topic = self._resolve("/odom")
        scan_topic = self._resolve("/scan")
        image_topic = self._resolve("/oakd/rgb/image_raw/compressed")
        map_topic = self._resolve("/map")  # namespaced: /T13/map

        # ========================================================
        # Subscriptions
        # ========================================================
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_cb, 20
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self._scan_cb, 20
        )
        self.image_sub = self.create_subscription(
            CompressedImage, image_topic, self._image_cb, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, map_topic, self._map_cb, 5
        )

        # ========================================================
        # Publishers (RViz real-time visualisation)
        # ========================================================
        self.traj_pub = self.create_publisher(RosPath, "phase1_traj", 10)
        self.cube_markers_pub = self.create_publisher(
            MarkerArray, "phase1_cube_markers", 10
        )
        self.map_pub = self.create_publisher(
            OccupancyGrid, "phase1_map", 10
        )
        self.free_markers_pub = self.create_publisher(
            MarkerArray, "phase1_free_cells", 10
        )
        self.blocked_markers_pub = self.create_publisher(
            MarkerArray, "phase1_blocked_cells", 10
        )

        # ========================================================
        # Periodic timer
        # ========================================================
        self.timer = self.create_timer(0.5, self._periodic)

        self.get_logger().info("Phase 1 Environment Mapper started.")
        self.get_logger().info(
            f"odom={odom_topic}, scan={scan_topic}, image={image_topic}, map={map_topic}"
        )
        self.get_logger().info(f"output_dir={self.output_dir}")

    # ============================================================
    # Helpers
    # ============================================================
    def _resolve(self, topic: str) -> str:
        """将话题名转换到命名空间下。
        例如 ns=/T13, topic=/map -> /T13/map
            ns=/T13, topic=scan -> /T13/scan
        """
        topic = topic.strip()
        if topic.startswith("/"):
            return f"{self.ns}{topic}"
        return f"{self.ns}/{topic}"

    def _world_to_cell(self, x: float, y: float):
        cx = int(math.floor((x - self.grid_origin_x) / self.grid_res))
        cy = int(math.floor((y - self.grid_origin_y) / self.grid_res))
        return cx, cy

    def _cell_in_bounds(self, cx, cy):
        return 0 <= cx < self.grid_w and 0 <= cy < self.grid_h

    def _get_intrinsics(self, img_w, img_h):
        fx = (img_w / 2.0) / math.tan(math.radians(OAK_RGB_HFOV_DEG / 2.0))
        fy = (img_h / 2.0) / math.tan(math.radians(OAK_RGB_VFOV_DEG / 2.0))
        return fx, fy

    # ============================================================
    # ROS Callbacks
    # ============================================================
    def _odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        if not self.origin_set:
            self.origin_x = px
            self.origin_y = py
            self.origin_yaw = yaw
            self.origin_set = True
            self.grid_origin_x = px - (GRID_DEFAULT_SIZE / 2.0) * GRID_RESOLUTION
            self.grid_origin_y = py - (GRID_DEFAULT_SIZE / 2.0) * GRID_RESOLUTION
            self.get_logger().info(
                f"Origin set: ({px:.3f}, {py:.3f}), yaw={math.degrees(yaw):.1f} deg"
            )

        self.robot_x = px
        self.robot_y = py
        self.robot_yaw = yaw
        self.have_odom = True

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg
        if not self.origin_set:
            return
        self._integrate_scan(msg)

    def _image_cb(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        self.latest_frame = frame
        if self.have_odom:
            self._check_for_red_cube(frame)

    def _map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg
        # Relay to RViz
        self.map_pub.publish(msg)

        # Update grid dimensions from live map
        if not self.grid_set and msg.info.width > 0 and msg.info.height > 0:
            self.grid_origin_x = msg.info.origin.position.x
            self.grid_origin_y = msg.info.origin.position.y
            self.grid_w = msg.info.width
            self.grid_h = msg.info.height
            self.grid_res = msg.info.resolution
            self.grid_set = True
            self.get_logger().info(
                f"Grid aligned to SLAM map: {self.grid_w}×{self.grid_h}, "
                f"res={self.grid_res}, origin=({self.grid_origin_x:.3f}, {self.grid_origin_y:.3f})"
            )

    # ============================================================
    # Red cube detection
    # ============================================================
    def _check_for_red_cube(self, frame):
        now = time.time()
        if now - self.last_sighting_time < CUBE_SIGHTING_COOLDOWN:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
        mask2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        red_pixels = int(cv2.countNonZero(mask))
        if red_pixels < MIN_RED_PIXELS:
            self.cube_seen_count = 0
            return

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.cube_seen_count = 0
            return

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_RED_PIXELS:
            self.cube_seen_count = 0
            return

        bx, by, bw, bh = cv2.boundingRect(largest)
        if bw < MIN_BOX_WIDTH_PX or bh < MIN_BOX_HEIGHT_PX:
            self.cube_seen_count = 0
            return

        aspect = bw / float(bh) if bh > 0 else 999
        if not (RED_ASPECT_MIN <= aspect <= RED_ASPECT_MAX):
            self.cube_seen_count = 0
            return

        # Confirmed frame
        self.cube_seen_count += 1
        if self.cube_seen_count < CUBE_CONFIRM_FRAMES:
            return

        # Estimate distance and global position
        img_h, img_w = frame.shape[:2]
        fx, fy = self._get_intrinsics(img_w, img_h)

        z = 0.5 * (
            (RED_CUBE_SIZE_M * fx) / float(bw)
            + (RED_CUBE_SIZE_M * fy) / float(bh)
        )

        box_cx = bx + bw / 2.0
        cube_robot_x = (box_cx - img_w / 2.0) * z / fx
        cube_robot_y = z

        # Convert robot-frame to global (odom frame)
        gx = self.robot_x + cube_robot_x * math.sin(self.robot_yaw) + cube_robot_y * math.cos(self.robot_yaw)
        gy = self.robot_y - cube_robot_x * math.cos(self.robot_yaw) + cube_robot_y * math.sin(self.robot_yaw)

        sighting = {
            "t": round(now - self.start_time, 2),
            "robot_x": round(self.robot_x, 3),
            "robot_y": round(self.robot_y, 3),
            "robot_yaw_deg": round(math.degrees(self.robot_yaw), 1),
            "cube_global_x": round(gx, 3),
            "cube_global_y": round(gy, 3),
            "distance_m": round(z, 3),
            "red_pixels": red_pixels,
            "bbox": (bx, by, bw, bh),
        }
        self.cube_sightings.append(sighting)
        self.last_sighting_time = now
        self.cube_seen_count = 0

        self.get_logger().info(
            f"CUBE SIGHTING #{len(self.cube_sightings)}: "
            f"global=({gx:.3f}, {gy:.3f}), "
            f"dist={z:.3f}m, robot=({self.robot_x:.3f}, {self.robot_y:.3f})"
        )

    # ============================================================
    # LiDAR integration
    # ============================================================
    def _integrate_scan(self, msg: LaserScan):
        rx, ry = self.robot_x, self.robot_y
        ryaw = self.robot_yaw

        start_cell = self._world_to_cell(rx, ry)
        if self._cell_in_bounds(*start_cell):
            self.free_cells.add(start_cell)

        ranges = msg.ranges
        for i in range(0, len(ranges), LIDAR_RAY_STRIDE):
            r = ranges[i]
            if not math.isfinite(r) or r <= msg.range_min:
                continue
            clipped = min(float(r), LIDAR_MAX_USEFUL, float(msg.range_max))

            angle = msg.angle_min + i * msg.angle_increment
            world_heading = ryaw + (angle - FRONT_ANGLE_RAD)

            ex = rx + clipped * math.cos(world_heading)
            ey = ry + clipped * math.sin(world_heading)
            end_cell = self._world_to_cell(ex, ey)
            if not self._cell_in_bounds(*end_cell):
                continue

            # Free along the ray
            for c in bresenham(start_cell[0], start_cell[1], end_cell[0], end_cell[1]):
                if self._cell_in_bounds(*c):
                    self.free_cells.add(c)

            # Blocked at endpoint if real hit
            if r < min(msg.range_max, LIDAR_MAX_USEFUL) - 0.05:
                self.blocked_cells.add(end_cell)

    # ============================================================
    # Trajectory recording
    # ============================================================
    def _maybe_record_breadcrumb(self):
        if not self.have_odom:
            return
        xy = (self.robot_x, self.robot_y)
        if self.last_breadcrumb is not None:
            if dist2d(xy, self.last_breadcrumb) < BREADCRUMB_SPACING_M:
                return
        if len(self.traj_points) >= MAX_TRAJ_POINTS:
            return

        self.traj_points.append({
            "t": round(time.time() - self.start_time, 2),
            "x": round(self.robot_x, 4),
            "y": round(self.robot_y, 4),
            "yaw": round(self.robot_yaw, 4),
            "yaw_deg": round(math.degrees(self.robot_yaw), 1),
        })
        self.last_breadcrumb = xy

    # ============================================================
    # RViz publishers
    # ============================================================
    def _publish_trajectory(self):
        if not self.traj_points:
            return
        msg = RosPath()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"

        for p in self.traj_points:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = p["x"]
            ps.pose.position.y = p["y"]
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self.traj_pub.publish(msg)

    def _publish_cube_markers(self):
        if not self.cube_sightings:
            return
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, s in enumerate(self.cube_sightings):
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = now
            m.ns = "cube_sightings"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = s["cube_global_x"]
            m.pose.position.y = s["cube_global_y"]
            m.pose.position.z = 0.03
            m.pose.orientation.w = 1.0
            m.scale.x = 0.06
            m.scale.y = 0.06
            m.scale.z = 0.06
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9)
            arr.markers.append(m)

            # Text label
            label = Marker()
            label.header.frame_id = "odom"
            label.header.stamp = now
            label.ns = "cube_labels"
            label.id = i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = s["cube_global_x"]
            label.pose.position.y = s["cube_global_y"]
            label.pose.position.z = 0.15
            label.scale.z = 0.12
            label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            label.text = f"Cube #{i+1}\n{s['distance_m']}m"
            arr.markers.append(label)

        self.cube_markers_pub.publish(arr)

    def _publish_cell_markers(self):
        """Publish free and blocked cells as small cubes (throttled: every ~2s)."""
        # Only publish every few cycles to avoid flooding RViz
        if not hasattr(self, "_cell_pub_counter"):
            self._cell_pub_counter = 0
        self._cell_pub_counter += 1
        if self._cell_pub_counter % 4 != 0:
            return

        now = self.get_clock().now().to_msg()
        half = self.grid_res / 2.0

        # Free cells (green)
        free_arr = MarkerArray()
        sample_free = list(self.free_cells)[:2000]  # cap
        for i, (cx, cy) in enumerate(sample_free):
            wx = self.grid_origin_x + (cx + 0.5) * self.grid_res
            wy = self.grid_origin_y + (cy + 0.5) * self.grid_res
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = now
            m.ns = "free_cells"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.0
            m.scale.x = half
            m.scale.y = half
            m.scale.z = half
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.25)
            free_arr.markers.append(m)
        self.free_markers_pub.publish(free_arr)

        # Blocked cells (red)
        blocked_arr = MarkerArray()
        sample_blocked = list(self.blocked_cells)[:2000]
        for i, (cx, cy) in enumerate(sample_blocked):
            wx = self.grid_origin_x + (cx + 0.5) * self.grid_res
            wy = self.grid_origin_y + (cy + 0.5) * self.grid_res
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = now
            m.ns = "blocked_cells"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.0
            m.scale.x = half
            m.scale.y = half
            m.scale.z = half
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.35)
            blocked_arr.markers.append(m)
        self.blocked_markers_pub.publish(blocked_arr)

    # ============================================================
    # Periodic loop
    # ============================================================
    def _periodic(self):
        if not self.origin_set:
            return

        self._maybe_record_breadcrumb()

        # Publish to RViz
        self._publish_trajectory()
        self._publish_cube_markers()
        self._publish_cell_markers()

        # Autosave
        now = time.time()
        if now - self.last_save_time >= AUTOSAVE_INTERVAL_SEC:
            self.last_save_time = now
            self._save_outputs(final=False)

    # ============================================================
    # Bridge data output
    # ============================================================
    def _save_outputs(self, final: bool = False):
        out_json = self.output_dir / f"{self.output_basename}.json"

        # If there's a latest SLAM map, save it as PGM + YAML too
        map_saved = False
        pgm_path = ""
        yaml_path = ""
        if self.latest_map is not None and final:
            pgm_path, yaml_path = self._save_slam_map()

        # Average cube position for Phase 2 to use as target hint
        cube_hint = None
        if self.cube_sightings:
            avg_x = np.mean([s["cube_global_x"] for s in self.cube_sightings])
            avg_y = np.mean([s["cube_global_y"] for s in self.cube_sightings])
            cube_hint = {
                "count": len(self.cube_sightings),
                "avg_global_x": round(float(avg_x), 3),
                "avg_global_y": round(float(avg_y), 3),
                "sightings": self.cube_sightings,
            }

        data = {
            "created_unix": time.time(),
            "final": final,
            "namespace": self.ns,
            "origin": {
                "x": self.origin_x,
                "y": self.origin_y,
                "yaw": self.origin_yaw,
                "yaw_deg": round(math.degrees(self.origin_yaw), 1),
            },
            "trajectory": {
                "count": len(self.traj_points),
                "points": self.traj_points,
            },
            "cube_hint": cube_hint,
            "grid": {
                "resolution": self.grid_res,
                "origin_x": self.grid_origin_x,
                "origin_y": self.grid_origin_y,
                "width": self.grid_w,
                "height": self.grid_h,
                "free_cells": len(self.free_cells),
                "blocked_cells": len(self.blocked_cells),
            },
            "slam_map_saved": {
                "pgm": pgm_path,
                "yaml": yaml_path,
            } if final else None,
            "notes": [
                "trajectory.points = the robot's odometry path during Phase 1 teleop.",
                "cube_hint = if the OAK-D camera saw the red cube during Phase 1, "
                "its estimated global position is here. Phase 2 can go directly to "
                "these coordinates instead of searching the entire corridor.",
                "slam_map_saved = final SLAM map saved as PGM+YAML (only on shutdown).",
                "Phase 2 can load this JSON to accelerate SEARCHING (go to cube_hint) "
                "and RETURNING (use trajectory or map for guidance).",
            ],
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.get_logger().info(
            f"Saved {'FINAL' if final else 'interim'} bridge data: "
            f"traj={len(self.traj_points)} pts, "
            f"cube_sightings={len(self.cube_sightings)}, "
            f"free_cells={len(self.free_cells)}, "
            f"blocked_cells={len(self.blocked_cells)} → {out_json}"
        )

    def _save_slam_map(self):
        """Save the latest SLAM OccupancyGrid as PGM + YAML."""
        if self.latest_map is None:
            return "", ""

        msg = self.latest_map
        w = msg.info.width
        h = msg.info.height

        # Convert OccupancyGrid data to PGM (0-255 grayscale)
        # Standard: 0=free (white), 100=occupied (black), -1=unknown (gray)
        data = np.array(msg.data, dtype=np.int16).reshape((h, w))
        pgm = np.full((h, w), 205, dtype=np.uint8)  # unknown = gray
        pgm[data == 0] = 254    # free = white
        pgm[data == 100] = 0    # occupied = black

        pgm_path = str(self.output_dir / f"{self.output_basename}_map.pgm")
        yaml_path = str(self.output_dir / f"{self.output_basename}_map.yaml")

        # OccupancyGrid row 0 is the map's bottom row, while image row 0 is
        # the top row. map_server expects the image convention, so flip Y.
        cv2.imwrite(pgm_path, cv2.flip(pgm, 0))

        q = msg.info.origin.orientation
        origin_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        meta = {
            "image": f"{self.output_basename}_map.pgm",
            "mode": "trinary",
            "resolution": float(msg.info.resolution),
            "origin": [
                float(msg.info.origin.position.x),
                float(msg.info.origin.position.y),
                float(origin_yaw),
            ],
            "negate": 0,
            "occupied_thresh": 0.65,
            "free_thresh": 0.196,
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

        self.get_logger().info(f"SLAM map saved: {pgm_path}, {yaml_path}")
        return pgm_path, yaml_path


def main(args=None):
    rclpy.init(args=args)
    node = Phase1EnvMapper()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt — saving final Phase 1 data.")
    finally:
        try:
            node._save_outputs(final=True)
        except Exception as e:
            node.get_logger().warn(f"Final save failed: {e}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

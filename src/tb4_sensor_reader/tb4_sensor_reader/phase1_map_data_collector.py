#!/usr/bin/env python3
"""
Phase 1 map-data collector for TurtleBot 4 Phase 2 searching / returning.

Purpose
-------
Run this node while you are doing Phase 1 SLAM + RViz + teleoperation.
It does NOT move the robot. It only records useful navigation memory:

1. Safe odometry breadcrumbs while the robot is manually driven.
2. Free / blocked / visited map cells estimated from LiDAR.
3. A sparse return waypoint path generated from the safe breadcrumbs.
4. Optional live OccupancyGrid from SLAM, or a saved lab_map.yaml + lab_map.pgm.
5. JSON/YAML/PNG outputs that Phase 2 can load.

Typical use
-----------
Terminal 1: launch SLAM / RViz as normal.
Terminal 2: teleop the robot around the C-shaped lab area.
Terminal 3: run this node.

The output file phase1_navigation_memory.json can be loaded by your Phase 2 node.
"""

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header


# ============================================================
# Default settings
# ============================================================
NAMESPACE = "/T13"

DEFAULT_MAP_BASENAME = "lab_map"
DEFAULT_OUTPUT_BASENAME = "phase1_navigation_memory"

# Record one safe breadcrumb every this many metres.
BREADCRUMB_SPACING_M = 0.15

# Keep return path sparse enough for Phase 2 waypoint following.
RETURN_WAYPOINT_SPACING_M = 0.35

# Cell size used for memory overlay if no saved map is available.
FALLBACK_GRID_RESOLUTION = 0.05

# LiDAR filtering.
LIDAR_MAX_USEFUL_RANGE_M = 6.0
LIDAR_RAY_STRIDE = 6

# Safety thresholds for marking a robot pose as a safe breadcrumb.
SAFE_FRONT_MIN_M = 0.45
SAFE_SIDE_MIN_M = 0.25
SAFE_UNKNOWN_OK = True

# Front direction correction from your robot setup.
# You observed that -pi/2 rad corresponds to robot forward.
FRONT_ANGLE_RAD = -math.pi / 2.0

# Output frequency.
AUTOSAVE_INTERVAL_SEC = 5.0


# ============================================================
# Helper functions
# ============================================================
def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_odom(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def dist2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bresenham(x0: int, y0: int, x1: int, y1: int):
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


# ============================================================
# Main node
# ============================================================
class Phase1MapDataCollector(Node):
    def __init__(self):
        super().__init__("phase1_map_data_collector")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("namespace", NAMESPACE)
        self.declare_parameter("map_basename", DEFAULT_MAP_BASENAME)
        self.declare_parameter("output_basename", DEFAULT_OUTPUT_BASENAME)
        self.declare_parameter("map_dir", "")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("subscribe_live_map", True)
        self.declare_parameter("map_topic", "map")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("scan_topic", "scan")
        self.declare_parameter("breadcrumb_spacing_m", BREADCRUMB_SPACING_M)
        self.declare_parameter("return_waypoint_spacing_m", RETURN_WAYPOINT_SPACING_M)
        self.declare_parameter("autosave_interval_sec", AUTOSAVE_INTERVAL_SEC)

        self.ns = str(self.get_parameter("namespace").value).rstrip("/")
        self.map_basename = str(self.get_parameter("map_basename").value)
        self.output_basename = str(self.get_parameter("output_basename").value)
        self.subscribe_live_map = bool(self.get_parameter("subscribe_live_map").value)

        map_dir_param = str(self.get_parameter("map_dir").value).strip()
        out_dir_param = str(self.get_parameter("output_dir").value).strip()

        self.map_dir = Path(map_dir_param).expanduser() if map_dir_param else Path.cwd()
        self.output_dir = Path(out_dir_param).expanduser() if out_dir_param else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.breadcrumb_spacing = float(self.get_parameter("breadcrumb_spacing_m").value)
        self.return_waypoint_spacing = float(self.get_parameter("return_waypoint_spacing_m").value)
        self.autosave_interval = float(self.get_parameter("autosave_interval_sec").value)

        odom_topic = self.resolve_topic(str(self.get_parameter("odom_topic").value))
        scan_topic = self.resolve_topic(str(self.get_parameter("scan_topic").value))
        map_topic = self.resolve_topic(str(self.get_parameter("map_topic").value))

        # -------------------------
        # Map representation
        # -------------------------
        self.map_resolution = FALLBACK_GRID_RESOLUTION
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_width = 0
        self.map_height = 0
        self.occupancy_grid: Optional[np.ndarray] = None
        self.map_source = "fallback_local_grid"

        self.load_saved_map_if_available()

        # Memory cells are stored as integer map cells.
        self.visited_cells = set()
        self.free_cells = set()
        self.blocked_cells = set()
        self.frontier_cells = set()

        # -------------------------
        # Odometry / pose state
        # -------------------------
        self.origin_set = False
        self.origin_raw_x = 0.0
        self.origin_raw_y = 0.0
        self.origin_raw_yaw = 0.0

        self.raw_x = 0.0
        self.raw_y = 0.0
        self.raw_yaw = 0.0
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_yaw = 0.0

        # -------------------------
        # LiDAR state
        # -------------------------
        self.latest_scan: Optional[LaserScan] = None
        self.front_min = float("inf")
        self.left_min = float("inf")
        self.right_min = float("inf")

        # -------------------------
        # Safe trajectory memory
        # -------------------------
        self.safe_path: List[Dict] = []
        self.last_breadcrumb_xy: Optional[Tuple[float, float]] = None
        self.start_time = time.time()
        self.last_save_time = 0.0

        # Optional published path for RViz visualisation.
        self.safe_path_pub = self.create_publisher(RosPath, "phase1_safe_path", 10)
        self.return_path_pub = self.create_publisher(RosPath, "phase1_return_path", 10)

        # -------------------------
        # Subscriptions
        # -------------------------
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 20)
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 20)

        if self.subscribe_live_map:
            self.create_subscription(OccupancyGrid, map_topic, self.map_callback, 5)

        self.timer = self.create_timer(0.5, self.periodic_update)

        self.get_logger().info("Phase 1 map-data collector started.")
        self.get_logger().info(f"odom_topic={odom_topic}, scan_topic={scan_topic}, map_topic={map_topic}")
        self.get_logger().info(f"map_source={self.map_source}")
        self.get_logger().info(f"output_dir={self.output_dir}")

    # ============================================================
    # Topic / map helpers
    # ============================================================
    def resolve_topic(self, topic: str) -> str:
        topic = topic.strip()
        if topic.startswith("/"):
            return topic
        if self.ns:
            return f"{self.ns}/{topic}"
        return topic

    def load_saved_map_if_available(self):
        yaml_file = self.map_dir / f"{self.map_basename}.yaml"

        # Also tolerate filenames such as lab_map(2).yaml if user passes basename.
        if not yaml_file.exists():
            candidates = sorted(self.map_dir.glob(f"{self.map_basename}*.yaml"))
            if candidates:
                yaml_file = candidates[0]

        if not yaml_file.exists():
            self.get_logger().warn(
                f"No saved map YAML found in {self.map_dir}. "
                "Will still collect odom/LiDAR memory."
            )
            return

        try:
            with yaml_file.open("r", encoding="utf-8") as f:
                meta = yaml.safe_load(f)

            image_name = meta.get("image", f"{self.map_basename}.pgm")
            image_path = (yaml_file.parent / image_name).resolve()
            if not image_path.exists():
                # Tolerate uploaded naming, e.g. lab_map(2).pgm.
                pgm_candidates = sorted(yaml_file.parent.glob(f"{self.map_basename}*.pgm"))
                if pgm_candidates:
                    image_path = pgm_candidates[0]

            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                self.get_logger().warn(f"Map image could not be read: {image_path}")
                return
            # map_server PGM row 0 is the top row; internal occupancy arrays use
            # OccupancyGrid ordering where row 0 is the map's bottom row.
            img = cv2.flip(img, 0)

            self.map_resolution = float(meta.get("resolution", FALLBACK_GRID_RESOLUTION))
            origin = meta.get("origin", [0.0, 0.0, 0.0])
            self.map_origin_x = float(origin[0])
            self.map_origin_y = float(origin[1])
            self.map_height, self.map_width = img.shape

            # Convert trinary map image to occupancy-style values:
            # 0 = free, 100 = occupied, -1 = unknown.
            occ = np.full(img.shape, -1, dtype=np.int16)
            occ[img >= 250] = 0
            occ[img <= 10] = 100
            self.occupancy_grid = occ
            self.map_source = f"saved_map:{yaml_file.name}"

            self.get_logger().info(
                f"Loaded saved map: {yaml_file.name}, image={image_path.name}, "
                f"size={self.map_width}x{self.map_height}, res={self.map_resolution}, "
                f"origin=({self.map_origin_x:.3f}, {self.map_origin_y:.3f})"
            )

        except Exception as e:
            self.get_logger().warn(f"Failed to load saved map: {e}")

    def map_callback(self, msg: OccupancyGrid):
        """Use live SLAM occupancy grid when available."""
        w = msg.info.width
        h = msg.info.height
        if w <= 0 or h <= 0:
            return

        arr = np.array(msg.data, dtype=np.int16).reshape((h, w))

        self.occupancy_grid = arr
        self.map_width = int(w)
        self.map_height = int(h)
        self.map_resolution = float(msg.info.resolution)
        self.map_origin_x = float(msg.info.origin.position.x)
        self.map_origin_y = float(msg.info.origin.position.y)
        self.map_source = "live_slam_occupancy_grid"

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int(math.floor((x - self.map_origin_x) / self.map_resolution))
        cy = int(math.floor((y - self.map_origin_y) / self.map_resolution))
        return cx, cy

    def cell_to_world(self, cx: int, cy: int) -> Tuple[float, float]:
        x = self.map_origin_x + (cx + 0.5) * self.map_resolution
        y = self.map_origin_y + (cy + 0.5) * self.map_resolution
        return x, y

    def cell_in_bounds(self, cx: int, cy: int) -> bool:
        if self.map_width <= 0 or self.map_height <= 0:
            return True
        return 0 <= cx < self.map_width and 0 <= cy < self.map_height

    # ============================================================
    # ROS callbacks
    # ============================================================
    def odom_callback(self, msg: Odometry):
        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        yaw = yaw_from_odom(msg)

        if not self.origin_set:
            self.origin_raw_x = px
            self.origin_raw_y = py
            self.origin_raw_yaw = yaw
            self.origin_set = True
            self.get_logger().info(
                f"Collector origin set: raw_x={px:.3f}, raw_y={py:.3f}, yaw={yaw:.3f}"
            )

        self.raw_x = px
        self.raw_y = py
        self.raw_yaw = yaw

        dx = px - self.origin_raw_x
        dy = py - self.origin_raw_y
        c = math.cos(self.origin_raw_yaw)
        s = math.sin(self.origin_raw_yaw)

        self.local_x = c * dx + s * dy
        self.local_y = -s * dx + c * dy
        self.local_yaw = normalize_angle(yaw - self.origin_raw_yaw)

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.update_lidar_sector_mins(msg)

        if self.origin_set:
            self.integrate_scan_into_memory(msg)

    # ============================================================
    # LiDAR processing
    # ============================================================
    def update_lidar_sector_mins(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return

        inc = msg.angle_increment
        front_i = int(round((FRONT_ANGLE_RAD - msg.angle_min) / inc)) % n
        side_90 = int(round(math.radians(90.0) / inc))
        half_30 = int(round(math.radians(30.0) / inc))

        def arc_min(center_index: int) -> float:
            vals = []
            for k in range(-half_30, half_30 + 1):
                i = (center_index + k) % n
                r = msg.ranges[i]
                if msg.range_min < r < msg.range_max and math.isfinite(r):
                    vals.append(float(r))
            return min(vals) if vals else float("inf")

        self.front_min = arc_min(front_i)
        self.left_min = arc_min((front_i + side_90) % n)
        self.right_min = arc_min((front_i - side_90) % n)

    def current_pose_is_safe(self) -> bool:
        front_ok = self.front_min >= SAFE_FRONT_MIN_M or (SAFE_UNKNOWN_OK and math.isinf(self.front_min))
        left_ok = self.left_min >= SAFE_SIDE_MIN_M or (SAFE_UNKNOWN_OK and math.isinf(self.left_min))
        right_ok = self.right_min >= SAFE_SIDE_MIN_M or (SAFE_UNKNOWN_OK and math.isinf(self.right_min))
        return front_ok and left_ok and right_ok

    def integrate_scan_into_memory(self, msg: LaserScan):
        """
        Project LiDAR rays into a lightweight grid memory.

        This does not replace SLAM. It creates practical Phase 2 data:
        - cells observed as free along each ray
        - obstacle cells at valid ray endpoints
        """
        if not self.origin_set:
            return

        robot_world_x = self.raw_x
        robot_world_y = self.raw_y
        robot_yaw = self.raw_yaw

        start_cell = self.world_to_cell(robot_world_x, robot_world_y)
        if self.cell_in_bounds(*start_cell):
            self.visited_cells.add(start_cell)

        ranges = msg.ranges
        for i in range(0, len(ranges), LIDAR_RAY_STRIDE):
            r = ranges[i]
            if not math.isfinite(r):
                continue
            if r <= msg.range_min:
                continue

            clipped_r = min(float(r), LIDAR_MAX_USEFUL_RANGE_M, float(msg.range_max))
            angle = msg.angle_min + i * msg.angle_increment

            # Convert LiDAR angle into world heading.
            # The robot's forward direction in scan frame is FRONT_ANGLE_RAD.
            relative_heading = angle - FRONT_ANGLE_RAD
            world_heading = robot_yaw + relative_heading

            end_x = robot_world_x + clipped_r * math.cos(world_heading)
            end_y = robot_world_y + clipped_r * math.sin(world_heading)
            end_cell = self.world_to_cell(end_x, end_y)

            if not self.cell_in_bounds(*end_cell):
                continue

            # Mark cells along the beam as free.
            ray_cells = list(bresenham(start_cell[0], start_cell[1], end_cell[0], end_cell[1]))
            if len(ray_cells) >= 2:
                for c in ray_cells[:-1]:
                    if self.cell_in_bounds(*c):
                        self.free_cells.add(c)

            # If the measurement is a real hit, mark endpoint blocked.
            real_hit = r < min(msg.range_max, LIDAR_MAX_USEFUL_RANGE_M) - 0.05
            if real_hit:
                self.blocked_cells.add(end_cell)

    # ============================================================
    # Memory recording
    # ============================================================
    def maybe_record_breadcrumb(self):
        if not self.origin_set:
            return

        if not self.current_pose_is_safe():
            return

        xy = (self.local_x, self.local_y)

        if self.last_breadcrumb_xy is not None:
            if dist2d(xy, self.last_breadcrumb_xy) < self.breadcrumb_spacing:
                return

        cell = self.world_to_cell(self.raw_x, self.raw_y)
        if not self.cell_in_bounds(*cell):
            return

        point = {
            "t": round(time.time() - self.start_time, 3),
            "raw_x": round(self.raw_x, 4),
            "raw_y": round(self.raw_y, 4),
            "raw_yaw": round(self.raw_yaw, 4),
            "local_x": round(self.local_x, 4),
            "local_y": round(self.local_y, 4),
            "local_yaw": round(self.local_yaw, 4),
            "cell_x": int(cell[0]),
            "cell_y": int(cell[1]),
            "front_min": round(self.front_min, 3) if math.isfinite(self.front_min) else None,
            "left_min": round(self.left_min, 3) if math.isfinite(self.left_min) else None,
            "right_min": round(self.right_min, 3) if math.isfinite(self.right_min) else None,
        }

        self.safe_path.append(point)
        self.last_breadcrumb_xy = xy
        self.visited_cells.add(cell)

    def make_sparse_return_path(self) -> List[Dict]:
        """
        Return reverse of safe path, sparsified by distance.

        This is the practical C-corridor return path: instead of trying to drive
        straight to (0, 0), Phase 2 can follow these remembered waypoints back.
        """
        if not self.safe_path:
            return []

        reverse_path = list(reversed(self.safe_path))
        sparse = [reverse_path[0]]
        last_xy = (reverse_path[0]["local_x"], reverse_path[0]["local_y"])

        for p in reverse_path[1:]:
            xy = (p["local_x"], p["local_y"])
            if dist2d(xy, last_xy) >= self.return_waypoint_spacing:
                sparse.append(p)
                last_xy = xy

        # Ensure the final point is close to origin / start.
        final_start = reverse_path[-1]
        if sparse[-1] is not final_start:
            sparse.append(final_start)

        return sparse

    def compute_frontiers_from_map(self):
        """
        Identify candidate frontier cells from the occupancy grid.

        Free cell adjacent to unknown cell = frontier. These are useful for a
        future search node to decide where unexplored boundaries are.
        """
        self.frontier_cells.clear()
        if self.occupancy_grid is None:
            return

        occ = self.occupancy_grid
        h, w = occ.shape

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if occ[y, x] != 0:
                    continue
                neighbourhood = occ[y - 1 : y + 2, x - 1 : x + 2]
                if np.any(neighbourhood == -1):
                    self.frontier_cells.add((x, y))

    # ============================================================
    # Visualisation / saving
    # ============================================================
    def publish_paths(self):
        if not self.origin_set:
            return

        now_msg = self.get_clock().now().to_msg()

        def make_path_msg(points: List[Dict], frame_id: str = "odom") -> RosPath:
            msg = RosPath()
            msg.header = Header()
            msg.header.stamp = now_msg
            msg.header.frame_id = frame_id

            for p in points:
                ps = PoseStamped()
                ps.header = msg.header
                ps.pose.position.x = float(p["raw_x"])
                ps.pose.position.y = float(p["raw_y"])
                ps.pose.position.z = 0.0
                ps.pose.orientation.w = 1.0
                msg.poses.append(ps)
            return msg

        self.safe_path_pub.publish(make_path_msg(self.safe_path))
        self.return_path_pub.publish(make_path_msg(self.make_sparse_return_path()))

    def save_outputs(self, final: bool = False):
        self.compute_frontiers_from_map()
        return_path = self.make_sparse_return_path()

        out_json = self.output_dir / f"{self.output_basename}.json"
        out_summary = self.output_dir / f"{self.output_basename}_summary.yaml"
        out_png = self.output_dir / f"{self.output_basename}_debug.png"

        data = {
            "created_unix_time": time.time(),
            "final": bool(final),
            "namespace": self.ns,
            "map": {
                "source": self.map_source,
                "resolution": self.map_resolution,
                "origin": [self.map_origin_x, self.map_origin_y, 0.0],
                "width": self.map_width,
                "height": self.map_height,
            },
            "collector_origin": {
                "raw_x": self.origin_raw_x,
                "raw_y": self.origin_raw_y,
                "raw_yaw": self.origin_raw_yaw,
            },
            "latest_pose": {
                "raw_x": self.raw_x,
                "raw_y": self.raw_y,
                "raw_yaw": self.raw_yaw,
                "local_x": self.local_x,
                "local_y": self.local_y,
                "local_yaw": self.local_yaw,
            },
            "safe_path": self.safe_path,
            "return_path": return_path,
            "visited_cells": [[int(x), int(y)] for x, y in sorted(self.visited_cells)],
            "free_cells": [[int(x), int(y)] for x, y in sorted(self.free_cells)],
            "blocked_cells": [[int(x), int(y)] for x, y in sorted(self.blocked_cells)],
            "frontier_cells": [[int(x), int(y)] for x, y in sorted(self.frontier_cells)],
            "notes": [
                "safe_path is the manually demonstrated safe trajectory from Phase 1.",
                "return_path is the reverse sparse waypoint path for Phase 2 returning.",
                "blocked/free/visited/frontier cells are lightweight memory, not a replacement for SLAM.",
            ],
        }

        with out_json.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        summary = {
            "map_source": self.map_source,
            "safe_path_points": len(self.safe_path),
            "return_waypoints": len(return_path),
            "visited_cells": len(self.visited_cells),
            "free_cells": len(self.free_cells),
            "blocked_cells": len(self.blocked_cells),
            "frontier_cells": len(self.frontier_cells),
            "latest_local_pose": [round(self.local_x, 3), round(self.local_y, 3), round(self.local_yaw, 3)],
            "outputs": {
                "json": str(out_json),
                "summary": str(out_summary),
                "debug_png": str(out_png),
            },
        }

        with out_summary.open("w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False)

        self.save_debug_image(out_png)

        self.get_logger().info(
            f"Saved memory: safe_path={len(self.safe_path)}, "
            f"return={len(return_path)}, visited={len(self.visited_cells)}, "
            f"blocked={len(self.blocked_cells)} -> {out_json}"
        )

    def save_debug_image(self, out_png: Path):
        if self.map_width <= 0 or self.map_height <= 0:
            return

        if self.occupancy_grid is not None:
            occ = self.occupancy_grid
            img = np.full((self.map_height, self.map_width, 3), 180, dtype=np.uint8)
            img[occ == 0] = (255, 255, 255)
            img[occ >= 50] = (0, 0, 0)
            img[occ < 0] = (160, 160, 160)
        else:
            img = np.full((self.map_height, self.map_width, 3), 220, dtype=np.uint8)

        # Draw memory overlays.
        for x, y in self.free_cells:
            if self.cell_in_bounds(x, y):
                img[y, x] = (220, 255, 220)

        for x, y in self.blocked_cells:
            if self.cell_in_bounds(x, y):
                img[y, x] = (0, 0, 255)

        for x, y in self.visited_cells:
            if self.cell_in_bounds(x, y):
                img[y, x] = (255, 180, 0)

        for x, y in self.frontier_cells:
            if self.cell_in_bounds(x, y):
                img[y, x] = (255, 0, 255)

        # Draw safe path and return path.
        def draw_points(points: List[Dict], color, radius=1):
            for p in points:
                cx = int(p.get("cell_x", -1))
                cy = int(p.get("cell_y", -1))
                if self.cell_in_bounds(cx, cy):
                    cv2.circle(img, (cx, cy), radius, color, -1)

        draw_points(self.safe_path, (255, 0, 0), radius=1)
        draw_points(self.make_sparse_return_path(), (0, 255, 255), radius=2)

        # PGM/YAML map y-axis convention can appear vertically flipped in image viewers;
        # flip for more intuitive debug viewing.
        img = cv2.flip(img, 0)
        scale = max(4, min(12, int(600 / max(1, max(self.map_width, self.map_height)))))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_png), img)

    # ============================================================
    # Periodic loop
    # ============================================================
    def periodic_update(self):
        if not self.origin_set:
            return

        self.maybe_record_breadcrumb()
        self.publish_paths()

        now = time.time()
        if now - self.last_save_time >= self.autosave_interval:
            self.last_save_time = now
            self.save_outputs(final=False)


def main(args=None):
    rclpy.init(args=args)
    node = Phase1MapDataCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt. Saving final Phase 1 memory.")
    finally:
        try:
            node.save_outputs(final=True)
        except Exception as e:
            node.get_logger().warn(f"Failed during final save: {e}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

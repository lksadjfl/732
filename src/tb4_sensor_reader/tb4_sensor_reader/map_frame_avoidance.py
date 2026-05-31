#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import time

import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PoseStamped
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


# ============================================================
# 全局参数区（按参数性质分组）
# ============================================================
# 说明：
# 1. 本版本使用 HSV 检测红色方块，不使用 YOLO。
# 2. 运动控制 / 状态机主体逻辑保持不变，仅整理参数分组和注释。
# 3. 调参优先改本区域，避免在函数内部写 magic number。


# ============================================================
# A. ROS2 节点、话题与文件路径
# ============================================================

NAMESPACE = "/T27"
NODE_NAME = "phase2_autonomous"

START_SIDE = "right"  # "right" 或 "left"
WALL_SIGN = 1.0 if START_SIDE == "right" else -1.0

SAVE_DIR = os.path.expanduser("~/tb4_phase2_evidence")
EVIDENCE_DIR_PREFIX = "tb4_red_evidence"
DEBUG_DIR_PREFIX = "tb4_lidar_debug"

# Phase 1 已经扫过 aisle 时，Phase 2 可以加载这些记忆路径：
#   1) phase1_navigation_memory.json: safe_path / return_path
#   2) phase1_env_data.json: trajectory.points
PHASE1_MEMORY_CANDIDATES = [
    "phase1_navigation_memory.json",
    "phase1_env_data.json",
]


# ============================================================
# B. 控制周期与状态行为开关
# ============================================================

CONTROL_DT = 0.10
ENABLE_DEBUG_LOG = False
STARTUP_WAIT_SEC = 3.0

STUCK_MONITORED_STATES = {
    "SEARCH_WALL_FOLLOW",
    "AVOID_OBSTACLE",
    "REJOIN_WALL",
    "RETURNING",
}

# 默认使用已经在实体机器人上验证过的右墙跟随三状态机。


# ============================================================
# C. 线速度参数（m/s）
# ============================================================

FORWARD_SPEED = 0.28
SLOW_FORWARD_SPEED = 0.10
BACKWARD_SPEED = -0.10
RETURN_MAX_LINEAR = 0.10
RETURN_MIN_LINEAR = 0.025
RETURN_WALL_FOLLOW_MAX_LINEAR = 0.065
RETURN_FINAL_MAX_LINEAR = 0.055
RETURN_FINAL_MIN_LINEAR = 0.018
RETURN_BACKUP_SPEED = -0.06
RETURN_ESCAPE_LINEAR = 0.055


# ============================================================
# D. 角速度参数（rad/s）
# ============================================================

TURN_SPEED = 0.45
SMALL_TURN_SPEED = 0.25
RETURN_MAX_ANGULAR = 0.45
RETURN_FINAL_MAX_ANGULAR = 0.35
RETURN_NO_WALL_TURN_SPEED = 0.28
RETURN_ESCAPE_ANGULAR = 0.38
POST_RETURN_TURN_SPEED = 0.35


# ============================================================
# E. 距离、尺寸与安全余量（m）
# ============================================================

ROBOT_DIAMETER = 0.30
ROBOT_RADIUS = ROBOT_DIAMETER / 2.0
ROBOT_SIDE_CLEARANCE = 0.09
ROBOT_FRONT_CLEARANCE = 0.18

FRONT_STOP_DIST = 0.35
FRONT_WARN_DIST = 0.50

TARGET_RIGHT_DIST = 0.38
MIN_RIGHT_DIST = 0.24
MAX_RIGHT_DIST = 0.58

PROTRUSION_FRONT_DANGER = max(0.30, ROBOT_RADIUS + ROBOT_FRONT_CLEARANCE)
PROTRUSION_FRONT_WARN = 0.55
PROTRUSION_EDGE_LOST_DIST = 0.75

AVOID_EXIT_YAW_TOL_DEG = 75.0

CHANGE_DIST_TH = 0.18
CHANGE_NEAR_DIST = 0.95
CHANGE_MAX_VALID = 3.5

PATTERN_NEAR_DIST = 0.50
PATTERN_EDGE_JUMP = 0.35
PATTERN_MAX_VALID = 3.5
OBSTACLE_MAX_WIDTH = 0.55

RIGHT_SHAPE_MAX_VALID = 3.5
RIGHT_SHAPE_VALLEY_MIN_DEPTH = 0.055
RIGHT_WALL_VISIBLE_MAX_DIST = 0.82

RETURN_FINAL_DIRECT_DIST = 0.15
# [Priority 2] 6cm 对轮式里程计过于严格，里程漂移/打滑/贴墙误差很容易超过它，
# 导致永远无法满足完成条件。放宽到 0.18m 显著提高返回成功率。
RETURN_STOP_DIST = 0.18
RETURN_FINAL_STRAIGHT_DIST = 0.20
RETURN_SLOW_DIST = 0.55
RETURN_SAFE_DIST = 0.65
RETURN_WARN_DIST = 0.85
RETURN_DANGER_DIST = 0.45
RETURN_SIDE_DANGER_DIST = 0.36

PATH_RECORD_MIN_DIST = 0.18
WAYPOINT_REACHED_DIST = 0.22
PHASE1_SEARCH_WAYPOINT_REACHED_DIST = 0.24
PHASE1_SEARCH_LOOKAHEAD = 3
PHASE1_SEARCH_MAX_LINEAR = 0.22
PHASE1_SEARCH_MIN_LINEAR = 0.08
PHASE1_SEARCH_ANGULAR_K = 1.10
PHASE1_SEARCH_ANGULAR_DEADBAND_DEG = 4.0
PHASE1_SEARCH_BLOCKED_DIST = 0.48
PHASE1_SEARCH_STRONG_BLOCKED_DIST = 0.34
PHASE1_REJOIN_MAX_SKIP_DIST = 0.80

STUCK_MOVE_DIST = 0.08

RED_CUBE_SIZE_M = 0.06


# ============================================================
# F. 角度、扇区宽度与角度容差（deg / rad）
# ============================================================

FRONT_ANGLE = -math.pi / 2.0

CHANGE_FRONT_ROI_DEG = 80.0

CHANGE_MIN_CLUSTER_DEG = 3.0
WALL_CLUSTER_DEG = 35.0
OBSTACLE_MIN_DEG = 5.0
OBSTACLE_MAX_DEG = 32.0

RIGHT_FRONT_ANGLE_DEG = -60.0
RIGHT_MID_ANGLE_DEG = -90.0
RIGHT_BACK_ANGLE_DEG = -120.0
RIGHT_PARALLEL_ARC_DEG = 22.0

RIGHT_SHAPE_CENTER_DEG = -90.0
RIGHT_SHAPE_ARC_DEG = 120.0
RIGHT_SHAPE_MIN_VALLEY_WIDTH_DEG = 20.0
RIGHT_WALL_VISIBLE_MIN_VALLEY_WIDTH_DEG = 10.0

GOAL_ARC_DEG = 35
OPENING_ARC_DEG = 25
RETURN_SCAN_MIN_DEG = -95
RETURN_SCAN_MAX_DEG = 95
RETURN_SCAN_STEP_DEG = 10
RETURN_WAYPOINT_DIRECT_ANGLE_DEG = 38.0
RETURN_ALIGN_ONLY_DEG = 58.0
RETURN_FINAL_ALIGN_ONLY_DEG = 45.0

PATH_RECORD_MIN_YAW_DEG = 15.0

POST_RETURN_TURN_ANGLE = math.pi
POST_RETURN_TURN_TOL = math.radians(4.0)


# ============================================================
# G. 时间参数（s）
# ============================================================

STUCK_TIME = 6.0

BACKUP_DURATION = 0.70
ESCAPE_TURN_DURATION = 1.20

RETURN_ESCAPE_BACKUP_TIME = 0.35
RETURN_ESCAPE_TURN_TIME = 0.65
RETURN_ESCAPE_ARC_TIME = 1.25

EVIDENCE_SNAPSHOT_COOLDOWN = 0.35
DEBUG_EVENT_COOLDOWN = 0.80


# ============================================================
# H. 控制增益与角速度叠加系数
# ============================================================

RIGHT_DIST_K = 1.00
RIGHT_PARALLEL_K = 0.70
RIGHT_SHAPE_ERROR_SCALE = 0.42

RETURN_LINEAR_K = 0.30
RETURN_ANGULAR_K = 1.25
RETURN_FINAL_LINEAR_K = 0.28

RETURN_TARGET_BIAS_K = 0.42
RETURN_TARGET_BIAS_MAX = 0.16


# ============================================================
# I. 比例、计数、防抖与形状判断阈值
# ============================================================

WALL_RATIO_TH = 0.60
PROTRUSION_REJOIN_FRONT_WALL_RATIO = 0.50

RIGHT_SHAPE_SMOOTH_WINDOW = 9
RIGHT_SHAPE_MIN_VALID_RATIO = 0.48
RIGHT_SHAPE_CENTER_TOL = 0.18
RIGHT_SHAPE_MONO_RATIO_TH = 0.62
RIGHT_SHAPE_MAX_JUMP = 0.16
RIGHT_SHAPE_MAX_JUMP_RATIO = 0.06
RIGHT_PARALLEL_TOL = 0.16
RIGHT_WALL_VISIBLE_MIN_VALID_RATIO = 0.42
RIGHT_WALL_VISIBLE_MAX_JUMP_RATIO = 0.16

RIGHT_WALL_STABLE_COUNT = 3
RIGHT_WALL_VISIBLE_STABLE_COUNT = 2
RETURN_BLOCKED_CONFIRM = 2
RETURN_GOAL_CLEAR_REQUIRED = 3

SCORE_DROP_TOLERANCE = 0.12

SCORE_MONOTONIC_DURATION = 60.0
SCORE_MONOTONIC_MIN_ACTIVATION = 0.30
SCORE_MONOTONIC_TOL_NORMAL = 0.12
SCORE_MONOTONIC_TOL_AVOID = 0.35
SCORE_MONOTONIC_DIR_EPS = 0.05
SCORE_SIGN_DEADZONE = 0.05

MAX_SAFE_PATH_POINTS = 800
EVIDENCE_MAX_AUTO_SNAPSHOTS = 8


# ============================================================
# I2.【停用实验代码】结构化场景解析参数
# ============================================================

CYLINDER_DIAMETER_M = 0.12
CYLINDER_RADIUS_M = CYLINDER_DIAMETER_M / 2.0
CYLINDER_RADIUS_TOL_M = 0.05
CORRIDOR_WIDTH_MIN_M = 1.10
CORRIDOR_WIDTH_MAX_M = 1.30

SCENE_MAX_VALID_M = 3.0
SCENE_MIN_VALID_M = 0.05
SCENE_FRONT_ROI_DEG = 55.0

SCENE_BREAK_K = 0.08
SCENE_BREAK_C = 0.06
SCENE_MIN_SEG_POINTS = 5

SCENE_IEPF_ENABLE = True
SCENE_IEPF_SPLIT_DIST = 0.10

SCENE_LINE_RES_MAX = 0.04
SCENE_CIRCLE_RES_MAX = 0.03

CYLINDER_AVOID_TRIGGER_DIST = 0.50
CYLINDER_AVOID_CLEARANCE = 0.30
# 固定方向的圆柱侧绕容易在 LiDAR 分类抖动时形成缓慢左右摆动。
# 关闭后仍保留 waypoint-biased gap 绕障和全局碰撞保护。
ENABLE_CRAB_WALK_AVOIDANCE = False

SCENE_GAP_FAR_DIST = 1.50
SCENE_GAP_MIN_WIDTH_DEG = 18.0
SCENE_GAP_HEADING_W = 1.0
SCENE_GAP_FORWARD_W = 0.35
SCENE_GAP_DEPTH_W = 0.10

SCENE_WALL_DIST_K = 1.10
SCENE_WALL_HEADING_K = 0.90
SCENE_FOLLOW_WALL_STABLE_COUNT = 2
# [Priority 4] 2 帧确认太少，单帧 LiDAR 伪影即可触发绕行。提高到 4 帧更贴合真实环境。
SCENE_CYLINDER_CONFIRM_FRAMES = 4

SCENE_SAFETYNET_FRONT_STOP = FRONT_STOP_DIST

# [Priority 7] gap 评分新增“与当前跟随墙连续性”权重：
# 偏向不会让机器人切换跟随侧/掉头的 gap，避免绕障后反向。
SCENE_GAP_CONTINUITY_W = 0.6

# ============================================================
# I3.【方案2 返回】RETURNING 一致化参数（Priority 1/3/5/9）
# ============================================================

# [Priority 1] RETURNING 在 scene 墙跟随角速度上叠加的弱原点吸引偏置增益与上限(rad/s)。
# angular = wall_follow_term + clamp(RETURN_ORIGIN_BIAS_K * goal_angle, ±MAX)
RETURN_ORIGIN_BIAS_K = 0.45
RETURN_ORIGIN_BIAS_MAX = 0.18

# [Priority 9] 全局碰撞保护：任意 LiDAR beam 距离小于该值，立刻覆盖所有行为，
# linear=0 并朝远离最近障碍方向急转。适用于 SEARCHING / RETURNING / ESCAPE。
CRITICAL_COLLISION_DIST = 0.15
CRITICAL_COLLISION_MIN_VALID_DIST = 0.05
# 只检查机器人前方和前侧。完整 360° 扫描会让后方近墙或 LiDAR 自反射
# 永久覆盖正常导航，表现为机器人在起点持续原地旋转。
CRITICAL_COLLISION_ARC_DEG = 110.0
# 急停转向角速度。
CRITICAL_COLLISION_TURN_SPEED = TURN_SPEED
# 旧版稳定控制器没有这一层全局覆盖。默认关闭，避免 LiDAR 近距离噪声
# 抢占右墙跟随并让机器人在起点持续原地旋转。
ENABLE_CRITICAL_COLLISION_OVERRIDE = False


# ============================================================
# J. 红色方块检测：HSV 阈值与 bbox 过滤
# ============================================================

RED_LOW1 = np.array([0, 150, 110])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 150, 110])
RED_HIGH2 = np.array([180, 255, 255])

MIN_RED_PIXELS = 5800
RED_CONFIRM_FRAMES = 3

RED_ASPECT_MIN = 0.5
RED_ASPECT_MAX = 2.0
MIN_BOX_WIDTH_PX = 8
MIN_BOX_HEIGHT_PX = 8


# ============================================================
# K. 相机单目测距参数
# ============================================================

OAK_RGB_HFOV_DEG = 66.0
OAK_RGB_VFOV_DEG = 54.0

USE_FIXED_FOCAL_LENGTH = False
FOCAL_LENGTH_PX = 615.0

# =========================
# 工具函数
# =========================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_to_index(angle, angle_min, angle_increment, n):
    return int(round((angle - angle_min) / angle_increment)) % n


def extract_arc(ranges, angle_min, angle_increment, center_angle, arc_deg):
    n = len(ranges)
    half = math.radians(arc_deg / 2.0)

    start_angle = center_angle - half
    end_angle = center_angle + half

    start_idx = angle_to_index(start_angle, angle_min, angle_increment, n)
    end_idx = angle_to_index(end_angle, angle_min, angle_increment, n)

    if start_idx <= end_idx:
        arc = ranges[start_idx:end_idx + 1]
    else:
        arc = ranges[start_idx:] + ranges[:end_idx + 1]

    return np.array(arc, dtype=float)


def clean_ranges(arc, max_valid=PATTERN_MAX_VALID):
    arc = np.array(arc, dtype=float)

    valid = np.isfinite(arc)
    arc_clean = arc.copy()

    arc_clean[~valid] = max_valid
    arc_clean = np.clip(arc_clean, 0.0, max_valid)

    return arc_clean, valid


def find_near_clusters(near_mask):
    clusters = []
    start = None

    for i, v in enumerate(near_mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            clusters.append((start, i - 1))
            start = None

    if start is not None:
        clusters.append((start, len(near_mask) - 1))

    return clusters


def find_true_clusters_circular(mask):
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)

    if n == 0 or not np.any(mask):
        return []

    if np.all(mask):
        return [list(range(n))]

    false_indices = np.where(~mask)[0]
    start_scan = int(false_indices[0])

    clusters = []
    current = []

    for k in range(1, n + 1):
        idx = (start_scan + k) % n
        if mask[idx]:
            current.append(idx)
        elif current:
            clusters.append(current)
            current = []

    if current:
        clusters.append(current)

    return clusters


# =========================
# 主节点
# =========================

class Phase2Autonomous(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        # ---- 从 ROS 参数读取 namespace（取代硬编码 NAMESPACE） ----
        self.declare_parameter("namespace", NAMESPACE)
        self.declare_parameter("phase1_memory_file", "")
        ns = str(self.get_parameter("namespace").value).rstrip("/")
        phase1_memory_file = str(self.get_parameter("phase1_memory_file").value).strip()

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.evidence_run_dir = os.path.join(
            os.getcwd(),
            f"{EVIDENCE_DIR_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.evidence_run_dir, exist_ok=True)

        self.evidence_csv_path = os.path.join(self.evidence_run_dir, "red_range_measurements.csv")
        self.evidence_snapshot_count = 0
        self.evidence_last_snapshot_time = 0.0
        self.latest_range_result = None
        self.init_evidence_log_file()

        self.debug_run_dir = None
        self.debug_csv_path = None
        self.debug_event_id = 0
        self.debug_last_save_time = {}

        if ENABLE_DEBUG_LOG:
            self.debug_run_dir = os.path.join(
                os.getcwd(),
                f"{DEBUG_DIR_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            os.makedirs(self.debug_run_dir, exist_ok=True)
            self.debug_csv_path = os.path.join(self.debug_run_dir, "debug_events.csv")
            self.init_debug_log_file()

        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(
            Twist,
            f"{ns}/cmd_vel",
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            f"{ns}/scan",
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            f"{ns}/odom",
            self.odom_callback,
            10
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            f"{ns}/oakd/rgb/image_raw/compressed",
            self.image_callback,
            10
        )

        # ---- AMCL / map-frame localization ----
        self.amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f"{ns}/amcl_pose",
            self.amcl_pose_callback,
            10
        )

        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            f"{ns}/initialpose",
            self.initial_pose_callback,
            10
        )

        # AMCL state
        self.amcl_pose_received = False
        self.amcl_x = 0.0
        self.amcl_y = 0.0
        self.amcl_yaw = 0.0
        self.home_map_x = 0.0
        self.home_map_y = 0.0
        self.home_map_yaw = 0.0
        self.home_map_set = False

        # ---- 搜索路径面包屑（SEARCHING 阶段记录，RETURNING 折返用） ----
        self.search_path = []              # list of (x, y, yaw) in AMCL map coords
        self.search_path_local = []        # list of (x, y, yaw) in Phase2 local odom frame
        self.search_path_last_x = 0.0
        self.search_path_last_y = 0.0
        self.search_path_local_last_x = 0.0
        self.search_path_local_last_y = 0.0
        self.search_path_record_dist = 0.25  # 每 0.25m 记录一个点
        self.return_waypoint_idx = -1        # RETURNING 当前目标 waypoint 索引
        self.return_local_waypoint_idx = -1  # RETURNING local search_path 当前目标 waypoint 索引
        self.phase1_return_waypoint_idx = -1 # RETURNING Phase1 return_path 当前目标 waypoint 索引

        # ---- Phase 1 aisle 记忆路径 ----
        # Phase 1 已经扫过 aisle，因此 Phase 2 SEARCHING 优先沿 safe_path 搜索；
        # Phase 2 实时 LiDAR 只作为新增障碍物绕行覆盖层。
        self.phase1_memory_file = phase1_memory_file
        self.phase1_memory_loaded = False
        self.phase1_memory_source = "none"
        self.phase1_safe_path = []           # list of (x_right, y_forward, yaw) in Phase2 local frame
        self.phase1_return_path = []         # same frame, sparse reverse path if available
        self.phase1_safe_path_map = []       # same paths anchored to AMCL home in map frame
        self.phase1_return_path_map = []
        self.phase1_search_idx = 0
        self.phase1_last_rejoin_time = 0.0

        self.latest_scan = None
        self.prev_scan_ranges = None
        self.curr_scan_ranges = None
        self.prev_scan_stamp = None
        self.curr_scan_stamp = None

        self.latest_frame = None
        self.latest_red_mask = None
        self.latest_red_bbox = None
        self.latest_red_hsv_stats = None

        self.x_raw = 0.0
        self.y_raw = 0.0
        self.yaw_raw = 0.0
        self.yaw = math.pi / 2.0
        self.have_odom = False

        self.start_x_raw = 0.0
        self.start_y_raw = 0.0
        self.start_yaw = 0.0
        self.start_recorded = False

        self.x = 0.0
        self.y = 0.0

        self.best_score = 0.0
        self.last_score = 0.0

        self.score_monotonic_start_time = None

        # Start after a short fixed delay. AMCL/initialpose callbacks are still
        # optional localization improvements; RViz "2D Pose Estimate" is not
        # required before autonomous search begins.
        self.state = "SEARCH_WALL_FOLLOW"
        self.state_start_time = time.time()
        self.startup_wait_until = time.time() + STARTUP_WAIT_SEC
        self.startup_wait_logged = False
        self.escape_resume_state = "SEARCH_WALL_FOLLOW"

        self.follow_wall_stable_count = 0
        self.cylinder_confirm_count = 0
        self.avoiding_cylinder = False
        self.avoid_cylinder_side = 0.0
        self.stuck_ref_x = 0.0
        self.stuck_ref_y = 0.0
        self.stuck_ref_time = time.time()

        self.red_detected = False
        self.red_pixels = 0
        self.saved_detection = False

        self.target_locked = False
        self.red_seen_count = 0

        self.cube_robot_x = None
        self.cube_robot_y = None
        self.cube_global_x = None
        self.cube_global_y = None
        self.cube_distance = None

        self.last_log_time = 0.0
        self.last_return_log_time = 0.0

        self.load_phase1_memory()

        self.rejoin_stable_count = 0

        self.avoid_entry_yaw = 0.0

        # [Priority 1/3/5] RETURNING 使用独立的多帧确认计数，避免与 SEARCHING 串扰。
        self.return_follow_wall_stable_count = 0
        self.return_cylinder_confirm_count = 0
        self.return_avoiding_cylinder = False
        self.return_avoid_cylinder_side = 0.0

        self.post_return_turn_start_yaw = 0.0

        self.timer = self.create_timer(CONTROL_DT, self.control_loop)

        self.get_logger().info("Phase2 autonomous node started.")
        self.get_logger().info(f"Waiting {STARTUP_WAIT_SEC:.1f}s before autonomous motion.")
        self.get_logger().info(f"Namespace: {ns} (AMCL home mode: map from Phase 1)")
        self.get_logger().info(
            "Strategy: SEARCHING=legacy right-wall follow + edge-follow avoidance | "
            "RETURNING=legacy right-wall follow + edge-follow avoidance | "
            "structured scene runtime=disabled"
        )
        if self.phase1_memory_loaded:
            self.get_logger().info(
                f"Phase1 memory loaded: source={self.phase1_memory_source}, "
                f"safe_path={len(self.phase1_safe_path)}, return_path={len(self.phase1_return_path)}"
            )
        else:
            self.get_logger().warn(
                "No Phase1 memory path loaded. Legacy wall-follow remains available. "
                "Pass -p phase1_memory_file:=/path/to/phase1_navigation_memory.json if needed."
            )
        if ENABLE_DEBUG_LOG:
            self.get_logger().info(f"Debug event folder: {self.debug_run_dir}")
            self.get_logger().info(f"Debug CSV log: {self.debug_csv_path}")
        else:
            self.get_logger().info("LiDAR debug event log disabled.")
        self.get_logger().info(f"Red ranging evidence folder: {self.evidence_run_dir}")
        self.get_logger().info(f"Red ranging CSV log: {self.evidence_csv_path}")

    # =========================
    # Phase 1 记忆路径加载 / 坐标转换
    # =========================

    def _candidate_phase1_memory_paths(self):
        paths = []
        if self.phase1_memory_file:
            paths.append(os.path.expanduser(self.phase1_memory_file))

        for name in PHASE1_MEMORY_CANDIDATES:
            paths.append(os.path.join(os.getcwd(), name))
            paths.append(os.path.expanduser(os.path.join("~", name)))

        # 去重但保持顺序
        unique = []
        seen = set()
        for p in paths:
            if p not in seen:
                unique.append(p)
                seen.add(p)
        return unique

    def _dedupe_path_points(self, pts, min_spacing=0.05):
        clean = []
        last = None
        for x, y, yaw in pts:
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if last is not None and math.hypot(x - last[0], y - last[1]) < min_spacing:
                continue
            clean.append((float(x), float(y), float(yaw)))
            last = clean[-1]
        return clean

    def _parse_phase1_local_path(self, raw_points):
        """解析 phase1_navigation_memory.json 的 safe_path/return_path。

        collector 的 local_x 是 Phase1 起点前进方向，local_y 是左侧方向；
        本节点内部坐标是 x=右，y=前，因此转换为：
            x_right = -local_y, y_forward = local_x。
        """
        pts = []
        for p in raw_points:
            if "local_x" not in p or "local_y" not in p:
                continue
            x_right = -float(p.get("local_y", 0.0))
            y_forward = float(p.get("local_x", 0.0))
            yaw = normalize_angle(float(p.get("local_yaw", 0.0)) + math.pi / 2.0)
            pts.append((x_right, y_forward, yaw))
        return self._dedupe_path_points(pts)

    def _parse_phase1_raw_trajectory(self, raw_points):
        """解析 phase1_env_data.json 的 trajectory.points。

        该文件保存 odom/raw x,y,yaw。这里用第一帧作为 Phase2 本地原点，
        转换到本节点 x=右, y=前 的相对坐标。
        """
        if not raw_points:
            return []
        first = raw_points[0]
        ox = float(first.get("x", first.get("robot_x", 0.0)))
        oy = float(first.get("y", first.get("robot_y", 0.0)))
        oyaw = float(first.get("yaw", 0.0))

        pts = []
        for p in raw_points:
            px = float(p.get("x", p.get("robot_x", ox)))
            py = float(p.get("y", p.get("robot_y", oy)))
            pyaw = float(p.get("yaw", oyaw))

            dx = px - ox
            dy = py - oy
            forward = dx * math.cos(oyaw) + dy * math.sin(oyaw)
            right = dx * math.sin(oyaw) - dy * math.cos(oyaw)
            yaw = normalize_angle((pyaw - oyaw) + math.pi / 2.0)
            pts.append((right, forward, yaw))
        return self._dedupe_path_points(pts)

    def load_phase1_memory(self):
        for path in self._candidate_phase1_memory_paths():
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                self.get_logger().warn(f"Failed to read Phase1 memory {path}: {e}")
                continue

            safe_path = []
            return_path = []
            source = os.path.basename(path)

            if isinstance(data.get("safe_path"), list):
                safe_path = self._parse_phase1_local_path(data.get("safe_path", []))
                if isinstance(data.get("return_path"), list):
                    return_path = self._parse_phase1_local_path(data.get("return_path", []))
            elif isinstance(data.get("trajectory"), dict):
                traj = data.get("trajectory", {}).get("points", [])
                safe_path = self._parse_phase1_raw_trajectory(traj)

            if len(safe_path) >= 2:
                self.phase1_memory_file = path
                self.phase1_memory_source = source
                self.phase1_safe_path = safe_path
                self.phase1_return_path = return_path if len(return_path) >= 2 else list(reversed(safe_path))
                self._refresh_phase1_map_paths()
                self.phase1_search_idx = 0
                self.phase1_memory_loaded = True
                return True

            self.get_logger().warn(
                f"Phase1 memory file found but no usable path: {path}"
            )

        return False

    def _phase1_local_point_to_map(self, point):
        """Anchor a Phase2-local (right, forward, yaw) point to the AMCL home pose."""
        x_right, y_forward, yaw_local = point
        c = math.cos(self.home_map_yaw)
        s = math.sin(self.home_map_yaw)
        map_x = self.home_map_x + y_forward * c + x_right * s
        map_y = self.home_map_y + y_forward * s - x_right * c
        map_yaw = normalize_angle(self.home_map_yaw + yaw_local - math.pi / 2.0)
        return map_x, map_y, map_yaw

    def _refresh_phase1_map_paths(self):
        """Rebuild map-frame memory paths after AMCL home is established or updated."""
        if not self.home_map_set:
            self.phase1_safe_path_map = []
            self.phase1_return_path_map = []
            return
        self.phase1_safe_path_map = [
            self._phase1_local_point_to_map(p) for p in self.phase1_safe_path
        ]
        self.phase1_return_path_map = [
            self._phase1_local_point_to_map(p) for p in self.phase1_return_path
        ]

    def _phase1_active_path(self, local_path, map_path):
        """Return path and current pose in the best available navigation frame."""
        if self.amcl_pose_received and self.home_map_set and len(map_path) == len(local_path):
            return map_path, self.amcl_x, self.amcl_y, True
        return local_path, self.x, self.y, False

    # =========================
    # ROS callbacks
    # =========================

    def scan_callback(self, msg):
        if self.curr_scan_ranges is not None:
            self.prev_scan_ranges = self.curr_scan_ranges.copy()
            self.prev_scan_stamp = self.curr_scan_stamp

        self.curr_scan_ranges = np.array(msg.ranges, dtype=float)
        self.curr_scan_stamp = self.get_clock().now().nanoseconds
        self.latest_scan = msg

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.x_raw = p.x
        self.y_raw = p.y
        self.yaw_raw = yaw_from_quaternion(q)
        self.have_odom = True

        if not self.start_recorded:
            self.start_x_raw = self.x_raw
            self.start_y_raw = self.y_raw
            self.start_yaw = self.yaw_raw
            self.start_recorded = True

            self.x = 0.0
            self.y = 0.0
            self.yaw = math.pi / 2.0

            self.stuck_ref_x = self.x
            self.stuck_ref_y = self.y
            self.stuck_ref_time = time.time()

            self.score_monotonic_start_time = time.time()

            self.get_logger().info(
                f"Start recorded as origin: x=0.000, y=0.000, "
                f"coordinate frame: +Y=initial forward, +X=initial right | "
                f"raw_x={self.start_x_raw:.3f}, raw_y={self.start_y_raw:.3f}, "
                f"raw_yaw={math.degrees(self.start_yaw):.1f} deg, "
                f"frame_yaw={math.degrees(self.yaw):.1f} deg"
            )
        else:
            dx_raw = self.x_raw - self.start_x_raw
            dy_raw = self.y_raw - self.start_y_raw

            forward = dx_raw * math.cos(self.start_yaw) + dy_raw * math.sin(self.start_yaw)
            right = dx_raw * math.sin(self.start_yaw) - dy_raw * math.cos(self.start_yaw)

            self.x = right
            self.y = forward

            self.yaw = normalize_angle((self.yaw_raw - self.start_yaw) + math.pi / 2.0)

    def amcl_pose_callback(self, msg):
        """接收 AMCL 位姿（map 坐标系）。第一次收到时记录 home 位置。"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.amcl_x = p.x
        self.amcl_y = p.y
        self.amcl_yaw = yaw_from_quaternion(q)
        self.amcl_pose_received = True

        if not self.home_map_set:
            self.home_map_x = self.amcl_x
            self.home_map_y = self.amcl_y
            self.home_map_yaw = self.amcl_yaw
            self.home_map_set = True
            self._refresh_phase1_map_paths()
            self.get_logger().info(
                f"AMCL home set in map frame: "
                f"({self.home_map_x:.3f}, {self.home_map_y:.3f}), "
                f"yaw={math.degrees(self.home_map_yaw):.1f} deg"
            )

    def initial_pose_callback(self, msg):
        """RViz '2D Pose Estimate' 回调 —— 更新 home 位置。"""
        if not self.amcl_pose_received:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.home_map_x = p.x
        self.home_map_y = p.y
        self.home_map_yaw = yaw_from_quaternion(q)
        self.home_map_set = True
        self._refresh_phase1_map_paths()
        self.get_logger().info(
            f"Home updated via initialpose: "
            f"({self.home_map_x:.3f}, {self.home_map_y:.3f}), "
            f"yaw={math.degrees(self.home_map_yaw):.1f} deg"
        )

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image decode failed: {e}")
            return

        self.latest_frame = frame

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
        mask2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.latest_red_mask = mask

        self.red_pixels = int(cv2.countNonZero(mask))
        self.latest_red_bbox = self.find_largest_red_bbox(mask)
        self.latest_red_hsv_stats = self.compute_red_hsv_stats(hsv, mask, self.latest_red_bbox)

        if self.target_locked:
            self.red_detected = True
        else:
            current_red = False

            if self.latest_red_bbox is not None:
                bx, by, bw, bh = self.latest_red_bbox

                if bh > 0:
                    aspect = bw / float(bh)
                else:
                    aspect = 999.0

                current_red = (
                    self.red_pixels >= MIN_RED_PIXELS
                    and bw >= MIN_BOX_WIDTH_PX
                    and bh >= MIN_BOX_HEIGHT_PX
                    and RED_ASPECT_MIN <= aspect <= RED_ASPECT_MAX
                )

            if current_red:
                self.red_seen_count += 1
            else:
                self.red_seen_count = 0

            if self.red_seen_count >= RED_CONFIRM_FRAMES:
                self.target_locked = True
                self.red_detected = True
                self.get_logger().info(
                    f"Red target locked after {self.red_seen_count} frames. red_pixels={self.red_pixels}"
                )
            else:
                self.red_detected = False

        if self.latest_red_bbox is not None and self.red_pixels >= MIN_RED_PIXELS:
            self.save_red_range_snapshot(trigger="threshold", force=False)

        display = frame.copy()

        if self.latest_red_bbox is not None:
            x, y, w, h = self.latest_red_bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(display, (x + w // 2, y + h // 2), 5, (255, 0, 0), -1)

        cv2.putText(display, f"state={self.state}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, f"red_pixels={self.red_pixels}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, f"odom_rel=({self.x:.2f},{self.y:.2f})", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.latest_range_result is not None:
            cv2.putText(
                display,
                f"cube=(x_right={self.latest_range_result['cube_robot_x']:.2f},y_forward={self.latest_range_result['cube_robot_y']:.2f})m dist={self.latest_range_result['distance_robot']:.2f}m",
                (20, 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2
            )

        if self.target_locked:
            status_text = "RED LOCKED"
        elif self.red_seen_count > 0:
            status_text = f"RED CONFIRM {self.red_seen_count}/{RED_CONFIRM_FRAMES}"
        else:
            status_text = ""

        if status_text:
            cv2.putText(display, status_text, (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("TB4 Camera", display)
        cv2.waitKey(1)

    # =========================
    # 红色方块测距 evidence 记录
    # =========================

    def compute_red_hsv_stats(self, hsv_image, red_mask, bbox):
        if hsv_image is None or red_mask is None or bbox is None:
            return None

        img_h, img_w = hsv_image.shape[:2]
        bx, by, bw, bh = bbox
        x0 = max(0, int(bx))
        y0 = max(0, int(by))
        x1 = min(img_w, int(bx + bw))
        y1 = min(img_h, int(by + bh))

        if x1 <= x0 or y1 <= y0:
            return None

        hsv_roi = hsv_image[y0:y1, x0:x1]
        mask_roi = red_mask[y0:y1, x0:x1]
        red_pixels = hsv_roi[mask_roi > 0]

        sample_mode = "mask_pixels"
        samples = red_pixels
        if samples.size == 0:
            sample_mode = "bbox_all_pixels"
            samples = hsv_roi.reshape(-1, 3)

        if samples.size == 0:
            return None

        center_x = int(clamp(round(bx + bw / 2.0), 0, img_w - 1))
        center_y = int(clamp(round(by + bh / 2.0), 0, img_h - 1))
        center_hsv = hsv_image[center_y, center_x].astype(float)

        mean_hsv = np.mean(samples, axis=0)
        median_hsv = np.median(samples, axis=0)
        min_hsv = np.min(samples, axis=0)
        max_hsv = np.max(samples, axis=0)
        std_hsv = np.std(samples, axis=0)

        return {
            "sample_mode": sample_mode,
            "sample_count": int(len(samples)),
            "center_hsv": tuple(float(v) for v in center_hsv),
            "mean_hsv": tuple(float(v) for v in mean_hsv),
            "median_hsv": tuple(float(v) for v in median_hsv),
            "min_hsv": tuple(float(v) for v in min_hsv),
            "max_hsv": tuple(float(v) for v in max_hsv),
            "std_hsv": tuple(float(v) for v in std_hsv),
        }

    def init_evidence_log_file(self):
        header = [
            "snapshot_id", "wall_time", "trigger", "state",
            "robot_x", "robot_y", "robot_yaw_deg", "red_pixels",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_cx", "bbox_cy",
            "img_w", "img_h", "fx_px", "fy_px", "hfov_deg", "vfov_deg",
            "z_from_width_m", "z_from_height_m",
            "cube_robot_x_right_m", "cube_robot_y_forward_m",
            "cube_distance_from_robot_m", "cube_global_x_m", "cube_global_y_m",
            "bearing_deg", "hsv_sample_mode", "hsv_sample_count",
            "hsv_center_h", "hsv_center_s", "hsv_center_v",
            "hsv_mean_h", "hsv_mean_s", "hsv_mean_v",
            "hsv_median_h", "hsv_median_s", "hsv_median_v",
            "image_path",
        ]

        with open(self.evidence_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def get_rgb_intrinsics_from_image_size(self, img_w, img_h):
        if USE_FIXED_FOCAL_LENGTH:
            return float(FOCAL_LENGTH_PX), float(FOCAL_LENGTH_PX)

        fx = (img_w / 2.0) / math.tan(math.radians(OAK_RGB_HFOV_DEG) / 2.0)
        fy = (img_h / 2.0) / math.tan(math.radians(OAK_RGB_VFOV_DEG) / 2.0)
        return float(fx), float(fy)

    def draw_red_range_overlay(self, frame, result, trigger):
        display = frame.copy()

        if self.latest_red_bbox is not None:
            bx, by, bw, bh = self.latest_red_bbox
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            cv2.circle(display, (bx + bw // 2, by + bh // 2), 5, (255, 0, 0), -1)

        lines = [
            f"trigger={trigger} state={self.state}",
            f"robot=({self.x:.2f},{self.y:.2f}) yaw={math.degrees(self.yaw):.1f} deg",
            f"red_pixels={self.red_pixels} bbox={self.latest_red_bbox}",
        ]

        if result is not None:
            lines.extend([
                f"cube_robot: x_right={result['cube_robot_x']:.2f}m y_forward={result['cube_robot_y']:.2f}m",
                f"distance={result['distance_robot']:.2f}m bearing={math.degrees(result['bearing_rad']):.1f} deg",
                f"cube_global=({result['cube_global_x']:.2f},{result['cube_global_y']:.2f})m",
            ])
        else:
            lines.append("range_result=None")

        y0 = 28
        for i, line in enumerate(lines):
            cv2.putText(display, line, (18, y0 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        return display

    def save_red_range_snapshot(self, trigger="threshold", force=False):
        if self.latest_frame is None or self.latest_red_bbox is None:
            return None

        if self.red_pixels < MIN_RED_PIXELS and not force:
            return None

        now = time.time()
        if not force:
            if self.evidence_snapshot_count >= EVIDENCE_MAX_AUTO_SNAPSHOTS:
                return self.latest_range_result
            if now - self.evidence_last_snapshot_time < EVIDENCE_SNAPSHOT_COOLDOWN:
                return self.latest_range_result

        result = self.estimate_red_cube_position()
        self.latest_range_result = result

        if result is None:
            return None

        self.evidence_snapshot_count += 1
        self.evidence_last_snapshot_time = now

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((now - int(now)) * 1000)
        image_name = f"snapshot_{self.evidence_snapshot_count:03d}_{timestamp}_{millis:03d}_{trigger}.jpg"
        image_path = os.path.join(self.evidence_run_dir, image_name)

        display = self.draw_red_range_overlay(self.latest_frame, result, trigger)
        cv2.imwrite(image_path, display)

        bx, by, bw, bh = result["bbox"]
        row = [
            self.evidence_snapshot_count,
            f"{now:.3f}",
            trigger,
            self.state,
            f"{self.x:.4f}",
            f"{self.y:.4f}",
            f"{math.degrees(self.yaw):.2f}",
            self.red_pixels,
            bx, by, bw, bh,
            f"{result['bbox_cx']:.2f}",
            f"{result['bbox_cy']:.2f}",
            result["image_size"][0],
            result["image_size"][1],
            f"{result['fx_px']:.3f}",
            f"{result['fy_px']:.3f}",
            f"{OAK_RGB_HFOV_DEG:.2f}",
            f"{OAK_RGB_VFOV_DEG:.2f}",
            f"{result['z_from_width']:.4f}",
            f"{result['z_from_height']:.4f}",
            f"{result['cube_robot_x']:.4f}",
            f"{result['cube_robot_y']:.4f}",
            f"{result['distance_robot']:.4f}",
            f"{result['cube_global_x']:.4f}",
            f"{result['cube_global_y']:.4f}",
            f"{math.degrees(result['bearing_rad']):.2f}",
            self.latest_red_hsv_stats.get("sample_mode", "") if self.latest_red_hsv_stats else "",
            self.latest_red_hsv_stats.get("sample_count", 0) if self.latest_red_hsv_stats else 0,
            f"{self.latest_red_hsv_stats['center_hsv'][0]:.2f}" if self.latest_red_hsv_stats else "",
            f"{self.latest_red_hsv_stats['center_hsv'][1]:.2f}" if self.latest_red_hsv_stats else "",
            f"{self.latest_red_hsv_stats['center_hsv'][2]:.2f}" if self.latest_red_hsv_stats else "",
            f"{self.latest_red_hsv_stats['mean_hsv'][0]:.2f}" if self.latest_red_hsv_stats else "",
            f"{self.latest_red_hsv_stats['mean_hsv'][1]:.2f}" if self.latest_red_hsv_stats else "",
            f"{self.latest_red_hsv_stats['mean_hsv'][2]:.2f}" if self.latest_red_hsv_stats else "",
            f"{self.latest_red_hsv_stats['median_hsv'][0]:.2f}" if self.latest_red_hsv_stats else "",
            f"{self.latest_red_hsv_stats['median_hsv'][1]:.2f}" if self.latest_red_hsv_stats else "",
            f"{self.latest_red_hsv_stats['median_hsv'][2]:.2f}" if self.latest_red_hsv_stats else "",
            image_path,
        ]

        with open(self.evidence_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.get_logger().info(
            f"RED_RANGE snapshot saved | trigger={trigger} | "
            f"robot=({self.x:.2f},{self.y:.2f}) | "
            f"cube_robot=({result['cube_robot_x']:.2f},{result['cube_robot_y']:.2f})m | "
            f"dist={result['distance_robot']:.2f}m | image={image_path}"
        )

        return result

    # =========================
    # Debug 事件记录
    # =========================

    def init_debug_log_file(self):
        if not ENABLE_DEBUG_LOG or self.debug_csv_path is None:
            return

        header = [
            "event_id", "wall_time", "trigger_type", "reason", "state",
            "x", "y", "yaw_deg", "score_abs_xy", "best_score_abs_xy",
            "front_min", "front_left_min", "front_right_min", "left_min", "right_min",
            "front_pattern", "front_left_pattern", "front_right_pattern",
            "near_ratio", "cluster_deg", "physical_width", "min_dist", "edge_count",
            "both_edges", "left_recovery", "right_recovery", "outside_recovery",
            "red_pixels", "target_locked", "image_path",
        ]

        with open(self.debug_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def save_debug_event(self, trigger_type, lidar, reason):
        if not ENABLE_DEBUG_LOG:
            return

        now = time.time()
        last_time = self.debug_last_save_time.get(trigger_type, 0.0)

        if now - last_time < DEBUG_EVENT_COOLDOWN:
            return

        self.debug_last_save_time[trigger_type] = now
        self.debug_event_id += 1

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_name = f"event_{self.debug_event_id:04d}_{timestamp}_{trigger_type}.jpg"
        image_path = os.path.join(self.debug_run_dir, image_name)

        front_info = lidar.get("front_info", {})

        if self.latest_frame is not None:
            display = self.latest_frame.copy()
        else:
            display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(display, "NO CAMERA FRAME", (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if self.latest_red_bbox is not None:
            bx, by, bw, bh = self.latest_red_bbox
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)

        overlay_lines = [
            f"DEBUG {trigger_type}",
            f"reason={reason}",
            f"state={self.state}",
            f"odom=({self.x:.2f},{self.y:.2f}) yaw={math.degrees(self.yaw):.1f}",
            f"front={lidar.get('front_min', float('inf')):.2f} "
            f"FL={lidar.get('front_left_min', float('inf')):.2f} "
            f"FR={lidar.get('front_right_min', float('inf')):.2f}",
            f"pattern={lidar.get('front_pattern', 'NA')} "
            f"near={front_info.get('near_ratio', 0.0):.2f} "
            f"deg={front_info.get('cluster_deg', 0.0):.1f} "
            f"width={front_info.get('physical_width', 0.0):.2f}",
        ]

        y0 = 30
        for i, line in enumerate(overlay_lines):
            cv2.putText(display, line, (20, y0 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imwrite(image_path, display)

        score = abs(self.x) + abs(self.y)

        row = [
            self.debug_event_id,
            f"{now:.3f}",
            trigger_type,
            reason,
            self.state,
            f"{self.x:.4f}",
            f"{self.y:.4f}",
            f"{math.degrees(self.yaw):.2f}",
            f"{score:.4f}",
            f"{self.best_score:.4f}",
            f"{lidar.get('front_min', float('inf')):.4f}",
            f"{lidar.get('front_left_min', float('inf')):.4f}",
            f"{lidar.get('front_right_min', float('inf')):.4f}",
            f"{lidar.get('left_min', float('inf')):.4f}",
            f"{lidar.get('right_min', float('inf')):.4f}",
            lidar.get("front_pattern", "NA"),
            lidar.get("front_left_pattern", "NA"),
            lidar.get("front_right_pattern", "NA"),
            f"{front_info.get('near_ratio', 0.0):.4f}",
            f"{front_info.get('cluster_deg', 0.0):.4f}",
            f"{front_info.get('physical_width', 0.0):.4f}",
            f"{front_info.get('min_dist', 0.0):.4f}",
            front_info.get("edge_count", 0),
            front_info.get("both_edges", False),
            f"{front_info.get('left_recovery', 0.0):.4f}",
            f"{front_info.get('right_recovery', 0.0):.4f}",
            f"{front_info.get('outside_recovery', 0.0):.4f}",
            self.red_pixels,
            self.target_locked,
            image_path,
        ]

        with open(self.debug_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        self.get_logger().warn(
            f"DEBUG_EVENT {trigger_type} saved | reason={reason} | "
            f"event_id={self.debug_event_id} | image={image_path}"
        )

    # =========================
    # 红色目标定位
    # =========================

    def find_largest_red_bbox(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < MIN_RED_PIXELS:
            return None

        x, y, w, h = cv2.boundingRect(largest)

        if w < MIN_BOX_WIDTH_PX or h < MIN_BOX_HEIGHT_PX:
            return None

        return x, y, w, h

    def estimate_red_cube_position(self):
        if self.latest_frame is None or self.latest_red_bbox is None:
            return None

        img_h, img_w = self.latest_frame.shape[:2]
        bx, by, bw, bh = self.latest_red_bbox

        if bw <= 0 or bh <= 0:
            return None

        fx, fy = self.get_rgb_intrinsics_from_image_size(img_w, img_h)

        z_from_width = (RED_CUBE_SIZE_M * fx) / float(bw)
        z_from_height = (RED_CUBE_SIZE_M * fy) / float(bh)

        distance_forward = 0.5 * (z_from_width + z_from_height)

        box_cx = bx + bw / 2.0
        box_cy = by + bh / 2.0
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0

        pixel_offset_x = box_cx - img_cx
        pixel_offset_y = box_cy - img_cy

        cube_robot_x = pixel_offset_x * distance_forward / fx
        cube_robot_y = distance_forward
        cube_robot_z_camera = -pixel_offset_y * distance_forward / fy

        distance_robot = math.hypot(cube_robot_x, cube_robot_y)

        bearing_rad = math.atan2(-cube_robot_x, cube_robot_y)

        gx = self.x + cube_robot_x * math.sin(self.yaw) + cube_robot_y * math.cos(self.yaw)
        gy = self.y - cube_robot_x * math.cos(self.yaw) + cube_robot_y * math.sin(self.yaw)

        self.cube_robot_x = cube_robot_x
        self.cube_robot_y = cube_robot_y
        self.cube_global_x = gx
        self.cube_global_y = gy
        self.cube_distance = distance_robot

        return {
            "pixel_size": max(bw, bh),
            "bbox": (bx, by, bw, bh),
            "bbox_cx": box_cx,
            "bbox_cy": box_cy,
            "image_size": (img_w, img_h),
            "fx_px": fx,
            "fy_px": fy,
            "z_from_width": z_from_width,
            "z_from_height": z_from_height,
            "distance_forward": distance_forward,
            "distance_robot": distance_robot,
            "bearing_rad": bearing_rad,
            "cube_robot_x": cube_robot_x,
            "cube_robot_y": cube_robot_y,
            "cube_robot_z_camera": cube_robot_z_camera,
            "cube_global_x": gx,
            "cube_global_y": gy,
            "distance": distance_robot,
        }

    def save_detection_evidence(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        img_path = os.path.join(self.evidence_run_dir, f"final_red_detection_{timestamp}.jpg")
        txt_path = os.path.join(self.evidence_run_dir, f"final_red_detection_{timestamp}.txt")

        result = self.save_red_range_snapshot(trigger="final_detection", force=True)
        if result is None:
            result = self.estimate_red_cube_position()

        if self.latest_frame is not None:
            display = self.latest_frame.copy()

            if self.latest_red_bbox is not None:
                bx, by, bw, bh = self.latest_red_bbox
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.circle(display, (bx + bw // 2, by + bh // 2), 5, (255, 0, 0), -1)

            cv2.putText(display, f"robot=({self.x:.2f},{self.y:.2f})", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if result is not None:
                cv2.putText(
                    display,
                    f"cube_robot=(x_right={result['cube_robot_x']:.2f}, y_forward={result['cube_robot_y']:.2f})m",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )
                cv2.putText(
                    display,
                    f"cube_global=({result['cube_global_x']:.2f},{result['cube_global_y']:.2f})m",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )

            cv2.imwrite(img_path, display)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("TB4 Phase 2 Red Cube Detection Evidence\n")
            f.write("======================================\n\n")
            f.write(f"timestamp: {timestamp}\n\n")

            f.write("Robot coordinate system:\n")
            f.write("  origin: start point = (0, 0)\n")
            f.write("  x axis: positive to the robot's initial right side\n")
            f.write("  y axis: positive to the robot's initial forward direction\n")
            f.write("  start heading: +Y direction\n\n")

            f.write("Robot position at detection:\n")
            f.write(f"  robot_x: {self.x:.4f} m\n")
            f.write(f"  robot_y: {self.y:.4f} m\n")
            f.write(f"  robot_yaw: {math.degrees(self.yaw):.2f} deg\n\n")

            f.write("Raw odometry at detection:\n")
            f.write(f"  raw_x: {self.x_raw:.4f} m\n")
            f.write(f"  raw_y: {self.y_raw:.4f} m\n")
            f.write(f"  start_raw_x: {self.start_x_raw:.4f} m\n")
            f.write(f"  start_raw_y: {self.start_y_raw:.4f} m\n\n")

            f.write("Red detection:\n")
            f.write(f"  red_pixels: {self.red_pixels}\n")
            f.write(f"  bbox: {self.latest_red_bbox}\n")
            f.write(f"  target_locked: {self.target_locked}\n")
            f.write(f"  red_seen_count: {self.red_seen_count}\n")
            f.write(f"  real_cube_size: {RED_CUBE_SIZE_M:.4f} m\n")
            f.write(f"  oak_rgb_hfov_deg: {OAK_RGB_HFOV_DEG:.2f}\n")
            f.write(f"  oak_rgb_vfov_deg: {OAK_RGB_VFOV_DEG:.2f}\n")
            f.write(f"  use_fixed_focal_length: {USE_FIXED_FOCAL_LENGTH}\n")
            f.write(f"  fallback_focal_length_px: {FOCAL_LENGTH_PX:.2f}\n\n")

            f.write("Detected target HSV statistics:\n")
            f.write("  HSV range note: OpenCV uses H=[0,179], S=[0,255], V=[0,255]\n")
            hsv_stats = self.latest_red_hsv_stats
            if hsv_stats is None:
                f.write("  hsv_result: unavailable\n\n")
            else:
                f.write(f"  sample_mode: {hsv_stats['sample_mode']}\n")
                f.write(f"  sample_count: {hsv_stats['sample_count']} pixels\n")
                f.write(
                    f"  center_hsv: H={hsv_stats['center_hsv'][0]:.2f}, "
                    f"S={hsv_stats['center_hsv'][1]:.2f}, V={hsv_stats['center_hsv'][2]:.2f}\n"
                )
                f.write(
                    f"  mean_hsv: H={hsv_stats['mean_hsv'][0]:.2f}, "
                    f"S={hsv_stats['mean_hsv'][1]:.2f}, V={hsv_stats['mean_hsv'][2]:.2f}\n"
                )
                f.write(
                    f"  median_hsv: H={hsv_stats['median_hsv'][0]:.2f}, "
                    f"S={hsv_stats['median_hsv'][1]:.2f}, V={hsv_stats['median_hsv'][2]:.2f}\n"
                )
                f.write(
                    f"  min_hsv: H={hsv_stats['min_hsv'][0]:.2f}, "
                    f"S={hsv_stats['min_hsv'][1]:.2f}, V={hsv_stats['min_hsv'][2]:.2f}\n"
                )
                f.write(
                    f"  max_hsv: H={hsv_stats['max_hsv'][0]:.2f}, "
                    f"S={hsv_stats['max_hsv'][1]:.2f}, V={hsv_stats['max_hsv'][2]:.2f}\n"
                )
                f.write(
                    f"  std_hsv: H={hsv_stats['std_hsv'][0]:.2f}, "
                    f"S={hsv_stats['std_hsv'][1]:.2f}, V={hsv_stats['std_hsv'][2]:.2f}\n\n"
                )

            if result is None:
                f.write("Monocular result: failed\n")
            else:
                f.write("Monocular result:\n")
                f.write(f"  pixel_size: {result['pixel_size']:.2f} px\n")
                f.write(f"  image_size: {result['image_size']}\n")
                f.write(f"  fx_px: {result['fx_px']:.3f}\n")
                f.write(f"  fy_px: {result['fy_px']:.3f}\n")
                f.write(f"  z_from_width: {result['z_from_width']:.4f} m\n")
                f.write(f"  z_from_height: {result['z_from_height']:.4f} m\n")
                f.write(f"  distance_forward: {result['distance_forward']:.4f} m\n")
                f.write(f"  distance_from_robot_xy: {result['distance_robot']:.4f} m\n")
                f.write(f"  bearing: {math.degrees(result['bearing_rad']):.2f} deg\n")
                f.write(f"  cube_robot_x_right: {result['cube_robot_x']:.4f} m\n")
                f.write(f"  cube_robot_y_forward: {result['cube_robot_y']:.4f} m\n")
                f.write(f"  cube_camera_z_vertical_est: {result['cube_robot_z_camera']:.4f} m\n")
                f.write(f"  cube_global_x: {result['cube_global_x']:.4f} m\n")
                f.write(f"  cube_global_y: {result['cube_global_y']:.4f} m\n\n")

            f.write(f"final_screenshot_path: {img_path}\n")
            f.write(f"evidence_folder: {self.evidence_run_dir}\n")
            f.write(f"evidence_csv: {self.evidence_csv_path}\n")

        self.get_logger().info("========== RED TARGET DETECTED ==========")
        self.get_logger().info(
            f"Robot at detection: x={self.x:.3f}, y={self.y:.3f}, yaw={math.degrees(self.yaw):.1f} deg"
        )

        if result is not None:
            self.get_logger().info(
                f"Cube robot coordinate: x_right={result['cube_robot_x']:.3f}, "
                f"y_forward={result['cube_robot_y']:.3f}, distance={result['distance']:.3f}"
            )
            self.get_logger().info(
                f"Cube global coordinate: x={result['cube_global_x']:.3f}, "
                f"y={result['cube_global_y']:.3f}"
            )
        else:
            self.get_logger().warn("Monocular cube localization failed.")

        self.get_logger().info(f"Screenshot saved: {img_path}")
        self.get_logger().info(f"Coordinate file saved: {txt_path}")
        self.get_logger().info("=========================================")

    # =========================
    # LiDAR 工具
    # =========================

    def get_arc_min(self, center_angle, arc_deg):
        if self.latest_scan is None:
            return float("inf")

        msg = self.latest_scan
        arc = extract_arc(list(msg.ranges), msg.angle_min, msg.angle_increment,
                          center_angle, arc_deg)

        arc = np.array(arc, dtype=float)
        valid = np.isfinite(arc)
        if not np.any(valid):
            return float("inf")

        return float(np.min(arc[valid]))

    def get_arc_median(self, center_angle, arc_deg, max_valid=PATTERN_MAX_VALID):
        if self.latest_scan is None:
            return float("inf")

        msg = self.latest_scan
        arc = extract_arc(list(msg.ranges), msg.angle_min, msg.angle_increment,
                          center_angle, arc_deg)

        arc = np.array(arc, dtype=float)
        valid = np.isfinite(arc) & (arc > 0.02) & (arc < max_valid)
        if not np.any(valid):
            return float("inf")

        return float(np.median(arc[valid]))

    def get_scan_change_summary(self):
        if self.latest_scan is None or self.curr_scan_ranges is None or self.prev_scan_ranges is None:
            return {
                "available": False, "beam_count": 0, "changed_count": 0,
                "changed_ratio": 0.0, "front_changed_ratio": 0.0,
                "front_closer": False, "front_opened": False,
                "clusters": [], "front_clusters": [],
            }

        msg = self.latest_scan
        n = min(len(self.curr_scan_ranges), len(self.prev_scan_ranges))
        if n == 0:
            return {
                "available": False, "beam_count": 0, "changed_count": 0,
                "changed_ratio": 0.0, "front_changed_ratio": 0.0,
                "front_closer": False, "front_opened": False,
                "clusters": [], "front_clusters": [],
            }

        curr_raw = self.curr_scan_ranges[:n]
        prev_raw = self.prev_scan_ranges[:n]

        curr, curr_valid = clean_ranges(curr_raw, CHANGE_MAX_VALID)
        prev, prev_valid = clean_ranges(prev_raw, CHANGE_MAX_VALID)

        valid = curr_valid | prev_valid
        delta = curr - prev
        changed_mask = (np.abs(delta) >= CHANGE_DIST_TH) & valid

        clusters_idx = find_true_clusters_circular(changed_mask)
        beam_deg = abs(math.degrees(msg.angle_increment))

        clusters = []
        front_clusters = []
        front_changed_count = 0
        front_total_count = 0
        front_closer = False
        front_opened = False

        for i in range(n):
            lidar_angle = msg.angle_min + i * msg.angle_increment
            robot_angle = normalize_angle(lidar_angle - FRONT_ANGLE)
            if abs(math.degrees(robot_angle)) <= CHANGE_FRONT_ROI_DEG:
                front_total_count += 1
                if changed_mask[i]:
                    front_changed_count += 1
                    if delta[i] < -CHANGE_DIST_TH and curr[i] < CHANGE_NEAR_DIST:
                        front_closer = True
                    if delta[i] > CHANGE_DIST_TH:
                        front_opened = True

        for idxs in clusters_idx:
            if not idxs:
                continue

            cluster_deg = len(idxs) * beam_deg
            if cluster_deg < CHANGE_MIN_CLUSTER_DEG:
                continue

            angles = []
            robot_angles = []
            for idx in idxs:
                lidar_angle = msg.angle_min + idx * msg.angle_increment
                angles.append(lidar_angle)
                robot_angles.append(normalize_angle(lidar_angle - FRONT_ANGLE))

            sin_sum = float(np.sum(np.sin(robot_angles)))
            cos_sum = float(np.sum(np.cos(robot_angles)))
            center_robot_angle = math.atan2(sin_sum, cos_sum)
            center_robot_deg = math.degrees(center_robot_angle)

            curr_vals = curr[idxs]
            delta_vals = delta[idxs]

            cluster = {
                "size": len(idxs),
                "deg": float(cluster_deg),
                "center_robot_deg": float(center_robot_deg),
                "min_curr": float(np.min(curr_vals)),
                "median_curr": float(np.median(curr_vals)),
                "mean_delta": float(np.mean(delta_vals)),
                "closer": bool(np.mean(delta_vals) < -CHANGE_DIST_TH),
                "opened": bool(np.mean(delta_vals) > CHANGE_DIST_TH),
            }
            clusters.append(cluster)

            if abs(center_robot_deg) <= CHANGE_FRONT_ROI_DEG:
                front_clusters.append(cluster)

        changed_count = int(np.sum(changed_mask))
        changed_ratio = float(changed_count / max(n, 1))
        front_changed_ratio = float(front_changed_count / max(front_total_count, 1))

        return {
            "available": True,
            "beam_count": int(n),
            "changed_count": changed_count,
            "changed_ratio": changed_ratio,
            "front_changed_ratio": front_changed_ratio,
            "front_closer": bool(front_closer),
            "front_opened": bool(front_opened),
            "clusters": clusters,
            "front_clusters": front_clusters,
        }

    def classify_arc_pattern(self, center_angle, arc_deg=60.0):
        if self.latest_scan is None:
            return "no_scan", {}

        msg = self.latest_scan

        raw_arc = extract_arc(list(msg.ranges), msg.angle_min, msg.angle_increment,
                              center_angle, arc_deg)

        arc, valid = clean_ranges(raw_arc)
        if len(arc) == 0:
            return "clear", {}

        near_mask = arc < PATTERN_NEAR_DIST
        near_ratio = float(np.mean(near_mask))
        clusters = find_near_clusters(near_mask)

        info = {
            "near_ratio": near_ratio,
            "cluster_deg": 0.0,
            "physical_width": 0.0,
            "edge_count": 0,
            "left_edge": False,
            "right_edge": False,
            "both_edges": False,
            "left_recovery": 0.0,
            "right_recovery": 0.0,
            "outside_recovery": 0.0,
            "min_dist": float(np.min(arc)),
            "valid_ratio": float(np.mean(valid)) if len(valid) > 0 else 0.0,
        }

        if not clusters:
            return "clear", info

        largest_cluster = max(clusters, key=lambda c: c[1] - c[0] + 1)
        c0, c1 = largest_cluster
        cluster_len = c1 - c0 + 1

        beam_deg = abs(math.degrees(msg.angle_increment))
        cluster_deg = cluster_len * beam_deg

        cluster_arc = arc[c0:c1 + 1]
        min_dist = float(np.min(cluster_arc))
        cluster_mean = float(np.mean(cluster_arc))
        physical_width = 2.0 * min_dist * math.tan(math.radians(cluster_deg) / 2.0)

        diff = np.diff(arc)
        abs_diff = np.abs(diff)
        edge_count = int(np.sum(abs_diff > PATTERN_EDGE_JUMP))

        left_edge = False
        right_edge = False
        if c0 > 1:
            left_edge = abs(arc[c0] - arc[c0 - 1]) > PATTERN_EDGE_JUMP
        if c1 < len(arc) - 2:
            right_edge = abs(arc[c1] - arc[c1 + 1]) > PATTERN_EDGE_JUMP

        outside_window = max(3, int(round(8.0 / max(beam_deg, 1e-6))))

        left_recovery = 0.0
        right_recovery = 0.0

        if c0 > 1:
            l0 = max(0, c0 - outside_window)
            left_outside = arc[l0:c0]
            if len(left_outside) > 0:
                left_recovery = float(np.median(left_outside) - cluster_mean)

        if c1 < len(arc) - 2:
            r1 = min(len(arc), c1 + 1 + outside_window)
            right_outside = arc[c1 + 1:r1]
            if len(right_outside) > 0:
                right_recovery = float(np.median(right_outside) - cluster_mean)

        outside_recovery = min(left_recovery, right_recovery)
        both_edges = left_edge and right_edge

        info.update({
            "cluster_deg": float(cluster_deg),
            "physical_width": float(physical_width),
            "edge_count": edge_count,
            "left_edge": bool(left_edge),
            "right_edge": bool(right_edge),
            "both_edges": bool(both_edges),
            "left_recovery": float(left_recovery),
            "right_recovery": float(right_recovery),
            "outside_recovery": float(outside_recovery),
            "min_dist": float(min_dist),
        })

        obstacle_like = (
            OBSTACLE_MIN_DEG <= cluster_deg <= 42.0
            and both_edges
            and physical_width <= max(OBSTACLE_MAX_WIDTH, 0.65)
            and outside_recovery > 0.18
        )

        strong_valley_obstacle = (
            both_edges
            and cluster_deg <= 48.0
            and outside_recovery > 0.28
            and min_dist < PATTERN_NEAR_DIST
        )

        if obstacle_like or strong_valley_obstacle:
            return "obstacle", info

        if near_ratio > WALL_RATIO_TH and cluster_deg > WALL_CLUSTER_DEG:
            return "wall", info

        if min_dist < FRONT_STOP_DIST and near_ratio > 0.35:
            return "corner_or_wall", info

        return "unknown_near_object", info

    def get_ordered_arc_with_angles(self, center_angle, arc_deg, max_valid=RIGHT_SHAPE_MAX_VALID):
        if self.latest_scan is None:
            return None, None, None

        msg = self.latest_scan
        ranges = list(msg.ranges)
        n = len(ranges)
        if n == 0:
            return None, None, None

        half = math.radians(arc_deg / 2.0)
        start_angle = center_angle - half
        end_angle = center_angle + half

        start_idx = angle_to_index(start_angle, msg.angle_min, msg.angle_increment, n)
        end_idx = angle_to_index(end_angle, msg.angle_min, msg.angle_increment, n)

        if start_idx <= end_idx:
            idxs = list(range(start_idx, end_idx + 1))
        else:
            idxs = list(range(start_idx, n)) + list(range(0, end_idx + 1))

        raw = np.array([ranges[i] for i in idxs], dtype=float)
        valid = np.isfinite(raw) & (raw > 0.02) & (raw < max_valid)

        cleaned = raw.copy()
        cleaned[~np.isfinite(cleaned)] = max_valid
        cleaned = np.clip(cleaned, 0.0, max_valid)

        robot_angles = []
        for i in idxs:
            lidar_angle = msg.angle_min + i * msg.angle_increment
            robot_angles.append(normalize_angle(lidar_angle - FRONT_ANGLE))

        return cleaned, valid, np.array(robot_angles, dtype=float)

    def smooth_1d(self, values, window=RIGHT_SHAPE_SMOOTH_WINDOW):
        values = np.array(values, dtype=float)
        if len(values) == 0:
            return values

        window = int(max(1, window))
        if window <= 1 or len(values) < window:
            return values.copy()

        if window % 2 == 0:
            window += 1

        pad = window // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(padded, kernel, mode="valid")

    def analyze_right_wall_shape(self):
        center_angle = FRONT_ANGLE + math.radians(WALL_SIGN * RIGHT_SHAPE_CENTER_DEG)
        arc, valid, robot_angles = self.get_ordered_arc_with_angles(
            center_angle, RIGHT_SHAPE_ARC_DEG, RIGHT_SHAPE_MAX_VALID
        )

        base = {
            "available": False, "shape_ok": False, "parallel_good": False,
            "right_distance": float("inf"), "right_front": float("inf"),
            "right_mid": float("inf"), "right_back": float("inf"),
            "right_min": float("inf"), "parallel_error": 0.0,
            "valley_index": -1, "valley_angle_deg": 0.0, "valley_offset": 0.0,
            "valid_ratio": 0.0, "valley_depth": 0.0, "valley_width_deg": 0.0,
            "mono_ratio": 0.0, "jump_ratio": 1.0,
            "left_mono_ratio": 0.0, "right_mono_ratio": 0.0,
        }

        if arc is None or valid is None or len(arc) < 12:
            return base

        n = len(arc)
        beam_deg = RIGHT_SHAPE_ARC_DEG / max(n - 1, 1)

        valid_ratio = float(np.mean(valid)) if len(valid) > 0 else 0.0
        smooth = self.smooth_1d(arc, RIGHT_SHAPE_SMOOTH_WINDOW)

        def sample_at_robot_deg(deg, half_width_deg=6.0):
            mask = np.abs(np.degrees(robot_angles) - deg) <= half_width_deg
            mask = mask & valid
            if not np.any(mask):
                return float("inf")
            return float(np.median(arc[mask]))

        right_back = sample_at_robot_deg(WALL_SIGN * -120.0)
        right_mid = sample_at_robot_deg(WALL_SIGN * -90.0)
        right_front = sample_at_robot_deg(WALL_SIGN * -60.0)

        if valid_ratio < RIGHT_SHAPE_MIN_VALID_RATIO:
            base.update({
                "available": True, "valid_ratio": valid_ratio,
                "right_front": right_front, "right_mid": right_mid,
                "right_back": right_back, "right_min": float(np.min(arc)),
            })
            return base

        valid_indices = np.where(valid)[0]
        valid_smooth = smooth[valid_indices]
        min_local = int(np.argmin(valid_smooth))
        min_idx = int(valid_indices[min_local])
        min_dist = float(smooth[min_idx])

        center_idx = (n - 1) / 2.0
        valley_offset = float((min_idx - center_idx) / max(center_idx, 1.0))
        valley_angle_deg = float(math.degrees(robot_angles[min_idx]))

        edge_n = max(5, int(0.12 * n))
        back_edge_vals = smooth[:edge_n][valid[:edge_n]]
        front_edge_vals = smooth[-edge_n:][valid[-edge_n:]]

        if len(back_edge_vals) == 0:
            back_edge = float(np.max(smooth[:edge_n]))
        else:
            back_edge = float(np.median(back_edge_vals))

        if len(front_edge_vals) == 0:
            front_edge = float(np.max(smooth[-edge_n:]))
        else:
            front_edge = float(np.median(front_edge_vals))

        valley_depth = float(min(back_edge, front_edge) - min_dist)

        valley_mask = (smooth <= min_dist + 0.14) & valid
        clusters = find_near_clusters(list(valley_mask))
        valley_width_deg = 0.0
        for c0, c1 in clusters:
            if c0 <= min_idx <= c1:
                valley_width_deg = float((c1 - c0 + 1) * beam_deg)
                break

        diff = np.diff(smooth)

        left_diff = diff[:max(min_idx, 1)]
        right_diff = diff[min_idx:]

        if len(left_diff) > 0:
            left_mono_ratio = float(np.mean(left_diff <= 0.035))
        else:
            left_mono_ratio = 0.0

        if len(right_diff) > 0:
            right_mono_ratio = float(np.mean(right_diff >= -0.035))
        else:
            right_mono_ratio = 0.0

        mono_ratio = min(left_mono_ratio, right_mono_ratio)

        jump_ratio = float(np.mean(np.abs(diff) > RIGHT_SHAPE_MAX_JUMP)) if len(diff) > 0 else 1.0

        shape_ok = (
            min_dist > ROBOT_RADIUS + 0.04
            and min_dist < PROTRUSION_EDGE_LOST_DIST
            and valley_depth >= RIGHT_SHAPE_VALLEY_MIN_DEPTH
            and valley_width_deg >= RIGHT_SHAPE_MIN_VALLEY_WIDTH_DEG
            and mono_ratio >= RIGHT_SHAPE_MONO_RATIO_TH
            and jump_ratio <= RIGHT_SHAPE_MAX_JUMP_RATIO
        )

        parallel_good = (
            shape_ok
            and abs(valley_offset) <= RIGHT_SHAPE_CENTER_TOL
            and MIN_RIGHT_DIST <= min_dist <= MAX_RIGHT_DIST
        )

        three_point_hint = 0.0
        if np.isfinite(right_back) and np.isfinite(right_front):
            three_point_hint = clamp(right_back - right_front, -0.35, 0.35)

        parallel_error = clamp(
            WALL_SIGN * (RIGHT_SHAPE_ERROR_SCALE * valley_offset + 0.25 * three_point_hint),
            -RIGHT_PARALLEL_TOL * 2.0,
            RIGHT_PARALLEL_TOL * 2.0
        )

        return {
            "available": True,
            "shape_ok": bool(shape_ok),
            "parallel_good": bool(parallel_good),
            "right_distance": float(min_dist),
            "right_front": float(right_front),
            "right_mid": float(right_mid),
            "right_back": float(right_back),
            "right_min": float(min_dist),
            "parallel_error": float(parallel_error),
            "valley_index": int(min_idx),
            "valley_angle_deg": valley_angle_deg,
            "valley_offset": float(valley_offset),
            "valid_ratio": valid_ratio,
            "valley_depth": float(valley_depth),
            "valley_width_deg": float(valley_width_deg),
            "mono_ratio": float(mono_ratio),
            "jump_ratio": float(jump_ratio),
            "left_mono_ratio": float(left_mono_ratio),
            "right_mono_ratio": float(right_mono_ratio),
        }

    def get_right_wall_geometry(self, lidar=None):
        geom = self.analyze_right_wall_shape()

        if lidar is not None:
            geom["right_min"] = float(min(lidar.get("right_min", geom["right_min"]), geom["right_min"]))

        return geom

    def is_right_wall_parallel_good(self, lidar):
        geom = self.get_right_wall_geometry(lidar)

        return (
            geom.get("parallel_good", False)
            and lidar["front_min"] > FRONT_STOP_DIST
        ), geom

    def is_right_wall_visible(self, lidar, require_distance_ok=True):
        geom = self.get_right_wall_geometry(lidar)

        right_distance = geom.get("right_distance", float("inf"))
        right_min = min(
            float(lidar.get("right_min", float("inf"))),
            float(geom.get("right_min", right_distance))
        )

        valid_ratio = geom.get("valid_ratio", 0.0)
        valley_width = geom.get("valley_width_deg", 0.0)
        jump_ratio = geom.get("jump_ratio", 1.0)
        mono_ratio = geom.get("mono_ratio", 0.0)
        shape_ok = geom.get("shape_ok", False)

        loose_continuous_wall = (
            geom.get("available", False)
            and valid_ratio >= RIGHT_WALL_VISIBLE_MIN_VALID_RATIO
            and valley_width >= RIGHT_WALL_VISIBLE_MIN_VALLEY_WIDTH_DEG
            and jump_ratio <= RIGHT_WALL_VISIBLE_MAX_JUMP_RATIO
            and mono_ratio >= 0.45
            and np.isfinite(right_distance)
            and right_distance < RIGHT_WALL_VISIBLE_MAX_DIST
        )

        distance_ok = (
            np.isfinite(right_min)
            and MIN_RIGHT_DIST <= right_min <= RIGHT_WALL_VISIBLE_MAX_DIST
        )

        visible = bool(shape_ok or loose_continuous_wall)
        if require_distance_ok:
            visible = visible and distance_ok

        geom["visible"] = visible
        geom["visible_distance_ok"] = bool(distance_ok)
        geom["loose_continuous_wall"] = bool(loose_continuous_wall)

        return visible, geom

    def compute_right_wall_follow_cmd(self, lidar, base_speed=FORWARD_SPEED):
        front_min = lidar["front_min"]
        front_left_min = lidar["front_left_min"]
        front_right_min = lidar["front_right_min"]
        geom = self.get_right_wall_geometry(lidar)

        right_distance = geom.get("right_distance", float("inf"))
        right_min = geom.get("right_min", right_distance)
        parallel_error = geom.get("parallel_error", 0.0)
        shape_ok = geom.get("shape_ok", False)
        available = geom.get("available", False)

        if front_min < FRONT_STOP_DIST:
            return 0.0, WALL_SIGN * SMALL_TURN_SPEED, geom

        if front_min < FRONT_WARN_DIST:
            if front_left_min >= front_right_min:
                return SLOW_FORWARD_SPEED, WALL_SIGN * SMALL_TURN_SPEED, geom
            return SLOW_FORWARD_SPEED, -WALL_SIGN * SMALL_TURN_SPEED, geom

        if (not available) or (not np.isfinite(right_distance)) or right_distance > PROTRUSION_EDGE_LOST_DIST:
            return SLOW_FORWARD_SPEED, -WALL_SIGN * SMALL_TURN_SPEED, geom

        if right_min < MIN_RIGHT_DIST:
            return SLOW_FORWARD_SPEED, WALL_SIGN * SMALL_TURN_SPEED, geom

        if right_distance > MAX_RIGHT_DIST:
            return SLOW_FORWARD_SPEED, -WALL_SIGN * SMALL_TURN_SPEED, geom

        dist_error = TARGET_RIGHT_DIST - right_distance
        angular = WALL_SIGN * (RIGHT_DIST_K * dist_error) + RIGHT_PARALLEL_K * parallel_error
        angular = clamp(angular, -SMALL_TURN_SPEED, SMALL_TURN_SPEED)

        linear = base_speed
        if (not shape_ok) or abs(parallel_error) > RIGHT_PARALLEL_TOL:
            linear = min(linear, SLOW_FORWARD_SPEED)

        return linear, angular, geom

    def front_wall_reappeared(self, lidar):
        front_info = lidar.get("front_info", {})
        front_pattern = lidar.get("front_pattern", "clear")

        large_front_wall = (
            front_pattern in ["wall", "corner_or_wall"]
            or (
                front_info.get("near_ratio", 0.0) >= PROTRUSION_REJOIN_FRONT_WALL_RATIO
                and front_info.get("cluster_deg", 0.0) >= WALL_CLUSTER_DEG
            )
        )

        side_wall_hint = (
            lidar.get("front_left_pattern") in ["wall", "corner_or_wall"]
            or lidar.get("front_right_pattern") in ["wall", "corner_or_wall"]
        )

        return bool(large_front_wall or side_wall_hint)

    # =========================
    # 【方案2】结构化场景解析 analyze_scene
    # =========================

    def _scene_polar_to_points(self):
        if self.latest_scan is None:
            return None, None, None, None

        msg = self.latest_scan
        ranges = np.array(msg.ranges, dtype=float)
        n = len(ranges)
        if n == 0:
            return None, None, None, None

        idx = np.arange(n)
        lidar_angles = msg.angle_min + idx * msg.angle_increment
        robot_angles = np.array(
            [normalize_angle(a - FRONT_ANGLE) for a in lidar_angles],
            dtype=float
        )

        valid = (
            np.isfinite(ranges)
            & (ranges >= SCENE_MIN_VALID_M)
            & (ranges <= SCENE_MAX_VALID_M)
        )

        safe_ranges = np.where(valid, ranges, 0.0)
        x = safe_ranges * np.cos(robot_angles)
        y = safe_ranges * np.sin(robot_angles)
        pts = np.stack([x, y], axis=1)

        return pts, robot_angles, ranges, valid

    def _scene_segment(self, pts, ranges, valid):
        n = len(ranges)
        if n == 0:
            return []

        if np.all(valid):
            start = 0
        else:
            start = int(np.where(~valid)[0][0])

        segments = []
        current = []
        prev_idx = None

        for k in range(n):
            i = (start + k) % n
            if not valid[i]:
                if current:
                    segments.append(current)
                    current = []
                prev_idx = None
                continue

            if prev_idx is None:
                current = [i]
                prev_idx = i
                continue

            d = math.hypot(pts[i, 0] - pts[prev_idx, 0], pts[i, 1] - pts[prev_idx, 1])
            thr = ranges[i] * SCENE_BREAK_K + SCENE_BREAK_C
            if d > thr:
                if current:
                    segments.append(current)
                current = [i]
            else:
                current.append(i)
            prev_idx = i

        if current:
            segments.append(current)

        return segments

    def _scene_iepf_split(self, seg, pts):
        if not SCENE_IEPF_ENABLE or len(seg) < 2 * SCENE_MIN_SEG_POINTS:
            return [seg]

        p0 = pts[seg[0]]
        p1 = pts[seg[-1]]
        d = p1 - p0
        L = math.hypot(d[0], d[1])
        if L < 1e-6:
            return [seg]

        nx, ny = -d[1] / L, d[0] / L
        max_dist = -1.0
        max_k = -1
        for k in range(1, len(seg) - 1):
            p = pts[seg[k]]
            dist = abs((p[0] - p0[0]) * nx + (p[1] - p0[1]) * ny)
            if dist > max_dist:
                max_dist = dist
                max_k = k

        if max_dist > SCENE_IEPF_SPLIT_DIST and max_k > 0:
            left = self._scene_iepf_split(seg[:max_k + 1], pts)
            right = self._scene_iepf_split(seg[max_k:], pts)
            return left + right

        return [seg]

    def _fit_line(self, P):
        c = P.mean(axis=0)
        Q = P - c
        cov = (Q.T @ Q) / max(len(P), 1)
        evals, evecs = np.linalg.eigh(cov)
        direction = evecs[:, -1]
        normal = evecs[:, 0]
        residuals = Q @ normal
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        heading = math.atan2(direction[1], direction[0])
        perp_dist = abs(float(c @ normal))
        return rms, heading, perp_dist, c

    def _fit_circle(self, P):
        if len(P) < 3:
            return float("inf"), float("inf"), None

        x = P[:, 0]
        y = P[:, 1]
        A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)
        b = x ** 2 + y ** 2
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return float("inf"), float("inf"), None

        cx, cy, c = sol
        r2 = c + cx ** 2 + cy ** 2
        if r2 <= 0:
            return float("inf"), float("inf"), None
        R = math.sqrt(r2)

        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        rms = float(np.sqrt(np.mean((dists - R) ** 2)))
        return rms, float(R), (float(cx), float(cy))

    def _classify_segment(self, seg, pts):
        P = pts[seg]
        npts = len(P)
        centroid = P.mean(axis=0)
        bearing = math.degrees(math.atan2(centroid[1], centroid[0]))
        seg_dist = float(math.hypot(centroid[0], centroid[1]))

        if npts < SCENE_MIN_SEG_POINTS:
            return {
                "type": "noise", "bearing_deg": bearing, "distance": seg_dist,
                "point_count": npts,
            }

        line_rms, heading, perp, lc = self._fit_line(P)
        circ_rms, R, cc = self._fit_circle(P)

        is_cylinder = False
        if cc is not None and np.isfinite(R):
            radius_ok = abs(R - CYLINDER_RADIUS_M) <= CYLINDER_RADIUS_TOL_M
            center_range = math.hypot(cc[0], cc[1])
            convex_outward = center_range > seg_dist
            if (circ_rms < SCENE_CIRCLE_RES_MAX and radius_ok and convex_outward
                    and circ_rms < line_rms):
                is_cylinder = True

        if is_cylinder:
            near_dist = float(np.min(np.hypot(P[:, 0], P[:, 1])))
            return {
                "type": "cylinder", "bearing_deg": bearing, "distance": near_dist,
                "centroid_distance": seg_dist, "radius": R, "fit_residual": circ_rms,
                "center": cc, "point_count": npts,
            }

        if line_rms < SCENE_LINE_RES_MAX:
            side = "left" if centroid[1] > 0 else "right"
            heading_err = normalize_angle(heading)
            if heading_err > math.pi / 2:
                heading_err -= math.pi
            elif heading_err < -math.pi / 2:
                heading_err += math.pi
            return {
                "type": "wall", "bearing_deg": bearing, "distance": perp,
                "side": side, "heading_err_rad": float(heading_err),
                "fit_residual": line_rms, "point_count": npts,
                "centroid": (float(centroid[0]), float(centroid[1])),
            }

        return {
            "type": "noise", "bearing_deg": bearing, "distance": seg_dist,
            "point_count": npts,
        }

    def _scene_find_gaps(self, robot_angles, ranges, valid):
        n = len(ranges)
        if n == 0:
            return []

        far_mask = (~valid) | (ranges > SCENE_GAP_FAR_DIST)

        msg = self.latest_scan
        beam_deg = abs(math.degrees(msg.angle_increment)) if msg else (360.0 / n)

        clusters = find_true_clusters_circular(far_mask)
        gaps = []
        for idxs in clusters:
            if not idxs:
                continue
            width_deg = len(idxs) * beam_deg
            if width_deg < SCENE_GAP_MIN_WIDTH_DEG:
                continue
            angs = robot_angles[idxs]
            sin_s = float(np.sum(np.sin(angs)))
            cos_s = float(np.sum(np.cos(angs)))
            center_deg = math.degrees(math.atan2(sin_s, cos_s))
            seg_ranges = ranges[idxs]
            finite = seg_ranges[np.isfinite(seg_ranges)]
            depth = float(np.max(finite)) if len(finite) else SCENE_MAX_VALID_M
            gaps.append({
                "center_bearing_deg": float(center_deg),
                "width_deg": float(width_deg),
                "depth": depth,
            })
        return gaps

    def analyze_scene(self):
        pts, robot_angles, ranges, valid = self._scene_polar_to_points()
        if pts is None:
            return {"available": False, "walls": [], "cylinders": [], "gaps": [],
                    "follow_wall": None, "front_blocker": None, "free_directions": []}

        raw_segments = self._scene_segment(pts, ranges, valid)

        segments = []
        for seg in raw_segments:
            if len(seg) >= 2 * SCENE_MIN_SEG_POINTS:
                segments.extend(self._scene_iepf_split(seg, pts))
            else:
                segments.append(seg)

        walls = []
        cylinders = []
        for seg in segments:
            cls = self._classify_segment(seg, pts)
            if cls["type"] == "wall":
                walls.append(cls)
            elif cls["type"] == "cylinder":
                cylinders.append(cls)

        gaps = self._scene_find_gaps(robot_angles, ranges, valid)

        follow_side = "right" if WALL_SIGN > 0 else "left"
        follow_candidates = [w for w in walls if w.get("side") == follow_side]
        follow_wall = None
        if follow_candidates:
            follow_candidates.sort(key=lambda w: (w["distance"], -w["point_count"]))
            fw = follow_candidates[0]
            follow_wall = {
                "valid": True,
                "distance": fw["distance"],
                "heading_err_rad": fw["heading_err_rad"],
                "bearing_deg": fw["bearing_deg"],
                "point_count": fw["point_count"],
                "fit_residual": fw["fit_residual"],
            }

        front_blocker = None
        roi = SCENE_FRONT_ROI_DEG
        front_cyl = [c for c in cylinders if abs(c["bearing_deg"]) <= roi]
        if front_cyl:
            front_cyl.sort(key=lambda c: c["distance"])
            c0 = front_cyl[0]
            front_blocker = {
                "type": "cylinder", "distance": c0["distance"],
                "bearing_deg": c0["bearing_deg"], "radius": c0["radius"],
            }
        else:
            front_walls = [w for w in walls if abs(w["bearing_deg"]) <= roi]
            if front_walls:
                front_walls.sort(key=lambda w: w["distance"])
                w0 = front_walls[0]
                front_blocker = {
                    "type": "wall", "distance": w0["distance"],
                    "bearing_deg": w0["bearing_deg"],
                }

        free_directions = [g["center_bearing_deg"] for g in gaps]

        front_roi_mask = valid & (np.abs(np.degrees(robot_angles)) <= roi)
        if np.any(front_roi_mask):
            front_near = float(np.min(ranges[front_roi_mask]))
        else:
            front_near = float("inf")

        return {
            "available": True,
            "walls": walls,
            "cylinders": cylinders,
            "gaps": gaps,
            "follow_wall": follow_wall,
            "front_blocker": front_blocker,
            "free_directions": free_directions,
            "front_near": front_near,
            "beam_count": int(len(ranges)),
        }

    # =========================
    # 【方案2】gap 选择（叠加 score 增大方向偏好）
    # =========================

    def _score_best_gap(self, scene, desired_robot):
        """在所有 gap 中选出综合代价最低者。
        desired_robot: 期望前进方向（机器人坐标系，rad）。
          SEARCHING 传入 score 增大方向；RETURNING 传入指向原点的 goal_angle。
        [Priority 7] 新增“与当前跟随墙连续性”代价：惩罚偏向跟随墙一侧的 gap，
          防止绕障后切换跟随侧 / 掉头。"""
        gaps = scene.get("gaps", [])
        if not gaps:
            return None, None

        follow_wall = scene.get("follow_wall")
        follow_valid = bool(follow_wall and follow_wall.get("valid"))
        # 跟随侧在机器人坐标系下的方位符号：右跟随(WALL_SIGN>0)时墙在物理右侧=负方位。
        follow_side_sign = -WALL_SIGN

        best = None
        best_score = float("inf")
        for g in gaps:
            ang = math.radians(g["center_bearing_deg"])
            heading_pen = SCENE_GAP_HEADING_W * abs(normalize_angle(ang - desired_robot))
            forward_pen = SCENE_GAP_FORWARD_W * abs(ang)
            depth_bonus = SCENE_GAP_DEPTH_W * g["depth"]
            cost = heading_pen + forward_pen - depth_bonus

            if follow_valid:
                toward_wall = follow_side_sign * ang  # >0 表示 gap 朝跟随墙一侧
                if toward_wall > 0.0:
                    cost += SCENE_GAP_CONTINUITY_W * toward_wall

            if cost < best_score:
                best_score = cost
                best = g

        if best is None:
            return None, None
        return math.radians(best["center_bearing_deg"]), best

    def choose_gap_direction(self, scene):
        # SEARCHING：期望方向 = score 增大（远离原点）方向。
        sx = 0.0 if abs(self.x) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.x)
        sy = 0.0 if abs(self.y) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.y)
        if sx == 0.0 and sy == 0.0:
            desired_global = self.yaw
        else:
            desired_global = math.atan2(sy, sx)
        desired_robot = normalize_angle(desired_global - self.yaw)
        return self._score_best_gap(scene, desired_robot)

    def choose_gap_toward_angle(self, scene, desired_robot):
        return self._score_best_gap(scene, desired_robot)

    def choose_gap_toward_origin(self, scene):
        # RETURNING：期望方向 = 指向原点 (0,0) 的方向。
        desired_robot = self.get_goal_angle_robot(0.0, 0.0)
        return self._score_best_gap(scene, desired_robot)

    def get_lidar_summary(self):
        front = FRONT_ANGLE
        follow_side = FRONT_ANGLE - WALL_SIGN * math.radians(90)
        outer_side = FRONT_ANGLE + WALL_SIGN * math.radians(90)
        front_follow = FRONT_ANGLE - WALL_SIGN * math.radians(35)
        front_outer = FRONT_ANGLE + WALL_SIGN * math.radians(35)

        summary = {}

        summary["front_min"] = self.get_arc_min(front, 50)
        summary["front_left_min"] = self.get_arc_min(front_outer, 35)
        summary["front_right_min"] = self.get_arc_min(front_follow, 35)
        summary["left_min"] = self.get_arc_min(outer_side, 45)
        summary["right_min"] = self.get_arc_min(follow_side, 45)

        summary["front_pattern"], summary["front_info"] = self.classify_arc_pattern(front, 60)
        summary["front_left_pattern"], summary["front_left_info"] = self.classify_arc_pattern(front_outer, 40)
        summary["front_right_pattern"], summary["front_right_info"] = self.classify_arc_pattern(front_follow, 40)

        summary["scan_change"] = self.get_scan_change_summary()
        summary["right_wall_geom"] = self.get_right_wall_geometry(summary)

        return summary

    # =========================
    # 返回阶段专用 LiDAR / goal 工具
    # =========================

    def get_goal_angle_robot(self, goal_x=0.0, goal_y=0.0, use_amcl=False):
        if use_amcl and self.amcl_pose_received:
            dx = goal_x - self.amcl_x
            dy = goal_y - self.amcl_y
            goal_yaw_global = math.atan2(dy, dx)
            goal_angle_robot = normalize_angle(goal_yaw_global - self.amcl_yaw)
        else:
            dx = goal_x - self.x
            dy = goal_y - self.y
            goal_yaw_global = math.atan2(dy, dx)
            goal_angle_robot = normalize_angle(goal_yaw_global - self.yaw)

        return goal_angle_robot

    def is_robot_angle_clear(self, robot_angle, arc_deg=GOAL_ARC_DEG, safe_dist=RETURN_SAFE_DIST):
        lidar_angle = FRONT_ANGLE + robot_angle
        dist = self.get_arc_min(lidar_angle, arc_deg)
        return dist > safe_dist, dist

    def find_best_clear_direction(self, desired_robot_angle):
        candidates = []

        for deg in range(RETURN_SCAN_MIN_DEG, RETURN_SCAN_MAX_DEG + 1, RETURN_SCAN_STEP_DEG):
            robot_angle = math.radians(deg)

            clear, dist = self.is_robot_angle_clear(
                robot_angle, arc_deg=OPENING_ARC_DEG, safe_dist=RETURN_SAFE_DIST
            )

            if not clear:
                continue

            angle_error = abs(normalize_angle(robot_angle - desired_robot_angle))
            side_penalty = 0.15 * abs(robot_angle)
            score = angle_error + side_penalty - 0.05 * dist

            candidates.append((score, angle_error, robot_angle, dist))

        if not candidates:
            return None, None

        candidates.sort(key=lambda x: x[0])
        _, _, best_angle, best_dist = candidates[0]

        return best_angle, best_dist

    # =========================
    # 运动控制
    # =========================

    def publish_cmd(self, linear_x=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    def reset_stuck_reference(self):
        self.stuck_ref_x = self.x
        self.stuck_ref_y = self.y
        self.stuck_ref_time = time.time()

    def set_state(self, new_state):
        if new_state != self.state:
            self.state = new_state
            self.state_start_time = time.time()
            if new_state in STUCK_MONITORED_STATES:
                self.reset_stuck_reference()
            self.get_logger().info(f"State -> {self.state}")

    # =========================
    # 启动阶段 score 单调前进机制
    # =========================

    def score_monotonic_active(self):
        if self.score_monotonic_start_time is None:
            return False
        if time.time() - self.score_monotonic_start_time >= SCORE_MONOTONIC_DURATION:
            return False
        if self.last_score < SCORE_MONOTONIC_MIN_ACTIVATION:
            return False
        return True

    def compute_score_derivative(self):
        sx = 0.0 if abs(self.x) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.x)
        sy = 0.0 if abs(self.y) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.y)
        return sx * math.cos(self.yaw) + sy * math.sin(self.yaw)

    def choose_score_increasing_turn(self):
        sx = 0.0 if abs(self.x) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.x)
        sy = 0.0 if abs(self.y) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.y)
        d_dscore_dyaw = -sx * math.sin(self.yaw) + sy * math.cos(self.yaw)
        return SMALL_TURN_SPEED if d_dscore_dyaw >= 0.0 else -SMALL_TURN_SPEED

    def score_monotonic_gate(self, linear, angular, mode="normal"):
        # [Priority 6] 不再把 score=abs(x)+abs(y) 当作硬运动门控。
        # 在 C 形走廊里，合法路线常常会暂时降低 score，硬门控会把机器人卡死。
        # score 现在只作为 gap 选择的弱偏好（见 choose_gap_direction），这里直接放行。
        return linear, angular, False

    def relax_best_score_after_detour(self):
        if not self.score_monotonic_active():
            return
        old_best = self.best_score
        self.best_score = self.last_score
        if old_best != self.best_score:
            self.get_logger().info(
                f"SCORE_GUARD: relax best_score {old_best:.2f} -> {self.best_score:.2f} after detour"
            )

    def avoid_exit_yaw_ok(self):
        yaw_diff_deg = abs(math.degrees(normalize_angle(self.yaw - self.avoid_entry_yaw)))
        ok = yaw_diff_deg <= AVOID_EXIT_YAW_TOL_DEG
        if not ok and time.time() - self.last_log_time > 0.8:
            self.get_logger().info(
                f"AVOID_EXIT_GUARD: yaw_diff={yaw_diff_deg:.0f}deg > tol={AVOID_EXIT_YAW_TOL_DEG:.0f}deg, "
                f"block exit to SEARCH_WALL_FOLLOW (entry_yaw={math.degrees(self.avoid_entry_yaw):.0f} "
                f"now_yaw={math.degrees(self.yaw):.0f})"
            )
        return ok

    # =========================
    # 卡住检测
    # =========================

    def check_stuck(self):
        if not self.have_odom:
            return False

        now = time.time()
        elapsed = now - self.stuck_ref_time

        if elapsed < STUCK_TIME:
            return False

        dx = self.x - self.stuck_ref_x
        dy = self.y - self.stuck_ref_y
        moved = math.hypot(dx, dy)

        self.stuck_ref_x = self.x
        self.stuck_ref_y = self.y
        self.stuck_ref_time = now

        if moved < STUCK_MOVE_DIST:
            return True

        return False

    def handle_stuck_if_needed(self):
        if self.state not in STUCK_MONITORED_STATES:
            return False

        if not self.check_stuck():
            return False

        self.escape_resume_state = self.state
        self.get_logger().warn(
            f"Stuck detected in {self.state}: moved less than {STUCK_MOVE_DIST:.3f} m "
            f"within {STUCK_TIME:.1f} s. Enter ESCAPE_BACKUP, then resume {self.escape_resume_state}."
        )
        self.stop_robot()
        self.set_state("ESCAPE_BACKUP")
        return True

    # =========================
    # 【方案2】SEARCHING 单一状态：消费结构化场景 scene
    # =========================

    def _wall_follow_from_scene(self, scene, base_speed=FORWARD_SPEED):
        fw = scene.get("follow_wall")
        if not fw or not fw.get("valid"):
            return None

        dist_error = TARGET_RIGHT_DIST - fw["distance"]
        heading_err = fw["heading_err_rad"]

        angular = WALL_SIGN * (-SCENE_WALL_DIST_K * dist_error) \
            + WALL_SIGN * (SCENE_WALL_HEADING_K * heading_err)
        angular = clamp(angular, -SMALL_TURN_SPEED, SMALL_TURN_SPEED)

        linear = base_speed
        if abs(dist_error) > 0.20 or abs(heading_err) > math.radians(25):
            linear = min(linear, SLOW_FORWARD_SPEED)
        return linear, angular

    def _record_search_path(self):
        """在 SEARCHING 阶段记录面包屑路径。

        AMCL/map 坐标优先用于返回；同时记录 Phase2 local odom 坐标，
        这样 AMCL 不可用时也能沿实际搜索路径反向返回。
        """
        if self.amcl_pose_received:
            if len(self.search_path) == 0:
                self.search_path.append((self.amcl_x, self.amcl_y, self.amcl_yaw))
                self.search_path_last_x = self.amcl_x
                self.search_path_last_y = self.amcl_y
            else:
                dist = math.hypot(self.amcl_x - self.search_path_last_x,
                                  self.amcl_y - self.search_path_last_y)
                if dist >= self.search_path_record_dist and len(self.search_path) < 2000:
                    self.search_path.append((self.amcl_x, self.amcl_y, self.amcl_yaw))
                    self.search_path_last_x = self.amcl_x
                    self.search_path_last_y = self.amcl_y

        if not self.have_odom:
            return
        if len(self.search_path_local) == 0:
            self.search_path_local.append((self.x, self.y, self.yaw))
            self.search_path_local_last_x = self.x
            self.search_path_local_last_y = self.y
            return
        local_dist = math.hypot(self.x - self.search_path_local_last_x,
                                self.y - self.search_path_local_last_y)
        if local_dist >= self.search_path_record_dist and len(self.search_path_local) < 2000:
            self.search_path_local.append((self.x, self.y, self.yaw))
            self.search_path_local_last_x = self.x
            self.search_path_local_last_y = self.y

    def _init_return_path(self):
        """初始化 RETURNING 路径索引。

        优先级：
        1) AMCL/map search_path 反向；
        2) local odom search_path 反向；
        3) Phase 1 return_path；
        4) home/origin fallback。
        """
        self.return_waypoint_idx = -1
        self.return_local_waypoint_idx = -1
        self.phase1_return_waypoint_idx = -1

        def nearest_idx(points, cx, cy):
            best_idx = 0
            best_dist = float("inf")
            for i, (px, py, _) in enumerate(points):
                d = math.hypot(cx - px, cy - py)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            return best_idx, best_dist

        if self.amcl_pose_received and len(self.search_path) >= 2:
            best_idx, best_dist = nearest_idx(self.search_path, self.amcl_x, self.amcl_y)
            self.return_waypoint_idx = best_idx
            self.get_logger().info(
                f"RETURN path init: source=search_path_amcl pts={len(self.search_path)}, "
                f"nearest idx={best_idx} dist={best_dist:.2f}m"
            )
            return

        if len(self.search_path_local) >= 2:
            best_idx, best_dist = nearest_idx(self.search_path_local, self.x, self.y)
            self.return_local_waypoint_idx = best_idx
            self.get_logger().info(
                f"RETURN path init: source=search_path_local pts={len(self.search_path_local)}, "
                f"nearest idx={best_idx} dist={best_dist:.2f}m"
            )
            return

        if self.phase1_memory_loaded and len(self.phase1_return_path) >= 2:
            phase1_path, current_x, current_y, _ = self._phase1_active_path(
                self.phase1_return_path, self.phase1_return_path_map)
            best_idx, best_dist = nearest_idx(phase1_path, current_x, current_y)
            self.phase1_return_waypoint_idx = best_idx
            self.get_logger().info(
                f"RETURN path init: source=phase1_return_path pts={len(self.phase1_return_path)}, "
                f"nearest idx={best_idx} dist={best_dist:.2f}m"
            )
            return

        self.get_logger().warn("RETURN path init: no waypoint path available, fallback to home/origin bias.")

    def _get_return_target(self):
        """返回当前 RETURNING 目标 waypoint/home。

        Returns dict with keys: source, goal_angle, waypoint_bias, use_waypoint, log.
        """
        # 1) 实际 SEARCHING AMCL/map 面包屑反向
        use_amcl_path = (self.return_waypoint_idx >= 0
                         and self.amcl_pose_received
                         and len(self.search_path) > 1)
        if use_amcl_path:
            wp_x, wp_y, _ = self.search_path[self.return_waypoint_idx]
            wp_dist = math.hypot(self.amcl_x - wp_x, self.amcl_y - wp_y)
            if wp_dist < WAYPOINT_REACHED_DIST and self.return_waypoint_idx > 0:
                self.return_waypoint_idx -= 1
                wp_x, wp_y, _ = self.search_path[self.return_waypoint_idx]
                wp_dist = math.hypot(self.amcl_x - wp_x, self.amcl_y - wp_y)
            goal_angle = self.get_goal_angle_robot(wp_x, wp_y, use_amcl=True)
            return {
                "source": "search_path_amcl",
                "goal_angle": goal_angle,
                "waypoint_bias": clamp(RETURN_ORIGIN_BIAS_K * goal_angle,
                                         -RETURN_ORIGIN_BIAS_MAX, RETURN_ORIGIN_BIAS_MAX),
                "use_waypoint": True,
                "log": f"wp_idx={self.return_waypoint_idx}/{len(self.search_path)-1} wp_dist={wp_dist:.2f}",
            }

        # 2) 实际 SEARCHING local odom 面包屑反向
        if self.return_local_waypoint_idx >= 0 and len(self.search_path_local) > 1:
            wp_x, wp_y, _ = self.search_path_local[self.return_local_waypoint_idx]
            wp_dist = math.hypot(self.x - wp_x, self.y - wp_y)
            if wp_dist < WAYPOINT_REACHED_DIST and self.return_local_waypoint_idx > 0:
                self.return_local_waypoint_idx -= 1
                wp_x, wp_y, _ = self.search_path_local[self.return_local_waypoint_idx]
                wp_dist = math.hypot(self.x - wp_x, self.y - wp_y)
            goal_angle = self.get_goal_angle_robot(wp_x, wp_y)
            return {
                "source": "search_path_local",
                "goal_angle": goal_angle,
                "waypoint_bias": clamp(RETURN_ORIGIN_BIAS_K * goal_angle,
                                         -RETURN_ORIGIN_BIAS_MAX, RETURN_ORIGIN_BIAS_MAX),
                "use_waypoint": True,
                "log": f"wp_idx={self.return_local_waypoint_idx}/{len(self.search_path_local)-1} wp_dist={wp_dist:.2f}",
            }

        # 3) Phase 1 return_path（若实际 search_path 不足）
        if self.phase1_return_waypoint_idx >= 0 and len(self.phase1_return_path) > 1:
            phase1_path, current_x, current_y, use_amcl = self._phase1_active_path(
                self.phase1_return_path, self.phase1_return_path_map)
            wp_x, wp_y, _ = phase1_path[self.phase1_return_waypoint_idx]
            wp_dist = math.hypot(current_x - wp_x, current_y - wp_y)
            if wp_dist < WAYPOINT_REACHED_DIST and self.phase1_return_waypoint_idx < len(self.phase1_return_path) - 1:
                self.phase1_return_waypoint_idx += 1
                wp_x, wp_y, _ = phase1_path[self.phase1_return_waypoint_idx]
                wp_dist = math.hypot(current_x - wp_x, current_y - wp_y)
            goal_angle = self.get_goal_angle_robot(wp_x, wp_y, use_amcl=use_amcl)
            return {
                "source": "phase1_return_path_amcl" if use_amcl else "phase1_return_path_local",
                "goal_angle": goal_angle,
                "waypoint_bias": clamp(RETURN_ORIGIN_BIAS_K * goal_angle,
                                         -RETURN_ORIGIN_BIAS_MAX, RETURN_ORIGIN_BIAS_MAX),
                "use_waypoint": True,
                "log": f"wp_idx={self.phase1_return_waypoint_idx}/{len(self.phase1_return_path)-1} wp_dist={wp_dist:.2f}",
            }

        # 4) fallback: home/origin
        if self.amcl_pose_received and self.home_map_set:
            goal_angle = self.get_goal_angle_robot(self.home_map_x, self.home_map_y, use_amcl=True)
            source = "home_amcl"
        else:
            goal_angle = self.get_goal_angle_robot(0.0, 0.0)
            source = "origin_local"
        return {
            "source": source,
            "goal_angle": goal_angle,
            "waypoint_bias": clamp(RETURN_ORIGIN_BIAS_K * goal_angle,
                                     -RETURN_ORIGIN_BIAS_MAX, RETURN_ORIGIN_BIAS_MAX),
            "use_waypoint": False,
            "log": "home/origin fallback",
        }

    def _phase1_nearest_safe_path_idx(self):
        if not self.phase1_safe_path:
            return -1, float("inf")

        phase1_path, current_x, current_y, _ = self._phase1_active_path(
            self.phase1_safe_path, self.phase1_safe_path_map)

        # 只允许小范围回退，避免绕障后跳回已走过很远的点。
        start = max(0, self.phase1_search_idx - 2)
        best_idx = start
        best_dist = float("inf")
        for i in range(start, len(phase1_path)):
            px, py, _ = phase1_path[i]
            d = math.hypot(current_x - px, current_y - py)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, best_dist

    def _phase1_current_search_target(self):
        if not self.phase1_memory_loaded or len(self.phase1_safe_path) < 2:
            return None

        nearest_idx, nearest_dist = self._phase1_nearest_safe_path_idx()
        if nearest_idx >= 0 and nearest_dist <= PHASE1_REJOIN_MAX_SKIP_DIST:
            self.phase1_search_idx = max(self.phase1_search_idx, nearest_idx)

        phase1_path, current_x, current_y, use_amcl = self._phase1_active_path(
            self.phase1_safe_path, self.phase1_safe_path_map)

        # 到达当前 waypoint 后向前推进；lookahead 让轨迹更平滑，不逐点抖动。
        while self.phase1_search_idx < len(self.phase1_safe_path) - 1:
            tx, ty, _ = phase1_path[self.phase1_search_idx]
            if math.hypot(current_x - tx, current_y - ty) > PHASE1_SEARCH_WAYPOINT_REACHED_DIST:
                break
            self.phase1_search_idx += 1

        target_idx = min(
            len(self.phase1_safe_path) - 1,
            self.phase1_search_idx + PHASE1_SEARCH_LOOKAHEAD
        )
        tx, ty, tyaw = phase1_path[target_idx]
        dist = math.hypot(tx - current_x, ty - current_y)
        angle = self.get_goal_angle_robot(tx, ty, use_amcl=use_amcl)
        return {
            "idx": target_idx,
            "base_idx": self.phase1_search_idx,
            "x": tx,
            "y": ty,
            "yaw": tyaw,
            "dist": dist,
            "angle": angle,
            "nearest_dist": nearest_dist,
            "frame": "map" if use_amcl else "local",
        }

    def _front_blocked_for_phase1_search(self, scene):
        front_near = scene.get("front_near", float("inf"))
        front_blocker = scene.get("front_blocker")
        if front_near < PHASE1_SEARCH_BLOCKED_DIST:
            return True
        if front_blocker is not None and front_blocker.get("distance", float("inf")) < FRONT_WARN_DIST:
            return True
        return False

    def _search_along_phase1_memory(self, scene):
        """沿 Phase 1 safe_path 搜索；实时 LiDAR 发现新增障碍物时临时 gap 绕行。"""
        target = self._phase1_current_search_target()
        if target is None:
            return False

        now = time.time()
        front_near = scene.get("front_near", float("inf"))
        desired_angle = target["angle"]
        blocked = self._front_blocked_for_phase1_search(scene)

        if blocked:
            best_angle, best_gap = self.choose_gap_toward_angle(scene, desired_angle)
            if best_angle is not None:
                if front_near < PHASE1_SEARCH_STRONG_BLOCKED_DIST:
                    linear = 0.0
                elif abs(best_angle) > math.radians(35.0):
                    linear = PHASE1_SEARCH_MIN_LINEAR
                else:
                    linear = SLOW_FORWARD_SPEED
                angular = clamp(PHASE1_SEARCH_ANGULAR_K * best_angle, -TURN_SPEED, TURN_SPEED)
                if now - self.last_log_time > 0.8:
                    self.get_logger().info(
                        f"PHASE1_SEARCH obstacle-gap | wp={target['idx']}/{len(self.phase1_safe_path)-1} "
                        f"desired={math.degrees(desired_angle):+.0f}deg gap={math.degrees(best_angle):+.0f}deg "
                        f"front={front_near:.2f} width={best_gap['width_deg']:.0f}"
                    )
                self.publish_cmd(linear, angular)
                return True

            # 没有 gap 时先朝更开阔侧原地/慢速转，避免继续撞向新增障碍物。
            turn_dir = 1.0 if desired_angle >= 0.0 else -1.0
            if front_near < PHASE1_SEARCH_STRONG_BLOCKED_DIST:
                self.publish_cmd(0.0, turn_dir * SMALL_TURN_SPEED)
            else:
                self.publish_cmd(PHASE1_SEARCH_MIN_LINEAR, turn_dir * SMALL_TURN_SPEED)
            return True

        # 正常情况：沿 Phase 1 已知安全 aisle 路径走。
        abs_err = abs(desired_angle)
        if abs_err > math.radians(70.0):
            linear = 0.0
        else:
            speed_scale = clamp(1.0 - abs_err / math.radians(70.0), 0.25, 1.0)
            linear = clamp(
                PHASE1_SEARCH_MAX_LINEAR * speed_scale,
                PHASE1_SEARCH_MIN_LINEAR,
                PHASE1_SEARCH_MAX_LINEAR,
            )
        if abs_err <= math.radians(PHASE1_SEARCH_ANGULAR_DEADBAND_DEG):
            angular = 0.0
        else:
            angular = clamp(PHASE1_SEARCH_ANGULAR_K * desired_angle, -TURN_SPEED, TURN_SPEED)

        if now - self.last_log_time > 0.8:
            self.get_logger().info(
                f"PHASE1_SEARCH path-follow | wp={target['idx']}/{len(self.phase1_safe_path)-1} "
                f"dist={target['dist']:.2f} angle={math.degrees(desired_angle):+.0f}deg "
                f"front={front_near:.2f} cmd=({linear:.2f},{angular:.2f})"
            )
        self.publish_cmd(linear, angular)
        return True

    def search_wall_follow(self, lidar):
        """实体机器人已验证的默认搜索：右墙跟随，遇到凸起后进入贴边绕行。"""
        self._record_search_path()

        if self.red_detected or self.target_locked:
            self.stop_robot()
            self.set_state("REPORTING")
            return

        front_min = lidar["front_min"]
        front_pattern = lidar["front_pattern"]
        scan_change = lidar.get("scan_change", {})
        dynamic_front_protrusion = (
            scan_change.get("available", False)
            and scan_change.get("front_closer", False)
            and scan_change.get("front_changed_ratio", 0.0) > 0.035
            and front_min < CHANGE_NEAR_DIST
        )

        if front_pattern == "obstacle" or dynamic_front_protrusion:
            self.save_debug_event(
                "PROTRUSION", lidar,
                "front_obstacle_or_360deg_change_enter_edge_follow")
            self.rejoin_stable_count = 0
            self.avoid_entry_yaw = self.yaw
            self.set_state("AVOID_OBSTACLE")
            return

        if front_pattern in ["wall", "corner_or_wall"] or front_min < FRONT_STOP_DIST:
            if front_min < FRONT_STOP_DIST:
                linear, angular = 0.0, SMALL_TURN_SPEED
            else:
                linear, angular = SLOW_FORWARD_SPEED, SMALL_TURN_SPEED
            self.publish_cmd(linear, angular)
            return

        linear, angular, _ = self.compute_right_wall_follow_cmd(
            lidar, base_speed=FORWARD_SPEED)
        self.publish_cmd(linear, angular)

    def avoid_obstacle(self, lidar):
        """沿障碍物边缘继续右墙跟随，直到重新找到稳定右墙。"""
        self._record_search_path()

        if self.red_detected or self.target_locked:
            self.stop_robot()
            self.set_state("REPORTING")
            return

        front_min = lidar["front_min"]
        front_right_min = lidar["front_right_min"]
        right_min = lidar["right_min"]

        if self.front_wall_reappeared(lidar) and front_min < PROTRUSION_FRONT_WARN:
            self.rejoin_stable_count = 0
            self.set_state("REJOIN_WALL")
            return

        visible, _ = self.is_right_wall_visible(lidar, require_distance_ok=True)
        if visible and front_min > FRONT_STOP_DIST:
            self.rejoin_stable_count += 1
        else:
            self.rejoin_stable_count = 0

        if self.rejoin_stable_count >= RIGHT_WALL_VISIBLE_STABLE_COUNT:
            if self.avoid_exit_yaw_ok():
                self.rejoin_stable_count = 0
                self.relax_best_score_after_detour()
                self.set_state("SEARCH_WALL_FOLLOW")
                return
            self.rejoin_stable_count = 0

        if front_min < PROTRUSION_FRONT_DANGER:
            self.publish_cmd(0.0, SMALL_TURN_SPEED)
            return
        if front_right_min < MIN_RIGHT_DIST or right_min < MIN_RIGHT_DIST:
            self.publish_cmd(SLOW_FORWARD_SPEED, SMALL_TURN_SPEED)
            return
        if right_min > PROTRUSION_EDGE_LOST_DIST:
            self.publish_cmd(SLOW_FORWARD_SPEED, -SMALL_TURN_SPEED)
            return

        linear, angular, _ = self.compute_right_wall_follow_cmd(
            lidar, base_speed=SLOW_FORWARD_SPEED)
        self.publish_cmd(linear, angular)

    def rejoin_wall(self, lidar):
        """绕过凸起后，将重新出现的墙面稳定地移回机器人右侧。"""
        self._record_search_path()

        if self.red_detected or self.target_locked:
            self.stop_robot()
            self.set_state("REPORTING")
            return

        front_min = lidar["front_min"]
        right_min = lidar["right_min"]
        visible, _ = self.is_right_wall_visible(lidar, require_distance_ok=True)
        if visible and front_min > FRONT_STOP_DIST:
            self.rejoin_stable_count += 1
        else:
            self.rejoin_stable_count = 0

        if self.rejoin_stable_count >= RIGHT_WALL_VISIBLE_STABLE_COUNT:
            if self.avoid_exit_yaw_ok():
                self.rejoin_stable_count = 0
                self.relax_best_score_after_detour()
                self.set_state("SEARCH_WALL_FOLLOW")
                return
            self.rejoin_stable_count = 0

        if front_min < FRONT_STOP_DIST:
            self.publish_cmd(0.0, SMALL_TURN_SPEED)
            return
        if self.front_wall_reappeared(lidar):
            self.publish_cmd(SLOW_FORWARD_SPEED, SMALL_TURN_SPEED)
            return
        if right_min > RIGHT_WALL_VISIBLE_MAX_DIST:
            self.publish_cmd(SLOW_FORWARD_SPEED, -SMALL_TURN_SPEED)
            return
        if right_min < MIN_RIGHT_DIST:
            self.publish_cmd(SLOW_FORWARD_SPEED, SMALL_TURN_SPEED)
            return

        linear, angular, _ = self.compute_right_wall_follow_cmd(
            lidar, base_speed=SLOW_FORWARD_SPEED)
        self.publish_cmd(linear, angular)

    def searching(self, lidar, scene):
        now = time.time()

        # 记录搜索路径（RETURNING 折返用）
        self._record_search_path()

        if self.red_detected or self.target_locked:
            self.stop_robot()
            self.avoiding_cylinder = False
            self.set_state("REPORTING")
            return

        front_blocker = scene.get("front_blocker")
        follow_wall = scene.get("follow_wall")
        front_near = scene.get("front_near", float("inf"))

        if follow_wall and follow_wall.get("valid"):
            self.follow_wall_stable_count += 1
        else:
            self.follow_wall_stable_count = 0
        follow_wall_ready = self.follow_wall_stable_count >= SCENE_FOLLOW_WALL_STABLE_COUNT

        cyl_blocker = (
            front_blocker is not None
            and front_blocker["type"] == "cylinder"
            and front_blocker["distance"] <= CYLINDER_AVOID_TRIGGER_DIST
        )
        if cyl_blocker:
            self.cylinder_confirm_count += 1
        else:
            self.cylinder_confirm_count = 0

        if (ENABLE_CRAB_WALK_AVOIDANCE
                and self.cylinder_confirm_count >= SCENE_CYLINDER_CONFIRM_FRAMES):
            bearing = front_blocker["bearing_deg"]
            if not self.avoiding_cylinder:
                self.avoiding_cylinder = True
                self.avoid_cylinder_side = -1.0 if bearing >= 0 else 1.0
                self.get_logger().info(
                    f"SCENE: cylinder ahead dist={front_blocker['distance']:.2f}m "
                    f"bearing={bearing:.0f}deg R={front_blocker.get('radius',0):.3f} -> avoid"
                )

            if front_near < PROTRUSION_FRONT_DANGER:
                linear, angular = 0.0, self.avoid_cylinder_side * SMALL_TURN_SPEED
            else:
                linear, angular = SLOW_FORWARD_SPEED, self.avoid_cylinder_side * SMALL_TURN_SPEED
            linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
            self.publish_cmd(linear, angular)
            return
        else:
            if self.avoiding_cylinder and not cyl_blocker:
                self.avoiding_cylinder = False
                self.relax_best_score_after_detour()

        if self._search_along_phase1_memory(scene):
            return

        if follow_wall_ready:
            wf = self._wall_follow_from_scene(scene, base_speed=FORWARD_SPEED)
            if wf is not None:
                linear, angular = wf
                if front_blocker and front_blocker["distance"] < FRONT_WARN_DIST:
                    linear = min(linear, SLOW_FORWARD_SPEED)
                if front_near < SCENE_SAFETYNET_FRONT_STOP:
                    linear, angular = 0.0, WALL_SIGN * SMALL_TURN_SPEED
                linear, angular, gated = self.score_monotonic_gate(linear, angular, mode="normal")
                if now - self.last_log_time > 0.8:
                    self.get_logger().info(
                        f"SCENE wall-follow | dist={follow_wall['distance']:.2f} "
                        f"head_err={math.degrees(follow_wall['heading_err_rad']):+.0f}deg "
                        f"front_near={front_near:.2f} cmd=({linear:.2f},{angular:.2f}) gated={gated}"
                    )
                self.publish_cmd(linear, angular)
                return

        best_angle, best_gap = self.choose_gap_direction(scene)
        if best_angle is not None:
            if front_near < SCENE_SAFETYNET_FRONT_STOP:
                linear = 0.0
            elif abs(best_angle) > math.radians(35):
                linear = SLOW_FORWARD_SPEED
            else:
                linear = FORWARD_SPEED
            angular = clamp(SCENE_WALL_HEADING_K * best_angle, -TURN_SPEED, TURN_SPEED)
            linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
            if now - self.last_log_time > 0.8:
                self.get_logger().info(
                    f"SCENE gap-steer | gap_bearing={math.degrees(best_angle):+.0f}deg "
                    f"width={best_gap['width_deg']:.0f}deg depth={best_gap['depth']:.2f} "
                    f"front_near={front_near:.2f}"
                )
            self.publish_cmd(linear, angular)
            return

        if front_near < SCENE_SAFETYNET_FRONT_STOP:
            self.publish_cmd(0.0, WALL_SIGN * SMALL_TURN_SPEED)
        else:
            linear, angular = SLOW_FORWARD_SPEED, -WALL_SIGN * SMALL_TURN_SPEED
            linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
            self.publish_cmd(linear, angular)
        if now - self.last_log_time > 0.8:
            self.get_logger().info(
                f"SCENE safety-net | no wall/gap | front_near={front_near:.2f}"
            )

    def escape_backup(self):
        t = time.time() - self.state_start_time

        if t < BACKUP_DURATION:
            self.publish_cmd(BACKWARD_SPEED, 0.0)
            return

        self.set_state("ESCAPE_TURN")

    def escape_turn(self, lidar):
        t = time.time() - self.state_start_time

        front_left_min = lidar["front_left_min"]
        front_right_min = lidar["front_right_min"]

        if t < ESCAPE_TURN_DURATION:
            # [Priority 5] 恢复转向方向的优先级：
            #   1) 碰撞规避：两侧明显不对称时，转向更开阔一侧；
            #   2) RETURNING 时朝原点方向旋转，避免越逃越远；
            #   3) 否则默认转向更开阔一侧。
            side_bias = None
            if abs(front_left_min - front_right_min) > 0.15:
                side_bias = 1.0 if front_left_min >= front_right_min else -1.0
            if side_bias is None and self.escape_resume_state == "RETURNING":
                if self.amcl_pose_received and self.home_map_set:
                    goal_angle = self.get_goal_angle_robot(
                        self.home_map_x, self.home_map_y, use_amcl=True)
                else:
                    goal_angle = self.get_goal_angle_robot(0.0, 0.0)
                side_bias = 1.0 if goal_angle >= 0.0 else -1.0
            if side_bias is None:
                side_bias = 1.0 if front_left_min >= front_right_min else -1.0
            self.publish_cmd(0.0, side_bias * TURN_SPEED)
            return

        self.reset_stuck_reference()

        resume_state = self.escape_resume_state
        self.escape_resume_state = "SEARCH_WALL_FOLLOW"
        self.relax_best_score_after_detour()
        self.set_state(resume_state)

    # =========================
    # REPORTING 阶段
    # =========================

    def reporting(self):
        self.stop_robot()

        if not self.saved_detection:
            self.save_detection_evidence()
            self.saved_detection = True

        # 初始化折返路径（沿 SEARCHING 面包屑反向走）
        self._init_return_path()

        self.set_state("RETURNING")

    # =========================
    # RETURNING 阶段
    # =========================

    def returning(self, lidar):
        """沿用旧版 RETURNING：右墙跟随 + 凸起贴边绕行，直到回到本地原点。"""
        now = time.time()
        dist_to_origin = math.hypot(self.x, self.y)

        if dist_to_origin <= RETURN_STOP_DIST:
            self.stop_robot()
            self.get_logger().info(
                f"Returned close to start. x={self.x:.3f}, y={self.y:.3f}, "
                f"dist={dist_to_origin:.3f}. Starting final 180 deg turn before DONE."
            )
            self.start_post_return_turn()
            return

        front_min = lidar["front_min"]
        front_right_min = lidar["front_right_min"]
        right_min = lidar["right_min"]
        front_pattern = lidar["front_pattern"]
        scan_change = lidar.get("scan_change", {})
        dynamic_front_protrusion = (
            scan_change.get("available", False)
            and scan_change.get("front_closer", False)
            and scan_change.get("front_changed_ratio", 0.0) > 0.035
            and front_min < CHANGE_NEAR_DIST
        )

        if front_pattern == "obstacle" or dynamic_front_protrusion:
            if front_min < PROTRUSION_FRONT_DANGER:
                self.publish_cmd(0.0, SMALL_TURN_SPEED)
            elif front_right_min < MIN_RIGHT_DIST or right_min < MIN_RIGHT_DIST:
                self.publish_cmd(SLOW_FORWARD_SPEED, SMALL_TURN_SPEED)
            elif right_min > PROTRUSION_EDGE_LOST_DIST:
                self.publish_cmd(SLOW_FORWARD_SPEED, -SMALL_TURN_SPEED)
            else:
                linear, angular, _ = self.compute_right_wall_follow_cmd(
                    lidar, base_speed=SLOW_FORWARD_SPEED)
                self.publish_cmd(linear, angular)
            if now - self.last_return_log_time > 0.6:
                self.get_logger().info(
                    f"RETURN PROTRUSION | dist_origin={dist_to_origin:.3f} | "
                    f"front={front_min:.2f} FR={front_right_min:.2f} right={right_min:.2f} | "
                    f"pattern={front_pattern} dyn={dynamic_front_protrusion}"
                )
                self.last_return_log_time = now
            return

        if front_pattern in ["wall", "corner_or_wall"] or front_min < FRONT_STOP_DIST:
            if front_min < FRONT_STOP_DIST:
                self.publish_cmd(0.0, SMALL_TURN_SPEED)
            else:
                self.publish_cmd(SLOW_FORWARD_SPEED, SMALL_TURN_SPEED)
            return

        linear, angular, _ = self.compute_right_wall_follow_cmd(
            lidar, base_speed=FORWARD_SPEED)
        self.publish_cmd(linear, angular)

    def start_post_return_turn(self):
        self.post_return_turn_start_yaw = self.yaw
        self.stop_robot()
        self.set_state("RETURN_FINAL_TURN")

    def return_final_turn(self):
        turned = abs(normalize_angle(self.yaw - self.post_return_turn_start_yaw))

        if turned >= POST_RETURN_TURN_ANGLE - POST_RETURN_TURN_TOL:
            self.stop_robot()
            self.get_logger().info(
                f"Final 180 deg turn completed: turned={math.degrees(turned):.1f} deg. DONE."
            )
            self.set_state("DONE")
            return

        self.publish_cmd(0.0, POST_RETURN_TURN_SPEED)

    def done(self):
        self.stop_robot()

    # =========================
    # 【Priority 9】全局碰撞保护
    # =========================

    def critical_collision_override(self):
        """扫描原始 LiDAR，若任一 beam 距离 < CRITICAL_COLLISION_DIST，
        立刻停车并朝远离最近障碍方向急转。触发返回 True（调用方应直接 return）。"""
        if self.latest_scan is None:
            return False

        msg = self.latest_scan
        ranges = np.array(msg.ranges, dtype=float)
        lidar_angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        robot_angles = np.array([
            normalize_angle(a - FRONT_ANGLE) for a in lidar_angles
        ])
        valid = (
            np.isfinite(ranges)
            & (ranges >= CRITICAL_COLLISION_MIN_VALID_DIST)
            & (np.abs(robot_angles) <= math.radians(CRITICAL_COLLISION_ARC_DEG))
        )
        if not np.any(valid):
            return False

        valid_ranges = np.where(valid, ranges, np.inf)
        min_idx = int(np.argmin(valid_ranges))
        min_range = float(valid_ranges[min_idx])

        if min_range >= CRITICAL_COLLISION_DIST:
            return False

        robot_angle = float(robot_angles[min_idx])

        # 最近障碍在左侧(robot_angle>0) -> 向右急转(负)，反之向左急转(正)。
        turn_dir = -1.0 if robot_angle >= 0.0 else 1.0
        self.publish_cmd(0.0, turn_dir * CRITICAL_COLLISION_TURN_SPEED)

        if time.time() - self.last_log_time > 0.5:
            self.get_logger().warn(
                f"CRITICAL_COLLISION: min_range={min_range:.3f}m @ "
                f"{math.degrees(robot_angle):+.0f}deg -> stop + emergency turn "
                f"{'R' if turn_dir < 0 else 'L'}"
            )
            self.last_log_time = time.time()
        return True

    # =========================
    # 主循环
    # =========================

    def control_loop(self):
        if time.time() < self.startup_wait_until:
            self.stop_robot()
            if not self.startup_wait_logged:
                self.get_logger().info("Startup wait active: robot remains stopped.")
                self.startup_wait_logged = True
            return

        if self.latest_scan is None:
            self.stop_robot()
            return

        lidar = self.get_lidar_summary()

        score = abs(self.x) + abs(self.y)
        self.last_score = score

        if score > self.best_score:
            self.best_score = score

        now = time.time()

        sm_active = self.score_monotonic_active()
        if self.score_monotonic_start_time is not None:
            sm_left = max(0.0, SCORE_MONOTONIC_DURATION - (now - self.score_monotonic_start_time))
        else:
            sm_left = SCORE_MONOTONIC_DURATION

        if now - self.last_log_time > 0.8:
            front_info = lidar.get("front_info", {})
            scan_change = lidar.get("scan_change", {})
            right_geom = lidar.get("right_wall_geom", {})
            self.get_logger().info(
                f"state={self.state} | "
                f"x={self.x:.2f}, y={self.y:.2f}, yaw={math.degrees(self.yaw):.1f} | "
                f"score={score:.2f}, best={self.best_score:.2f} | "
                f"sm={sm_active} left={sm_left:.0f}s | "
                f"front={lidar['front_min']:.2f}, right={lidar['right_min']:.2f}, "
                f"FL={lidar['front_left_min']:.2f}, FR={lidar['front_right_min']:.2f} | "
                f"pattern={lidar['front_pattern']} "
                f"near_ratio={front_info.get('near_ratio', 0.0):.2f} "
                f"cluster_deg={front_info.get('cluster_deg', 0.0):.1f} | "
                f"right_dist={right_geom.get('right_distance', float('inf')):.2f} "
                f"shape_ok={right_geom.get('shape_ok', False)} | "
                f"change_front={scan_change.get('front_changed_ratio', 0.0):.2f} "
                f"front_closer={scan_change.get('front_closer', False)} | "
                f"red={self.red_pixels} confirm={self.red_seen_count}/{RED_CONFIRM_FRAMES} "
                f"locked={self.target_locked}"
            )
            self.last_log_time = now

        if self.handle_stuck_if_needed():
            return

        # [Priority 9] 全局碰撞保护：在状态分发之前检查最近 beam，
        # 触发则立刻覆盖所有运动（停车 + 远离最近障碍急转）。
        # DONE / REPORTING（本身停车）不参与。
        if ENABLE_CRITICAL_COLLISION_OVERRIDE and self.state not in ("DONE", "REPORTING"):
            if self.critical_collision_override():
                return

        if self.state == "SEARCH_WALL_FOLLOW":
            self.search_wall_follow(lidar)

        elif self.state == "AVOID_OBSTACLE":
            self.avoid_obstacle(lidar)

        elif self.state == "REJOIN_WALL":
            self.rejoin_wall(lidar)

        elif self.state == "ESCAPE_BACKUP":
            self.escape_backup()

        elif self.state == "ESCAPE_TURN":
            self.escape_turn(lidar)

        elif self.state == "REPORTING":
            self.reporting()

        elif self.state == "RETURNING":
            self.returning(lidar)

        elif self.state == "RETURN_FINAL_TURN":
            self.return_final_turn()

        elif self.state == "DONE":
            self.done()

        else:
            self.get_logger().warn(f"Unknown state: {self.state}")
            self.stop_robot()


def main(args=None):
    rclpy.init(args=args)
    node = Phase2Autonomous()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

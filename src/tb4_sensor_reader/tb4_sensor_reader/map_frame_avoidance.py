#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import time

import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Twist
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

# TurtleBot 4 namespace。不同机器人编号时只需要改这里，例如 /T6、/T13。
NAMESPACE = "/T13"

# ROS2 节点名称。
NODE_NAME = "phase2_autonomous"

# 旧版 evidence 备份目录。当前红色方块 evidence 会主要保存在运行目录下的时间戳文件夹。
SAVE_DIR = os.path.expanduser("~/tb4_phase2_evidence")

# 红色方块 evidence 文件夹前缀：运行目录下创建 tb4_red_evidence_YYYYMMDD_HHMMSS。
EVIDENCE_DIR_PREFIX = "tb4_red_evidence"

# LiDAR debug 文件夹前缀。默认关闭，避免 demo 时刷大量文件。
DEBUG_DIR_PREFIX = "tb4_lidar_debug"


# ============================================================
# B. 控制周期与状态行为开关
# ============================================================

# 控制循环周期，单位：秒。0.10 s = 10 Hz。
CONTROL_DT = 0.10

# LiDAR obstacle/wall/corner debug 截图与 CSV。红色 evidence 保存不受此开关影响。
ENABLE_DEBUG_LOG = False

# 这些状态下启用 odom 卡住检测。
# SEARCHING 与 RETURNING 都会生效；最终 5 cm 前进和 180° 转向不启用，避免误判。
STUCK_MONITORED_STATES = {
    "SEARCH_WALL_FOLLOW",
    "AVOID_OBSTACLE",
    "REJOIN_WALL",
    "RETURNING",
}


# ============================================================
# C. 线速度参数（m/s）
# ============================================================

# 搜索阶段正常右墙跟随时的前进速度。
FORWARD_SPEED = 0.28

# 慢速前进：贴墙调整、绕障边缘、rejoin wall 等更谨慎动作。
SLOW_FORWARD_SPEED = 0.10

# 脱困后退速度，负数表示后退。
BACKWARD_SPEED = -0.10

# RETURNING waypoint 追踪最大/最小线速度。
RETURN_MAX_LINEAR = 0.10
RETURN_MIN_LINEAR = 0.025

# RETURNING 右墙贴边返回时的最大线速度。
RETURN_WALL_FOLLOW_MAX_LINEAR = 0.065

# RETURNING 最后直线回原点的线速度上限/下限。
RETURN_FINAL_MAX_LINEAR = 0.055
RETURN_FINAL_MIN_LINEAR = 0.018

# RETURNING 短时 escape 保留速度参数。
RETURN_BACKUP_SPEED = -0.06
RETURN_ESCAPE_LINEAR = 0.055


# ============================================================
# D. 角速度参数（rad/s）
# ============================================================

# 大角速度：明显原地转向或脱困。
TURN_SPEED = 0.45

# 小角速度：贴墙微调、缓慢修正方向。
SMALL_TURN_SPEED = 0.25

# RETURNING waypoint 追踪最大角速度。
RETURN_MAX_ANGULAR = 0.45

# RETURNING 最后直线回原点时最大角速度。
RETURN_FINAL_MAX_ANGULAR = 0.35

# RETURNING 无右墙可跟随时的搜索转向速度。
RETURN_NO_WALL_TURN_SPEED = 0.28

# RETURNING 短时 escape 保留角速度。
RETURN_ESCAPE_ANGULAR = 0.38

# 回到原点后、DONE 前的 180° 转向角速度。
POST_RETURN_TURN_SPEED = 0.35


# ============================================================
# E. 距离、尺寸与安全余量（m）
# ============================================================

# 机器人直径约 0.30 m，半径 0.15 m。所有安全距离必须大于机器人半径。
ROBOT_DIAMETER = 0.30
ROBOT_RADIUS = ROBOT_DIAMETER / 2.0
ROBOT_SIDE_CLEARANCE = 0.09
ROBOT_FRONT_CLEARANCE = 0.18

# 前方安全距离。
FRONT_STOP_DIST = 0.35
FRONT_WARN_DIST = 0.50

# 右墙跟随目标距离范围。
TARGET_RIGHT_DIST = 0.38
MIN_RIGHT_DIST = 0.24
MAX_RIGHT_DIST = 0.58

# 右墙凸起/障碍物贴边绕行距离阈值。
PROTRUSION_FRONT_DANGER = max(0.30, ROBOT_RADIUS + ROBOT_FRONT_CLEARANCE)
PROTRUSION_FRONT_WARN = 0.55
PROTRUSION_EDGE_LOST_DIST = 0.75

# 绕障退出方向保护：
# 从 AVOID_OBSTACLE / REJOIN_WALL 退回 SEARCH_WALL_FOLLOW 时，
# 如果当前 yaw 和进入绕障时的 yaw 偏差超过此阈值（度），
# 则不允许退出——防止绕障中途被误判"右墙重新出现"，
# 导致机器人带着反方向的 yaw 回到右墙跟随（朝原点走）。
AVOID_EXIT_YAW_TOL_DEG = 75.0

# LiDAR 当前帧/上一帧差分距离阈值。
CHANGE_DIST_TH = 0.18
CHANGE_NEAR_DIST = 0.95
CHANGE_MAX_VALID = 3.5

# LiDAR pattern 分类距离阈值。
PATTERN_NEAR_DIST = 0.50
PATTERN_EDGE_JUMP = 0.35
PATTERN_MAX_VALID = 3.5
OBSTACLE_MAX_WIDTH = 0.55

# 右侧墙面 V 型分析距离阈值。
RIGHT_SHAPE_MAX_VALID = 3.5
RIGHT_SHAPE_VALLEY_MIN_DEPTH = 0.055
RIGHT_WALL_VISIBLE_MAX_DIST = 0.82

# RETURNING 目标/障碍安全距离。
RETURN_FINAL_DIRECT_DIST = 0.40
RETURN_STOP_DIST = 0.15
RETURN_FINAL_STRAIGHT_DIST = 0.20
RETURN_SLOW_DIST = 0.55
RETURN_SAFE_DIST = 0.65
RETURN_WARN_DIST = 0.85
RETURN_DANGER_DIST = 0.45
RETURN_SIDE_DANGER_DIST = 0.36

# 搜索阶段路径记录距离间隔。
PATH_RECORD_MIN_DIST = 0.18
WAYPOINT_REACHED_DIST = 0.22

# 卡住检测距离阈值：在 STUCK_TIME 内移动小于此距离，认为卡住。
STUCK_MOVE_DIST = 0.08

# 红色方块真实边长，单目测距使用，单位 m。默认 6 cm。
RED_CUBE_SIZE_M = 0.06


# ============================================================
# F. 角度、扇区宽度与角度容差（deg / rad）
# ============================================================

# 你们确认过：当前 LiDAR 坐标系中，机器人正前方对应 -pi/2。
FRONT_ANGLE = -math.pi / 2.0

# 360° LiDAR 差分前方 ROI 半角范围。
CHANGE_FRONT_ROI_DEG = 80.0

# 变化 cluster 与障碍/墙面分类角宽阈值。
CHANGE_MIN_CLUSTER_DEG = 3.0
WALL_CLUSTER_DEG = 35.0
OBSTACLE_MIN_DEG = 5.0
OBSTACLE_MAX_DEG = 32.0

# 右墙三点采样角度，仅用于 log / 轻微辅助。
RIGHT_FRONT_ANGLE_DEG = -60.0
RIGHT_MID_ANGLE_DEG = -90.0
RIGHT_BACK_ANGLE_DEG = -120.0
RIGHT_PARALLEL_ARC_DEG = 22.0

# 右侧整段 V 型曲线分析窗口。
RIGHT_SHAPE_CENTER_DEG = -90.0
RIGHT_SHAPE_ARC_DEG = 120.0
RIGHT_SHAPE_MIN_VALLEY_WIDTH_DEG = 20.0
RIGHT_WALL_VISIBLE_MIN_VALLEY_WIDTH_DEG = 10.0

# RETURNING 目标方向扫描/判断角度。
GOAL_ARC_DEG = 35
OPENING_ARC_DEG = 25
RETURN_SCAN_MIN_DEG = -95
RETURN_SCAN_MAX_DEG = 95
RETURN_SCAN_STEP_DEG = 10
RETURN_WAYPOINT_DIRECT_ANGLE_DEG = 38.0
RETURN_ALIGN_ONLY_DEG = 58.0
RETURN_FINAL_ALIGN_ONLY_DEG = 45.0

# 搜索路径记录 yaw 变化阈值。
PATH_RECORD_MIN_YAW_DEG = 15.0

# 回到原点后、DONE 前最终旋转 180°。
POST_RETURN_TURN_ANGLE = math.pi
POST_RETURN_TURN_TOL = math.radians(4.0)


# ============================================================
# G. 时间参数（s）
# ============================================================

# 卡住检测时间窗口。现在 SEARCHING 和 RETURNING 都会检查。
STUCK_TIME = 6.0

# 搜索阶段脱困动作持续时间。
BACKUP_DURATION = 0.70
ESCAPE_TURN_DURATION = 1.20

# RETURNING 短时 escape 保留持续时间参数。
RETURN_ESCAPE_BACKUP_TIME = 0.35
RETURN_ESCAPE_TURN_TIME = 0.65
RETURN_ESCAPE_ARC_TIME = 1.25

# 红色 evidence 自动快照冷却时间。
EVIDENCE_SNAPSHOT_COOLDOWN = 0.35

# LiDAR debug 事件冷却时间。
DEBUG_EVENT_COOLDOWN = 0.80


# ============================================================
# H. 控制增益与角速度叠加系数
# ============================================================

# 右墙跟随控制增益。
RIGHT_DIST_K = 1.00
RIGHT_PARALLEL_K = 0.70
RIGHT_SHAPE_ERROR_SCALE = 0.42

# RETURNING 目标追踪控制增益。
RETURN_LINEAR_K = 0.30
RETURN_ANGULAR_K = 1.25
RETURN_FINAL_LINEAR_K = 0.28

# RETURNING 右墙贴边时叠加 waypoint/origin 目标偏置，避免沿墙一直走远。
RETURN_TARGET_BIAS_K = 0.42
RETURN_TARGET_BIAS_MAX = 0.16


# ============================================================
# I. 比例、计数、防抖与形状判断阈值
# ============================================================

# 墙面 / 近距离 cluster 比例。
WALL_RATIO_TH = 0.60
PROTRUSION_REJOIN_FRONT_WALL_RATIO = 0.50

# 右墙 V 型曲线平滑与形状判断。
RIGHT_SHAPE_SMOOTH_WINDOW = 9
RIGHT_SHAPE_MIN_VALID_RATIO = 0.48
RIGHT_SHAPE_CENTER_TOL = 0.18
RIGHT_SHAPE_MONO_RATIO_TH = 0.62
RIGHT_SHAPE_MAX_JUMP = 0.16
RIGHT_SHAPE_MAX_JUMP_RATIO = 0.06
RIGHT_PARALLEL_TOL = 0.16
RIGHT_WALL_VISIBLE_MIN_VALID_RATIO = 0.42
RIGHT_WALL_VISIBLE_MAX_JUMP_RATIO = 0.16

# 多帧防抖计数。
RIGHT_WALL_STABLE_COUNT = 3
RIGHT_WALL_VISIBLE_STABLE_COUNT = 2
RETURN_BLOCKED_CONFIRM = 2
RETURN_GOAL_CLEAR_REQUIRED = 3

# 搜索阶段允许 abs(x)+abs(y) score 小幅下降，避免转弯/绕障被过度限制。
SCORE_DROP_TOLERANCE = 0.12

# ---- 启动阶段 score 单调前进机制（仅在 SEARCHING 前 60s 生效）----
# 在 SEARCH_WALL_FOLLOW / AVOID_OBSTACLE / REJOIN_WALL 三个状态中：
#   如果 score = |x|+|y| 比历史最大值 best_score 下降超过容差，
#   并且当前车头方向继续前进会让 score 进一步减小，
#   才禁止前进（linear=0），原地旋转到 score 增大方向后自动放行。

# 机制生效时长（秒）。
SCORE_MONOTONIC_DURATION = 60.0

# score 太小时跳过机制，避免在原点附近死锁。
SCORE_MONOTONIC_MIN_ACTIVATION = 0.30

# SEARCH_WALL_FOLLOW 中的 score 下降容差。
SCORE_MONOTONIC_TOL_NORMAL = 0.12

# AVOID_OBSTACLE / REJOIN_WALL 中使用更宽松的容差。
SCORE_MONOTONIC_TOL_AVOID = 0.35

# dscore/dt > 此阈值时认为"不再朝原点靠近"，放行。
SCORE_MONOTONIC_DIR_EPS = 0.05

# sign(x)/sign(y) 的死区半径：|x| 或 |y| 小于此值时视 sign 为 0，
# 避免 odom 噪声导致 dscore 抖动。
SCORE_SIGN_DEADZONE = 0.05

# 最大路径点数量与 evidence 自动快照数量。
MAX_SAFE_PATH_POINTS = 800
EVIDENCE_MAX_AUTO_SNAPSHOTS = 8


# ============================================================
# J. 红色方块检测：HSV 阈值与 bbox 过滤
# ============================================================
# 本版本已经关闭/移除 YOLO 检测入口，恢复为 HSV 颜色阈值检测。
# 检测流程：BGR -> HSV -> 两段红色 hue mask -> 形态学滤波 -> 最大红色轮廓 bbox -> 连续帧确认。
# 如需恢复 YOLO，请另建版本，不建议在 demo 前混合两套检测逻辑。

# HSV 中红色跨越 0/180 hue 边界，所以使用两段阈值。
RED_LOW1 = np.array([0, 150, 110])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 150, 110])
RED_HIGH2 = np.array([180, 255, 255])

# 红色 mask 总像素阈值；低于该值不认为看到目标。
MIN_RED_PIXELS = 5800

# 连续多少帧满足红色目标条件后才锁定目标。
RED_CONFIRM_FRAMES = 3

# 红色方块 bbox 形状与尺寸过滤。
RED_ASPECT_MIN = 0.5
RED_ASPECT_MAX = 2.0
MIN_BOX_WIDTH_PX = 8
MIN_BOX_HEIGHT_PX = 8

# YOLO 代码状态说明：当前文件不启用 YOLO import、model loading 或 predict。
# # from ultralytics import YOLO
# # YOLO_MODEL_PATH = os.path.expanduser("~/ros2_ws/model/best.pt")
# # self.yolo_model = YOLO(YOLO_MODEL_PATH)


# ============================================================
# K. 相机单目测距参数
# ============================================================
# 当前测距不是 OAK-D stereo depth；它使用：bbox 像素大小 + 已知方块边长 + RGB 相机 FOV 估算距离。

# OAK-D Pro RGB 近似视场角。根据图像分辨率动态换算 fx/fy。
OAK_RGB_HFOV_DEG = 66.0
OAK_RGB_VFOV_DEG = 54.0

# 若现场用标定板测得更准确焦距，可设 True 并修改 FOCAL_LENGTH_PX。
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
    """
    在 360° LiDAR 布尔数组中寻找连续 True 区域。
    支持首尾相连的 cluster，例如 index 1070~1079 和 0~8 会被视作同一区域。
    返回值为若干 index list，而不是 (start, end)，这样可以自然表示跨越数组边界的 cluster。
    """
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

        os.makedirs(SAVE_DIR, exist_ok=True)

        # 红色方块测距 evidence 文件夹：保存在 ros2 run 命令启动时的当前路径下。
        # 例如从 ~/ros2_ws 启动，就会生成 ~/ros2_ws/tb4_red_evidence_YYYYMMDD_HHMMSS。
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

        # LiDAR debug 事件日志默认关闭。
        # 注意：红色方块 evidence 文件夹和 CSV 仍然会正常保存。
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
            f"{NAMESPACE}/cmd_vel",
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            f"{NAMESPACE}/scan",
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            f"{NAMESPACE}/odom",
            self.odom_callback,
            10
        )

        self.image_sub = self.create_subscription(
            CompressedImage,
            f"{NAMESPACE}/oakd/rgb/image_raw/compressed",
            self.image_callback,
            10
        )

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
        # self.yaw 使用“起点坐标系”的标准数学角度：
        # +X = 起点时机器人右侧，+Y = 起点时机器人前方。
        # 因此启动时 self.yaw = pi/2，表示机器人朝向 +Y。
        self.yaw = math.pi / 2.0
        self.have_odom = False

        self.start_x_raw = 0.0
        self.start_y_raw = 0.0
        self.start_yaw = 0.0
        self.start_recorded = False

        # 起点坐标系位置。起点=(0,0)，+Y=启动时前方，+X=启动时右侧。
        self.x = 0.0
        self.y = 0.0

        self.best_score = 0.0
        self.last_score = 0.0

        # score 单调机制启动时刻。None = 尚未收到 odom。
        self.score_monotonic_start_time = None

        self.state = "SEARCH_WALL_FOLLOW"
        self.state_start_time = time.time()
        self.escape_resume_state = "SEARCH_WALL_FOLLOW"

        self.stuck_ref_x = 0.0
        self.stuck_ref_y = 0.0
        self.stuck_ref_time = time.time()

        self.red_detected = False
        self.red_pixels = 0
        self.saved_detection = False

        # 红色目标锁定逻辑
        self.target_locked = False
        self.red_seen_count = 0

        self.cube_robot_x = None
        self.cube_robot_y = None
        self.cube_global_x = None
        self.cube_global_y = None
        self.cube_distance = None

        self.last_log_time = 0.0
        self.last_return_log_time = 0.0

        # 搜索阶段记录安全路径，返回阶段倒序跟踪这些 waypoint。
        self.safe_path = []
        self.path_record_last_x = 0.0
        self.path_record_last_y = 0.0
        self.path_record_last_yaw = 0.0
        self.return_path = []
        self.return_path_index = None
        self.return_initialized = False

        # REJOIN_WALL 需要连续多帧满足右墙平行，避免刚碰到边缘就误切回 SEARCH。
        self.rejoin_stable_count = 0

        # 进入 AVOID_OBSTACLE 时记录的 yaw，用于退出方向保护。
        self.avoid_entry_yaw = 0.0

        # RETURNING 阶段不再记录新路径；这里保留少量状态计数用于多 beam 判断防抖。
        # 返回时优先追踪冻结的 searching path，路径受阻时用右墙贴边逻辑绕回。
        self.return_escape_phase = "IDLE"
        self.return_escape_start_time = 0.0
        self.return_escape_dir = 1.0
        self.return_blocked_count = 0
        self.return_goal_clear_count = 0

        # RETURNING 完成后、DONE 前的最终 180° 转向起始 yaw。
        self.post_return_turn_start_yaw = 0.0

        self.timer = self.create_timer(CONTROL_DT, self.control_loop)

        self.get_logger().info("Phase2 autonomous node started.")
        self.get_logger().info(f"Namespace: {NAMESPACE}")
        self.get_logger().info(
            "Strategy: 360deg current/previous LiDAR change detection + right-sector V-shape wall following + non-timed protrusion edge following + frozen search-path right-wall return."
        )
        if ENABLE_DEBUG_LOG:
            self.get_logger().info(f"Debug event folder: {self.debug_run_dir}")
            self.get_logger().info(f"Debug CSV log: {self.debug_csv_path}")
        else:
            self.get_logger().info("LiDAR debug event log disabled.")
        self.get_logger().info(f"Red ranging evidence folder: {self.evidence_run_dir}")
        self.get_logger().info(f"Red ranging CSV log: {self.evidence_csv_path}")

    # =========================
    # ROS callbacks
    # =========================

    def scan_callback(self, msg):
        # 保存上一时刻和当前时刻的完整 360° ranges。
        # RPLIDAR-A1 常见为 1080 beams；这里按实际 msg.ranges 长度处理。
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

            # 起点坐标系定义：
            # - start point = (0, 0)
            # - robot initial forward direction = +Y
            # - robot initial right side = +X
            self.x = 0.0
            self.y = 0.0
            self.yaw = math.pi / 2.0

            self.stuck_ref_x = self.x
            self.stuck_ref_y = self.y
            self.stuck_ref_time = time.time()

            self.safe_path = [(0.0, 0.0)]
            self.path_record_last_x = 0.0
            self.path_record_last_y = 0.0
            self.path_record_last_yaw = self.yaw

            self.score_monotonic_start_time = time.time()

            self.get_logger().info(
                f"Start recorded as origin: x=0.000, y=0.000, "
                f"coordinate frame: +Y=initial forward, +X=initial right | "
                f"raw_x={self.start_x_raw:.3f}, raw_y={self.start_y_raw:.3f}, "
                f"raw_yaw={math.degrees(self.start_yaw):.1f} deg, "
                f"frame_yaw={math.degrees(self.yaw):.1f} deg"
            )
            self.get_logger().info(
                f"Score-monotonic guard armed for first {SCORE_MONOTONIC_DURATION:.0f}s "
                f"(normal_tol={SCORE_MONOTONIC_TOL_NORMAL}, avoid_tol={SCORE_MONOTONIC_TOL_AVOID})"
            )
        else:
            dx_raw = self.x_raw - self.start_x_raw
            dy_raw = self.y_raw - self.start_y_raw

            # 将 ROS odom 原始位移旋转到起点坐标系。
            # forward_axis = robot 启动时的朝向，记为 +Y。
            # right_axis   = robot 启动时的右侧，记为 +X。
            forward = dx_raw * math.cos(self.start_yaw) + dy_raw * math.sin(self.start_yaw)
            right = dx_raw * math.sin(self.start_yaw) - dy_raw * math.cos(self.start_yaw)

            self.x = right
            self.y = forward

            # 在起点坐标系中，标准数学角度从 +X 逆时针量起；
            # 启动时机器人朝向 +Y，所以 yaw = pi/2。
            self.yaw = normalize_angle((self.yaw_raw - self.start_yaw) + math.pi / 2.0)

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

        # =========================
        # 连续 3 帧确认 + target lock
        # =========================
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

        # 红色像素达到阈值后，自动保存若干张测距 evidence 快照。
        # 这里不改变任何状态机/运动控制，只记录当前图像、odom 和估算坐标。
        if self.latest_red_bbox is not None and self.red_pixels >= MIN_RED_PIXELS:
            self.save_red_range_snapshot(trigger="threshold", force=False)

        display = frame.copy()

        if self.latest_red_bbox is not None:
            x, y, w, h = self.latest_red_bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(display, (x + w // 2, y + h // 2), 5, (255, 0, 0), -1)

        cv2.putText(
            display,
            f"state={self.state}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"red_pixels={self.red_pixels}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"odom_rel=({self.x:.2f},{self.y:.2f})",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if self.latest_range_result is not None:
            cv2.putText(
                display,
                f"cube=(x_right={self.latest_range_result['cube_robot_x']:.2f},y_forward={self.latest_range_result['cube_robot_y']:.2f})m dist={self.latest_range_result['distance_robot']:.2f}m",
                (20, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2
            )

        if self.target_locked:
            status_text = "RED LOCKED"
        elif self.red_seen_count > 0:
            status_text = f"RED CONFIRM {self.red_seen_count}/{RED_CONFIRM_FRAMES}"
        else:
            status_text = ""

        if status_text:
            cv2.putText(
                display,
                status_text,
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        cv2.imshow("TB4 Camera", display)
        cv2.waitKey(1)


    # =========================
    # 红色方块测距 evidence 记录
    # =========================

    def compute_red_hsv_stats(self, hsv_image, red_mask, bbox):
        """
        统计检测到的红色目标 HSV 值，用于 final evidence txt。

        统计范围优先使用 bbox 内 mask>0 的红色像素；如果 bbox 内没有有效红色像素，
        则退化为 bbox 内所有像素。OpenCV HSV 范围：H=[0,179], S=[0,255], V=[0,255]。
        """
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
        """创建本次运行的红色方块测距 CSV 文件。"""
        header = [
            "snapshot_id",
            "wall_time",
            "trigger",
            "state",
            "robot_x",
            "robot_y",
            "robot_yaw_deg",
            "red_pixels",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "bbox_cx",
            "bbox_cy",
            "img_w",
            "img_h",
            "fx_px",
            "fy_px",
            "hfov_deg",
            "vfov_deg",
            "z_from_width_m",
            "z_from_height_m",
            "cube_robot_x_right_m",
            "cube_robot_y_forward_m",
            "cube_distance_from_robot_m",
            "cube_global_x_m",
            "cube_global_y_m",
            "bearing_deg",
            "hsv_sample_mode",
            "hsv_sample_count",
            "hsv_center_h",
            "hsv_center_s",
            "hsv_center_v",
            "hsv_mean_h",
            "hsv_mean_s",
            "hsv_mean_v",
            "hsv_median_h",
            "hsv_median_s",
            "hsv_median_v",
            "image_path",
        ]

        with open(self.evidence_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def get_rgb_intrinsics_from_image_size(self, img_w, img_h):
        """
        根据 OAK-D Pro RGB 相机 FOV 和当前图像分辨率估算 pinhole intrinsics。

        fx = (W/2) / tan(HFOV/2)
        fy = (H/2) / tan(VFOV/2)

        如果 USE_FIXED_FOCAL_LENGTH=True，则 fx=fy=FOCAL_LENGTH_PX，方便现场标定后覆盖。
        """
        if USE_FIXED_FOCAL_LENGTH:
            return float(FOCAL_LENGTH_PX), float(FOCAL_LENGTH_PX)

        fx = (img_w / 2.0) / math.tan(math.radians(OAK_RGB_HFOV_DEG) / 2.0)
        fy = (img_h / 2.0) / math.tan(math.radians(OAK_RGB_VFOV_DEG) / 2.0)
        return float(fx), float(fy)

    def draw_red_range_overlay(self, frame, result, trigger):
        """在 evidence 图片上画 bbox、机器人位姿和测距结果。"""
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
            cv2.putText(
                display,
                line,
                (18, y0 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2
            )

        return display

    def save_red_range_snapshot(self, trigger="threshold", force=False):
        """
        保存红色方块测距快照和 CSV 行。

        这个函数只记录 evidence，不改变运动控制、状态机或 target lock 逻辑。
        """
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
            bx,
            by,
            bw,
            bh,
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
        """创建本次运行的 debug CSV 文件。"""
        if not ENABLE_DEBUG_LOG or self.debug_csv_path is None:
            return

        header = [
            "event_id",
            "wall_time",
            "trigger_type",
            "reason",
            "state",
            "x",
            "y",
            "yaw_deg",
            "score_abs_xy",
            "best_score_abs_xy",
            "front_min",
            "front_left_min",
            "front_right_min",
            "left_min",
            "right_min",
            "front_pattern",
            "front_left_pattern",
            "front_right_pattern",
            "near_ratio",
            "cluster_deg",
            "physical_width",
            "min_dist",
            "edge_count",
            "both_edges",
            "left_recovery",
            "right_recovery",
            "outside_recovery",
            "red_pixels",
            "target_locked",
            "image_path",
        ]

        with open(self.debug_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def save_debug_event(self, trigger_type, lidar, reason):
        """
        保存 LiDAR 分类触发事件。

        trigger_type:
        - OBSTACLE: 前方被 classify_arc_pattern 判定为独立障碍物
        - WALL: 前方被判定为墙
        - CORNER_OR_WALL: 前方被判定为墙角/死角
        """
        if not ENABLE_DEBUG_LOG:
            return

        now = time.time()
        last_time = self.debug_last_save_time.get(trigger_type, 0.0)

        # wall/corner 可能连续 10 Hz 触发。这里做轻微防抖，避免短时间刷爆硬盘。
        if now - last_time < DEBUG_EVENT_COOLDOWN:
            return

        self.debug_last_save_time[trigger_type] = now
        self.debug_event_id += 1

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_name = f"event_{self.debug_event_id:04d}_{timestamp}_{trigger_type}.jpg"
        image_path = os.path.join(self.debug_run_dir, image_name)

        front_info = lidar.get("front_info", {})

        # 优先保存当前相机帧。如果相机还没收到图像，保存一张黑底占位图并写明 NO CAMERA FRAME。
        if self.latest_frame is not None:
            display = self.latest_frame.copy()
        else:
            display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                display,
                "NO CAMERA FRAME",
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

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
            cv2.putText(
                display,
                line,
                (20, y0 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

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
        """
        使用 RGB 图像 bbox + OAK-D Pro RGB FOV + 已知红色方块尺寸估算目标位置。

        坐标定义：
        - start/global x: 起点时机器人右侧为 +X
        - start/global y: 起点时机器人前方为 +Y
        - robot-local x: 当前机器人右侧为 +X
        - robot-local y: 当前机器人前方为 +Y

        因此图像中目标偏右时 cube_robot_x 为正，目标在正前方时 cube_robot_y 为正。

        注意：这是基于单目尺寸的估计，不是 OAK-D stereo depth topic 的真实深度。
        如果现场能稳定读取 depth image，可以后续再替换为 depth[y, x]。
        """
        if self.latest_frame is None or self.latest_red_bbox is None:
            return None

        img_h, img_w = self.latest_frame.shape[:2]
        bx, by, bw, bh = self.latest_red_bbox

        if bw <= 0 or bh <= 0:
            return None

        fx, fy = self.get_rgb_intrinsics_from_image_size(img_w, img_h)

        z_from_width = (RED_CUBE_SIZE_M * fx) / float(bw)
        z_from_height = (RED_CUBE_SIZE_M * fy) / float(bh)

        # 方块可能因为姿态/遮挡导致宽高估计略有差异。这里取平均作为前向距离估计。
        distance_forward = 0.5 * (z_from_width + z_from_height)

        box_cx = bx + bw / 2.0
        box_cy = by + bh / 2.0
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0

        pixel_offset_x = box_cx - img_cx
        pixel_offset_y = box_cy - img_cy

        # robot-local 坐标：x = right, y = forward
        cube_robot_x = pixel_offset_x * distance_forward / fx
        cube_robot_y = distance_forward
        cube_robot_z_camera = -pixel_offset_y * distance_forward / fy

        distance_robot = math.hypot(cube_robot_x, cube_robot_y)

        # bearing_rad 仍然使用 LiDAR/控制中的机器人角度习惯：
        # 0 = 正前方，正数 = 左侧，负数 = 右侧。
        bearing_rad = math.atan2(-cube_robot_x, cube_robot_y)

        # 将 robot-local 坐标投影到起点/global 坐标。
        # self.yaw 是当前朝向在起点坐标系中的角度，启动时为 pi/2，即朝向 +Y。
        # forward_unit = (cos(yaw), sin(yaw))
        # right_unit   = (sin(yaw), -cos(yaw))
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

            cv2.putText(
                display,
                f"robot=({self.x:.2f},{self.y:.2f})",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if result is not None:
                cv2.putText(
                    display,
                    f"cube_robot=(x_right={result['cube_robot_x']:.2f}, y_forward={result['cube_robot_y']:.2f})m",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    display,
                    f"cube_global=({result['cube_global_x']:.2f},{result['cube_global_y']:.2f})m",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
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
        arc = extract_arc(
            list(msg.ranges),
            msg.angle_min,
            msg.angle_increment,
            center_angle,
            arc_deg
        )

        arc = np.array(arc, dtype=float)
        valid = np.isfinite(arc)
        if not np.any(valid):
            return float("inf")

        return float(np.min(arc[valid]))

    def get_arc_median(self, center_angle, arc_deg, max_valid=PATTERN_MAX_VALID):
        """返回某个角度扇区的有效中位数，比 min 更适合估计墙面距离。"""
        if self.latest_scan is None:
            return float("inf")

        msg = self.latest_scan
        arc = extract_arc(
            list(msg.ranges),
            msg.angle_min,
            msg.angle_increment,
            center_angle,
            arc_deg
        )

        arc = np.array(arc, dtype=float)
        valid = np.isfinite(arc) & (arc > 0.02) & (arc < max_valid)
        if not np.any(valid):
            return float("inf")

        return float(np.median(arc[valid]))

    def get_scan_change_summary(self):
        """
        比较当前一整圈 LiDAR beams 与上一时刻一整圈 beams，找出发生明显变化的区域。

        关键输出：
        - clusters: 所有 360° changed clusters
        - front_closer: 前方 ROI 内是否有物体明显变近
        - front_changed_ratio: 前方 ROI 内变化 beam 比例

        这不是单独替代墙/障碍物判断，而是作为“当前环境几何正在变化”的额外证据。
        """
        if self.latest_scan is None or self.curr_scan_ranges is None or self.prev_scan_ranges is None:
            return {
                "available": False,
                "beam_count": 0,
                "changed_count": 0,
                "changed_ratio": 0.0,
                "front_changed_ratio": 0.0,
                "front_closer": False,
                "front_opened": False,
                "clusters": [],
                "front_clusters": [],
            }

        msg = self.latest_scan
        n = min(len(self.curr_scan_ranges), len(self.prev_scan_ranges))
        if n == 0:
            return {
                "available": False,
                "beam_count": 0,
                "changed_count": 0,
                "changed_ratio": 0.0,
                "front_changed_ratio": 0.0,
                "front_closer": False,
                "front_opened": False,
                "clusters": [],
                "front_clusters": [],
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

            # 用 circular mean 计算中心角，避免 -pi/pi 附近跳变。
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
        """
        基于一个 LiDAR 扇区内的 beam 形状进行分类。

        注意：这里仍然保留 wall / obstacle / corner_or_wall 的几何分类，
        但 obstacle 不再进入固定时间绕行动作，而是作为“右墙凸起”贴边绕行。
        """
        if self.latest_scan is None:
            return "no_scan", {}

        msg = self.latest_scan

        raw_arc = extract_arc(
            list(msg.ranges),
            msg.angle_min,
            msg.angle_increment,
            center_angle,
            arc_deg
        )

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
        """
        返回某个扇形内按角度顺序排列的距离值和机器人坐标系角度。

        robot_angle = 0 表示正前方，负角度表示右侧，正角度表示左侧。
        例如右侧扇形约为 -150° -> -90° -> -30°。
        """
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
        """简单移动平均，用于降低单帧 LiDAR 噪声。"""
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
        """
        使用右侧整段扇形 LiDAR 曲线判断墙面几何。

        理想右侧平行墙面的距离序列应满足：
        - 从右后方到正右方向：距离逐渐变近；
        - 从正右方向到右前方：距离逐渐变远；
        - 整体变化连续，不能有大量突变边缘；
        - 谷底应靠近正右方向。谷底偏前，说明车头更靠近墙，需要左转；
          谷底偏后，说明车尾更靠近墙，需要右转。
        """
        center_angle = FRONT_ANGLE + math.radians(RIGHT_SHAPE_CENTER_DEG)
        arc, valid, robot_angles = self.get_ordered_arc_with_angles(
            center_angle,
            RIGHT_SHAPE_ARC_DEG,
            RIGHT_SHAPE_MAX_VALID
        )

        base = {
            "available": False,
            "shape_ok": False,
            "parallel_good": False,
            "right_distance": float("inf"),
            "right_front": float("inf"),
            "right_mid": float("inf"),
            "right_back": float("inf"),
            "right_min": float("inf"),
            "parallel_error": 0.0,
            "valley_index": -1,
            "valley_angle_deg": 0.0,
            "valley_offset": 0.0,
            "valid_ratio": 0.0,
            "valley_depth": 0.0,
            "valley_width_deg": 0.0,
            "mono_ratio": 0.0,
            "jump_ratio": 1.0,
            "left_mono_ratio": 0.0,
            "right_mono_ratio": 0.0,
        }

        if arc is None or valid is None or len(arc) < 12:
            return base

        n = len(arc)
        beam_deg = RIGHT_SHAPE_ARC_DEG / max(n - 1, 1)

        valid_ratio = float(np.mean(valid)) if len(valid) > 0 else 0.0
        smooth = self.smooth_1d(arc, RIGHT_SHAPE_SMOOTH_WINDOW)

        # 三个值仍然保留在 debug/log 里，但不再用于核心平行判断。
        def sample_at_robot_deg(deg, half_width_deg=6.0):
            mask = np.abs(np.degrees(robot_angles) - deg) <= half_width_deg
            mask = mask & valid
            if not np.any(mask):
                return float("inf")
            return float(np.median(arc[mask]))

        right_back = sample_at_robot_deg(-120.0)
        right_mid = sample_at_robot_deg(-90.0)
        right_front = sample_at_robot_deg(-60.0)

        if valid_ratio < RIGHT_SHAPE_MIN_VALID_RATIO:
            base.update({
                "available": True,
                "valid_ratio": valid_ratio,
                "right_front": right_front,
                "right_mid": right_mid,
                "right_back": right_back,
                "right_min": float(np.min(arc)),
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

        # 边缘距离：如果边缘有少量 invalid，用前/后 12% 区域的有效中位数。
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

        # valley 宽度：距离在谷底附近的一段应有一定角宽。圆柱通常很窄，墙面更宽。
        valley_mask = (smooth <= min_dist + 0.14) & valid
        clusters = find_near_clusters(list(valley_mask))
        valley_width_deg = 0.0
        for c0, c1 in clusters:
            if c0 <= min_idx <= c1:
                valley_width_deg = float((c1 - c0 + 1) * beam_deg)
                break

        diff = np.diff(smooth)

        # 从右后 -> 谷底，应该整体下降；从谷底 -> 右前，应该整体上升。
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

        # 突变比例：平滑墙面不应该出现大量连续 beam 的大跳变。
        # 注意：这是对 smoothed curve 做判断，所以阈值可以比原始 beam 更严格。
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

        # valley_offset 正值表示谷底偏右前方，车头更靠近墙，需要左转；负值则需要右转。
        # 用三点差值做轻微辅助，但核心仍然是整段 V 型曲线。
        three_point_hint = 0.0
        if np.isfinite(right_back) and np.isfinite(right_front):
            three_point_hint = clamp(right_back - right_front, -0.35, 0.35)

        parallel_error = clamp(
            RIGHT_SHAPE_ERROR_SCALE * valley_offset + 0.25 * three_point_hint,
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
        """
        返回右侧墙面几何。核心来自右侧扇形 V 型曲线分析，
        而不是只看 right_front / right_mid / right_back 三个点。
        """
        geom = self.analyze_right_wall_shape()

        if lidar is not None:
            # 碰撞安全仍然使用 min，因为 min 对近距离物体更敏感。
            geom["right_min"] = float(min(lidar.get("right_min", geom["right_min"]), geom["right_min"]))

        return geom

    def is_right_wall_parallel_good(self, lidar):
        geom = self.get_right_wall_geometry(lidar)

        return (
            geom.get("parallel_good", False)
            and lidar["front_min"] > FRONT_STOP_DIST
        ), geom

    def is_right_wall_visible(self, lidar, require_distance_ok=True):
        """
        判断右侧是否已经重新出现连续墙面。

        注意：这里不是要求机器人已经完全平行。
        - parallel_good: 用于“已经贴好且平行”；
        - visible: 只表示右侧多 beam 曲线已经像墙面，可以切回 SEARCH_WALL_FOLLOW，
          后续由 wall-follow controller 慢慢调平行。
        """
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

        # shape_ok 是较严格的 V 型墙面判断；loose_continuous_wall 用于 rejoin 退出，
        # 允许墙面还没完全居中/平行，但必须是连续、多 beam、低突变的右侧结构。
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
        """
        右墙扇形 V 型曲线跟随控制器：
        - 前方危险：优先左转，避免撞墙/凸起；
        - 右侧曲线无效或墙丢失：慢速右转重新找右侧边缘；
        - 右侧太近：左转拉开；
        - 正常：用谷底距离误差 + 谷底偏移误差保持与右墙平行。
        """
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
            return 0.0, SMALL_TURN_SPEED, geom

        if front_min < FRONT_WARN_DIST:
            if front_left_min >= front_right_min:
                return SLOW_FORWARD_SPEED, SMALL_TURN_SPEED, geom
            return SLOW_FORWARD_SPEED, -SMALL_TURN_SPEED, geom

        if (not available) or (not np.isfinite(right_distance)) or right_distance > PROTRUSION_EDGE_LOST_DIST:
            # 右边完全丢失墙/凸起边缘时，慢速右转重新找边缘。
            return SLOW_FORWARD_SPEED, -SMALL_TURN_SPEED, geom

        if right_min < MIN_RIGHT_DIST:
            return SLOW_FORWARD_SPEED, SMALL_TURN_SPEED, geom

        if right_distance > MAX_RIGHT_DIST:
            return SLOW_FORWARD_SPEED, -SMALL_TURN_SPEED, geom

        dist_error = TARGET_RIGHT_DIST - right_distance
        angular = RIGHT_DIST_K * dist_error + RIGHT_PARALLEL_K * parallel_error
        angular = clamp(angular, -SMALL_TURN_SPEED, SMALL_TURN_SPEED)

        # 如果右侧曲线还不是稳定 V 型，降速贴边，避免把柱子/墙角误认为平行墙。
        linear = base_speed
        if (not shape_ok) or abs(parallel_error) > RIGHT_PARALLEL_TOL:
            linear = min(linear, SLOW_FORWARD_SPEED)

        return linear, angular, geom

    def front_wall_reappeared(self, lidar):
        """
        绕过凸起后，如果大面积墙面重新出现在前方，进入 REJOIN_WALL。
        REJOIN_WALL 会通过左转/位移让这面墙回到右侧 LiDAR 区域。
        """
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

    def get_lidar_summary(self):
        front = FRONT_ANGLE
        front_left = FRONT_ANGLE + math.radians(35)
        front_right = FRONT_ANGLE - math.radians(35)
        left = FRONT_ANGLE + math.radians(90)
        right = FRONT_ANGLE - math.radians(90)

        summary = {}

        summary["front_min"] = self.get_arc_min(front, 50)
        summary["front_left_min"] = self.get_arc_min(front_left, 35)
        summary["front_right_min"] = self.get_arc_min(front_right, 35)
        summary["left_min"] = self.get_arc_min(left, 45)
        summary["right_min"] = self.get_arc_min(right, 45)

        summary["front_pattern"], summary["front_info"] = self.classify_arc_pattern(front, 60)
        summary["front_left_pattern"], summary["front_left_info"] = self.classify_arc_pattern(front_left, 40)
        summary["front_right_pattern"], summary["front_right_info"] = self.classify_arc_pattern(front_right, 40)

        summary["scan_change"] = self.get_scan_change_summary()
        summary["right_wall_geom"] = self.get_right_wall_geometry(summary)

        return summary

    # =========================
    # 返回阶段专用 LiDAR / goal 工具
    # =========================

    def get_goal_angle_robot(self, goal_x=0.0, goal_y=0.0):
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
                robot_angle,
                arc_deg=OPENING_ARC_DEG,
                safe_dist=RETURN_SAFE_DIST
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
        """重置卡住检测参考点，避免刚切换状态时用旧参考点误判。"""
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
        """前 60s 内、且已经走出起点一定距离时，返回 True。"""
        if self.score_monotonic_start_time is None:
            return False
        if time.time() - self.score_monotonic_start_time >= SCORE_MONOTONIC_DURATION:
            return False
        if self.last_score < SCORE_MONOTONIC_MIN_ACTIVATION:
            return False
        return True

    def compute_score_derivative(self):
        """
        沿当前 yaw 前进时 score = |x|+|y| 的变化率（单位速度）。
        dscore/dt ≈ sign(x)*cos(yaw) + sign(y)*sin(yaw)
        > 0 = 远离原点，< 0 = 靠近原点。
        """
        sx = 0.0 if abs(self.x) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.x)
        sy = 0.0 if abs(self.y) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.y)
        return sx * math.cos(self.yaw) + sy * math.sin(self.yaw)

    def choose_score_increasing_turn(self):
        """
        选择原地旋转方向，使 dscore/dt 尽快变大（即车头转向远离原点的方向）。
        d(dscore)/d(yaw) = -sign(x)*sin(yaw) + sign(y)*cos(yaw)
        > 0 → 左转让 dscore 增大；< 0 → 右转更好。
        """
        sx = 0.0 if abs(self.x) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.x)
        sy = 0.0 if abs(self.y) < SCORE_SIGN_DEADZONE else math.copysign(1.0, self.y)
        d_dscore_dyaw = -sx * math.sin(self.yaw) + sy * math.cos(self.yaw)
        return SMALL_TURN_SPEED if d_dscore_dyaw >= 0.0 else -SMALL_TURN_SPEED

    def score_monotonic_gate(self, linear, angular, mode="normal"):
        """
        前 60s 内对线速度做门控。三个条件同时满足时才锁：
          1. score 比 best_score 下降超过容差
          2. linear > 0（正在前进）
          3. dscore < -eps（车头方向会继续让 score 减小）
        被锁时 linear=0，angular 替换为朝 score 增大方向的原地转向。
        返回 (linear, angular, gated)。
        """
        if not self.score_monotonic_active():
            return linear, angular, False

        tol = SCORE_MONOTONIC_TOL_AVOID if mode == "avoid" else SCORE_MONOTONIC_TOL_NORMAL

        if self.last_score >= (self.best_score - tol):
            return linear, angular, False
        if linear <= 0.0:
            return linear, angular, False
        if self.compute_score_derivative() >= -SCORE_MONOTONIC_DIR_EPS:
            return linear, angular, False

        return 0.0, self.choose_score_increasing_turn(), True

    def relax_best_score_after_detour(self):
        """
        避障/脱困后回到 SEARCH_WALL_FOLLOW 时调用。
        将 best_score 降级为当前 score，防止 normal_tol（0.12）
        在绕障造成的 score 下降未恢复时立刻锁住前进。
        """
        if not self.score_monotonic_active():
            return
        old_best = self.best_score
        self.best_score = self.last_score
        if old_best != self.best_score:
            self.get_logger().info(
                f"SCORE_GUARD: relax best_score {old_best:.2f} -> {self.best_score:.2f} after detour"
            )

    def avoid_exit_yaw_ok(self):
        """
        绕障退出方向保护：当前 yaw 是否和进入绕障时的 yaw 偏差在容差内。
        防止绕障中途误判"右墙重新出现"导致带着反方向退出。
        """
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
        """在 SEARCHING 和 RETURNING 相关状态中统一处理卡住检测。"""
        if self.state not in STUCK_MONITORED_STATES:
            return False

        if not self.check_stuck():
            return False

        self.escape_resume_state = "RETURNING" if self.state == "RETURNING" else "SEARCH_WALL_FOLLOW"
        self.get_logger().warn(
            f"Stuck detected in {self.state}: moved less than {STUCK_MOVE_DIST:.3f} m "
            f"within {STUCK_TIME:.1f} s. Enter ESCAPE_BACKUP, then resume {self.escape_resume_state}."
        )
        self.stop_robot()
        self.set_state("ESCAPE_BACKUP")
        return True

    # =========================
    # SEARCHING 阶段：右墙平行跟随 + 360° LiDAR 差分凸起检测
    # =========================

    def search_wall_follow(self, lidar):
        front_min = lidar["front_min"]
        front_pattern = lidar["front_pattern"]
        scan_change = lidar.get("scan_change", {})

        if self.red_detected or self.target_locked:
            self.stop_robot()
            self.set_state("REPORTING")
            return

        # 当前/上一时刻 360° LiDAR 差分：前方突然变近，说明有新凸起/障碍进入路径。
        dynamic_front_protrusion = (
            scan_change.get("available", False)
            and scan_change.get("front_closer", False)
            and scan_change.get("front_changed_ratio", 0.0) > 0.035
            and front_min < CHANGE_NEAR_DIST
        )

        # 独立障碍物不再执行固定时间绕行，而是当作右墙凸起，进入贴边模式。
        if front_pattern == "obstacle" or dynamic_front_protrusion:
            self.save_debug_event(
                "PROTRUSION",
                lidar,
                "front_obstacle_or_360deg_change_enter_edge_follow"
            )
            self.get_logger().info(
                "Front protrusion detected by pattern/change. Enter AVOID_OBSTACLE edge-follow mode."
            )
            self.rejoin_stable_count = 0
            self.avoid_entry_yaw = self.yaw
            self.set_state("AVOID_OBSTACLE")
            return

        # 前方如果是大面积墙/墙角，右墙策略下优先左转，让前方墙面逐渐转移到右侧。
        if front_pattern in ["wall", "corner_or_wall"] or front_min < FRONT_STOP_DIST:
            if front_pattern == "wall":
                self.save_debug_event("WALL", lidar, "front_wall_turn_left_to_keep_wall_on_right")
            elif front_pattern == "corner_or_wall":
                self.save_debug_event("CORNER_OR_WALL", lidar, "front_corner_turn_left_to_keep_wall_on_right")

            if front_min < FRONT_STOP_DIST:
                linear, angular = 0.0, SMALL_TURN_SPEED
            else:
                linear, angular = SLOW_FORWARD_SPEED, SMALL_TURN_SPEED
            linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="normal")
            self.publish_cmd(linear, angular)
            return

        linear, angular, _ = self.compute_right_wall_follow_cmd(lidar, base_speed=FORWARD_SPEED)
        linear, angular, gated = self.score_monotonic_gate(linear, angular, mode="normal")
        if gated and time.time() - self.last_log_time > 0.8:
            self.get_logger().info(
                f"SCORE_GUARD(search): block forward. "
                f"score={self.last_score:.2f} best={self.best_score:.2f} "
                f"dscore={self.compute_score_derivative():+.2f}"
            )
        self.publish_cmd(linear, angular)

    def avoid_obstacle(self, lidar):
        """
        非定时绕障：把障碍物视为右墙的凸起，沿着凸起边缘继续右墙跟随。

        退出不看时间：
        1. 前方重新出现大面积墙面 -> REJOIN_WALL；
        2. 右侧多 beam 已经重新形成连续墙面 -> SEARCH_WALL_FOLLOW。
        第二点不再要求“完全平行”，因为平行调整应交给 SEARCH_WALL_FOLLOW 持续完成。
        """
        if self.red_detected or self.target_locked:
            self.stop_robot()
            self.set_state("REPORTING")
            return

        front_min = lidar["front_min"]
        front_right_min = lidar["front_right_min"]
        right_min = lidar["right_min"]

        # 绕过凸起后，原来的墙面/走廊面通常会先出现在前方。
        # 这时进入 REJOIN_WALL，通过左转/位移让墙回到右侧区域。
        if self.front_wall_reappeared(lidar) and front_min < PROTRUSION_FRONT_WARN:
            self.get_logger().info("Front wall shape reappeared after protrusion. Enter REJOIN_WALL.")
            self.rejoin_stable_count = 0
            self.set_state("REJOIN_WALL")
            return

        visible, geom = self.is_right_wall_visible(lidar, require_distance_ok=True)
        if visible and front_min > FRONT_STOP_DIST:
            self.rejoin_stable_count += 1
        else:
            self.rejoin_stable_count = 0

        if self.rejoin_stable_count >= RIGHT_WALL_VISIBLE_STABLE_COUNT:
            if self.avoid_exit_yaw_ok():
                self.get_logger().info(
                    "Right wall visible again after protrusion. Back to SEARCH_WALL_FOLLOW; "
                    "parallel tuning continues in follow mode."
                )
                self.rejoin_stable_count = 0
                self.relax_best_score_after_detour()
                self.set_state("SEARCH_WALL_FOLLOW")
                return
            else:
                # 方向偏差太大，不退出，重置计数继续绕。
                self.rejoin_stable_count = 0

        # 太近时原地左转，避免前脸撞上凸起（linear=0，gate 不会触发）。
        if front_min < PROTRUSION_FRONT_DANGER:
            self.publish_cmd(0.0, SMALL_TURN_SPEED)
            return

        # 右前太近时说明正在擦凸起边缘，左转拉开。
        if front_right_min < MIN_RIGHT_DIST or right_min < MIN_RIGHT_DIST:
            linear, angular = SLOW_FORWARD_SPEED, SMALL_TURN_SPEED
            linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
            self.publish_cmd(linear, angular)
            return

        # 边缘丢失时，慢速右转重新贴回凸起/右墙边缘。
        if right_min > PROTRUSION_EDGE_LOST_DIST:
            linear, angular = SLOW_FORWARD_SPEED, -SMALL_TURN_SPEED
            linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
            self.publish_cmd(linear, angular)
            return

        # 默认继续右墙多 beam 跟随，但降速。
        linear, angular, _ = self.compute_right_wall_follow_cmd(lidar, base_speed=SLOW_FORWARD_SPEED)
        linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
        self.publish_cmd(linear, angular)

    def rejoin_wall(self, lidar):
        """
        绕过凸起后，当前墙面通常先出现在前方。
        REJOIN_WALL 的任务是根据 LiDAR 几何让这面墙移动到机器人右侧。

        退出条件已放宽：
        - 右侧多 beam 重新出现连续墙面；
        - 右侧距离在安全/可跟随范围内；
        - 前方没有直接碰撞风险。

        不再要求在 REJOIN_WALL 内已经完全平行；平行调整由 SEARCH_WALL_FOLLOW 继续完成。
        """
        if self.red_detected or self.target_locked:
            self.stop_robot()
            self.set_state("REPORTING")
            return

        front_min = lidar["front_min"]
        right_min = lidar["right_min"]

        visible, geom = self.is_right_wall_visible(lidar, require_distance_ok=True)
        if visible and front_min > FRONT_STOP_DIST:
            self.rejoin_stable_count += 1
        else:
            self.rejoin_stable_count = 0

        if self.rejoin_stable_count >= RIGHT_WALL_VISIBLE_STABLE_COUNT:
            if self.avoid_exit_yaw_ok():
                self.get_logger().info(
                    f"REJOIN complete: right wall visible | "
                    f"right_distance={geom.get('right_distance', float('inf')):.2f}, "
                    f"valley_angle={geom.get('valley_angle_deg', 0.0):.1f}, "
                    f"shape_ok={geom.get('shape_ok', False)}, "
                    f"parallel_good={geom.get('parallel_good', False)}"
                )
                self.rejoin_stable_count = 0
                self.relax_best_score_after_detour()
                self.set_state("SEARCH_WALL_FOLLOW")
                return
            else:
                self.rejoin_stable_count = 0

        # 前方墙很近：原地左转（linear=0，gate 不触发）。
        if front_min < FRONT_STOP_DIST:
            self.publish_cmd(0.0, SMALL_TURN_SPEED)
            return

        # 前方仍有墙面形状：慢速前进 + 左转，等它转移到右侧。
        if self.front_wall_reappeared(lidar):
            linear, angular = SLOW_FORWARD_SPEED, SMALL_TURN_SPEED
            linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
            self.publish_cmd(linear, angular)
            return

        # 如果右侧还没有墙，慢速右转找回右边缘。
        if right_min > RIGHT_WALL_VISIBLE_MAX_DIST:
            linear, angular = SLOW_FORWARD_SPEED, -SMALL_TURN_SPEED
            linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
            self.publish_cmd(linear, angular)
            return

        # 如果右侧太近，左转拉开。
        if right_min < MIN_RIGHT_DIST:
            linear, angular = SLOW_FORWARD_SPEED, SMALL_TURN_SPEED
            linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
            self.publish_cmd(linear, angular)
            return

        # 右侧已有墙但还不够平行：用右墙多 beam 控制微调。
        linear, angular, _ = self.compute_right_wall_follow_cmd(lidar, base_speed=SLOW_FORWARD_SPEED)
        linear, angular, _ = self.score_monotonic_gate(linear, angular, mode="avoid")
        self.publish_cmd(linear, angular)

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
            if front_left_min >= front_right_min:
                self.publish_cmd(0.0, TURN_SPEED)
            else:
                self.publish_cmd(0.0, -TURN_SPEED)
            return

        self.reset_stuck_reference()

        resume_state = self.escape_resume_state
        self.escape_resume_state = "SEARCH_WALL_FOLLOW"
        # 脱困后 score 可能已经偏离 best，relax 避免立即被 gate 锁住。
        self.relax_best_score_after_detour()
        self.set_state(resume_state)

    # =========================
    # SEARCHING 阶段安全路径记录
    # =========================

    def record_safe_path(self):
        """只在 SEARCHING 相关状态记录安全 waypoint。RETURNING 阶段绝不写入路径。"""
        if not self.have_odom or not self.start_recorded:
            return

        if self.target_locked or self.red_detected:
            return

        if self.state not in ["SEARCH_WALL_FOLLOW", "AVOID_OBSTACLE", "REJOIN_WALL"]:
            return

        if not self.safe_path:
            self.safe_path.append((self.x, self.y))
            self.path_record_last_x = self.x
            self.path_record_last_y = self.y
            self.path_record_last_yaw = self.yaw
            return

        moved = math.hypot(self.x - self.path_record_last_x, self.y - self.path_record_last_y)
        yaw_changed = abs(normalize_angle(self.yaw - self.path_record_last_yaw))

        if moved >= PATH_RECORD_MIN_DIST or yaw_changed >= math.radians(PATH_RECORD_MIN_YAW_DEG):
            self.safe_path.append((self.x, self.y))
            self.path_record_last_x = self.x
            self.path_record_last_y = self.y
            self.path_record_last_yaw = self.yaw

            if len(self.safe_path) > MAX_SAFE_PATH_POINTS:
                # 保留起点，删除中间最旧点。
                self.safe_path = [self.safe_path[0]] + self.safe_path[-(MAX_SAFE_PATH_POINTS - 1):]

    def prepare_return_path(self):
        """检测到目标后，冻结搜索阶段安全路径，并设置倒序返回 index。"""
        if self.return_initialized:
            return

        if not self.safe_path:
            self.safe_path = [(0.0, 0.0)]

        # 冻结一份 return_path。之后 RETURNING 阶段不再修改这份路径，
        # 避免把返回阶段新走出的轨迹又加入路径，造成绕圈或路径污染。
        self.return_path = list(self.safe_path)

        last_x, last_y = self.return_path[-1]
        if math.hypot(self.x - last_x, self.y - last_y) > 0.08:
            self.return_path.append((self.x, self.y))

        self.return_path_index = len(self.return_path) - 1

        # 如果末端 waypoint 基本就是当前位置，先追踪前一个点。
        if self.return_path_index > 0:
            tx, ty = self.return_path[self.return_path_index]
            if math.hypot(tx - self.x, ty - self.y) < WAYPOINT_REACHED_DIST:
                self.return_path_index -= 1

        self.return_initialized = True
        self.return_escape_phase = "IDLE"
        self.return_blocked_count = 0
        self.return_goal_clear_count = 0

        self.get_logger().info(
            f"Return path frozen: search_points={len(self.safe_path)}, "
            f"return_points={len(self.return_path)}, start_index={self.return_path_index}"
        )

    def get_return_target(self):
        """返回当前要追踪的倒序 waypoint。到达后自动切到上一个 waypoint。"""
        if self.return_path_index is None or not self.return_path:
            return 0.0, 0.0, "ORIGIN"

        # 已经接近起点时，直接追踪 (0,0)，避免在起点附近抖动。
        if math.hypot(self.x, self.y) < RETURN_FINAL_DIRECT_DIST:
            self.return_path_index = 0
            return 0.0, 0.0, "ORIGIN_FINAL"

        while self.return_path_index > 0:
            tx, ty = self.return_path[self.return_path_index]
            if math.hypot(tx - self.x, ty - self.y) <= WAYPOINT_REACHED_DIST:
                self.return_path_index -= 1
            else:
                break

        if self.return_path_index <= 0:
            return 0.0, 0.0, "ORIGIN"

        tx, ty = self.return_path[self.return_path_index]
        return tx, ty, f"PATH[{self.return_path_index}]"

    def get_angle_to_point_robot(self, tx, ty):
        dx = tx - self.x
        dy = ty - self.y
        target_yaw_global = math.atan2(dy, dx)
        return normalize_angle(target_yaw_global - self.yaw)

    def choose_return_escape_direction(self, lidar, desired_angle):
        """选择短时绕障方向。优先朝更空的一侧，若差不多则偏向 waypoint 所在方向。"""
        fl = lidar["front_left_min"]
        fr = lidar["front_right_min"]
        left = lidar["left_min"]
        right = lidar["right_min"]

        left_score = fl + 0.45 * left
        right_score = fr + 0.45 * right

        if abs(left_score - right_score) > 0.08:
            return 1.0 if left_score > right_score else -1.0

        return 1.0 if desired_angle >= 0.0 else -1.0

    # =========================
    # REPORTING 阶段
    # =========================

    def reporting(self):
        self.stop_robot()

        if not self.saved_detection:
            self.save_detection_evidence()
            self.saved_detection = True

        self.prepare_return_path()
        self.set_state("RETURNING")

    # =========================
    # RETURNING 阶段
    # =========================

    def returning(self, lidar):
        """
        RETURNING 阶段（沿墙返回版）：
        与 SEARCHING 阶段使用完全相同的右墙跟随 + 凸起绕行逻辑，
        直到 odom 距离原点 (0,0) 足够近为止。

        改动重点：
        - 取消 waypoint 追踪，不再使用 frozen return_path / get_return_target；
        - 不再依赖 desired_angle / target_bias 偏置；
        - 行为完全模仿 search_wall_follow + avoid_obstacle（凸起绕行内联处理，
          不离开 RETURNING 状态）；
        - 唯一退出条件：dist_to_origin <= RETURN_STOP_DIST -> 直接进入 RETURN_FINAL_TURN。
        """
        now = time.time()
        dist_to_origin = math.hypot(self.x, self.y)

        # 唯一退出条件：到达原点附近。直接进入 180° 转向。
        if dist_to_origin <= RETURN_STOP_DIST:
            self.stop_robot()
            self.get_logger().info(
                f"Returned close to start. x={self.x:.3f}, y={self.y:.3f}, dist={dist_to_origin:.3f}. "
                "Starting final 180 deg turn before DONE."
            )
            self.start_post_return_turn()
            return

        front_min = lidar["front_min"]
        front_right_min = lidar["front_right_min"]
        right_min = lidar["right_min"]
        front_pattern = lidar["front_pattern"]
        scan_change = lidar.get("scan_change", {})

        # =========================
        # 与 search_wall_follow 完全相同的凸起检测
        # =========================
        dynamic_front_protrusion = (
            scan_change.get("available", False)
            and scan_change.get("front_closer", False)
            and scan_change.get("front_changed_ratio", 0.0) > 0.035
            and front_min < CHANGE_NEAR_DIST
        )

        # 凸起或独立障碍物：内联使用 avoid_obstacle 的贴边逻辑，但不切换状态。
        if front_pattern == "obstacle" or dynamic_front_protrusion:
            if front_min < PROTRUSION_FRONT_DANGER:
                # 前脸太近：原地左转避撞。
                self.publish_cmd(0.0, SMALL_TURN_SPEED)
            elif front_right_min < MIN_RIGHT_DIST or right_min < MIN_RIGHT_DIST:
                # 右前/右侧太近：擦边，左转拉开。
                self.publish_cmd(SLOW_FORWARD_SPEED, SMALL_TURN_SPEED)
            elif right_min > PROTRUSION_EDGE_LOST_DIST:
                # 边缘丢失：慢速右转贴回。
                self.publish_cmd(SLOW_FORWARD_SPEED, -SMALL_TURN_SPEED)
            else:
                linear, angular, _ = self.compute_right_wall_follow_cmd(
                    lidar, base_speed=SLOW_FORWARD_SPEED
                )
                self.publish_cmd(linear, angular)

            if now - self.last_return_log_time > 0.6:
                self.get_logger().info(
                    f"RETURN PROTRUSION | dist_origin={dist_to_origin:.3f} | "
                    f"front={front_min:.2f} FR={front_right_min:.2f} right={right_min:.2f} | "
                    f"pattern={front_pattern} dyn={dynamic_front_protrusion}"
                )
                self.last_return_log_time = now
            return

        # 前方是大面积墙/墙角：和 search_wall_follow 一样左转把墙转移到右侧。
        if front_pattern in ["wall", "corner_or_wall"] or front_min < FRONT_STOP_DIST:
            if front_min < FRONT_STOP_DIST:
                self.publish_cmd(0.0, SMALL_TURN_SPEED)
            else:
                self.publish_cmd(SLOW_FORWARD_SPEED, SMALL_TURN_SPEED)

            if now - self.last_return_log_time > 0.6:
                self.get_logger().info(
                    f"RETURN WALL_TURN_LEFT | dist_origin={dist_to_origin:.3f} | "
                    f"front={front_min:.2f} pattern={front_pattern}"
                )
                self.last_return_log_time = now
            return

        # 默认：和 search_wall_follow 完全相同的多 beam 右墙跟随。
        linear, angular, geom = self.compute_right_wall_follow_cmd(
            lidar, base_speed=FORWARD_SPEED
        )
        self.publish_cmd(linear, angular)

        if now - self.last_return_log_time > 0.6:
            self.get_logger().info(
                f"RETURN WALL_FOLLOW | dist_origin={dist_to_origin:.3f} | "
                f"front={front_min:.2f} right={right_min:.2f} | "
                f"right_dist={geom.get('right_distance', float('inf')):.2f} "
                f"shape_ok={geom.get('shape_ok', False)} | "
                f"cmd=({linear:.2f}, {angular:.2f})"
            )
            self.last_return_log_time = now

    def start_post_return_turn(self):
        """返回原点后、DONE 前：记录起始 yaw，直接进入 180° 转向。"""
        self.post_return_turn_start_yaw = self.yaw
        self.stop_robot()
        self.set_state("RETURN_FINAL_TURN")

    def return_final_turn(self):
        """前进 5 cm 后原地转 180°，然后进入 DONE。"""
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
    # 主循环
    # =========================

    def control_loop(self):
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
                f"cluster_deg={front_info.get('cluster_deg', 0.0):.1f} "
                f"width={front_info.get('physical_width', 0.0):.2f} | "
                f"right_dist={right_geom.get('right_distance', float('inf')):.2f} "
                f"valley_angle={right_geom.get('valley_angle_deg', 0.0):.1f} "
                f"shape_ok={right_geom.get('shape_ok', False)} "
                f"parallel_err={right_geom.get('parallel_error', 0.0):.2f} | "
                f"change_front={scan_change.get('front_changed_ratio', 0.0):.2f} "
                f"front_closer={scan_change.get('front_closer', False)} "
                f"beams={scan_change.get('beam_count', 0)} | "
                f"red={self.red_pixels} "
                f"confirm={self.red_seen_count}/{RED_CONFIRM_FRAMES} "
                f"locked={self.target_locked}"
            )
            self.last_log_time = now

        # SEARCHING / RETURNING 统一卡住检测。触发后本轮不再执行原状态控制。
        if self.handle_stuck_if_needed():
            return

        # 搜索阶段记录已经实际走过的安全路径，供 RETURNING 倒序跟踪。
        self.record_safe_path()

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

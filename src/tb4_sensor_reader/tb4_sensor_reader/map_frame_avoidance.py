import io
import math
import time

import rclpy
import yaml
from geometry_msgs.msg import Twist
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

NAMESPACE = "/T21"
NODE_NAME = "phase2_autonomous"

START_SIDE = "right"  # "right" 或 "left"
WALL_SIGN = 1.0 if START_SIDE == "right" else -1.0

SAVE_DIR = os.path.expanduser("~/ros2_ws/tb4_phase2_evidence")
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
        super().__init__('map_frame_avoidance')

        self.cmd_pub = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, f'{NAMESPACE}/odom', self.odom_callback, 10)
        self.create_subscription(CompressedImage, f'{NAMESPACE}/oakd/rgb/image_raw/compressed', self.image_callback, 10)

        self.origin_set = False
        self.origin_x = self.origin_y = self.origin_yaw = 0.0
        self.current_x = self.current_y = self.current_yaw = 0.0

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

        self.last_camera_image = None
        self.red_pixels = 0
        self.red_detected = False
        self.red_pixels = 0
        self.saved_detection = False

        self.state = MissionState.SEARCHING
        self.detected_local_xy = None

        self.cube_robot_x = None
        self.cube_robot_y = None
        self.cube_global_x = None
        self.cube_global_y = None
        self.cube_distance = None

        self.viewer_root = tk.Tk()
        self.viewer_root.title('tb4 camera viewer')
        self.viewer_label = tk.Label(self.viewer_root)
        self.viewer_label.pack()
        self.viewer_photo = None

        self.load_phase1_map_artifacts()
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Autonomous mission node started (Pillow/Tk viewer)')

    def load_phase1_map_artifacts(self):
        root = Path.cwd()
        targets = [root / f'{MAP_BASENAME}{ext}' for ext in ('.pgm', '.yaml', '.posegraph', '.data')]
        if not all(p.exists() for p in targets):
            self.get_logger().warn('Phase-1 map artifacts not fully found in current directory')
            return
        with (root / f'{MAP_BASENAME}.yaml').open('r', encoding='utf-8') as f:
            meta = yaml.safe_load(f)
        self.get_logger().info(f"Loaded map metadata: resolution={meta.get('resolution')}, origin={meta.get('origin')}")

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
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if not self.origin_set:
            self.origin_x, self.origin_y, self.origin_yaw = px, py, yaw
            self.origin_set = True
            self.get_logger().info('Origin established at first odometry sample')
        self.current_x, self.current_y, self.current_yaw = px, py, yaw

    def scan_callback(self, msg: LaserScan):
        n = len(msg.ranges)
        inc = msg.angle_increment
        front_i = int(round(((-math.pi / 2) - msg.angle_min) / inc)) % n
        side_a = int(round(math.radians(90) / inc))
        half = int(round(math.radians(30) / inc))

        def arc_min(center):
            vals = []
            for k in range(-half, half + 1):
                i = (center + k) % n
                r = msg.ranges[i]
                if msg.range_min < r < msg.range_max:
                    vals.append(r)
            return min(vals) if vals else float('inf')

        self.front_min = arc_min(front_i)
        self.left_min = arc_min((front_i + side_a) % n)
        self.right_min = arc_min((front_i - side_a) % n)

    def image_callback(self, msg: CompressedImage):
        image = Image.open(io.BytesIO(bytes(msg.data))).convert('RGB')
        self.last_camera_image = image

        thumb = image.resize((160, 120))
        self.red_pixels = self.count_red_pixels(thumb)
        self.red_detected = self.red_pixels >= RED_MIN_PIXELS

        display = image.resize((640, 360))
        self.viewer_photo = ImageTk.PhotoImage(display)
        self.viewer_label.config(image=self.viewer_photo)
        self.viewer_root.update_idletasks()
        self.viewer_root.update()

    @staticmethod
    def count_red_pixels(image):
        red = 0
        for r, g, b in image.getdata():
            if r > 130 and g < 90 and b < 90 and r > g * 1.3 and r > b * 1.3:
                red += 1
        return red

    def local_xy(self):
        dx = self.current_x - self.origin_x
        dy = self.current_y - self.origin_y
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def transition(self, new_state):
        if self.state != new_state:
            self.get_logger().info(f'{self.state.value} -> {new_state.value}')
            self.state = new_state

    def save_detection_evidence(self):
        if self.last_camera_image is None:
            return
        out = Path.cwd() / f'red_cube_evidence_{int(time.time())}.png'
        self.last_camera_image.save(out)
        self.get_logger().info(f'Saved detection screenshot: {out}')

    def searching_control(self):
        cmd = Twist()
        if self.front_min <= AVOID_DISTANCE:
            cmd.angular.z = TURN_SPEED if self.left_min >= self.right_min else -TURN_SPEED
            return cmd
        elapsed = time.time() - self.sweep_phase_started
        if elapsed <= SWEEP_LEG_SECONDS:
            cmd.linear.x = FORWARD_SPEED
        elif elapsed <= SWEEP_LEG_SECONDS + SWEEP_TURN_SECONDS:
            cmd.angular.z = TURN_SPEED if (self.sweep_leg_idx % 2 == 0) else -TURN_SPEED
        else:
            self.sweep_phase_started = time.time()
            self.sweep_leg_idx += 1
        return cmd

    def returning_control(self):
        cmd = Twist()
        lx, ly = self.local_xy()
        dist = math.hypot(lx, ly)
        if dist <= GOAL_TOLERANCE_M:
            self.transition(MissionState.DONE)
            return cmd
        target_heading = math.atan2(-ly, -lx)
        heading_error = self.normalize_angle(target_heading - (self.current_yaw - self.origin_yaw))
        if abs(heading_error) > TURN_IN_PLACE_THRESHOLD:
            cmd.angular.z = max(-TURN_SPEED, min(TURN_SPEED, RETURN_HEADING_GAIN * heading_error))
            return cmd
        cmd.linear.x = min(RETURN_MAX_FORWARD, 0.10 + 0.25 * dist)
        cmd.angular.z = max(-TURN_SPEED, min(TURN_SPEED, RETURN_HEADING_GAIN * heading_error))
        return cmd

        if result is None:
            return None

        self.evidence_snapshot_count += 1
        self.evidence_last_snapshot_time = now

        if self.state == MissionState.SEARCHING and self.red_detected:
            lx, ly = self.local_xy()
            self.get_logger().info(f'Red cube detected at odom local x={lx:.3f}, y={ly:.3f}')
            self.save_detection_evidence()
            self.transition(MissionState.REPORTING)

        cmd = Twist()
        if self.state == MissionState.SEARCHING:
            cmd = self.searching_control()
        elif self.state == MissionState.REPORTING:
            self.transition(MissionState.RETURNING)
        elif self.state == MissionState.RETURNING:
            cmd = self.returning_control()
        self.cmd_pub.publish(cmd)

        with open(self.evidence_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

def main(args=None):
    rclpy.init(args=args)
    node = MapFrameAvoidance()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.viewer_root.destroy()
        except Exception:
            pass
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

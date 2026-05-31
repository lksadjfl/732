import io
import math
import time

import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image, ImageTk
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan
import tkinter as tk

NAMESPACE = '/T13'
FORWARD_SPEED = 0.16
TURN_SPEED = 0.7
AVOID_DISTANCE = 0.55
RED_MIN_PIXELS = 500
GOAL_TOLERANCE_M = 0.25
RETURN_HEADING_GAIN = 1.2
RETURN_MAX_FORWARD = 0.16
TURN_IN_PLACE_THRESHOLD = 0.5
SWEEP_LEG_SECONDS = 8.0
SWEEP_TURN_SECONDS = 2.0
MAP_BASENAME = 'lab_map'


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

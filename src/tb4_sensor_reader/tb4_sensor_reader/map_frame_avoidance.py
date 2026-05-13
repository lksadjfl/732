import io
import math
import time
from enum import Enum
from pathlib import Path

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


class MissionState(Enum):
    SEARCHING = 'SEARCHING'
    REPORTING = 'REPORTING'
    RETURNING = 'RETURNING'
    DONE = 'DONE'


class MapFrameAvoidance(Node):
    def __init__(self):
        super().__init__('map_frame_avoidance')

        self.cmd_pub = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, f'{NAMESPACE}/odom', self.odom_callback, 10)
        self.create_subscription(CompressedImage, f'{NAMESPACE}/oakd/rgb/image_raw/compressed', self.image_callback, 10)

        self.origin_set = False
        self.origin_x = self.origin_y = self.origin_yaw = 0.0
        self.current_x = self.current_y = self.current_yaw = 0.0

        self.front_min = float('inf')
        self.left_min = float('inf')
        self.right_min = float('inf')

        self.last_camera_image = None
        self.red_pixels = 0
        self.red_detected = False

        self.state = MissionState.SEARCHING
        self.detected_local_xy = None

        self.sweep_phase_started = time.time()
        self.sweep_leg_idx = 0

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

    def odom_callback(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
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

    @staticmethod
    def normalize_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def control_loop(self):
        if not self.origin_set:
            return

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
        rclpy.shutdown()


if __name__ == '__main__':
    main()

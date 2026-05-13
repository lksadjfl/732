import math
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan

NAMESPACE = '/T13'

FORWARD_SPEED = 0.16
TURN_SPEED = 0.7
AVOID_DISTANCE = 0.55
RED_MIN_PIXELS = 650

GOAL_TOLERANCE_M = 0.25
RETURN_HEADING_GAIN = 1.2
RETURN_MAX_FORWARD = 0.16
TURN_IN_PLACE_THRESHOLD = 0.5

SWEEP_LEG_SECONDS = 8.0
SWEEP_TURN_SECONDS = 2.0

MAP_BASENAME = 'lab_map'

RED_LOW1 = np.array([0, 150, 100])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 150, 100])
RED_HIGH2 = np.array([180, 255, 255])


class MissionState(Enum):
    SEARCHING = 'SEARCHING'
    REPORTING = 'REPORTING'
    RETURNING = 'RETURNING'
    DONE = 'DONE'


class MapFrameAvoidance(Node):
    def __init__(self):
        super().__init__('map_frame_avoidance')
        self.bridge = CvBridge()

        self.cmd_pub = self.create_publisher(Twist, f'{NAMESPACE}/cmd_vel', 10)
        self.create_subscription(LaserScan, f'{NAMESPACE}/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, f'{NAMESPACE}/odom', self.odom_callback, 10)
        self.create_subscription(
            CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback,
            10,
        )

        self.origin_set = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.front_min = float('inf')
        self.left_min = float('inf')
        self.right_min = float('inf')

        self.last_camera_frame = None
        self.red_pixels = 0
        self.red_detected = False

        self.state = MissionState.SEARCHING
        self.detected_local_xy = None
        self.evidence_path = None

        self.sweep_phase_started = time.time()
        self.sweep_leg_idx = 0

        self.load_phase1_map_artifacts()

        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Autonomous mission node started')

    def load_phase1_map_artifacts(self):
        root = Path.cwd()
        targets = {
            'pgm': root / f'{MAP_BASENAME}.pgm',
            'yaml': root / f'{MAP_BASENAME}.yaml',
            'posegraph': root / f'{MAP_BASENAME}.posegraph',
            'data': root / f'{MAP_BASENAME}.data',
        }

        missing = [k for k, p in targets.items() if not p.exists()]
        if missing:
            self.get_logger().warn(
                f'Phase-1 map artifacts missing ({missing}). '
                'Node will still run using live sensing.'
            )
            return

        with targets['yaml'].open('r', encoding='utf-8') as f:
            meta = yaml.safe_load(f)
        self.get_logger().info(
            f"Loaded map metadata: resolution={meta.get('resolution')}, origin={meta.get('origin')}"
        )

    def odom_callback(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        if not self.origin_set:
            self.origin_x = px
            self.origin_y = py
            self.origin_yaw = yaw
            self.origin_set = True
            self.get_logger().info('Origin established at first odometry sample')

        self.current_x = px
        self.current_y = py
        self.current_yaw = yaw

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
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.last_camera_frame = img

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2),
        )

        self.red_pixels = int(cv2.countNonZero(mask))
        self.red_detected = self.red_pixels >= RED_MIN_PIXELS

        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]
        cv2.putText(overlay, f'State: {self.state.value}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)
        cv2.putText(overlay, f'Red pixels: {self.red_pixels}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)
        cv2.imshow('camera_detector', overlay)
        cv2.waitKey(1)

    def local_xy(self):
        dx = self.current_x - self.origin_x
        dy = self.current_y - self.origin_y
        c = math.cos(self.origin_yaw)
        s = math.sin(self.origin_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def transition(self, new_state: MissionState):
        if self.state != new_state:
            self.get_logger().info(f'{self.state.value} -> {new_state.value}')
            self.state = new_state

    def save_detection_evidence(self):
        if self.last_camera_frame is None:
            return
        stamp = int(time.time())
        out = Path.cwd() / f'red_cube_evidence_{stamp}.png'
        cv2.imwrite(str(out), self.last_camera_frame)
        self.evidence_path = out
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
            self.detected_local_xy = (lx, ly)
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
        elif self.state == MissionState.DONE:
            pass

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MapFrameAvoidance()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

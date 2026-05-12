import csv
import math
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan

NAMESPACE = '/T13'
FORWARD_SPEED = 0.15
TURN_SPEED = 0.6
AVOID_DISTANCE = 0.6
MIN_RED_PIXELS = 500
MAP_SAVE_INTERVAL_S = 2.0

RED_LOW1 = np.array([0, 150, 100])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 150, 100])
RED_HIGH2 = np.array([180, 255, 255])


class MapFrameAvoidance(Node):
    """Build a simple local map, establish a local frame, avoid obstacles, and stop at red cube."""

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

        self.nearest_front = float('inf')
        self.red_detected = False

        self.origin_set = False
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.map_points = []
        self.map_output = Path.home() / 'tb4_saved_map_points.csv'

        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.map_timer = self.create_timer(MAP_SAVE_INTERVAL_S, self.save_map)

        self.get_logger().info('Map/frame/avoidance node started')

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
            self.get_logger().info('Local coordinate frame established from first odom sample')

        self.current_x = px
        self.current_y = py
        self.current_yaw = yaw

    def scan_callback(self, msg: LaserScan):
        if not self.origin_set:
            return

        n = len(msg.ranges)
        inc = msg.angle_increment
        front_angle = -math.pi / 2
        front_i = int(round((front_angle - msg.angle_min) / inc)) % n
        half_a = int(round(math.radians(45) / inc))

        front_vals = []

        for k in range(-half_a, half_a + 1):
            i = (front_i + k) % n
            r = msg.ranges[i]
            if msg.range_min < r < msg.range_max:
                front_vals.append(r)

        self.nearest_front = min(front_vals) if front_vals else float('inf')

        c = math.cos(self.current_yaw)
        s = math.sin(self.current_yaw)
        for i, r in enumerate(msg.ranges):
            if not (msg.range_min < r < msg.range_max):
                continue
            beam = msg.angle_min + i * msg.angle_increment
            gx = self.current_x + r * math.cos(self.current_yaw + beam)
            gy = self.current_y + r * math.sin(self.current_yaw + beam)

            lx = c * (gx - self.origin_x) + s * (gy - self.origin_y)
            ly = -s * (gx - self.origin_x) + c * (gy - self.origin_y)
            self.map_points.append((round(lx, 3), round(ly, 3)))

        if len(self.map_points) > 5000:
            self.map_points = self.map_points[-5000:]

    def image_callback(self, msg: CompressedImage):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2),
        )
        pixels = cv2.countNonZero(mask)
        self.red_detected = pixels >= MIN_RED_PIXELS

    def control_loop(self):
        cmd = Twist()

        if self.red_detected:
            self.get_logger().info('Red cube detected: stopping robot')
        elif self.nearest_front <= AVOID_DISTANCE:
            cmd.angular.z = TURN_SPEED
            self.get_logger().warn(f'Obstacle ahead ({self.nearest_front:.2f} m), turning')
        else:
            cmd.linear.x = FORWARD_SPEED

        self.cmd_pub.publish(cmd)

    def save_map(self):
        if not self.map_points:
            return

        unique_points = sorted(set(self.map_points))
        with self.map_output.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_local_m', 'y_local_m'])
            writer.writerows(unique_points)

        self.get_logger().info(
            f'Saved map snapshot with {len(unique_points)} points to {self.map_output}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = MapFrameAvoidance()
    try:
        rclpy.spin(node)
    finally:
        node.save_map()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

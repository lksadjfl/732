#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

# Change this to your robot namespace
NAMESPACE = '/T13'


class TestNode(Node):

    def __init__(self):
        super().__init__('test_node')

        self.cmd_pub = self.create_publisher(
            Twist,
            f'{NAMESPACE}/cmd_vel',
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            f'{NAMESPACE}/scan',
            self.scan_callback,
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)
        self.received_scan = False

        self.get_logger().info('Front-wall LiDAR check node started')
        self.get_logger().info('Robot will remain stopped')
        self.get_logger().info(f'Subscribing to {NAMESPACE}/scan')

    def stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def scan_callback(self, msg):
        if not self.received_scan:
            self.received_scan = True
            self.get_logger().info('LiDAR data received successfully')

        n = len(msg.ranges)
        if n == 0:
            self.get_logger().info('No LiDAR beams received')
            return

        # Match lidar_logger.py:
        # TurtleBot 4 robot front / camera side corresponds to angle = pi
        target_angle = -math.pi / 2

        # Use the scan's angle_increment directly
        raw_idx = int(round((target_angle - msg.angle_min) / msg.angle_increment))
        raw_idx = max(0, min(n - 1, raw_idx))

        # Average a small window around the front beam
        window = 5
        lo = max(0, raw_idx - window)
        hi = min(n - 1, raw_idx + window)

        window_ranges = [
            msg.ranges[i] for i in range(lo, hi + 1)
            if msg.range_min <= msg.ranges[i] <= msg.range_max
        ]

        if window_ranges:
            front_range = sum(window_ranges) / len(window_ranges)
            self.get_logger().info(
                f'Front wall check | beam index: {raw_idx} | '
                f'front avg range: {front_range:.3f} m | '
                f'window beams used: {len(window_ranges)}'
            )
        else:
            self.get_logger().info(
                f'Front wall check | beam index: {raw_idx} | no valid front-wall readings'
            )

    def control_loop(self):
        self.stop_robot()


def main(args=None):
    rclpy.init(args=args)
    node = TestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

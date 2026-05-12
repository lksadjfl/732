import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry   # NEW

NAMESPACE      = '/T13'
FORWARD_SPEED  = 0.15
TURN_SPEED     = 0.5
AVOID_DISTANCE = 0.55
FRONT_ARC_DEG  = 60


class AvoidancePhysical(Node):
    def __init__(self):
        super().__init__('avoidance_physical')

        self.publisher = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback,
            10
        )

        # NEW: odometry subscription
        self.current_x = 0.0
        self.current_y = 0.0

        self.odom_sub = self.create_subscription(
            Odometry,
            f'{NAMESPACE}/odom',
            self.odom_callback,
            10
        )

        self.nearest_front = float('inf')
        self.nearest_left  = float('inf')
        self.nearest_right = float('inf')

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Avoidance controller started')

    def scan_callback(self, msg):
        inc = msg.angle_increment
        arc_r = math.radians(FRONT_ARC_DEG)
        side_r = math.radians(90)

        half_a = int(round(arc_r / inc))
        side_a = int(round(side_r / inc))
        n = len(msg.ranges)

        front_angle = -math.pi / 2
        front_i = int(round((front_angle - msg.angle_min) / inc)) % n

        left_i = (front_i + side_a) % n
        right_i = (front_i - side_a) % n

        def arc_min(center_i, half_width_i):
            indices = [
                (center_i + k) % n
                for k in range(-half_width_i, half_width_i + 1)
            ]

            vals = [
                msg.ranges[i]
                for i in indices
                if msg.range_min < msg.ranges[i] < msg.range_max
            ]

            return min(vals) if vals else float('inf')

        self.nearest_front = arc_min(front_i, half_a)
        self.nearest_left  = arc_min(left_i, half_a)
        self.nearest_right = arc_min(right_i, half_a)

    # NEW: odometry callback
    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def control_loop(self):
        msg = Twist()

        if self.nearest_front > AVOID_DISTANCE:
            msg.linear.x = FORWARD_SPEED
            msg.angular.z = 0.0

            self.get_logger().info(
                f'Fwd | front={self.nearest_front:.2f} m | '
                f'pos=({self.current_x:.2f}, {self.current_y:.2f})'
            )

        else:
            msg.linear.x = 0.0

            if self.nearest_left >= self.nearest_right:
                msg.angular.z = TURN_SPEED
                self.get_logger().warn('Obstacle — turning LEFT')
            else:
                msg.angular.z = -TURN_SPEED
                self.get_logger().warn('Obstacle — turning RIGHT')

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AvoidancePhysical()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

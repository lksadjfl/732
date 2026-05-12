import rclpy, cv2, math
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

NAMESPACE      = '/T13'
FORWARD_SPEED  = 0.15
TURN_SPEED     = 0.5
AVOID_DISTANCE = 0.55
FRONT_ARC_DEG  = 60

RED_LOW1  = np.array([5,   150, 100])
RED_HIGH1 = np.array([10,  255, 255])
RED_LOW2  = np.array([170, 150, 100])
RED_HIGH2 = np.array([180, 255, 255])
MIN_PIXELS = 500


class DetectAndStop(Node):
    def __init__(self):
        super().__init__('detect_and_stop')

        self.bridge = CvBridge()

        self.publisher = self.create_publisher(
            Twist, f'{NAMESPACE}/cmd_vel', 10
        )

        self.create_subscription(
            LaserScan, f'{NAMESPACE}/scan',
            self.scan_callback, 10
        )

        self.create_subscription(
            CompressedImage,
            f'{NAMESPACE}/oakd/rgb/image_raw/compressed',
            self.image_callback, 10
        )

        self.create_subscription(
            Odometry, f'{NAMESPACE}/odom',
            self.odom_callback, 10
        )

        self.nearest_front = float('inf')
        self.nearest_left  = float('inf')
        self.nearest_right = float('inf')

        self.current_x = 0.0
        self.current_y = 0.0

        self.cube_detected = False

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Detect-and-stop node started')

    def scan_callback(self, msg):
        inc = msg.angle_increment
        arc_r = math.radians(FRONT_ARC_DEG)
        side_r = math.radians(90)

        half_a = int(round(arc_r / inc))
        side_a = int(round(side_r / inc))
        n = len(msg.ranges)

        # On your robot, -pi/2 is the forward direction
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

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def image_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
            cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
        )

        pixels = cv2.countNonZero(mask)

        # Continuously update detection state
        self.cube_detected = pixels >= MIN_PIXELS

        overlay = img.copy()
        overlay[mask > 0] = [0, 0, 255]

        cv2.putText(
            overlay,
            f'Red pixels: {pixels}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if self.cube_detected:
            cv2.putText(
                overlay,
                'DETECTED',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        cv2.imshow('Detection', overlay)
        cv2.waitKey(1)

    def stop(self):
        self.publisher.publish(Twist())

    def control_loop(self):
        msg = Twist()

        # If red cube is detected, stop
        if self.cube_detected:
            msg.linear.x = 0.0
            msg.angular.z = 0.0

            self.get_logger().info(
                f'RED DETECTED | STOP | '
                f'pos=({self.current_x:.2f}, {self.current_y:.2f})'
            )

        # If red cube is not detected, continue obstacle avoidance
        else:
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

    def destroy_node(self):
        self.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectAndStop()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

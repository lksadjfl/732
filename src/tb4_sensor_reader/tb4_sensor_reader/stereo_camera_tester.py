import re
import time
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


IMAGE_TYPE = 'sensor_msgs/msg/Image'
COMPRESSED_IMAGE_TYPE = 'sensor_msgs/msg/CompressedImage'


@dataclass
class TopicStat:
    count: int = 0
    first_time: float = 0.0
    last_time: float = 0.0
    width: int = 0
    height: int = 0
    encoding: str = ''
    camera_name: str = ''


class StereoCameraTester(Node):
    def __init__(self):
        super().__init__('stereo_camera_tester')

        self.declare_parameter('robot_namespace', '/T21')
        self.declare_parameter('report_period', 5.0)
        self.declare_parameter('show_windows', True)
        self.declare_parameter('scan_graph', True)

        robot_namespace = self.get_parameter('robot_namespace').get_parameter_value().string_value
        report_period = self.get_parameter('report_period').get_parameter_value().double_value
        self.show_windows = self.get_parameter('show_windows').get_parameter_value().bool_value
        self.scan_graph = self.get_parameter('scan_graph').get_parameter_value().bool_value

        ns = robot_namespace.strip()
        if ns and not ns.startswith('/'):
            ns = '/' + ns
        if ns == '/':
            ns = ''
        self.ns = ns

        self.bridge = CvBridge()
        self.stats = {}
        self.topic_subscriptions = []
        self.subscribed_topics = set()
        self.discovered_image_topics = {}

        self.latest_frames = {
            'rgb': None,
            'left': None,
            'right': None,
        }
        self.latest_topics = {
            'rgb': '',
            'left': '',
            'right': '',
        }
        self.window_names = {
            'rgb': 'OAK-D RGB camera',
            'left': 'OAK-D LEFT mono camera',
            'right': 'OAK-D RIGHT mono camera',
        }

        self._subscribe_common_candidates()

        self.create_timer(report_period, self.report_status)
        if self.scan_graph:
            self.create_timer(2.0, self.scan_and_subscribe_graph_topics)
        if self.show_windows:
            self._init_windows()
            self.create_timer(0.03, self.update_windows)

        self.get_logger().info('Auto-discovery three-camera tester started.')
        self.get_logger().info(f'robot_namespace = {ns or "/"}')
        self.get_logger().info('This node will scan ROS graph for all sensor_msgs/Image and CompressedImage topics.')
        self.get_logger().info('If left/right image topics exist, it will auto-subscribe and show them.')
        self.get_logger().info('Waiting for image data...')
        self.scan_and_subscribe_graph_topics()

    def _subscribe_common_candidates(self):
        ns = self.ns

        rgb_raw = self._unique_topics([
            f'{ns}/oakd/rgb/image_raw',
            f'{ns}/oakd/rgb/preview/image_raw',
            f'{ns}/oakd/rgb/image_rect',
            f'{ns}/color/image',
            f'{ns}/rgb/image',
            f'{ns}/camera/color/image_raw',
            '/oakd/rgb/image_raw',
            '/oakd/rgb/preview/image_raw',
            '/oakd/rgb/image_rect',
            '/color/image',
            '/rgb/image',
            '/camera/color/image_raw',
        ])
        rgb_compressed = self._unique_topics([
            topic + '/compressed' for topic in rgb_raw
        ])
        left_raw = self._unique_topics([
            f'{ns}/oakd/left/image_rect',
            f'{ns}/oakd/left/image_raw',
            f'{ns}/oakd/left/image',
            f'{ns}/left/image',
            f'{ns}/left/image_rect',
            f'{ns}/left/image_raw',
            f'{ns}/camera/left/image_raw',
            '/oakd/left/image_rect',
            '/oakd/left/image_raw',
            '/oakd/left/image',
            '/left/image',
            '/left/image_rect',
            '/left/image_raw',
            '/camera/left/image_raw',
        ])
        right_raw = self._unique_topics([
            f'{ns}/oakd/right/image_rect',
            f'{ns}/oakd/right/image_raw',
            f'{ns}/oakd/right/image',
            f'{ns}/right/image',
            f'{ns}/right/image_rect',
            f'{ns}/right/image_raw',
            f'{ns}/camera/right/image_raw',
            '/oakd/right/image_rect',
            '/oakd/right/image_raw',
            '/oakd/right/image',
            '/right/image',
            '/right/image_rect',
            '/right/image_raw',
            '/camera/right/image_raw',
        ])
        left_compressed = self._unique_topics([topic + '/compressed' for topic in left_raw])
        right_compressed = self._unique_topics([topic + '/compressed' for topic in right_raw])

        self._create_image_subscriptions('rgb', rgb_raw, compressed=False, source='candidate')
        self._create_image_subscriptions('rgb', rgb_compressed, compressed=True, source='candidate')
        self._create_image_subscriptions('left', left_raw, compressed=False, source='candidate')
        self._create_image_subscriptions('left', left_compressed, compressed=True, source='candidate')
        self._create_image_subscriptions('right', right_raw, compressed=False, source='candidate')
        self._create_image_subscriptions('right', right_compressed, compressed=True, source='candidate')

    @staticmethod
    def _unique_topics(topics):
        seen = set()
        result = []
        for topic in topics:
            if topic and topic not in seen:
                seen.add(topic)
                result.append(topic)
        return result

    def scan_and_subscribe_graph_topics(self):
        try:
            topic_info = self.get_topic_names_and_types()
        except Exception as exc:
            self.get_logger().warn(f'Could not scan ROS graph topics: {exc}')
            return

        new_image_topics = []
        for topic, types in sorted(topic_info):
            for msg_type in types:
                if msg_type not in (IMAGE_TYPE, COMPRESSED_IMAGE_TYPE):
                    continue
                if topic not in self.discovered_image_topics:
                    self.discovered_image_topics[topic] = msg_type
                    new_image_topics.append((topic, msg_type))

                camera_name = self._classify_topic(topic)
                if camera_name in ('rgb', 'left', 'right'):
                    self._create_image_subscriptions(
                        camera_name,
                        [topic],
                        compressed=(msg_type == COMPRESSED_IMAGE_TYPE),
                        source='graph',
                    )

        if new_image_topics:
            self.get_logger().info('Discovered image topics in current ROS graph:')
            for topic, msg_type in new_image_topics:
                label = self._classify_topic(topic)
                self.get_logger().info(f'  [{label}] {topic} [{msg_type}]')

    def _classify_topic(self, topic):
        t = topic.lower()
        if re.search(r'(^|/)left(/|_|$)', t):
            return 'left'
        if re.search(r'(^|/)right(/|_|$)', t):
            return 'right'
        if any(word in t for word in ('/rgb/', '/color/', '/preview/', 'image_raw/compressed')):
            return 'rgb'
        if 'depth' in t or 'stereo' in t:
            return 'depth/stereo'
        return 'other'

    def _create_image_subscriptions(self, camera_name, topics, compressed=False, source='candidate'):
        msg_type = CompressedImage if compressed else Image
        for topic in topics:
            key = (topic, 'compressed' if compressed else 'raw')
            if key in self.subscribed_topics:
                continue
            self.subscribed_topics.add(key)
            self.topic_subscriptions.append(
                self.create_subscription(
                    msg_type,
                    topic,
                    lambda msg, c=camera_name, t=topic, z=compressed: self.image_callback(c, t, msg, z),
                    10,
                )
            )
            self.get_logger().info(
                f'Subscribed {source}: [{camera_name.upper()}] {topic} '
                f'[{COMPRESSED_IMAGE_TYPE if compressed else IMAGE_TYPE}]'
            )

    def _init_windows(self):
        try:
            for camera_name, window_name in self.window_names.items():
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 480, 360)
                cv2.imshow(window_name, self._placeholder(camera_name))
            cv2.waitKey(1)
            self.get_logger().info('Opened three camera windows: RGB, LEFT, RIGHT.')
        except Exception as exc:
            self.show_windows = False
            self.get_logger().error(f'Could not open OpenCV windows. Display disabled. Error: {exc}')

    def _placeholder(self, camera_name):
        img = np.full((360, 480, 3), 255, dtype=np.uint8)
        text = f'Waiting for {camera_name.upper()} image...'
        cv2.putText(img, text, (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        topic = self.latest_topics.get(camera_name, '')
        if topic:
            cv2.putText(img, topic[-55:], (15, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        return img

    def image_callback(self, camera_name, topic, msg, compressed=False):
        now = time.time()
        stat = self.stats.setdefault(topic, TopicStat(camera_name=camera_name))
        stat.count += 1
        stat.last_time = now

        if compressed:
            stat.width = 0
            stat.height = 0
            stat.encoding = 'compressed'
        else:
            stat.width = msg.width
            stat.height = msg.height
            stat.encoding = msg.encoding

        if stat.first_time == 0.0:
            stat.first_time = now
            self.get_logger().info(
                f'[{camera_name.upper()}] First image received on {topic} '
                f'({stat.width}x{stat.height}, encoding={stat.encoding})'
            )

        if self.show_windows and camera_name in self.latest_frames:
            try:
                frame = self._message_to_bgr(msg, compressed)
                self.latest_frames[camera_name] = frame
                self.latest_topics[camera_name] = topic
            except Exception as exc:
                self.get_logger().warn(f'Failed to convert image from {topic}: {exc}')

        if stat.count % 30 == 0:
            duration = max(now - stat.first_time, 1e-6)
            fps = stat.count / duration
            self.get_logger().info(
                f'[{camera_name.upper()}] {topic}: count={stat.count}, '
                f'fps≈{fps:.2f}, size={stat.width}x{stat.height}, encoding={stat.encoding}'
            )

    def _message_to_bgr(self, msg, compressed):
        if compressed:
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        encoding = msg.encoding.lower()

        if frame is None:
            raise RuntimeError('cv_bridge returned None')

        if len(frame.shape) == 2:
            if frame.dtype == np.uint16:
                nonzero = frame[frame > 0]
                if nonzero.size > 0:
                    max_val = np.percentile(nonzero, 95)
                    max_val = max(max_val, 1.0)
                    display = np.clip((frame.astype(np.float32) / max_val) * 255.0, 0, 255).astype(np.uint8)
                else:
                    display = np.zeros_like(frame, dtype=np.uint8)
                return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            if frame.dtype != np.uint8:
                display = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if encoding == 'rgb8':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if encoding == 'rgba8':
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if encoding == 'bgra8':
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame

    def update_windows(self):
        if not self.show_windows:
            return
        try:
            for camera_name, window_name in self.window_names.items():
                frame = self.latest_frames.get(camera_name)
                if frame is None:
                    frame = self._placeholder(camera_name)
                else:
                    frame = frame.copy()
                    topic = self.latest_topics.get(camera_name, '')
                    if topic:
                        cv2.putText(frame, topic[-70:], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info('q pressed in OpenCV window. Shutting down node.')
                rclpy.shutdown()
        except Exception as exc:
            self.show_windows = False
            self.get_logger().error(f'OpenCV window update failed. Display disabled. Error: {exc}')

    def report_status(self):
        if self.discovered_image_topics:
            self.get_logger().info('Known image topics from ROS graph:')
            for topic, msg_type in sorted(self.discovered_image_topics.items()):
                self.get_logger().info(f'  [{self._classify_topic(topic)}] {topic} [{msg_type}]')
        else:
            self.get_logger().warn('No Image/CompressedImage topics discovered in ROS graph yet.')

        self._report_for_side('rgb')
        self._report_for_side('left')
        self._report_for_side('right')

    def _report_for_side(self, side):
        active = []
        for topic, stat in sorted(self.stats.items()):
            if stat.camera_name != side or stat.count <= 0:
                continue
            duration = max((stat.last_time - stat.first_time), 1e-6)
            fps = 1.0 if stat.count == 1 else stat.count / duration
            active.append(
                f'{topic} -> count={stat.count}, fps≈{fps:.2f}, '
                f'size={stat.width}x{stat.height}, encoding={stat.encoding}'
            )

        if active:
            self.get_logger().info(f'[{side.upper()}] Working topics:')
            for item in active:
                self.get_logger().info(f'  {item}')
        else:
            self.get_logger().warn(f'[{side.upper()}] No image data received yet.')


def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_windows:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

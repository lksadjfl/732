#!/usr/bin/env python3

import os
import csv
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# ===== Change this to your robot namespace =====
NAMESPACE = '/T13'

# ===== Detection settings =====
RED_PIXEL_THRESHOLD = 1500      # save image / csv only if red pixels exceed this
MIN_CONTOUR_AREA = 500          # ignore tiny noisy blobs
DISPLAY_SCALE = 1.0             # 1.0 = original size
SAVE_DIR = os.path.expanduser('~')
CSV_PATH = os.path.join(SAVE_DIR, 'detections.csv')


class RedObjectDetector(Node):
    def __init__(self):
        super().__init__('camera_c2_node')

        self.topic = f'{NAMESPACE}/oakd/rgb/image_raw/compressed'
        self.sub = self.create_subscription(
            CompressedImage,
            self.topic,
            self.image_callback,
            10
        )

        self.detection_count = 0
        self.csv_initialized = False
        self.last_saved_time_ns = 0
        self.save_cooldown_ns = int(1.5 * 1e9)   # 1.5 seconds cooldown

        self.get_logger().info(f'Subscribing to {self.topic}')
        self.get_logger().info('Press q in the image window to quit.')

        self.init_csv()

    def init_csv(self):
        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'detection_id',
                    'timestamp_sec',
                    'red_pixels',
                    'bbox_x',
                    'bbox_y',
                    'bbox_w',
                    'bbox_h',
                    'image_file'
                ])
        self.csv_initialized = True
        self.get_logger().info(f'CSV logging to: {CSV_PATH}')

    def image_callback(self, msg):
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warning('Failed to decode image frame')
            return

        # Convert BGR -> HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wraps around HSV hue range, so use two masks
        lower_red_1 = np.array([0, 120, 70])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 120, 70])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        red_pixels = int(cv2.countNonZero(red_mask))

        # Find biggest contour
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox = None
        object_visible = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area >= MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(largest)
                bbox = (x, y, w, h)
                object_visible = True

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    'Red object detected',
                    (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # Overlay info
        cv2.putText(
            frame,
            f'Red pixels: {red_pixels}',
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )
        cv2.putText(
            frame,
            f'Visible: {"Y" if object_visible else "N"}',
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        # Save detection if above threshold
        now_ns = self.get_clock().now().nanoseconds
        if red_pixels >= RED_PIXEL_THRESHOLD and (now_ns - self.last_saved_time_ns) > self.save_cooldown_ns:
            self.detection_count += 1
            self.last_saved_time_ns = now_ns

            image_filename = f'detection_{self.detection_count}.png'
            image_path = os.path.join(SAVE_DIR, image_filename)
            cv2.imwrite(image_path, frame)

            timestamp_sec = now_ns / 1e9

            if bbox is None:
                x = y = w = h = -1
            else:
                x, y, w, h = bbox

            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.detection_count,
                    f'{timestamp_sec:.3f}',
                    red_pixels,
                    x, y, w, h,
                    image_filename
                ])

            self.get_logger().info(
                f'Saved {image_filename} | red_pixels={red_pixels} | visible={"Y" if object_visible else "N"}'
            )

        # Resize for display if needed
        if DISPLAY_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

        cv2.imshow('C2 Red Object Detection', frame)
        cv2.imshow('Red Mask', red_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('q pressed, shutting down...')
            cv2.destroyAllWindows()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RedObjectDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

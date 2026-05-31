#!/usr/bin/env python3
"""Detect a red cube during Nav2 search and request an automatic return home."""

import csv
import json
import math
import os
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose, Spin
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


RED_LOW1 = np.array([0, 150, 110])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 150, 110])
RED_HIGH2 = np.array([180, 255, 255])

MIN_RED_PIXELS = 5800
MIN_BOX_WIDTH_PX = 8
MIN_BOX_HEIGHT_PX = 8
RED_ASPECT_MIN = 0.5
RED_ASPECT_MAX = 2.0
RED_CONFIRM_FRAMES = 3

RED_CUBE_SIZE_M = 0.06
OAK_RGB_HFOV_DEG = 66.0
OAK_RGB_VFOV_DEG = 54.0
SPIN_TARGET_YAW = math.pi
MAX_RETURN_ATTEMPTS = 2


class MissionState(str, Enum):
    WAITING_FOR_HOME = "WAITING_FOR_HOME"
    SEARCHING = "SEARCHING"
    RED_LOCKED = "RED_LOCKED"
    RETURNING = "RETURNING"
    SPINNING = "SPINNING"
    DONE = "DONE"
    FAILED = "FAILED"


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def find_red_bbox(frame):
    """Return the largest valid red bbox and mask, or (None, mask)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOW1, RED_HIGH1),
        cv2.inRange(hsv, RED_LOW2, RED_HIGH2),
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_RED_PIXELS:
        return None, mask

    x, y, w, h = cv2.boundingRect(largest)
    aspect = w / float(h) if h > 0 else float("inf")
    valid = (
        w >= MIN_BOX_WIDTH_PX
        and h >= MIN_BOX_HEIGHT_PX
        and RED_ASPECT_MIN <= aspect <= RED_ASPECT_MAX
    )
    return ((x, y, w, h) if valid else None), mask


def estimate_cube_robot_coordinates(image_shape, bbox):
    """Estimate cube (x_right, y_forward) from its image bbox."""
    img_h, img_w = image_shape[:2]
    bx, by, bw, bh = bbox
    fx = (img_w / 2.0) / math.tan(math.radians(OAK_RGB_HFOV_DEG) / 2.0)
    fy = (img_h / 2.0) / math.tan(math.radians(OAK_RGB_VFOV_DEG) / 2.0)
    z_width = RED_CUBE_SIZE_M * fx / float(bw)
    z_height = RED_CUBE_SIZE_M * fy / float(bh)
    y_forward = 0.5 * (z_width + z_height)
    x_right = ((bx + bw / 2.0) - img_w / 2.0) * y_forward / fx
    return {
        "x_right": x_right,
        "y_forward": y_forward,
        "distance": math.hypot(x_right, y_forward),
        "fx_px": fx,
        "fy_px": fy,
        "z_from_width": z_width,
        "z_from_height": z_height,
    }


def robot_to_map(robot_x, robot_y, robot_yaw, x_right, y_forward):
    return (
        robot_x + y_forward * math.cos(robot_yaw) + x_right * math.sin(robot_yaw),
        robot_y + y_forward * math.sin(robot_yaw) - x_right * math.cos(robot_yaw),
    )


def map_to_task_local(home_x, home_y, home_yaw, map_x, map_y):
    dx = map_x - home_x
    dy = map_y - home_y
    return (
        dx * math.sin(home_yaw) - dy * math.cos(home_yaw),
        dx * math.cos(home_yaw) + dy * math.sin(home_yaw),
    )


class Phase2Nav2RedReturn(Node):
    def __init__(self):
        super().__init__("phase2_nav2_red_return")
        self.declare_parameter("namespace", "/T13")
        self.declare_parameter("image_topic", "")
        self.declare_parameter("return_retry_delay", 2.0)
        self.declare_parameter("show_camera", True)
        self.declare_parameter("evidence_dir", "~/tb4_phase2_evidence")

        self.ns = str(self.get_parameter("namespace").value).rstrip("/")
        image_topic = str(self.get_parameter("image_topic").value).strip()
        self.image_topic = image_topic or f"{self.ns}/oakd/rgb/image_raw/compressed"
        self.return_retry_delay = float(self.get_parameter("return_retry_delay").value)
        self.show_camera = bool(self.get_parameter("show_camera").value)

        evidence_root = Path(str(self.get_parameter("evidence_dir").value)).expanduser()
        self.run_dir = evidence_root / time.strftime("nav2_red_return_%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "red_cube_measurements.csv"
        self._init_csv()

        self.bridge = CvBridge()
        self.state = MissionState.WAITING_FOR_HOME
        self.home_pose = None
        self.current_pose = None
        self.latest_frame = None
        self.latest_bbox = None
        self.red_seen_count = 0
        self.red_locked = False
        self.return_attempts = 0
        self.retry_timer = None

        self.navigate_client = ActionClient(self, NavigateToPose, f"{self.ns}/navigate_to_pose")
        self.spin_client = ActionClient(self, Spin, f"{self.ns}/spin")
        self.create_subscription(
            PoseWithCovarianceStamped, f"{self.ns}/amcl_pose", self.amcl_pose_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, f"{self.ns}/initialpose", self.initial_pose_callback, 10)
        self.create_subscription(CompressedImage, self.image_topic, self.image_callback, 10)

        self.get_logger().info(
            f"Nav2 red-return coordinator started. image={self.image_topic}. "
            "Use RViz 2D Pose Estimate to define mission home.")

    def _init_csv(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "wall_time", "state", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "robot_x_right_m", "robot_y_forward_m", "robot_distance_m",
                "task_x_right_m", "task_y_forward_m", "map_x_m", "map_y_m",
            ])

    def _pose_tuple(self, msg):
        p = msg.pose.pose.position
        return p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation)

    def amcl_pose_callback(self, msg):
        self.current_pose = self._pose_tuple(msg)

    def initial_pose_callback(self, msg):
        if self.state not in (MissionState.WAITING_FOR_HOME, MissionState.SEARCHING):
            self.get_logger().warn("Ignoring initialpose update after return sequence started.")
            return
        self.home_pose = self._pose_tuple(msg)
        self.state = MissionState.SEARCHING
        self.get_logger().info(
            f"Mission home set: map=({self.home_pose[0]:.3f}, {self.home_pose[1]:.3f}), "
            f"yaw={math.degrees(self.home_pose[2]):.1f}deg. SEARCHING.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Image decode failed: {exc}")
            return

        self.latest_frame = frame
        self.latest_bbox, mask = find_red_bbox(frame)
        red_pixels = int(cv2.countNonZero(mask))

        if self.state == MissionState.SEARCHING and self.latest_bbox is not None:
            self.red_seen_count += 1
        elif self.state == MissionState.SEARCHING:
            self.red_seen_count = 0

        if (
            self.state == MissionState.SEARCHING
            and self.red_seen_count >= RED_CONFIRM_FRAMES
            and not self.red_locked
        ):
            self.red_locked = True
            self.state = MissionState.RED_LOCKED
            result = self._build_cube_result(red_pixels)
            self._save_final_evidence(result)
            self.get_logger().info("Red cube locked. Requesting Nav2 return to mission home.")
            self._send_return_goal()

        if self.show_camera:
            self._show_camera(frame, red_pixels)

    def _build_cube_result(self, red_pixels):
        if self.latest_frame is None or self.latest_bbox is None or self.current_pose is None:
            return None
        estimate = estimate_cube_robot_coordinates(self.latest_frame.shape, self.latest_bbox)
        map_x, map_y = robot_to_map(
            *self.current_pose, estimate["x_right"], estimate["y_forward"])
        if self.home_pose is None:
            task_x, task_y = None, None
        else:
            task_x, task_y = map_to_task_local(*self.home_pose, map_x, map_y)
        return {
            "timestamp": time.time(),
            "state": self.state.value,
            "bbox": self.latest_bbox,
            "red_pixels": red_pixels,
            "robot": estimate,
            "task_local": {"x_right": task_x, "y_forward": task_y},
            "map": {"x": map_x, "y": map_y},
            "robot_pose_map": {
                "x": self.current_pose[0],
                "y": self.current_pose[1],
                "yaw": self.current_pose[2],
            },
            "home_pose_map": {
                "x": self.home_pose[0],
                "y": self.home_pose[1],
                "yaw": self.home_pose[2],
            },
        }

    def _save_final_evidence(self, result):
        if result is None:
            self.get_logger().warn("Red cube locked but coordinates are unavailable.")
            return
        (self.run_dir / "red_cube_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        (self.run_dir / "red_cube_summary.txt").write_text(
            "TB4 Phase 2 red cube result\n"
            f"robot: x_right={result['robot']['x_right']:.4f} m, "
            f"y_forward={result['robot']['y_forward']:.4f} m\n"
            f"task_local: x_right={result['task_local']['x_right']:.4f} m, "
            f"y_forward={result['task_local']['y_forward']:.4f} m\n"
            f"map: x={result['map']['x']:.4f} m, y={result['map']['y']:.4f} m\n",
            encoding="utf-8",
        )
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                f"{result['timestamp']:.3f}", result["state"], *result["bbox"],
                f"{result['robot']['x_right']:.4f}",
                f"{result['robot']['y_forward']:.4f}",
                f"{result['robot']['distance']:.4f}",
                f"{result['task_local']['x_right']:.4f}",
                f"{result['task_local']['y_forward']:.4f}",
                f"{result['map']['x']:.4f}", f"{result['map']['y']:.4f}",
            ])
        display = self.latest_frame.copy()
        x, y, w, h = self.latest_bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imwrite(str(self.run_dir / "red_cube_final.jpg"), display)

    def _show_camera(self, frame, red_pixels):
        try:
            display = frame.copy()
            if self.latest_bbox is not None:
                x, y, w, h = self.latest_bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                display, f"state={self.state.value} red={red_pixels}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.imshow("TB4 Nav2 Red Return", display)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.show_camera = False
            self.get_logger().warn(f"Camera window disabled: {exc}")

    def build_home_goal(self):
        if self.home_pose is None:
            return None
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self.home_pose[0]
        goal.pose.pose.position.y = self.home_pose[1]
        qx, qy, qz, qw = quaternion_from_yaw(self.home_pose[2])
        goal.pose.pose.orientation.x = qx
        goal.pose.pose.orientation.y = qy
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw
        return goal

    def _send_return_goal(self):
        if self.home_pose is None:
            self._fail("Cannot return: mission home has not been set.")
            return
        self.return_attempts += 1
        if not self.navigate_client.wait_for_server(timeout_sec=1.0):
            self._retry_or_fail("NavigateToPose action server unavailable")
            return
        self.state = MissionState.RETURNING
        future = self.navigate_client.send_goal_async(self.build_home_goal())
        future.add_done_callback(self._return_goal_response)
        self.get_logger().info(
            f"Nav2 return goal sent: attempt={self.return_attempts}/{MAX_RETURN_ATTEMPTS}.")

    def _return_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._retry_or_fail("Nav2 return goal rejected")
            return
        goal_handle.get_result_async().add_done_callback(self._return_result)

    def _return_result(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Nav2 returned home. Requesting 180deg spin.")
            self._send_spin_goal()
        else:
            self._retry_or_fail(f"Nav2 return failed with status={status}")

    def _retry_or_fail(self, reason):
        if self.return_attempts >= MAX_RETURN_ATTEMPTS:
            self._fail(f"{reason}; retry exhausted")
            return
        self.get_logger().warn(
            f"{reason}; retrying in {self.return_retry_delay:.1f}s.")
        self.retry_timer = self.create_timer(self.return_retry_delay, self._retry_once)

    def _retry_once(self):
        if self.retry_timer is not None:
            self.retry_timer.cancel()
            self.destroy_timer(self.retry_timer)
            self.retry_timer = None
        self._send_return_goal()

    def _send_spin_goal(self):
        if not self.spin_client.wait_for_server(timeout_sec=1.0):
            self._fail("Spin action server unavailable")
            return
        self.state = MissionState.SPINNING
        goal = Spin.Goal()
        goal.target_yaw = float(SPIN_TARGET_YAW)
        goal.time_allowance.sec = 30
        future = self.spin_client.send_goal_async(goal)
        future.add_done_callback(self._spin_goal_response)

    def _spin_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._fail("Spin goal rejected")
            return
        goal_handle.get_result_async().add_done_callback(self._spin_result)

    def _spin_result(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.state = MissionState.DONE
            self.get_logger().info("Mission DONE: returned home and completed 180deg spin.")
        else:
            self._fail(f"Spin failed with status={status}")

    def _fail(self, reason):
        self.state = MissionState.FAILED
        self.get_logger().error(f"Mission FAILED: {reason}")


def main(args=None):
    rclpy.init(args=args)
    node = Phase2Nav2RedReturn()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

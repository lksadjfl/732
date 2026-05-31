#!/usr/bin/env python3
"""Scan for a red cube at a Nav2 goal, save evidence, and return home."""

import csv
import json
import math
import statistics
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
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
SCAN_SPIN_TARGET_YAW = 2.0 * math.pi
MAX_SCAN_ATTEMPTS = 2
MAX_RETURN_ATTEMPTS = 2
ACTION_SERVER_WAIT_SEC = 10.0


class MissionState(str, Enum):
    WAITING_FOR_HOME = "WAITING_FOR_HOME"
    SEARCHING = "SEARCHING"
    SCANNING = "SCANNING"
    RED_LOCKED = "RED_LOCKED"
    RETURNING = "RETURNING"
    DONE = "DONE"
    FAILED = "FAILED"


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def goal_uuid_key(goal_id):
    return tuple(int(value) for value in goal_id.uuid)


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
    """Estimate cube position in the robot frame from its image bbox."""
    img_h, img_w = image_shape[:2]
    bx, by, bw, bh = bbox
    fx = (img_w / 2.0) / math.tan(math.radians(OAK_RGB_HFOV_DEG) / 2.0)
    fy = (img_h / 2.0) / math.tan(math.radians(OAK_RGB_VFOV_DEG) / 2.0)
    z_width = RED_CUBE_SIZE_M * fx / float(bw)
    z_height = RED_CUBE_SIZE_M * fy / float(bh)
    y_forward = 0.5 * (z_width + z_height)
    bbox_cx = bx + bw / 2.0
    bbox_cy = by + bh / 2.0
    x_right = (bbox_cx - img_w / 2.0) * y_forward / fx
    z_camera = -(bbox_cy - img_h / 2.0) * y_forward / fy
    return {
        "bbox_cx": bbox_cx,
        "bbox_cy": bbox_cy,
        "image_width": img_w,
        "image_height": img_h,
        "fx_px": fx,
        "fy_px": fy,
        "z_from_width": z_width,
        "z_from_height": z_height,
        "distance_forward": y_forward,
        "x_right": x_right,
        "y_forward": y_forward,
        "z_camera": z_camera,
        "distance": math.hypot(x_right, y_forward),
        "bearing_rad": math.atan2(-x_right, y_forward),
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


def median_cube_result(measurements):
    """Build a stable final coordinate result from consecutive observations."""
    if not measurements:
        return None

    def med(path):
        group, field = path
        return statistics.median(float(item[group][field]) for item in measurements)

    return {
        "measurement_count": len(measurements),
        "robot": {
            "x_right": med(("robot", "x_right")),
            "y_forward": med(("robot", "y_forward")),
            "z_camera": med(("robot", "z_camera")),
            "distance": med(("robot", "distance")),
            "bearing_rad": med(("robot", "bearing_rad")),
        },
        "task_local": {
            "x_right": med(("task_local", "x_right")),
            "y_forward": med(("task_local", "y_forward")),
        },
        "map": {
            "x": med(("map", "x")),
            "y": med(("map", "y")),
        },
    }


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
        self.scan_measurements = []
        self.scan_frames = []
        self.scan_attempts = 0
        self.return_attempts = 0
        self.retry_timer = None
        self.spin_goal_handle = None
        self.return_started = False
        self.final_cube_result = None
        self.own_return_goal_ids = set()
        self.external_goal_ids = set()
        self.handled_external_goal_ids = set()

        self.navigate_client = ActionClient(self, NavigateToPose, f"{self.ns}/navigate_to_pose")
        self.spin_client = ActionClient(self, Spin, f"{self.ns}/spin")
        self.create_subscription(
            PoseWithCovarianceStamped, f"{self.ns}/amcl_pose", self.amcl_pose_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, f"{self.ns}/initialpose", self.initial_pose_callback, 10)
        self.create_subscription(CompressedImage, self.image_topic, self.image_callback, 10)
        self.create_subscription(
            GoalStatusArray,
            f"{self.ns}/navigate_to_pose/_action/status",
            self.navigate_status_callback,
            10,
        )

        self.get_logger().info(
            f"Nav2 scan-return coordinator started. image={self.image_topic}. "
            "Use RViz 2D Pose Estimate, then send a Nav2 Goal.")

    def _init_csv(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "wall_time", "scan_attempt", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "bbox_cx", "bbox_cy", "img_w", "img_h", "fx_px", "fy_px",
                "z_from_width_m", "z_from_height_m", "distance_forward_m",
                "cube_robot_x_right_m", "cube_robot_y_forward_m", "cube_camera_z_m",
                "cube_distance_robot_m", "bearing_deg", "map_x_m", "map_y_m",
                "task_local_x_right_m", "task_local_y_forward_m",
            ])

    def _pose_tuple(self, msg):
        p = msg.pose.pose.position
        return p.x, p.y, yaw_from_quaternion(msg.pose.pose.orientation)

    def amcl_pose_callback(self, msg):
        self.current_pose = self._pose_tuple(msg)

    def initial_pose_callback(self, msg):
        if self.state not in (MissionState.WAITING_FOR_HOME, MissionState.SEARCHING):
            self.get_logger().warn("Ignoring initialpose update after scanning started.")
            return
        self.home_pose = self._pose_tuple(msg)
        self.state = MissionState.SEARCHING
        self.get_logger().info(
            f"Mission home set: map=({self.home_pose[0]:.3f}, {self.home_pose[1]:.3f}), "
            f"yaw={math.degrees(self.home_pose[2]):.1f}deg. SEARCHING.")

    def navigate_status_callback(self, msg):
        for status in msg.status_list:
            key = goal_uuid_key(status.goal_info.goal_id)
            if key in self.own_return_goal_ids:
                continue
            if status.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING):
                self.external_goal_ids.add(key)
            elif (
                status.status == GoalStatus.STATUS_SUCCEEDED
                and key in self.external_goal_ids
                and key not in self.handled_external_goal_ids
                and self.state == MissionState.SEARCHING
            ):
                self.handled_external_goal_ids.add(key)
                self.get_logger().info("External Nav2 search goal reached. Starting 360deg scan.")
                self._start_scan()

    def image_callback(self, msg):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Image decode failed: {exc}")
            return

        self.latest_frame = frame
        self.latest_bbox, mask = find_red_bbox(frame)
        red_pixels = int(cv2.countNonZero(mask))

        if self.state == MissionState.SCANNING:
            measurement = self._build_cube_measurement(red_pixels)
            if measurement is None:
                self.red_seen_count = 0
                self.scan_measurements = []
                self.scan_frames = []
            else:
                self.red_seen_count += 1
                self.scan_measurements.append(measurement)
                self.scan_frames.append(frame.copy())
                self.scan_measurements = self.scan_measurements[-RED_CONFIRM_FRAMES:]
                self.scan_frames = self.scan_frames[-RED_CONFIRM_FRAMES:]
                if self.red_seen_count >= RED_CONFIRM_FRAMES:
                    self.state = MissionState.RED_LOCKED
                    self._save_final_evidence(self.scan_measurements)
                    self.get_logger().info(
                        "Red cube locked during scan. Canceling Spin before return.")
                    self._cancel_scan_for_return()

        if self.show_camera:
            self._show_camera(frame, red_pixels)

    def _build_cube_measurement(self, red_pixels):
        if (
            self.latest_frame is None
            or self.latest_bbox is None
            or self.current_pose is None
            or self.home_pose is None
        ):
            return None
        estimate = estimate_cube_robot_coordinates(self.latest_frame.shape, self.latest_bbox)
        map_x, map_y = robot_to_map(
            *self.current_pose, estimate["x_right"], estimate["y_forward"])
        task_x, task_y = map_to_task_local(*self.home_pose, map_x, map_y)
        return {
            "timestamp": time.time(),
            "scan_attempt": self.scan_attempts,
            "red_pixels": red_pixels,
            "bbox": self.latest_bbox,
            "robot": estimate,
            "map": {"x": map_x, "y": map_y},
            "task_local": {"x_right": task_x, "y_forward": task_y},
            "robot_pose_map": {
                "x": self.current_pose[0],
                "y": self.current_pose[1],
                "yaw": self.current_pose[2],
            },
        }

    def _save_final_evidence(self, measurements):
        final = median_cube_result(measurements)
        if final is None:
            self._fail("Cannot save red cube evidence: no valid measurements")
            return
        result = {
            "timestamp": time.time(),
            "state": MissionState.RED_LOCKED.value,
            "coordinate_frame": {
                "origin": "mission home from RViz 2D Pose Estimate",
                "x_axis": "positive to robot initial right",
                "y_axis": "positive to robot initial forward",
            },
            "home_pose_map": {
                "x": self.home_pose[0], "y": self.home_pose[1], "yaw": self.home_pose[2]},
            "measurements": measurements,
            "median_result": final,
        }
        self.final_cube_result = final
        (self.run_dir / "red_cube_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        (self.run_dir / "red_cube_summary.txt").write_text(
            "TB4 Phase 2 red cube result\n"
            f"measurements: {final['measurement_count']}\n"
            f"robot: x_right={final['robot']['x_right']:.4f} m, "
            f"y_forward={final['robot']['y_forward']:.4f} m\n"
            f"task_local: x_right={final['task_local']['x_right']:.4f} m, "
            f"y_forward={final['task_local']['y_forward']:.4f} m\n"
            f"map: x={final['map']['x']:.4f} m, y={final['map']['y']:.4f} m\n",
            encoding="utf-8",
        )
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for measurement in measurements:
                robot = measurement["robot"]
                bx, by, bw, bh = measurement["bbox"]
                writer.writerow([
                    f"{measurement['timestamp']:.3f}", measurement["scan_attempt"],
                    bx, by, bw, bh, f"{robot['bbox_cx']:.2f}", f"{robot['bbox_cy']:.2f}",
                    robot["image_width"], robot["image_height"],
                    f"{robot['fx_px']:.3f}", f"{robot['fy_px']:.3f}",
                    f"{robot['z_from_width']:.4f}", f"{robot['z_from_height']:.4f}",
                    f"{robot['distance_forward']:.4f}", f"{robot['x_right']:.4f}",
                    f"{robot['y_forward']:.4f}", f"{robot['z_camera']:.4f}",
                    f"{robot['distance']:.4f}", f"{math.degrees(robot['bearing_rad']):.2f}",
                    f"{measurement['map']['x']:.4f}", f"{measurement['map']['y']:.4f}",
                    f"{measurement['task_local']['x_right']:.4f}",
                    f"{measurement['task_local']['y_forward']:.4f}",
                ])
        best_index = min(
            range(len(measurements)),
            key=lambda index: abs(
                measurements[index]["robot"]["bbox_cx"]
                - measurements[index]["robot"]["image_width"] / 2.0),
        )
        best = measurements[best_index]
        display = self.scan_frames[best_index].copy()
        x, y, w, h = best["bbox"]
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.imwrite(str(self.run_dir / "red_cube_final.jpg"), display)
        self._save_mission_result("RED_LOCKED", "red cube found during scan", final)

    def _save_mission_result(self, state, reason, cube_result=None):
        payload = {
            "timestamp": time.time(),
            "state": state,
            "reason": reason,
            "scan_attempts": self.scan_attempts,
            "return_attempts": self.return_attempts,
            "cube_result": cube_result,
        }
        (self.run_dir / "mission_result.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")

    def _show_camera(self, frame, red_pixels):
        try:
            display = frame.copy()
            if self.latest_bbox is not None:
                x, y, w, h = self.latest_bbox
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                display,
                f"state={self.state.value} scan={self.scan_attempts}/{MAX_SCAN_ATTEMPTS} "
                f"red={red_pixels} confirm={self.red_seen_count}/{RED_CONFIRM_FRAMES}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("TB4 Nav2 Scan Return", display)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.show_camera = False
            self.get_logger().warn(f"Camera window disabled: {exc}")

    def _start_scan(self):
        if self.home_pose is None:
            self._fail("Cannot scan: mission home has not been set")
            return
        if not self.spin_client.wait_for_server(timeout_sec=ACTION_SERVER_WAIT_SEC):
            self._fail("Spin action server unavailable")
            return
        self.scan_attempts += 1
        self.red_seen_count = 0
        self.scan_measurements = []
        self.scan_frames = []
        self.spin_goal_handle = None
        self.state = MissionState.SCANNING
        goal = Spin.Goal()
        goal.target_yaw = float(SCAN_SPIN_TARGET_YAW)
        goal.time_allowance.sec = 90
        future = self.spin_client.send_goal_async(goal)
        future.add_done_callback(self._scan_goal_response)
        self.get_logger().info(
            f"360deg scan sent: attempt={self.scan_attempts}/{MAX_SCAN_ATTEMPTS}.")

    def _scan_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._fail("Spin scan goal rejected")
            return
        self.spin_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self._scan_result)
        if self.state == MissionState.RED_LOCKED:
            self._cancel_scan_for_return()

    def _cancel_scan_for_return(self):
        if self.return_started:
            return
        if self.spin_goal_handle is None:
            return
        self.spin_goal_handle.cancel_goal_async().add_done_callback(self._scan_cancel_response)

    def _scan_cancel_response(self, future):
        if not future.result().goals_canceling:
            self._fail("Spin scan cancellation rejected")

    def _scan_result(self, future):
        status = future.result().status
        self.spin_goal_handle = None
        if self.state == MissionState.RED_LOCKED:
            if status in (GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_SUCCEEDED):
                self._send_return_goal()
            else:
                self._fail(f"Spin scan failed while stopping with status={status}")
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            if self.scan_attempts < MAX_SCAN_ATTEMPTS:
                self.get_logger().warn("Red cube not found. Starting second 360deg scan.")
                self._start_scan()
            else:
                self._fail("Red cube not found after two 360deg scans")
        else:
            self._fail(f"Spin scan failed with status={status}")

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
        if self.return_started:
            return
        if self.home_pose is None:
            self._fail("Cannot return: mission home has not been set")
            return
        self.return_started = True
        self.return_attempts += 1
        if not self.navigate_client.wait_for_server(timeout_sec=ACTION_SERVER_WAIT_SEC):
            self.return_started = False
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
            self.return_started = False
            self._retry_or_fail("Nav2 return goal rejected")
            return
        self.own_return_goal_ids.add(goal_uuid_key(goal_handle.goal_id))
        goal_handle.get_result_async().add_done_callback(self._return_result)

    def _return_result(self, future):
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.state = MissionState.DONE
            self._save_mission_result(
                "DONE", "returned home after red cube detection", self.final_cube_result)
            self.get_logger().info("Mission DONE: returned home.")
        else:
            self.return_started = False
            self._retry_or_fail(f"Nav2 return failed with status={status}")

    def _retry_or_fail(self, reason):
        if self.return_attempts >= MAX_RETURN_ATTEMPTS:
            self._fail(f"{reason}; retry exhausted")
            return
        self.get_logger().warn(f"{reason}; retrying in {self.return_retry_delay:.1f}s.")
        self.retry_timer = self.create_timer(self.return_retry_delay, self._retry_once)

    def _retry_once(self):
        if self.retry_timer is not None:
            self.retry_timer.cancel()
            self.destroy_timer(self.retry_timer)
            self.retry_timer = None
        self._send_return_goal()

    def _fail(self, reason):
        self.state = MissionState.FAILED
        self._save_mission_result("FAILED", reason)
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

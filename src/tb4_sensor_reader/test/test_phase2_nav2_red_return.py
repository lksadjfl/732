import inspect
import math
from types import SimpleNamespace

import cv2
import numpy as np
from builtin_interfaces.msg import Time

from tb4_sensor_reader.phase2_nav2_red_return import (
    MAX_RETURN_ATTEMPTS,
    MissionState,
    Phase2Nav2RedReturn,
    SPIN_TARGET_YAW,
    estimate_cube_robot_coordinates,
    find_red_bbox,
    map_to_task_local,
    robot_to_map,
)


class _Clock:
    def now(self):
        return SimpleNamespace(to_msg=lambda: Time(sec=12, nanosec=34))


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warn(self, message):
        self.messages.append(("warn", message))

    def error(self, message):
        self.messages.append(("error", message))


def test_red_bbox_requires_a_large_red_region():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (500, 280), (620, 400), (0, 0, 255), -1)
    bbox, mask = find_red_bbox(frame)

    assert bbox is not None
    assert cv2.countNonZero(mask) >= 5800

    small = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(small, (10, 10), (20, 20), (0, 0, 255), -1)
    assert find_red_bbox(small)[0] is None


def test_red_lock_requires_three_consecutive_frames():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (500, 280), (620, 400), (0, 0, 255), -1)
    sent_returns = []

    class _Bridge:
        def compressed_imgmsg_to_cv2(self, _msg, desired_encoding):
            assert desired_encoding == "bgr8"
            return frame

    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.state = MissionState.SEARCHING
    node.bridge = _Bridge()
    node.show_camera = False
    node.red_seen_count = 0
    node.red_locked = False
    node.current_pose = (0.0, 0.0, 0.0)
    node.home_pose = (0.0, 0.0, 0.0)
    node.get_logger = lambda: _Logger()
    node._save_final_evidence = lambda _result: None
    node._send_return_goal = lambda: sent_returns.append(True)

    node.image_callback(object())
    node.image_callback(object())
    assert sent_returns == []

    node.image_callback(object())
    assert sent_returns == [True]
    assert node.state == MissionState.RED_LOCKED


def test_bbox_range_estimate_is_centered_and_positive():
    result = estimate_cube_robot_coordinates((720, 1280, 3), (590, 300, 100, 100))
    assert math.isclose(result["x_right"], 0.0, abs_tol=1e-9)
    assert result["y_forward"] > 0.0
    assert math.isclose(result["distance"], result["y_forward"])


def test_map_coordinates_round_trip_through_home_frame():
    map_x, map_y = robot_to_map(2.0, 3.0, math.pi / 2.0, 0.4, 1.2)
    local_x, local_y = map_to_task_local(2.0, 3.0, math.pi / 2.0, map_x, map_y)
    assert math.isclose(local_x, 0.4)
    assert math.isclose(local_y, 1.2)


def test_home_goal_uses_recorded_map_pose():
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.home_pose = (1.25, -0.75, math.pi / 2.0)
    node.get_clock = lambda: _Clock()
    goal = node.build_home_goal()

    assert goal.pose.header.frame_id == "map"
    assert goal.pose.header.stamp.sec == 12
    assert math.isclose(goal.pose.pose.position.x, 1.25)
    assert math.isclose(goal.pose.pose.position.y, -0.75)
    assert math.isclose(goal.pose.pose.orientation.z, math.sin(math.pi / 4.0))
    assert math.isclose(goal.pose.pose.orientation.w, math.cos(math.pi / 4.0))


def test_initialpose_updates_home_before_return_but_amcl_does_not_drift_it():
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.state = MissionState.WAITING_FOR_HOME
    node.home_pose = None
    node.current_pose = None
    node.get_logger = lambda: _Logger()
    msg = SimpleNamespace(pose=SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )))
    node.initial_pose_callback(msg)
    home = node.home_pose

    msg.pose.pose.position.x = 7.0
    node.amcl_pose_callback(msg)
    assert node.home_pose == home
    assert node.current_pose[0] == 7.0


def test_return_failure_retries_once_then_fails():
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.return_attempts = 1
    node.return_retry_delay = 2.0
    node.retry_timer = None
    node.get_logger = lambda: _Logger()
    timers = []
    node.create_timer = lambda delay, callback: timers.append((delay, callback)) or object()
    node._retry_or_fail("first failure")
    assert len(timers) == 1

    node.return_attempts = MAX_RETURN_ATTEMPTS
    node._retry_or_fail("second failure")
    assert node.state == MissionState.FAILED


def test_spin_goal_requests_pi_radians():
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    sent = []
    node.spin_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True,
        send_goal_async=lambda goal: sent.append(goal) or SimpleNamespace(
            add_done_callback=lambda callback: None),
    )
    node._send_spin_goal()
    assert node.state == MissionState.SPINNING
    assert len(sent) == 1
    assert math.isclose(sent[0].target_yaw, SPIN_TARGET_YAW)


def test_coordinator_does_not_publish_twist_commands():
    source = inspect.getsource(Phase2Nav2RedReturn)
    assert "create_publisher" not in source
    assert "cmd_vel" not in source

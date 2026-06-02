import inspect
import math
from types import SimpleNamespace

import cv2
import numpy as np
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time

from tb4_sensor_reader.phase2_nav2_red_return import (
    MAX_RETURN_ATTEMPTS,
    MissionState,
    Phase2Nav2RedReturn,
    SCAN_SPIN_TARGET_YAW,
    estimate_cube_robot_coordinates,
    find_red_bbox,
    map_to_task_local,
    median_cube_result,
    robot_to_map,
)


class _Clock:
    def now(self):
        return SimpleNamespace(to_msg=lambda: Time(sec=12, nanosec=34))


class _Logger:
    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, _message):
        pass


def _uuid(value):
    return SimpleNamespace(uuid=[value] * 16)


def _status(value, status):
    return SimpleNamespace(goal_info=SimpleNamespace(goal_id=_uuid(value)), status=status)


def _result(status):
    return SimpleNamespace(result=lambda: SimpleNamespace(status=status))


def test_red_bbox_requires_a_large_red_region():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (500, 280), (620, 400), (0, 0, 255), -1)
    bbox, mask = find_red_bbox(frame)
    assert bbox is not None
    assert cv2.countNonZero(mask) >= 5800

    small = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(small, (10, 10), (20, 20), (0, 0, 255), -1)
    assert find_red_bbox(small)[0] is None


def test_searching_does_not_lock_red_cube():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (500, 280), (620, 400), (0, 0, 255), -1)
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.state = MissionState.SEARCHING
    node.bridge = SimpleNamespace(
        compressed_imgmsg_to_cv2=lambda _msg, desired_encoding: frame)
    node.show_camera = False
    node.latest_frame = None
    node.latest_bbox = None
    node.red_seen_count = 0
    node.image_callback(object())
    assert node.state == MissionState.SEARCHING
    assert node.red_seen_count == 0


def test_scanning_red_lock_requires_three_consecutive_frames():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (500, 280), (620, 400), (0, 0, 255), -1)
    canceled = []
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.state = MissionState.SCANNING
    node.bridge = SimpleNamespace(
        compressed_imgmsg_to_cv2=lambda _msg, desired_encoding: frame)
    node.show_camera = False
    node.latest_frame = None
    node.latest_bbox = None
    node.red_seen_count = 0
    node.scan_measurements = []
    node.scan_frames = []
    node.get_logger = lambda: _Logger()
    node._build_cube_measurement = lambda _pixels: {"value": True}
    node._save_final_evidence = lambda measurements: None
    node._cancel_scan_for_return = lambda: canceled.append(True)
    node.image_callback(object())
    node.image_callback(object())
    assert canceled == []
    node.image_callback(object())
    assert canceled == [True]
    assert node.state == MissionState.RED_LOCKED


def test_bbox_range_estimate_includes_legacy_fields():
    result = estimate_cube_robot_coordinates((720, 1280, 3), (590, 300, 100, 100))
    assert math.isclose(result["x_right"], 0.0, abs_tol=1e-9)
    assert result["y_forward"] > 0.0
    assert math.isclose(result["distance"], result["y_forward"])
    assert "bbox_cx" in result
    assert "z_from_width" in result
    assert "z_from_height" in result
    assert "bearing_rad" in result
    assert "z_camera" in result


def test_map_coordinates_round_trip_through_home_frame():
    map_x, map_y = robot_to_map(2.0, 3.0, math.pi / 2.0, 0.4, 1.2)
    local_x, local_y = map_to_task_local(2.0, 3.0, math.pi / 2.0, map_x, map_y)
    assert math.isclose(local_x, 0.4)
    assert math.isclose(local_y, 1.2)


def test_median_cube_result_uses_three_measurements():
    measurements = []
    for value in (1.0, 9.0, 3.0):
        measurements.append({
            "robot": {
                "x_right": value, "y_forward": value + 1, "z_camera": value + 2,
                "distance": value + 3, "bearing_rad": value + 4,
            },
            "task_local": {"x_right": value + 5, "y_forward": value + 6},
            "map": {"x": value + 7, "y": value + 8},
        })
    result = median_cube_result(measurements)
    assert result["measurement_count"] == 3
    assert result["robot"]["x_right"] == 3.0
    assert result["task_local"]["x_right"] == 8.0
    assert result["map"]["y"] == 11.0


def test_final_evidence_saves_three_confirmation_photos(tmp_path):
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.home_pose = (0.0, 0.0, 0.0)
    node.run_dir = tmp_path
    node.csv_path = tmp_path / "red_cube_measurements.csv"
    node.csv_path.write_text("", encoding="utf-8")
    node.scan_frames = [
        np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(3)]
    node._save_mission_result = lambda state, reason, cube=None: None
    measurements = []
    for value in (1.0, 2.0, 3.0):
        measurements.append({
            "timestamp": value,
            "scan_attempt": 1,
            "bbox": (50, 40, 30, 30),
            "robot": {
                "bbox_cx": 65.0, "bbox_cy": 55.0,
                "image_width": 160, "image_height": 120,
                "fx_px": 100.0, "fy_px": 100.0,
                "z_from_width": value, "z_from_height": value,
                "distance_forward": value, "x_right": value,
                "y_forward": value, "z_camera": value,
                "distance": value, "bearing_rad": value,
            },
            "map": {"x": value, "y": value},
            "task_local": {"x_right": value, "y_forward": value},
        })
    node._save_final_evidence(measurements)
    assert (tmp_path / "red_cube_confirm_01.jpg").exists()
    assert (tmp_path / "red_cube_confirm_02.jpg").exists()
    assert (tmp_path / "red_cube_confirm_03.jpg").exists()
    assert (tmp_path / "red_cube_final.jpg").exists()


def test_external_search_goal_success_starts_scan():
    started = []
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.state = MissionState.SEARCHING
    node.own_return_goal_ids = set()
    node.external_goal_ids = set()
    node.handled_external_goal_ids = set()
    node.get_logger = lambda: _Logger()
    node._start_scan = lambda: started.append(True)
    node.navigate_status_callback(SimpleNamespace(status_list=[
        _status(3, GoalStatus.STATUS_EXECUTING)]))
    node.navigate_status_callback(SimpleNamespace(status_list=[
        _status(3, GoalStatus.STATUS_SUCCEEDED)]))
    assert started == [True]


def test_coordinator_return_goal_success_does_not_start_scan():
    started = []
    key = tuple([4] * 16)
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.state = MissionState.SEARCHING
    node.own_return_goal_ids = {key}
    node.external_goal_ids = set()
    node.handled_external_goal_ids = set()
    node._start_scan = lambda: started.append(True)
    node.navigate_status_callback(SimpleNamespace(status_list=[
        _status(4, GoalStatus.STATUS_SUCCEEDED)]))
    assert started == []


def test_start_scan_requests_two_pi_radians():
    sent = []
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.home_pose = (0.0, 0.0, 0.0)
    node.scan_attempts = 0
    node.spin_client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True,
        send_goal_async=lambda goal: sent.append(goal) or SimpleNamespace(
            add_done_callback=lambda callback: None),
    )
    node.get_logger = lambda: _Logger()
    node._start_scan()
    assert node.state == MissionState.SCANNING
    assert math.isclose(sent[0].target_yaw, SCAN_SPIN_TARGET_YAW)


def test_first_scan_miss_retries_and_second_scan_miss_fails():
    started = []
    failed = []
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.state = MissionState.SCANNING
    node.spin_goal_handle = object()
    node.scan_attempts = 1
    node.get_logger = lambda: _Logger()
    node._start_scan = lambda: started.append(True)
    node._fail = lambda reason: failed.append(reason)
    node._scan_result(_result(GoalStatus.STATUS_SUCCEEDED))
    assert started == [True]

    node.state = MissionState.SCANNING
    node.spin_goal_handle = object()
    node.scan_attempts = 2
    node._scan_result(_result(GoalStatus.STATUS_SUCCEEDED))
    assert "not found" in failed[0]


def test_red_lock_canceled_scan_sends_home_goal():
    sent = []
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.state = MissionState.RED_LOCKED
    node.spin_goal_handle = object()
    node._send_return_goal = lambda: sent.append(True)
    node._fail = lambda reason: None
    node._scan_result(_result(GoalStatus.STATUS_CANCELED))
    assert sent == [True]


def test_home_goal_uses_recorded_map_pose():
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.home_pose = (1.25, -0.75, math.pi / 2.0)
    node.get_clock = lambda: _Clock()
    goal = node.build_home_goal()
    assert goal.pose.header.frame_id == "map"
    assert math.isclose(goal.pose.pose.position.x, 1.25)
    assert math.isclose(goal.pose.pose.position.y, -0.75)


def test_return_success_goes_directly_to_done():
    saved = []
    node = Phase2Nav2RedReturn.__new__(Phase2Nav2RedReturn)
    node.final_cube_result = {"task_local": {"x_right": 1.0, "y_forward": 2.0}}
    node._save_mission_result = lambda state, reason, cube=None: saved.append((state, cube))
    node.get_logger = lambda: _Logger()
    node._return_result(_result(GoalStatus.STATUS_SUCCEEDED))
    assert node.state == MissionState.DONE
    assert saved == [("DONE", node.final_cube_result)]


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
    node._fail = lambda reason: setattr(node, "state", MissionState.FAILED)
    node._retry_or_fail("second failure")
    assert node.state == MissionState.FAILED


def test_coordinator_does_not_publish_twist_commands():
    source = inspect.getsource(Phase2Nav2RedReturn)
    assert "create_publisher" not in source
    assert "cmd_vel" not in source

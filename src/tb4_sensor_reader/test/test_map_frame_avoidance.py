import math
from types import SimpleNamespace

from builtin_interfaces.msg import Time

from tb4_sensor_reader.map_frame_avoidance import (
    CELL_SIZE,
    MAP_FRAME,
    MissionState,
    MapFrameAvoidance,
    NAMESPACE,
    NAV2_GOAL_FRAME,
    RETURN_MODE,
)


class _FakeClock:
    def now(self):
        return SimpleNamespace(to_msg=lambda: Time(sec=123, nanosec=456))


def test_namespace_is_hard_coded_to_t13():
    assert NAMESPACE == 'T27'


def test_default_return_mode_prefers_nav2_placeholder():
    assert RETURN_MODE == 'nav2'
    assert NAV2_GOAL_FRAME in {'odom', MAP_FRAME}


def test_mission_states_include_snapshot_and_backtracking():
    assert MissionState.SEARCHING.value == 'SEARCHING'
    assert MissionState.BACKTRACKING.value == 'BACKTRACKING'
    assert MissionState.SNAPSHOT.value == 'SNAPSHOT'
    assert MissionState.RETURNING.value == 'RETURNING'
    assert MissionState.DONE.value == 'DONE'


def test_world_to_cell_rounds_by_cell_size():
    node = MapFrameAvoidance.__new__(MapFrameAvoidance)
    assert node.world_to_cell(0.0, 0.0) == (0, 0)
    assert node.world_to_cell(CELL_SIZE * 0.51, CELL_SIZE * 1.49) == (1, 1)
    assert node.world_to_cell(-CELL_SIZE * 0.51, -CELL_SIZE * 1.49) == (-1, -1)


def test_normalize_angle_wraps_to_pi_range():
    assert math.isclose(MapFrameAvoidance.normalize_angle(3 * math.pi), math.pi)
    assert math.isclose(MapFrameAvoidance.normalize_angle(-3 * math.pi), -math.pi)


def test_build_nav2_goal_pose_targets_origin_with_origin_heading():
    node = MapFrameAvoidance.__new__(MapFrameAvoidance)
    node.nav2_goal_frame = 'odom'
    node.origin_x = 1.25
    node.origin_y = -0.75
    node.origin_yaw = math.pi / 2.0
    node.get_clock = lambda: _FakeClock()

    pose = node.build_nav2_goal_pose()

    assert pose.header.frame_id == 'odom'
    assert pose.header.stamp.sec == 123
    assert pose.header.stamp.nanosec == 456
    assert math.isclose(pose.pose.position.x, 1.25)
    assert math.isclose(pose.pose.position.y, -0.75)
    assert math.isclose(pose.pose.orientation.z, math.sin(math.pi / 4.0))
    assert math.isclose(pose.pose.orientation.w, math.cos(math.pi / 4.0))


def test_activate_breadcrumb_return_switches_mode_and_reason():
    node = MapFrameAvoidance.__new__(MapFrameAvoidance)
    node.return_mode = 'nav2'
    node.nav2_fallback_reason = None
    node.get_logger = lambda: SimpleNamespace(warn=lambda msg: None)

    node.activate_breadcrumb_return('server unavailable')

    assert node.return_mode == 'breadcrumbs'
    assert node.nav2_fallback_reason == 'server unavailable'

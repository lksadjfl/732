#!/usr/bin/env python3
"""
Phase 2 — 双模式启动：自主搜索 或 Nav2 手动导航。

用法：
  自主模式:  ros2 launch tb4_sensor_reader phase2_with_map.launch.py mode:=auto
  Nav2模式:  ros2 launch tb4_sensor_reader phase2_with_map.launch.py mode:=nav2

必须先单独开 RViz。auto 模式还需要相机，nav2 手动导航不需要:
  ros2 launch turtlebot4_viz view_robot.launch.py namespace:=/T27
  ros2 launch tb4_sensor_reader oakd_left_right.launch.py namespace:=T27
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    map_yaml = LaunchConfiguration('map')
    phase1_memory_file = LaunchConfiguration('phase1_memory_file')
    mode = LaunchConfiguration('mode')
    nav2_start_delay = LaunchConfiguration('nav2_start_delay')

    declare_namespace = DeclareLaunchArgument(
        'namespace', default_value='/T27')
    declare_map = DeclareLaunchArgument(
        'map', default_value=os.path.expanduser('~/phase1_env_data_map.yaml'))
    declare_phase1_memory_file = DeclareLaunchArgument(
        'phase1_memory_file',
        default_value=os.path.expanduser('~/phase1_navigation_memory.json'),
        description='Phase 1 safe_path / return_path JSON used by autonomous mode')
    declare_mode = DeclareLaunchArgument(
        'mode', default_value='auto',
        choices=['auto', 'nav2'],
        description='auto=Phase2自主搜索 | nav2=Nav2手动导航')
    declare_nav2_start_delay = DeclareLaunchArgument(
        'nav2_start_delay', default_value='12.0',
        description='Delay Nav2 startup so map_server and AMCL can finish lifecycle bringup')

    pkg_tb4_nav = '/opt/ros/humble/share/turtlebot4_navigation'

    # ---- 公共：localization（map_server + AMCL） ----
    localization = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg_tb4_nav, 'launch', 'localization.launch.py'])),
            launch_arguments={
                'namespace': namespace,
                'map': map_yaml,
            }.items(),
        ),
    ])

    # ---- auto 模式：Phase 2 自主节点 ----
    auto_mode = GroupAction(
        condition=IfCondition(PythonExpression(["'", mode, "' == 'auto'"])),
        actions=[
            Node(
                package='tb4_sensor_reader',
                executable='map_frame_avoidance',
                name='phase2_autonomous',
                output='screen',
                parameters=[{
                    'namespace': namespace,
                    'phase1_memory_file': phase1_memory_file,
                }],
            ),
        ])

    # ---- nav2 模式：Nav2 手动导航 ----
    nav2_mode = GroupAction(
        condition=IfCondition(PythonExpression(["'", mode, "' == 'nav2'"])),
        actions=[
            TimerAction(
                period=nav2_start_delay,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            PathJoinSubstitution([pkg_tb4_nav, 'launch', 'nav2.launch.py'])),
                        launch_arguments={
                            'namespace': namespace,
                        }.items(),
                    ),
                ]),
        ])

    ld = LaunchDescription()
    ld.add_action(declare_namespace)
    ld.add_action(declare_map)
    ld.add_action(declare_phase1_memory_file)
    ld.add_action(declare_mode)
    ld.add_action(declare_nav2_start_delay)
    ld.add_action(localization)
    ld.add_action(auto_mode)
    ld.add_action(nav2_mode)
    return ld

#!/usr/bin/env python3
"""Launch Phase 2 map localization, Nav2, and red-cube automatic return."""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    map_yaml = LaunchConfiguration("map")
    image_topic = LaunchConfiguration("image_topic")
    nav2_start_delay = LaunchConfiguration("nav2_start_delay")
    return_retry_delay = LaunchConfiguration("return_retry_delay")
    bond_timeout = LaunchConfiguration("bond_timeout")

    package_share = FindPackageShare("tb4_sensor_reader")
    turtlebot4_navigation = FindPackageShare("turtlebot4_navigation")
    localization_params = PathJoinSubstitution([
        package_share, "config", "localization_phase2.yaml"])
    nav2_params = PathJoinSubstitution([
        package_share, "config", "nav2_phase2.yaml"])

    localization = GroupAction(
        actions=[
            SetParameter(name="bond_timeout", value=bond_timeout),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution([
                    turtlebot4_navigation, "launch", "localization.launch.py"])),
                launch_arguments={
                    "namespace": namespace,
                    "map": map_yaml,
                    "params": localization_params,
                }.items(),
            ),
        ],
    )

    nav2 = TimerAction(
        period=nav2_start_delay,
        actions=[
            GroupAction(
                actions=[
                    SetParameter(name="bond_timeout", value=bond_timeout),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(PathJoinSubstitution([
                            turtlebot4_navigation, "launch", "nav2.launch.py"])),
                        launch_arguments={
                            "namespace": namespace,
                            "params_file": nav2_params,
                        }.items(),
                    ),
                ],
            ),
        ],
    )

    coordinator = Node(
        package="tb4_sensor_reader",
        executable="phase2_nav2_red_return",
        name="phase2_nav2_red_return",
        output="screen",
        parameters=[{
            "namespace": namespace,
            "image_topic": image_topic,
            "return_retry_delay": return_retry_delay,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value="/T13"),
        DeclareLaunchArgument(
            "map",
            default_value=os.path.expanduser("~/ros2_ws/maps/phase1_env_data_map.yaml"),
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="",
        ),
        DeclareLaunchArgument("nav2_start_delay", default_value="20.0"),
        DeclareLaunchArgument("return_retry_delay", default_value="2.0"),
        DeclareLaunchArgument(
            "bond_timeout",
            default_value="0.0",
            description=(
                "Lifecycle bond timeout in seconds. Zero disables heartbeat "
                "monitoring for discovery-server lab environments."
            ),
        ),
        localization,
        nav2,
        coordinator,
    ])

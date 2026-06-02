#!/usr/bin/env python3
"""
Phase 1 data collection without camera startup.

Run this alongside SLAM, RViz, and teleoperation. It saves:
  - phase1_env_data.json and phase1_env_data_map.yaml/.pgm
  - phase1_navigation_memory.json and its debug summary/image
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    output_dir = LaunchConfiguration("output_dir")

    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value="/T21"),
        DeclareLaunchArgument(
            "output_dir",
            default_value=os.path.expanduser("~/ros2_ws/maps"),
            description="Directory for Phase 1 map and navigation-memory outputs",
        ),
        Node(
            package="tb4_sensor_reader",
            executable="phase1_env_mapper",
            name="phase1_env_mapper",
            output="screen",
            parameters=[{
                "namespace": namespace,
                "output_dir": output_dir,
            }],
        ),
        Node(
            package="tb4_sensor_reader",
            executable="phase1_map_data_collector",
            name="phase1_map_data_collector",
            output="screen",
            parameters=[{
                "namespace": namespace,
                "output_dir": output_dir,
            }],
        ),
    ])

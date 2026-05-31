from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    declare_namespace = DeclareLaunchArgument(
        'namespace', default_value='T27',
        description='Robot namespace (without leading /)')

    namespace = LaunchConfiguration('namespace')

    params_file = PathJoinSubstitution([
        FindPackageShare('tb4_sensor_reader'),
        'config',
        'oakd_left_right.yaml',
    ])

    depthai_camera_launch = PathJoinSubstitution([
        FindPackageShare('depthai_ros_driver'),
        'launch',
        'camera.launch.py',
    ])

    return LaunchDescription([
        declare_namespace,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(depthai_camera_launch),
            launch_arguments={
                'name': 'oak',
                'namespace': namespace,
                'camera_model': 'OAK-D-PRO',
                'params_file': params_file,
                'rs_compat': 'true',
                'enable_color': 'true',
                'enable_depth': 'true',
                'enable_infra1': 'true',
                'enable_infra2': 'true',
                'rectify_rgb': 'true',
                'pointcloud.enable': 'false',
                'depth_module.depth_profile': '1280,720,10',
                'depth_module.infra_profile': '1280,720,10',
                'rgb_camera.color_profile': '1280,720,10',
            }.items(),
        ),
    ])

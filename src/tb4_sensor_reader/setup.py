from setuptools import find_packages, setup

package_name = 'tb4_sensor_reader'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jxia219',
    maintainer_email='jxia219@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    scripts=['scripts/map_frame_avoidance'],
    entry_points={
    'console_scripts': [
        'odom_reader = tb4_sensor_reader.odom_reader:main',
        'motion_controller = tb4_sensor_reader.motion_controller:main',
        'reactive_controller = tb4_sensor_reader.reactive_controller:main',
        'avoid_controller = tb4_sensor_reader.avoid_controller:main',
        'test_node = tb4_sensor_reader.test_node_template:main',
        'camera_c2_node = tb4_sensor_reader.camera_c2_node:main',
        'pose_reader = tb4_sensor_reader.pose_reader:main',
        'physical_motion = tb4_sensor_reader.physical_motion:main',
        'reactive_physical = tb4_sensor_reader.reactive_physical:main',
        'avoidance_physical = tb4_sensor_reader.avoidance_physical:main',
        'camera_viewer = tb4_sensor_reader.camera_viewer:main',
        'camera_detector = tb4_sensor_reader.camera_detector:main',
        'detect_and_stop = tb4_sensor_reader.detect_and_stop:main',
        'map_frame_avoidance = tb4_sensor_reader.map_frame_avoidance:main',
    	],
},
)

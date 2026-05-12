#!/usr/bin/env python3

import math
import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# Change this to your robot namespace
NAMESPACE = '/T13'

# Direction to monitor: robot front = -pi/2
TARGET_ANGLE = -math.pi / 2.0

# Phase durations (seconds)
MANUAL_ROTATE_RECORD_SECONDS = 20.0   # You manually rotate robot during this phase
STATIONARY_RECORD_SECONDS = 5.0       # Robot remains still during this phase

# Output file in home directory
OUTPUT_PATH = os.path.expanduser('~/lidar_b2_manual_rotate_results.txt')


class EnvironmentSnapshotNode(Node):
    def __init__(self):
        super().__init__('environment_snapshot_node')

        self.scan_sub = self.create_subscription(
            LaserScan,
            f'{NAMESPACE}/scan',
            self.scan_callback,
            10
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            f'{NAMESPACE}/cmd_vel',
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.phase = 0
        self.phase_start_time = time.time()
        self.last_scan = None
        self.has_written = False

        # Phase 0: you manually rotate the robot while node records data
        self.manual_stats = self.new_stats_dict('manual_rotation_phase')
        # Phase 1: robot stands still and node records data
        self.stationary_stats = self.new_stats_dict('stationary_phase')

        self.get_logger().info(f'Subscribing to {NAMESPACE}/scan')
        self.get_logger().info(
            'Phase 0: robot will NOT rotate automatically. '
            'Please manually rotate the robot about 360 degrees while data is recorded.'
        )

    def new_stats_dict(self, name):
        return {
            'name': name,
            'scan_count': 0,
            'beam_counts': [],
            'min_valid_observed': float('inf'),
            'max_valid_observed': float('-inf'),
            'straight_ahead_values': [],
            'anomalies': [],
        }

    def stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def active_stats(self):
        if self.phase == 0:
            return self.manual_stats
        if self.phase == 1:
            return self.stationary_stats
        return None

    def scan_callback(self, msg):
        self.last_scan = msg
        stats = self.active_stats()
        if stats is None:
            return

        n_beams = len(msg.ranges)
        stats['scan_count'] += 1
        stats['beam_counts'].append(n_beams)

        if n_beams == 0:
            stats['anomalies'].append('empty scan received')
            return

        valid_ranges = [
            r for r in msg.ranges
            if math.isfinite(r) and msg.range_min <= r <= msg.range_max
        ]

        if valid_ranges:
            min_valid_range = min(valid_ranges)
            max_valid_range = max(valid_ranges)
            stats['min_valid_observed'] = min(stats['min_valid_observed'], min_valid_range)
            stats['max_valid_observed'] = max(stats['max_valid_observed'], max_valid_range)
        else:
            stats['anomalies'].append('no valid ranges in scan')

        inf_count = sum(1 for r in msg.ranges if math.isinf(r))
        nan_count = sum(1 for r in msg.ranges if math.isnan(r))
        invalid_count = sum(
            1 for r in msg.ranges
            if (not math.isfinite(r)) or (r < msg.range_min) or (r > msg.range_max)
        )

        if inf_count > 0:
            stats['anomalies'].append(f'inf values observed ({inf_count})')
        if nan_count > 0:
            stats['anomalies'].append(f'nan values observed ({nan_count})')
        if invalid_count > 0:
            stats['anomalies'].append(f'invalid beams observed ({invalid_count})')

        idx = int(round((TARGET_ANGLE - msg.angle_min) / msg.angle_increment))
        idx = max(0, min(n_beams - 1, idx))
        straight_ahead_range = msg.ranges[idx]

        if math.isfinite(straight_ahead_range) and msg.range_min <= straight_ahead_range <= msg.range_max:
            stats['straight_ahead_values'].append(straight_ahead_range)
        else:
            stats['anomalies'].append('straight-ahead beam invalid/out of range')

        self.get_logger().info(
            f'phase={self.phase} | total_beams={n_beams} | '
            f'min_valid={min(valid_ranges):.3f} m ' if valid_ranges else f'phase={self.phase} | total_beams={n_beams} | min_valid=nan | '
        )
        if valid_ranges:
            self.get_logger().info(
                f'phase={self.phase} | max_valid={max(valid_ranges):.3f} m | '
                f'straight_ahead(-pi/2)_index={idx} | '
                f'straight_ahead(-pi/2)={straight_ahead_range:.3f} m'
            )
        else:
            self.get_logger().info(
                f'phase={self.phase} | max_valid=nan | '
                f'straight_ahead(-pi/2)_index={idx} | '
                f'straight_ahead(-pi/2)={straight_ahead_range}'
            )

    def summarize_stats(self, stats):
        beam_value = stats['beam_counts'][-1] if stats['beam_counts'] else 'N/A'

        if stats['min_valid_observed'] == float('inf'):
            min_valid_text = 'N/A'
        else:
            min_valid_text = f"{stats['min_valid_observed']:.3f} m"

        if stats['max_valid_observed'] == float('-inf'):
            max_valid_text = 'N/A'
        else:
            max_valid_text = f"{stats['max_valid_observed']:.3f} m"

        if stats['straight_ahead_values']:
            straight_ahead_text = (
                f"latest={stats['straight_ahead_values'][-1]:.3f} m, "
                f"min={min(stats['straight_ahead_values']):.3f} m, "
                f"max={max(stats['straight_ahead_values']):.3f} m"
            )
        else:
            straight_ahead_text = 'N/A'

        if stats['anomalies']:
            unique_anomalies = []
            for item in stats['anomalies']:
                if item not in unique_anomalies:
                    unique_anomalies.append(item)
            anomalies_text = '; '.join(unique_anomalies)
        else:
            anomalies_text = 'none'

        lines = [
            f"[{stats['name']} ]",
            f"Total scans recorded: {stats['scan_count']}",
            f"Total beams per scan: {beam_value}",
            f"Minimum valid range observed: {min_valid_text}",
            f"Maximum valid range observed: {max_valid_text}",
            f"Range straight ahead (-1/2pi): {straight_ahead_text}",
            f"Any anomalies noted: {anomalies_text}",
            ''
        ]
        return '\n'.join(lines)

    def write_results(self):
        text = []
        text.append('LiDAR B2 phased results')
        text.append(f'Namespace: {NAMESPACE}')
        text.append(f'Target direction: -pi/2 ({TARGET_ANGLE:.6f} rad)')
        text.append(f'Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        text.append('')
        text.append(self.summarize_stats(self.manual_stats))
        text.append(self.summarize_stats(self.stationary_stats))

        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text))

        self.get_logger().info(f'Results written to {OUTPUT_PATH}')

    def control_loop(self):
        # Do not rotate robot automatically in either phase.
        self.stop_robot()

        now = time.time()
        elapsed = now - self.phase_start_time

        if self.phase == 0 and elapsed >= MANUAL_ROTATE_RECORD_SECONDS:
            self.phase = 1
            self.phase_start_time = now
            self.get_logger().info(
                'Phase 1 started: keep robot stationary now. Recording stationary data.'
            )
            return

        if self.phase == 1 and elapsed >= STATIONARY_RECORD_SECONDS:
            if not self.has_written:
                self.write_results()
                self.has_written = True
                self.get_logger().info('All phases complete. Press Ctrl+C to exit.')
            self.phase = 2


def main(args=None):
    rclpy.init(args=args)
    node = EnvironmentSnapshotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        if not node.has_written:
            node.write_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

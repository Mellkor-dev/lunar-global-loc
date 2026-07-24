#!/usr/bin/env python3
"""
Capture ONE PointCloud2 message from a ROS2 topic and save it as a plain
numpy array (.npy) of shape (N, 3) [x, y, z], or (N, 4) if --intensity is
set and the message actually carries an intensity field.

Usage:
    python3 sim/capture_cloud.py --out scan_site_01.npy
"""


import argparse
import sys
from pathlib import Path
import time
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
OUT_DIR = PROJECT_ROOT / "LargeScale_Implement" / "sim" / "pointcloud_scans"

class SingleScanSaver(Node):
    def __init__(self, out_path: str, include_intensity: bool):
        super().__init__("single_scan_saver")
        self.out_path = out_path
        self.include_intensity = include_intensity
        self.saved = False
        self.create_subscription(PointCloud2, "/pointcloud", self.callback, 10)
        self.get_logger().info(f"Waiting for one message on /pointcloud ...")

    def callback(self, msg: PointCloud2):
        if self.saved:
            return
        fields = ("x", "y", "z", "intensity") if self.include_intensity else ("x", "y", "z")
        try:
            points = np.array(list(pc2.read_points(msg, field_names=fields, skip_nans=True)))
        except Exception as e:
            self.get_logger().warn(f"requested fields not available ({e}), falling back to xyz only")
            points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

        np.savez(OUT_DIR / self.out_path, points)
        self.get_logger().info(f"Saved {points.shape[0]} points -> {OUT_DIR / self.out_path}")
        self.saved = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)    
    parser.add_argument("--out", required=True, help="Output .npy path")
    parser.add_argument("--intensity", action="store_true", help="Also save intensity as a 4th column")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait before giving up")
    args = parser.parse_args()

    rclpy.init()
    node = SingleScanSaver(args.out, args.intensity)

    start = time.time()
    while rclpy.ok() and not node.saved:
        rclpy.spin_once(node, timeout_sec=0.5)
        if time.time() - start > args.timeout:
            node.get_logger().error(f"Timed out after {args.timeout}s waiting on /pointcloud")
            break

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if node.saved else 1)


if __name__ == "__main__":
    main()
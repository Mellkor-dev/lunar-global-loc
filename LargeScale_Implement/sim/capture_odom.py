#!/usr/bin/env python3

import argparse
import sys
import os
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

# Fix 1: Properly resolve 'lunar-global-loc' repository root directory (2 levels up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fix 2: Construct the absolute target directory explicitly
OUT_DIR = PROJECT_ROOT / "sim" / "odom_scans"

class OdomSaver(Node):
    def __init__(self, out_path: str):
        super().__init__("odom_saver")
        # Ensure we only grab the filename to prevent folder-escaping
        self.out_filename = Path(out_path).name
        self.saved = False
        self.create_subscription(Odometry, "/odom", self.callback, 10)
        self.get_logger().info("Waiting for one message on /odom ...")

    def callback(self, msg: Odometry):
        if self.saved:
            return
        
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        odom_data = np.array([position.x, position.y, position.z,
                              orientation.x, orientation.y, orientation.z, orientation.w])
        
        # Ensure the destination folder structures physically exist
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_filepath = OUT_DIR / self.out_filename
        
        np.savez(save_filepath, odom_data)
        self.get_logger().info(f"Saved odometry data -> {save_filepath}")
        self.saved = True
        
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output .npy filename")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait before giving up")
    args = parser.parse_args()

    rclpy.init()
    node = OdomSaver(args.out)
    
    start = time.time()
    while rclpy.ok() and not node.saved:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - start > args.timeout:
            node.get_logger().error(f"Timed out after {args.timeout}s waiting on /odom")
            break
    
    # Clean up node and context layers safely
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass
        
    # Fix 3: Force-exit using os._exit to bypass CycloneDDS cleanup deadlock hangs
    sys.stdout.flush()
    os._exit(0 if node.saved else 1)

if __name__ == "__main__":
    main()
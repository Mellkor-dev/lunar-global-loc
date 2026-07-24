#!/usr/bin/env python3

import argparse
import numpy as np
import rclpy

from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation


class TransformSaver(Node):

    def __init__(self, target_frame, source_frame, out_file):
        super().__init__("transform_saver")

        self.target_frame = target_frame
        self.source_frame = source_frame
        self.out_file = out_file

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

    def save_transform(self):

        try:

            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time()
            )

            t = tf.transform.translation
            q = tf.transform.rotation

            quat = np.array([
                q.x,
                q.y,
                q.z,
                q.w
            ])

            R = Rotation.from_quat(quat).as_matrix()

            T = np.eye(4)

            T[:3, :3] = R

            T[:3, 3] = [
                t.x,
                t.y,
                t.z
            ]

            np.save(self.out_file, T)

            self.get_logger().info(
                f"Saved transform matrix to {self.out_file}"
            )

            print("\nTransform matrix:\n")
            print(T)

            return True

        except Exception as e:

            self.get_logger().warn(
                f"Transform unavailable: {e}"
            )

            return False
        
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        default="odom"
    )

    parser.add_argument(
        "--source",
        default="vlp16"
    )

    parser.add_argument(
        "--out",
        default="odom_to_vlp16.npy"
    )

    args = parser.parse_args()

    rclpy.init()

    node = TransformSaver(
        args.target,
        args.source,
        args.out
    )

    success = False

    while rclpy.ok() and not success:

        rclpy.spin_once(
            node,
            timeout_sec=0.1
        )

        success = node.save_transform()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
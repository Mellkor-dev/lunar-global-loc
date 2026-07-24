#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
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

        self.out_file = Path(out_file)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.get_logger().info(
            f"Waiting for transform "
            f"{target_frame} -> {source_frame}"
        )

    def save_transform(self):

        try:

            if not self.tf_buffer.can_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time(),
            ):
                return False

            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time(),
            )

            t = tf.transform.translation
            q = tf.transform.rotation

            translation = np.array([
                t.x,
                t.y,
                t.z
            ])

            quaternion = np.array([
                q.x,
                q.y,
                q.z,
                q.w
            ])

            R = Rotation.from_quat(
                quaternion
            ).as_matrix()

            T = np.eye(4)

            T[:3, :3] = R
            T[:3, 3] = translation

            # Save binary version
            np.savez(
                PROJECT_ROOT / "sim" /"transform_scan" / self.out_file,
                T=T,
                translation=translation,
                quaternion=quaternion,
                target_frame=self.target_frame,
                source_frame=self.source_frame,
            )

            # Save human-readable matrix
            txt_file = PROJECT_ROOT / "sim" / "transform_scan" / self.out_file.with_suffix(".txt")

            with open(txt_file, "w") as f:

                f.write(
                    f"Target frame : {self.target_frame}\n"
                )

                f.write(
                    f"Source frame : {self.source_frame}\n\n"
                )

                f.write(
                    "Translation (m)\n"
                )

                f.write(
                    str(translation) + "\n\n"
                )

                f.write(
                    "Quaternion (x y z w)\n"
                )

                f.write(
                    str(quaternion) + "\n\n"
                )

                f.write(
                    "Homogeneous Transform Matrix\n"
                )

                np.savetxt(
                    f,
                    T,
                    fmt="%.10f"
                )

            self.get_logger().info(
                f"Saved transform -> {self.out_file}"
            )

            print("\nTranslation:")
            print(translation)

            print("\nQuaternion:")
            print(quaternion)

            print("\nTransform Matrix:")
            print(T)

            return True

        except Exception as e:

            self.get_logger().error(
                f"Failed: {e}"
            )

            return False


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        default="odom",
        help="Target frame"
    )

    parser.add_argument(
        "--source",
        default="vlp16",
        help="Source frame"
    )

    parser.add_argument(
        "--out",
        default="odom_to_vlp16.npz",
        help="Output NPZ file"
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds"
    )

    args = parser.parse_args()

    rclpy.init()

    node = TransformSaver(
        args.target,
        args.source,
        args.out
    )

    start_time = time.time()

    success = False

    while rclpy.ok() and not success:

        rclpy.spin_once(
            node,
            timeout_sec=0.1
        )

        success = node.save_transform()

        if time.time() - start_time > args.timeout:

            node.get_logger().error(
                f"Timed out after "
                f"{args.timeout} seconds"
            )

            break

    node.destroy_node()

    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
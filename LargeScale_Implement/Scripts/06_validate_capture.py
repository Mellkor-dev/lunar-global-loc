#!/usr/bin/env python3

from pathlib import Path
import csv
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation




PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import load_pipeline_config

CONFIG = load_pipeline_config()

CAPTURE_DIRECTORY = CONFIG.captures_path
CLOUD_DIRECTORY = CAPTURE_DIRECTORY / "pointcloud_scans"
ODOM_DIRECTORY = CAPTURE_DIRECTORY / "odom_scans"
TRANSFORM_DIRECTORY = CAPTURE_DIRECTORY / "transform_scan"

OUTPUT_DIRECTORY = CONFIG.capture_validation_path
SUMMARY_PATH = OUTPUT_DIRECTORY / "capture_validation.csv"
EXTRINSIC_PATH = OUTPUT_DIRECTORY / "recovered_lidar_extrinsics.npz"


# Known Husky-to-LiDAR translation from the USD/TF setup.
EXPECTED_BASE_TO_LIDAR_TRANSLATION_M = np.array(
    [-0.15, 0.0, 0.415],
    dtype=float,
)

# These are validation tolerances, not sensor uncertainty values.
TRANSLATION_CONSISTENCY_TOLERANCE_M = 0.03
ROTATION_CONSISTENCY_TOLERANCE_DEG = 1.0


def site_number(path: Path) -> int:
    """Extract the integer after 'site_' from a filename."""
    match = re.search(r"site_(\d+)", path.name)

    if match is None:
        raise ValueError(f"Cannot determine site number from {path.name}")

    return int(match.group(1))


def pointcloud_xyz(path: Path) -> np.ndarray:
    """Load either a structured x/y/z array or a normal N x 3 array."""
    cloud = np.load(path)

    
    if cloud.dtype.names is not None:
        required = {"x", "y", "z"}

        if not required.issubset(cloud.dtype.names):
            raise ValueError(
                f"{path.name} fields are {cloud.dtype.names}; "
                "expected x, y, z"
            )

        xyz = np.column_stack(
            (
                cloud["x"],
                cloud["y"],
                cloud["z"],
            )
        )

    else:
        cloud = np.asarray(cloud)

        if cloud.ndim != 2 or cloud.shape[1] < 3:
            raise ValueError(
                f"{path.name} must have shape (N, 3+) or named xyz fields; "
                f"received {cloud.shape}"
            )

        xyz = cloud[:, :3]

    return np.asarray(xyz, dtype=float)


def transform_from_pose(pose: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] into a 4 x 4 matrix."""
    pose = np.asarray(pose, dtype=float).reshape(-1)

    if pose.shape != (7,):
        raise ValueError(
            f"Odometry pose must contain 7 values, received {pose.shape}"
        )

    if not np.isfinite(pose).all():
        raise ValueError("Odometry pose contains invalid values")

    translation = pose[:3]
    quaternion_xyzw = pose[3:]

    quaternion_norm = np.linalg.norm(quaternion_xyzw)

    if quaternion_norm == 0.0:
        raise ValueError("Odometry quaternion has zero norm")

    quaternion_xyzw = quaternion_xyzw / quaternion_norm

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(
        quaternion_xyzw
    ).as_matrix()
    transform[:3, 3] = translation

    return transform


def rotation_difference_deg(
    reference_rotation: np.ndarray,
    other_rotation: np.ndarray,
) -> float:
    """Angular difference between two 3 x 3 rotation matrices."""
    relative_rotation = reference_rotation.T @ other_rotation

    return float(
        np.degrees(
            Rotation.from_matrix(relative_rotation).magnitude()
        )
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    cloud_files = {
        site_number(path): path
        for path in CLOUD_DIRECTORY.glob("scan_site_*.npy")
    }

    odom_files = {
        site_number(path): path
        for path in ODOM_DIRECTORY.glob("odom_site_*.npy")
    }

    transform_files = {
        site_number(path): path
        for path in TRANSFORM_DIRECTORY.glob("transform_site_*.npz")
    }

    cloud_sites = set(cloud_files)
    odom_sites = set(odom_files)
    transform_sites = set(transform_files)

    all_sites = sorted(cloud_sites | odom_sites | transform_sites)
    complete_sites = sorted(cloud_sites & odom_sites & transform_sites)

    print("Dataset files")
    print("-------------")
    print(f"Cloud sites:      {sorted(cloud_sites)}")
    print(f"Odometry sites:   {sorted(odom_sites)}")
    print(f"Transform sites:  {sorted(transform_sites)}")
    print(f"Complete sites:   {complete_sites}")
    print()

    incomplete_sites = sorted(
        set(all_sites) - set(complete_sites)
    )

    if incomplete_sites:
        print(f"WARNING: incomplete sites: {incomplete_sites}")
        print()

    if not complete_sites:
        raise RuntimeError("No complete scan/odom/transform site was found")

    results = []
    recovered_transforms = []
    map_base_transforms = []
    map_lidar_transforms = []

    previous_position = None
    cumulative_distance_m = 0.0

    for site in complete_sites:
        cloud_path = cloud_files[site]
        odom_path = odom_files[site]
        transform_path = transform_files[site]       

        xyz = pointcloud_xyz(cloud_path)
        finite_mask = np.isfinite(xyz).all(axis=1)

        finite_xyz = xyz[finite_mask]

        if len(finite_xyz) == 0:
            raise ValueError(
                f"Site {site:02d} contains no finite point-cloud data"
            )

        ranges = np.linalg.norm(finite_xyz, axis=1)       

        odom_pose = np.load(odom_path)
        T_map_base = transform_from_pose(odom_pose)  
        transform_data = np.load(transform_path)
        if "T" not in transform_data.files:
            raise ValueError(
                f"{transform_path.name} does not contain matrix T"
            )

        T_map_lidar = np.asarray(
            transform_data["T"],
            dtype=float,
        )

        if T_map_lidar.shape != (4, 4):
            raise ValueError(
                f"{transform_path.name}: T has shape "
                f"{T_map_lidar.shape}, expected (4, 4)"
            )

        if not np.isfinite(T_map_lidar).all():
            raise ValueError(
                f"{transform_path.name} contains invalid values"
            )
        T_base_lidar = np.linalg.inv(T_map_base) @ T_map_lidar

        recovered_translation = T_base_lidar[:3, 3]
        recovered_rotation = T_base_lidar[:3, :3]

        expected_translation_error = float(
            np.linalg.norm(
                recovered_translation
                - EXPECTED_BASE_TO_LIDAR_TRANSLATION_M
            )
        )

        map_position = T_map_base[:3, 3]

        if previous_position is None:
            step_distance_m = 0.0
        else:
            step_distance_m = float(
                np.linalg.norm(
                    map_position - previous_position
                )
            )
            cumulative_distance_m += step_distance_m

        previous_position = map_position.copy()

        results.append(
            {
                "site": site,
                "point_count_total": len(xyz),
                "point_count_finite": len(finite_xyz),
                "minimum_range_m": float(np.min(ranges)),
                "median_range_m": float(np.median(ranges)),
                "maximum_range_m": float(np.max(ranges)),
                "map_base_x_m": float(map_position[0]),
                "map_base_y_m": float(map_position[1]),
                "map_base_z_m": float(map_position[2]),
                "step_distance_m": step_distance_m,
                "cumulative_distance_m": cumulative_distance_m,
                "base_lidar_x_m": float(recovered_translation[0]),
                "base_lidar_y_m": float(recovered_translation[1]),
                "base_lidar_z_m": float(recovered_translation[2]),
                "expected_translation_error_m": (
                    expected_translation_error
                ),
            }
        )

        recovered_transforms.append(T_base_lidar)
        map_base_transforms.append(T_map_base)
        map_lidar_transforms.append(T_map_lidar)

    recovered_transforms = np.stack(recovered_transforms)
    map_base_transforms = np.stack(map_base_transforms)
    map_lidar_transforms = np.stack(map_lidar_transforms)

    
    reference_transform = recovered_transforms[0]
    reference_rotation = reference_transform[:3, :3]
    reference_translation = reference_transform[:3, 3]

    translation_differences = []
    rotation_differences_deg = []

    for index, transform in enumerate(recovered_transforms):
        translation_difference = float(
            np.linalg.norm(
                transform[:3, 3] - reference_translation
            )
        )

        rotation_difference = rotation_difference_deg(
            reference_rotation,
            transform[:3, :3],
        )

        translation_differences.append(translation_difference)
        rotation_differences_deg.append(rotation_difference)

        results[index]["extrinsic_translation_change_m"] = (
            translation_difference
        )

        results[index]["extrinsic_rotation_change_deg"] = (
            rotation_difference
        )

    translation_differences = np.asarray(
        translation_differences
    )

    rotation_differences_deg = np.asarray(
        rotation_differences_deg
    )

    translations = recovered_transforms[:, :3, 3]   

    print("Per-site summary")
    print("----------------")

    for row in results:
        print(
            f"Site {row['site']:02d}: "
            f"points={row['point_count_finite']:6d}, "
            f"range=[{row['minimum_range_m']:6.2f}, "
            f"{row['maximum_range_m']:6.2f}] m, "
            f"step={row['step_distance_m']:6.2f} m, "
            f"T_base_lidar=({row['base_lidar_x_m']:+.4f}, "
            f"{row['base_lidar_y_m']:+.4f}, "
            f"{row['base_lidar_z_m']:+.4f}) m, "
            f"Δt={row['extrinsic_translation_change_m']:.5f} m, "
            f"ΔR={row['extrinsic_rotation_change_deg']:.5f}°"
        )

    print()
    print("Recovered static LiDAR extrinsic")
    print("--------------------------------")
    print("Mean translation [m]:")
    print(np.mean(translations, axis=0))
    print("Translation standard deviation [m]:")
    print(np.std(translations, axis=0))
    print()
    print(
        "Maximum change from reference translation: "
        f"{np.max(translation_differences):.6f} m"
    )
    print(
        "Maximum change from reference rotation: "
        f"{np.max(rotation_differences_deg):.6f} deg"
    )
    print(
        "Maximum error from expected translation "
        f"{EXPECTED_BASE_TO_LIDAR_TRANSLATION_M.tolist()}: "
        f"{max(row['expected_translation_error_m'] for row in results):.6f} m"
    )
    print()
    print(f"Total trajectory distance: {cumulative_distance_m:.3f} m")

    translation_consistent = (
        np.max(translation_differences)
        <= TRANSLATION_CONSISTENCY_TOLERANCE_M
    )

    rotation_consistent = (
        np.max(rotation_differences_deg)
        <= ROTATION_CONSISTENCY_TOLERANCE_DEG
    )

    print()
    print("Validation result")
    print("-----------------")
    print(
        "Static translation consistency: "
        f"{'PASS' if translation_consistent else 'FAIL'}"
    )
    print(
        "Static rotation consistency:    "
        f"{'PASS' if rotation_consistent else 'FAIL'}"
    )   

    fieldnames = list(results[0].keys())

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    np.savez_compressed(
        EXTRINSIC_PATH,
        site_numbers=np.asarray(complete_sites, dtype=int),
        T_map_base=map_base_transforms,
        T_map_lidar=map_lidar_transforms,
        T_base_lidar=recovered_transforms,
        mean_base_lidar_translation_m=np.mean(
            translations,
            axis=0,
        ),
        translation_standard_deviation_m=np.std(
            translations,
            axis=0,
        ),
        translation_change_from_reference_m=(
            translation_differences
        ),
        rotation_change_from_reference_deg=(
            rotation_differences_deg
        ),
        expected_translation_m=(
            EXPECTED_BASE_TO_LIDAR_TRANSLATION_M
        ),
    )

    print()
    print(f"CSV summary:      {SUMMARY_PATH}")
    print(f"Transform arrays: {EXTRINSIC_PATH}")


if __name__ == "__main__":
    main()

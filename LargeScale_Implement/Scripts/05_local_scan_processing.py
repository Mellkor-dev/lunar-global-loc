#!/usr/bin/env python3
"""Load and preprocess the captured local-scan dataset.

Point clouds are deliberately stored as a tuple because every scan can contain
a different number of points.  Odometry poses and transforms have fixed shapes
and are therefore stacked into normal NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence
import math
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = PROJECT_ROOT / "sim"
ODOM_SCAN = SIM_PATH / "odom_scans"
TRANSFORM_SCAN = SIM_PATH / "transform_scan"
POINTCLOUD_SCAN = SIM_PATH / "pointcloud_scans"


@dataclass(frozen=True)
class LocalScanDataset:
    """Preprocessed, site-aligned scan data ready for later computations."""

    site_numbers: np.ndarray
    pointclouds: tuple[np.ndarray, ...]
    odometry: np.ndarray
    transforms: np.ndarray

    def __len__(self) -> int:
        return len(self.site_numbers)

    def index_for_site(self, site_number: int) -> int:
        """Return the array/list index corresponding to a site number."""
        matches = np.flatnonzero(self.site_numbers == site_number)
        if len(matches) == 0:
            raise KeyError(f"Site {site_number:02d} is not in this dataset")
        return int(matches[0])

    def site(
        self,
        site_number: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(pointcloud, odometry, transform)`` for one site."""
        index = self.index_for_site(site_number)
        return (
            self.pointclouds[index],
            self.odometry[index],
            self.transforms[index],
        )

    def pointcloud_in_map(self, site_number: int) -> np.ndarray:
        """Transform one preprocessed LiDAR-frame point cloud into map frame."""
        index = self.index_for_site(site_number)
        xyz = self.pointclouds[index]
        rotation = self.transforms[index, :3, :3]
        translation = self.transforms[index, :3, 3]
        return xyz @ rotation.T + translation


def _site_number(path: Path) -> int:
    match = re.search(r"site_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot extract a site number from {path.name}")
    return int(match.group(1))


def _files_by_site(directory: Path, pattern: str) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {directory}")

    files: dict[int, Path] = {}
    for path in directory.glob(pattern):
        site_number = _site_number(path)
        if site_number in files:
            raise ValueError(
                f"Duplicate files for site {site_number:02d} in {directory}"
            )
        files[site_number] = path
    return files


def _preprocess_pointcloud(path: Path) -> np.ndarray:
    """Convert a structured or ordinary cloud to finite, nonzero ``N x 3`` XYZ."""
    cloud = np.load(path, allow_pickle=False)

    if cloud.dtype.names is not None:
        required_fields = {"x", "y", "z"}
        if not required_fields.issubset(cloud.dtype.names):
            raise ValueError(
                f"{path.name} has fields {cloud.dtype.names}; expected x, y, z"
            )
        xyz = np.column_stack((cloud["x"], cloud["y"], cloud["z"]))
    else:
        cloud = np.asarray(cloud)
        if cloud.ndim != 2 or cloud.shape[1] < 3:
            raise ValueError(
                f"{path.name} must be an N x 3+ array; got {cloud.shape}"
            )
        xyz = cloud[:, :3]

    xyz = np.asarray(xyz, dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=1)
    valid &= np.any(xyz != 0.0, axis=1)
    xyz = np.ascontiguousarray(xyz[valid])

    if len(xyz) == 0:
        raise ValueError(f"{path.name} has no valid points after preprocessing")
    return xyz


def _load_odometry(paths: Sequence[Path]) -> np.ndarray:
    poses = []
    for path in paths:
        pose = np.asarray(
            np.load(path, allow_pickle=False),
            dtype=np.float64,
        ).reshape(-1)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(
                f"{path.name} must contain finite [x,y,z,qx,qy,qz,qw]; "
                f"got shape {pose.shape}"
            )

        quaternion_norm = np.linalg.norm(pose[3:])
        if quaternion_norm == 0.0:
            raise ValueError(f"{path.name} contains a zero-length quaternion")
        pose[3:] /= quaternion_norm
        poses.append(pose)

    return np.stack(poses)


def _load_transforms(paths: Sequence[Path]) -> np.ndarray:
    transforms = []
    for path in paths:
        with np.load(path, allow_pickle=False) as transform_file:
            if "T" not in transform_file.files:
                raise ValueError(f"{path.name} does not contain a 'T' array")
            transform = np.asarray(transform_file["T"], dtype=np.float64)

        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(
                f"{path.name} must contain a finite 4 x 4 transform; "
                f"got {transform.shape}"
            )
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError(f"{path.name} is not a homogeneous transform")
        transforms.append(transform)

    return np.stack(transforms)


def load_local_scan_dataset(
    pointcloud_directory: Path = POINTCLOUD_SCAN,
    odometry_directory: Path = ODOM_SCAN,
    transform_directory: Path = TRANSFORM_SCAN,
) -> LocalScanDataset:
    """Load, validate, preprocess, and align all inputs by site number."""
    cloud_files = _files_by_site(Path(pointcloud_directory), "scan_site_*.npy")
    odom_files = _files_by_site(Path(odometry_directory), "odom_site_*.npy")
    transform_files = _files_by_site(
        Path(transform_directory),
        "transform_site_*.npz",
    )

    site_sets = (set(cloud_files), set(odom_files), set(transform_files))
    if not all(site_sets):
        raise FileNotFoundError("One or more dataset directories contain no data")
    if not (site_sets[0] == site_sets[1] == site_sets[2]):
        raise ValueError(
            "Point-cloud, odometry, and transform site numbers do not match: "
            f"cloud={sorted(site_sets[0])}, odom={sorted(site_sets[1])}, "
            f"transform={sorted(site_sets[2])}"
        )

    site_numbers = np.asarray(sorted(site_sets[0]), dtype=np.int64)
    pointclouds = tuple(
        _preprocess_pointcloud(cloud_files[site])
        for site in site_numbers
    )
    odometry = _load_odometry([odom_files[site] for site in site_numbers])
    transforms = _load_transforms(
        [transform_files[site] for site in site_numbers]
    )

    return LocalScanDataset(
        site_numbers=site_numbers,
        pointclouds=pointclouds,
        odometry=odometry,
        transforms=transforms,
    )
    
def pose_transform(pose : np.ndarray) -> np.ndarray:
    pose =np.asarray(pose,dtype=np.float64).reshape(-1)
    if pose.shape != 7:
        print("Shape Error")
        
    quaternion = pose[3:]
    quaternion_mag = np.linalg.norm(quaternion)
    quaternion = quaternion/quaternion_mag
    transform = np.eye(4,dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = pose[:3]

    return transform

def yaw_rotation(yaw_rad: float) -> np.ndarray:
    """Rotation from a yaw-aligned local frame into the map frame."""
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)

    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )      
    

def level_pointcloud(
    points_lidar: np.ndarray,
    odometry_pose: np.ndarray,
    transform_array:np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Convert LiDAR points to base frame and remove roll/pitch.

    Returns:
        points_base
        points_leveled
        lidar_origin_leveled
        yaw_rad
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)

    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError(
            f"points_lidar must have shape (N, 3), got {points_lidar.shape}"
        )

    # Odometry gives base_link pose in map/odom.
    T_map_base = pose_transform(odometry_pose)

    # Static LiDAR mount: lidar frame -> base_link frame.
    T_base_lidar = np.array(
        [
            [1.0, 0.0, 0.0, -0.15],
            [0.0, 1.0, 0.0,  0.00],
            [0.0, 0.0, 1.0,  0.415],
            [0.0, 0.0, 0.0,  1.00],
        ],
        dtype=np.float64,
    )

    R_base_lidar = T_base_lidar[:3, :3]
    t_base_lidar = T_base_lidar[:3, 3]

    # LiDAR frame -> base_link frame.
    points_base = (
        points_lidar @ R_base_lidar.T
        + t_base_lidar
    )

    R_map_base = T_map_base[:3, :3]

    # Extract heading only.
    yaw_rad = float(
        np.arctan2(
            R_map_base[1, 0],
            R_map_base[0, 0],
        )
    )

    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)

    R_map_level = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Remove roll and pitch while preserving rover-relative yaw.
    R_level_base = R_map_level.T @ R_map_base

    points_leveled = points_base @ R_level_base.T

    # Position of LiDAR origin relative to base_link,
    # expressed in the leveled frame.
    lidar_origin_leveled = R_level_base @ t_base_lidar
    
    T_map_base = pose_transform(odometry_pose)

    T_map_lidar_predicted = T_map_base @ T_base_lidar
    T_map_lidar_saved = transform_array

    error = np.linalg.inv(T_map_lidar_predicted) @ T_map_lidar_saved

    print("Predicted T_map_lidar:")
    print(T_map_lidar_predicted)

    print("Saved T_map_lidar:")
    print(T_map_lidar_saved)

    print("Difference:")
    print(error)

    return (
        points_base,
        points_leveled,
        T_base_lidar,
        lidar_origin_leveled,
        yaw_rad,
    )
    

def main() -> None:
    dataset = load_local_scan_dataset()

    site_number = 16
    index = dataset.index_for_site(site_number)

    points_lidar = dataset.pointclouds[index]
    odometry_pose = dataset.odometry[index]
    T_map_lidar = dataset.transforms[index]

    (
        points_base,
        points_leveled,
        T_base_lidar,
        lidar_origin_leveled,
        yaw_rad,
    ) = level_pointcloud(
        points_lidar=points_lidar,
        odometry_pose=odometry_pose,
        transform_array=T_map_lidar,
    )

    print(f"Site: {site_number:02d}")
    print(f"Raw cloud shape:     {points_lidar.shape}")
    print(f"Base cloud shape:    {points_base.shape}")
    print(f"Leveled cloud shape: {points_leveled.shape}")

    print("\nRecovered T_base_lidar:")
    print(T_base_lidar)

    print("\nLiDAR origin in leveled frame:")
    print(lidar_origin_leveled)

    print(f"\nRover yaw: {np.degrees(yaw_rad):.3f} deg")

    print("\nRaw LiDAR z range:")
    print(points_lidar[:, 2].min(), points_lidar[:, 2].max())

    print("\nBase-frame z range:")
    print(points_base[:, 2].min(), points_base[:, 2].max())

    print("\nLeveled z range:")
    print(points_leveled[:, 2].min(), points_leveled[:, 2].max())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Load, level, and grid every captured local LiDAR scan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import add_resolution_argument, load_resolution_config
from site_selection import (
    remove_unselected_site_artifacts,
    selected_sites_for_config,
)

_DEFAULT_CONFIG = load_resolution_config("5m")
CAPTURE_PATH = _DEFAULT_CONFIG.captures_path
ODOM_SCAN = CAPTURE_PATH / "odom_scans"
TRANSFORM_SCAN = CAPTURE_PATH / "transform_scan"
POINTCLOUD_SCAN = CAPTURE_PATH / "pointcloud_scans"
LEVELED_SCAN = _DEFAULT_CONFIG.leveled_maps_path
GRIDDED_SCAN = _DEFAULT_CONFIG.gridded_maps_path
LOCAL_GRID_RESOLUTION_M = _DEFAULT_CONFIG.orbital_raster.resolution_m


T_BASE_LIDAR = np.eye(4, dtype=np.float64)
T_BASE_LIDAR[:3, 3] = _DEFAULT_CONFIG.base_to_lidar_translation_m


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
        matches = np.flatnonzero(self.site_numbers == site_number)
        if len(matches) == 0:
            raise KeyError(f"Site {site_number:02d} is not in this dataset")
        return int(matches[0])

    def site(
        self,
        site_number: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = self.index_for_site(site_number)
        return (
            self.pointclouds[index],
            self.odometry[index],
            self.transforms[index],
        )

    def pointcloud_in_map(self, site_number: int) -> np.ndarray:
        """Evaluation helper; this uses the saved absolute ground-truth pose."""
        index = self.index_for_site(site_number)
        xyz = self.pointclouds[index]
        rotation = self.transforms[index, :3, :3]
        translation = self.transforms[index, :3, 3]
        return xyz @ rotation.T + translation


@dataclass(frozen=True)
class LeveledPointCloud:
    """One cloud expressed in a gravity-leveled rover-local frame."""

    points_xyz: np.ndarray
    lidar_origin_xyz: np.ndarray
    yaw_rad: float


@dataclass(frozen=True)
class LocalElevationGrid:
    """A north-up local elevation raster and its coordinate vectors."""

    elevation: np.ndarray
    valid_mask: np.ndarray
    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    resolution_m: float
    max_neighbor_distance_m: float


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
    """Convert a structured or ordinary cloud to finite, nonzero N x 3 XYZ."""
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
    pointcloud_directory: Path | None = None,
    odometry_directory: Path | None = None,
    transform_directory: Path | None = None,
    selected_sites: Sequence[int] | None = None,
) -> LocalScanDataset:
    """Load, validate, preprocess, and align all inputs by site number."""
    pointcloud_directory = pointcloud_directory or POINTCLOUD_SCAN
    odometry_directory = odometry_directory or ODOM_SCAN
    transform_directory = transform_directory or TRANSFORM_SCAN
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

    available_sites = sorted(site_sets[0])
    if selected_sites is not None:
        requested = sorted(set(int(site) for site in selected_sites))
        missing = sorted(set(requested).difference(available_sites))
        if missing:
            raise FileNotFoundError(
                f"Selected sites are missing synchronized captures: {missing}"
            )
        available_sites = requested
    site_numbers = np.asarray(available_sites, dtype=np.int64)
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


def pose_transform(pose: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] to a 4 x 4 transform."""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"Pose must have shape (7,), got {pose.shape}")
    if not np.isfinite(pose).all():
        raise ValueError("Pose contains NaN or infinite values")

    quaternion = pose[3:]
    quaternion_norm = np.linalg.norm(quaternion)
    if quaternion_norm == 0.0:
        raise ValueError("Pose quaternion has zero length")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(
        quaternion / quaternion_norm
    ).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def yaw_rotation(yaw_rad: float) -> np.ndarray:
    
    cosine = np.cos(yaw_rad)
    sine = np.sin(yaw_rad)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def level_pointcloud(
    points_lidar: np.ndarray,
    odometry_pose: np.ndarray,
    T_base_lidar: np.ndarray = T_BASE_LIDAR,
) -> LeveledPointCloud:
    
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError(
            f"points_lidar must have shape (N, 3), got {points_lidar.shape}"
        )


    T_base_lidar = np.asarray(T_base_lidar, dtype=np.float64)
    if T_base_lidar.shape != (4, 4):
        raise ValueError(
            f"T_base_lidar must have shape (4, 4), got {T_base_lidar.shape}"
        )
    if not np.isfinite(T_base_lidar).all():
        raise ValueError("T_base_lidar contains invalid values")
    if not np.allclose(T_base_lidar[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError("T_base_lidar is not a homogeneous transform")

    R_base_lidar = T_base_lidar[:3, :3]
    t_base_lidar = T_base_lidar[:3, 3]
    points_base = points_lidar @ R_base_lidar.T + t_base_lidar

    R_map_base = pose_transform(odometry_pose)[:3, :3]
    yaw_rad = float(np.arctan2(R_map_base[1, 0], R_map_base[0, 0]))
    R_level_base = yaw_rotation(yaw_rad).T @ R_map_base

    points_leveled = points_base @ R_level_base.T
    lidar_origin_leveled = R_level_base @ t_base_lidar
    return LeveledPointCloud(
        points_xyz=np.ascontiguousarray(points_leveled, dtype=np.float32),
        lidar_origin_xyz=np.asarray(lidar_origin_leveled, dtype=np.float64),
        yaw_rad=yaw_rad,
    )


def grid_pointcloud_nearest(
    points_xyz: np.ndarray,
    resolution_m: float = LOCAL_GRID_RESOLUTION_M,
    max_neighbor_distance_m: float | None = None,
) -> LocalElevationGrid:
    """Grid a leveled cloud with nearest-neighbour XY interpolation."""
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(
            f"points_xyz must have shape (N, 3), got {points_xyz.shape}"
        )
    points_xyz = points_xyz[np.isfinite(points_xyz).all(axis=1)]
    if len(points_xyz) == 0:
        raise ValueError("Point cloud has no finite points to grid")
    if not np.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("resolution_m must be finite and positive")

    if max_neighbor_distance_m is None:
        max_neighbor_distance_m = resolution_m / np.sqrt(2.0)
    if (
        not np.isfinite(max_neighbor_distance_m)
        or max_neighbor_distance_m <= 0.0
    ):
        raise ValueError(
            "max_neighbor_distance_m must be finite and positive"
        )

    x_min = np.floor(points_xyz[:, 0].min() / resolution_m) * resolution_m
    x_max = np.ceil(points_xyz[:, 0].max() / resolution_m) * resolution_m
    y_min = np.floor(points_xyz[:, 1].min() / resolution_m) * resolution_m
    y_max = np.ceil(points_xyz[:, 1].max() / resolution_m) * resolution_m
    if x_max <= x_min:
        x_max = x_min + resolution_m
    if y_max <= y_min:
        y_max = y_min + resolution_m

    x_centers = np.arange(
        x_min + resolution_m / 2.0,
        x_max,
        resolution_m,
        dtype=np.float64,
    )
    # North-up raster convention: row zero has the greatest local y.
    y_centers = np.arange(
        y_max - resolution_m / 2.0,
        y_min,
        -resolution_m,
        dtype=np.float64,
    )

    query_y, query_x = np.meshgrid(y_centers, x_centers, indexing="ij")
    query_xy = np.column_stack((query_x.ravel(), query_y.ravel()))
    distances, nearest_indices = cKDTree(points_xyz[:, :2]).query(
        query_xy,
        k=1,
        workers=-1,
    )
    valid = distances <= max_neighbor_distance_m

    elevation = np.full(len(query_xy), np.nan, dtype=np.float32)
    elevation[valid] = points_xyz[nearest_indices[valid], 2]
    shape = (len(y_centers), len(x_centers))
    return LocalElevationGrid(
        elevation=elevation.reshape(shape),
        valid_mask=valid.reshape(shape),
        x_centers_m=x_centers,
        y_centers_m=y_centers,
        resolution_m=float(resolution_m),
        max_neighbor_distance_m=float(max_neighbor_distance_m),
    )


def process_all_sites(
    dataset: LocalScanDataset,
    leveled_directory: Path = LEVELED_SCAN,
    gridded_directory: Path = GRIDDED_SCAN,
    resolution_m: float = LOCAL_GRID_RESOLUTION_M,
    max_neighbor_distance_m: float | None = None,
    T_base_lidar: np.ndarray = T_BASE_LIDAR,
) -> list[LocalElevationGrid]:
    """Level and grid every site, saving reusable NumPy artifacts."""
    leveled_directory = Path(leveled_directory)
    gridded_directory = Path(gridded_directory)
    leveled_directory.mkdir(parents=True, exist_ok=True)
    gridded_directory.mkdir(parents=True, exist_ok=True)

    grids = []
    for index, site_value in enumerate(dataset.site_numbers):
        site_number = int(site_value)
        leveled = level_pointcloud(
            dataset.pointclouds[index],
            dataset.odometry[index],
            T_base_lidar=T_base_lidar,
        )
        grid = grid_pointcloud_nearest(
            leveled.points_xyz,
            resolution_m=resolution_m,
            max_neighbor_distance_m=max_neighbor_distance_m,
        )

        np.save(
            leveled_directory / f"leveled_site_{site_number:02d}.npy",
            leveled.points_xyz,
        )
        np.savez_compressed(
            gridded_directory / f"grid_site_{site_number:02d}.npz",
            elevation=grid.elevation,
            valid_mask=grid.valid_mask,
            x_centers_m=grid.x_centers_m,
            y_centers_m=grid.y_centers_m,
            resolution_m=np.float64(grid.resolution_m),
            max_neighbor_distance_m=np.float64(
                grid.max_neighbor_distance_m
            ),
            lidar_origin_xyz_m=leveled.lidar_origin_xyz,
            measured_yaw_rad=np.float64(leveled.yaw_rad),
            site_number=np.int64(site_number),
            frame=np.asarray("gravity_leveled_rover_local"),
            raster_convention=np.asarray(
                "columns:+x, rows:-y, elevation:+z"
            ),
        )
        grids.append(grid)
        print(
            f"Site {site_number:02d}: points={len(leveled.points_xyz):6d}, "
            f"grid={grid.elevation.shape}, "
            f"valid={int(grid.valid_mask.sum()):5d}/{grid.valid_mask.size:5d}"
        )

    return grids


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_resolution_argument(parser)
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=None,
        help="Override the selected profile's grid resolution in metres",
    )
    parser.add_argument(
        "--max-neighbor-distance",
        type=float,
        default=None,
        help="Maximum XY interpolation distance (default: half-cell diagonal)",
    )
    parser.add_argument(
        "--leveled-directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--gridded-directory",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    global CAPTURE_PATH, ODOM_SCAN, TRANSFORM_SCAN, POINTCLOUD_SCAN
    args = parse_arguments()
    config = load_resolution_config(args.resolution)
    CAPTURE_PATH = config.captures_path
    ODOM_SCAN = CAPTURE_PATH / "odom_scans"
    TRANSFORM_SCAN = CAPTURE_PATH / "transform_scan"
    POINTCLOUD_SCAN = CAPTURE_PATH / "pointcloud_scans"
    resolution_m = args.grid_resolution or config.orbital_raster.resolution_m
    leveled_directory = args.leveled_directory or config.leveled_maps_path
    gridded_directory = args.gridded_directory or config.gridded_maps_path
    selected_sites = selected_sites_for_config(config)
    dataset = load_local_scan_dataset(selected_sites=selected_sites)
    T_base_lidar = np.eye(4, dtype=np.float64)
    T_base_lidar[:3, 3] = config.base_to_lidar_translation_m
    removed_leveled = remove_unselected_site_artifacts(
        leveled_directory, "leveled_site_*.npy", selected_sites
    )
    removed_gridded = remove_unselected_site_artifacts(
        gridded_directory, "grid_site_*.npz", selected_sites
    )
    grids = process_all_sites(
        dataset,
        leveled_directory=leveled_directory,
        gridded_directory=gridded_directory,
        resolution_m=resolution_m,
        max_neighbor_distance_m=args.max_neighbor_distance,
        T_base_lidar=T_base_lidar,
    )
    print()
    print(f"Saved {len(grids)} leveled clouds to {leveled_directory}")
    print(f"Saved {len(grids)} local grids to {gridded_directory}")
    print(f"Shared site manifest: {config.site_selection_manifest_path}")
    if removed_leveled or removed_gridded:
        print(
            "Removed stale unselected artifacts: "
            f"leveled={removed_leveled}, gridded={removed_gridded}"
        )


if __name__ == "__main__":
    main()

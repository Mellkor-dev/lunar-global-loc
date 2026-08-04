#!/usr/bin/env python3
"""Detect crater-like features in every gridded local LiDAR map."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.dilation_detector import detect_craters, detect_peaks
from pipeline_config import add_resolution_argument, load_resolution_config


CONFIG = load_resolution_config("5m")
GRID_DIRECTORY = CONFIG.gridded_maps_path
FEATURE_DIRECTORY = CONFIG.local_features_path
PREVIEW_DIRECTORY = CONFIG.plots_path / "local_features"
SUMMARY_PATH = FEATURE_DIRECTORY / "local_feature_summary.csv"

RESOLUTION_M = CONFIG.orbital_raster.resolution_m
DETECTION_DISTANCE_M = CONFIG.features.distance_m
DETECTION_RADIUS_CELLS = CONFIG.features.radius_for_resolution(RESOLUTION_M)
FLATNESS_EPS_M = CONFIG.features.flatness_threshold_m
MIN_VALID_FRACTION = CONFIG.features.local_min_valid_fraction
SIGMA_LOCAL_MAP_M = 0.5
UNCERTAINTY_MODEL = "symmetric_crater_occlusion"


def has_noncollinear_triplet(xy: np.ndarray, area_epsilon_m2: float = 1.0) -> bool:
    """Return whether at least one feature triple defines a usable triangle."""
    if len(xy) < 3:
        return False

    # Testing triangles anchored at each pair is inexpensive for these small
    # catalogues and avoids treating three collinear detections as matchable.
    for first in range(len(xy) - 2):
        vectors = xy[first + 1:] - xy[first]
        for left in range(len(vectors) - 1):
            twice_areas = np.abs(
                vectors[left, 0] * vectors[left + 1:, 1]
                - vectors[left, 1] * vectors[left + 1:, 0]
            )
            if np.any(twice_areas > 2.0 * area_epsilon_m2):
                return True
    return False


def save_preview(
    path: Path,
    elevation: np.ndarray,
    valid_mask: np.ndarray,
    x_centers_m: np.ndarray,
    y_centers_m: np.ndarray,
    feature_xy_m: np.ndarray,
    site_number: int,
) -> None:
    masked_elevation = np.ma.masked_where(~valid_mask, elevation)
    half_cell = RESOLUTION_M / 2.0
    extent = (
        x_centers_m[0] - half_cell,
        x_centers_m[-1] + half_cell,
        y_centers_m[-1] - half_cell,
        y_centers_m[0] + half_cell,
    )

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(
        masked_elevation,
        origin="upper",
        extent=extent,
        cmap="terrain",
    )
    if len(feature_xy_m):
        axis.scatter(
            feature_xy_m[:, 0],
            feature_xy_m[:, 1],
            marker="x",
            color="red",
            s=55,
            linewidths=1.8,
            label=f"Detected {CONFIG.features.kind}",
        )

    axis.scatter(0.0, 0.0, marker="^", color="black", s=45, label="Rover")
    axis.legend()
    axis.set_title(
        f"Site {site_number:02d}: local {CONFIG.features.kind} features "
        f"(N={len(feature_xy_m)})"
    )
    axis.set_xlabel("Rover-local x [m]")
    axis.set_ylabel("Rover-local y [m]")
    axis.set_aspect("equal")
    figure.colorbar(image, ax=axis, label="Leveled elevation [m]")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)

def crater_feature_covariances(
    feature_xyz_m: np.ndarray,
    lidar_origin_xyz_m: np.ndarray,
    *,
    d_detect_m: float,
    sigma_local_map_m: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Create crater-feature covariance matrices.

    This is a lunar crater adaptation of the Carle occlusion model.
    It uses the absolute elevation angle because crater minima are
    generally below the LiDAR and can be hidden by the near crater rim.
    """
    feature_xyz_m = np.asarray(
        feature_xyz_m,
        dtype=np.float64,
    )

    lidar_origin_xyz_m = np.asarray(
        lidar_origin_xyz_m,
        dtype=np.float64,
    ).reshape(-1)

    if feature_xyz_m.ndim != 2 or feature_xyz_m.shape[1] != 3:
        raise ValueError(
            "feature_xyz_m must have shape (N, 3)"
        )

    if lidar_origin_xyz_m.shape != (3,):
        raise ValueError(
            "lidar_origin_xyz_m must have shape (3,)"
        )

    if d_detect_m <= 0.0:
        raise ValueError("d_detect_m must be positive")

    if sigma_local_map_m <= 0.0:
        raise ValueError(
            "sigma_local_map_m must be positive"
        )

    delta_xyz_m = (
        feature_xyz_m - lidar_origin_xyz_m
    )

    horizontal_range_m = np.hypot(
        delta_xyz_m[:, 0],
        delta_xyz_m[:, 1],
    )

    elevation_angle_rad = np.arctan2(
        delta_xyz_m[:, 2],
        horizontal_range_m,
    )

    sigma_xy_m = d_detect_m / 2.0

    # tan(theta) = dz / horizontal range.
    safe_horizontal_range_m = np.maximum(
        horizontal_range_m,
        1e-6,
    )

    absolute_slope = np.abs(
        delta_xyz_m[:, 2]
        / safe_horizontal_range_m
    )

    sigma_z_squared_m2 = (
        sigma_local_map_m**2
        + (sigma_xy_m * absolute_slope) ** 2
    )

    covariance = np.zeros(
        (len(feature_xyz_m), 3, 3),
        dtype=np.float64,
    )

    covariance[:, 0, 0] = sigma_xy_m**2
    covariance[:, 1, 1] = sigma_xy_m**2
    covariance[:, 2, 2] = sigma_z_squared_m2

    return covariance, elevation_angle_rad


def process_grid(path: Path) -> dict[str, int | float | bool]:
    with np.load(path, allow_pickle=False) as grid:
        elevation = np.asarray(
            grid["elevation"],
            dtype=np.float32,
        )

        valid_mask = np.asarray(
            grid["valid_mask"],
            dtype=bool,
        )

        x_centers_m = np.asarray(
            grid["x_centers_m"],
            dtype=np.float64,
        )

        y_centers_m = np.asarray(
            grid["y_centers_m"],
            dtype=np.float64,
        )

        resolution_m = float(grid["resolution_m"])
        site_number = int(grid["site_number"])

        if "lidar_origin_xyz_m" not in grid.files:
            raise ValueError(
                f"{path.name}: lidar_origin_xyz_m is missing. "
                "Regenerate the local grids using "
                "Scripts/05_local_scan_processing.py."
            )

        lidar_origin_leveled = np.asarray(
            grid["lidar_origin_xyz_m"],
            dtype=np.float64,
        ).reshape(-1)
        
    if not np.array_equal(valid_mask, np.isfinite(elevation)):
        raise ValueError(
            f"{path.name}: valid mask and finite cells disagree"
        )
    if elevation.shape != valid_mask.shape:
        raise ValueError(f"{path.name}: elevation/mask shapes do not match")
    if elevation.shape != (len(y_centers_m), len(x_centers_m)):
        raise ValueError(f"{path.name}: coordinate vectors do not match raster")
    if not np.isclose(resolution_m, RESOLUTION_M):
        raise ValueError(
            f"{path.name}: expected {RESOLUTION_M} m cells, got {resolution_m}"
        )
    if not np.array_equal(valid_mask, np.isfinite(elevation)):
        raise ValueError(f"{path.name}: valid mask and finite cells disagree")

    detector = (
        detect_craters
        if CONFIG.features.kind == "crater"
        else detect_peaks
    )
    crater_rc = detector(
        elevation,
        n=DETECTION_RADIUS_CELLS,
        flatness_eps=FLATNESS_EPS_M,
        min_valid_fraction=MIN_VALID_FRACTION,
    )
    rows = crater_rc[:, 0] if len(crater_rc) else np.empty(0, dtype=int)
    columns = crater_rc[:, 1] if len(crater_rc) else np.empty(0, dtype=int)
    x_m = x_centers_m[columns]
    y_m = y_centers_m[rows]
    z_m = elevation[rows, columns]
    feature_xy_m = np.column_stack((x_m, y_m))
    usable_for_darces = has_noncollinear_triplet(feature_xy_m)

    FEATURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    local_features_xyz = np.column_stack(
        (
            x_m,
            y_m,
            z_m,
        )
    ).astype(np.float64)

    local_covariances, elevation_angles_rad = (
        crater_feature_covariances(
            feature_xyz_m=local_features_xyz,
            lidar_origin_xyz_m=lidar_origin_leveled,
            d_detect_m=DETECTION_DISTANCE_M,
            sigma_local_map_m=SIGMA_LOCAL_MAP_M,
        )
    )

    local_sigma_z_m = np.sqrt(
        local_covariances[:, 2, 2]
    )
    np.savez_compressed(
        FEATURE_DIRECTORY / f"local_craters_site_{site_number:02d}.npz",
        row=rows.astype(np.int64),
        column=columns.astype(np.int64),
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        resolution_m=np.float64(resolution_m),
        detection_radius_cells=np.int64(DETECTION_RADIUS_CELLS),
        detection_distance_m=np.float64(DETECTION_DISTANCE_M),
        flatness_eps_m=np.float64(FLATNESS_EPS_M),
        min_valid_fraction=np.float64(MIN_VALID_FRACTION),
        site_number=np.int64(site_number),
        feature_kind=np.asarray(CONFIG.features.kind),
        config_path=np.asarray(str(CONFIG.config_path)),
        usable_for_darces=np.bool_(usable_for_darces),
        local_covariances=local_covariances,
        covariance=local_covariances,
        elevation_angle_rad=elevation_angles_rad,
        lidar_origin_xyz_m=lidar_origin_leveled,
        sigma_local_xy_m=np.float64(DETECTION_DISTANCE_M / 2.0),
        sigma_local_z_m=local_sigma_z_m,
        sigma_local_map_m=np.float64(SIGMA_LOCAL_MAP_M),
        uncertainty_model=np.asarray(UNCERTAINTY_MODEL),
    )
    save_preview(
        PREVIEW_DIRECTORY / f"local_craters_site_{site_number:02d}.png",
        elevation,
        valid_mask,
        x_centers_m,
        y_centers_m,
        feature_xy_m,
        site_number,
    )

    return {
        "site": site_number,
        "valid_grid_cells": int(valid_mask.sum()),
        "feature_count": len(crater_rc),
        "has_three_features": len(crater_rc) >= 3,
        "has_noncollinear_triplet": usable_for_darces,
    }


def main() -> None:
    global CONFIG, GRID_DIRECTORY, FEATURE_DIRECTORY, PREVIEW_DIRECTORY
    global SUMMARY_PATH, RESOLUTION_M, DETECTION_DISTANCE_M
    global DETECTION_RADIUS_CELLS, FLATNESS_EPS_M, MIN_VALID_FRACTION
    parser = argparse.ArgumentParser(description=__doc__)
    add_resolution_argument(parser)
    args = parser.parse_args()
    CONFIG = load_resolution_config(args.resolution)
    GRID_DIRECTORY = CONFIG.gridded_maps_path
    FEATURE_DIRECTORY = CONFIG.local_features_path
    PREVIEW_DIRECTORY = CONFIG.plots_path / "local_features"
    SUMMARY_PATH = FEATURE_DIRECTORY / "local_feature_summary.csv"
    RESOLUTION_M = CONFIG.orbital_raster.resolution_m
    DETECTION_DISTANCE_M = CONFIG.features.distance_m
    DETECTION_RADIUS_CELLS = CONFIG.features.radius_for_resolution(RESOLUTION_M)
    FLATNESS_EPS_M = CONFIG.features.flatness_threshold_m
    MIN_VALID_FRACTION = CONFIG.features.local_min_valid_fraction

    grid_paths = sorted(GRID_DIRECTORY.glob("grid_site_*.npz"))
    if not grid_paths:
        raise FileNotFoundError(f"No local grids found in {GRID_DIRECTORY}")

    rows = [process_grid(path) for path in grid_paths]
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Local {CONFIG.features.kind} feature detection")
    print("--------------------------------")
    for row in rows:
        status = "DARCES-ready" if row["has_noncollinear_triplet"] else "insufficient"
        print(
            f"Site {row['site']:02d}: features={row['feature_count']:2d}, "
            f"valid_cells={row['valid_grid_cells']:4d}, {status}"
        )

    ready_count = sum(bool(row["has_noncollinear_triplet"]) for row in rows)
    print()
    print(f"DARCES-ready sites: {ready_count}/{len(rows)}")
    print(f"Feature catalogues: {FEATURE_DIRECTORY}")
    print(f"Preview plots:      {PREVIEW_DIRECTORY}")
    print(f"Summary:            {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

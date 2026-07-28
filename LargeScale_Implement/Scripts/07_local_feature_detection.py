#!/usr/bin/env python3
"""Detect crater-like features in every gridded local LiDAR map."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.dilation_detector import detect_craters


GRID_DIRECTORY = PROJECT_ROOT / "local_maps" / "gridded_5m"
FEATURE_DIRECTORY = PROJECT_ROOT / "local_maps" / "features"
PREVIEW_DIRECTORY = PROJECT_ROOT / "plots" / "local_features"
SUMMARY_PATH = FEATURE_DIRECTORY / "local_feature_summary.csv"

RESOLUTION_M = 5.0
DETECTION_DISTANCE_M = 30.0
DETECTION_RADIUS_CELLS = int(round(DETECTION_DISTANCE_M / RESOLUTION_M))
FLATNESS_EPS_M = 0.15
MIN_VALID_FRACTION = 0.10


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
            label="Detected crater",
        )

    axis.scatter(0.0, 0.0, marker="^", color="black", s=45, label="Rover")
    axis.legend()
    axis.set_title(
        f"Site {site_number:02d}: local crater features "
        f"(N={len(feature_xy_m)})"
    )
    axis.set_xlabel("Rover-local x [m]")
    axis.set_ylabel("Rover-local y [m]")
    axis.set_aspect("equal")
    figure.colorbar(image, ax=axis, label="Leveled elevation [m]")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def process_grid(path: Path) -> dict[str, int | float | bool]:
    with np.load(path, allow_pickle=False) as grid:
        elevation = np.asarray(grid["elevation"], dtype=np.float32)
        valid_mask = np.asarray(grid["valid_mask"], dtype=bool)
        x_centers_m = np.asarray(grid["x_centers_m"], dtype=np.float64)
        y_centers_m = np.asarray(grid["y_centers_m"], dtype=np.float64)
        resolution_m = float(grid["resolution_m"])
        site_number = int(grid["site_number"])

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

    crater_rc = detect_craters(
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
        feature_kind=np.asarray("crater"),
        usable_for_darces=np.bool_(usable_for_darces),
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
    grid_paths = sorted(GRID_DIRECTORY.glob("grid_site_*.npz"))
    if not grid_paths:
        raise FileNotFoundError(f"No local grids found in {GRID_DIRECTORY}")

    rows = [process_grid(path) for path in grid_paths]
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Local crater feature detection")
    print("------------------------------")
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

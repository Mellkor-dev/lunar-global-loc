#!/usr/bin/env python3
"""Validate alignment between the 1.5 m truth DEM and 5 m orbital prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from scipy.interpolate import RegularGridInterpolator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_config import add_resolution_argument, load_resolution_config


def raster_coordinates(
    rows: int,
    columns: int,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Return x/y cell centers and edge bounds for a centered north-up DEM.

    Columns increase eastward.
    Rows increase southward.
    Map x increases eastward.
    Map y increases northward.
    """
    width_m = columns * resolution_m
    height_m = rows * resolution_m

    x_min_edge = -width_m / 2.0
    x_max_edge = width_m / 2.0
    y_min_edge = -height_m / 2.0
    y_max_edge = height_m / 2.0

    x_centers = (
        x_min_edge
        + (np.arange(columns, dtype=np.float64) + 0.5) * resolution_m
    )

    y_centers = (
        y_max_edge
        - (np.arange(rows, dtype=np.float64) + 0.5) * resolution_m
    )

    bounds = (
        x_min_edge,
        x_max_edge,
        y_min_edge,
        y_max_edge,
    )

    return x_centers, y_centers, bounds


def build_interpolator(
    dem: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> RegularGridInterpolator:
    """Create a DEM interpolator using ascending coordinate axes."""
    # Raster rows run from north to south, so y_centers is descending.
    # RegularGridInterpolator requires monotonically ascending axes.
    return RegularGridInterpolator(
        points=(y_centers[::-1], x_centers),
        values=dem[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def interpolate_dem(
    interpolator: RegularGridInterpolator,
    x_query: np.ndarray,
    y_query: np.ndarray,
) -> np.ndarray:
    """Interpolate a DEM on a Cartesian product of query coordinates."""
    query_y, query_x = np.meshgrid(
        y_query,
        x_query,
        indexing="ij",
    )

    query_points = np.column_stack(
        (query_y.ravel(), query_x.ravel())
    )

    result = interpolator(query_points)

    return result.reshape(query_y.shape)


def residual_statistics(residual: np.ndarray) -> dict[str, float | int]:
    finite = residual[np.isfinite(residual)]

    if finite.size == 0:
        raise ValueError("No finite residual samples were generated")

    absolute = np.abs(finite)

    return {
        "sample_count": int(finite.size),
        "mean_m": float(np.mean(finite)),
        "median_m": float(np.median(finite)),
        "standard_deviation_m": float(np.std(finite)),
        "mae_m": float(np.mean(absolute)),
        "rmse_m": float(np.sqrt(np.mean(finite**2))),
        "absolute_p95_m": float(np.percentile(absolute, 95.0)),
        "absolute_max_m": float(np.max(absolute)),
    }


def save_alignment_plot(
    truth: np.ndarray,
    orbital: np.ndarray,
    orbital_on_truth: np.ndarray,
    residual: np.ndarray,
    truth_bounds: tuple[float, float, float, float],
    orbital_bounds: tuple[float, float, float, float],
) -> None:
    minimum = min(float(np.min(truth)), float(np.min(orbital)))
    maximum = max(float(np.max(truth)), float(np.max(orbital)))

    finite_residual = residual[np.isfinite(residual)]
    residual_limit = float(
        np.percentile(np.abs(finite_residual), 99.0)
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 12))

    truth_image = axes[0, 0].imshow(
        truth,
        origin="upper",
        extent=truth_bounds,
        cmap="terrain",
        vmin=minimum,
        vmax=maximum,
    )
    axes[0, 0].set_title("Truth DEM — 1.5 m/cell")
    axes[0, 0].set_xlabel("Map x / east [m]")
    axes[0, 0].set_ylabel("Map y / north [m]")
    axes[0, 0].axhline(0.0, linewidth=0.5)
    axes[0, 0].axvline(0.0, linewidth=0.5)

    orbital_image = axes[0, 1].imshow(
        orbital,
        origin="upper",
        extent=orbital_bounds,
        cmap="terrain",
        vmin=minimum,
        vmax=maximum,
    )
    axes[0, 1].set_title("Orbital prior — 5 m/cell")
    axes[0, 1].set_xlabel("Map x / east [m]")
    axes[0, 1].set_ylabel("Map y / north [m]")
    axes[0, 1].axhline(0.0, linewidth=0.5)
    axes[0, 1].axvline(0.0, linewidth=0.5)

    interpolated_image = axes[1, 0].imshow(
        orbital_on_truth,
        origin="upper",
        extent=truth_bounds,
        cmap="terrain",
        vmin=minimum,
        vmax=maximum,
    )
    axes[1, 0].set_title("5 m prior interpolated onto 1.5 m centers")
    axes[1, 0].set_xlabel("Map x / east [m]")
    axes[1, 0].set_ylabel("Map y / north [m]")

    residual_image = axes[1, 1].imshow(
        residual,
        origin="upper",
        extent=truth_bounds,
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[1, 1].set_title("Truth − interpolated orbital prior")
    axes[1, 1].set_xlabel("Map x / east [m]")
    axes[1, 1].set_ylabel("Map y / north [m]")

    figure.colorbar(
        truth_image,
        ax=[axes[0, 0], axes[0, 1], axes[1, 0]],
        label="Elevation [m]",
        shrink=0.75,
    )

    figure.colorbar(
        residual_image,
        ax=axes[1, 1],
        label="Elevation residual [m]",
    )

    figure.suptitle(
        "Apollo 17 truth/prior DEM alignment",
        fontsize=14,
    )

    figure.savefig(
        ALIGNMENT_PLOT_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_profile_plot(
    truth_interpolator: RegularGridInterpolator,
    orbital_interpolator: RegularGridInterpolator,
) -> None:
    coordinates = np.linspace(-990.0, 990.0, 1000)

    east_west_points = np.column_stack(
        (
            np.zeros_like(coordinates),
            coordinates,
        )
    )

    south_north_points = np.column_stack(
        (
            coordinates,
            np.zeros_like(coordinates),
        )
    )

    truth_east_west = truth_interpolator(east_west_points)
    orbital_east_west = orbital_interpolator(east_west_points)

    truth_south_north = truth_interpolator(south_north_points)
    orbital_south_north = orbital_interpolator(south_north_points)

    figure, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(
        coordinates,
        truth_east_west,
        label="Truth 1.5 m",
    )
    axes[0].plot(
        coordinates,
        orbital_east_west,
        label="Orbital prior 5 m",
    )
    axes[0].set_title("East–west profile through map origin")
    axes[0].set_xlabel("Map x / east [m]")
    axes[0].set_ylabel("Elevation [m]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        coordinates,
        truth_south_north,
        label="Truth 1.5 m",
    )
    axes[1].plot(
        coordinates,
        orbital_south_north,
        label="Orbital prior 5 m",
    )
    axes[1].set_title("South–north profile through map origin")
    axes[1].set_xlabel("Map y / north [m]")
    axes[1].set_ylabel("Elevation [m]")
    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(PROFILE_PLOT_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    global CONFIG, TRUTH_PATH, ORBITAL_PATH, MASK_PATH
    global OUTPUT_DIRECTORY, ALIGNMENT_PLOT_PATH, PROFILE_PLOT_PATH, REPORT_PATH
    global TRUTH_RESOLUTION_M, ORBITAL_RESOLUTION_M

    parser = argparse.ArgumentParser(description=__doc__)
    add_resolution_argument(parser)
    args = parser.parse_args()
    CONFIG = load_resolution_config(args.resolution)
    TRUTH_PATH = CONFIG.truth_dem_path
    ORBITAL_PATH = CONFIG.orbital_dem_path
    MASK_PATH = CONFIG.orbital_mask_path
    OUTPUT_DIRECTORY = CONFIG.dem_qa_path
    ALIGNMENT_PLOT_PATH = OUTPUT_DIRECTORY / "dem_alignment.png"
    PROFILE_PLOT_PATH = OUTPUT_DIRECTORY / "dem_center_profiles.png"
    REPORT_PATH = OUTPUT_DIRECTORY / "dem_alignment_report.json"
    TRUTH_RESOLUTION_M = CONFIG.truth_raster.resolution_m
    ORBITAL_RESOLUTION_M = CONFIG.orbital_raster.resolution_m

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    truth = np.load(TRUTH_PATH)
    orbital = np.load(ORBITAL_PATH)
    orbital_mask = (
        np.load(MASK_PATH)
        if MASK_PATH.is_file()
        else np.isfinite(orbital)
    )

    if truth.shape != CONFIG.truth_raster.shape:
        raise ValueError(f"Unexpected truth shape: {truth.shape}")

    if orbital.shape != CONFIG.orbital_raster.shape:
        raise ValueError(f"Unexpected orbital shape: {orbital.shape}")

    if orbital_mask.shape != orbital.shape:
        raise ValueError("Orbital mask shape does not match DEM")

    if not orbital_mask.all():
        raise ValueError("Orbital prior contains invalid cells")

    truth_x, truth_y, truth_bounds = raster_coordinates(
        rows=truth.shape[0],
        columns=truth.shape[1],
        resolution_m=TRUTH_RESOLUTION_M,
    )

    orbital_x, orbital_y, orbital_bounds = raster_coordinates(
        rows=orbital.shape[0],
        columns=orbital.shape[1],
        resolution_m=ORBITAL_RESOLUTION_M,
    )

    truth_interpolator = build_interpolator(
        dem=truth,
        x_centers=truth_x,
        y_centers=truth_y,
    )

    orbital_interpolator = build_interpolator(
        dem=orbital,
        x_centers=orbital_x,
        y_centers=orbital_y,
    )

    orbital_on_truth = interpolate_dem(
        interpolator=orbital_interpolator,
        x_query=truth_x,
        y_query=truth_y,
    )

    residual = truth.astype(np.float64) - orbital_on_truth
    statistics = residual_statistics(residual)

    save_alignment_plot(
        truth=truth,
        orbital=orbital,
        orbital_on_truth=orbital_on_truth,
        residual=residual,
        truth_bounds=truth_bounds,
        orbital_bounds=orbital_bounds,
    )

    save_profile_plot(
        truth_interpolator=truth_interpolator,
        orbital_interpolator=orbital_interpolator,
    )

    report = {
        "coordinate_contract": {
            "origin": "center_of_dem",
            "x_direction": "east",
            "y_direction": "north",
            "raster_column_direction": "east",
            "raster_row_direction": "south",
        },
        "truth_dem": {
            "shape": list(truth.shape),
            "resolution_m": TRUTH_RESOLUTION_M,
            "edge_bounds_m": {
                "x_min": truth_bounds[0],
                "x_max": truth_bounds[1],
                "y_min": truth_bounds[2],
                "y_max": truth_bounds[3],
            },
            "first_center_m": {
                "x": float(truth_x[0]),
                "y": float(truth_y[0]),
            },
            "last_center_m": {
                "x": float(truth_x[-1]),
                "y": float(truth_y[-1]),
            },
        },
        "orbital_dem": {
            "shape": list(orbital.shape),
            "resolution_m": ORBITAL_RESOLUTION_M,
            "edge_bounds_m": {
                "x_min": orbital_bounds[0],
                "x_max": orbital_bounds[1],
                "y_min": orbital_bounds[2],
                "y_max": orbital_bounds[3],
            },
            "first_center_m": {
                "x": float(orbital_x[0]),
                "y": float(orbital_y[0]),
            },
            "last_center_m": {
                "x": float(orbital_x[-1]),
                "y": float(orbital_y[-1]),
            },
        },
        "truth_minus_interpolated_prior": statistics,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("DEM alignment validation")
    print("------------------------")
    print(f"Truth bounds:   {truth_bounds}")
    print(f"Orbital bounds: {orbital_bounds}")
    print()
    print("Truth first cell:")
    print(f"  x = {truth_x[0]:.6f} m")
    print(f"  y = {truth_y[0]:.6f} m")
    print("Truth last cell:")
    print(f"  x = {truth_x[-1]:.6f} m")
    print(f"  y = {truth_y[-1]:.6f} m")
    print()
    print("Orbital first cell:")
    print(f"  x = {orbital_x[0]:.6f} m")
    print(f"  y = {orbital_y[0]:.6f} m")
    print("Orbital last cell:")
    print(f"  x = {orbital_x[-1]:.6f} m")
    print(f"  y = {orbital_y[-1]:.6f} m")
    print()
    print("Truth − interpolated prior residuals")
    for key, value in statistics.items():
        print(f"{key:24s}: {value}")

    print()
    print(f"Alignment plot: {ALIGNMENT_PLOT_PATH}")
    print(f"Profile plot:   {PROFILE_PLOT_PATH}")
    print(f"JSON report:    {REPORT_PATH}")


if __name__ == "__main__":
    main()

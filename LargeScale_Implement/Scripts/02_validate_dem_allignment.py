#!/usr/bin/env python3
"""Validate alignment between the truth DEM and a selected resolution DEM."""

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
    first_x_center_m: float,
    first_y_center_m: float,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Return configured x/y cell centers and edge bounds for a north-up DEM.

    Columns increase eastward.
    Rows increase southward.
    Map x increases eastward.
    Map y increases northward.
    """
    x_centers = first_x_center_m + np.arange(columns) * resolution_m
    y_centers = first_y_center_m - np.arange(rows) * resolution_m

    bounds = (
        float(x_centers[0] - resolution_m / 2.0),
        float(x_centers[-1] + resolution_m / 2.0),
        float(y_centers[-1] - resolution_m / 2.0),
        float(y_centers[0] + resolution_m / 2.0),
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
    selected_profile: str,
    truth_source_label: str,
    truth_resolution_m: float,
    selected_resolution_m: float,
    statistics: dict[str, float | int],
) -> None:
    minimum = min(float(np.min(truth)), float(np.min(orbital)))
    maximum = max(float(np.max(truth)), float(np.max(orbital)))

    finite_residual = residual[np.isfinite(residual)]
    residual_limit = float(
        np.percentile(np.abs(finite_residual), 99.0)
    )
    if residual_limit == 0.0:
        residual_limit = float(np.finfo(np.float64).eps)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 12),
        layout="constrained",
    )
    truth_label = truth_source_label
    selected_label = (
        f"Selected DEM ({selected_profile}) — "
        f"{selected_resolution_m:g} m/cell"
    )

    truth_image = axes[0, 0].imshow(
        truth,
        origin="upper",
        extent=truth_bounds,
        cmap="terrain",
        vmin=minimum,
        vmax=maximum,
    )
    axes[0, 0].set_title(truth_label)
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
    axes[0, 1].set_title(selected_label)
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
    axes[1, 0].set_title(
        f"{selected_profile} DEM interpolated onto "
        f"{truth_resolution_m:g} m truth centers"
    )
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
    axes[1, 1].set_title("Truth − interpolated selected DEM")
    axes[1, 1].set_xlabel("Map x / east [m]")
    axes[1, 1].set_ylabel("Map y / north [m]")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(
            (
                f"mean = {float(statistics['mean_m']):.4f} m",
                f"RMSE = {float(statistics['rmse_m']):.4f} m",
                f"MAE = {float(statistics['mae_m']):.4f} m",
                f"|error| p95 = "
                f"{float(statistics['absolute_p95_m']):.4f} m",
            )
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

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
        f"DEM alignment: {truth_source_label} vs {selected_profile}",
        fontsize=14,
    )

    figure.savefig(
        ALIGNMENT_PLOT_PATH,
        dpi=180,
    )
    plt.close(figure)


def save_profile_plot(
    truth_interpolator: RegularGridInterpolator,
    orbital_interpolator: RegularGridInterpolator,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    selected_profile: str,
    truth_source_label: str,
    truth_resolution_m: float,
    selected_resolution_m: float,
) -> None:
    east_west_points = np.column_stack(
        (
            np.zeros_like(x_coordinates),
            x_coordinates,
        )
    )

    south_north_points = np.column_stack(
        (
            y_coordinates,
            np.zeros_like(y_coordinates),
        )
    )

    truth_east_west = truth_interpolator(east_west_points)
    orbital_east_west = orbital_interpolator(east_west_points)

    truth_south_north = truth_interpolator(south_north_points)
    orbital_south_north = orbital_interpolator(south_north_points)

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    truth_label = truth_source_label
    selected_label = (
        f"Selected {selected_profile} DEM ({selected_resolution_m:g} m)"
    )

    axes[0, 0].plot(
        x_coordinates,
        truth_east_west,
        label=truth_label,
        linewidth=2.2,
    )
    axes[0, 0].plot(
        x_coordinates,
        orbital_east_west,
        label=selected_label,
        linestyle="--",
        linewidth=1.4,
    )
    axes[0, 0].set_title("East–west profile through map origin")
    axes[0, 0].set_xlabel("Map x / east [m]")
    axes[0, 0].set_ylabel("Elevation [m]")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[1, 0].plot(
        y_coordinates,
        truth_south_north,
        label=truth_label,
        linewidth=2.2,
    )
    axes[1, 0].plot(
        y_coordinates,
        orbital_south_north,
        label=selected_label,
        linestyle="--",
        linewidth=1.4,
    )
    axes[1, 0].set_title("South–north profile through map origin")
    axes[1, 0].set_xlabel("Map y / north [m]")
    axes[1, 0].set_ylabel("Elevation [m]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[0, 1].plot(
        x_coordinates,
        truth_east_west - orbital_east_west,
        color="tab:purple",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[0, 1].set_title("East–west residual: truth − selected")
    axes[0, 1].set_xlabel("Map x / east [m]")
    axes[0, 1].set_ylabel("Elevation residual [m]")
    axes[0, 1].grid(True)

    axes[1, 1].plot(
        y_coordinates,
        truth_south_north - orbital_south_north,
        color="tab:purple",
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.7)
    axes[1, 1].set_title("South–north residual: truth − selected")
    axes[1, 1].set_xlabel("Map y / north [m]")
    axes[1, 1].set_ylabel("Elevation residual [m]")
    axes[1, 1].grid(True)

    figure.suptitle(
        f"Centerline comparison: {truth_source_label} vs {selected_profile}"
    )
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
        first_x_center_m=CONFIG.truth_raster.first_x_center_m,
        first_y_center_m=CONFIG.truth_raster.first_y_center_m,
    )

    orbital_x, orbital_y, orbital_bounds = raster_coordinates(
        rows=orbital.shape[0],
        columns=orbital.shape[1],
        resolution_m=ORBITAL_RESOLUTION_M,
        first_x_center_m=CONFIG.orbital_raster.first_x_center_m,
        first_y_center_m=CONFIG.orbital_raster.first_y_center_m,
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
        selected_profile=args.resolution,
        truth_source_label=CONFIG.truth_source_label,
        truth_resolution_m=TRUTH_RESOLUTION_M,
        selected_resolution_m=ORBITAL_RESOLUTION_M,
        statistics=statistics,
    )

    common_x_min = max(float(truth_x[0]), float(orbital_x[0]))
    common_x_max = min(float(truth_x[-1]), float(orbital_x[-1]))
    common_y_min = max(float(truth_y[-1]), float(orbital_y[-1]))
    common_y_max = min(float(truth_y[0]), float(orbital_y[0]))
    save_profile_plot(
        truth_interpolator=truth_interpolator,
        orbital_interpolator=orbital_interpolator,
        x_coordinates=np.linspace(common_x_min, common_x_max, 2000),
        y_coordinates=np.linspace(common_y_min, common_y_max, 2000),
        selected_profile=args.resolution,
        truth_source_label=CONFIG.truth_source_label,
        truth_resolution_m=TRUTH_RESOLUTION_M,
        selected_resolution_m=ORBITAL_RESOLUTION_M,
    )

    report = {
        "comparison": {
            "selected_profile": args.resolution,
            "truth_source_label": CONFIG.truth_source_label,
            "truth_source_profile": CONFIG.truth_source_profile,
            "operation": "truth minus selected DEM interpolated to truth centers",
        },
        "coordinate_contract": {
            "origin": "configured_map_frame",
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
        "selected_dem": {
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
        "truth_minus_interpolated_selected_dem": statistics,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("DEM alignment validation")
    print("------------------------")
    print(f"Truth bounds:   {truth_bounds}")
    print(f"Selected DEM bounds ({args.resolution}): {orbital_bounds}")
    print()
    print("Truth first cell:")
    print(f"  x = {truth_x[0]:.6f} m")
    print(f"  y = {truth_y[0]:.6f} m")
    print("Truth last cell:")
    print(f"  x = {truth_x[-1]:.6f} m")
    print(f"  y = {truth_y[-1]:.6f} m")
    print()
    print("Selected DEM first cell:")
    print(f"  x = {orbital_x[0]:.6f} m")
    print(f"  y = {orbital_y[0]:.6f} m")
    print("Selected DEM last cell:")
    print(f"  x = {orbital_x[-1]:.6f} m")
    print(f"  y = {orbital_y[-1]:.6f} m")
    print()
    print("Truth − interpolated selected DEM residuals")
    for key, value in statistics.items():
        print(f"{key:24s}: {value}")

    print()
    print(f"Alignment plot: {ALIGNMENT_PLOT_PATH}")
    print(f"Profile plot:   {PROFILE_PLOT_PATH}")
    print(f"JSON report:    {REPORT_PATH}")


if __name__ == "__main__":
    main()

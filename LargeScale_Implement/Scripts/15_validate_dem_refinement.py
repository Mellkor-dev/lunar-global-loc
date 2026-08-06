#!/usr/bin/env python3
"""Compare 0.25 m and 1.5 m OmniLRS DEMs after controlled downsampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import load_resolution_config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "DEM" / "resolution_validation",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=PROJECT_ROOT / "plots" / "dem_resolution_validation",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def overlap_matrix(
    source_count: int,
    source_resolution_m: float,
    target_count: int,
    target_resolution_m: float,
    target_offset_m: float,
) -> csr_matrix:
    """Return normalized 1-D cell-area overlap weights."""
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    source_extent = source_count * source_resolution_m
    tolerance = 1e-9
    for target_index in range(target_count):
        target_left = target_offset_m + target_index * target_resolution_m
        target_right = target_left + target_resolution_m
        if target_left < -tolerance or target_right > source_extent + tolerance:
            raise ValueError("Target grid extends outside source DEM")
        first = max(0, int(np.floor(target_left / source_resolution_m)))
        last = min(
            source_count - 1,
            int(np.ceil(target_right / source_resolution_m)) - 1,
        )
        for source_index in range(first, last + 1):
            source_left = source_index * source_resolution_m
            source_right = source_left + source_resolution_m
            overlap = max(
                0.0,
                min(target_right, source_right) - max(target_left, source_left),
            )
            if overlap > 0.0:
                rows.append(target_index)
                columns.append(source_index)
                values.append(overlap / target_resolution_m)
    matrix = csr_matrix(
        (values, (rows, columns)),
        shape=(target_count, source_count),
        dtype=np.float64,
    )
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    if not np.allclose(sums, 1.0, atol=1e-10):
        raise RuntimeError(f"Invalid overlap weights: row sums {sums.min()}–{sums.max()}")
    return matrix


def area_average(
    source: np.ndarray,
    source_resolution_m: float,
    target_shape: tuple[int, int],
    target_resolution_m: float,
    row_offset_m: float,
    column_offset_m: float,
) -> np.ndarray:
    row_weights = overlap_matrix(
        source.shape[0], source_resolution_m, target_shape[0],
        target_resolution_m, row_offset_m,
    )
    column_weights = overlap_matrix(
        source.shape[1], source_resolution_m, target_shape[1],
        target_resolution_m, column_offset_m,
    )
    intermediate = row_weights @ np.asarray(source, dtype=np.float64)
    return np.asarray((column_weights @ intermediate.T).T, dtype=np.float32)


def exact_block_average(source: np.ndarray, factor: int) -> np.ndarray:
    rows, columns = source.shape
    if rows % factor or columns % factor:
        raise ValueError(f"Shape {source.shape} is not divisible by factor {factor}")
    reshaped = source.reshape(rows // factor, factor, columns // factor, factor)
    return np.asarray(reshaped.mean(axis=(1, 3), dtype=np.float64), dtype=np.float32)


def metrics(residual: np.ndarray, reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    absolute = np.abs(residual)
    return {
        "mean_bias_m": float(np.mean(residual)),
        "mae_m": float(np.mean(absolute)),
        "rmse_m": float(np.sqrt(np.mean(residual.astype(np.float64) ** 2))),
        "standard_deviation_m": float(np.std(residual)),
        "absolute_p50_m": float(np.percentile(absolute, 50)),
        "absolute_p95_m": float(np.percentile(absolute, 95)),
        "absolute_p99_m": float(np.percentile(absolute, 99)),
        "maximum_absolute_m": float(np.max(absolute)),
        "pearson_correlation": float(np.corrcoef(reference.ravel(), candidate.ravel())[0, 1]),
    }


def robust_limit(array: np.ndarray, percentile: float = 99.0) -> float:
    value = float(np.percentile(np.abs(array), percentile))
    return value if value > 0.0 else 1.0


def plot_comparison_10m(
    path: Path,
    dem_0p25_10m: np.ndarray,
    dem_1p5_10m: np.ndarray,
    residual: np.ndarray,
) -> None:
    elevation_min = float(min(dem_0p25_10m.min(), dem_1p5_10m.min()))
    elevation_max = float(max(dem_0p25_10m.max(), dem_1p5_10m.max()))
    residual_limit = robust_limit(residual)
    extent = (-1000.0, 1000.0, -1000.0, 1000.0)
    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    first = axes[0, 0].imshow(
        dem_0p25_10m, origin="upper", extent=extent, cmap="terrain",
        vmin=elevation_min, vmax=elevation_max,
    )
    axes[0, 0].set_title("0.25 m DEM → 10 m/cell")
    axes[0, 1].imshow(
        dem_1p5_10m, origin="upper", extent=extent, cmap="terrain",
        vmin=elevation_min, vmax=elevation_max,
    )
    axes[0, 1].set_title("1.5 m DEM → 10 m/cell")
    difference_image = axes[1, 0].imshow(
        residual, origin="upper", extent=extent, cmap="coolwarm",
        vmin=-residual_limit, vmax=residual_limit,
    )
    axes[1, 0].set_title("0.25→10 m − 1.5→10 m")
    axes[1, 1].hist(residual.ravel(), bins=100, color="#455a64", alpha=0.9)
    axes[1, 1].axvline(0.0, color="black", linewidth=1)
    axes[1, 1].set_title("10 m residual distribution")
    axes[1, 1].set_xlabel("Elevation residual [m]")
    axes[1, 1].set_ylabel("Cells")
    for axis in axes.flat[:3]:
        axis.set_xlabel("Map x / east [m]")
        axis.set_ylabel("Map y / north [m]")
    figure.colorbar(first, ax=axes[0, :], label="Elevation [m]", shrink=0.85)
    figure.colorbar(difference_image, ax=axes[1, 0], label="Residual [m]", shrink=0.85)
    figure.suptitle("DEM comparison on a common 10 m grid", fontsize=16, fontweight="bold")
    figure.subplots_adjust(left=0.07, right=0.92, bottom=0.07, top=0.91, wspace=0.24, hspace=0.24)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_refinement_1p5m(
    path: Path,
    native_1p5: np.ndarray,
    downsampled_0p25: np.ndarray,
    residual: np.ndarray,
    subcell_std: np.ndarray,
) -> None:
    elevation_min = float(min(native_1p5.min(), downsampled_0p25.min()))
    elevation_max = float(max(native_1p5.max(), downsampled_0p25.max()))
    residual_limit = robust_limit(residual)
    extent = (-999.0, 999.0, -999.0, 999.0)
    figure, axes = plt.subplots(2, 2, figsize=(13, 11))
    first = axes[0, 0].imshow(
        native_1p5, origin="upper", extent=extent, cmap="terrain",
        vmin=elevation_min, vmax=elevation_max,
    )
    axes[0, 0].set_title("Native OmniLRS 1.5 m DEM")
    axes[0, 1].imshow(
        downsampled_0p25, origin="upper", extent=extent, cmap="terrain",
        vmin=elevation_min, vmax=elevation_max,
    )
    axes[0, 1].set_title("0.25 m refined DEM → 1.5 m/cell")
    difference_image = axes[1, 0].imshow(
        residual, origin="upper", extent=extent, cmap="coolwarm",
        vmin=-residual_limit, vmax=residual_limit,
    )
    axes[1, 0].set_title("Downsampled refined − native 1.5 m")
    detail_image = axes[1, 1].imshow(
        subcell_std, origin="upper", extent=extent, cmap="magma",
        vmin=0.0, vmax=float(np.percentile(subcell_std, 99)),
    )
    axes[1, 1].set_title("Within-cell 0.25 m elevation variation")
    for axis in axes.flat:
        axis.set_xlabel("Map x / east [m]")
        axis.set_ylabel("Map y / north [m]")
    figure.colorbar(first, ax=axes[0, :], label="Elevation [m]", shrink=0.85)
    figure.colorbar(difference_image, ax=axes[1, 0], label="Residual [m]", shrink=0.85)
    figure.colorbar(detail_image, ax=axes[1, 1], label="Subcell standard deviation [m]", shrink=0.85)
    figure.suptitle("OmniLRS 0.25 m refinement validation", fontsize=16, fontweight="bold")
    figure.subplots_adjust(left=0.07, right=0.92, bottom=0.07, top=0.91, wspace=0.24, hspace=0.24)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    config_0p25 = load_resolution_config("0p25m")
    config_1p5 = load_resolution_config("1p5m")
    dem_0p25 = np.load(config_0p25.orbital_dem_path, mmap_mode="r")
    dem_1p5 = np.load(config_1p5.orbital_dem_path, mmap_mode="r")
    if dem_0p25.shape != (8000, 8000) or dem_1p5.shape != (1334, 1334):
        raise ValueError(
            f"Unexpected input shapes: 0.25 m {dem_0p25.shape}, 1.5 m {dem_1p5.shape}"
        )
    if not np.isfinite(dem_0p25).all() or not np.isfinite(dem_1p5).all():
        raise ValueError("Input DEMs must be fully finite")

    # Both inputs are reduced to the same centered [-1000, 1000] m target.
    dem_0p25_10m = exact_block_average(dem_0p25, 40)
    dem_1p5_10m = area_average(
        dem_1p5, 1.5, (200, 200), 10.0, 0.5, 0.5
    )
    residual_10m = np.asarray(dem_0p25_10m - dem_1p5_10m, dtype=np.float32)

    # The outer native 1.5 m cells extend beyond the 0.25 m DEM. Removing one
    # cell on every edge leaves exactly [-999, 999] m, or 1332 cells.
    refined_common = np.asarray(dem_0p25[4:7996, 4:7996])
    blocks = refined_common.reshape(1332, 6, 1332, 6)
    dem_0p25_1p5m = np.asarray(blocks.mean(axis=(1, 3), dtype=np.float64), dtype=np.float32)
    subcell_std = np.asarray(blocks.std(axis=(1, 3), dtype=np.float64), dtype=np.float32)
    native_1p5_common = np.asarray(dem_1p5[1:-1, 1:-1], dtype=np.float32)
    residual_1p5m = np.asarray(dem_0p25_1p5m - native_1p5_common, dtype=np.float32)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    products = {
        "dem_0p25_to_10m.npy": dem_0p25_10m,
        "dem_1p5_to_10m.npy": dem_1p5_10m,
        "residual_0p25_vs_1p5_at_10m.npy": residual_10m,
        "dem_0p25_to_1p5m_common.npy": dem_0p25_1p5m,
        "residual_refined_vs_native_1p5m.npy": residual_1p5m,
        "subcell_std_0p25_within_1p5m.npy": subcell_std,
    }
    for filename, array in products.items():
        np.save(output_dir / filename, array)

    maximum_absolute_detail = 0.0
    for row_start in range(0, 1332, 64):
        row_stop = min(row_start + 64, 1332)
        detail = (
            blocks[row_start:row_stop]
            - dem_0p25_1p5m[row_start:row_stop, None, :, None]
        )
        maximum_absolute_detail = max(
            maximum_absolute_detail, float(np.max(np.abs(detail)))
        )
    report = {
        "inputs": {
            "dem_0p25m": str(config_0p25.orbital_dem_path),
            "dem_1p5m": str(config_1p5.orbital_dem_path),
        },
        "comparison_10m": {
            "target_shape": [200, 200],
            "target_resolution_m": 10.0,
            "target_edge_bounds_m": [-1000.0, 1000.0, -1000.0, 1000.0],
            **metrics(residual_10m, dem_1p5_10m, dem_0p25_10m),
        },
        "refinement_comparison_1p5m": {
            "target_shape": [1332, 1332],
            "target_resolution_m": 1.5,
            "target_edge_bounds_m": [-999.0, 999.0, -999.0, 999.0],
            **metrics(residual_1p5m, native_1p5_common, dem_0p25_1p5m),
        },
        "sub_1p5m_detail_in_refined_dem": {
            "mean_within_cell_std_m": float(np.mean(subcell_std)),
            "median_within_cell_std_m": float(np.median(subcell_std)),
            "p95_within_cell_std_m": float(np.percentile(subcell_std, 95)),
            "p99_within_cell_std_m": float(np.percentile(subcell_std, 99)),
            "rms_about_1p5m_cell_mean_m": float(
                np.sqrt(np.mean(subcell_std.astype(np.float64) ** 2))
            ),
            "maximum_absolute_detail_m": maximum_absolute_detail,
        },
        "interpretation_note": (
            "Residuals quantify differences between the two OmniLRS products. "
            "They cannot by themselves distinguish interpolation from procedural "
            "crater/rock generation or other terrain-generation changes."
        ),
    }
    report_path = output_dir / "dem_refinement_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    plot_dir = args.plot_dir.resolve()
    if not args.no_plots:
        plot_comparison_10m(
            plot_dir / "dem_comparison_10m.png",
            dem_0p25_10m,
            dem_1p5_10m,
            residual_10m,
        )
        plot_refinement_1p5m(
            plot_dir / "dem_refinement_comparison_1p5m.png",
            native_1p5_common,
            dem_0p25_1p5m,
            residual_1p5m,
            subcell_std,
        )

    print("DEM refinement validation")
    print("-------------------------")
    print(
        "10 m comparison: "
        f"MAE={report['comparison_10m']['mae_m']:.4f} m, "
        f"RMSE={report['comparison_10m']['rmse_m']:.4f} m, "
        f"|error| p95={report['comparison_10m']['absolute_p95_m']:.4f} m"
    )
    print(
        "1.5 m comparison: "
        f"MAE={report['refinement_comparison_1p5m']['mae_m']:.4f} m, "
        f"RMSE={report['refinement_comparison_1p5m']['rmse_m']:.4f} m, "
        f"|error| p95={report['refinement_comparison_1p5m']['absolute_p95_m']:.4f} m"
    )
    print(
        "Subcell detail:   "
        f"mean std={report['sub_1p5m_detail_in_refined_dem']['mean_within_cell_std_m']:.4f} m, "
        f"p95 std={report['sub_1p5m_detail_in_refined_dem']['p95_within_cell_std_m']:.4f} m"
    )
    print(f"Report:           {report_path}")
    print(f"Arrays:           {output_dir}")
    if not args.no_plots:
        print(f"Plots:            {plot_dir}")


if __name__ == "__main__":
    main()

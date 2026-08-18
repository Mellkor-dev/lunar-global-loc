#!/usr/bin/env python3
"""Refine every DARCES result with covariance-aware exhaustive RANSAC."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching.darces import decimate_reference_points_xy
from matching.ransac import exhaustive_ransac
from pipeline_config import add_resolution_argument, load_resolution_config
from traversal_presentation import environment_display_name


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_resolution_argument(parser)
    parser.add_argument(
        "--darces-results",
        type=Path,
        help="Override results/<resolution>_px/darces_all_sites.json",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--minimum-overlap", type=float, default=0.50)
    parser.add_argument("--minimum-triangle-area", type=float, default=1.0)
    parser.add_argument("--maximum-refinement-iterations", type=int, default=5)
    parser.add_argument(
        "--minimum-fitness-improvement-m",
        type=float,
        default=0.01,
        help="Minimum terrain-MAE reduction required to select RANSAC (default: 0.01 m)",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_features(
    path: Path,
    expected_kind: str,
    fallback_covariance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        kind = str(data["feature_kind"])
        xyz = np.column_stack((data["x_m"], data["y_m"], data["z_m"])).astype(
            np.float64
        )
        covariance = (
            np.asarray(data["covariance"], dtype=np.float64)
            if "covariance" in data.files
            else None
        )
    if kind != expected_kind:
        raise ValueError(f"{path}: feature kind {kind}, expected {expected_kind}")
    if covariance is None:
        if fallback_covariance is None:
            raise ValueError(f"{path}: no feature covariance is available")
        covariance = np.repeat(
            np.asarray(fallback_covariance, dtype=np.float64)[None, :, :],
            len(xyz),
            axis=0,
        )
    return xyz, covariance


def preserved_record(
    darces: dict[str, object],
    status: str,
    reason: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "site": int(darces["site"]),
        "status": status,
        "estimate_source": "darces" if darces.get("status") == "solution" else "none",
        "fallback_reason": reason,
        "darces": darces,
    }
    if darces.get("status") == "solution":
        for key in (
            "estimated_x_m",
            "estimated_y_m",
            "estimated_z_m",
            "estimated_heading_deg",
            "fitness",
            "overlap",
            "truth_x_m",
            "truth_y_m",
            "truth_z_m",
            "truth_z_stage_m",
            "stage_to_dem_vertical_offset_m",
            "xy_error_m",
            "z_error_m",
            "heading_error_deg",
        ):
            if key in darces:
                record[key] = darces[key]
    return record


def serializable_ransac(result: dict[str, object]) -> dict[str, object]:
    converted: dict[str, object] = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        elif isinstance(value, np.integer):
            converted[key] = int(value)
        elif isinstance(value, np.floating):
            converted[key] = float(value)
        else:
            converted[key] = value
    return converted


def plot_correspondences(
    path: Path,
    site: int,
    local_features_xyz: np.ndarray,
    global_features_xyz: np.ndarray,
    local_indices: np.ndarray,
    global_indices: np.ndarray,
    result: dict[str, object],
    darces: dict[str, object],
    *,
    environment_name: str,
    resolution_m: float,
) -> None:
    rotation = np.asarray(result["rotation"], dtype=np.float64)
    translation = np.asarray(result["translation_xy_m"], dtype=np.float64)
    projected = local_features_xyz[local_indices, :2] @ rotation.T + translation
    targets = global_features_xyz[global_indices, :2]
    inlier_pairs = {
        (int(local), int(global_))
        for local, global_ in zip(
            result["inlier_local_indices"],
            result["inlier_global_indices"],
            strict=True,
        )
    }
    figure, axis = plt.subplots(figsize=(9.2, 7.5), layout="constrained")
    for index, (local_index, global_index) in enumerate(
        zip(local_indices, global_indices, strict=True)
    ):
        is_inlier = (int(local_index), int(global_index)) in inlier_pairs
        color = "#2e7d32" if is_inlier else "#c62828"
        label = None
        if is_inlier and "Inlier" not in axis.get_legend_handles_labels()[1]:
            label = "Inlier"
        if not is_inlier and "Outlier" not in axis.get_legend_handles_labels()[1]:
            label = "Outlier"
        axis.plot(
            [projected[index, 0], targets[index, 0]],
            [projected[index, 1], targets[index, 1]],
            color=color,
            linewidth=2.2,
            alpha=0.9,
            solid_capstyle="round",
            zorder=2,
            label=label,
        )
    axis.scatter(
        targets[:, 0], targets[:, 1], marker="o", facecolors="none",
        edgecolors="black", linewidths=2.0, s=145, zorder=4,
        label="Global feature",
    )
    axis.scatter(
        projected[:, 0], projected[:, 1], marker="x", color="#1565c0",
        s=135, linewidths=2.5, zorder=5,
        label="Transformed local feature",
    )
    axis.scatter(
        translation[0], translation[1], marker="^", color="#6a1b9a",
        edgecolors="white", linewidths=1.0, s=190, zorder=6,
        label="RANSAC pose",
    )
    if "estimated_x_m" in darces and "estimated_y_m" in darces:
        axis.scatter(
            darces["estimated_x_m"], darces["estimated_y_m"], marker="+",
            color="#ef6c00", linewidths=2.8, s=190, zorder=7,
            label="DARCES pose",
        )
    if "truth_x_m" in darces and "truth_y_m" in darces:
        axis.scatter(
            darces["truth_x_m"], darces["truth_y_m"], marker="*",
            color="#00838f", edgecolors="white", linewidths=0.9,
            s=220, zorder=8, label="Truth (evaluation only)",
        )
    axis.set_title(
        f"{environment_name} Site {site:02d} "
        f"{resolution_m:g}m/px RANSAC correspondences\n"
        f"Inliers={result['inlier_count']}/"
        f"{result['input_correspondence_count']}",
        fontsize=18,
        fontweight="bold",
        pad=9,
    )
    axis.set_xlabel("Map x / east [m]", fontsize=14)
    axis.set_ylabel("Map y / north [m]", fontsize=14)
    axis.tick_params(axis="both", labelsize=11)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend(
        loc="upper right",
        fontsize=10.5,
        framealpha=0.92,
        borderpad=0.55,
        handletextpad=0.6,
        handlelength=2.0,
        labelspacing=0.4,
        markerscale=1.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie in (0, 1)")
    config = load_resolution_config(args.resolution)
    if args.minimum_fitness_improvement_m < 0.0:
        raise ValueError("--minimum-fitness-improvement-m must be non-negative")
    darces_path = args.darces_results or config.results_path / "darces_all_sites.json"
    with darces_path.open("r", encoding="utf-8") as stream:
        darces_payload = json.load(stream)
    darces_settings = darces_payload.get("settings", {})
    if darces_settings.get("fitness_reference") != "raw_lidar_xy_decimated":
        raise ValueError(
            f"{darces_path} uses legacy grid-cell terrain fitness; rerun "
            "Scripts/09_darces_all_sites.py before RANSAC"
        )
    darces_sites = darces_payload.get("sites")
    if not isinstance(darces_sites, list):
        raise ValueError(f"{darces_path}: 'sites' must be a list")

    with config.feature_uncertainty_path.open("r", encoding="utf-8") as stream:
        uncertainty = json.load(stream)
    global_covariance = np.asarray(uncertainty["covariance_m2"], dtype=np.float64)
    global_features_xyz, global_covariances = load_features(
        config.global_features_path,
        config.features.kind,
        fallback_covariance=global_covariance,
    )
    global_dem = np.asarray(np.load(config.orbital_dem_path), dtype=np.float64)
    global_x, global_y = config.orbital_raster.coordinates()
    records: list[dict[str, object]] = []
    plot_directory = config.plots_path / "ransac"

    for darces in darces_sites:
        site = int(darces["site"])
        if darces.get("status") != "solution":
            records.append(
                preserved_record(darces, "darces_unavailable", "DARCES has no pose estimate")
            )
            continue
        if (
            "correspondence_local_indices" not in darces
            or "correspondence_global_indices" not in darces
        ):
            records.append(
                preserved_record(
                    darces,
                    "fallback_darces",
                    "DARCES result predates expanded correspondences",
                )
            )
            continue

        try:
            feature_path = (
                config.local_features_path / f"local_craters_site_{site:02d}.npz"
            )
            grid_path = config.gridded_maps_path / f"grid_site_{site:02d}.npz"
            local_features_xyz, local_covariances = load_features(
                feature_path, config.features.kind
            )
            with np.load(grid_path, allow_pickle=False) as grid:
                local_grid = np.asarray(grid["elevation"], dtype=np.float64)
                local_x = np.asarray(grid["x_centers_m"], dtype=np.float64)
                local_y = np.asarray(grid["y_centers_m"], dtype=np.float64)
            reference_spacing_m = config.orbital_raster.resolution_m / 2.0
            reference_points_xyz = decimate_reference_points_xy(
                np.asarray(
                    np.load(
                        config.leveled_maps_path
                        / f"leveled_site_{site:02d}.npy",
                        allow_pickle=False,
                    ),
                    dtype=np.float64,
                ),
                reference_spacing_m,
            )
            local_indices = np.asarray(
                darces["correspondence_local_indices"], dtype=np.int64
            )
            global_indices = np.asarray(
                darces["correspondence_global_indices"], dtype=np.int64
            )
            start = time.perf_counter()
            result = exhaustive_ransac(
                local_features_xyz,
                global_features_xyz,
                local_covariances,
                global_covariances,
                local_indices,
                global_indices,
                local_grid,
                global_dem,
                local_x,
                local_y,
                global_x,
                global_y,
                confidence=args.confidence,
                minimum_overlap=args.minimum_overlap,
                minimum_triangle_area_m2=args.minimum_triangle_area,
                maximum_refinement_iterations=args.maximum_refinement_iterations,
                reference_points_xyz=reference_points_xyz,
            )
            runtime_s = time.perf_counter() - start
        except Exception as error:
            records.append(
                preserved_record(darces, "fallback_darces", f"RANSAC input error: {error}")
            )
            continue

        if result["status"] in {"solution", "minimal_unverified"}:
            ransac_xy = np.asarray(result["translation_xy_m"], dtype=np.float64)
            rejected_outlier = int(result["outlier_count"]) > 0
            fitness_improvement_m = float(result["terrain_fitness"]) - float(
                darces["fitness"]
            )
            improves_fitness = (
                fitness_improvement_m >= args.minimum_fitness_improvement_m
            )
            use_ransac_estimate = (
                result["status"] == "solution"
                and improves_fitness
            )
            if use_ransac_estimate:
                estimate_xy = ransac_xy
                estimated_z_m = float(result["translation_z_m"])
                estimated_heading_deg = float(result["heading_deg"])
                estimate_fitness = float(result["terrain_fitness"])
                estimate_overlap = float(result["terrain_overlap"])
                estimate_source = "ransac"
                preservation_reason = None
            else:
                estimate_xy = np.array(
                    [darces["estimated_x_m"], darces["estimated_y_m"]],
                    dtype=np.float64,
                )
                estimated_z_m = float(darces["estimated_z_m"])
                estimated_heading_deg = float(darces["estimated_heading_deg"])
                estimate_fitness = float(darces["fitness"])
                estimate_overlap = float(darces["overlap"])
                estimate_source = "darces"
                preservation_reason = (
                    "exactly three correspondences cannot be outlier-tested"
                    if result["status"] == "minimal_unverified"
                    else "RANSAC terrain fitness improvement was below threshold"
                )
            record = {
                "site": site,
                "status": result["status"],
                "estimate_source": estimate_source,
                "runtime_s": runtime_s,
                "estimated_x_m": float(estimate_xy[0]),
                "estimated_y_m": float(estimate_xy[1]),
                "estimated_z_m": estimated_z_m,
                "estimated_heading_deg": estimated_heading_deg,
                "fitness": estimate_fitness,
                "overlap": estimate_overlap,
                "preservation_reason": preservation_reason,
                "fitness_improvement_m": fitness_improvement_m,
                "ransac_rejected_outlier": rejected_outlier,
                "fitness_reference": "raw_lidar_xy_decimated",
                "fitness_reference_spacing_m": reference_spacing_m,
                "fitness_reference_point_count": len(reference_points_xyz),
                "ransac": serializable_ransac(result),
                "darces": darces,
            }
            if "truth_x_m" in darces and "truth_y_m" in darces:
                truth_xy = np.array(
                    [darces["truth_x_m"], darces["truth_y_m"]], dtype=np.float64
                )
                record["truth_x_m"] = float(truth_xy[0])
                record["truth_y_m"] = float(truth_xy[1])
                record["truth_z_m"] = float(darces["truth_z_m"])
                if "truth_z_stage_m" in darces:
                    record["truth_z_stage_m"] = float(
                        darces["truth_z_stage_m"]
                    )
                if "stage_to_dem_vertical_offset_m" in darces:
                    record["stage_to_dem_vertical_offset_m"] = float(
                        darces["stage_to_dem_vertical_offset_m"]
                    )
                record["xy_error_m"] = float(np.linalg.norm(estimate_xy - truth_xy))
                record["z_error_m"] = float(
                    abs(estimated_z_m - float(darces["truth_z_m"]))
                )
                if "heading_measurement_deg" in darces:
                    record["heading_error_deg"] = float(
                        abs(
                            (
                                estimated_heading_deg
                                - float(darces["heading_measurement_deg"])
                                + 180.0
                            )
                            % 360.0
                            - 180.0
                        )
                    )
            records.append(record)
            if not args.no_plots:
                plot_correspondences(
                    plot_directory / f"ransac_site_{site:02d}.png",
                    site,
                    local_features_xyz,
                    global_features_xyz,
                    local_indices,
                    global_indices,
                    result,
                    darces,
                    environment_name=environment_display_name(config),
                    resolution_m=config.orbital_raster.resolution_m,
                )
        else:
            records.append(
                preserved_record(
                    darces,
                    "fallback_darces",
                    f"RANSAC returned {result['status']}",
                )
            )

    output_json = config.results_path / "ransac_all_sites.json"
    output_csv = config.results_path / "ransac_all_sites.csv"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "resolution": args.resolution,
            "confidence": args.confidence,
            "minimum_overlap": args.minimum_overlap,
            "minimum_triangle_area_m2": args.minimum_triangle_area,
            "maximum_refinement_iterations": args.maximum_refinement_iterations,
            "minimum_fitness_improvement_m": args.minimum_fitness_improvement_m,
            "fitness_reference": "raw_lidar_xy_decimated",
            "fitness_reference_spacing_factor": 0.5,
            "stage_to_dem_vertical_offset_m": (
                config.stage_to_dem_vertical_offset_m
            ),
            "darces_results": str(darces_path.resolve()),
        },
        "sites": records,
    }
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    flat_records = [
        {key: value for key, value in record.items() if key not in {"ransac", "darces"}}
        for record in records
    ]
    fieldnames = sorted({key for record in flat_records for key in record})
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_records)

    print("RANSAC all-sites refinement")
    print("---------------------------")
    for status in sorted({str(record["status"]) for record in records}):
        print(f"{status}: {sum(record['status'] == status for record in records)}")
    print(f"JSON:  {output_json}")
    print(f"CSV:   {output_csv}")
    if not args.no_plots:
        print(f"Plots: {plot_directory}")


if __name__ == "__main__":
    main()

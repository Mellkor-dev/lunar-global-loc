#!/usr/bin/env python3
"""Run independent 2.5-D single-frame MOGA on all captured sites."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import add_resolution_argument, load_resolution_config
from site_selection import selected_sites_for_config
from refinement.moga import (
    FeatureObservation,
    MogaProblem,
    solve_single_frame_moga,
    wrap_angle,
)


ODOMETRY_PATTERN = re.compile(r"odom_site_(\d+)\.npy$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_resolution_argument(parser)
    parser.add_argument(
        "--ransac-results",
        type=Path,
        help="Override results/<resolution>_px/ransac_all_sites.json",
    )
    parser.add_argument("--heading-sigma-deg", type=float, default=1.0)
    parser.add_argument("--max-evaluations", type=int, default=300)
    parser.add_argument("--relative-tolerance", type=float, default=1e-10)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def load_odometry(
    directory: Path,
    selected_sites: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    records: list[tuple[int, np.ndarray]] = []
    for path in directory.glob("odom_site_*.npy"):
        match = ODOMETRY_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        pose = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"{path} must contain [x,y,z,qx,qy,qz,qw]")
        records.append((int(match.group(1)), pose))
    if not records:
        raise FileNotFoundError(f"No odometry scans found in {directory}")
    if selected_sites is not None:
        selected = set(selected_sites)
        records = [item for item in records if item[0] in selected]
        missing = sorted(selected.difference(site for site, _pose in records))
        if missing:
            raise FileNotFoundError(f"Selected odometry sites are missing: {missing}")
    records.sort(key=lambda item: item[0])
    return np.asarray([item[0] for item in records]), np.stack([item[1] for item in records])


def headings_from_odometry(odometry: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_quat(odometry[:, 3:]).as_matrix()
    return np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])


def localized_pose(record: dict[str, object]) -> np.ndarray | None:
    if record.get("estimate_source") not in {"darces", "ransac"}:
        return None
    required = ("estimated_x_m", "estimated_y_m", "estimated_heading_deg")
    if not all(key in record for key in required):
        return None
    pose = np.array(
        [
            float(record["estimated_x_m"]),
            float(record["estimated_y_m"]),
            np.radians(float(record["estimated_heading_deg"])),
        ],
        dtype=np.float64,
    )
    return pose if np.isfinite(pose).all() else None


def load_global_features(config) -> tuple[np.ndarray, np.ndarray]:
    with np.load(config.global_features_path, allow_pickle=False) as data:
        xyz = np.column_stack((data["x_m"], data["y_m"], data["z_m"])).astype(
            np.float64
        )
        covariance = (
            np.asarray(data["covariance"], dtype=np.float64)
            if "covariance" in data.files
            else None
        )
    if covariance is None:
        with config.feature_uncertainty_path.open("r", encoding="utf-8") as stream:
            uncertainty = json.load(stream)
        covariance = np.repeat(
            np.asarray(uncertainty["covariance_m2"], dtype=np.float64)[None, :, :],
            len(xyz),
            axis=0,
        )
    return xyz, covariance


def load_local_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        xyz = np.column_stack((data["x_m"], data["y_m"], data["z_m"])).astype(
            np.float64
        )
        covariance = np.asarray(data["covariance"], dtype=np.float64)
    return xyz, covariance


def build_site_problem(
    config,
    site: int,
    heading_measurement: float,
    record: dict[str, object],
    global_features_xyz: np.ndarray,
    global_covariances: np.ndarray,
) -> tuple[MogaProblem, float, str] | None:
    """Build one isolated frame with private landmark state."""
    initial_pose = localized_pose(record)
    ransac = record.get("ransac")
    if initial_pose is None or not isinstance(ransac, dict):
        return None
    local_indices = np.asarray(ransac.get("inlier_local_indices", []), dtype=np.int64)
    global_indices = np.asarray(ransac.get("inlier_global_indices", []), dtype=np.int64)
    if len(local_indices) != len(global_indices) or len(local_indices) < 3:
        return None
    if len(np.unique(local_indices)) != len(local_indices):
        raise ValueError(f"Site {site:02d} has duplicate local feature indices")
    if len(np.unique(global_indices)) != len(global_indices):
        raise ValueError(f"Site {site:02d} has duplicate global feature indices")

    local_xyz, local_covariance = load_local_features(
        config.local_features_path / f"local_craters_site_{site:02d}.npz"
    )
    if np.any(local_indices < 0) or np.any(local_indices >= len(local_xyz)):
        raise IndexError(f"Site {site:02d} has an out-of-range local feature index")
    if np.any(global_indices < 0) or np.any(global_indices >= len(global_features_xyz)):
        raise IndexError(f"Site {site:02d} has an out-of-range global feature index")

    landmark_global_indices = np.asarray(global_indices, dtype=np.int64)
    landmark_lookup = {
        int(global_index): index
        for index, global_index in enumerate(landmark_global_indices)
    }
    observations = tuple(
        FeatureObservation(
            pose_index=0,
            landmark_index=landmark_lookup[int(global_index)],
            local_xy_m=local_xyz[local_index, :2].copy(),
            local_covariance_xy_m2=local_covariance[local_index, :2, :2].copy(),
        )
        for local_index, global_index in zip(local_indices, global_indices, strict=True)
    )
    problem = MogaProblem(
        site_numbers=np.asarray([site], dtype=np.int64),
        initial_poses=initial_pose[None, :],
        heading_measurements_rad=np.asarray([heading_measurement], dtype=np.float64),
        landmark_global_indices=landmark_global_indices,
        initial_landmarks_xy_m=global_features_xyz[landmark_global_indices, :2].copy(),
        landmark_global_covariances_xy_m2=global_covariances[
            landmark_global_indices, :2, :2
        ].copy(),
        observations=observations,
    )
    return problem, float(record["estimated_z_m"]), str(record["estimate_source"])


def plot_solution(
    path: Path,
    site_numbers: np.ndarray,
    truth_xy: np.ndarray,
    initial_poses: np.ndarray,
    optimized_poses: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.plot(truth_xy[:, 0], truth_xy[:, 1], "k.-", label="Simulation truth")
    axis.scatter(
        initial_poses[:, 0], initial_poses[:, 1], marker="o", color="#ef6c00",
        alpha=0.8, label="Initial RANSAC/DARCES",
    )
    axis.scatter(
        optimized_poses[:, 0], optimized_poses[:, 1], marker="o", color="#1565c0",
        label="MOGA",
    )
    for site, point in zip(site_numbers, optimized_poses, strict=True):
        axis.annotate(f"{site:02d}", point[:2], xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set_title(
        "Independent single-frame MOGA (odometry position is evaluation only)"
    )
    axis.set_xlabel("Map x / east [m]")
    axis.set_ylabel("Map y / north [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    config = load_resolution_config(args.resolution)
    ransac_path = args.ransac_results or config.results_path / "ransac_all_sites.json"
    with ransac_path.open("r", encoding="utf-8") as stream:
        ransac_payload = json.load(stream)
    records = ransac_payload.get("sites")
    if not isinstance(records, list):
        raise ValueError(f"{ransac_path}: 'sites' must be a list")
    records_by_site = {int(record["site"]): record for record in records}
    selected_sites = selected_sites_for_config(config)
    site_numbers, odometry = load_odometry(
        config.captures_path / "odom_scans", selected_sites
    )
    if set(site_numbers) != set(records_by_site):
        raise ValueError("RANSAC result sites and odometry sites do not match")
    records_by_site = {int(site): records_by_site[int(site)] for site in site_numbers}
    odometry_headings = headings_from_odometry(odometry)
    global_features_xyz, global_covariances = load_global_features(config)
    site_runs: dict[int, tuple[MogaProblem, float, str, dict[str, object]]] = {}
    for site_value, heading_measurement in zip(
        site_numbers, odometry_headings, strict=True
    ):
        site = int(site_value)
        built = build_site_problem(
            config,
            site,
            float(heading_measurement),
            records_by_site[site],
            global_features_xyz,
            global_covariances,
        )
        if built is None:
            continue
        problem, elevation, initial_source = built
        solution = solve_single_frame_moga(
            problem,
            heading_sigma_deg=args.heading_sigma_deg,
            maximum_function_evaluations=args.max_evaluations,
            relative_tolerance=args.relative_tolerance,
        )
        site_runs[site] = (problem, elevation, initial_source, solution)
    truth_index = {int(site): index for index, site in enumerate(site_numbers)}
    output_sites: list[dict[str, object]] = []
    for site_value in site_numbers:
        site = int(site_value)
        truth_i = truth_index[site]
        truth_z_stage = float(odometry[truth_i, 2])
        truth_z = config.stage_z_to_dem_datum(truth_z_stage)
        if site not in site_runs:
            output_sites.append(
                {
                    "site": site,
                    "status": "feature_pose_unavailable",
                    "initial_source": "none",
                    "truth_x_m": float(odometry[truth_i, 0]),
                    "truth_y_m": float(odometry[truth_i, 1]),
                    "truth_z_m": truth_z,
                    "truth_z_stage_m": truth_z_stage,
                    "stage_to_dem_vertical_offset_m": (
                        config.stage_to_dem_vertical_offset_m
                    ),
                    "xy_error_m": None,
                    "z_error_m": None,
                    "heading_error_deg": None,
                }
            )
            continue
        problem, elevation, initial_source, solution = site_runs[site]
        optimized_pose = np.asarray(solution["poses"], dtype=np.float64)[0]
        pose_covariance = np.asarray(solution["pose_covariances"], dtype=np.float64)[0]
        xy_error = float(np.linalg.norm(optimized_pose[:2] - odometry[truth_i, :2]))
        heading_error_deg = float(
            abs(np.degrees(wrap_angle(optimized_pose[2] - odometry_headings[truth_i])))
        )
        output_sites.append(
            {
                "site": site,
                "status": "solution" if solution["success"] else "optimizer_incomplete",
                "initial_source": initial_source,
                "initial_x_m": float(problem.initial_poses[0, 0]),
                "initial_y_m": float(problem.initial_poses[0, 1]),
                "initial_heading_deg": float(np.degrees(problem.initial_poses[0, 2])),
                "estimated_x_m": float(optimized_pose[0]),
                "estimated_y_m": float(optimized_pose[1]),
                "estimated_z_m": elevation,
                "estimated_heading_deg": float(np.degrees(optimized_pose[2])),
                "pose_covariance_xyyaw": pose_covariance.tolist(),
                "truth_x_m": float(odometry[truth_i, 0]),
                "truth_y_m": float(odometry[truth_i, 1]),
                "truth_z_m": truth_z,
                "truth_z_stage_m": truth_z_stage,
                "stage_to_dem_vertical_offset_m": (
                    config.stage_to_dem_vertical_offset_m
                ),
                "xy_error_m": xy_error,
                "z_error_m": float(abs(elevation - truth_z)),
                "heading_error_deg": heading_error_deg,
                "moga_initial_cost": float(solution["initial_cost"]),
                "moga_final_cost": float(solution["final_cost"]),
            }
        )
    output_json = config.results_path / "moga_all_sites.json"
    output_csv = config.results_path / "moga_all_sites.csv"
    payload = {
        "settings": {
            "resolution": args.resolution,
            "mode": "independent_single_frame_2p5d",
            "heading_sigma_deg": args.heading_sigma_deg,
            "odometry_position_usage": "evaluation_only",
            "odometry_heading_usage": "heading_sensor_constraint",
            "relative_odometry_usage": "none",
            "stage_to_dem_vertical_offset_m": (
                config.stage_to_dem_vertical_offset_m
            ),
            "maximum_function_evaluations": args.max_evaluations,
            "relative_tolerance": args.relative_tolerance,
            "ransac_results": str(ransac_path.resolve()),
        },
        "optimization": {
            "mode": "independent_single_frame",
            "success": bool(site_runs)
            and all(bool(run[3]["success"]) for run in site_runs.values()),
            "message": (
                "Independent single-frame refinements completed"
                if site_runs
                else "No feature-derived RANSAC/DARCES poses were available"
            ),
            "solved_frames": len(site_runs),
            "initial_cost": float(
                sum(float(run[3]["initial_cost"]) for run in site_runs.values())
            ),
            "final_cost": float(
                sum(float(run[3]["final_cost"]) for run in site_runs.values())
            ),
            "function_evaluations": int(
                sum(int(run[3]["function_evaluations"]) for run in site_runs.values())
            ),
        },
        "site_optimizations": [
            {
                "site": site,
                "success": bool(solution["success"]),
                "message": str(solution["message"]),
                "status_code": int(solution["status_code"]),
                "function_evaluations": int(solution["function_evaluations"]),
                "initial_cost": float(solution["initial_cost"]),
                "final_cost": float(solution["final_cost"]),
                "optimality": float(solution["optimality"]),
                "landmark_global_indices": problem.landmark_global_indices.tolist(),
                "optimized_landmarks_xy_m": np.asarray(
                    solution["landmarks_xy_m"]
                ).tolist(),
            }
            for site, (problem, _elevation, _source, solution) in site_runs.items()
        ],
        "sites": output_sites,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    flat_sites = [
        {key: value for key, value in record.items() if key != "pose_covariance_xyyaw"}
        for record in output_sites
    ]
    fieldnames = sorted({key for record in flat_sites for key in record})
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_sites)
    plot_path = config.plots_path / "moga" / "moga_traversal.png"
    if not args.no_plot:
        plotted_sites = np.asarray(list(site_runs), dtype=np.int64)
        if len(plotted_sites):
            initial_poses = np.vstack(
                [site_runs[int(site)][0].initial_poses[0] for site in plotted_sites]
            )
            optimized_poses = np.vstack(
                [np.asarray(site_runs[int(site)][3]["poses"])[0] for site in plotted_sites]
            )
        else:
            initial_poses = np.empty((0, 3), dtype=np.float64)
            optimized_poses = np.empty((0, 3), dtype=np.float64)
        plot_solution(
            plot_path, plotted_sites, odometry[:, :2], initial_poses, optimized_poses
        )
    total_observations = sum(len(run[0].observations) for run in site_runs.values())
    total_landmarks = sum(run[0].landmark_count for run in site_runs.values())
    initial_cost = float(sum(float(run[3]["initial_cost"]) for run in site_runs.values()))
    final_cost = float(sum(float(run[3]["final_cost"]) for run in site_runs.values()))
    print("MOGA independent single-frame refinement")
    print("----------------------------------------")
    print(f"Successful:    {sum(bool(run[3]['success']) for run in site_runs.values())}/{len(site_runs)}")
    print(f"Observations:  {total_observations}")
    print(f"Landmarks:     {total_landmarks} (private per frame)")
    print(f"Feature poses: {len(site_runs)}/{len(site_numbers)}")
    print(f"Unavailable:   {len(site_numbers) - len(site_runs)}")
    print(f"Initial cost:  {initial_cost:.6f}")
    print(f"Final cost:    {final_cost:.6f}")
    errors = [site["xy_error_m"] for site in output_sites if site["xy_error_m"] is not None]
    if errors:
        print(f"Median XY err: {np.median(errors):.3f} m (available feature poses only)")
    else:
        print("Median XY err: N/A (no available feature poses)")
    print(f"JSON:          {output_json}")
    print(f"CSV:           {output_csv}")
    if not args.no_plot:
        print(f"Plot:          {plot_path}")


if __name__ == "__main__":
    main()

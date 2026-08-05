#!/usr/bin/env python3
"""Run 2.5-D multi-frame MOGA from RANSAC consensus results."""

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
from refinement.moga import (
    FeatureObservation,
    MogaProblem,
    solve_moga,
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


def load_odometry(directory: Path) -> tuple[np.ndarray, np.ndarray]:
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


def initialize_feature_poses(
    site_numbers: np.ndarray,
    records_by_site: dict[int, dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    available_sites: list[int] = []
    poses: list[np.ndarray] = []
    elevations: list[float] = []
    sources: list[str] = []
    for site_value in site_numbers:
        site = int(site_value)
        record = records_by_site[site]
        pose = localized_pose(record)
        if pose is None or not isinstance(record.get("ransac"), dict):
            continue
        available_sites.append(site)
        poses.append(pose)
        elevations.append(float(record["estimated_z_m"]))
        sources.append(str(record["estimate_source"]))
    if not poses:
        raise ValueError("No feature-derived RANSAC/DARCES poses are available for MOGA")
    return (
        np.asarray(available_sites, dtype=np.int64),
        np.stack(poses),
        np.asarray(elevations, dtype=np.float64),
        np.asarray(sources, dtype=object),
    )


def build_problem(
    config,
    site_numbers: np.ndarray,
    heading_measurements: np.ndarray,
    records_by_site: dict[int, dict[str, object]],
    global_features_xyz: np.ndarray,
    global_covariances: np.ndarray,
) -> tuple[MogaProblem, np.ndarray, np.ndarray]:
    available_sites, initial_poses, elevations, sources = initialize_feature_poses(
        site_numbers, records_by_site
    )
    raw_observations: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    used_global_indices: set[int] = set()
    site_to_pose = {int(site): index for index, site in enumerate(available_sites)}
    for site, record in records_by_site.items():
        if site not in site_to_pose:
            continue
        ransac = record.get("ransac")
        if not isinstance(ransac, dict):
            continue
        local_indices = np.asarray(ransac.get("inlier_local_indices", []), dtype=np.int64)
        global_indices = np.asarray(ransac.get("inlier_global_indices", []), dtype=np.int64)
        if len(local_indices) != len(global_indices) or len(local_indices) == 0:
            continue
        local_xyz, local_covariance = load_local_features(
            config.local_features_path / f"local_craters_site_{site:02d}.npz"
        )
        pose_index = site_to_pose[site]
        for local_index, global_index in zip(local_indices, global_indices, strict=True):
            used_global_indices.add(int(global_index))
            raw_observations.append(
                (
                    pose_index,
                    int(global_index),
                    local_xyz[local_index, :2].copy(),
                    local_covariance[local_index, :2, :2].copy(),
                )
            )
    landmark_global_indices = np.asarray(sorted(used_global_indices), dtype=np.int64)
    if len(landmark_global_indices) == 0:
        raise ValueError("RANSAC results contain no inlier feature observations")
    landmark_lookup = {
        int(global_index): index
        for index, global_index in enumerate(landmark_global_indices)
    }
    observations = tuple(
        FeatureObservation(
            pose_index=pose_index,
            landmark_index=landmark_lookup[global_index],
            local_xy_m=local_xy,
            local_covariance_xy_m2=local_covariance,
        )
        for pose_index, global_index, local_xy, local_covariance in raw_observations
    )
    heading_by_site = {
        int(site): float(heading) for site, heading in zip(site_numbers, heading_measurements, strict=True)
    }
    problem = MogaProblem(
        site_numbers=available_sites,
        initial_poses=initial_poses,
        heading_measurements_rad=np.asarray(
            [heading_by_site[int(site)] for site in available_sites], dtype=np.float64
        ),
        landmark_global_indices=landmark_global_indices,
        initial_landmarks_xy_m=global_features_xyz[landmark_global_indices, :2].copy(),
        landmark_global_covariances_xy_m2=global_covariances[
            landmark_global_indices, :2, :2
        ].copy(),
        observations=observations,
    )
    return problem, elevations, sources


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
    axis.set_title("Feature-only MOGA estimates (odometry position is evaluation only)")
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
    site_numbers, odometry = load_odometry(config.captures_path / "odom_scans")
    if set(site_numbers) != set(records_by_site):
        raise ValueError("RANSAC result sites and odometry sites do not match")
    records_by_site = {int(site): records_by_site[int(site)] for site in site_numbers}
    odometry_headings = headings_from_odometry(odometry)
    global_features_xyz, global_covariances = load_global_features(config)
    problem, elevations, initial_sources = build_problem(
        config,
        site_numbers,
        odometry_headings,
        records_by_site,
        global_features_xyz,
        global_covariances,
    )
    solution = solve_moga(
        problem,
        heading_sigma_deg=args.heading_sigma_deg,
        maximum_function_evaluations=args.max_evaluations,
        relative_tolerance=args.relative_tolerance,
    )
    optimized = np.asarray(solution["poses"], dtype=np.float64)
    pose_covariances = np.asarray(solution["pose_covariances"], dtype=np.float64)
    optimized_index = {int(site): index for index, site in enumerate(problem.site_numbers)}
    truth_index = {int(site): index for index, site in enumerate(site_numbers)}
    output_sites: list[dict[str, object]] = []
    for site_value in site_numbers:
        site = int(site_value)
        truth_i = truth_index[site]
        if site not in optimized_index:
            output_sites.append(
                {
                    "site": site,
                    "status": "feature_pose_unavailable",
                    "initial_source": "none",
                    "truth_x_m": float(odometry[truth_i, 0]),
                    "truth_y_m": float(odometry[truth_i, 1]),
                    "truth_z_m": float(odometry[truth_i, 2]),
                    "xy_error_m": None,
                    "z_error_m": None,
                    "heading_error_deg": None,
                }
            )
            continue
        index = optimized_index[site]
        xy_error = float(np.linalg.norm(optimized[index, :2] - odometry[truth_i, :2]))
        heading_error_deg = float(
            abs(np.degrees(wrap_angle(optimized[index, 2] - odometry_headings[truth_i])))
        )
        output_sites.append(
            {
                "site": site,
                "status": "solution" if solution["success"] else "optimizer_incomplete",
                "initial_source": str(initial_sources[index]),
                "initial_x_m": float(problem.initial_poses[index, 0]),
                "initial_y_m": float(problem.initial_poses[index, 1]),
                "initial_heading_deg": float(np.degrees(problem.initial_poses[index, 2])),
                "estimated_x_m": float(optimized[index, 0]),
                "estimated_y_m": float(optimized[index, 1]),
                "estimated_z_m": float(elevations[index]),
                "estimated_heading_deg": float(np.degrees(optimized[index, 2])),
                "pose_covariance_xyyaw": pose_covariances[index].tolist(),
                "truth_x_m": float(odometry[truth_i, 0]),
                "truth_y_m": float(odometry[truth_i, 1]),
                "truth_z_m": float(odometry[truth_i, 2]),
                "xy_error_m": xy_error,
                "z_error_m": float(abs(elevations[index] - odometry[truth_i, 2])),
                "heading_error_deg": heading_error_deg,
            }
        )
    output_json = config.results_path / "moga_all_sites.json"
    output_csv = config.results_path / "moga_all_sites.csv"
    payload = {
        "settings": {
            "resolution": args.resolution,
            "heading_sigma_deg": args.heading_sigma_deg,
            "odometry_position_usage": "evaluation_only",
            "odometry_heading_usage": "heading_sensor_constraint",
            "maximum_function_evaluations": args.max_evaluations,
            "relative_tolerance": args.relative_tolerance,
            "ransac_results": str(ransac_path.resolve()),
        },
        "optimization": {
            key: value
            for key, value in solution.items()
            if key not in {"poses", "landmarks_xy_m", "pose_covariances", "posterior_covariance"}
        },
        "landmark_global_indices": problem.landmark_global_indices.tolist(),
        "optimized_landmarks_xy_m": np.asarray(solution["landmarks_xy_m"]).tolist(),
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
        plot_solution(
            plot_path, problem.site_numbers, odometry[:, :2], problem.initial_poses, optimized
        )
    print("MOGA multi-frame refinement")
    print("---------------------------")
    print(f"Success:       {solution['success']}")
    print(f"Message:       {solution['message']}")
    print(f"Observations:  {len(problem.observations)}")
    print(f"Landmarks:     {problem.landmark_count}")
    print(f"Feature poses: {problem.pose_count}/{len(site_numbers)}")
    print(f"Unavailable:   {len(site_numbers) - problem.pose_count}")
    print(f"Initial cost:  {solution['initial_cost']:.6f}")
    print(f"Final cost:    {solution['final_cost']:.6f}")
    errors = [site["xy_error_m"] for site in output_sites if site["xy_error_m"] is not None]
    print(f"Median XY err: {np.median(errors):.3f} m (available feature poses only)")
    print(f"JSON:          {output_json}")
    print(f"CSV:           {output_csv}")
    if not args.no_plot:
        print(f"Plot:          {plot_path}")


if __name__ == "__main__":
    main()

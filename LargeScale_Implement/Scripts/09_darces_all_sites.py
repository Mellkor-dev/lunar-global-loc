#!/usr/bin/env python3
"""Run repaired DARCES independently on every captured local-map site."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching.darces import run_darces
from pipeline_config import load_pipeline_config


RESULT_DIRECTORY = PROJECT_ROOT / "results"
JSON_PATH = RESULT_DIRECTORY / "darces_all_sites.json"
CSV_PATH = RESULT_DIRECTORY / "darces_all_sites.csv"


# ALL-SITES EDIT 1: One reproducible CLI controls every independent site run.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000_000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--heading-tolerance", type=float, default=5.0)
    parser.add_argument("--minimum-cluster-size", type=int, default=2)
    parser.add_argument("--cluster-position-radius", type=float,default = 50.0)
    parser.add_argument("--top-hypotheses",type=int, default = 5)
    parser.add_argument("--consensus-radius",type=float,default=15.0)
    parser.add_argument("--minimum-consensus-features", type=int, default=4)
    return parser.parse_args()


def _heading_deg(odometry_pose: np.ndarray) -> float:
    rotation = Rotation.from_quat(odometry_pose[3:]).as_matrix()
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def _angle_error_deg(first: float, second: float) -> float:
    return float(abs((first - second + 180.0) % 360.0 - 180.0))


# ALL-SITES EDIT 2: Catalogue loading validates shared physical feature types.
def _load_features(
    path: Path,
    expected_kind: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as data:
        kind = str(data["feature_kind"])

        xyz = np.column_stack(
            (
                data["x_m"],
                data["y_m"],
                data["z_m"],
            )
        ).astype(np.float64)

        covariance = (
            np.asarray(
                data["covariance"],
                dtype=np.float64,
            )
            if "covariance" in data.files
            else None
        )

    if kind != expected_kind:
        raise ValueError(
            f"{path.name} contains {kind}, expected {expected_kind}"
        )

    if not np.isfinite(xyz).all():
        raise ValueError(
            f"{path.name} contains invalid feature coordinates"
        )

    if covariance is not None:
        expected_shape = (len(xyz), 3, 3)

        if covariance.shape != expected_shape:
            raise ValueError(
                f"{path.name}: covariance shape "
                f"{covariance.shape}, expected {expected_shape}"
            )

    return xyz, covariance


# ALL-SITES EDIT 3: Each site is matched without absolute odometry position.
def run_site(
    site_number: int,
    *,
    config,
    global_features_xyz: np.ndarray,
    global_dem: np.ndarray,
    global_x_centers_m: np.ndarray,
    global_y_centers_m: np.ndarray,
    distance_tolerance_m: float,
    z_residual_tolerance_m: float,
    args: argparse.Namespace,    
    global_covariances: np.ndarray | None,
    covariance_sigma_multiplier: float = 2.0,
    use_feature_consensus: bool = False,
) -> dict[str, object]:
    feature_path = (
        config.local_features_path
        / f"local_craters_site_{site_number:02d}.npz"
    )
    grid_path = (
        config.gridded_maps_path / f"grid_site_{site_number:02d}.npz"
    )
    odometry_path = (
        PROJECT_ROOT
        / "sim"
        / "5m_px"
        / "odom_scans"
        / f"odom_site_{site_number:02d}.npy"
    )

    local_features_xyz, local_feature_covariances = _load_features(
        feature_path,
        config.features.kind,
    )
    odometry_pose = np.asarray(
        np.load(odometry_path, allow_pickle=False),
        dtype=np.float64,
    )
    heading_measurement_deg = _heading_deg(odometry_pose)
    base_result: dict[str, object] = {
        "site": site_number,
        "local_feature_count": len(local_features_xyz),
        "heading_measurement_deg": heading_measurement_deg,
        "status": "pending",
    }
    if len(local_features_xyz) < 3:
        base_result["status"] = "skipped_insufficient_features"
        return base_result

    with np.load(grid_path, allow_pickle=False) as data:
        local_grid = np.asarray(data["elevation"], dtype=np.float64)
        local_x_centers_m = np.asarray(
            data["x_centers_m"],
            dtype=np.float64,
        )
        local_y_centers_m = np.asarray(
            data["y_centers_m"],
            dtype=np.float64,
        )

    start = time.perf_counter()
    result = run_darces(
        local_features_xyz=local_features_xyz,
        global_features_xyz=global_features_xyz,
        local_grid=local_grid,
        global_dem=global_dem,
        local_x_centers_m=local_x_centers_m,
        local_y_centers_m=local_y_centers_m,
        global_x_centers_m=global_x_centers_m,
        global_y_centers_m=global_y_centers_m,
        n_trials=100000,
        distance_tolerance_m=distance_tolerance_m,
        z_residual_tolerance_m=z_residual_tolerance_m,
        heading_measurement_deg=heading_measurement_deg,
        heading_tolerance_deg=args.heading_tolerance,
        cluster_position_radius_m=args.cluster_position_radius,         
        top_hypothesis_count =  args.top_hypotheses,
        cluster_heading_radius_deg=5.0,
        minimum_cluster_size=args.minimum_cluster_size,
        seed=args.seed + site_number,
        consensus_xy_tolerance_m=args.consensus_radius,
        minimum_consensus_features=args.minimum_consensus_features,
        local_covariances=local_feature_covariances,
        global_covariances=global_covariances,
        covariance_sigma_multiplier=covariance_sigma_multiplier,
        use_feature_consensus=use_feature_consensus,
    )
    elapsed_seconds = time.perf_counter() - start
    base_result["runtime_s"] = elapsed_seconds
    if result is None:
        base_result["status"] = "no_solution"
        return base_result

    # ALL-SITES EDIT 4: Odometry XYZ enters only after DARCES returns.
    estimated_xy = np.asarray(result["t"], dtype=np.float64)
    estimated_z = float(result["tz"])
    base_result.update(
        {
            "status": "solution",
            "estimated_x_m": float(estimated_xy[0]),
            "estimated_y_m": float(estimated_xy[1]),
            "estimated_z_m": estimated_z,
            "estimated_heading_deg": float(result["heading_deg"]),
            "fitness": float(result["fitness"]),
            "overlap": float(result["overlap"]),
            "cluster_size": int(result["cluster_size"]),
            "evaluated_hypothesis_count": int(
                result["evaluated_hypothesis_count"]
            ),
            "truth_x_m": float(odometry_pose[0]),
            "truth_y_m": float(odometry_pose[1]),
            "truth_z_m": float(odometry_pose[2]),
            "xy_error_m": float(
                np.linalg.norm(estimated_xy - odometry_pose[:2])
            ),
            "z_error_m": float(abs(estimated_z - odometry_pose[2])),
            "heading_error_deg": _angle_error_deg(
                float(result["heading_deg"]),
                heading_measurement_deg,
            ),
            "consensus_count": int(result["consensus_count"]),
            "consensus_xy_rmse_m": float(
                result["consensus_xy_rmse_m"]
            ),
            "consensus_z_rmse_m": float(
                result["consensus_z_rmse_m"]
            ),
        }
    )
    return base_result


def main() -> None:
    args = parse_arguments()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.minimum_cluster_size <= 0:
        raise ValueError("--minimum-cluster-size must be positive")

    config = load_pipeline_config()
    global_features_xyz, _ = _load_features(
        config.global_features_path,
        config.features.kind,
    )
    GLOBAL_FEATURE_COVARIANCE_M2 = np.array(
        [
            [110.50479094, 0.0,          0.0],
            [0.0,          110.50479094, 0.0],
            [0.0,          0.0,          1.70066948],
        ],
        dtype=np.float64,
    )
    global_covariances = np.repeat(
        GLOBAL_FEATURE_COVARIANCE_M2[None, :, :],
        len(global_features_xyz),
        axis=0,
    )
    global_dem = np.asarray(
        np.load(config.orbital_dem_path, allow_pickle=False),
        dtype=np.float64,
    )
    global_x_centers_m, global_y_centers_m = (
        config.orbital_raster.coordinates()
    )
    with config.feature_uncertainty_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        uncertainty = json.load(stream)
    distance_tolerance_m = 3.0 * float(uncertainty["sigma_xy_m"])
    z_residual_tolerance_m = 3.0 * float(uncertainty["sigma_z_m"])

    # ALL-SITES EDIT 5: Available grid files define the tested site set.
    grid_paths = sorted(config.gridded_maps_path.glob("grid_site_*.npz"))
    site_numbers = [
        int(path.stem.rsplit("_", 1)[1])
        for path in grid_paths
    ]
    if not site_numbers:
        raise FileNotFoundError("No gridded local sites were found")

    print("DARCES all-sites evaluation")
    print("---------------------------")
    print(f"Sites:                {site_numbers}")
    print(f"Trials per site:      {args.trials}")
    print(f"Global features:      {len(global_features_xyz)}")
    print(f"Distance tolerance:   {distance_tolerance_m:.3f} m")
    print(f"Z residual tolerance: {z_residual_tolerance_m:.3f} m")
    print()

    results = []
    for site_number in site_numbers:
        print(f"=== Site {site_number:02d} ===")
        result = run_site(
            site_number,
            config=config,
            global_features_xyz=global_features_xyz,
            global_dem=global_dem,
            global_x_centers_m=global_x_centers_m,
            global_y_centers_m=global_y_centers_m,
            distance_tolerance_m=distance_tolerance_m,
            z_residual_tolerance_m=z_residual_tolerance_m,
            args=args,            
            global_covariances=global_covariances,
            covariance_sigma_multiplier=2.0,
            use_feature_consensus=False,
        )
        results.append(result)
        print(
            f"status={result['status']}, "
            f"features={result['local_feature_count']}, "
            f"runtime={float(result.get('runtime_s', 0.0)):.2f}s"
        )
        if result["status"] == "solution":
            print(
                f"xy_error={result['xy_error_m']:.3f}m, "
                f"z_error={result['z_error_m']:.3f}m, "
                f"heading_error={result['heading_error_deg']:.3f}deg"
                f"Hypothesis cap:       {args.trials}"
            )
        print()

    # ALL-SITES EDIT 6: Machine-readable JSON and flat CSV share one result.
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "trials_per_site": args.trials,
            "seed": args.seed,
            "heading_tolerance_deg": args.heading_tolerance,
            "minimum_cluster_size": args.minimum_cluster_size,
            "distance_tolerance_m": distance_tolerance_m,
            "z_residual_tolerance_m": z_residual_tolerance_m,
        },
        "sites": results,
    }
    with JSON_PATH.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")

    fieldnames = sorted({key for result in results for key in result})
    with CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    statuses = {
        status: sum(result["status"] == status for result in results)
        for status in sorted({str(result["status"]) for result in results})
    }
    print("Summary")
    print("-------")
    for status, count in statuses.items():
        print(f"{status}: {count}")
    print(f"JSON: {JSON_PATH}")
    print(f"CSV:  {CSV_PATH}")


if __name__ == "__main__":
    main()

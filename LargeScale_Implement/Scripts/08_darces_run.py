#!/usr/bin/env python3
"""Run the repaired DARCES implementation on local scan Site 08 only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching.darces import run_darces
from pipeline_config import load_pipeline_config


SITE_NUMBER = 8
RESULT_DIRECTORY = PROJECT_ROOT / "results"
RESULT_PATH = RESULT_DIRECTORY / "darces_site_08.json"


# RUNNER EDIT 1: All experimental controls are explicit CLI parameters.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--heading-tolerance", type=float, default=10.0)
    parser.add_argument("--minimum-cluster-size", type=int, default=2)
    return parser.parse_args()


def _load_uncertainty(path: Path) -> tuple[float, float]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return float(data["sigma_xy_m"]), float(data["sigma_z_m"])


def main() -> None:
    args = parse_arguments()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.minimum_cluster_size <= 0:
        raise ValueError("--minimum-cluster-size must be positive")

    config = load_pipeline_config()
    grid_path = (
        config.gridded_maps_path / f"grid_site_{SITE_NUMBER:02d}.npz"
    )
    feature_path = (
        config.local_features_path
        / f"local_craters_site_{SITE_NUMBER:02d}.npz"
    )
    odometry_path = (
        PROJECT_ROOT
        / "sim"
        / "odom_scans"
        / f"odom_site_{SITE_NUMBER:02d}.npy"
    )

    # RUNNER EDIT 2: Physical XYZ feature coordinates are loaded directly;
    # raster indices are no longer reconstructed inside DARCES.
    with np.load(feature_path, allow_pickle=False) as data:
        local_features_xyz = np.column_stack(
            (data["x_m"], data["y_m"], data["z_m"])
        ).astype(np.float64)
        local_feature_kind = str(data["feature_kind"])

    with np.load(config.global_features_path, allow_pickle=False) as data:
        global_features_xyz = np.column_stack(
            (data["x_m"], data["y_m"], data["z_m"])
        ).astype(np.float64)
        global_feature_kind = str(data["feature_kind"])

    # RUNNER EDIT 3: North-up grids and their true coordinate vectors are
    # passed unchanged; the old flip-and-origin compatibility shim is removed.
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
        grid_resolution_m = float(data["resolution_m"])

    global_dem = np.asarray(
        np.load(config.orbital_dem_path, allow_pickle=False),
        dtype=np.float64,
    )
    global_x_centers_m, global_y_centers_m = (
        config.orbital_raster.coordinates()
    )
    odometry_pose = np.asarray(
        np.load(odometry_path, allow_pickle=False),
        dtype=np.float64,
    )

    if local_feature_kind != config.features.kind:
        raise ValueError("Local feature kind differs from configuration")
    if global_feature_kind != local_feature_kind:
        raise ValueError("Local and global feature kinds do not match")
    if len(local_features_xyz) < 3:
        raise RuntimeError(
            f"Site {SITE_NUMBER:02d} has only "
            f"{len(local_features_xyz)} local features"
        )
    if not np.isclose(
        grid_resolution_m,
        config.orbital_raster.resolution_m,
    ):
        raise ValueError("Local and orbital grid resolutions differ")

    # RUNNER EDIT 4: Odometry contributes heading only. Absolute X/Y/Z are
    # withheld from matching and used strictly in the evaluation block.
    odometry_rotation = Rotation.from_quat(odometry_pose[3:]).as_matrix()
    heading_measurement_deg = float(
        np.degrees(
            np.arctan2(
                odometry_rotation[1, 0],
                odometry_rotation[0, 0],
            )
        )
    )

    # RUNNER EDIT 5: Screening gates are derived from the Phase 1 uncertainty
    # report instead of the obsolete 0.5 m and 0.3 m constants.
    sigma_xy_m, sigma_z_m = _load_uncertainty(
        config.feature_uncertainty_path
    )
    distance_tolerance_m = 3.0 * sigma_xy_m
    z_residual_tolerance_m = 3.0 * sigma_z_m

    print("DARCES single-site input")
    print("------------------------")
    print(f"Site:                 {SITE_NUMBER:02d}")
    print(f"Local features:       {len(local_features_xyz)}")
    print(f"Global features:      {len(global_features_xyz)}")
    print(f"Grid resolution:      {grid_resolution_m:g} m")
    print(f"Heading measurement:  {heading_measurement_deg:.3f} deg")
    print(f"Distance tolerance:   {distance_tolerance_m:.3f} m")
    print(f"Z residual tolerance: {z_residual_tolerance_m:.3f} m")
    print(f"Trials:               {args.trials}")
    print()

    # RUNNER EDIT 6: The runner calls the coordinate-safe 2.5-D DARCES API.
    result = run_darces(
        local_features_xyz=local_features_xyz,
        global_features_xyz=global_features_xyz,
        local_grid=local_grid,
        global_dem=global_dem,
        local_x_centers_m=local_x_centers_m,
        local_y_centers_m=local_y_centers_m,
        global_x_centers_m=global_x_centers_m,
        global_y_centers_m=global_y_centers_m,
        n_trials=args.trials,
        distance_tolerance_m=distance_tolerance_m,
        z_residual_tolerance_m=z_residual_tolerance_m,
        heading_measurement_deg=heading_measurement_deg,
        heading_tolerance_deg=args.heading_tolerance,
        cluster_position_radius_m=distance_tolerance_m,
        cluster_heading_radius_deg=5.0,
        minimum_cluster_size=args.minimum_cluster_size,
        seed=args.seed,
    )

    # RUNNER EDIT 7: Matching inputs and evaluation-only truth are clearly
    # separated in the saved result.
    payload: dict[str, object] = {
        "site": SITE_NUMBER,
        "status": "no_solution" if result is None else "solution",
        "matching_inputs": {
            "trials": args.trials,
            "seed": args.seed,
            "local_feature_count": len(local_features_xyz),
            "global_feature_count": len(global_features_xyz),
            "heading_measurement_deg": heading_measurement_deg,
            "distance_tolerance_m": distance_tolerance_m,
            "z_residual_tolerance_m": z_residual_tolerance_m,
        },
    }
    if result is not None:
        estimated_xy = np.asarray(result["t"], dtype=np.float64)
        truth_xy = odometry_pose[:2]
        estimated_z = float(result["tz"])
        truth_z = float(odometry_pose[2])
        payload["estimate"] = {
            "xy_m": estimated_xy.tolist(),
            "z_m": estimated_z,
            "heading_deg": float(result["heading_deg"]),
            "fitness": float(result["fitness"]),
            "overlap": float(result["overlap"]),
            "cluster_size": int(result["cluster_size"]),
            "evaluated_hypothesis_count": int(
                result["evaluated_hypothesis_count"]
            ),
            "local_feature_indices": np.asarray(
                result["local_idx"]
            ).tolist(),
            "global_feature_indices": np.asarray(
                result["global_idx"]
            ).tolist(),
        }
        payload["evaluation_only"] = {
            "truth_xyz_m": odometry_pose[:3].tolist(),
            "xy_error_m": float(np.linalg.norm(estimated_xy - truth_xy)),
            "z_error_m": float(abs(estimated_z - truth_z)),
            "heading_error_deg": float(
                abs(
                    (
                        result["heading_deg"]
                        - heading_measurement_deg
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                )
            ),
        }

    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")

    print()
    print("Evaluation")
    print("----------")
    if result is None:
        print("No DARCES solution was returned.")
    else:
        evaluation = payload["evaluation_only"]
        print(f"Estimated XYZ: {payload['estimate']['xy_m'] + [payload['estimate']['z_m']]}")
        print(f"Truth XYZ:     {evaluation['truth_xyz_m']} (evaluation only)")
        print(f"XY error:      {evaluation['xy_error_m']:.3f} m")
        print(f"Z error:       {evaluation['z_error_m']:.3f} m")
        print(f"Heading error: {evaluation['heading_error_deg']:.3f} deg")
    print(f"Result:        {RESULT_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run repaired DARCES independently on every captured local-map site."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matching.darces import (
    DEFAULT_CLUSTER_HEADING_RADIUS_DEG,
    DEFAULT_CLUSTER_POSITION_RADIUS_M,
    DEFAULT_CONSENSUS_XY_TOLERANCE_M,
    DEFAULT_CONTROL_RMS_TOLERANCE_M,
    DEFAULT_COVARIANCE_SIGMA_MULTIPLIER,
    DEFAULT_HEADING_TOLERANCE_DEG,
    DEFAULT_MAXIMUM_TERRAIN_MAE_M,
    DEFAULT_MINIMUM_CLUSTER_SIZE,
    DEFAULT_MINIMUM_CONSENSUS_FEATURES,
    DEFAULT_MINIMUM_OVERLAP,
    DEFAULT_MINIMUM_TRIANGLE_ANGLE_DEG,
    DEFAULT_SIDE_RATIO_TOLERANCE,
    DEFAULT_TOP_HYPOTHESIS_COUNT,
    DEFAULT_TRIALS_PER_SITE,
    decimate_reference_points_xy,
    run_darces,
)
from pipeline_config import add_resolution_argument, load_resolution_config
from site_selection import selected_sites_for_config


# ALL-SITES EDIT 1: One reproducible CLI controls every independent site run.
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_resolution_argument(parser)
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS_PER_SITE,
        help="Maximum accepted control hypotheses per site (default: 100000)",
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--heading-tolerance",
        type=float,
        default=DEFAULT_HEADING_TOLERANCE_DEG,
    )
    parser.add_argument(
        "--minimum-cluster-size",
        type=int,
        default=DEFAULT_MINIMUM_CLUSTER_SIZE,
    )
    parser.add_argument(
        "--cluster-position-radius",
        type=float,
        default=DEFAULT_CLUSTER_POSITION_RADIUS_M,
    )
    parser.add_argument(
        "--top-hypotheses",
        type=int,
        default=DEFAULT_TOP_HYPOTHESIS_COUNT,
    )
    parser.add_argument(
        "--control-rms-tolerance",
        type=float,
        default=DEFAULT_CONTROL_RMS_TOLERANCE_M,
        help=(
            "Maximum RMS XY alignment error of a control triangle in metres "
            "(default: 10)"
        ),
    )
    parser.add_argument(
        "--side-ratio-tolerance",
        type=float,
        default=DEFAULT_SIDE_RATIO_TOLERANCE,
    )
    parser.add_argument(
        "--minimum-triangle-angle",
        type=float,
        default=DEFAULT_MINIMUM_TRIANGLE_ANGLE_DEG,
    )
    parser.add_argument(
        "--covariance-sigma-multiplier",
        type=float,
        default=DEFAULT_COVARIANCE_SIGMA_MULTIPLIER,
    )
    parser.add_argument("--distance-tolerance-sigma", type=float, default=3.0)
    parser.add_argument("--z-residual-tolerance-sigma", type=float, default=3.0)
    parser.add_argument(
        "--minimum-overlap",
        type=float,
        default=DEFAULT_MINIMUM_OVERLAP,
    )
    parser.add_argument(
        "--maximum-terrain-mae",
        type=float,
        default=DEFAULT_MAXIMUM_TERRAIN_MAE_M,
        help="Maximum mean absolute LiDAR-to-DEM elevation residual in metres",
    )
    parser.add_argument(
        "--cluster-heading-radius",
        type=float,
        default=DEFAULT_CLUSTER_HEADING_RADIUS_DEG,
    )
    parser.add_argument("--reference-spacing-factor", type=float, default=0.50)
    parser.add_argument(
        "--consensus-radius",
        type=float,
        default=DEFAULT_CONSENSUS_XY_TOLERANCE_M,
    )
    parser.add_argument(
        "--minimum-consensus-features",
        type=int,
        default=DEFAULT_MINIMUM_CONSENSUS_FEATURES,
    )
    parser.add_argument(
        "--use-feature-consensus",
        action="store_true",
        help=(
            "Require each control hypothesis to pass feature-consensus "
            "screening (disabled by default)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Independent site processes. Results remain deterministic because "
            "each site uses seed + site number"
        ),
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        help=(
            "Checkpoint root (default: "
            "results/<resolution>_px/darces_checkpoints)"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore compatible per-site checkpoints and recompute every site",
    )
    parser.add_argument(
        "--sites",
        type=int,
        nargs="+",
        help="Run only these site numbers (default: every gridded site)",
    )
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
    covariance_sigma_multiplier: float,
    use_feature_consensus: bool = False,
) -> dict[str, object]:
    feature_path = (
        config.local_features_path
        / f"local_craters_site_{site_number:02d}.npz"
    )
    grid_path = (
        config.gridded_maps_path / f"grid_site_{site_number:02d}.npz"
    )
    leveled_path = (
        config.leveled_maps_path / f"leveled_site_{site_number:02d}.npy"
    )
    odometry_path = (
        config.captures_path
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

    reference_spacing_m = (
        config.orbital_raster.resolution_m * args.reference_spacing_factor
    )
    reference_points_xyz = decimate_reference_points_xy(
        np.asarray(np.load(leveled_path, allow_pickle=False), dtype=np.float64),
        reference_spacing_m,
    )
    base_result.update(
        {
            "fitness_reference": "raw_lidar_xy_decimated",
            "fitness_reference_spacing_m": reference_spacing_m,
            "fitness_reference_point_count": len(reference_points_xyz),
        }
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
        n_trials=args.trials,
        distance_tolerance_m=distance_tolerance_m,
        z_residual_tolerance_m=z_residual_tolerance_m,
        heading_measurement_deg=heading_measurement_deg,
        heading_tolerance_deg=args.heading_tolerance,
        cluster_position_radius_m=args.cluster_position_radius,         
        top_hypothesis_count =  args.top_hypotheses,
        cluster_heading_radius_deg=args.cluster_heading_radius,
        minimum_cluster_size=args.minimum_cluster_size,
        control_rms_tolerance_m=args.control_rms_tolerance,
        side_ratio_tolerance=args.side_ratio_tolerance,
        minimum_triangle_angle_deg=args.minimum_triangle_angle,
        minimum_overlap=args.minimum_overlap,
        maximum_terrain_mae_m=args.maximum_terrain_mae,
        seed=args.seed + site_number,
        consensus_xy_tolerance_m=args.consensus_radius,
        minimum_consensus_features=args.minimum_consensus_features,
        local_covariances=local_feature_covariances,
        global_covariances=global_covariances,
        covariance_sigma_multiplier=covariance_sigma_multiplier,
        use_feature_consensus=use_feature_consensus,
        reference_points_xyz=reference_points_xyz,
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
            "correspondence_count": int(result["correspondence_count"]),
            "correspondence_local_indices": np.asarray(
                result["correspondence_local_idx"], dtype=np.int64
            ).tolist(),
            "correspondence_global_indices": np.asarray(
                result["correspondence_global_idx"], dtype=np.int64
            ).tolist(),
            "correspondence_xy_rmse_m": float(
                result["correspondence_xy_rmse_m"]
            ),
            "correspondence_z_rmse_m": float(
                result["correspondence_z_rmse_m"]
            ),
        }
    )
    return base_result


_WORKER_CONTEXT: dict[str, object] | None = None


def _initialize_worker(context: dict[str, object]) -> None:
    """Install immutable shared inputs once in each worker process."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _run_site_from_context(
    site_number: int,
    context: dict[str, object],
) -> dict[str, object]:
    return run_site(
        site_number,
        config=context["config"],
        global_features_xyz=context["global_features_xyz"],
        global_dem=context["global_dem"],
        global_x_centers_m=context["global_x_centers_m"],
        global_y_centers_m=context["global_y_centers_m"],
        distance_tolerance_m=context["distance_tolerance_m"],
        z_residual_tolerance_m=context["z_residual_tolerance_m"],
        args=context["args"],
        global_covariances=context["global_covariances"],
        covariance_sigma_multiplier=context["covariance_sigma_multiplier"],
        use_feature_consensus=context["use_feature_consensus"],
    )


def _run_site_worker(site_number: int) -> dict[str, object]:
    """Run one site and keep verbose DARCES output in a site-specific log."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("DARCES worker context was not initialized")
    log_directory = Path(_WORKER_CONTEXT["site_log_directory"])
    log_path = log_directory / f"site_{site_number:04d}.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            print(f"=== Site {site_number:02d} ===", flush=True)
            try:
                return _run_site_from_context(site_number, _WORKER_CONTEXT)
            except BaseException:
                traceback.print_exc()
                raise


def _settings_payload(
    args: argparse.Namespace,
    *,
    distance_tolerance_m: float,
    z_residual_tolerance_m: float,
    global_covariances: np.ndarray,
) -> dict[str, object]:
    """Return numerical settings that define a reproducible DARCES run."""
    return {
        "trials_per_site": args.trials,
        "seed": args.seed,
        "heading_tolerance_deg": args.heading_tolerance,
        "minimum_cluster_size": args.minimum_cluster_size,
        "cluster_position_radius_m": args.cluster_position_radius,
        "top_hypothesis_count": args.top_hypotheses,
        "control_rms_tolerance_m": args.control_rms_tolerance,
        "effective_control_rms_tolerance_m": (
            args.control_rms_tolerance
            if args.control_rms_tolerance is not None
            else distance_tolerance_m
        ),
        "side_ratio_tolerance": args.side_ratio_tolerance,
        "minimum_triangle_angle_deg": args.minimum_triangle_angle,
        "minimum_overlap": args.minimum_overlap,
        "maximum_terrain_mae_m": args.maximum_terrain_mae,
        "cluster_heading_radius_deg": args.cluster_heading_radius,
        "distance_tolerance_sigma_multiplier": args.distance_tolerance_sigma,
        "z_residual_tolerance_sigma_multiplier": args.z_residual_tolerance_sigma,
        "use_feature_consensus": args.use_feature_consensus,
        "consensus_xy_tolerance_m": args.consensus_radius,
        "minimum_consensus_features": args.minimum_consensus_features,
        "distance_tolerance_m": distance_tolerance_m,
        "z_residual_tolerance_m": z_residual_tolerance_m,
        "covariance_sigma_multiplier": args.covariance_sigma_multiplier,
        "global_covariance_m2": global_covariances[0].tolist(),
        "fitness_reference": "raw_lidar_xy_decimated",
        "fitness_reference_spacing_factor": args.reference_spacing_factor,
    }


def _input_token(path: Path) -> dict[str, object]:
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _checkpoint_signature(
    *,
    config,
    settings: dict[str, object],
    site_numbers: list[int],
) -> tuple[str, dict[str, object]]:
    """Fingerprint numerical settings and every per-site input file."""
    input_paths = [
        config.config_path,
        Path(run_darces.__code__.co_filename),
        config.global_features_path,
        config.orbital_dem_path,
        config.feature_uncertainty_path,
    ]
    for site_number in site_numbers:
        input_paths.extend(
            (
                config.local_features_path
                / f"local_craters_site_{site_number:02d}.npz",
                config.gridded_maps_path / f"grid_site_{site_number:02d}.npz",
                config.leveled_maps_path / f"leveled_site_{site_number:02d}.npy",
                config.captures_path
                / "odom_scans"
                / f"odom_site_{site_number:02d}.npy",
            )
        )
    manifest = {
        "settings": settings,
        "inputs": [_input_token(path) for path in input_paths],
    }
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), manifest


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = sorted({key for result in results for key in result})
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _hypothesis_counts_from_log(
    path: Path,
) -> tuple[int | None, int | None]:
    """Return ordered and fully screened hypothesis counts from a site log."""
    if not path.is_file():
        return None, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    generated_match = re.search(
        r"Generated\s+([0-9]+)\s+ordered hypotheses",
        text,
    )
    screened_match = re.search(r"Passed all screening:\s*([0-9]+)", text)
    return (
        int(generated_match.group(1)) if generated_match else None,
        int(screened_match.group(1)) if screened_match else None,
    )


def _write_runtime_ledger(
    path: Path,
    *,
    resolution: str,
    settings: dict[str, object],
    signature: str,
    completed: dict[int, dict[str, object]],
    site_log_directory: Path,
) -> None:
    """Atomically refresh the completed-site runtime/hypothesis table."""
    rows: list[dict[str, object]] = []
    hypothesis_cap = settings.get("trials_per_site")
    for site_number, result in sorted(completed.items()):
        generated, screened = _hypothesis_counts_from_log(
            site_log_directory / f"site_{site_number:04d}.log"
        )
        runtime = result.get("runtime_s")
        rate = None
        if generated is not None and runtime is not None and float(runtime) > 0.0:
            rate = generated / float(runtime)
        rows.append(
            {
                "resolution": resolution,
                "site": site_number,
                "status": result.get("status"),
                "local_feature_count": result.get("local_feature_count"),
                "hypothesis_cap": hypothesis_cap,
                "generated_hypothesis_count": generated,
                "screened_hypothesis_count": screened,
                "evaluated_hypothesis_count": result.get(
                    "evaluated_hypothesis_count"
                ),
                "runtime_s": runtime,
                "generated_hypotheses_per_s": rate,
                "cluster_size": result.get("cluster_size"),
                "correspondence_count": result.get("correspondence_count"),
                "xy_error_m": result.get("xy_error_m"),
                "heading_error_deg": result.get("heading_error_deg"),
                "checkpoint_signature": signature,
            }
        )
    _atomic_write_csv(path, rows)


def _load_checkpoint(path: Path, signature: str) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("signature") != signature:
            return None
        result = payload["result"]
        if int(result["site"]) <= 0:
            return None
        return result
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_checkpoint(
    path: Path,
    signature: str,
    result: dict[str, object],
) -> None:
    _atomic_write_json(path, {"signature": signature, "result": result})


def _print_site_result(
    result: dict[str, object],
    *,
    completed_count: int,
    total_count: int,
    trials: int,
    resumed: bool,
    log_path: Path | None = None,
) -> None:
    site_number = int(result["site"])
    source = "checkpoint" if resumed else "worker"
    print(
        f"[{completed_count:3d}/{total_count}] Site {site_number:02d}: "
        f"status={result['status']}, "
        f"features={result['local_feature_count']}, "
        f"runtime={float(result.get('runtime_s', 0.0)):.2f}s "
        f"({source})",
        flush=True,
    )
    if result["status"] == "solution":
        print(
            f"  xy_error={result['xy_error_m']:.3f}m, "
            f"z_error={result['z_error_m']:.3f}m, "
            f"heading_error={result['heading_error_deg']:.3f}deg, "
            f"hypothesis_cap={trials}",
            flush=True,
        )
    if log_path is not None:
        print(f"  detail_log={log_path}", flush=True)


def main() -> None:
    args = parse_arguments()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.minimum_cluster_size <= 0:
        raise ValueError("--minimum-cluster-size must be positive")
    if (
        args.control_rms_tolerance is not None
        and args.control_rms_tolerance <= 0.0
    ):
        raise ValueError("--control-rms-tolerance must be positive")
    if not 0.0 <= args.side_ratio_tolerance < 1.0:
        raise ValueError("--side-ratio-tolerance must lie in [0, 1)")
    if not 0.0 < args.minimum_triangle_angle < 60.0:
        raise ValueError("--minimum-triangle-angle must lie in (0, 60)")
    if args.covariance_sigma_multiplier <= 0.0:
        raise ValueError("--covariance-sigma-multiplier must be positive")
    if args.distance_tolerance_sigma <= 0.0:
        raise ValueError("--distance-tolerance-sigma must be positive")
    if args.z_residual_tolerance_sigma <= 0.0:
        raise ValueError("--z-residual-tolerance-sigma must be positive")
    if not 0.0 < args.minimum_overlap <= 1.0:
        raise ValueError("--minimum-overlap must lie in (0, 1]")
    if (
        args.maximum_terrain_mae is not None
        and args.maximum_terrain_mae <= 0.0
    ):
        raise ValueError("--maximum-terrain-mae must be positive")
    if args.cluster_heading_radius <= 0.0:
        raise ValueError("--cluster-heading-radius must be positive")
    if args.reference_spacing_factor <= 0.0:
        raise ValueError("--reference-spacing-factor must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    config = load_resolution_config(args.resolution)
    result_directory = config.results_path
    json_path = result_directory / "darces_all_sites.json"
    csv_path = result_directory / "darces_all_sites.csv"
    global_features_xyz, catalogue_global_covariances = _load_features(
        config.global_features_path,
        config.features.kind,
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
    if catalogue_global_covariances is not None:
        global_covariances = catalogue_global_covariances
    else:
        global_covariance_m2 = np.asarray(
            uncertainty["covariance_m2"], dtype=np.float64
        )
        global_covariances = np.repeat(
            global_covariance_m2[None, :, :],
            len(global_features_xyz),
            axis=0,
        )
    distance_tolerance_m = (
        args.distance_tolerance_sigma * float(uncertainty["sigma_xy_m"])
    )
    z_residual_tolerance_m = (
        args.z_residual_tolerance_sigma * float(uncertainty["sigma_z_m"])
    )

    # The dataset-level manifest defines one common site set for every resolution.
    grid_paths = config.gridded_maps_path.glob("grid_site_*.npz")
    available_grid_sites = sorted(
        int(path.stem.rsplit("_", 1)[1])
        for path in grid_paths
    )
    if not available_grid_sites:
        raise FileNotFoundError("No gridded local sites were found")
    selected_sites = selected_sites_for_config(config)
    missing_selected = sorted(set(selected_sites).difference(available_grid_sites))
    if missing_selected:
        raise FileNotFoundError(
            f"Selected sites have no local grids: {missing_selected}"
        )
    site_numbers = selected_sites
    if args.sites is not None:
        requested_sites = sorted(set(args.sites))
        if any(site <= 0 for site in requested_sites):
            raise ValueError("--sites values must be positive")
        unavailable = sorted(set(requested_sites).difference(selected_sites))
        if unavailable:
            raise ValueError(
                f"Requested sites are outside the shared site manifest: {unavailable}"
            )
        site_numbers = requested_sites

    print("DARCES all-sites evaluation")
    print("---------------------------")
    print(f"Sites:                {site_numbers}")
    print(f"Trials per site:      {args.trials}")
    print(f"Global features:      {len(global_features_xyz)}")
    print(f"Distance tolerance:   {distance_tolerance_m:.3f} m")
    print(f"Z residual tolerance: {z_residual_tolerance_m:.3f} m")
    print(f"Worker processes:      {args.workers}")
    print()

    settings = _settings_payload(
        args,
        distance_tolerance_m=distance_tolerance_m,
        z_residual_tolerance_m=z_residual_tolerance_m,
        global_covariances=global_covariances,
    )
    signature, checkpoint_manifest = _checkpoint_signature(
        config=config,
        settings=settings,
        site_numbers=site_numbers,
    )
    checkpoint_root = (
        args.checkpoint_directory.resolve()
        if args.checkpoint_directory is not None
        else result_directory / "darces_checkpoints"
    )
    checkpoint_run_directory = checkpoint_root / signature
    site_log_directory = checkpoint_run_directory / "logs"
    site_log_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        checkpoint_run_directory / "manifest.json",
        {
            "signature": signature,
            **checkpoint_manifest,
        },
    )

    completed: dict[int, dict[str, object]] = {}
    if not args.no_resume:
        for site_number in site_numbers:
            checkpoint_path = (
                checkpoint_run_directory / f"site_{site_number:04d}.json"
            )
            result = _load_checkpoint(checkpoint_path, signature)
            if result is not None and int(result["site"]) == site_number:
                completed[site_number] = result

    pending = [site for site in site_numbers if site not in completed]
    print(f"Checkpoint signature: {signature[:16]}")
    print(f"Checkpoint directory: {checkpoint_run_directory}")
    print(f"Resumed sites:        {len(completed)}")
    print(f"Pending sites:        {len(pending)}")
    print()

    runtime_ledger_path = result_directory / "darces_runtime_by_site.csv"
    _write_runtime_ledger(
        runtime_ledger_path,
        resolution=args.resolution,
        settings=settings,
        signature=signature,
        completed=completed,
        site_log_directory=site_log_directory,
    )

    total_count = len(site_numbers)
    for completed_count, site_number in enumerate(sorted(completed), start=1):
        _print_site_result(
            completed[site_number],
            completed_count=completed_count,
            total_count=total_count,
            trials=args.trials,
            resumed=True,
        )

    context: dict[str, object] = {
        "config": config,
        "global_features_xyz": global_features_xyz,
        "global_dem": global_dem,
        "global_x_centers_m": global_x_centers_m,
        "global_y_centers_m": global_y_centers_m,
        "distance_tolerance_m": distance_tolerance_m,
        "z_residual_tolerance_m": z_residual_tolerance_m,
        "args": args,
        "global_covariances": global_covariances,
        "covariance_sigma_multiplier": args.covariance_sigma_multiplier,
        "use_feature_consensus": args.use_feature_consensus,
        "site_log_directory": site_log_directory,
    }

    if args.workers == 1:
        for site_number in pending:
            print(f"=== Site {site_number:02d} ===", flush=True)
            result = _run_site_from_context(site_number, context)
            completed[site_number] = result
            _save_checkpoint(
                checkpoint_run_directory / f"site_{site_number:04d}.json",
                signature,
                result,
            )
            _write_runtime_ledger(
                runtime_ledger_path,
                resolution=args.resolution,
                settings=settings,
                signature=signature,
                completed=completed,
                site_log_directory=site_log_directory,
            )
            _print_site_result(
                result,
                completed_count=len(completed),
                total_count=total_count,
                trials=args.trials,
                resumed=False,
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=(context,),
        ) as executor:
            futures = {
                executor.submit(_run_site_worker, site_number): site_number
                for site_number in pending
            }
            for future in as_completed(futures):
                site_number = futures[future]
                result = future.result()
                if int(result["site"]) != site_number:
                    raise RuntimeError(
                        f"Worker returned Site {result['site']} for Site "
                        f"{site_number}"
                    )
                completed[site_number] = result
                _save_checkpoint(
                    checkpoint_run_directory / f"site_{site_number:04d}.json",
                    signature,
                    result,
                )
                _write_runtime_ledger(
                    runtime_ledger_path,
                    resolution=args.resolution,
                    settings=settings,
                    signature=signature,
                    completed=completed,
                    site_log_directory=site_log_directory,
                )
                _print_site_result(
                    result,
                    completed_count=len(completed),
                    total_count=total_count,
                    trials=args.trials,
                    resumed=False,
                    log_path=(
                        site_log_directory / f"site_{site_number:04d}.log"
                    ),
                )

    results = [completed[site_number] for site_number in site_numbers]

    # ALL-SITES EDIT 6: Machine-readable JSON and flat CSV share one result.
    result_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": settings,
        "execution": {
            "worker_processes": args.workers,
            "checkpoint_signature": signature,
        },
        "sites": results,
    }
    _atomic_write_json(json_path, payload)
    _atomic_write_csv(csv_path, results)
    _write_runtime_ledger(
        runtime_ledger_path,
        resolution=args.resolution,
        settings=settings,
        signature=signature,
        completed=completed,
        site_log_directory=site_log_directory,
    )

    statuses = {
        status: sum(result["status"] == status for result in results)
        for status in sorted({str(result["status"]) for result in results})
    }
    print("Summary")
    print("-------")
    for status, count in statuses.items():
        print(f"{status}: {count}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"Runtime ledger: {runtime_ledger_path}")


if __name__ == "__main__":
    main()

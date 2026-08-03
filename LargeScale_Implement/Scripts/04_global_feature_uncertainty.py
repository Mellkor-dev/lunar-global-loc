#!/usr/bin/env python3
"""Estimate global-feature uncertainty against synchronized truth features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import PipelineConfig, load_pipeline_config


CONFIG = load_pipeline_config()
OUTPUT_DIRECTORY = CONFIG.feature_validation_path
MATCHES_PATH = OUTPUT_DIRECTORY / "downsampling_feature_matches.npz"
SUMMARY_PATH = OUTPUT_DIRECTORY / "downsampling_uncertainty_summary.csv"
PLOT_PATH = CONFIG.plots_path / "downsampling_uncertainty.png"
REQUIRED_FIELDS = {"x_m", "y_m", "z_m", "feature_kind"}


@dataclass(frozen=True)
class GlobalFeatureUncertainty:
    sigma_xy_m: float
    sigma_z_m: float
    sigma_xy_intrinsic_m: float
    sigma_z_intrinsic_m: float
    sigma_xy_downsampling_m: float
    sigma_z_downsampling_m: float

    @property
    def covariance_m2(self) -> np.ndarray:
        return np.diag(
            (self.sigma_xy_m**2, self.sigma_xy_m**2, self.sigma_z_m**2)
        )


def _load_catalogue(
    path: Path,
    expected_kind: str,
) -> tuple[np.ndarray, dict[str, float | str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature catalogue not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_FIELDS.difference(data.files)
        if missing:
            raise ValueError(f"{path.name} is missing fields {sorted(missing)}")
        xyz = np.column_stack((data["x_m"], data["y_m"], data["z_m"]))
        metadata = {
            "feature_kind": str(data["feature_kind"]),
            "detection_distance_m": float(data["detection_distance_m"]),
            "flatness_eps_m": float(data["flatness_eps_m"]),
            "resolution_m": float(data["resolution_m"]),
        }

    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError(f"{path.name} must contain a nonempty N x 3 catalogue")
    if not np.isfinite(xyz).all():
        raise ValueError(f"{path.name} contains invalid coordinates")
    if metadata["feature_kind"] != expected_kind:
        raise ValueError(
            f"{path.name} contains {metadata['feature_kind']} features, "
            f"expected {expected_kind}"
        )
    return xyz, metadata


def _mutual_nearest_matches(
    orbital_xyz: np.ndarray,
    truth_xyz: np.ndarray,
    gate_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-to-one mutual nearest neighbours within an XY gate."""
    truth_tree = cKDTree(truth_xyz[:, :2])
    distance, truth_index = truth_tree.query(orbital_xyz[:, :2], k=1)
    orbital_tree = cKDTree(orbital_xyz[:, :2])
    _, orbital_index_for_truth = orbital_tree.query(truth_xyz[:, :2], k=1)

    orbital_index = np.arange(len(orbital_xyz), dtype=np.int64)
    mutual = orbital_index_for_truth[truth_index] == orbital_index
    accepted = mutual & (distance <= gate_m)
    return orbital_index[accepted], truth_index[accepted], distance[accepted]


def estimate_uncertainty(
    config: PipelineConfig,
) -> tuple[GlobalFeatureUncertainty, dict[str, int | float]]:
    orbital_xyz, orbital_metadata = _load_catalogue(
        config.global_features_path,
        config.features.kind,
    )
    truth_xyz, truth_metadata = _load_catalogue(
        config.truth_features_path,
        config.features.kind,
    )
    for key in ("detection_distance_m", "flatness_eps_m"):
        if not np.isclose(
            float(orbital_metadata[key]),
            float(truth_metadata[key]),
        ):
            raise ValueError(f"Global and truth catalogues disagree on {key}")

    gate_m = config.features.distance_m * config.match_gate_fraction
    orbital_index, truth_index, xy_distance = _mutual_nearest_matches(
        orbital_xyz,
        truth_xyz,
        gate_m,
    )
    if len(orbital_index) < 3:
        raise RuntimeError(
            f"Only {len(orbital_index)} feature matches passed the {gate_m:g} m "
            "mutual-nearest gate"
        )

    matched_orbital = orbital_xyz[orbital_index]
    matched_truth = truth_xyz[truth_index]
    difference = matched_orbital - matched_truth
    sigma_xy_downsampling = float(np.sqrt(np.mean(xy_distance**2)))
    sigma_z_downsampling = float(np.sqrt(np.mean(difference[:, 2] ** 2)))
    uncertainty = GlobalFeatureUncertainty(
        sigma_xy_m=float(
            np.hypot(
                config.intrinsic_horizontal_sigma_m,
                sigma_xy_downsampling,
            )
        ),
        sigma_z_m=float(
            np.hypot(
                config.intrinsic_vertical_sigma_m,
                sigma_z_downsampling,
            )
        ),
        sigma_xy_intrinsic_m=config.intrinsic_horizontal_sigma_m,
        sigma_z_intrinsic_m=config.intrinsic_vertical_sigma_m,
        sigma_xy_downsampling_m=sigma_xy_downsampling,
        sigma_z_downsampling_m=sigma_z_downsampling,
    )
    counts = {
        "orbital_feature_count": len(orbital_xyz),
        "truth_feature_count": len(truth_xyz),
        "mutual_match_count": len(orbital_index),
        "unmatched_or_rejected_orbital_count": (
            len(orbital_xyz) - len(orbital_index)
        ),
        "match_gate_m": gate_m,
    }

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MATCHES_PATH,
        orbital_feature_index=orbital_index,
        truth_feature_index=truth_index,
        orbital_xyz_m=matched_orbital,
        truth_xyz_m=matched_truth,
        dx_m=difference[:, 0],
        dy_m=difference[:, 1],
        dz_m=difference[:, 2],
        xy_distance_m=xy_distance,
    )
    payload = {
        **asdict(uncertainty),
        **counts,
        "feature_kind": config.features.kind,
        "detection_distance_m": config.features.distance_m,
        "flatness_threshold_m": config.features.flatness_threshold_m,
        "orbital_feature_path": str(config.global_features_path),
        "truth_feature_path": str(config.truth_features_path),
        "covariance_m2": uncertainty.covariance_m2.tolist(),
    }
    config.feature_uncertainty_path.parent.mkdir(parents=True, exist_ok=True)
    with config.feature_uncertainty_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")

    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        for key, value in payload.items():
            if not isinstance(value, (list, dict)):
                writer.writerow((key, value))

    return uncertainty, counts


def _save_plot() -> None:
    with np.load(MATCHES_PATH, allow_pickle=False) as data:
        dx = data["dx_m"]
        dy = data["dy_m"]
        dz = data["dz_m"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    components = (
        (dx, "x error [m]"),
        (dy, "y error [m]"),
        (dz, "z error [m]"),
    )

    for axis, (errors, x_label) in zip(axes, components):
        mean = float(np.mean(errors))
        standard_deviation = float(np.std(errors))
        axis.hist(
            errors,
            bins="auto",
            color="tab:blue",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        axis.axvline(
            mean,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label=f"mean={mean:.3f} m",
        )
        axis.set_title(f"σ={standard_deviation:.3f} m")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Matched craters")
        axis.legend(loc="upper right")
        axis.grid(axis="y", alpha=0.2)

    figure.suptitle(
        "Coordinate differences: orbital − matched truth feature",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    config = CONFIG
    uncertainty, counts = estimate_uncertainty(config)
    _save_plot()

    print("Global feature uncertainty")
    print("--------------------------")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"sigma_xy_m: {uncertainty.sigma_xy_m:.6f}")
    print(f"sigma_z_m:  {uncertainty.sigma_z_m:.6f}")
    print("covariance_m2:")
    print(uncertainty.covariance_m2)
    print(f"Report: {config.feature_uncertainty_path}")


if __name__ == "__main__":
    main()

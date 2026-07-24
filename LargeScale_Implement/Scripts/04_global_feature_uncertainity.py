#!/usr/bin/env python3


from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import csv
import json
from typing import List, Tuple, Callable, Optional, Union, Any, Dict, Sequence, Iterable
from dataclasses import asdict, dataclass


DEM_PATH = PROJECT_ROOT / "DEM" / "orbital_dem_5m.npy"
TRUTH_DEM_PATH = PROJECT_ROOT / "DEM" / "truth_dem_1p5m.npy"
PLOT_DIRECTORY = PROJECT_ROOT / "plots"
PLOT_PATH = PLOT_DIRECTORY / "global_features_validated.png"
FEATURE_PATH = PROJECT_ROOT / "DEM" / "global_craters_n12.npz"
VALIDATION_PATH = PROJECT_ROOT / "DEM" / "validation"
SUMMARY_PATH = VALIDATION_PATH / "summary.csv"


DOWNSAMPLED_FEATURE_PATH = PROJECT_ROOT / "DEM" / "global_craters_n12.npz"
TRUTH_FEATURE_PATH = PROJECT_ROOT / "DEM" / "truth_craters_n40.npz"

OUTPUT_DIRECTORY = PROJECT_ROOT / "DEM" / "validation"
MATCHES_PATH = OUTPUT_DIRECTORY / "downsampling_feature_matches.npz"
SUMMARY_PATH = OUTPUT_DIRECTORY / "downsampling_uncertainty_summary.csv"
VALIDATION_DATA_PATH = OUTPUT_DIRECTORY / "global_feature_uncertainty.json"
PLOT_PATH = PROJECT_ROOT / "plots" / "downsampling_uncertainty.png"

REQUIRED_FIELDS = ("row", "column", "x_m", "y_m", "z_m")

INTRINSIC_HORIZONTAL_ACCURACY_M = 10.0  # Haase/DLR DTM metadata
INTRINSIC_VERTICAL_ACCURACY_M = 1.3



LXY_M = 5.0
N_CELLS = 12
FLATNESS_EPS_M = 0.5
TRUTH_DEM_LXY_M = 1.5
D_DETECT = 60.0
MATCH_DISTANCE_GATE_M = D_DETECT / 2.0

SCALE_CONFIG = (
    {"D_DETECT": 60.0,
     "N_CELLS": 12,
     "N_CELLS_TRUTH": 40
    },
    {"D_DETECT": 40.0,
     "N_CELLS": 8,
     "N_CELLS_TRUTH": 27
    },
    {"D_DETECT": 80.0,
     "N_CELLS": 16,
     "N_CELLS_TRUTH": 54
    },
    {"D_DETECT": 25.0,
     "N_CELLS": 5,
     "N_CELLS_TRUTH": 17
    }
)

def normalize_feature_indices(
    features: np.ndarray,
    *,
    dem_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    
    features = np.asarray(features)

    if features.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape (N, 2), got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(f"{name} contains NaN or infinite indices")

    rounded = np.rint(features)

    if not np.allclose(features, rounded):
        raise ValueError(f"{name} contains non-integer raster indices")

    features = rounded.astype(np.int64)

    rows = features[:, 0]
    columns = features[:, 1]

    if (
        np.any(rows < 0)
        or np.any(rows >= dem_shape[0])
        or np.any(columns < 0)
        or np.any(columns >= dem_shape[1])
    ):
        raise IndexError(f"{name} contains indices outside the DEM")

    # Remove accidental duplicate detections.
    features = np.unique(features, axis=0)

    return features

def gloabal_feature_covariance(
    sigma_xy: float,
    sigma_z: float,
) -> np.ndarray:
    return np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2])

@dataclass
class GlobalFeatureUncertainty:
    sigma_xy: float
    sigma_z: float
    sigma_z_intrinsic: float
    sigma_xy_intrinsic: float
    sigma_z_downsample: float
    sigma_xy_downsample: float

def estimate_downsampled_uncertainty(
    sigma_xy: float,
    sigma_z: float,
    downsample_factor: int,
) -> GlobalFeatureUncertainty:
    """
    Estimate the uncertainty of downsampled features based on the original
    uncertainty and the downsampling factor.

    Args:
        sigma_xy: The original uncertainty in the XY plane.
        sigma_z: The original uncertainty in the Z direction.
        downsample_factor: The factor by which the features are downsampled."""
        
    if sigma_xy < 0.0 or sigma_z < 0.0:
        raise ValueError("Intrinsic uncertainties must be non-negative")
    if downsample_factor <= 0:
        raise ValueError("downsample_factor must be positive")

    with np.load(DOWNSAMPLED_FEATURE_PATH) as downsampled_data:
        missing = set(REQUIRED_FIELDS).difference(downsampled_data.files)
        if missing:
            raise ValueError(
                f"{DOWNSAMPLED_FEATURE_PATH.name} is missing {sorted(missing)}"
            )
        downsampled_xyz = np.column_stack(
            [
                downsampled_data["x_m"],
                downsampled_data["y_m"],
                downsampled_data["z_m"],
            ]
        ).astype(np.float64)

    with np.load(TRUTH_FEATURE_PATH) as truth_data:
        missing = set(REQUIRED_FIELDS).difference(truth_data.files)
        if missing:
            raise ValueError(
                f"{TRUTH_FEATURE_PATH.name} is missing {sorted(missing)}"
            )
        truth_xyz = np.column_stack(
            [truth_data["x_m"], truth_data["y_m"], truth_data["z_m"]]
        ).astype(np.float64)

    if len(downsampled_xyz) == 0 or len(truth_xyz) == 0:
        raise ValueError("Both feature catalogues must contain features")
    if not np.isfinite(downsampled_xyz).all() or not np.isfinite(truth_xyz).all():
        raise ValueError("Feature catalogues contain invalid coordinates")

    # Match each crater from the downsampled DEM to the closest truth crater.
    xy_distance, truth_index = cKDTree(truth_xyz[:, :2]).query(
        downsampled_xyz[:, :2],
        k=1,
    )
    
    matched_truth_xyz = truth_xyz[truth_index]
    coordinate_difference = downsampled_xyz - matched_truth_xyz
    keep = xy_distance <= MATCH_DISTANCE_GATE_M
    n_rejected = int((~keep).sum())
    if n_rejected > 0:
        print(f"Rejected {n_rejected}/{len(xy_distance)} matches beyond "
              f"{MATCH_DISTANCE_GATE_M:.1f}m as likely wrong-feature pairings")

    xy_distance_clean = xy_distance[keep]
    dz_clean = coordinate_difference[keep, 2]
    z_distance = np.abs(coordinate_difference[:, 2])
    
    
    sigma_xy_downsample = float(np.sqrt(np.mean(xy_distance_clean**2)))
    sigma_z_downsample = float(np.sqrt(np.mean(np.mean(dz_clean**2))))

    # Independent uncertainty sources combine in quadrature.
    result = GlobalFeatureUncertainty(
        sigma_xy=float(np.hypot(sigma_xy, sigma_xy_downsample)),
        sigma_z=float(np.hypot(sigma_z, sigma_z_downsample)),
        sigma_z_intrinsic=float(sigma_z),
        sigma_xy_intrinsic=float(sigma_xy),
        sigma_z_downsample=sigma_z_downsample,
        sigma_xy_downsample=sigma_xy_downsample,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Save the dataclass in a portable, human-readable representation.
    validation_data = {
        **asdict(result),
        "downsample_factor": int(downsample_factor),
        "match_count": int(len(downsampled_xyz)),
        "match_count_used": int(keep.sum()),
        "match_count_rejected": n_rejected,
        "match_distance_gate_m": float(MATCH_DISTANCE_GATE_M),
        "downsampled_feature_path": str(DOWNSAMPLED_FEATURE_PATH),
        "truth_feature_path": str(TRUTH_FEATURE_PATH),
    }
    with VALIDATION_DATA_PATH.open("w", encoding="utf-8") as stream:
        json.dump(validation_data, stream, indent=2)
        stream.write("\n")

    # Preserve all validation measurements used to calculate the dataclass.
    np.savez_compressed(
        MATCHES_PATH,
        downsampled_feature_index=np.arange(len(downsampled_xyz)),
        truth_feature_index=truth_index,
        downsampled_xyz_m=downsampled_xyz,
        truth_xyz_m=matched_truth_xyz,
        dx_m=coordinate_difference[:, 0],
        dy_m=coordinate_difference[:, 1],
        dz_m=coordinate_difference[:, 2],
        xy_distance_m=xy_distance,
        z_distance_m=z_distance,
    )
    def inspect_match(rank_idx, downsampled_xyz, truth_xyz, dem_5m, dem_1p5m,
                   x_coords_5m, y_coords_5m, x_coords_1p5m, y_coords_1p5m,
                   window=40.0):
        dx, dy = downsampled_xyz[rank_idx, 0], downsampled_xyz[rank_idx, 1]
        tx, ty = truth_xyz[rank_idx, 0], truth_xyz[rank_idx, 1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, dem, xs, ys, title in [
            (axes[0], dem_5m, x_coords_5m, y_coords_5m, "5m downsampled"),
            (axes[1], dem_1p5m, x_coords_1p5m, y_coords_1p5m, "1.5m truth"),
        ]:
            cx = (dx + tx) / 2
            cy = (dy + ty) / 2
            x_mask = (xs >= cx - window) & (xs <= cx + window)
            y_mask = (ys >= cy - window) & (ys <= cy + window)
            ax.imshow(dem[np.ix_(np.where(y_mask)[0], np.where(x_mask)[0])],
                    extent=[xs[x_mask].min(), xs[x_mask].max(),
                            ys[y_mask].min(), ys[y_mask].max()],
                    origin="lower", cmap="terrain")
            ax.plot(dx, dy, "rx", markersize=12, markeredgewidth=3, label="downsampled")
            ax.plot(tx, ty, "b+", markersize=12, markeredgewidth=3, label="truth")
            ax.set_title(title)
            ax.legend()
        plt.tight_layout()
        plt.savefig(f"/tmp/match_inspect_rank{rank_idx}.png")
        print(f"saved match_inspect_rank{rank_idx}.png")

    
    data = np.load(MATCHES_PATH)
    xy = data["xy_distance_m"]
    downsampled_xyz = data["downsampled_xyz_m"]
    truth_xyz = data["truth_xyz_m"]
    
    order = np.argsort(xy)[::-1]  # largest first
    print(f"{'rank':>4} {'xy_dist_m':>10} {'downsampled_xy':>25} {'truth_xy':>25}")
    for rank, i in enumerate(order[:15]):
        print(f"{rank:>4} {xy[i]:>10.2f} "
            f"({downsampled_xyz[i,0]:>8.1f},{downsampled_xyz[i,1]:>8.1f})   "
            f"({truth_xyz[i,0]:>8.1f},{truth_xyz[i,1]:>8.1f})")
    
    for rank in [1, 2, 3, 4, 5]:
        inspect_match(order[rank], downsampled_xyz, truth_xyz, np.load(DEM_PATH), np.load(TRUTH_DEM_PATH), downsampled_xyz[:, 0], downsampled_xyz[:, 1], truth_xyz[:, 0], truth_xyz[:, 1])

    return result




def main() -> None:
    uncertainty = estimate_downsampled_uncertainty(
        sigma_xy=INTRINSIC_HORIZONTAL_ACCURACY_M,
        sigma_z=INTRINSIC_VERTICAL_ACCURACY_M,
        downsample_factor=round(LXY_M / TRUTH_DEM_LXY_M),
    )
    print(json.dumps(asdict(uncertainty), indent=2))
    print(f"Saved validation dataclass: {VALIDATION_DATA_PATH}")
    print(f"Saved nearest-neighbour data: {MATCHES_PATH}")
    print(f"global feature covariance matrix:\n{gloabal_feature_covariance(uncertainty.sigma_xy, uncertainty.sigma_z)}")
    
if __name__ == "__main__":
    
    main()
    

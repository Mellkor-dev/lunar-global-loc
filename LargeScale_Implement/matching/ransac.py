"""Covariance-aware exhaustive consensus refinement for DARCES matches."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import chi2

from matching.darces import (
    build_dem_interpolator,
    rigid_transform_2d,
    terrain_fitness,
    triangle_area,
)


def _points_xyz(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return points


def _covariances(
    covariance: np.ndarray,
    feature_count: int,
    name: str,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    expected = (feature_count, 3, 3)
    if covariance.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {covariance.shape}")
    if not np.isfinite(covariance).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return covariance


def _indices(indices: np.ndarray, upper_bound: int, name: str) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= upper_bound):
        raise IndexError(f"{name} contains an out-of-range feature index")
    return indices


def _local_grid_xyz(
    local_grid: np.ndarray,
    local_x_centers_m: np.ndarray,
    local_y_centers_m: np.ndarray,
) -> np.ndarray:
    local_grid = np.asarray(local_grid, dtype=np.float64)
    x = np.asarray(local_x_centers_m, dtype=np.float64)
    y = np.asarray(local_y_centers_m, dtype=np.float64)
    if local_grid.shape != (len(y), len(x)):
        raise ValueError("Local grid shape does not match its coordinate vectors")
    rows, columns = np.nonzero(np.isfinite(local_grid))
    if len(rows) == 0:
        raise ValueError("Local grid contains no finite terrain samples")
    return np.column_stack((x[columns], y[rows], local_grid[rows, columns]))


def _fit_2p5d(
    local_xyz: np.ndarray,
    global_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(local_xyz) < 3:
        raise ValueError("At least three correspondences are required")
    rotation, translation_xy_m = rigid_transform_2d(
        local_xyz[:, :2], global_xyz[:, :2]
    )
    translation_z_m = float(np.median(global_xyz[:, 2] - local_xyz[:, 2]))
    return rotation, translation_xy_m, translation_z_m


def _has_noncollinear_triplet(points_xy: np.ndarray, minimum_area_m2: float) -> bool:
    return any(
        triangle_area(points_xy[np.asarray(indices), :2]) >= minimum_area_m2
        for indices in combinations(range(len(points_xy)), 3)
    )


def covariance_residuals(
    rotation: np.ndarray,
    translation_xy_m: np.ndarray,
    translation_z_m: float,
    local_xyz: np.ndarray,
    global_xyz: np.ndarray,
    local_covariances: np.ndarray,
    global_covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return XYZ residuals and squared Mahalanobis distances."""
    rotation_3d = np.eye(3, dtype=np.float64)
    rotation_3d[:2, :2] = rotation
    transformed = np.empty_like(local_xyz)
    transformed[:, :2] = local_xyz[:, :2] @ rotation.T + translation_xy_m
    transformed[:, 2] = local_xyz[:, 2] + translation_z_m
    residuals = transformed - global_xyz
    mahalanobis_squared = np.empty(len(residuals), dtype=np.float64)
    for index, residual in enumerate(residuals):
        residual_covariance = (
            rotation_3d
            @ local_covariances[index]
            @ rotation_3d.T
            + global_covariances[index]
        )
        mahalanobis_squared[index] = float(
            residual @ np.linalg.pinv(residual_covariance) @ residual
        )
    return residuals, mahalanobis_squared


def exhaustive_ransac(
    local_features_xyz: np.ndarray,
    global_features_xyz: np.ndarray,
    local_feature_covariances: np.ndarray,
    global_feature_covariances: np.ndarray,
    correspondence_local_indices: np.ndarray,
    correspondence_global_indices: np.ndarray,
    local_grid: np.ndarray,
    global_dem: np.ndarray,
    local_x_centers_m: np.ndarray,
    local_y_centers_m: np.ndarray,
    global_x_centers_m: np.ndarray,
    global_y_centers_m: np.ndarray,
    *,
    confidence: float = 0.95,
    minimum_overlap: float = 0.50,
    minimum_triangle_area_m2: float = 1.0,
    maximum_refinement_iterations: int = 5,
    reference_points_xyz: np.ndarray | None = None,
) -> dict[str, object]:
    """Test every unique three-match model and return the best consensus.

    Models retain the rover pipeline's working 2.5-D representation. Inliers
    are classified with the combined local/global feature covariance, and
    valid consensus sets are ranked by DARCES terrain fitness.
    """
    local_features_xyz = _points_xyz(local_features_xyz, "local_features_xyz")
    global_features_xyz = _points_xyz(global_features_xyz, "global_features_xyz")
    local_feature_covariances = _covariances(
        local_feature_covariances, len(local_features_xyz), "local_feature_covariances"
    )
    global_feature_covariances = _covariances(
        global_feature_covariances,
        len(global_features_xyz),
        "global_feature_covariances",
    )
    local_indices = _indices(
        correspondence_local_indices, len(local_features_xyz), "local indices"
    )
    global_indices = _indices(
        correspondence_global_indices, len(global_features_xyz), "global indices"
    )
    if len(local_indices) != len(global_indices):
        raise ValueError("Local and global correspondence counts differ")
    if len(np.unique(local_indices)) != len(local_indices):
        raise ValueError("Local correspondence indices must be unique")
    if len(np.unique(global_indices)) != len(global_indices):
        raise ValueError("Global correspondence indices must be unique")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if maximum_refinement_iterations <= 0:
        raise ValueError("maximum_refinement_iterations must be positive")

    correspondence_count = len(local_indices)
    if correspondence_count < 3:
        return {
            "status": "insufficient_correspondences",
            "input_correspondence_count": correspondence_count,
            "model_count_tested": 0,
        }

    local_points = local_features_xyz[local_indices]
    global_points = global_features_xyz[global_indices]
    local_covariances = local_feature_covariances[local_indices]
    global_covariances = global_feature_covariances[global_indices]
    terrain_points = (
        _local_grid_xyz(local_grid, local_x_centers_m, local_y_centers_m)
        if reference_points_xyz is None
        else _points_xyz(reference_points_xyz, "reference_points_xyz")
    )
    interpolator = build_dem_interpolator(
        global_dem, global_x_centers_m, global_y_centers_m
    )
    threshold = float(chi2.ppf(confidence, df=3))
    best: dict[str, object] | None = None
    model_count = 0

    for sample_tuple in combinations(range(correspondence_count), 3):
        sample = np.asarray(sample_tuple, dtype=np.int64)
        if triangle_area(local_points[sample, :2]) < minimum_triangle_area_m2:
            continue
        if triangle_area(global_points[sample, :2]) < minimum_triangle_area_m2:
            continue
        model_count += 1
        try:
            rotation, translation_xy_m, translation_z_m = _fit_2p5d(
                local_points[sample], global_points[sample]
            )
        except (ValueError, np.linalg.LinAlgError):
            continue

        previous_inliers: np.ndarray | None = None
        inliers = sample
        for _ in range(maximum_refinement_iterations):
            _, mahalanobis_squared = covariance_residuals(
                rotation,
                translation_xy_m,
                translation_z_m,
                local_points,
                global_points,
                local_covariances,
                global_covariances,
            )
            inliers = np.flatnonzero(mahalanobis_squared <= threshold)
            if len(inliers) < 3:
                break
            if previous_inliers is not None and np.array_equal(inliers, previous_inliers):
                break
            if not _has_noncollinear_triplet(
                local_points[inliers], minimum_triangle_area_m2
            ):
                break
            previous_inliers = inliers.copy()
            rotation, translation_xy_m, translation_z_m = _fit_2p5d(
                local_points[inliers], global_points[inliers]
            )
        if len(inliers) < 3:
            continue

        residuals, mahalanobis_squared = covariance_residuals(
            rotation,
            translation_xy_m,
            translation_z_m,
            local_points,
            global_points,
            local_covariances,
            global_covariances,
        )
        inliers = np.flatnonzero(mahalanobis_squared <= threshold)
        if len(inliers) < 3:
            continue
        fitness, overlap = terrain_fitness(
            rotation,
            translation_xy_m,
            translation_z_m,
            terrain_points,
            interpolator,
            minimum_overlap=minimum_overlap,
        )
        if not np.isfinite(fitness):
            continue
        median_mahalanobis_squared = float(np.median(mahalanobis_squared[inliers]))
        ranking = (fitness, len(inliers), -median_mahalanobis_squared)
        if best is None or ranking > best["ranking"]:
            best = {
                "ranking": ranking,
                "rotation": rotation.copy(),
                "translation_xy_m": translation_xy_m.copy(),
                "translation_z_m": translation_z_m,
                "fitness": float(fitness),
                "overlap": float(overlap),
                "inliers": inliers.copy(),
                "residuals": residuals.copy(),
                "mahalanobis_squared": mahalanobis_squared.copy(),
                "median_mahalanobis_squared": median_mahalanobis_squared,
            }

    if best is None:
        return {
            "status": "no_valid_consensus",
            "input_correspondence_count": correspondence_count,
            "model_count_tested": model_count,
            "mahalanobis_threshold_squared": threshold,
        }

    inliers = np.asarray(best["inliers"], dtype=np.int64)
    outliers = np.setdiff1d(
        np.arange(correspondence_count, dtype=np.int64), inliers
    )
    rotation = np.asarray(best["rotation"], dtype=np.float64)
    heading_deg = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    status = "minimal_unverified" if correspondence_count == 3 else "solution"
    return {
        "status": status,
        "input_correspondence_count": correspondence_count,
        "inlier_count": len(inliers),
        "outlier_count": len(outliers),
        "inlier_local_indices": local_indices[inliers],
        "inlier_global_indices": global_indices[inliers],
        "outlier_local_indices": local_indices[outliers],
        "outlier_global_indices": global_indices[outliers],
        "rotation": rotation,
        "translation_xy_m": np.asarray(best["translation_xy_m"]),
        "translation_z_m": float(best["translation_z_m"]),
        "heading_deg": heading_deg,
        "terrain_fitness": float(best["fitness"]),
        "terrain_overlap": float(best["overlap"]),
        "median_mahalanobis_squared": float(best["median_mahalanobis_squared"]),
        "mahalanobis_squared": np.asarray(best["mahalanobis_squared"]),
        "residuals_xyz_m": np.asarray(best["residuals"]),
        "mahalanobis_threshold_squared": threshold,
        "model_count_tested": model_count,
    }

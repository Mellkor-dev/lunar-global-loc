#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


# Final Apollo tuning defaults. Keep these in one place so the public DARCES
# API and both command-line runners cannot silently drift apart.
DEFAULT_TRIALS_PER_SITE = 100_000
DEFAULT_HEADING_TOLERANCE_DEG = 5.0
DEFAULT_MINIMUM_OVERLAP = 0.50
DEFAULT_CLUSTER_POSITION_RADIUS_M = 50.0
DEFAULT_CLUSTER_HEADING_RADIUS_DEG = 5.0
DEFAULT_MINIMUM_CLUSTER_SIZE = 2
DEFAULT_SIDE_RATIO_TOLERANCE = 0.12
DEFAULT_MINIMUM_TRIANGLE_ANGLE_DEG = 10.0
DEFAULT_CONTROL_RMS_TOLERANCE_M = 10.0
DEFAULT_TOP_HYPOTHESIS_COUNT = 5
DEFAULT_COVARIANCE_SIGMA_MULTIPLIER = 2.0
DEFAULT_CONSENSUS_XY_TOLERANCE_M = 15.0
DEFAULT_MINIMUM_CONSENSUS_FEATURES = 4
# Terrain MAE remains the DARCES hypothesis fitness used for ranking. The
# historical pipeline did not impose a hard acceptance cutoff after ranking.
DEFAULT_MAXIMUM_TERRAIN_MAE_M: float | None = None


@dataclass(frozen=True)
class TriangleHypothesis:
    """One ordered local/global control-triangle correspondence."""

    local_indices: np.ndarray
    global_indices: np.ndarray
    edge_errors_m: np.ndarray
    control_rms_m: float

@dataclass(frozen=True)
class EvaluatedHypothesis:
    """A screened pose hypothesis with feature and terrain support."""

    rotation: np.ndarray
    translation_xy_m: np.ndarray
    translation_z_m: float

    fitness: float
    overlap: float

    consensus_count: int
    consensus_xy_rmse_m: float
    consensus_z_rmse_m: float

    local_indices: np.ndarray
    global_indices: np.ndarray
    control_rms_m: float

    @property
    def heading_deg(self) -> float:
        return float(
            np.degrees(
                np.arctan2(
                    self.rotation[1, 0],
                    self.rotation[0, 0],
                )
            )
        )



def _points_xy(points: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite N x 2 floating-point array."""
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {points.shape}")

    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    return points


def _points_xyz(points: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite N x 3 floating-point array."""
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")

    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    return points


def pairwise_dists(points_xy: np.ndarray) -> np.ndarray:
    """Return the complete pairwise Euclidean-distance matrix."""
    points_xy = _points_xy(points_xy, "points_xy")
    difference = points_xy[:, None, :] - points_xy[None, :, :]
    return np.linalg.norm(difference, axis=-1)


def triangle_area(points_xy: np.ndarray) -> float:
    """Return the unsigned area of a 2-D triangle."""
    points_xy = _points_xy(points_xy, "triangle")

    if points_xy.shape[0] != 3:
        raise ValueError("triangle must contain exactly three points")

    first = points_xy[1] - points_xy[0]
    second = points_xy[2] - points_xy[0]

    # Explicit 2-D determinant. NumPy 2.0 deprecated and newer releases reject
    # np.cross() on two-component vectors because cross products are formally
    # represented as three-dimensional vectors.
    twice_area = first[0] * second[1] - first[1] * second[0]
    return 0.5 * abs(float(twice_area))


def _minimum_triangle_angle_deg(points_xy: np.ndarray) -> float:
    """Return the smallest internal angle of a 2-D triangle."""
    points_xy = _points_xy(points_xy, "triangle")

    if points_xy.shape[0] != 3:
        raise ValueError("triangle must contain exactly three points")

    side_lengths = np.array(
        [
            np.linalg.norm(points_xy[1] - points_xy[2]),
            np.linalg.norm(points_xy[0] - points_xy[2]),
            np.linalg.norm(points_xy[0] - points_xy[1]),
        ],
        dtype=np.float64,
    )

    if np.any(side_lengths <= 0.0):
        return 0.0

    angles_deg: list[float] = []

    for opposite, adjacent_a, adjacent_b in (
        (side_lengths[0], side_lengths[1], side_lengths[2]),
        (side_lengths[1], side_lengths[0], side_lengths[2]),
        (side_lengths[2], side_lengths[0], side_lengths[1]),
    ):
        cosine = (
            adjacent_a**2 + adjacent_b**2 - opposite**2
        ) / (2.0 * adjacent_a * adjacent_b)

        angles_deg.append(
            float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        )

    return min(angles_deg)


def rigid_transform_2d(
    source_xy: np.ndarray,
    destination_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the proper 2-D rigid transform destination = R source + t."""
    source_xy = _points_xy(source_xy, "source_xy")
    destination_xy = _points_xy(destination_xy, "destination_xy")

    if source_xy.shape != destination_xy.shape:
        raise ValueError("source and destination must have matching shapes")

    if len(source_xy) < 2:
        raise ValueError("at least two point correspondences are required")

    source_mean = source_xy.mean(axis=0)
    destination_mean = destination_xy.mean(axis=0)

    source_centered = source_xy - source_mean
    destination_centered = destination_xy - destination_mean

    covariance = source_centered.T @ destination_centered
    left, _, right_transposed = np.linalg.svd(covariance)

    rotation = right_transposed.T @ left.T

    if np.linalg.det(rotation) < 0.0:
        right_transposed[-1, :] *= -1.0
        rotation = right_transposed.T @ left.T

    translation = destination_mean - rotation @ source_mean

    return rotation, translation


def _triangle_edge_lengths(points_xy: np.ndarray) -> np.ndarray:
    """Return edges [d01, d02, d12] for three ordered vertices."""
    return np.array(
        [
            np.linalg.norm(points_xy[0] - points_xy[1]),
            np.linalg.norm(points_xy[0] - points_xy[2]),
            np.linalg.norm(points_xy[1] - points_xy[2]),
        ],
        dtype=np.float64,
    )


def _edge_tolerances(
    local_triangle_xy: np.ndarray,
    global_triangle_xy: np.ndarray,
    local_indices: np.ndarray,
    global_indices: np.ndarray,
    *,
    fixed_tolerance_m: float,
    local_covariances: np.ndarray | None,
    global_covariances: np.ndarray | None,
    sigma_multiplier: float,
) -> np.ndarray:
    """Return one allowed length error for each corresponding triangle edge.

    When both covariance arrays are supplied, edge-length uncertainty is
    propagated along each edge. Otherwise one fixed tolerance is used.
    """
    if local_covariances is None or global_covariances is None:
        return np.full(3, fixed_tolerance_m, dtype=np.float64)

    edge_pairs = ((0, 1), (0, 2), (1, 2))
    tolerances = np.empty(3, dtype=np.float64)

    for edge_index, (first, second) in enumerate(edge_pairs):
        local_delta = local_triangle_xy[first] - local_triangle_xy[second]
        global_delta = global_triangle_xy[first] - global_triangle_xy[second]

        local_distance = float(np.linalg.norm(local_delta))
        global_distance = float(np.linalg.norm(global_delta))

        if local_distance <= 0.0 or global_distance <= 0.0:
            tolerances[edge_index] = 0.0
            continue

        local_covariance = (
            local_covariances[local_indices[first], :2, :2]
            + local_covariances[local_indices[second], :2, :2]
        )
        global_covariance = (
            global_covariances[global_indices[first], :2, :2]
            + global_covariances[global_indices[second], :2, :2]
        )

        local_variance = float(
            local_delta.T @ local_covariance @ local_delta
            / local_distance**2
        )
        global_variance = float(
            global_delta.T @ global_covariance @ global_delta
            / global_distance**2
        )

        propagated_tolerance = sigma_multiplier * np.sqrt(
            max(local_variance + global_variance, 0.0)
        )

        # Carle et al. Eq. (2): |dG - dL| <= k sigma_EGL. The fixed
        # tolerance is only the fallback when covariance is unavailable; it
        # must not clip the propagated uncertainty gate.
        tolerances[edge_index] = propagated_tolerance

    return tolerances


def _validate_covariances(
    covariances: np.ndarray | None,
    feature_count: int,
    name: str,
) -> np.ndarray | None:
    if covariances is None:
        return None

    covariances = np.asarray(covariances, dtype=np.float64)

    if covariances.shape != (feature_count, 3, 3):
        raise ValueError(
            f"{name} must have shape ({feature_count}, 3, 3), "
            f"got {covariances.shape}"
        )

    if not np.isfinite(covariances).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    return covariances


def generate_hypotheses(
    local_xy: np.ndarray,
    global_xy: np.ndarray,
    distance_tolerance_m: float,
    *,
    side_ratio_tolerance: float = 0.12,
    minimum_angle_deg: float = 10.0,
    control_rms_tolerance_m: float | None = None,
    maximum_hypotheses: int = 100_000,
    local_covariances: np.ndarray | None = None,
    global_covariances: np.ndarray | None = None,
    sigma_multiplier: float = 2.0,
) -> list[TriangleHypothesis]:
    """Generate all geometry-compatible ordered triangle correspondences.

    The search is deterministic. For each local triangle:

    1. Find ordered global point pairs compatible with local edge d01.
    2. Find every third global point compatible with edges d02 and d12.
    3. Reject degenerate triangles and inconsistent edge ratios.
    4. Estimate a rigid transform and retain only low-RMS controls.

    This covers all vertex permutations without randomly drawing millions of
    local/global triangle pairs.
    """
    local_xy = _points_xy(local_xy, "local_xy")
    global_xy = _points_xy(global_xy, "global_xy")

    if len(local_xy) < 3 or len(global_xy) < 3:
        return []

    if distance_tolerance_m <= 0.0:
        raise ValueError("distance_tolerance_m must be positive")

    if not 0.0 <= side_ratio_tolerance < 1.0:
        raise ValueError("side_ratio_tolerance must lie in [0, 1)")

    if minimum_angle_deg <= 0.0 or minimum_angle_deg >= 60.0:
        raise ValueError("minimum_angle_deg must lie between 0 and 60")

    if maximum_hypotheses <= 0:
        raise ValueError("maximum_hypotheses must be positive")

    if sigma_multiplier <= 0.0:
        raise ValueError("sigma_multiplier must be positive")

    if control_rms_tolerance_m is None:
        control_rms_tolerance_m = distance_tolerance_m

    local_covariances = _validate_covariances(
        local_covariances,
        len(local_xy),
        "local_covariances",
    )
    global_covariances = _validate_covariances(
        global_covariances,
        len(global_xy),
        "global_covariances",
    )

    global_distances = pairwise_dists(global_xy)
    lookup_tolerance_m = distance_tolerance_m
    if local_covariances is not None and global_covariances is not None:
        max_local_xy_variance = max(
            np.linalg.eigvalsh(covariance[:2, :2]).max()
            for covariance in local_covariances
        )
        max_global_xy_variance = max(
            np.linalg.eigvalsh(covariance[:2, :2]).max()
            for covariance in global_covariances
        )
        lookup_tolerance_m = max(
            lookup_tolerance_m,
            sigma_multiplier
            * np.sqrt(
                2.0 * max_local_xy_variance
                + 2.0 * max_global_xy_variance
            ),
        )
    hypotheses: list[TriangleHypothesis] = []

    local_triangle_indices = list(combinations(range(len(local_xy)), 3))

    for local_tuple in local_triangle_indices:
        local_indices = np.asarray(local_tuple, dtype=np.int64)
        local_triangle = local_xy[local_indices]

        if _minimum_triangle_angle_deg(local_triangle) < minimum_angle_deg:
            continue

        local_edges = _triangle_edge_lengths(local_triangle)
        if np.any(local_edges <= 0.0):
            continue

        # Fixed tolerance is used for the fast candidate lookup. Optional
        # covariance propagation is applied exactly after a candidate triangle
        # has been formed.
        first_edge_mask = (
            np.abs(global_distances - local_edges[0])
            <= lookup_tolerance_m
        )
        np.fill_diagonal(first_edge_mask, False)

        first_vertices, second_vertices = np.nonzero(first_edge_mask)

        for global_first, global_second in zip(
            first_vertices,
            second_vertices,
            strict=True,
        ):
            third_mask = (
                np.abs(
                    global_distances[global_first] - local_edges[1]
                )
                    <= lookup_tolerance_m
            )
            third_mask &= (
                np.abs(
                    global_distances[global_second] - local_edges[2]
                )
                    <= lookup_tolerance_m
            )
            third_mask[global_first] = False
            third_mask[global_second] = False

            for global_third in np.flatnonzero(third_mask):
                global_indices = np.array(
                    [global_first, global_second, global_third],
                    dtype=np.int64,
                )
                global_triangle = global_xy[global_indices]

                if (
                    _minimum_triangle_angle_deg(global_triangle)
                    < minimum_angle_deg
                ):
                    continue

                global_edges = _triangle_edge_lengths(global_triangle)
                edge_errors = np.abs(local_edges - global_edges)

                tolerances = _edge_tolerances(
                    local_triangle,
                    global_triangle,
                    local_indices,
                    global_indices,
                    fixed_tolerance_m=distance_tolerance_m,
                    local_covariances=local_covariances,
                    global_covariances=global_covariances,
                    sigma_multiplier=sigma_multiplier,
                )

                if not np.all(edge_errors <= tolerances):
                    continue

                edge_scales = np.maximum(local_edges, global_edges)
                if np.any(edge_scales <= 0.0):
                    continue

                if np.max(edge_errors / edge_scales) > side_ratio_tolerance:
                    continue

                rotation, translation = rigid_transform_2d(
                    local_triangle,
                    global_triangle,
                )

                aligned = local_triangle @ rotation.T + translation
                control_rms = float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (aligned - global_triangle) ** 2,
                                axis=1,
                            )
                        )
                    )
                )

                if control_rms > control_rms_tolerance_m:
                    continue

                hypotheses.append(
                    TriangleHypothesis(
                        local_indices=local_indices.copy(),
                        global_indices=global_indices,
                        edge_errors_m=edge_errors,
                        control_rms_m=control_rms,
                    )
                )

                if len(hypotheses) >= maximum_hypotheses:
                    return hypotheses

    return hypotheses


def _angle_error_deg(first_deg: float, second_deg: float) -> float:
    """Smallest absolute wrapped angular difference in degrees."""
    return float(
        abs((first_deg - second_deg + 180.0) % 360.0 - 180.0)
    )


def hypothesis_within_global_bounds(
    translation_xy_m: np.ndarray,
    global_x_centers_m: np.ndarray,
    global_y_centers_m: np.ndarray,
) -> bool:
    """Check whether the estimated rover origin lies inside the DEM."""
    translation_xy_m = np.asarray(
        translation_xy_m,
        dtype=np.float64,
    ).reshape(-1)

    if translation_xy_m.shape != (2,):
        raise ValueError("translation_xy_m must contain two values")

    x_min = float(np.min(global_x_centers_m))
    x_max = float(np.max(global_x_centers_m))
    y_min = float(np.min(global_y_centers_m))
    y_max = float(np.max(global_y_centers_m))

    rover_x, rover_y = translation_xy_m

    return x_min <= rover_x <= x_max and y_min <= rover_y <= y_max


def hypothesis_matches_heading(
    rotation: np.ndarray,
    heading_measurement_deg: float,
    heading_tolerance_deg: float,
) -> bool:
    """Check the candidate heading against the independent measurement."""
    heading_deg = float(
        np.degrees(
            np.arctan2(rotation[1, 0], rotation[0, 0])
        )
    )

    return (
        _angle_error_deg(heading_deg, heading_measurement_deg)
        <= heading_tolerance_deg
    )


def estimate_vertical_translation(
    local_z_m: np.ndarray,
    global_z_m: np.ndarray,
    z_residual_tolerance_m: float,
) -> tuple[float, np.ndarray] | None:
    """Estimate robust vertical offset and reject inconsistent controls."""
    local_z_m = np.asarray(local_z_m, dtype=np.float64)
    global_z_m = np.asarray(global_z_m, dtype=np.float64)

    if local_z_m.shape != global_z_m.shape:
        raise ValueError("local and global Z arrays must have equal shapes")

    if local_z_m.ndim != 1:
        raise ValueError("local and global Z arrays must be one-dimensional")

    if z_residual_tolerance_m <= 0.0:
        raise ValueError("z_residual_tolerance_m must be positive")

    translation_z_m = float(np.median(global_z_m - local_z_m))
    residual_m = local_z_m + translation_z_m - global_z_m

    if np.max(np.abs(residual_m)) > z_residual_tolerance_m:
        return None
    if translation_z_m is None:
        print(translation_z_m)
    return translation_z_m, residual_m


def build_dem_interpolator(
    global_dem: np.ndarray,
    global_x_centers_m: np.ndarray,
    global_y_centers_m: np.ndarray,
) -> RegularGridInterpolator:
    """Build an interpolator for a north-up DEM.

    X coordinates increase from west to east. Y coordinates decrease with
    raster row, so Y and the DEM rows are reversed for SciPy's interpolator.
    """
    global_dem = np.asarray(global_dem, dtype=np.float64)
    global_x_centers_m = np.asarray(
        global_x_centers_m,
        dtype=np.float64,
    )
    global_y_centers_m = np.asarray(
        global_y_centers_m,
        dtype=np.float64,
    )

    expected_shape = (
        len(global_y_centers_m),
        len(global_x_centers_m),
    )

    if global_dem.shape != expected_shape:
        raise ValueError(
            "Global DEM shape does not match coordinate vectors: "
            f"{global_dem.shape} != {expected_shape}"
        )

    if not np.all(np.diff(global_x_centers_m) > 0.0):
        raise ValueError("Global X centers must increase")

    if not np.all(np.diff(global_y_centers_m) < 0.0):
        raise ValueError(
            "Global Y centers must decrease for a north-up raster"
        )

    return RegularGridInterpolator(
        (global_y_centers_m[::-1], global_x_centers_m),
        global_dem[::-1],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

def feature_consensus(
    rotation: np.ndarray,
    translation_xy_m: np.ndarray,
    translation_z_m: float,
    local_features_xyz: np.ndarray,
    global_features_xyz: np.ndarray,
    *,
    xy_tolerance_m: float,
    z_tolerance_m: float,
    local_covariances: np.ndarray | None = None,
    global_covariances: np.ndarray | None = None,
    sigma_multiplier: float = 2.0,
) -> dict[str, object]:
    """Expand a pose into unique, uncertainty-compatible correspondences."""
    if xy_tolerance_m <= 0.0:
        raise ValueError("xy_tolerance_m must be positive")

    if z_tolerance_m <= 0.0:
        raise ValueError("z_tolerance_m must be positive")

    local_features_xyz = _points_xyz(
        local_features_xyz, "local_features_xyz"
    )
    global_features_xyz = _points_xyz(
        global_features_xyz, "global_features_xyz"
    )
    local_covariances = _validate_covariances(
        local_covariances, len(local_features_xyz), "local_covariances"
    )
    global_covariances = _validate_covariances(
        global_covariances, len(global_features_xyz), "global_covariances"
    )
    if sigma_multiplier <= 0.0:
        raise ValueError("sigma_multiplier must be positive")

    transformed_xyz = np.empty_like(local_features_xyz, dtype=np.float64)

    transformed_xyz[:, :2] = (
        local_features_xyz[:, :2] @ rotation.T
        + translation_xy_m
    )

    transformed_xyz[:, 2] = (
        local_features_xyz[:, 2]
        + translation_z_m
    )

    rotation_3d = np.eye(3, dtype=np.float64)
    rotation_3d[:2, :2] = rotation
    covariance_gating = (
        local_covariances is not None and global_covariances is not None
    )
    search_radius_m = xy_tolerance_m
    if covariance_gating:
        max_local_xy_variance = max(
            np.linalg.eigvalsh(covariance[:2, :2]).max()
            for covariance in local_covariances
        )
        max_global_xy_variance = max(
            np.linalg.eigvalsh(covariance[:2, :2]).max()
            for covariance in global_covariances
        )
        search_radius_m = sigma_multiplier * np.sqrt(
            max_local_xy_variance + max_global_xy_variance
        )

    global_tree = cKDTree(global_features_xyz[:, :2])
    proposals: list[tuple[float, int, int, float, float]] = []
    for local_index, transformed_feature in enumerate(transformed_xyz):
        nearby_global_indices = global_tree.query_ball_point(
            transformed_feature[:2], search_radius_m
        )
        for global_index in nearby_global_indices:
            residual = transformed_feature - global_features_xyz[global_index]
            xy_error_m = float(np.linalg.norm(residual[:2]))
            z_error_m = float(abs(residual[2]))
            if covariance_gating:
                residual_covariance = (
                    rotation_3d
                    @ local_covariances[local_index]
                    @ rotation_3d.T
                    + global_covariances[global_index]
                )
                normalized_error = float(
                    residual @ np.linalg.pinv(residual_covariance) @ residual
                )
                if normalized_error > sigma_multiplier**2:
                    continue
            else:
                if z_error_m > z_tolerance_m:
                    continue
                normalized_error = (
                    (xy_error_m / xy_tolerance_m) ** 2
                    + (z_error_m / z_tolerance_m) ** 2
                )
            proposals.append(
                (
                    normalized_error,
                    local_index,
                    int(global_index),
                    xy_error_m,
                    z_error_m,
                )
            )

    # Greedy assignment by normalized uncertainty error enforces one-to-one
    # local/global associations and avoids duplicate landmark matches.
    accepted_local_indices: list[int] = []
    accepted_global_indices: list[int] = []
    accepted_xy_errors_m: list[float] = []
    accepted_z_errors_m: list[float] = []
    used_local_indices: set[int] = set()
    used_global_indices: set[int] = set()
    for _, local_index, global_index, xy_error_m, z_error_m in sorted(proposals):
        if (
            local_index in used_local_indices
            or global_index in used_global_indices
        ):
            continue
        used_local_indices.add(local_index)
        used_global_indices.add(global_index)
        accepted_local_indices.append(local_index)
        accepted_global_indices.append(global_index)
        accepted_xy_errors_m.append(xy_error_m)
        accepted_z_errors_m.append(z_error_m)

    accepted_local = np.asarray(
        accepted_local_indices,
        dtype=np.int64,
    )

    accepted_global = np.asarray(
        accepted_global_indices,
        dtype=np.int64,
    )

    if len(accepted_local) == 0:
        xy_rmse_m = np.inf
        z_rmse_m = np.inf
    else:
        xy_rmse_m = float(
            np.sqrt(
                np.mean(
                    np.asarray(accepted_xy_errors_m) ** 2
                )
            )
        )

        z_rmse_m = float(
            np.sqrt(
                np.mean(
                    np.asarray(accepted_z_errors_m) ** 2
                )
            )
        )

    return {
        "count": len(accepted_local),
        "local_indices": accepted_local,
        "global_indices": accepted_global,
        "xy_rmse_m": xy_rmse_m,
        "z_rmse_m": z_rmse_m,
    }
    
    
    
def terrain_fitness(
    rotation: np.ndarray,
    translation_xy_m: np.ndarray,
    translation_z_m: float,
    local_xyz: np.ndarray,
    dem_interpolator: RegularGridInterpolator,
    *,
    minimum_overlap: float = 0.50,
) -> tuple[float, float]:
    """Return negative mean absolute elevation error and map overlap."""
    local_xyz = _points_xyz(local_xyz, "local_xyz")

    if not 0.0 < minimum_overlap <= 1.0:
        raise ValueError("minimum_overlap must lie in (0, 1]")

    world_xy = (
        local_xyz[:, :2] @ rotation.T
        + translation_xy_m
    )

    global_z_m = dem_interpolator(
        np.column_stack(
            (world_xy[:, 1], world_xy[:, 0])
        )
    )

    valid = np.isfinite(global_z_m)
    overlap = float(valid.mean()) if len(valid) else 0.0

    if overlap < minimum_overlap or not valid.any():
        return -np.inf, overlap

    transformed_z_m = (
        local_xyz[valid, 2] + translation_z_m
    )

    mean_absolute_error_m = float(
        np.mean(
            np.abs(
                transformed_z_m - global_z_m[valid]
            )
        )
    )

    return -mean_absolute_error_m, overlap


def decimate_reference_points_xy(
    points_xyz: np.ndarray,
    spacing_m: float,
) -> np.ndarray:
    """Retain one observed LiDAR point per XY cell of ``spacing_m``.

    The retained sample is the observed point nearest each cell centre.  This
    avoids averaging elevations or otherwise synthesising terrain while making
    DARCES fitness sampling approximately uniform in the map plane.
    """
    points_xyz = _points_xyz(points_xyz, "points_xyz")
    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")

    cell_xy = np.floor(points_xyz[:, :2] / spacing_m).astype(np.int64)
    cell_centres_xy = (cell_xy.astype(np.float64) + 0.5) * spacing_m
    distance_squared = np.sum(
        (points_xyz[:, :2] - cell_centres_xy) ** 2,
        axis=1,
    )
    order = np.lexsort((distance_squared, cell_xy[:, 1], cell_xy[:, 0]))
    ordered_cells = cell_xy[order]
    first_in_cell = np.ones(len(order), dtype=bool)
    first_in_cell[1:] = np.any(ordered_cells[1:] != ordered_cells[:-1], axis=1)
    return np.ascontiguousarray(points_xyz[order[first_in_cell]])


def _local_grid_xyz(
    local_grid: np.ndarray,
    local_x_centers_m: np.ndarray,
    local_y_centers_m: np.ndarray,
) -> np.ndarray:
    """Convert all finite local raster cells to physical XYZ coordinates."""
    local_grid = np.asarray(local_grid, dtype=np.float64)
    local_x_centers_m = np.asarray(
        local_x_centers_m,
        dtype=np.float64,
    )
    local_y_centers_m = np.asarray(
        local_y_centers_m,
        dtype=np.float64,
    )

    expected_shape = (
        len(local_y_centers_m),
        len(local_x_centers_m),
    )

    if local_grid.shape != expected_shape:
        raise ValueError(
            "Local DEM shape does not match coordinate vectors: "
            f"{local_grid.shape} != {expected_shape}"
        )

    valid_rows, valid_columns = np.nonzero(
        np.isfinite(local_grid)
    )

    if len(valid_rows) == 0:
        raise ValueError(
            "Local elevation grid contains no finite cells"
        )

    return np.column_stack(
        (
            local_x_centers_m[valid_columns],
            local_y_centers_m[valid_rows],
            local_grid[valid_rows, valid_columns],
        )
    )


def select_hypothesis_cluster(
    candidates: list[EvaluatedHypothesis],
    *,
    position_radius_m: float,
    heading_radius_deg: float,
    minimum_cluster_size: int,
    top_hypothesis_count: int = 5,
) -> tuple[EvaluatedHypothesis, int, int] | None:
    
    if not candidates:
        return None

    if position_radius_m <= 0.0:
        raise ValueError("position_radius_m must be positive")

    if heading_radius_deg <= 0.0:
        raise ValueError("heading_radius_deg must be positive")

    if minimum_cluster_size <= 0:
        raise ValueError("minimum_cluster_size must be positive")

    if top_hypothesis_count <= 0:
        raise ValueError("top_hypothesis_count must be positive")

    ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.fitness,
                -candidate.control_rms_m,
            ),
            reverse=True,
        )

    top_candidates = ranked[:top_hypothesis_count]

    best_members: list[int] = []
    best_distinct_triangle_count = 0

    for center in top_candidates:
        members: list[int] = []

        for index, candidate in enumerate(top_candidates):
            position_difference_m = float(
                np.linalg.norm(
                    candidate.translation_xy_m
                    - center.translation_xy_m
                )
            )
            heading_difference_deg = _angle_error_deg(
                candidate.heading_deg,
                center.heading_deg,
            )

            if (
                position_difference_m <= position_radius_m
                and heading_difference_deg <= heading_radius_deg
            ):
                members.append(index)

        distinct_local_triangles = {
            tuple(
                sorted(
                    int(value)
                    for value in top_candidates[index].local_indices
                )
            )
            for index in members
        }

        distinct_triangle_count = len(distinct_local_triangles)

        if distinct_triangle_count > best_distinct_triangle_count:
            best_members = members
            best_distinct_triangle_count = distinct_triangle_count
        elif (
            distinct_triangle_count == best_distinct_triangle_count
            and members
            and best_members
        ):
            member_best_fitness = max(
                top_candidates[index].fitness
                for index in members
            )
            existing_best_fitness = max(
                top_candidates[index].fitness
                for index in best_members
            )

            if member_best_fitness > existing_best_fitness:
                best_members = members

    if best_distinct_triangle_count < minimum_cluster_size:
        return None

    selected = max(
        (top_candidates[index] for index in best_members),
        key=lambda candidate: candidate.fitness,
    )

    return (
        selected,
        best_distinct_triangle_count,
        len(best_members),
    )


def run_darces(
    local_features_xyz: np.ndarray,
    global_features_xyz: np.ndarray,
    local_grid: np.ndarray,
    global_dem: np.ndarray,
    local_x_centers_m: np.ndarray,
    local_y_centers_m: np.ndarray,
    global_x_centers_m: np.ndarray,
    global_y_centers_m: np.ndarray,
    *,
    n_trials: int = DEFAULT_TRIALS_PER_SITE,
    distance_tolerance_m: float,
    z_residual_tolerance_m: float,
    heading_measurement_deg: float,
    heading_tolerance_deg: float = DEFAULT_HEADING_TOLERANCE_DEG,
    minimum_overlap: float = DEFAULT_MINIMUM_OVERLAP,
    cluster_position_radius_m: float | None = (
        DEFAULT_CLUSTER_POSITION_RADIUS_M
    ),
    cluster_heading_radius_deg: float = DEFAULT_CLUSTER_HEADING_RADIUS_DEG,
    minimum_cluster_size: int = DEFAULT_MINIMUM_CLUSTER_SIZE,
    seed: int | None = None,
    side_ratio_tolerance: float = DEFAULT_SIDE_RATIO_TOLERANCE,
    minimum_triangle_angle_deg: float = DEFAULT_MINIMUM_TRIANGLE_ANGLE_DEG,
    control_rms_tolerance_m: float | None = (
        DEFAULT_CONTROL_RMS_TOLERANCE_M
    ),
    top_hypothesis_count: int = DEFAULT_TOP_HYPOTHESIS_COUNT,
    local_covariances: np.ndarray | None = None,
    global_covariances: np.ndarray | None = None,
    covariance_sigma_multiplier: float = DEFAULT_COVARIANCE_SIGMA_MULTIPLIER,
    consensus_xy_tolerance_m: float = DEFAULT_CONSENSUS_XY_TOLERANCE_M,
    minimum_consensus_features: int = DEFAULT_MINIMUM_CONSENSUS_FEATURES,
    use_feature_consensus: bool = False,
    reference_points_xyz: np.ndarray | None = None,
    maximum_terrain_mae_m: float | None = DEFAULT_MAXIMUM_TERRAIN_MAE_M,
) -> dict[str, object] | None:
    """Run deterministic DARCES registration.

    ``n_trials`` is retained for compatibility with existing runners, but now
    acts as the maximum number of accepted control hypotheses. The search
    itself is deterministic and exhaustive within the supplied geometry gates.
    ``seed`` is retained for API compatibility and is intentionally unused.
    """
    del seed
    
    if use_feature_consensus:
        if consensus_xy_tolerance_m <= 0.0:
            raise ValueError(
                "consensus_xy_tolerance_m must be positive"
            )

        if minimum_consensus_features < 3:
            raise ValueError(
                "minimum_consensus_features must be at least 3"
            )
            

    local_features_xyz = _points_xyz(
        local_features_xyz,
        "local_features_xyz",
    )
    global_features_xyz = _points_xyz(
        global_features_xyz,
        "global_features_xyz",
    )

    if len(local_features_xyz) < 3:
        raise ValueError(
            "DARCES requires at least three local features"
        )

    if len(global_features_xyz) < 3:
        raise ValueError(
            "DARCES requires at least three global features"
        )

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    if heading_tolerance_deg <= 0.0:
        raise ValueError("heading_tolerance_deg must be positive")

    if maximum_terrain_mae_m is not None and maximum_terrain_mae_m <= 0.0:
        raise ValueError("maximum_terrain_mae_m must be positive or None")

    if reference_points_xyz is None:
        # Compatibility fallback for direct callers and older tests.
        terrain_points_xyz = _local_grid_xyz(
            local_grid,
            local_x_centers_m,
            local_y_centers_m,
        )
    else:
        # Preferred method: use observed LiDAR points decimated by the caller
        # to half of the selected global-map cell spacing.
        terrain_points_xyz = _points_xyz(
            reference_points_xyz,
            "reference_points_xyz",
        )

    dem_interpolator = build_dem_interpolator(
        global_dem,
        global_x_centers_m,
        global_y_centers_m,
    )

    hypotheses = generate_hypotheses(
        local_features_xyz[:, :2],
        global_features_xyz[:, :2],
        distance_tolerance_m,
        side_ratio_tolerance=side_ratio_tolerance,
        minimum_angle_deg=minimum_triangle_angle_deg,
        control_rms_tolerance_m=control_rms_tolerance_m,
        maximum_hypotheses=n_trials,
        local_covariances=local_covariances,
        global_covariances=global_covariances,
        sigma_multiplier=covariance_sigma_multiplier,
    )

    print(
        f"Generated {len(hypotheses)} ordered hypotheses "
        "by deterministic edge lookup"
    )

    rejection_counts = {
        "outside_map": 0,
        "heading": 0,
        "vertical": 0,
        "feature_consensus": 0,
        "overlap_or_fitness": 0,
    }

    candidates: list[EvaluatedHypothesis] = []

    for hypothesis in hypotheses:
        local_control = local_features_xyz[
            hypothesis.local_indices
        ]
        global_control = global_features_xyz[
            hypothesis.global_indices
        ]

        rotation, translation_xy_m = rigid_transform_2d(
            local_control[:, :2],
            global_control[:, :2],
        )

        if not hypothesis_within_global_bounds(
            translation_xy_m,
            global_x_centers_m,
            global_y_centers_m,
        ):
            rejection_counts["outside_map"] += 1
            continue

        if not hypothesis_matches_heading(
            rotation,
            heading_measurement_deg,
            heading_tolerance_deg,
        ):
            rejection_counts["heading"] += 1
            continue

        vertical_result = estimate_vertical_translation(
            local_control[:, 2],
            global_control[:, 2],
            z_residual_tolerance_m,
        )
        if vertical_result is None:
            rejection_counts["vertical"] += 1
            continue
        
        translation_z_m, _ = vertical_result

        # Default diagnostic values when consensus screening is disabled.
        consensus_count = 0
        consensus_xy_rmse_m = np.inf
        consensus_z_rmse_m = np.inf

        if use_feature_consensus:
            consensus = feature_consensus(
                rotation=rotation,
                translation_xy_m=translation_xy_m,
                translation_z_m=translation_z_m,
                local_features_xyz=local_features_xyz,
                global_features_xyz=global_features_xyz,
                xy_tolerance_m=consensus_xy_tolerance_m,
                z_tolerance_m=z_residual_tolerance_m,
                local_covariances=local_covariances,
                global_covariances=global_covariances,
                sigma_multiplier=covariance_sigma_multiplier,
            )
            consensus_count = int(consensus["count"])
            consensus_xy_rmse_m = float(consensus["xy_rmse_m"])
            consensus_z_rmse_m = float(consensus["z_rmse_m"])

            if consensus_count < minimum_consensus_features:
                rejection_counts["feature_consensus"] += 1
                continue

        fitness, overlap = terrain_fitness(
            rotation,
            translation_xy_m,
            translation_z_m,
            terrain_points_xyz,
            dem_interpolator,
            minimum_overlap=minimum_overlap,
        )

        if not np.isfinite(fitness):
            rejection_counts["overlap_or_fitness"] += 1
            continue

        candidates.append(
            EvaluatedHypothesis(
                rotation=rotation,
                translation_xy_m=translation_xy_m,
                translation_z_m=translation_z_m,
                fitness=fitness,
                overlap=overlap,

                # Use the initialized scalar variables, not consensus["..."].
                consensus_count=consensus_count,
                consensus_xy_rmse_m=consensus_xy_rmse_m,
                consensus_z_rmse_m=consensus_z_rmse_m,

                local_indices=hypothesis.local_indices,
                global_indices=hypothesis.global_indices,
                control_rms_m=hypothesis.control_rms_m,
            )
        )

    print("Hypothesis screening")
    print("--------------------")
    print(f"Passed all screening: {len(candidates)}")
    for reason, count in rejection_counts.items():
        print(f"Rejected by {reason:18s}: {count}")

    if cluster_position_radius_m is None:
        cluster_position_radius_m = distance_tolerance_m

    clustered = select_hypothesis_cluster(
        candidates,
        position_radius_m=cluster_position_radius_m,
        heading_radius_deg=cluster_heading_radius_deg,
        minimum_cluster_size=minimum_cluster_size,
        top_hypothesis_count=top_hypothesis_count,
    )

    if clustered is None:
        print(
            "DARCES: no valid high-fitness pose cluster found"
        )
        return None

    best, distinct_triangle_count, cluster_member_count = clustered

    selected_terrain_mae_m = -best.fitness
    if (
        maximum_terrain_mae_m is not None
        and selected_terrain_mae_m > maximum_terrain_mae_m
    ):
        print(
            "DARCES: selected pose rejected by terrain MAE gate "
            f"({selected_terrain_mae_m:.4f} m > "
            f"{maximum_terrain_mae_m:.4f} m)"
        )
        return None

    # Expand the three control-point matches into all pose-compatible feature
    # correspondences for the downstream RANSAC stage.
    expanded_correspondences = feature_consensus(
        rotation=best.rotation,
        translation_xy_m=best.translation_xy_m,
        translation_z_m=best.translation_z_m,
        local_features_xyz=local_features_xyz,
        global_features_xyz=global_features_xyz,
        xy_tolerance_m=consensus_xy_tolerance_m,
        z_tolerance_m=z_residual_tolerance_m,
        local_covariances=local_covariances,
        global_covariances=global_covariances,
        sigma_multiplier=covariance_sigma_multiplier,
    )

    print(f"Best fitness:             {best.fitness:.4f}")
    print(
        "Estimated XY translation: "
        f"{best.translation_xy_m}"
    )
    print(
        f"Estimated Z translation:  "
        f"{best.translation_z_m:.4f}"
    )
    print(f"Estimated heading:        {best.heading_deg:.2f} deg")
    print(
        "Distinct local triangles: "
        f"{distinct_triangle_count}"
    )
    print(f"Cluster members:          {cluster_member_count}")
    if use_feature_consensus:
        print(
            f"Feature consensus:        "
            f"{best.consensus_count}/{len(local_features_xyz)}"
        )
        print(
            f"Consensus XY RMSE:        "
            f"{best.consensus_xy_rmse_m:.3f} m"
        )
        print(
            f"Consensus Z RMSE:         "
            f"{best.consensus_z_rmse_m:.3f} m"
        )

    return {
        "R": best.rotation,
        "t": best.translation_xy_m,
        "tz": best.translation_z_m,
        "fitness": best.fitness,
        "overlap": best.overlap,
        "heading_deg": best.heading_deg,
        "local_idx": best.local_indices,
        "global_idx": best.global_indices,
        "correspondence_local_idx": expanded_correspondences["local_indices"],
        "correspondence_global_idx": expanded_correspondences["global_indices"],
        "correspondence_count": int(expanded_correspondences["count"]),
        "correspondence_xy_rmse_m": float(
            expanded_correspondences["xy_rmse_m"]
        ),
        "correspondence_z_rmse_m": float(
            expanded_correspondences["z_rmse_m"]
        ),
        "control_rms_m": best.control_rms_m,
        "cluster_size": distinct_triangle_count,
        "cluster_member_count": cluster_member_count,
        "evaluated_hypothesis_count": len(candidates),
        "generated_hypothesis_count": len(hypotheses),
        "rejection_counts": rejection_counts,
        "consensus_count": best.consensus_count,
        "consensus_xy_rmse_m": best.consensus_xy_rmse_m,
        "consensus_z_rmse_m": best.consensus_z_rmse_m,
    }

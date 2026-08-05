"""Synthetic regression tests for repaired DARCES geometry."""

from itertools import permutations

import numpy as np

from matching.darces import (
    feature_consensus,
    generate_hypotheses,
    rigid_transform_2d,
    run_darces,
)


def _rotation(yaw_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    return np.array(
        ((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw)))
    )


def test_all_triangle_vertex_orders_recover_transform() -> None:
    local = np.array(((0.0, 0.0), (20.0, 0.0), (5.0, 30.0)))
    expected_rotation = _rotation(37.0)
    expected_translation = np.array((120.0, -45.0))
    global_points = local @ expected_rotation.T + expected_translation

    for local_order in permutations(range(3)):
        for global_order in permutations(range(3)):
            shuffled_local = local[list(local_order)]
            shuffled_global = global_points[list(global_order)]
            hypotheses = generate_hypotheses(
                shuffled_local,
                shuffled_global,
                distance_tolerance_m=0.01,
                control_rms_tolerance_m=0.01,
            )
            recovered = []
            for hypothesis in hypotheses:
                rotation, translation = rigid_transform_2d(
                    shuffled_local[hypothesis.local_indices],
                    shuffled_global[hypothesis.global_indices],
                )
                recovered.append((rotation, translation))
            assert any(
                np.allclose(rotation, expected_rotation, atol=1e-10)
                and np.allclose(translation, expected_translation, atol=1e-10)
                for rotation, translation in recovered
            )


def test_end_to_end_synthetic_2p5d_pose() -> None:
    local_x = np.arange(-20.0, 25.0, 5.0)
    local_y = np.arange(20.0, -25.0, -5.0)
    global_x = np.arange(-100.0, 105.0, 5.0)
    global_y = np.arange(100.0, -105.0, -5.0)
    query_y, query_x = np.meshgrid(global_y, global_x, indexing="ij")
    global_dem = 0.01 * query_x + 0.02 * query_y

    expected_rotation = _rotation(20.0)
    expected_xy = np.array((30.0, -20.0))
    expected_z = 5.0
    local_query_y, local_query_x = np.meshgrid(
        local_y,
        local_x,
        indexing="ij",
    )
    local_xy = np.column_stack(
        (local_query_x.ravel(), local_query_y.ravel())
    )
    world_xy = local_xy @ expected_rotation.T + expected_xy
    local_grid = (
        0.01 * world_xy[:, 0] + 0.02 * world_xy[:, 1] - expected_z
    ).reshape(local_query_x.shape)

    feature_local_xy = np.array(
        ((-15.0, -10.0), (15.0, -5.0), (-5.0, 15.0))
    )
    feature_global_xy = (
        feature_local_xy @ expected_rotation.T + expected_xy
    )
    feature_global_z = (
        0.01 * feature_global_xy[:, 0]
        + 0.02 * feature_global_xy[:, 1]
    )
    feature_local_z = feature_global_z - expected_z

    result = run_darces(
        local_features_xyz=np.column_stack(
            (feature_local_xy, feature_local_z)
        ),
        global_features_xyz=np.column_stack(
            (feature_global_xy, feature_global_z)
        ),
        local_grid=local_grid,
        global_dem=global_dem,
        local_x_centers_m=local_x,
        local_y_centers_m=local_y,
        global_x_centers_m=global_x,
        global_y_centers_m=global_y,
        n_trials=100,
        distance_tolerance_m=0.1,
        z_residual_tolerance_m=0.1,
        heading_measurement_deg=20.0,
        minimum_cluster_size=1,
        seed=2,
    )

    assert result is not None
    assert np.allclose(result["R"], expected_rotation, atol=1e-10)
    assert np.allclose(result["t"], expected_xy, atol=1e-10)
    assert np.isclose(result["tz"], expected_z, atol=1e-10)
    assert result["correspondence_count"] == 3


def test_covariance_gate_is_not_clipped_by_lookup_tolerance() -> None:
    local = np.array(((0.0, 0.0), (20.0, 0.0), (5.0, 30.0)))
    global_points = local.copy()
    global_points[1, 0] += 1.0
    covariance = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    hypotheses = generate_hypotheses(
        local,
        global_points,
        distance_tolerance_m=0.1,
        control_rms_tolerance_m=2.0,
        local_covariances=covariance,
        global_covariances=covariance,
        sigma_multiplier=2.0,
    )
    assert hypotheses


def test_feature_expansion_is_covariance_gated_and_one_to_one() -> None:
    local = np.array(
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.4, 0.0, 0.0))
    )
    global_points = np.array(((5.0, 2.0, 1.0), (15.2, 2.0, 1.0)))
    local_covariance = np.repeat((0.25 * np.eye(3))[None, :, :], 3, axis=0)
    global_covariance = np.repeat((0.25 * np.eye(3))[None, :, :], 2, axis=0)
    result = feature_consensus(
        rotation=np.eye(2),
        translation_xy_m=np.array((5.0, 2.0)),
        translation_z_m=1.0,
        local_features_xyz=local,
        global_features_xyz=global_points,
        xy_tolerance_m=0.01,
        z_tolerance_m=0.01,
        local_covariances=local_covariance,
        global_covariances=global_covariance,
        sigma_multiplier=2.0,
    )
    assert result["count"] == 2
    assert len(np.unique(result["global_indices"])) == 2

"""Synthetic tests for covariance-aware exhaustive RANSAC."""

import numpy as np

from matching.ransac import exhaustive_ransac


def _rotation(yaw_deg: float) -> np.ndarray:
    yaw = np.radians(yaw_deg)
    return np.array(
        ((np.cos(yaw), -np.sin(yaw)), (np.sin(yaw), np.cos(yaw)))
    )


def _problem() -> dict[str, np.ndarray]:
    local_xy = np.array(
        ((-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0), (0.0, 18.0))
    )
    rotation = _rotation(25.0)
    translation = np.array((40.0, -30.0))
    local_xyz = np.column_stack((local_xy, np.full(5, -2.0)))
    global_xyz = np.column_stack(
        (local_xy @ rotation.T + translation, np.zeros(5))
    )
    global_xyz[4, :2] += np.array((35.0, -20.0))
    covariance = np.repeat((0.25 * np.eye(3))[None, :, :], 5, axis=0)
    local_x = np.arange(-20.0, 25.0, 5.0)
    local_y = np.arange(20.0, -25.0, -5.0)
    global_x = np.arange(-100.0, 105.0, 5.0)
    global_y = np.arange(100.0, -105.0, -5.0)
    return {
        "local_xyz": local_xyz,
        "global_xyz": global_xyz,
        "covariance": covariance,
        "local_grid": np.full((len(local_y), len(local_x)), -2.0),
        "global_dem": np.zeros((len(global_y), len(global_x))),
        "local_x": local_x,
        "local_y": local_y,
        "global_x": global_x,
        "global_y": global_y,
        "rotation": rotation,
        "translation": translation,
    }


def test_exhaustive_ransac_rejects_correspondence_outlier() -> None:
    problem = _problem()
    result = exhaustive_ransac(
        problem["local_xyz"],
        problem["global_xyz"],
        problem["covariance"],
        problem["covariance"],
        np.arange(5),
        np.arange(5),
        problem["local_grid"],
        problem["global_dem"],
        problem["local_x"],
        problem["local_y"],
        problem["global_x"],
        problem["global_y"],
    )
    assert result["status"] == "solution"
    assert result["inlier_count"] == 4
    assert result["outlier_local_indices"].tolist() == [4]
    assert np.allclose(result["rotation"], problem["rotation"])
    assert np.allclose(result["translation_xy_m"], problem["translation"])
    assert np.isclose(result["translation_z_m"], 2.0)


def test_three_correspondences_are_preserved_but_marked_unverified() -> None:
    problem = _problem()
    result = exhaustive_ransac(
        problem["local_xyz"][:3],
        problem["global_xyz"][:3],
        problem["covariance"][:3],
        problem["covariance"][:3],
        np.arange(3),
        np.arange(3),
        problem["local_grid"],
        problem["global_dem"],
        problem["local_x"],
        problem["local_y"],
        problem["global_x"],
        problem["global_y"],
    )
    assert result["status"] == "minimal_unverified"
    assert result["inlier_count"] == 3


def test_too_few_correspondences_return_explicit_status() -> None:
    problem = _problem()
    result = exhaustive_ransac(
        problem["local_xyz"],
        problem["global_xyz"],
        problem["covariance"],
        problem["covariance"],
        np.arange(2),
        np.arange(2),
        problem["local_grid"],
        problem["global_dem"],
        problem["local_x"],
        problem["local_y"],
        problem["global_x"],
        problem["global_y"],
    )
    assert result["status"] == "insufficient_correspondences"

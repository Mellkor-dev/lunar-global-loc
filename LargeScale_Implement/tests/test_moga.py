"""Synthetic regression test for 2.5-D MOGA."""

import numpy as np

from refinement.moga import (
    FeatureObservation,
    MogaProblem,
    solve_moga,
    yaw_rotation,
)


def test_moga_recovers_connected_pose_and_landmark_chain() -> None:
    true_poses = np.array(
        ((0.0, 0.0, 0.0), (10.0, 0.0, 0.05), (20.0, 2.0, 0.10))
    )
    global_landmarks = np.array(((5.0, 12.0), (16.0, -8.0), (27.0, 9.0)))
    initial_poses = true_poses + np.array(
        ((2.0, -1.0, 0.02), (2.5, -1.5, -0.01), (3.0, -2.0, 0.01))
    )
    observations = []
    for pose_index, pose in enumerate(true_poses):
        for landmark_index, landmark in enumerate(global_landmarks):
            local = yaw_rotation(pose[2]).T @ (landmark - pose[:2])
            observations.append(
                FeatureObservation(
                    pose_index=pose_index,
                    landmark_index=landmark_index,
                    local_xy_m=local,
                    local_covariance_xy_m2=0.04 * np.eye(2),
                )
            )
    problem = MogaProblem(
        site_numbers=np.array((1, 2, 3)),
        initial_poses=initial_poses,
        heading_measurements_rad=true_poses[:, 2],
        landmark_global_indices=np.array((10, 20, 30)),
        initial_landmarks_xy_m=global_landmarks.copy(),
        landmark_global_covariances_xy_m2=np.repeat(
            (0.25 * np.eye(2))[None, :, :], 3, axis=0
        ),
        observations=tuple(observations),
    )
    result = solve_moga(
        problem,
        heading_sigma_deg=0.5,
    )
    assert result["success"]
    assert result["final_cost"] < result["initial_cost"]
    assert np.allclose(result["poses"], true_poses, atol=0.2)
    assert np.allclose(result["landmarks_xy_m"], global_landmarks, atol=0.2)
    assert result["pose_covariances"].shape == (3, 3, 3)

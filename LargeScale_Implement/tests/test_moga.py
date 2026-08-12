"""Synthetic regression test for 2.5-D MOGA."""

import numpy as np

from refinement.moga import (
    FeatureObservation,
    MogaProblem,
    solve_moga,
    solve_single_frame_moga,
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


def _single_frame_problem(
    *, initial_pose: np.ndarray, true_pose: np.ndarray, site: int
) -> MogaProblem:
    landmarks = np.array(((5.0, 12.0), (16.0, -8.0), (27.0, 9.0)))
    observations = tuple(
        FeatureObservation(
            pose_index=0,
            landmark_index=index,
            local_xy_m=yaw_rotation(true_pose[2]).T @ (landmark - true_pose[:2]),
            local_covariance_xy_m2=0.04 * np.eye(2),
        )
        for index, landmark in enumerate(landmarks)
    )
    return MogaProblem(
        site_numbers=np.array((site,)),
        initial_poses=initial_pose[None, :],
        heading_measurements_rad=np.array((true_pose[2],)),
        landmark_global_indices=np.array((10, 20, 30)),
        initial_landmarks_xy_m=landmarks,
        landmark_global_covariances_xy_m2=np.repeat(
            (0.25 * np.eye(2))[None, :, :], 3, axis=0
        ),
        observations=observations,
    )


def test_single_frame_moga_is_independent_of_another_site() -> None:
    true_pose = np.array((0.0, 0.0, 0.0))
    first = _single_frame_problem(
        initial_pose=np.array((2.0, -1.0, 0.02)), true_pose=true_pose, site=1
    )
    unrelated = _single_frame_problem(
        initial_pose=np.array((110.0, 90.0, 0.5)),
        true_pose=np.array((100.0, 100.0, 0.5)),
        site=2,
    )

    first_result = solve_single_frame_moga(first, heading_sigma_deg=0.5)
    solve_single_frame_moga(unrelated, heading_sigma_deg=0.5)
    repeated_result = solve_single_frame_moga(first, heading_sigma_deg=0.5)

    assert np.array_equal(first_result["poses"], repeated_result["poses"])
    assert np.array_equal(
        first_result["landmarks_xy_m"], repeated_result["landmarks_xy_m"]
    )


def test_single_frame_moga_rejects_multiple_pose_problem() -> None:
    true_pose = np.array((0.0, 0.0, 0.0))
    problem = _single_frame_problem(
        initial_pose=np.array((2.0, -1.0, 0.02)), true_pose=true_pose, site=1
    )
    invalid = MogaProblem(
        site_numbers=np.array((1, 2)),
        initial_poses=np.vstack((problem.initial_poses, problem.initial_poses)),
        heading_measurements_rad=np.array((0.0, 0.0)),
        landmark_global_indices=problem.landmark_global_indices,
        initial_landmarks_xy_m=problem.initial_landmarks_xy_m,
        landmark_global_covariances_xy_m2=problem.landmark_global_covariances_xy_m2,
        observations=problem.observations,
    )

    with np.testing.assert_raises_regex(ValueError, "exactly one pose"):
        solve_single_frame_moga(invalid)

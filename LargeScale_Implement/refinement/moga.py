"""Multi-frame odometry-compensated global alignment in 2.5-D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_rotation(yaw_rad: float) -> np.ndarray:
    cosine = np.cos(yaw_rad)
    sine = np.sin(yaw_rad)
    return np.array(((cosine, -sine), (sine, cosine)), dtype=np.float64)


@dataclass(frozen=True)
class FeatureObservation:
    pose_index: int
    landmark_index: int
    local_xy_m: np.ndarray
    local_covariance_xy_m2: np.ndarray


@dataclass(frozen=True)
class MogaProblem:
    site_numbers: np.ndarray
    initial_poses: np.ndarray
    heading_measurements_rad: np.ndarray
    landmark_global_indices: np.ndarray
    initial_landmarks_xy_m: np.ndarray
    landmark_global_covariances_xy_m2: np.ndarray
    observations: tuple[FeatureObservation, ...]

    @property
    def pose_count(self) -> int:
        return len(self.site_numbers)

    @property
    def landmark_count(self) -> int:
        return len(self.landmark_global_indices)


def pack_state(poses: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    return np.concatenate((np.asarray(poses).ravel(), np.asarray(landmarks).ravel()))


def unpack_state(
    state: np.ndarray,
    pose_count: int,
    landmark_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state, dtype=np.float64)
    pose_values = 3 * pose_count
    expected = pose_values + 2 * landmark_count
    if state.shape != (expected,):
        raise ValueError(f"State has shape {state.shape}, expected ({expected},)")
    return (
        state[:pose_values].reshape(pose_count, 3),
        state[pose_values:].reshape(landmark_count, 2),
    )


def _whiten(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    try:
        factor = np.linalg.cholesky(covariance)
        return np.linalg.solve(factor, residual)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        inverse_sqrt = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
        return inverse_sqrt @ residual


def residual_vector(
    state: np.ndarray,
    problem: MogaProblem,
    *,
    heading_sigma_rad: float,
) -> np.ndarray:
    poses, landmarks = unpack_state(
        state, problem.pose_count, problem.landmark_count
    )
    residuals: list[np.ndarray] = []

    for landmark_index in range(problem.landmark_count):
        residuals.append(
            _whiten(
                landmarks[landmark_index] - problem.initial_landmarks_xy_m[landmark_index],
                problem.landmark_global_covariances_xy_m2[landmark_index],
            )
        )

    for observation in problem.observations:
        pose = poses[observation.pose_index]
        rotation = yaw_rotation(pose[2])
        predicted_landmark = rotation @ observation.local_xy_m + pose[:2]
        covariance_world = (
            rotation
            @ observation.local_covariance_xy_m2
            @ rotation.T
        )
        residuals.append(
            _whiten(
                predicted_landmark - landmarks[observation.landmark_index],
                covariance_world,
            )
        )

    heading_residual = wrap_angle(poses[:, 2] - problem.heading_measurements_rad)
    residuals.append(np.asarray(heading_residual) / heading_sigma_rad)
    return np.concatenate(residuals)


def solve_moga(
    problem: MogaProblem,
    *,
    heading_sigma_deg: float = 1.0,
    maximum_function_evaluations: int = 300,
    relative_tolerance: float = 1e-10,
) -> dict[str, object]:
    """Jointly optimize poses and shared landmark positions."""
    if problem.pose_count == 0:
        raise ValueError("MOGA requires at least one pose")
    if problem.landmark_count == 0:
        raise ValueError("MOGA requires at least one globally anchored landmark")
    if len(problem.observations) < 3:
        raise ValueError("MOGA requires at least three feature observations")
    if heading_sigma_deg <= 0.0:
        raise ValueError("heading sigma must be positive")

    initial_state = pack_state(
        problem.initial_poses, problem.initial_landmarks_xy_m
    )
    heading_sigma_rad = np.radians(heading_sigma_deg)

    def residual(state: np.ndarray) -> np.ndarray:
        return residual_vector(
            state,
            problem,
            heading_sigma_rad=heading_sigma_rad,
        )

    initial_residual = residual(initial_state)
    solution = least_squares(
        residual,
        initial_state,
        method="trf",
        loss="linear",
        max_nfev=maximum_function_evaluations,
        ftol=relative_tolerance,
        xtol=relative_tolerance,
        gtol=relative_tolerance,
    )
    optimized_poses, optimized_landmarks = unpack_state(
        solution.x, problem.pose_count, problem.landmark_count
    )
    optimized_poses[:, 2] = wrap_angle(optimized_poses[:, 2])

    information = solution.jac.T @ solution.jac
    posterior_covariance = np.linalg.pinv(information)
    pose_covariances = np.empty((problem.pose_count, 3, 3), dtype=np.float64)
    for pose_index in range(problem.pose_count):
        start = 3 * pose_index
        pose_covariances[pose_index] = posterior_covariance[
            start : start + 3, start : start + 3
        ]

    return {
        "success": bool(solution.success),
        "message": str(solution.message),
        "status_code": int(solution.status),
        "function_evaluations": int(solution.nfev),
        "initial_cost": float(0.5 * initial_residual @ initial_residual),
        "final_cost": float(solution.cost),
        "optimality": float(solution.optimality),
        "poses": optimized_poses,
        "landmarks_xy_m": optimized_landmarks,
        "pose_covariances": pose_covariances,
        "posterior_covariance": posterior_covariance,
    }

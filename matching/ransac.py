"""
RANSAC outlier rejectio sits between DARCES and MOGA.

 This consensus set — not DARCES's original 3 points — is what gets fed to MOGA.
"""

import numpy as np
from scipy.spatial import cKDTree


from matching.darces import rigid_transform_2d


def expand_correspondences(
    local_xy: np.ndarray,
    global_xy: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    match_threshold: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    
    local_xy = np.asarray(local_xy, dtype=float)
    global_xy = np.asarray(global_xy, dtype=float)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)

    if local_xy.ndim != 2 or local_xy.shape[1] != 2:
        raise ValueError("local_xy must have shape (N, 2)")

    if global_xy.ndim != 2 or global_xy.shape[1] != 2:
        raise ValueError("global_xy must have shape (M, 2)")

    if R.shape != (2, 2):
        raise ValueError("R must have shape (2, 2)")

    if t.shape != (2,):
        raise ValueError("t must have shape (2,)")

    if match_threshold <= 0.0:
        raise ValueError("match_threshold must be positive")

    if len(local_xy) == 0 or len(global_xy) == 0:
        return []

    projected_local = (R @ local_xy.T).T + t

    global_tree = cKDTree(global_xy)
    local_to_global_distance, local_to_global_index = global_tree.query(
        projected_local,
        k=1,
    )

    projected_tree = cKDTree(projected_local)
    _, global_to_local_index = projected_tree.query(global_xy, k=1)

    matches: list[tuple[np.ndarray, np.ndarray]] = []

    for local_index, (distance, global_index) in enumerate(
        zip(local_to_global_distance, local_to_global_index)
    ):
        global_index = int(global_index)

        is_mutual = (
            int(global_to_local_index[global_index]) == local_index
        )

        if distance <= match_threshold and is_mutual:
            matches.append(
                (local_xy[local_index].copy(), global_xy[global_index].copy())
            )

    return matches


def ransac_refine(candidate_pairs, n_iterations=200, inlier_threshold=0.5, seed=None):
    
    rng = np.random.default_rng(seed)
    n = len(candidate_pairs)
    if n < 3:
        return candidate_pairs, None, None

    local_pts = np.array([p[0] for p in candidate_pairs])
    global_pts = np.array([p[1] for p in candidate_pairs])

    best_inliers_idx = []
    best_R, best_t = None, None

    for _ in range(n_iterations):
        sample_idx = rng.choice(n, size=3, replace=False)
        try:
            R, t = rigid_transform_2d(local_pts[sample_idx], global_pts[sample_idx])
        except Exception:
            continue

        predicted = (R @ local_pts.T).T + t
        residuals = np.linalg.norm(predicted - global_pts, axis=1)
        inliers_idx = np.where(residuals < inlier_threshold)[0]

        if len(inliers_idx) > len(best_inliers_idx):
            best_inliers_idx = inliers_idx
            best_R, best_t = R, t

    if len(best_inliers_idx) >= 3:
        
        R_final, t_final = rigid_transform_2d(
            local_pts[best_inliers_idx], global_pts[best_inliers_idx]
        )
        inlier_pairs = [candidate_pairs[i] for i in best_inliers_idx]
        return inlier_pairs, R_final, t_final

    return candidate_pairs, best_R, best_t  # RANSAC found nothing better

"""
RANSAC outlier rejection (Sec 6.8) — sits between DARCES and MOGA.

DARCES gives a transform from only 3 control points. This stage uses that
transform to project ALL detected local features into the global frame,
finds nearest-neighbor candidate matches against ALL global features, then
runs RANSAC over this expanded candidate set to find the largest group of
mutually-consistent correspondences (the "consensus set"). This consensus
set — not DARCES's original 3 points — is what gets fed to MOGA.
"""

import numpy as np
from scipy.spatial import cKDTree
from matching.darces import rigid_transform_2d


def expand_correspondences(local_xy, global_xy, R, t, match_threshold):
    """
    Project all local features into the global frame using DARCES's (R,t),
    then nearest-neighbor match against all global features. Returns list
    of (local_xy_i, global_xy_i) candidate pairs within match_threshold.
    """
    projected = (R @ local_xy.T).T + t
    tree = cKDTree(global_xy)
    dists, idx = tree.query(projected)

    pairs = []
    for i, (d, j) in enumerate(zip(dists, idx)):
        if d < match_threshold:
            pairs.append((local_xy[i], global_xy[j]))
    return pairs


def ransac_refine(candidate_pairs, n_iterations=200, inlier_threshold=0.5, seed=None):
    """
    RANSAC: repeatedly sample 3 candidate pairs, fit a rigid transform,
    count how many OTHER candidate pairs agree with it (inliers), keep the
    largest/best consensus set. Returns (best_inlier_pairs, R, t).
    """
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
        # Refit transform using the FULL consensus set, not just the seed 3
        R_final, t_final = rigid_transform_2d(
            local_pts[best_inliers_idx], global_pts[best_inliers_idx]
        )
        inlier_pairs = [candidate_pairs[i] for i in best_inliers_idx]
        return inlier_pairs, R_final, t_final

    return candidate_pairs, best_R, best_t  # fallback: RANSAC found nothing better

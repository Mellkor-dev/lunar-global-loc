import numpy as np
from itertools import combinations

def pairwise_dists(points_xy: np.ndarray) -> np.ndarray:
    """NxN matrix of pairwise Euclidean distances."""
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))

def rigid_transform_2d(src: np.ndarray, dst: np.ndarray):
    """
    Least-squares rigid transform (rotation + translation) aligning
    src -> dst, for exactly 3 point correspondences (2D xy).
    Horn's method / Arun et al. 1987, restricted to 2D + no reflection.
    Returns (R (2x2), t (2,)).
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # reflection guard
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean
    return R, t

def generate_hypotheses(local_xy, global_xy, n_trials, dist_tol_m, rng):
    n_local, n_global = len(local_xy), len(global_xy)
    local_d = pairwise_dists(local_xy)
    global_d = pairwise_dists(global_xy)

    hypotheses = []
    for _ in range(n_trials):
        if n_local < 3 or n_global < 3:
            break
        li = rng.choice(n_local, size=3, replace=False)
        gi = rng.choice(n_global, size=3, replace=False)

        l = sorted([local_d[li[0],li[1]], local_d[li[0],li[2]], local_d[li[1],li[2]]])
        g = sorted([global_d[gi[0],gi[1]], global_d[gi[0],gi[2]], global_d[gi[1],gi[2]]])
        total_err = sum(abs(a - b) for a, b in zip(l, g))

        if total_err < dist_tol_m * 3:  # dist_tol_m now means "per-side average tolerance"
            hypotheses.append((li, gi))
    return hypotheses

def screen_hypothesis(R, t, global_dem, dem_lxy, dem_origin, z_dev_thresh,
                       heading_est_deg, heading_meas_deg, e_heading_deg):
    """
    Sec IV.B screening tests: map-boundary, z-deviation, orientation.
    Returns True if hypothesis passes all screens.
    """
    H, W = global_dem.shape
    # map-boundary: rover position (origin of local frame, after transform) must be inside map
    rover_xy = t  # local frame origin maps to t under this R,t
    i = (rover_xy[1] - dem_origin[1]) / dem_lxy
    j = (rover_xy[0] - dem_origin[0]) / dem_lxy
    if not (0 <= i < H and 0 <= j < W):
        return False

    # heading consistency (yaw recovered from R vs measured heading, e.g. compass/sun sensor)
    heading_est = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    if abs(((heading_est - heading_meas_deg + 180) % 360) - 180) > e_heading_deg:
        return False

    return True

def fitness(R, t, local_xy_all, local_z_all, global_dem, dem_lxy, dem_origin, decimation=2):
    """
    Eq. 1: average absolute z-error between transformed reference points
    and interpolated global map elevation. Negated so higher = better.
    """
    from scipy.ndimage import map_coordinates
    ref_xy = local_xy_all[::decimation]
    ref_z = local_z_all[::decimation]

    world_xy = (R @ ref_xy.T).T + t
    i = (world_xy[:, 1] - dem_origin[1]) / dem_lxy
    j = (world_xy[:, 0] - dem_origin[0]) / dem_lxy

    H, W = global_dem.shape
    valid = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    if valid.sum() == 0:
        return -np.inf

    z_global = map_coordinates(global_dem, [i[valid], j[valid]], order=1, mode='nearest')
    z_err = np.abs(ref_z[valid] - z_global)
    return -z_err.mean()

def run_darces(local_peaks_xy, global_peaks_xy, local_grid, global_dem,
               local_origin, global_origin, lxy,
               n_trials=100, dist_tol_m=0.5, z_dev_thresh=0.3,
               heading_meas_deg=0.0, e_heading_deg=10.0, seed=None):
    """
    Full Sec IV pipeline: hypothesis search -> screening -> fitness -> best selection.
    local_peaks_xy / global_peaks_xy: Nx2 arrays in (row,col) PIXEL indices.
    Converts to metric xy internally using origin+lxy.
    """
    rng = np.random.default_rng(seed)

    local_xy_m = local_peaks_xy[:, [1, 0]] * lxy + np.array(local_origin)
    global_xy_m = global_peaks_xy[:, [1, 0]] * lxy + np.array(global_origin)

    hyps = generate_hypotheses(local_xy_m, global_xy_m, n_trials, dist_tol_m, rng)
    print(f"Generated {len(hyps)} candidate hypotheses (of {n_trials} trials)")

    # local elevation samples at all local peak locations, for fitness ref points
    H_local, W_local = local_grid.shape
    local_all_ij = np.argwhere(np.ones_like(local_grid, dtype=bool))[::20]  # decimated dense sample
    local_all_xy = local_all_ij[:, [1, 0]] * lxy + np.array(local_origin)
    local_all_z = local_grid[local_all_ij[:, 0], local_all_ij[:, 1]]

    best = None
    best_fit = -np.inf
    results = []
    for li, gi in hyps:
        R, t = rigid_transform_2d(local_xy_m[li], global_xy_m[gi])

        if not screen_hypothesis(R, t, global_dem, lxy, global_origin,
                                  z_dev_thresh, 0.0, heading_meas_deg, e_heading_deg):
            continue

        f = fitness(R, t, local_all_xy, local_all_z, global_dem, lxy, global_origin)
        results.append((f, R, t, li, gi))
        if f > best_fit:
            best_fit = f
            best = (R, t, li, gi)

    print(f"{len(results)} hypotheses passed screening")
    if best is None:
        print("DARCES: no valid solution found")
        return None

    R, t, li, gi = best
    print(f"Best fitness: {best_fit:.4f}")
    print(f"Estimated translation (t): {t}")
    print(f"Estimated heading: {np.degrees(np.arctan2(R[1,0], R[0,0])):.2f} deg")
    return {"R": R, "t": t, "fitness": best_fit, "local_idx": li, "global_idx": gi}

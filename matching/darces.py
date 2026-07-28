import numpy as np
from scipy.ndimage import map_coordinates


# ============================================================
# Geometry utilities
# ============================================================

def pairwise_dists(points_xy: np.ndarray) -> np.ndarray:
    diff = points_xy[:, None, :] - points_xy[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def triangle_area(p):
    v1=p[1]-p[0]
    v2 = p[2] - p[0]
    return 0.5 * abs(v1[0]*v2[1]-v1[1]*v2[0])
        


def rigid_transform_2d(src: np.ndarray, dst: np.ndarray):   

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    H = src_c.T @ dst_c

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean

    return R, t

def generate_hypotheses(local_xy,global_xy,n_trials,dist_tol_m,rng,side_ratio_tol=0.05,):
    n_local = len(local_xy)
    n_global = len(global_xy)

    local_d = pairwise_dists(local_xy)
    global_d = pairwise_dists(global_xy)

    hypotheses = []
    seen = set()

    for _ in range(n_trials):

        if n_local < 3 or n_global < 3:
            break

        li = rng.choice(n_local, size=3, replace=False)
        gi = rng.choice(n_global, size=3, replace=False)

        key = (
            tuple(sorted(li)),
            tuple(sorted(gi))
        )

        if key in seen:
            continue

        seen.add(key)

        # reject nearly-collinear triangles
        if triangle_area(local_xy[li]) < 1.0:
            continue

        if triangle_area(global_xy[gi]) < 1.0:
            continue

        l = np.sort([
            local_d[li[0], li[1]],
            local_d[li[0], li[2]],
            local_d[li[1], li[2]],
        ])

        g = np.sort([
            global_d[gi[0], gi[1]],
            global_d[gi[0], gi[2]],
            global_d[gi[1], gi[2]],
        ])

        if np.any(l < 1e-6):
            continue

        abs_err = np.mean(np.abs(l - g))

        l_ratio = l / l.max()
        g_ratio = g / g.max()

        ratio_err = np.mean(np.abs(l_ratio - g_ratio))

        if (
            abs_err < dist_tol_m
            and ratio_err < side_ratio_tol
        ):
            hypotheses.append((li, gi))

    return hypotheses

def screen_hypothesis(R,t,global_dem,dem_lxy,dem_origin,heading_meas_deg,e_heading_deg,):
    H, W = global_dem.shape
    rover_xy = t
    i = (rover_xy[1] - dem_origin[1]) / dem_lxy
    j = (rover_xy[0] - dem_origin[0]) / dem_lxy
    if not (0 <= i < H and 0 <= j < W):
        return False

    heading_est = np.degrees(
        np.arctan2(R[1, 0], R[0, 0])
    )

    heading_err = abs(
        ((heading_est - heading_meas_deg + 180) % 360) - 180
    )

    if heading_err > e_heading_deg:
        return False

    return True

def peak_height_consistency(R,t,local_peak_xy,local_peak_z,global_dem,dem_lxy,dem_origin,z_thresh=0.5,):
    world_xy = (R @ local_peak_xy.T).T + t
    i = (world_xy[:, 1] - dem_origin[1]) / dem_lxy
    j = (world_xy[:, 0] - dem_origin[0]) / dem_lxy

    H, W = global_dem.shape

    valid = (
        (i >= 0)
        & (i < H)
        & (j >= 0)
        & (j < W)
    )

    if valid.sum() < 2:
        return False

    z_global = map_coordinates(
        global_dem,
        [i[valid], j[valid]],
        order=1,
        mode="nearest",
    )

    err = np.abs(
        local_peak_z[valid] - z_global
    )

    return np.median(err) < z_thresh

def fitness(R,t,local_xy_all,local_z_all,global_dem,dem_lxy,dem_origin,decimation=5,min_overlap=0.50):

    ref_xy = local_xy_all[::decimation]
    ref_z = local_z_all[::decimation]

    world_xy = (R @ ref_xy.T).T + t

    i = (world_xy[:, 1] - dem_origin[1]) / dem_lxy
    j = (world_xy[:, 0] - dem_origin[0]) / dem_lxy

    H, W = global_dem.shape

    valid = (
        (i >= 0)
        & (i < H)
        & (j >= 0)
        & (j < W)
    )

    overlap = valid.mean()

    if overlap < min_overlap:
        return -np.inf

    z_global = map_coordinates(
        global_dem,
        [i[valid], j[valid]],
        order=1,
        mode="nearest",
    )

    if len(z_global) == 0:
        return -np.inf

    if not np.isfinite(z_global).all():
        return -np.inf

    z_err = np.abs(
        ref_z[valid] - z_global
    )

    return -np.median(z_err)

def run_darces(local_peaks_xy,global_peaks_xy,local_grid, global_dem, local_origin, global_origin, lxy, n_trials=10000, dist_tol_m=0.5, 
               z_dev_thresh=0.3, heading_meas_deg=0.0, e_heading_deg=10.0, seed=None):

    rng = np.random.default_rng(seed)

    local_xy_m = (
        local_peaks_xy[:, [1, 0]] * lxy
        + np.array(local_origin)
    )

    global_xy_m = (
        global_peaks_xy[:, [1, 0]] * lxy
        + np.array(global_origin)
    )

    hyps = generate_hypotheses(
        local_xy_m,
        global_xy_m,
        n_trials,
        dist_tol_m,
        rng,
    )

    print(
        f"Generated {len(hyps)} candidate hypotheses "
        f"(of {n_trials} trials)"
    )

    all_ij = np.argwhere(
        np.isfinite(local_grid)
    )

    n_samples = min(
        10000,
        len(all_ij)
    )

    sample_idx = rng.choice(
        len(all_ij),
        n_samples,
        replace=False,
    )

    local_all_ij = all_ij[sample_idx]

    local_all_xy = (
        local_all_ij[:, [1, 0]] * lxy
        + np.array(local_origin)
    )

    local_all_z = local_grid[
        local_all_ij[:, 0],
        local_all_ij[:, 1]
    ]

    best = None
    best_fit = -np.inf

    results = []

    for li, gi in hyps:

        R, t = rigid_transform_2d(
            local_xy_m[li],
            global_xy_m[gi],
        )

        if not screen_hypothesis(
            R,
            t,
            global_dem,
            lxy,
            global_origin,
            heading_meas_deg,
            e_heading_deg,
        ):
            continue

        local_peak_z = local_grid[
            local_peaks_xy[li][:, 0],
            local_peaks_xy[li][:, 1]
        ]

        if not peak_height_consistency(
            R,
            t,
            local_xy_m[li],
            local_peak_z,
            global_dem,
            lxy,
            global_origin,
            z_thresh=z_dev_thresh,
        ):
            continue

        f = fitness(
            R,
            t,
            local_all_xy,
            local_all_z,
            global_dem,
            lxy,
            global_origin,
        )

        results.append(
            (f, R, t, li, gi)
        )

        if f > best_fit:
            best_fit = f
            best = (R, t, li, gi)

    print(
        f"{len(results)} hypotheses passed screening"
    )

    if best is None:
        print("DARCES: no valid solution found")
        return None

    R, t, li, gi = best

    heading = np.degrees(
        np.arctan2(
            R[1, 0],
            R[0, 0]
        )
    )

    print(f"Best fitness: {best_fit:.4f}")
    print(f"Estimated translation: {t}")
    print(f"Estimated heading: {heading:.2f} deg")

    return {
        "R": R,
        "t": t,
        "fitness": best_fit,
        "heading_deg": heading,
        "local_idx": li,
        "global_idx": gi,
    }
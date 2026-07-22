import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maps.local_map import quat_to_rotmat, level_points, grid_local_map, level_points_tilt_only
from features.dilation_detector import detect_craters
from matching.darces import run_darces
from sim.traverse_recorder import RAW_ODOM, SPAWN_XYZ
from matching.ransac import expand_correspondences, ransac_refine

VLP16_OFFSET = np.array([-0.14999991655349731, -2.3034954210743308e-07, 0.41500020027160645])
LXY = 0.025

global_dem = np.load("maps/data/global_dem.npy")
global_peaks = detect_craters(global_dem, n=80, flatness_eps=0.0065)
print(f"Global peaks: {len(global_peaks)}\n")

results = {}
for site in sorted(RAW_ODOM.keys()):
    print(f"=== {site} ===")
    pos, quat = RAW_ODOM[site]
    
    world_pos = SPAWN_XYZ + np.array(pos)

    pts_struct = np.load(f"sim/data/traverse/local_scan_{site}.npy")
    pts_local = np.stack([pts_struct['x'], pts_struct['y'], pts_struct['z']], axis=1)
    pts_base_link = pts_local + VLP16_OFFSET
    
    R = quat_to_rotmat(*quat)
    true_yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pts_leveled = level_points_tilt_only(pts_base_link, R)
    z_offset = -np.median(pts_leveled[:, 2])
    pts_leveled[:, 2] += z_offset

    local_grid, local_origin, lxy = grid_local_map(pts_leveled, LXY)
    local_peaks = detect_craters(local_grid, n=80, flatness_eps=0.0065)
    print(f"  grid shape: {local_grid.shape}, local peaks: {len(local_peaks)}")

    result = run_darces(
        local_peaks, global_peaks, local_grid, global_dem,
        local_origin, (0.0, 0.0), lxy,
        n_trials=5000, dist_tol_m=0.5, z_dev_thresh=0.2,
        heading_meas_deg=0.0, e_heading_deg=180.0, seed=42
    )

    if result is not None:
        err = np.linalg.norm(result['t'] - world_pos[:2])
        print(f"  DARCES estimate: t={result['t']}, fitness={result['fitness']:.4f}")
        print(f"  Ground truth: {world_pos[:2]}, error={err:.2f}m\n")

        
        local_xy_m = local_peaks[:, [1, 0]] * lxy + np.array(local_origin)
        global_xy_m = global_peaks[:, [1, 0]] * lxy + np.array((0.0, 0.0))

        
        candidates = expand_correspondences(
            local_xy_m, global_xy_m, result["R"], result["t"], match_threshold=2.5
        )
        matched_pairs, ransac_R, ransac_t = ransac_refine(
            candidates, n_iterations=200, inlier_threshold=0.5, seed=42
        )
        print(f"  RANSAC: {len(candidates)} candidates -> {len(matched_pairs)} inliers")

        if ransac_t is not None:  #be careful this is wrong
           
            result["t"] = ransac_t
            result["R"] = ransac_R
            err = np.linalg.norm(result['t'] - world_pos[:2])
            print(f"  RANSAC-refined estimate: t={result['t']}, error={err:.2f}m")

        results[site] = {
            "local_grid": local_grid, "local_peaks": local_peaks,
            "local_origin": local_origin, "lxy": lxy,
            "darces_R": result["R"], "darces_t": result["t"],
            "darces_fitness": result["fitness"],
            "ground_truth_xy": world_pos[:2],
            "error_m": err,
            "matched_pairs": matched_pairs,
            "true_yaw_deg": true_yaw,
        }
    else:
        print("  DARCES: no solution\n")
        results[site] = None

np.save("maps/data/traverse_darces_results.npy", results, allow_pickle=True)
print("Saved: maps/data/traverse_darces_results.npy")

print("\n     Summary      ")
for site, r in results.items():
    if r is not None:
        print(f"{site}: fitness={r['darces_fitness']:.4f}, error={r['error_m']:.2f}m")
    else:
        print(f"{site}: no DARCES solution")

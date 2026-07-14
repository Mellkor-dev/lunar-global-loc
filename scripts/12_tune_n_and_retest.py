# scripts/12_tune_n_and_retest.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features.dilation_detector import detect_craters
from matching.darces import pairwise_dists, rigid_transform_2d, screen_hypothesis, fitness, generate_hypotheses

global_dem = np.load("maps/data/global_dem.npy")
local_data = np.load("maps/data/local_features.npz")
local_grid = local_data["local_grid"]
local_origin = tuple(local_data["origin_xy"])
lxy = float(local_data["lxy"])

for n in [40, 60, 80, 100]:
    global_peaks = detect_craters(global_dem, n=n, flatness_eps=0.007)
    local_peaks = detect_craters(local_grid, n=n, flatness_eps=0.0065)
    print(f"\n=== n={n} (D_detect={n*lxy:.2f}m) ===")
    print(f"Global peaks: {len(global_peaks)}, Local peaks: {len(local_peaks)}")

    if len(global_peaks) < 3 or len(local_peaks) < 3:
        print("Too few peaks, skipping.")
        continue

    local_xy_m = local_peaks[:, [1, 0]] * lxy + np.array(local_origin)
    global_xy_m = global_peaks[:, [1, 0]] * lxy + np.array((0.0, 0.0))

    rng = np.random.default_rng(42)
    hyps = generate_hypotheses(local_xy_m, global_xy_m, 5000, 0.5, rng)

    local_all_ij = np.argwhere(np.ones_like(local_grid, dtype=bool))[::10]
    local_all_xy = local_all_ij[:, [1, 0]] * lxy + np.array(local_origin)
    local_all_z = local_grid[local_all_ij[:, 0], local_all_ij[:, 1]]

    results = []
    for li, gi in hyps:
        R, t = rigid_transform_2d(local_xy_m[li], global_xy_m[gi])
        if not screen_hypothesis(R, t, global_dem, lxy, (0.0,0.0), 0.3, 0.0, 0.0, 180.0):
            continue
        f = fitness(R, t, local_all_xy, local_all_z, global_dem, lxy, (0.0,0.0))
        results.append((f, t, np.degrees(np.arctan2(R[1,0], R[0,0]))))

    if not results:
        print("No hypotheses passed screening.")
        continue

    results.sort(key=lambda r: -r[0])
    fits = [r[0] for r in results]
    print(f"{len(results)} hypotheses passed. Fitness range: {min(fits):.4f} to {max(fits):.4f} (spread: {max(fits)-min(fits):.4f})")
    print("Top 3:")
    for f, t, h in results[:3]:
        print(f"  t=({t[0]:.2f},{t[1]:.2f}) heading={h:.1f} fitness={f:.4f}")
    print(f"  [ground truth: t=(9.964, 9.967)]")

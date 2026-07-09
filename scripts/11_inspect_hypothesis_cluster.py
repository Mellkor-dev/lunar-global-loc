# scripts/11_inspect_hypothesis_cluster.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from matching.darces import pairwise_dists, rigid_transform_2d, screen_hypothesis, fitness, generate_hypotheses

global_dem = np.load("maps/data/global_dem.npy")
from features.dilation_detector import detect_craters
global_peaks = detect_craters(global_dem, n=60, flatness_eps=0.005)

local_data = np.load("maps/data/local_features.npz")
local_peaks = local_data["local_peaks"]
local_origin = tuple(local_data["origin_xy"])
lxy = float(local_data["lxy"])
local_grid = local_data["local_grid"]

local_xy_m = local_peaks[:, [1, 0]] * lxy + np.array(local_origin)
global_xy_m = global_peaks[:, [1, 0]] * lxy + np.array((0.0, 0.0))

rng = np.random.default_rng(42)
hyps = generate_hypotheses(local_xy_m, global_xy_m, 5000, 0.5, rng)

local_all_ij = np.argwhere(np.ones_like(local_grid, dtype=bool))[::20]
local_all_xy = local_all_ij[:, [1, 0]] * lxy + np.array(local_origin)
local_all_z = local_grid[local_all_ij[:, 0], local_all_ij[:, 1]]

print(f"{'t_x':>8} {'t_y':>8} {'heading':>8} {'fitness':>9}")
for li, gi in hyps:
    R, t = rigid_transform_2d(local_xy_m[li], global_xy_m[gi])
    if not screen_hypothesis(R, t, global_dem, lxy, (0.0,0.0), 0.3, 0.0, 0.0, 180.0):
        continue
    f = fitness(R, t, local_all_xy, local_all_z, global_dem, lxy, (0.0,0.0))
    heading = np.degrees(np.arctan2(R[1,0], R[0,0]))
    print(f"{t[0]:8.2f} {t[1]:8.2f} {heading:8.1f} {f:9.4f}")

# scripts/10_diagnose_darces_tolerance.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from matching.darces import pairwise_dists
from features.dilation_detector import detect_craters

global_dem = np.load("maps/data/global_dem.npy")
global_peaks = detect_craters(global_dem, n=80, flatness_eps=0.0065)

local_data = np.load("maps/data/local_features.npz")
local_peaks = local_data["local_peaks"]
local_origin = tuple(local_data["origin_xy"])
lxy = float(local_data["lxy"])

local_xy_m = local_peaks[:, [1, 0]] * lxy + np.array(local_origin)
global_xy_m = global_peaks[:, [1, 0]] * lxy + np.array((0.0, 0.0))

local_d = pairwise_dists(local_xy_m)
global_d = pairwise_dists(global_xy_m)

rng = np.random.default_rng(42)
n_trials = 20000
best_err = np.inf
hits = 0
for _ in range(n_trials):
    li = rng.choice(len(local_xy_m), 3, replace=False)
    gi = rng.choice(len(global_xy_m), 3, replace=False)
    l = sorted([local_d[li[0],li[1]], local_d[li[0],li[2]], local_d[li[1],li[2]]])
    g = sorted([global_d[gi[0],gi[1]], global_d[gi[0],gi[2]], global_d[gi[1],gi[2]]])
    err = sum(abs(a-b) for a,b in zip(l,g))
    if err < best_err:
        best_err = err
    if err < 1.5:  # loose tolerance, 0.5m per side roughly
        hits += 1

print(f"Best total pairwise-distance error over {n_trials} trials: {best_err:.3f}")
print(f"Hits within loose tolerance (1.5m total): {hits}")

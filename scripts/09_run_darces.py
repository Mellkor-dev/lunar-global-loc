import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from matching.darces import run_darces

global_dem = np.load("maps/data/global_dem.npy")
global_peaks = np.load("maps/data/global_peaks.npy") if os.path.exists("maps/data/global_peaks.npy") else None

local_data = np.load("maps/data/local_features.npz")
local_grid = local_data["local_grid"]
local_peaks = local_data["local_peaks"]
local_origin = tuple(local_data["origin_xy"])
lxy = float(local_data["lxy"])

# regenerate global peaks fresh (script 02 didn't save them to disk previously)
from features.dilation_detector import detect_craters
global_peaks = detect_craters(global_dem, n=80, flatness_eps=0.0065)
global_origin = (0.0, 0.0)

print(f"Local peaks: {len(local_peaks)}, Global peaks: {len(global_peaks)}")

result = run_darces(
    local_peaks, global_peaks, local_grid, global_dem,
    local_origin, global_origin, lxy,
    n_trials=5000, dist_tol_m=0.5, z_dev_thresh=0.3,
    heading_meas_deg=5.79,
    e_heading_deg=180.0,   # keep disabled for this pass
    seed=42
)

if result is not None:
    # ground truth for comparison: rover world position was (9.964, 9.967)
    print(f"\nGround truth position: (9.964, 9.967)")
    print(f"DARCES estimated position: {result['t']}")
    err = np.linalg.norm(result['t'] - np.array([9.964, 9.967]))
    print(f"Position error: {err:.3f} m")

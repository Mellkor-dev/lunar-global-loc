# scripts/05_match_diagnostics.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from maps.global_dem import GlobalDEM
from features.dilation_detector import detect_craters
from scipy.spatial import cKDTree

dem = np.load("maps/data/global_dem.npy")
craters = np.load("maps/data/craters_data.npy")
lxy = 0.025
gdem = GlobalDEM(elevation=dem, lxy=lxy, origin_xy=(0.0, 0.0))
craters_ij = np.array([gdem.xy_to_ij(c['x'], c['y']) for c in craters])

n = 80
peaks = detect_craters(dem, n=n, flatness_eps=0.0065)
print(f"{len(peaks)} peaks detected")

tree = cKDTree(craters_ij)  # ALL 162 craters, no size filter
dists_px, idx = tree.query(peaks)
dists_m = dists_px * lxy

for thresh in [0.1, 0.25, 0.5, 1.0, 2.0]:
    matched = (dists_m < thresh).sum()
    print(f"  within {thresh}m: {matched}/{len(peaks)}")

print(f"\nClosest 5 distances (m): {np.sort(dists_m)[:5]}")
print(f"Matched crater sizes for closest 5: {craters['size'][idx[np.argsort(dists_m)[:5]]]}")

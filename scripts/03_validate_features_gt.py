import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maps.global_dem import GlobalDEM
from features.dilation_detector import detect_peaks, detect_craters

dem = np.load("maps/data/global_dem.npy")
craters = np.load("maps/data/craters_data.npy")

lxy = 0.025
gdem = GlobalDEM(elevation=dem, lxy=lxy, origin_xy=(0.0, 0.0))

n = 60  # D_detect = 1.5m, matches largest crater tier
peaks = detect_craters(dem, n=n, flatness_eps=0.005)

# convert crater world coords (m) -> pixel coords for plotting
craters_ij = np.array([gdem.xy_to_ij(c['x'], c['y']) for c in craters])  # (i, j) = (row, col)

# only compare against the largest craters (size proxy for radius; adjust threshold after inspecting distribution)
print("size distribution: min={:.1f} max={:.1f} median={:.1f}".format(
    craters['size'].min(), craters['size'].max(), np.median(craters['size'])))

size_thresh = np.percentile(craters['size'], 80)  # top 20% largest craters
large_craters_ij = craters_ij[craters['size'] >= size_thresh]
print(f"Comparing against {len(large_craters_ij)} large craters (size >= {size_thresh:.1f})")

# nearest-neighbor match: for each detected peak, distance to nearest large crater center
if len(peaks) > 0 and len(large_craters_ij) > 0:
    from scipy.spatial import cKDTree
    tree = cKDTree(large_craters_ij)
    dists_px, _ = tree.query(peaks)
    dists_m = dists_px * lxy
    print(f"Peak-to-nearest-large-crater distances (m): "
          f"mean={dists_m.mean():.3f}, median={np.median(dists_m):.3f}, max={dists_m.max():.3f}")
    matched = (dists_m < 0.5).sum()  # within 0.5m counts as a match
    print(f"{matched}/{len(peaks)} detected peaks matched a large crater within 0.5m")

plt.figure(figsize=(8, 7))
plt.imshow(dem, cmap="terrain", origin="lower")
plt.scatter(craters_ij[:, 1], craters_ij[:, 0], c="cyan", s=8, alpha=0.4, label=f"All craters (n={len(craters)})")
plt.scatter(large_craters_ij[:, 1], large_craters_ij[:, 0], c="yellow", s=60, marker="o",
            facecolors="none", edgecolors="yellow", linewidths=1.5, label=f"Large craters (n={len(large_craters_ij)})")
plt.scatter(peaks[:, 1], peaks[:, 0], c="red", s=40, marker="x", label=f"Detected peaks (n={len(peaks)})")
plt.colorbar(label="Elevation (m)")
plt.legend(loc="upper right", fontsize=8)
plt.title("Detected features vs ground-truth craters")
plt.savefig("maps/data/gt_validation.png", dpi=150)
print("Saved: maps/data/gt_validation.png")

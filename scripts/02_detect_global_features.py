import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maps.global_dem import GlobalDEM
from features.dilation_detector import detect_peaks

dem = np.load("maps/data/global_dem.npy")
gdem = GlobalDEM(elevation=dem, lxy=0.025, origin_xy=(0.0, 0.0))

n = 60  # cell-radius; D_detect = n * lxy = 60*0.025 = 1.5m
peaks = detect_peaks(dem, n=n, flatness_eps=0.005)

print(f"Detected {len(peaks)} peaks with n={n} (D_detect={n*gdem.lxy:.2f}m)")
print(peaks[:10], "..." if len(peaks) > 10 else "")

plt.figure(figsize=(7, 6))
plt.imshow(dem, cmap="terrain", origin="lower")
if len(peaks) > 0:
    plt.scatter(peaks[:, 1], peaks[:, 0], c="red", s=30, marker="x", label="Detected features")
plt.colorbar(label="Elevation (m)")
plt.title(f"Lunaryard global features (n={n}, {len(peaks)} peaks)")
plt.legend()
plt.savefig("maps/data/global_features_preview.png", dpi=150)
print("Saved: maps/data/global_features_preview.png")

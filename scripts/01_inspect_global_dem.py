import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maps.global_dem import GlobalDEM

dem = np.load("maps/data/global_dem.npy")
mask = np.load("maps/data/global_mask.npy")

print(f"DEM shape: {dem.shape}")
print(f"DEM min/max elevation: {dem.min():.3f} / {dem.max():.3f} m")
print(f"Mask unique values: {np.unique(mask)}")

gdem = GlobalDEM(elevation=dem, lxy=0.025, origin_xy=(0.0, 0.0))

plt.figure(figsize=(7, 6))
plt.imshow(dem, cmap="terrain", origin="lower")
plt.colorbar(label="Elevation (m)")
plt.title(f"Lunaryard 20m global DEM ({dem.shape[0]}x{dem.shape[1]} @ {gdem.lxy}m/px)")
plt.savefig("maps/data/global_dem_preview.png", dpi=150)
print("Saved preview: maps/data/global_dem_preview.png")

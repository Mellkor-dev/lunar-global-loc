# scripts/04_check_crater_alignment.py
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from maps.global_dem import GlobalDEM

dem = np.load("maps/data/global_dem.npy")
craters = np.load("maps/data/craters_data.npy")
lxy = 0.025
gdem = GlobalDEM(elevation=dem, lxy=lxy, origin_xy=(0.0, 0.0))

craters_ij = np.array([gdem.xy_to_ij(c['x'], c['y']) for c in craters])

print(f"DEM shape: {dem.shape}")
print(f"Crater x range: {craters['x'].min():.2f} to {craters['x'].max():.2f}")
print(f"Crater y range: {craters['y'].min():.2f} to {craters['y'].max():.2f}")
print(f"Implied pixel range (i/row): {craters_ij[:,0].min():.1f} to {craters_ij[:,0].max():.1f}")
print(f"Implied pixel range (j/col): {craters_ij[:,1].min():.1f} to {craters_ij[:,1].max():.1f}")

plt.figure(figsize=(8, 7))
plt.imshow(dem, cmap="terrain", origin="lower")
plt.scatter(craters_ij[:, 1], craters_ij[:, 0], c="red", s=15, alpha=0.7)
plt.title("Raw crater coords over DEM (no detector)")
plt.savefig("maps/data/alignment_check.png", dpi=150)
print("Saved: maps/data/alignment_check.png")

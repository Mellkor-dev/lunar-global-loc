import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from maps.local_map import level_points, grid_local_map, quat_to_rotmat
from features.dilation_detector import detect_craters

# raw local scan (vlp16 frame) — same file used before the world transform
pts_struct = np.load("sim/data/local_scan.npy")
pts_local = np.stack([pts_struct['x'], pts_struct['y'], pts_struct['z']], axis=1)

# vlp16 -> base_link static offset
vlp16_offset = np.array([-0.14999991655349731, -2.3034954210743308e-07, 0.41500020027160645])
pts_base_link = pts_local + vlp16_offset

# roll/pitch from /odom orientation (yaw deliberately excluded)
qx, qy, qz, qw = 0.05050792578746255, 0.004088629400219434, -0.00013808546076639217, 0.9987152814865112
R = quat_to_rotmat(qx, qy, qz, qw)
#print(f"roll={np.degrees(roll):.2f} deg, pitch={np.degrees(pitch):.2f} deg")
print(f"R (base_link -> world):\n{R}")
pts_leveled = level_points(pts_base_link, R)
# Known (design-time) vertical offset: sensor sits at a fixed height above
# the local ground plane, independent of world position. Approximate this
# as the negative of the local scan's own elevation median (crude but
# principled: assumes locally-scanned terrain averages near ground level
# under the rover, which is what "ground plane" means).
z_offset = -np.median(pts_leveled[:, 2])
pts_leveled[:, 2] += z_offset
print(f"Applied vertical calibration offset: {z_offset:.3f} m")
lxy = 0.025
local_grid, origin_xy, lxy = grid_local_map(pts_leveled, lxy)
print(f"Local grid shape: {local_grid.shape}, origin: {origin_xy}")

n_local = 60  # same D_detect as global, per paper Sec III
local_peaks = detect_craters(local_grid, n=n_local, flatness_eps=0.005)
print(f"Detected {len(local_peaks)} local features")

plt.figure(figsize=(6, 5))
plt.imshow(local_grid, cmap="terrain", origin="lower")
if len(local_peaks) > 0:
    plt.scatter(local_peaks[:, 1], local_peaks[:, 0], c="red", s=40, marker="x")
plt.colorbar(label="Elevation (m)")
plt.title(f"Local map features (leveled, n={n_local})")
plt.savefig("maps/data/local_features_preview.png", dpi=150)
print("Saved: maps/data/local_features_preview.png")

np.savez("maps/data/local_features.npz",
         local_grid=local_grid, local_peaks=local_peaks,
         origin_xy=origin_xy, lxy=lxy, R=R)

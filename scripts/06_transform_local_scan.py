# scripts/06_transform_local_scan.py  (corrected)
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def quat_to_rotmat(x, y, z, w):
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x*x+z*z),   2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x*x+y*y)]
    ])

# --- Static transform: base_link -> vlp16 (from /tf_static) ---
vlp16_offset = np.array([-0.14999991655349731, -2.3034954210743308e-07, 0.41500020027160645])
# rotation is identity (x,y,z ~0, w~1), so no rotation needed here

# --- Dynamic transform: odom -> base_link (from /odom) ---
spawn_xyz = np.array([10.0, 10.0, 0.5])
odom_xyz = np.array([-0.03562355041503906, -0.03276252746582031, -0.4269917607307434])
world_xyz = spawn_xyz + odom_xyz

qx, qy, qz, qw = 0.05050792578746255, 0.004088629400219434, -0.00013808546076639217, 0.9987152814865112
R = quat_to_rotmat(qx, qy, qz, qw)

print(f"Rover world position (base_link): {world_xyz}")

# --- Load local scan (vlp16 frame), shift to base_link, then to world ---
pts = np.load("sim/data/local_scan.npy")
local_xyz = np.stack([pts['x'], pts['y'], pts['z']], axis=1)

base_link_xyz = local_xyz + vlp16_offset          # vlp16 -> base_link
world_pts = (R @ base_link_xyz.T).T + world_xyz    # base_link -> world (odom)

print(f"World-frame point range: x[{world_pts[:,0].min():.2f},{world_pts[:,0].max():.2f}] "
      f"y[{world_pts[:,1].min():.2f},{world_pts[:,1].max():.2f}] "
      f"z[{world_pts[:,2].min():.2f},{world_pts[:,2].max():.2f}]")

np.save("sim/data/local_scan_world.npy", world_pts)
print("Saved: sim/data/local_scan_world.npy")
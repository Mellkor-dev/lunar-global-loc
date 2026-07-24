import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pointcloud = np.load("sim/data/traverse/local_scan_site01.npy")
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

print("Shape:", pointcloud.shape)
print("Data Type:", pointcloud.dtype)
print("First 20 elements:", pointcloud[:20])

x = pointcloud['x']
y = pointcloud['y']
z = pointcloud['z']


valid_mask = ~((x == 0) & (y == 0) & (z == 0)) & (np.abs(x) < 50) & (np.abs(y) < 50)
x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]


fig = plt.figure(figsize=(14, 6))


ax1 = fig.add_subplot(121, projection='3d')

scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=2, alpha=0.8)
ax1.set_xlabel('X (Forward/Backward)')
ax1.set_ylabel('Y (Left/Right)')
ax1.set_zlabel('Z (Height)')
ax1.set_title('3D Local Map View')
fig.colorbar(scatter1, ax=ax1, label='Height (Z)', pad=0.1)


ax2 = fig.add_subplot(122)

scatter2 = ax2.scatter(x, y, c=z, cmap='viridis', s=2, alpha=0.8)
ax2.set_xlabel('X (Forward/Backward)')
ax2.set_ylabel('Y (Left/Right)')
ax2.set_title('2D Top-Down Local Map')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.axis('equal') 

plt.tight_layout()
plt.show






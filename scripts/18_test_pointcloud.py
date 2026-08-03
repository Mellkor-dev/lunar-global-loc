import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pointcloud = np.load("sim/data/traverse/scan_site_28.npy")
import matplotlib.pyplot as plt
#import open3d as o3d
import numpy as np

print("Shape:", pointcloud.shape)
print("Data Type:", pointcloud.dtype)
print("First 20 elements:", pointcloud[:20])

x = pointcloud['x']
y = pointcloud['y']
z = pointcloud['z']

odom_to_lidar_transform = np.array([[0.9963381999, 0.0020336656, -0.0854754685, -0.1848350956],
                                    [-0.0035239641, 0.9998443390, -0.0172881293, -0.0018247955],
                                    [0.0854270051, 0.0175260361, 0.9961902754, -0.0892609051],
                                    [0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000]])

valid_mask = ~((x == 0) & (y == 0) & (z == 0)) & (np.abs(x) < 50) & (np.abs(y) < 50)
x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]

ones = np.ones((x.shape[0], 1))
points_homogeneous = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1), ones))
transformed_points = (odom_to_lidar_transform @ points_homogeneous.T).T
x, y, z = transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2]
fig = plt.figure(figsize=(14, 6))


ax1 = fig.add_subplot(121, projection='3d')

scatter1 = ax1.scatter(
    x, y, z,
    c=z,
    cmap='viridis',
    s=2,
    alpha=0.8
)

ax1.set_xlabel('X (Forward/Backward)')
ax1.set_ylabel('Y (Left/Right)')
ax1.set_zlabel('Z (Height)')
ax1.set_title('3D Local Map View')

# Equal metric scaling
ax1.set_box_aspect([
    np.ptp(x),
    np.ptp(y),
    np.ptp(z)
])

fig.colorbar(
    scatter1,
    ax=ax1,
    label='Height (Z)',
    pad=0.1
)

plt.tight_layout()
plt.show()






import numpy as np
from scipy.interpolate import griddata

def quat_to_rotmat(x, y, z, w):
    """Same as script 06 — reuse directly, don't re-derive via Euler angles."""
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x*x+z*z),   2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x*x+y*y)]
    ])

def level_points(points_local: np.ndarray, R_world_from_base: np.ndarray) -> np.ndarray:
    """
    Rotate points from base_link frame into a world-aligned (leveled) frame
    using the SAME rotation matrix derived from the full quaternion — no
    separate roll/pitch reconstruction, avoids Euler-order mismatches.
    """
    return (R_world_from_base @ points_local.T).T

def grid_local_map(points_leveled: np.ndarray, lxy: float, pad: float = 0.5):
    x, y, z = points_leveled[:, 0], points_leveled[:, 1], points_leveled[:, 2]
    x_min, x_max = x.min() - pad, x.max() + pad
    y_min, y_max = y.min() - pad, y.max() + pad
    nx = int((x_max - x_min) / lxy)
    ny = int((y_max - y_min) / lxy)
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='nearest')
    return grid_z, (x_min, y_min), lxy

def level_points_tilt_only(points_body: np.ndarray, R_body_to_world: np.ndarray) -> np.ndarray:
    """
    Apply full R (correct tilt+yaw), then rotate back by -yaw only about
    world Z, leaving pure tilt applied and yaw reset to zero. Avoids
    reconstructing R from scratch (no Euler-order ambiguity vs the
    original quaternion).
    """
    world_pts_full = (R_body_to_world @ points_body.T).T  # correct tilt AND yaw

    # yaw = rotation of R's gix-axis projected onto world XY plane
    x_axis_world = R_body_to_world @ np.array([1.0, 0.0, 0.0])
    yaw = np.arctan2(x_axis_world[1], x_axis_world[0])

    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (Rz_inv @ world_pts_full.T).T
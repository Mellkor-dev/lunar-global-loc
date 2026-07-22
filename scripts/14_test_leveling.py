# scripts/14_test_leveling.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from maps.local_map import quat_to_rotmat, level_points_tilt_only

def quat_from_axis_angle(axis, angle_deg):
    angle = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)
    return (*(axis * np.sin(angle/2)), np.cos(angle/2))

# Test 1: pure yaw 
qx, qy, qz, qw = quat_from_axis_angle(np.array([0,0,1]), 90)
R = quat_to_rotmat(qx, qy, qz, qw)
pts = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
leveled = level_points_tilt_only(pts, R)
print("Test 1 (pure 90deg yaw) -- expect near-identity (no rotation):")
print(leveled, "\nvs original:\n", pts)

# Test 2: pure roll 
qx, qy, qz, qw = quat_from_axis_angle(np.array([1,0,0]), 30)
R = quat_to_rotmat(qx, qy, qz, qw)
pts = np.array([[0.0, 1.0, 0.0]])
leveled = level_points_tilt_only(pts, R)
expected = (R @ pts.T).T  # full R be careful, this is not the same as leveled
print("\nTest 2 (pure 30deg roll) -- expect leveled == full-R result:")
print("leveled:", leveled, " full-R:", expected)

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from refinement.moga import run_moga, wrap_angle

# --- Ground truth: 3 sites ---
truth = {
    "s1": (0.0, 0.0, 0.0),
    "s2": (5.0, 2.0, np.radians(30)),
    "s3": (9.0, 6.0, np.radians(70)),
}
site_names = ["s1", "s2", "s3"]

def pose_R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

# --- Fabricate global landmarks (fixed, arbitrary) ---
global_landmarks = {
    "L1": np.array([2.0, 1.0]),
    "L2": np.array([6.0, 3.0]),
    "L3": np.array([10.0, 7.0]),
}

# --- For each site, create a local feature that EXACTLY corresponds to
#     one global landmark under the ground-truth pose:
#     global = R(theta) @ local + t   =>   local = R(theta).T @ (global - t)
site_to_landmark = {"s1": "L1", "s2": "L2", "s3": "L3"}
feature_correspondences = {}
for s in site_names:
    x, y, th = truth[s]
    R = pose_R(th)
    t = np.array([x, y])
    g = global_landmarks[site_to_landmark[s]]
    local = R.T @ (g - t)
    feature_correspondences[s] = [(local, g)]

# --- Fabricate EXACT odometry between consecutive sites ---
odometry_chain = []
for a, b in [("s1", "s2"), ("s2", "s3")]:
    xa, ya, tha = truth[a]
    xb, yb, thb = truth[b]
    Ra = pose_R(tha)
    rho = Ra.T @ (np.array([xb, yb]) - np.array([xa, ya]))
    dtheta = wrap_angle(thb - tha)
    odometry_chain.append({"from": a, "to": b, "rho_xy": rho, "dtheta": dtheta})

# --- Start MOGA from a deliberately WRONG initial guess ---
initial_poses = {
    "s1": (1.0, -1.0, np.radians(10)),
    "s2": (4.0, 4.0, np.radians(10)),
    "s3": (7.0, 3.0, np.radians(10)),
}

print("=== Ground truth ===")
for s in site_names:
    x, y, th = truth[s]
    print(f"{s}: x={x:.3f} y={y:.3f} theta={np.degrees(th):.2f}deg")

print("\n=== Initial (wrong) guess ===")
for s in site_names:
    x, y, th = initial_poses[s]
    print(f"{s}: x={x:.3f} y={y:.3f} theta={np.degrees(th):.2f}deg")

orientation_measurements = {
    s: truth[s][2] + np.radians(1.0)  # simulate ~1deg sensor error
    for s in site_names
}

refined = run_moga(
    site_names, initial_poses, feature_correspondences, odometry_chain,
    orientation_measurements=orientation_measurements,
    sigma_feature=0.1, sigma_odom_trans=0.05, sigma_odom_rot_deg=1.0,
    sigma_heading_deg=1.0,
    max_iter=50, e_converge=1e-12, verbose=True
)

print("\n=== MOGA refined ===")
max_err = 0.0
for s in site_names:
    x, y, th = refined[s]
    tx, ty, tth = truth[s]
    err_xy = np.sqrt((x-tx)**2 + (y-ty)**2)
    err_th = abs(np.degrees(wrap_angle(th - tth)))
    max_err = max(max_err, err_xy, err_th)
    print(f"{s}: x={x:.6f} y={y:.6f} theta={np.degrees(th):.4f}deg  "
          f"(pos_err={err_xy:.6f}m, heading_err={err_th:.6f}deg)")

print(f"\nMax error across all sites/dims: {max_err:.8f}")
print("PASS" if max_err < 1.5 else "FAIL")

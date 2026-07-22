"""""

Design simplification vs. the paper's full Eq. 2:
    - Global feature positions are treated as FIXED (
    

Three error terms (of the paper's four — global/local feature terms are
merged, see above):   

Solved via Gauss-Newton: linearize all residual term
"""

import numpy as np


def wrap_angle(a):
    
    return (a + np.pi) % (2 * np.pi) - np.pi


def pose_to_matrix(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t = np.array([x, y])
    return R, t


def build_feature_residual_and_jacobian(x, y, theta, local_xy, global_xy):
    
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pred = R @ local_xy + np.array([x, y])
    residual = pred - global_xy

    dR_dtheta = np.array([[-s, -c], [c, -s]])
    J = np.zeros((2, 3))
    J[:, 0] = [1, 0]
    J[:, 1] = [0, 1]
    J[:, 2] = dR_dtheta @ local_xy
    return residual, J


def build_orientation_residual_and_jacobian(theta, theta_meas):
    
    r = np.array([wrap_angle(theta - theta_meas)])
    J = np.array([[1.0]])
    return r, J


def build_odometry_residual_and_jacobian(xa, ya, tha, xb, yb, thb, rho_meas, dtheta_meas):
    
    ca, sa = np.cos(tha), np.sin(tha)
    Ra = np.array([[ca, -sa], [sa, ca]])
    t_diff = np.array([xb - xa, yb - ya])
    pred_rho = Ra.T @ t_diff #check later
    pred_dtheta = wrap_angle(thb - tha) #something wrong here

    res_rho = pred_rho - rho_meas
    res_dtheta = wrap_angle(pred_dtheta - dtheta_meas) #check later
    residual = np.array([res_rho[0], res_rho[1], res_dtheta])

    dRa_dtha = np.array([[-sa, -ca], [ca, -sa]])
    J = np.zeros((3, 6))
    J[0:2, 0:2] = -Ra.T
    J[0:2, 2] = (dRa_dtha.T @ t_diff)
    J[0:2, 3:5] = Ra.T
    J[2, 2] = -1
    J[2, 5] = 1
    return residual, J


def run_moga(site_names, initial_poses, feature_correspondences, odometry_chain,
             orientation_measurements=None,
             sigma_feature=0.4, sigma_odom_trans=0.1, sigma_odom_rot_deg=1.0,
             sigma_heading_deg=1.0,
             max_iter=50, e_converge=1e-6, verbose=True):
    """
    site_names: list of site name strings, in traverse order
    initial_poses: dict {site_name: (x, y, theta)} — from DARCES
    feature_correspondences: dict {site_name: list of (local_xy, global_xy)}
    odometry_chain: list of dicts {"from", "to", "rho_xy" (2-vec), "dtheta"}
    orientation_measurements: dict {site_name: theta_meas} — independent
        absolute heading. Resolves the global rotational gauge freedom —
        without it, a single-landmark-per-site system is under-constrained.
        If None, falls back to a hard anchor on site_names[0].

    Returns dict {site_name: (x, y, theta)} — refined poses.
    """
    n_sites = len(site_names)
    idx = {s: i for i, s in enumerate(site_names)}

    z = np.zeros(3 * n_sites)
    for s in site_names:
        i = idx[s]
        x, y, th = initial_poses[s]
        z[3*i:3*i+3] = [x, y, th]

    sigma_odom_rot = np.radians(sigma_odom_rot_deg)
    sigma_heading = np.radians(sigma_heading_deg)
    W_feature = np.diag([1/sigma_feature**2, 1/sigma_feature**2])
    W_odom = np.diag([1/sigma_odom_trans**2, 1/sigma_odom_trans**2, 1/sigma_odom_rot**2])
    W_heading = np.array([[1/sigma_heading**2]])

    prev_cost = None
    for iteration in range(max_iter):
        H = np.zeros((3*n_sites, 3*n_sites))
        b = np.zeros(3*n_sites)
        cost = 0.0

        for s in site_names:
            i = idx[s]
            x, y, th = z[3*i:3*i+3]
            for local_xy, global_xy in feature_correspondences.get(s, []):
                r, J = build_feature_residual_and_jacobian(x, y, th, local_xy, global_xy)
                H[3*i:3*i+3, 3*i:3*i+3] += J.T @ W_feature @ J
                b[3*i:3*i+3] += J.T @ W_feature @ r
                cost += 0.5 * r.T @ W_feature @ r

        for edge in odometry_chain:
            a, bsite = edge["from"], edge["to"]
            ia, ib = idx[a], idx[bsite]
            xa, ya, tha = z[3*ia:3*ia+3]
            xb, yb, thb = z[3*ib:3*ib+3]
            rho_meas = edge["rho_xy"]
            dtheta_meas = edge["dtheta"]

            r, J = build_odometry_residual_and_jacobian(xa, ya, tha, xb, yb, thb, rho_meas, dtheta_meas)
            idx_block = list(range(3*ia, 3*ia+3)) + list(range(3*ib, 3*ib+3))
            for p in range(6):
                for q in range(6):
                    H[idx_block[p], idx_block[q]] += (J[:, p].T @ W_odom @ J[:, q])
                b[idx_block[p]] += J[:, p].T @ W_odom @ r
            cost += 0.5 * r.T @ W_odom @ r

        if orientation_measurements is not None:
            for s in site_names:
                if s not in orientation_measurements:
                    continue
                i = idx[s]
                th = z[3*i+2]
                theta_meas = orientation_measurements[s]
                r, J = build_orientation_residual_and_jacobian(th, theta_meas)
                H[3*i+2, 3*i+2] += (J.T @ W_heading @ J)[0, 0]
                b[3*i+2] += (J.T @ W_heading @ r)[0]
                cost += 0.5 * float(r.T @ W_heading @ r)
        else:
            H[0:3, :] = 0.0
            H[:, 0:3] = 0.0
            H[0:3, 0:3] = np.eye(3)
            b[0:3] = 0.0

        try:
            dz = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            print("MOGA: singular H matrix, stopping.")
            break

        z += dz

        if verbose:
            print(f"iter {iteration}: cost={cost:.6f}, |dz|={np.linalg.norm(dz):.6f}")

        if prev_cost is not None and abs(prev_cost - cost) < e_converge:
            if verbose:
                print(f"Converged at iteration {iteration}")
            break
        prev_cost = cost

    refined = {}
    for s in site_names:
        i = idx[s]
        refined[s] = tuple(z[3*i:3*i+3])
    return refined
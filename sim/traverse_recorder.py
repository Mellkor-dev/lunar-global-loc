import numpy as np

def quat_to_rotmat(x, y, z, w):
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x*x+z*z),   2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x*x+y*y)]
    ])

SPAWN_XYZ = np.array([10.0, 10.0, 0.5])

# (position, quaternion x,y,z,w), from /odom at each site
RAW_ODOM = {
    "site01": ((3.1273012161254883, 2.3025007247924805, -0.6429600715637207),
               (-0.03698534315603117, 0.02151380862943881, 0.38480776212815365, 0.9220044612884521)),
    "site02": ((6.854265213012695, -4.261559963226318, -0.430521696805954),
               (-0.022717915907412452, 0.012086715659754964, -0.6007036291985641, 0.7990575432777405)),
    "site03": ((-5.314756393432617, -1.5379981994628906, -0.6164887547492981),
               (0.031908699591060144, 0.03485072908948109, -0.9361809251384015, -0.3483282029628753)),
    "site04": ((-3.220860481262207, 5.191847801208496, -0.4904630482196808),
               (-0.0016240545908121982, -0.02667094520260749, -0.5649236513882344, -0.8247104287147522)),
    "site05": ((0.5108470916748047, -1.7831220626831055, -0.6171184778213501),
               (0.023142535666483267, -0.06635113393931005, 0.8855143236158297, -0.45926716923713695)),
}

def build_traverse():
    sites = sorted(RAW_ODOM.keys())
    world_t = {}
    world_R = {}
    for s in sites:
        pos, quat = RAW_ODOM[s]
        world_t[s] = SPAWN_XYZ + np.array(pos)
        world_R[s] = quat_to_rotmat(*quat)
        print(f"{s}: world_pos={world_t[s]}, heading={np.degrees(np.arctan2(world_R[s][1,0], world_R[s][0,0])):.1f}deg")

    # relative odometry between consecutive sites (Fig. 2 notation: T_B^A)
    odometry_chain = []
    for i in range(len(sites) - 1):
        a, b = sites[i], sites[i+1]
        R_a, t_a = world_R[a], world_t[a]
        R_b, t_b = world_R[b], world_t[b]

        R_ab = R_a.T @ R_b               # rotation of b expressed in a's frame
        rho_ab = R_a.T @ (t_b - t_a)      # translation of b expressed in a's frame

        dist = np.linalg.norm(rho_ab)
        print(f"odometry {a}->{b}: translation={rho_ab}, distance={dist:.2f}m")
        odometry_chain.append({"from": a, "to": b, "R": R_ab, "rho": rho_ab})

    return {"sites": sites, "world_t": world_t, "world_R": world_R, "odometry_chain": odometry_chain}

if __name__ == "__main__":
    traverse = build_traverse()
    np.savez("maps/data/traverse.npz",
              sites=traverse["sites"],
              world_t={s: traverse["world_t"][s] for s in traverse["sites"]},
              allow_pickle=True)
    print("\nSaved: maps/data/traverse.npz")

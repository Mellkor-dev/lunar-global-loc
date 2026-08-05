#!/usr/bin/env python3
"""Inspect a captured LiDAR cloud with several metric-scale views."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


DEFAULT_SCAN = Path("LargeScale_Implement/sim/5m_px/pointcloud_scans/scan_site_01.npy")


def load_xyz(path: Path) -> np.ndarray:
    """Load either a structured ROS array or a plain N-by-3/N-by-4 array."""
    cloud = np.load(path, allow_pickle=False)
    if cloud.dtype.names is not None:
        missing = {"x", "y", "z"}.difference(cloud.dtype.names)
        if missing:
            raise ValueError(f"{path} is missing point fields: {sorted(missing)}")
        xyz = np.column_stack((cloud["x"], cloud["y"], cloud["z"]))
    else:
        cloud = np.asarray(cloud)
        if cloud.ndim != 2 or cloud.shape[1] < 3:
            raise ValueError(f"{path} must have shape (N, >=3), got {cloud.shape}")
        xyz = cloud[:, :3]
    return np.asarray(xyz, dtype=np.float64)


def filter_xyz(xyz: np.ndarray, min_range: float, max_range: float) -> tuple[np.ndarray, dict[str, int]]:
    finite = np.isfinite(xyz).all(axis=1)
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-6
    ranges = np.linalg.norm(xyz, axis=1)
    in_range = (ranges >= min_range) & (ranges <= max_range)
    keep = finite & nonzero & in_range
    counts = {
        "input": int(len(xyz)),
        "non_finite": int(np.count_nonzero(~finite)),
        "zero": int(np.count_nonzero(finite & ~nonzero)),
        "outside_range": int(np.count_nonzero(finite & nonzero & ~in_range)),
        "kept": int(np.count_nonzero(keep)),
    }
    return xyz[keep], counts


def sample_points(xyz: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(xyz) <= limit:
        return xyz
    rng = np.random.default_rng(seed)
    return xyz[rng.choice(len(xyz), size=limit, replace=False)]


def equal_xy_limits(ax, xyz: np.ndarray) -> None:
    lo = np.min(xyz[:, :2], axis=0)
    hi = np.max(xyz[:, :2], axis=0)
    center = (lo + hi) / 2.0
    half_span = max(float(np.max(hi - lo)) / 2.0, 1e-3)
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan", nargs="?", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--min-range", type=float, default=0.5, help="Minimum retained radial range in metres")
    parser.add_argument("--max-range", type=float, default=250.0, help="Maximum retained radial range in metres")
    parser.add_argument("--max-points", type=int, default=200_000, help="Random display-point limit; 0 keeps all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="Save the figure instead of only displaying it")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_xyz = load_xyz(args.scan)
    xyz, counts = filter_xyz(raw_xyz, args.min_range, args.max_range)
    if not len(xyz):
        raise RuntimeError("No valid points remain after filtering")

    display_xyz = sample_points(xyz, args.max_points, args.seed)
    ranges = np.linalg.norm(xyz, axis=1)
    z_lo, z_hi = np.percentile(display_xyz[:, 2], (1.0, 99.0))
    if z_hi <= z_lo:
        z_hi = z_lo + 1e-6
    color_norm = Normalize(z_lo, z_hi, clip=True)

    print(f"Scan: {args.scan.resolve()}")
    print(f"Array: shape={raw_xyz.shape}, dtype={raw_xyz.dtype}")
    print("Filtering: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Displayed: {len(display_xyz):,} points")
    print(
        "Range [m]: "
        f"min={ranges.min():.3f}, median={np.median(ranges):.3f}, "
        f"p95={np.percentile(ranges, 95):.3f}, max={ranges.max():.3f}"
    )
    print(
        "XYZ bounds [m]: "
        f"min={np.min(xyz, axis=0).round(3).tolist()}, "
        f"max={np.max(xyz, axis=0).round(3).tolist()}"
    )

    # Explicit spacing is more reliable than constrained_layout for a mixed
    # 3D/2D figure with a shared colorbar, especially with the Tk backend.
    fig = plt.figure(figsize=(16, 10))
    fig.subplots_adjust(
        left=0.06,
        right=0.93,
        bottom=0.07,
        top=0.90,
        wspace=0.24,
        hspace=0.30,
    )
    grid = fig.add_gridspec(2, 2)
    ax_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_top = fig.add_subplot(grid[0, 1])
    ax_side = fig.add_subplot(grid[1, 0])
    ax_range = fig.add_subplot(grid[1, 1])

    point_size = max(0.15, min(2.0, 100_000 / max(len(display_xyz), 1)))
    common = dict(c=display_xyz[:, 2], cmap="terrain", norm=color_norm, s=point_size, alpha=0.75, linewidths=0)

    cloud_plot = ax_3d.scatter(display_xyz[:, 0], display_xyz[:, 1], display_xyz[:, 2], **common)
    ax_3d.scatter([0], [0], [0], marker="^", s=45, c="black", label="LiDAR")
    ax_3d.set_title("Oblique 3D view")
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    spans = np.ptp(display_xyz, axis=0)
    ax_3d.set_box_aspect(np.maximum(spans, 1e-3))
    ax_3d.view_init(elev=32, azim=-55)
    ax_3d.legend(loc="upper right")

    ax_top.scatter(display_xyz[:, 0], display_xyz[:, 1], **common)
    ax_top.scatter([0], [0], marker="^", s=45, c="black")
    ax_top.set_title("Top-down footprint (colour = elevation)")
    ax_top.set_xlabel("X [m]")
    ax_top.set_ylabel("Y [m]")
    ax_top.set_aspect("equal", adjustable="box")
    equal_xy_limits(ax_top, display_xyz)

    ax_side.scatter(
        display_xyz[:, 0], display_xyz[:, 2],
        c=display_xyz[:, 1], cmap="coolwarm", s=point_size, alpha=0.65, linewidths=0,
    )
    ax_side.scatter([0], [0], marker="^", s=45, c="black")
    ax_side.set_title("Side profile (colour = lateral Y)")
    ax_side.set_xlabel("X [m]")
    ax_side.set_ylabel("Z [m]")
    ax_side.grid(alpha=0.25)

    bins = min(160, max(30, int(np.sqrt(len(ranges)))))
    ax_range.hist(ranges, bins=bins, color="slateblue", alpha=0.85)
    ax_range.axvline(np.median(ranges), color="black", linestyle="--", label="median")
    ax_range.axvline(np.percentile(ranges, 95), color="crimson", linestyle="--", label="p95")
    ax_range.set_title("Radial range distribution")
    ax_range.set_xlabel("Range [m]")
    ax_range.set_ylabel("Point count")
    ax_range.set_yscale("log")
    ax_range.grid(alpha=0.25)
    ax_range.legend()

    fig.colorbar(cloud_plot, ax=[ax_3d, ax_top], label="Elevation Z [m]", shrink=0.8)
    fig.suptitle(f"LiDAR scan quality overview — {args.scan.name}\n{counts['kept']:,} valid points", fontsize=14)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

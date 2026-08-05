#!/usr/bin/env python3
"""Plot captured rover traversal and DARCES estimates over the selected DEM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import add_resolution_argument, load_resolution_config


SITE_PATTERN = re.compile(r"odom_site_(\d+)\.npy$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_resolution_argument(parser)
    parser.add_argument(
        "--results",
        type=Path,
        help="Override the default results/<resolution>_px/darces_all_sites.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override plots/<resolution>_px/traversal/darces_traversal_map.png",
    )
    parser.add_argument(
        "--zoom-padding",
        type=float,
        default=100.0,
        help="Terrain-detail padding around the odometry traverse in metres",
    )
    parser.add_argument(
        "--max-background-size",
        type=int,
        default=1600,
        help="Maximum displayed DEM rows/columns before display decimation",
    )
    parser.add_argument(
        "--no-global-features",
        action="store_true",
        help="Do not overlay the selected resolution's global feature catalogue",
    )
    parser.add_argument("--show", action="store_true", help="Also open the figure interactively")
    return parser.parse_args()


def load_odometry(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    records: list[tuple[int, np.ndarray]] = []
    for path in directory.glob("odom_site_*.npy"):
        match = SITE_PATTERN.search(path.name)
        if match is None:
            continue
        pose = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"{path} must contain seven finite pose values")
        records.append((int(match.group(1)), pose))
    if not records:
        raise FileNotFoundError(f"No odom_site_*.npy files found in {directory}")
    records.sort(key=lambda item: item[0])
    return (
        np.asarray([site for site, _ in records], dtype=np.int64),
        np.stack([pose for _, pose in records]),
    )


def load_estimates(path: Path) -> dict[int, dict[str, float | str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    sites = payload.get("sites", [])
    if not isinstance(sites, list):
        raise ValueError(f"{path}: 'sites' must be a list")
    estimates: dict[int, dict[str, float | str]] = {}
    for record in sites:
        if not isinstance(record, dict) or "site" not in record:
            continue
        estimates[int(record["site"])] = record
    return estimates


def display_dem(dem: np.ndarray, maximum_size: int) -> tuple[np.ndarray, int]:
    if maximum_size <= 0:
        raise ValueError("--max-background-size must be positive")
    stride = max(1, int(np.ceil(max(dem.shape) / maximum_size)))
    return dem[::stride, ::stride], stride


def dem_extent(config) -> tuple[float, float, float, float]:
    raster = config.orbital_raster
    x_first = raster.first_x_center_m
    y_first = raster.first_y_center_m
    x_last = x_first + (raster.shape[1] - 1) * raster.resolution_m
    y_last = y_first - (raster.shape[0] - 1) * raster.resolution_m
    half = raster.resolution_m / 2.0
    return (x_first - half, x_last + half, y_last - half, y_first + half)


def terrain_rgb(dem: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(dem)
    if not finite.any():
        raise ValueError("DEM has no finite cells")
    low, high = np.nanpercentile(dem, (1.0, 99.0))
    if high <= low:
        high = low + 1e-6
    filled = np.where(finite, dem, np.nanmedian(dem))
    light = LightSource(azdeg=315, altdeg=42)
    rgb = light.shade(
        filled,
        cmap=plt.get_cmap("terrain"),
        vert_exag=2.0,
        dx=cell_size_m,
        dy=cell_size_m,
        vmin=low,
        vmax=high,
        blend_mode="soft",
    )
    rgb[~finite, 3] = 0.0
    return rgb, float(low), float(high)


def headings_xy(poses: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()
    return rotations[:, :2, 0]


def add_background(ax, rgb: np.ndarray, extent, title: str) -> None:
    ax.imshow(rgb, extent=extent, origin="upper", interpolation="bilinear")
    ax.set_title(title)
    ax.set_xlabel("Map x / east [m]")
    ax.set_ylabel("Map y / north [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", alpha=0.16, linewidth=0.6)


def plot_route(ax, sites: np.ndarray, poses: np.ndarray, *, labels: bool) -> None:
    xy = poses[:, :2]
    ax.plot(xy[:, 0], xy[:, 1], color="#00e5ff", linewidth=2.2, zorder=6, label="Odometry traverse")
    ax.scatter(
        xy[:, 0], xy[:, 1], s=28, c="#072b36", edgecolors="white",
        linewidths=0.8, zorder=7, label="Captured sites",
    )
    directions = headings_xy(poses)
    ax.quiver(
        xy[:, 0], xy[:, 1], directions[:, 0], directions[:, 1],
        color="white", angles="xy", scale_units="xy", scale=0.035,
        width=0.003, headwidth=3.5, zorder=8,
    )
    if labels:
        for site, (x, y) in zip(sites, xy):
            ax.annotate(
                f"{site:02d}", (x, y), xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="white", weight="bold", zorder=9,
            )


def plot_estimates(
    ax,
    sites: np.ndarray,
    poses: np.ndarray,
    estimates: dict[int, dict[str, float | str]],
    *,
    labels: bool,
) -> tuple[int, list[float]]:
    pose_by_site = {int(site): pose for site, pose in zip(sites, poses)}
    estimate_rows = []
    errors = []
    for site, result in sorted(estimates.items()):
        if result.get("status") != "solution" or site not in pose_by_site:
            continue
        try:
            estimated = np.array(
                [float(result["estimated_x_m"]), float(result["estimated_y_m"])],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(estimated).all():
            continue
        truth = pose_by_site[site][:2]
        error = float(np.linalg.norm(estimated - truth))
        estimate_rows.append((site, estimated, truth, error))
        errors.append(error)

    for index, (site, estimated, truth, error) in enumerate(estimate_rows):
        ax.plot(
            [truth[0], estimated[0]], [truth[1], estimated[1]],
            color="#ffcc80", linewidth=0.9, alpha=0.75, linestyle="--", zorder=4,
            label="DARCES error vector" if index == 0 else None,
        )
        ax.scatter(
            estimated[0], estimated[1], marker="x", s=65, linewidths=2.0,
            color="#ff1744", zorder=10,
            label="DARCES estimate" if index == 0 else None,
        )
        if labels:
            ax.annotate(
                f"E{site:02d}\n{error:.1f} m", estimated, xytext=(5, -12),
                textcoords="offset points", fontsize=7, color="#ffebee",
                weight="bold", zorder=11,
            )
    return len(estimate_rows), errors


def main() -> None:
    args = parse_arguments()
    config = load_resolution_config(args.resolution)
    output_path = args.output or (
        config.plots_path / "traversal" / "darces_traversal_map.png"
    )
    results_path = args.results or (
        config.results_path / "darces_all_sites.json"
    )

    sites, poses = load_odometry(config.captures_path / "odom_scans")
    estimates = load_estimates(results_path)
    dem = np.load(config.orbital_dem_path, mmap_mode="r", allow_pickle=False)
    if dem.shape != config.orbital_raster.shape:
        raise ValueError(
            f"DEM shape {dem.shape} differs from configured {config.orbital_raster.shape}"
        )
    shown_dem, stride = display_dem(np.asarray(dem), args.max_background_size)
    rgb, elevation_low, elevation_high = terrain_rgb(
        shown_dem,
        config.orbital_raster.resolution_m * stride,
    )
    extent = dem_extent(config)

    global_features = np.empty((0, 2), dtype=np.float64)
    if not args.no_global_features and config.global_features_path.exists():
        with np.load(config.global_features_path, allow_pickle=False) as catalogue:
            global_features = np.column_stack((catalogue["x_m"], catalogue["y_m"]))

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.08, top=0.90, wspace=0.14)
    add_background(axes[0], rgb, extent, "Full DEM and DARCES localization overview")
    add_background(axes[1], rgb, extent, "Terrain detail around captured traverse")

    for ax in axes:
        if len(global_features):
            ax.scatter(
                global_features[:, 0], global_features[:, 1], marker="o", s=14,
                facecolors="none", edgecolors="#fff176", linewidths=0.55,
                alpha=0.72, zorder=3, label="Global detected feature",
            )
        plot_route(ax, sites, poses, labels=ax is axes[1])

    solution_count, errors = plot_estimates(
        axes[0], sites, poses, estimates, labels=True,
    )
    plot_estimates(axes[1], sites, poses, estimates, labels=False)

    xy = poses[:, :2]
    x_min, y_min = np.min(xy, axis=0) - args.zoom_padding
    x_max, y_max = np.max(xy, axis=0) + args.zoom_padding
    map_x_min, map_x_max, map_y_min, map_y_max = extent
    axes[1].set_xlim(max(x_min, map_x_min), min(x_max, map_x_max))
    axes[1].set_ylim(max(y_min, map_y_min), min(y_max, map_y_max))

    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.88)
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.88)

    step_distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    trajectory_distance = float(np.sum(step_distances))
    error_summary = "no successful estimates"
    if errors:
        error_summary = (
            f"DARCES XY error: median {np.median(errors):.1f} m, "
            f"max {np.max(errors):.1f} m"
        )
    fig.suptitle(
        f"Apollo 17 traverse over {args.resolution} DEM\n"
        f"{len(sites)} sites · {trajectory_distance:.1f} m sampled path · "
        f"{solution_count} DARCES solutions · {error_summary}",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.025,
        f"Hillshade elevation display: p1={elevation_low:.2f} m, "
        f"p99={elevation_high:.2f} m · background display stride={stride} · "
        "red crosses are estimates; dashed lines show current odometry-to-estimate error",
        ha="center",
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print("Traversal/DARCES map")
    print("--------------------")
    print(f"Resolution:        {args.resolution}")
    print(f"DEM:               {config.orbital_dem_path}")
    print(f"Captured sites:    {len(sites)}")
    print(f"Trajectory length: {trajectory_distance:.3f} m")
    print(f"DARCES results:    {results_path if results_path.exists() else 'not found'}")
    print(f"Solutions plotted: {solution_count}")
    print(f"Global features:   {len(global_features)}")
    print(f"Saved plot:        {output_path}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

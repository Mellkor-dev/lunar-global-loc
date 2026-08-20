"""Presentation-oriented DEM contour and localization pose visualization."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np


DISPLAY_NAMES = {
    "apollo17_5m.yaml": "Apollo 17",
    "apollo11.yaml": "Apollo 11",
    "haworth.yaml": "Haworth",
}

LOCALIZATION_STAGES = ("darces", "ransac", "moga")
STAGE_PRIORITY = {
    stage: priority for priority, stage in enumerate(LOCALIZATION_STAGES)
}


def environment_display_name(config: Any) -> str:
    """Return the human-readable environment name for plot titles."""
    configured = DISPLAY_NAMES.get(config.config_path.name)
    if configured is not None:
        return configured
    label = str(config.truth_source_label).strip()
    marker = " native "
    return label.split(marker, 1)[0] if marker in label else label


def spatial_route_order(sites: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Order sites spatially from north to south using nearest neighbours."""
    sites = np.asarray(sites, dtype=np.int64)
    xy = np.asarray(xy, dtype=np.float64)
    if sites.ndim != 1 or xy.shape != (len(sites), 2):
        raise ValueError("sites and xy must have shapes (N,) and (N, 2)")
    if len(sites) == 0:
        return np.empty(0, dtype=np.int64)
    remaining = set(range(len(sites)))
    start = max(remaining, key=lambda index: (xy[index, 1], -xy[index, 0]))
    order = [start]
    remaining.remove(start)
    while remaining:
        current = order[-1]
        next_index = min(
            remaining,
            key=lambda index: (
                float(np.linalg.norm(xy[index] - xy[current])),
                int(sites[index]),
            ),
        )
        order.append(next_index)
        remaining.remove(next_index)
    return np.asarray(order, dtype=np.int64)


def _dem_extent(config: Any) -> tuple[float, float, float, float]:
    raster = config.orbital_raster
    half = raster.resolution_m / 2.0
    x_last = raster.first_x_center_m + (raster.shape[1] - 1) * raster.resolution_m
    y_last = raster.first_y_center_m - (raster.shape[0] - 1) * raster.resolution_m
    return (
        raster.first_x_center_m - half,
        x_last + half,
        y_last - half,
        raster.first_y_center_m + half,
    )


def _grayscale_hillshade(dem: np.ndarray, cell_size_m: float) -> np.ndarray:
    finite = np.isfinite(dem)
    if not finite.any():
        raise ValueError("DEM has no finite elevation cells")
    low, high = np.nanpercentile(dem, (1.0, 99.0))
    if high <= low:
        high = low + 1e-6
    filled = np.where(finite, dem, np.nanmedian(dem))
    image = LightSource(azdeg=315, altdeg=38).shade(
        filled,
        cmap=plt.get_cmap("gray"),
        vert_exag=2.0,
        dx=cell_size_m,
        dy=cell_size_m,
        vmin=low,
        vmax=high,
        blend_mode="soft",
    )
    image[~finite, 3] = 0.0
    return image


def _prediction_rows(
    sites: np.ndarray,
    poses: np.ndarray,
    raw_sites_in_display_order: np.ndarray,
    estimates: dict[int, dict[str, float | str]],
) -> list[tuple[int, int, np.ndarray, float, np.ndarray, float]]:
    display_by_raw = {
        int(raw_site): display_site
        for display_site, raw_site in enumerate(raw_sites_in_display_order, 1)
    }
    truth_by_raw = {int(site): pose[:2] for site, pose in zip(sites, poses)}
    rows: list[tuple[int, int, np.ndarray, float, np.ndarray, float]] = []
    for raw_site, record in estimates.items():
        if record.get("status") != "solution":
            continue
        if raw_site not in truth_by_raw or raw_site not in display_by_raw:
            continue
        try:
            estimated = np.array(
                [float(record["estimated_x_m"]), float(record["estimated_y_m"])],
                dtype=np.float64,
            )
            heading_deg = float(record["estimated_heading_deg"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(estimated).all() or not np.isfinite(heading_deg):
            continue
        truth = truth_by_raw[raw_site]
        rows.append(
            (
                display_by_raw[raw_site],
                raw_site,
                estimated,
                heading_deg,
                truth,
                float(np.linalg.norm(estimated - truth)),
            )
        )
    rows.sort(key=lambda row: row[0])
    return rows


def select_best_pose_estimates(
    sites: np.ndarray,
    poses: np.ndarray,
    stage_estimates: dict[str, dict[int, dict[str, Any]]],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    """Select the lowest-XY-error available stage at each site for display.

    This is an evaluation-time selection because it compares estimates with
    the captured truth position. Exact ties prefer the later pipeline stage,
    matching the consolidated localization-results figure.
    """
    sites = np.asarray(sites, dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float64)
    if sites.ndim != 1 or poses.shape != (len(sites), 7):
        raise ValueError("sites and poses must have shapes (N,) and (N, 7)")

    truth_by_site = {int(site): pose[:2] for site, pose in zip(sites, poses)}
    selected: dict[int, dict[str, Any]] = {}
    counts = {stage: 0 for stage in LOCALIZATION_STAGES}
    for site in sorted(truth_by_site):
        candidates: list[tuple[float, int, str, dict[str, Any]]] = []
        for stage in LOCALIZATION_STAGES:
            record = stage_estimates.get(stage, {}).get(site)
            if record is None:
                continue
            try:
                estimated = np.asarray(
                    (record["estimated_x_m"], record["estimated_y_m"]),
                    dtype=np.float64,
                )
                heading_deg = float(record["estimated_heading_deg"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(estimated).all() or not np.isfinite(heading_deg):
                continue
            error = float(np.linalg.norm(estimated - truth_by_site[site]))
            candidates.append(
                (error, -STAGE_PRIORITY[stage], stage, record)
            )
        if not candidates:
            continue
        error, _priority, stage, record = min(candidates)
        chosen = dict(record)
        chosen["status"] = "solution"
        chosen["selected_stage"] = stage
        chosen["xy_error_m"] = error
        selected[site] = chosen
        counts[stage] += 1
    return selected, counts


def _write_site_mapping(
    path: Path,
    raw_sites: np.ndarray,
    xy: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["display_site", "raw_capture_site", "map_x_m", "map_y_m"])
        for display_site, (raw_site, position) in enumerate(zip(raw_sites, xy), 1):
            writer.writerow(
                [
                    display_site,
                    int(raw_site),
                    f"{position[0]:.6f}",
                    f"{position[1]:.6f}",
                ]
            )


def generate_localization_prediction_contour_map(
    *,
    config: Any,
    sites: np.ndarray,
    poses: np.ndarray,
    estimates: dict[int, dict[str, float | str]],
    dem: np.ndarray,
    output_path: Path,
    contour_interval_m: float = 2.0,
    error_label_threshold_m: float = 50.0,
    padding_m: float = 70.0,
    maximum_background_size: int = 1600,
) -> dict[str, Any]:
    """Create the grayscale contour map while preserving raw-site traceability."""
    if contour_interval_m <= 0:
        raise ValueError("contour_interval_m must be positive")
    if error_label_threshold_m < 0:
        raise ValueError("error_label_threshold_m cannot be negative")
    if padding_m < 0:
        raise ValueError("padding_m cannot be negative")
    if maximum_background_size <= 0:
        raise ValueError("maximum_background_size must be positive")

    sites = np.asarray(sites, dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float64)
    if sites.ndim != 1 or poses.shape != (len(sites), 7):
        raise ValueError("sites and poses must have shapes (N,) and (N, 7)")
    if len(sites) == 0:
        raise ValueError("At least one captured site is required")
    dem = np.asarray(dem)
    if dem.shape != config.orbital_raster.shape:
        raise ValueError(
            f"DEM shape {dem.shape} differs from configured {config.orbital_raster.shape}"
        )

    order = spatial_route_order(sites, poses[:, :2])
    raw_sites = sites[order]
    xy = poses[order, :2]
    prediction_rows = _prediction_rows(sites, poses, raw_sites, estimates)

    stride = max(1, int(np.ceil(max(dem.shape) / maximum_background_size)))
    shown_dem = dem[::stride, ::stride]
    x_centers, y_centers = config.orbital_raster.coordinates()
    shown_x = x_centers[::stride]
    shown_y = y_centers[::stride]
    finite = np.isfinite(shown_dem)
    low = float(np.nanmin(shown_dem))
    high = float(np.nanmax(shown_dem))
    first_level = np.ceil(low / contour_interval_m) * contour_interval_m
    last_level = np.floor(high / contour_interval_m) * contour_interval_m
    levels = np.arange(
        first_level,
        last_level + 0.5 * contour_interval_m,
        contour_interval_m,
    )

    fig, ax = plt.subplots(figsize=(11.5, 10.0), constrained_layout=True)
    ax.imshow(
        _grayscale_hillshade(
            shown_dem,
            config.orbital_raster.resolution_m * stride,
        ),
        extent=_dem_extent(config),
        origin="upper",
        interpolation="bilinear",
        zorder=0,
    )
    contours = ax.contour(
        shown_x,
        shown_y,
        np.ma.masked_where(~finite, shown_dem),
        levels=levels,
        colors="#1769c2",
        linewidths=0.62,
        alpha=0.82,
        zorder=2,
    )
    label_levels = levels[::4]
    if len(label_levels):
        ax.clabel(
            contours,
            levels=label_levels,
            inline=True,
            inline_spacing=3,
            fontsize=6.5,
            colors="#0d4f9a",
            fmt=lambda value: f"{value:g} m",
        )

    ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=3.5, alpha=0.82, zorder=5)
    ax.plot(
        xy[:, 0],
        xy[:, 1],
        color="#e32626",
        linewidth=1.8,
        zorder=6,
        label="Spatially ordered scan-site route",
    )
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=29,
        facecolors="#f7f7f7",
        edgecolors="black",
        linewidths=0.9,
        zorder=7,
        label="LiDAR scan site",
    )
    for display_site, (x, y) in enumerate(xy, 1):
        ax.annotate(
            f"{display_site:02d}",
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#101010",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "alpha": 0.72,
                "edgecolor": "none",
            },
            zorder=8,
        )

    for index, (_display, _raw, estimated, _heading, truth, _error) in enumerate(
        prediction_rows
    ):
        ax.plot(
            [truth[0], estimated[0]],
            [truth[1], estimated[1]],
            color="#ff8f4c",
            linewidth=1.05,
            linestyle=(0, (1.4, 2.2)),
            alpha=0.88,
            zorder=3,
            label="Site-to-prediction error" if index == 0 else None,
        )

    if prediction_rows:
        estimated_xy = np.stack([row[2] for row in prediction_rows])
        heading_rad = np.deg2rad([row[3] for row in prediction_rows])
        arrow_scale_points = np.vstack((xy, estimated_xy))
        plotted_span_m = float(np.max(np.ptp(arrow_scale_points, axis=0)))
        arrow_length_m = float(np.clip(0.015 * plotted_span_m, 3.0, 16.0))
        u = np.cos(heading_rad) * arrow_length_m
        v = np.sin(heading_rad) * arrow_length_m
        arrow_options = {
            "angles": "xy",
            "scale_units": "xy",
            "scale": 1,
            "headwidth": 4.2,
            "headlength": 5.0,
            "headaxislength": 4.4,
        }
        ax.quiver(
            estimated_xy[:, 0],
            estimated_xy[:, 1],
            u,
            v,
            width=0.0060,
            color="white",
            zorder=9,
            **arrow_options,
        )
        ax.quiver(
            estimated_xy[:, 0],
            estimated_xy[:, 1],
            u,
            v,
            width=0.0035,
            color="#e6007e",
            zorder=10,
            label="Best per-site pose and heading",
            **arrow_options,
        )
        for display_site, _raw, estimated, _heading, truth, error in prediction_rows:
            prediction_label = f"P{display_site:02d}"
            label_position = estimated
            label_offset = (7, 7)
            if error >= error_label_threshold_m:
                prediction_label += f"\n{error:.0f} m"
                label_position = 0.5 * (estimated + truth)
                label_offset = (0, 0)
            ax.annotate(
                prediction_label,
                label_position,
                xytext=label_offset,
                textcoords="offset points",
                ha="center" if error >= error_label_threshold_m else "left",
                va="center" if error >= error_label_threshold_m else "bottom",
                fontsize=7.0,
                color="#9c0057",
                weight="bold",
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": "white",
                    "alpha": 0.78,
                    "edgecolor": "none",
                },
                zorder=11,
            )

    plotted_xy = xy
    if prediction_rows:
        plotted_xy = np.vstack((xy, np.stack([row[2] for row in prediction_rows])))
    x_limits = (
        float(np.min(plotted_xy[:, 0]) - padding_m),
        float(np.max(plotted_xy[:, 0]) + padding_m),
    )
    y_limits = (
        float(np.min(plotted_xy[:, 1]) - padding_m),
        float(np.max(plotted_xy[:, 1]) + padding_m),
    )
    map_x_min, map_x_max, map_y_min, map_y_max = _dem_extent(config)
    ax.set_xlim(max(x_limits[0], map_x_min), min(x_limits[1], map_x_max))
    ax.set_ylim(max(y_limits[0], map_y_min), min(y_limits[1], map_y_max))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Map x / east [m]")
    ax.set_ylabel("Map y / north [m]")
    ax.set_title(
        f"{environment_display_name(config)} Localization Predictions",
        fontsize=16,
        weight="bold",
        pad=12,
    )
    ax.grid(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=4,
        framealpha=0.94,
        fontsize=8.5,
        borderaxespad=0.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    mapping_path = output_path.with_name(output_path.stem + "_site_mapping.csv")
    _write_site_mapping(mapping_path, raw_sites, xy)

    step_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return {
        "output_path": output_path,
        "mapping_path": mapping_path,
        "prediction_count": len(prediction_rows),
        "route_length_m": float(step_lengths.sum()) if len(step_lengths) else 0.0,
        "largest_route_link_m": float(step_lengths.max()) if len(step_lengths) else 0.0,
        "display_order": raw_sites.tolist(),
        "display_stride": stride,
    }


# Backward-compatible name for callers outside this repository. The input is
# now expected to contain whichever per-site estimates the caller wants shown.
generate_darces_prediction_contour_map = (
    generate_localization_prediction_contour_map
)

#!/usr/bin/env python3
"""Create a concise, presentation-ready localization results figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import load_pipeline_config, load_resolution_config


STAGES = ("darces", "ransac", "moga")
STAGE_LABELS = ("DARCES", "RANSAC", "MOGA")
COLORS = ("#ef6c00", "#7b1fa2", "#1565c0")
MARKERS = ("o", "s", "^")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "plots" / "results_summary" / "consolidated_localization_results.png",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_stage(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    sites = payload.get("sites", [])
    if not isinstance(sites, list):
        raise ValueError(f"{path}: 'sites' must be a list")
    return sites


def finite_xy(records: list[dict[str, object]], x_key: str, y_key: str) -> np.ndarray:
    points = []
    for record in records:
        x = record.get(x_key)
        y = record.get(y_key)
        if x is None or y is None:
            continue
        point = np.asarray((float(x), float(y)), dtype=np.float64)
        if np.isfinite(point).all():
            points.append(point)
    return np.stack(points) if points else np.empty((0, 2), dtype=np.float64)


def finite_errors(records: list[dict[str, object]]) -> np.ndarray:
    errors = []
    for record in records:
        value = record.get("xy_error_m")
        if value is not None and np.isfinite(float(value)):
            errors.append(float(value))
    return np.asarray(errors, dtype=np.float64)


def truth_xy(config, stage_records: dict[str, list[dict[str, object]]]) -> np.ndarray:
    for stage in reversed(STAGES):
        points = finite_xy(stage_records[stage], "truth_x_m", "truth_y_m")
        if len(points):
            return points
    poses = []
    odom_paths = sorted(
        (config.captures_path / "odom_scans").glob("odom_site_*.npy"),
        key=lambda item: int(item.stem.rsplit("_", 1)[1]),
    )
    for odom_path in odom_paths:
        pose = np.asarray(np.load(odom_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if len(pose) >= 2 and np.isfinite(pose[:2]).all():
            poses.append(pose[:2])
    return np.stack(poses) if poses else np.empty((0, 2), dtype=np.float64)


def plot_positions(axis, resolution: str, truth: np.ndarray,
                   records: dict[str, list[dict[str, object]]]) -> None:
    if len(truth):
        axis.plot(truth[:, 0], truth[:, 1], "k.-", linewidth=1.2,
                  markersize=4, label="Truth", zorder=1)
    for stage, label, color, marker in zip(
        STAGES, STAGE_LABELS, COLORS, MARKERS, strict=True
    ):
        points = finite_xy(records[stage], "estimated_x_m", "estimated_y_m")
        if not len(points):
            continue
        axis.scatter(
            points[:, 0], points[:, 1], s=38, marker=marker,
            facecolors="none" if stage == "darces" else color,
            edgecolors=color, linewidths=1.4, label=label, zorder=3,
        )
    axis.set_title(f"{resolution} position estimates", fontweight="bold")
    axis.set_xlabel("Map x / east [m]")
    axis.set_ylabel("Map y / north [m]")
    axis.grid(alpha=0.2)
    axis.set_aspect("equal", adjustable="datalim")


def plot_error_ranges(axis, resolution: str,
                      records: dict[str, list[dict[str, object]]]) -> None:
    tick_labels = []
    positive_exists = False
    for index, (stage, label, color, marker) in enumerate(
        zip(STAGES, STAGE_LABELS, COLORS, MARKERS, strict=True), start=1
    ):
        errors = finite_errors(records[stage])
        tick_labels.append(f"{label}\nn={len(errors)}")
        if not len(errors):
            axis.text(index, 0.5, "N/A", transform=axis.get_xaxis_transform(),
                      ha="center", va="center", color="0.45")
            continue
        positive_exists = positive_exists or bool(np.any(errors > 0.0))
        offsets = np.linspace(-0.075, 0.075, len(errors))
        axis.scatter(index + offsets, errors, s=22, marker=marker, color=color,
                     alpha=0.55, linewidths=0.5, zorder=2)
        minimum, median, maximum = np.min(errors), np.median(errors), np.max(errors)
        axis.vlines(index, minimum, maximum, color=color, linewidth=3, zorder=3)
        axis.hlines((minimum, maximum), index - 0.11, index + 0.11,
                    color=color, linewidth=2, zorder=3)
        axis.scatter(index, median, s=90, marker="D", color=color,
                     edgecolor="white", linewidth=1.2, zorder=4)
    if positive_exists:
        axis.set_yscale("log")
    axis.set_xticks((1, 2, 3), tick_labels)
    axis.set_xlim(0.55, 3.45)
    axis.set_title(f"{resolution} horizontal error range", fontweight="bold")
    axis.set_ylabel("Horizontal position error [m]")
    axis.grid(axis="y", alpha=0.2)


def make_figure(path: Path, dpi: int) -> None:
    resolutions = list(load_pipeline_config().available_resolutions)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure, axes = plt.subplots(2, len(resolutions), figsize=(16, 10))
    figure.suptitle("Localization Results Across DEM Resolutions",
                    fontsize=19, fontweight="bold", y=0.985)
    for column, resolution in enumerate(resolutions):
        config = load_resolution_config(resolution)
        records = {
            stage: load_stage(config.results_path / f"{stage}_all_sites.json")
            for stage in STAGES
        }
        plot_positions(axes[0, column], resolution, truth_xy(config, records), records)
        plot_error_ranges(axes[1, column], resolution, records)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper center", ncol=4,
                      frameon=False, bbox_to_anchor=(0.5, 0.948))
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.91), h_pad=2.5, w_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    output = args.output.resolve()
    make_figure(output, args.dpi)
    print("Presentation results figure")
    print("---------------------------")
    print(f"PNG: {output}")


if __name__ == "__main__":
    main()

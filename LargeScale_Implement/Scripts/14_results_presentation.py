#!/usr/bin/env python3
"""Create a concise, presentation-ready localization results summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import load_pipeline_config, load_resolution_config
from traversal_presentation import environment_display_name


STAGES = ("darces", "ransac", "moga")
STAGE_LABELS = {"darces": "DARCES", "ransac": "RANSAC", "moga": "MOGA"}
STAGE_COLORS = {"darces": "#ef6c00", "ransac": "#7b1fa2", "moga": "#1565c0"}
STAGE_MARKERS = {"darces": "o", "ransac": "s", "moga": "^"}
STAGE_PRIORITY = {stage: index for index, stage in enumerate(STAGES)}
SUMMARY_COLOR = "#087f8c"
COUNT_COLOR = "#90a4ae"


def resolution_label(resolution: str) -> str:
    """Return a compact human-readable DEM resolution label."""
    value = resolution.removesuffix("m").replace("p", ".")
    return f"{value} m/px"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "plots"
            / "results_summary"
            / "consolidated_localization_results.png"
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Statistics CSV (default: same path as --output with .csv suffix)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        help=(
            "Read per-resolution result directories from this root instead "
            "of the active workspace results directory"
        ),
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


def finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def record_xy(record: dict[str, object], prefix: str = "estimated") -> np.ndarray | None:
    x = finite_float(record.get(f"{prefix}_x_m"))
    y = finite_float(record.get(f"{prefix}_y_m"))
    if x is None or y is None:
        return None
    return np.asarray((x, y), dtype=np.float64)


def records_by_site(records: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    output: dict[int, dict[str, object]] = {}
    for record in records:
        if "site" not in record:
            continue
        site = int(record["site"])
        if site in output:
            raise ValueError(f"Duplicate result for site {site:02d}")
        output[site] = record
    return output


def select_oracle_best(
    stage_records: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Select the lowest-truth-error stage for each site for evaluation only.

    Exact error ties prefer the later pipeline stage (MOGA, then RANSAC) so the
    stage breakdown does not misleadingly attribute unchanged downstream poses
    to DARCES.
    """
    indexed = {stage: records_by_site(stage_records[stage]) for stage in STAGES}
    sites = sorted(set().union(*(set(records) for records in indexed.values())))
    selected: list[dict[str, object]] = []
    for site in sites:
        candidates: list[tuple[float, int, str, dict[str, object]]] = []
        for stage in STAGES:
            record = indexed[stage].get(site)
            if record is None or record_xy(record) is None:
                continue
            error = finite_float(record.get("xy_error_m"))
            if error is None:
                continue
            candidates.append((error, -STAGE_PRIORITY[stage], stage, record))
        if not candidates:
            continue
        error, _priority, stage, record = min(candidates)
        item = dict(record)
        item["selected_stage"] = stage
        item["xy_error_m"] = error
        selected.append(item)
    return selected


def truth_by_site(
    config,
    stage_records: dict[str, list[dict[str, object]]],
) -> dict[int, np.ndarray]:
    truth: dict[int, np.ndarray] = {}
    for stage in reversed(STAGES):
        for site, record in records_by_site(stage_records[stage]).items():
            point = record_xy(record, "truth")
            if point is not None:
                truth.setdefault(site, point)
    if truth:
        return truth
    for path in sorted(
        (config.captures_path / "odom_scans").glob("odom_site_*.npy"),
        key=lambda item: int(item.stem.rsplit("_", 1)[1]),
    ):
        pose = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if len(pose) >= 2 and np.isfinite(pose[:2]).all():
            truth[int(path.stem.rsplit("_", 1)[1])] = pose[:2]
    return truth


def error_statistics(errors: np.ndarray) -> dict[str, float | int | None]:
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return {
            "solved_sites": 0,
            "minimum_xy_error_m": None,
            "p10_xy_error_m": None,
            "p25_xy_error_m": None,
            "median_xy_error_m": None,
            "mean_xy_error_m": None,
            "std_xy_error_m": None,
            "p75_xy_error_m": None,
            "p90_xy_error_m": None,
            "maximum_xy_error_m": None,
        }
    percentiles = np.percentile(errors, (10, 25, 50, 75, 90))
    return {
        "solved_sites": int(len(errors)),
        "minimum_xy_error_m": float(np.min(errors)),
        "p10_xy_error_m": float(percentiles[0]),
        "p25_xy_error_m": float(percentiles[1]),
        "median_xy_error_m": float(percentiles[2]),
        "mean_xy_error_m": float(np.mean(errors)),
        "std_xy_error_m": float(np.std(errors)),
        "p75_xy_error_m": float(percentiles[3]),
        "p90_xy_error_m": float(percentiles[4]),
        "maximum_xy_error_m": float(np.max(errors)),
    }


def build_summary(
    resolution: str,
    total_sites: int,
    selected: list[dict[str, object]],
) -> dict[str, object]:
    errors = np.asarray([float(record["xy_error_m"]) for record in selected])
    counts = {
        stage: sum(record["selected_stage"] == stage for record in selected)
        for stage in STAGES
    }
    return {
        "resolution": resolution,
        "total_sites": total_sites,
        **error_statistics(errors),
        "darces_selected": counts["darces"],
        "ransac_selected": counts["ransac"],
        "moga_selected": counts["moga"],
    }


def plot_positions(
    axis,
    resolution: str,
    truth: dict[int, np.ndarray],
    selected: list[dict[str, object]],
) -> None:
    if truth:
        ordered_truth = np.vstack([truth[site] for site in sorted(truth)])
        axis.plot(
            ordered_truth[:, 0], ordered_truth[:, 1], "k.-", linewidth=1.15,
            markersize=2.8, alpha=0.8, zorder=1,
        )
    for stage in STAGES:
        points = [
            record_xy(record)
            for record in selected
            if record["selected_stage"] == stage
        ]
        points = [point for point in points if point is not None]
        if not points:
            continue
        points_array = np.vstack(points)
        axis.scatter(
            points_array[:, 0], points_array[:, 1], s=31,
            marker=STAGE_MARKERS[stage], color=STAGE_COLORS[stage],
            edgecolors="white", linewidths=0.45, alpha=0.9, zorder=3,
        )
    axis.set_title(
        f"{resolution}  ·  solved {len(selected)}/{len(truth)}",
        fontweight="bold", pad=7,
    )
    axis.set_xlabel("Map x / east [m]")
    axis.set_ylabel("Map y / north [m]")
    axis.grid(alpha=0.16)
    axis.set_aspect("equal", adjustable="datalim")


def plot_distribution(
    axis,
    resolution: str,
    selected: list[dict[str, object]],
    total_sites: int,
) -> None:
    errors = np.asarray([float(record["xy_error_m"]) for record in selected])
    if len(errors) == 0:
        axis.text(0.5, 0.5, "No pose solution", transform=axis.transAxes,
                  ha="center", va="center", color="0.40", fontsize=13,
                  fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(
            f"{resolution_label(resolution)} — 0/{total_sites} solved",
            fontsize=13, fontweight="bold", pad=26,
        )
        return
    stats = error_statistics(errors)
    p10 = float(stats["p10_xy_error_m"])
    p25 = float(stats["p25_xy_error_m"])
    median = float(stats["median_xy_error_m"])
    mean = float(stats["mean_xy_error_m"])
    std = float(stats["std_xy_error_m"])
    p75 = float(stats["p75_xy_error_m"])
    p90 = float(stats["p90_xy_error_m"])

    rng = np.random.default_rng(1701 + STAGE_PRIORITY["moga"] + len(errors))
    jitter = rng.uniform(-0.105, 0.105, size=len(errors))
    colors = [STAGE_COLORS[str(record["selected_stage"])] for record in selected]
    axis.scatter(
        jitter, errors, s=38, c=colors, alpha=0.72,
        edgecolors="white", linewidths=0.45, zorder=2,
    )
    axis.add_patch(Rectangle(
        (-0.16, p25), 0.32, max(p75 - p25, np.finfo(float).eps),
        facecolor=SUMMARY_COLOR, edgecolor=SUMMARY_COLOR,
        alpha=0.23, linewidth=2.5, zorder=3,
    ))
    axis.vlines(0.0, p10, p90, color=SUMMARY_COLOR, linewidth=3.0, zorder=3)
    axis.hlines((p10, p90), -0.08, 0.08, color=SUMMARY_COLOR,
                linewidth=2.6, zorder=3)
    lower_std = min(std, mean * 0.98) if mean > 0.0 else 0.0
    axis.errorbar(
        0.0, mean, yerr=np.asarray([[lower_std], [std]]), fmt="o",
        color="#d84315", ecolor="#d84315", capsize=6,
        elinewidth=2.2, capthick=2.2, markersize=8,
        markeredgecolor="white", zorder=5,
    )
    axis.scatter(0.0, median, marker="D", s=95, color="#003f5c",
                 edgecolor="white", linewidth=1.1, zorder=6)
    axis.text(
        0.5, 1.012,
        f"Median {median:.1f} m · Mean {mean:.1f} m\nσ {std:.1f} m",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=8.5,
        color="#37474f", fontweight="bold", clip_on=False,
        linespacing=1.05,
    )
    if np.all(errors > 0.0):
        axis.set_yscale("log")
    axis.set_xlim(-0.28, 0.28)
    axis.set_xticks([])
    axis.set_title(
        f"{resolution_label(resolution)} — {len(errors)}/{total_sites} solved",
        fontsize=13, fontweight="bold", pad=26,
    )
    axis.set_ylabel("Horizontal error [m]", fontsize=11)
    axis.tick_params(axis="y", labelsize=10)
    axis.grid(axis="y", alpha=0.28, linewidth=0.9)


def plot_resolution_summary(axis, summaries: list[dict[str, object]]) -> None:
    x = np.arange(len(summaries), dtype=np.float64)
    labels = [str(row["resolution"]) for row in summaries]
    solved = np.asarray([int(row["solved_sites"]) for row in summaries])
    totals = np.asarray([int(row["total_sites"]) for row in summaries])

    def values(key: str) -> np.ndarray:
        return np.asarray([
            np.nan if row[key] is None else float(row[key]) for row in summaries
        ])

    p10, p90 = values("p10_xy_error_m"), values("p90_xy_error_m")
    median, mean, std = (
        values("median_xy_error_m"), values("mean_xy_error_m"),
        values("std_xy_error_m"),
    )
    valid_band = np.isfinite(p10) & np.isfinite(p90)
    if np.any(valid_band):
        axis.fill_between(
            x, p10, p90, where=valid_band, color=SUMMARY_COLOR,
            alpha=0.22, interpolate=False, label="P10–P90 spread",
        )
        axis.plot(x, p10, color=SUMMARY_COLOR, linewidth=1.6, alpha=0.70)
        axis.plot(x, p90, color=SUMMARY_COLOR, linewidth=1.6, alpha=0.70)
    axis.plot(x, median, "D-", color="#003f5c", linewidth=3.2,
              markersize=9, label="Median", zorder=4)
    axis.plot(x, mean, "o--", color="#d84315", linewidth=2.5,
              markersize=7.5, label="Mean", zorder=4)
    valid_mean = np.isfinite(mean) & np.isfinite(std)
    if np.any(valid_mean):
        lower = np.minimum(std[valid_mean], mean[valid_mean] * 0.98)
        axis.errorbar(
            x[valid_mean], mean[valid_mean],
            yerr=np.vstack((lower, std[valid_mean])), fmt="none",
            ecolor="#d84315", alpha=0.72, capsize=5,
            capthick=1.8, linewidth=1.8, label="Mean ± 1σ",
        )
    if np.any(np.isfinite(median) & (median > 0.0)):
        axis.set_yscale("log")
    axis.set_xticks(x, [resolution_label(label) for label in labels])
    axis.set_xlabel("DEM resolution", fontsize=12)
    axis.set_ylabel("Horizontal error [m]", fontsize=12)
    axis.tick_params(axis="both", labelsize=10)
    axis.set_title(
        "Best per-site horizontal error and solved-site count",
        fontsize=14, fontweight="bold", loc="left", pad=9,
    )
    axis.grid(axis="y", alpha=0.28, linewidth=0.9)

    count_axis = axis.twinx()
    bars = count_axis.bar(
        x, solved, width=0.58, color=COUNT_COLOR, alpha=0.20,
        edgecolor="none", zorder=0,
    )
    count_axis.set_ylim(0.0, max(float(np.max(totals)) * 1.20, 1.0))
    count_axis.set_ylabel("Solved sites [count]", color="#607d8b")
    count_axis.tick_params(axis="y", colors="#607d8b")
    count_axis.spines["right"].set_color("#b0bec5")
    for bar, count, total in zip(bars, solved, totals, strict=True):
        count_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(float(np.max(totals)) * 0.018, 0.2),
            f"{count}/{total}", ha="center", va="bottom",
            color="#455a64", fontsize=10, fontweight="bold",
        )
    axis.legend(loc="upper left", ncol=4, frameon=False, fontsize=10)


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0]) if summaries else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def make_figure(
    path: Path,
    summary_csv: Path,
    dpi: int,
    results_root: Path | None = None,
) -> None:
    resolutions = list(load_pipeline_config().available_resolutions)
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    figure = plt.figure(figsize=(18, 8.4), constrained_layout=False)
    grid = figure.add_gridspec(
        2, len(resolutions), height_ratios=(1.0, 0.88),
        hspace=0.34, wspace=0.34,
    )
    distribution_axes = [
        figure.add_subplot(grid[0, index]) for index in range(len(resolutions))
    ]
    summary_axis = figure.add_subplot(grid[1, :])

    summaries: list[dict[str, object]] = []
    for column, resolution in enumerate(resolutions):
        config = load_resolution_config(resolution)
        resolution_results_path = (
            config.results_path
            if results_root is None
            else results_root / f"{resolution}_px"
        )
        records = {
            stage: load_stage(resolution_results_path / f"{stage}_all_sites.json")
            for stage in STAGES
        }
        truth = truth_by_site(config, records)
        selected = select_oracle_best(records)
        summaries.append(build_summary(resolution, len(truth), selected))
        plot_distribution(
            distribution_axes[column], resolution, selected, len(truth)
        )

    plot_resolution_summary(summary_axis, summaries)
    write_summary_csv(summary_csv, summaries)

    environment_name = environment_display_name(load_resolution_config(resolutions[0]))
    figure.suptitle(
        f"{environment_name} Localization Accuracy vs DEM Resolution",
        fontsize=21, fontweight="bold", y=0.985,
    )
    stage_handles = [
        Line2D([], [], linestyle="none", marker=STAGE_MARKERS[stage],
               color=STAGE_COLORS[stage], markersize=8,
               label=STAGE_LABELS[stage])
        for stage in STAGES
    ]
    statistic_handles = [
        Patch(facecolor=SUMMARY_COLOR, edgecolor=SUMMARY_COLOR, alpha=0.18,
              label="IQR / P10–P90"),
        Line2D([], [], linestyle="none", marker="D", color="#003f5c",
               markersize=7, label="Median"),
        Line2D([], [], linestyle="none", marker="o", color="#d84315",
               markersize=7, label="Mean ± 1σ"),
    ]
    distribution_axes[0].legend(
        handles=stage_handles + statistic_handles,
        loc="lower center", ncol=2, frameon=True,
        bbox_to_anchor=(0.5, 0.035), fontsize=8.3,
        framealpha=0.92, borderpad=0.55, columnspacing=0.9,
        handletextpad=0.55, labelspacing=0.45,
    )
    figure.text(
        0.5, 0.018,
        "Best/site = lowest evaluation error among available DARCES, RANSAC and MOGA poses.",
        ha="center", va="bottom", fontsize=10, color="#5d4037",
    )
    figure.subplots_adjust(left=0.052, right=0.962, top=0.865, bottom=0.095)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    output = args.output.resolve()
    summary_csv = (
        args.summary_csv.resolve()
        if args.summary_csv is not None
        else output.with_suffix(".csv")
    )
    results_root = args.results_root.resolve() if args.results_root else None
    make_figure(output, summary_csv, args.dpi, results_root)
    print("Presentation results summary")
    print("----------------------------")
    print(f"PNG: {output}")
    print(f"CSV: {summary_csv}")
    if results_root is not None:
        print(f"Results root: {results_root}")


if __name__ == "__main__":
    main()

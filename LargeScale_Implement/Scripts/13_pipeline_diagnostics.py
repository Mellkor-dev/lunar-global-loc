#!/usr/bin/env python3
"""Generate reproducible DARCES/RANSAC/MOGA diagnostics for one or all resolutions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import add_resolution_argument, load_pipeline_config, load_resolution_config


STAGES = ("darces", "ransac", "moga")
STAGE_LABELS = {"darces": "DARCES", "ransac": "RANSAC", "moga": "MOGA"}
COLORS = {"darces": "#ef6c00", "ransac": "#6a1b9a", "moga": "#1565c0"}
MARKERS = {"darces": "o", "ransac": "s", "moga": "^"}
LINE_STYLES = {"darces": "--", "ransac": ":", "moga": "-"}
SITE_OFFSETS = {"darces": -0.16, "ransac": 0.0, "moga": 0.16}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_resolution_argument(parser, include_all=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "diagnostics",
        help="Directory for combined CSV and Markdown tables",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_stage(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    sites = payload.get("sites", [])
    if not isinstance(sites, list):
        raise ValueError(f"{path}: 'sites' must be a list")
    return {int(record["site"]): record for record in sites}


def finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def stage_values(record: dict[str, object] | None) -> dict[str, object]:
    if record is None:
        return {
            "status": "stage_not_run",
            "available": False,
            "source": None,
            "x_m": None,
            "y_m": None,
            "z_m": None,
            "heading_deg": None,
            "xy_error_m": None,
            "z_error_m": None,
            "heading_error_deg": None,
        }
    x = finite_float(record.get("estimated_x_m"))
    y = finite_float(record.get("estimated_y_m"))
    return {
        "status": str(record.get("status", "unknown")),
        "available": x is not None and y is not None,
        "source": record.get("estimate_source", record.get("initial_source")),
        "x_m": x,
        "y_m": y,
        "z_m": finite_float(record.get("estimated_z_m")),
        "heading_deg": finite_float(record.get("estimated_heading_deg")),
        "xy_error_m": finite_float(record.get("xy_error_m")),
        "z_error_m": finite_float(record.get("z_error_m")),
        "heading_error_deg": finite_float(record.get("heading_error_deg")),
    }


def improvement(before: float | None, after: float | None) -> tuple[float | None, str]:
    if before is None or after is None:
        return None, "not_comparable"
    reduction = before - after
    tolerance = 1e-9
    if reduction > tolerance:
        return reduction, "improved"
    if reduction < -tolerance:
        return reduction, "worsened"
    return reduction, "unchanged"


def diagnostic_rows(resolution: str) -> list[dict[str, object]]:
    config = load_resolution_config(resolution)
    stage_records = {
        stage: load_stage(config.results_path / f"{stage}_all_sites.json")
        for stage in STAGES
    }
    site_numbers = sorted(set().union(*(records.keys() for records in stage_records.values())))
    rows: list[dict[str, object]] = []
    for site in site_numbers:
        values = {stage: stage_values(stage_records[stage].get(site)) for stage in STAGES}
        truth_record = next(
            (stage_records[stage].get(site) for stage in reversed(STAGES)
             if stage_records[stage].get(site, {}).get("truth_x_m") is not None),
            {},
        )
        row: dict[str, object] = {
            "resolution": resolution,
            "site": site,
            "truth_x_m": finite_float(truth_record.get("truth_x_m")),
            "truth_y_m": finite_float(truth_record.get("truth_y_m")),
            "truth_z_m": finite_float(truth_record.get("truth_z_m")),
        }
        for stage in STAGES:
            for key, value in values[stage].items():
                row[f"{stage}_{key}"] = value
        dr_delta, dr_result = improvement(
            values["darces"].get("xy_error_m"), values["ransac"].get("xy_error_m")
        )
        rm_delta, rm_result = improvement(
            values["ransac"].get("xy_error_m"), values["moga"].get("xy_error_m")
        )
        dm_delta, dm_result = improvement(
            values["darces"].get("xy_error_m"), values["moga"].get("xy_error_m")
        )
        row.update(
            {
                "darces_to_ransac_error_reduction_m": dr_delta,
                "darces_to_ransac_result": dr_result,
                "ransac_to_moga_error_reduction_m": rm_delta,
                "ransac_to_moga_result": rm_result,
                "darces_to_moga_error_reduction_m": dm_delta,
                "darces_to_moga_result": dm_result,
                "best_available_stage": min(
                    (
                        (values[stage].get("xy_error_m"), STAGE_LABELS[stage])
                        for stage in STAGES
                        if values[stage].get("xy_error_m") is not None
                    ),
                    default=(None, "none"),
                    key=lambda item: float("inf") if item[0] is None else item[0],
                )[1],
            }
        )
        rows.append(row)
    return rows


def display(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    columns = (
        "resolution", "site",
        "darces_status", "darces_x_m", "darces_y_m", "darces_xy_error_m",
        "ransac_status", "ransac_x_m", "ransac_y_m", "ransac_xy_error_m",
        "darces_to_ransac_result",
        "moga_status", "moga_x_m", "moga_y_m", "moga_xy_error_m",
        "ransac_to_moga_result", "best_available_stage",
    )
    labels = (
        "Resolution", "Site", "DARCES status", "DARCES x", "DARCES y", "DARCES error",
        "RANSAC status", "RANSAC x", "RANSAC y", "RANSAC error", "D→R",
        "MOGA status", "MOGA x", "MOGA y", "MOGA error", "R→M", "Best",
    )
    lines = [
        "# Pipeline localization diagnostics",
        "",
        "Errors are horizontal radial errors in metres. An em dash means no feature-derived pose was available.",
        "Odometry position is used only as evaluation truth.",
        "",
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(display(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    present = {str(row["resolution"]) for row in rows}
    resolutions = [
        resolution
        for resolution in load_pipeline_config().available_resolutions
        if resolution in present
    ]
    for resolution in resolutions:
        selected = [row for row in rows if row["resolution"] == resolution]
        for stage in STAGES:
            errors = np.asarray(
                [row[f"{stage}_xy_error_m"] for row in selected if row[f"{stage}_xy_error_m"] is not None],
                dtype=np.float64,
            )
            summaries.append(
                {
                    "resolution": resolution,
                    "stage": STAGE_LABELS[stage],
                    "available": int(len(errors)),
                    "total_sites": len(selected),
                    "availability_percent": 100.0 * len(errors) / len(selected) if selected else 0.0,
                    "median_xy_error_m": float(np.median(errors)) if len(errors) else None,
                    "mean_xy_error_m": float(np.mean(errors)) if len(errors) else None,
                    "rmse_xy_m": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                    "maximum_xy_error_m": float(np.max(errors)) if len(errors) else None,
                    "within_50m": int(np.count_nonzero(errors <= 50.0)),
                    "within_50m_percent_of_available": (
                        100.0 * np.count_nonzero(errors <= 50.0) / len(errors) if len(errors) else None
                    ),
                }
            )
    return summaries


def plot_paper_style_errors(path: Path, resolution: str, rows: list[dict[str, object]]) -> None:
    figure, axis = plt.subplots(figsize=(13, 5.5))
    sites = np.asarray([int(row["site"]) for row in rows])
    positive_error_exists = False
    for stage in STAGES:
        errors = np.asarray(
            [np.nan if row[f"{stage}_xy_error_m"] is None else row[f"{stage}_xy_error_m"] for row in rows],
            dtype=np.float64,
        )
        positive_error_exists = positive_error_exists or bool(np.any(errors > 0.0))
        axis.plot(
            sites + SITE_OFFSETS[stage],
            errors,
            marker=MARKERS[stage],
            linestyle=LINE_STYLES[stage],
            linewidth=1.35,
            markersize=5,
            markerfacecolor="none" if stage == "darces" else COLORS[stage],
            markeredgewidth=1.4,
            color=COLORS[stage],
            label=STAGE_LABELS[stage],
        )
    axis.axhline(50.0, color="0.35", linestyle="--", linewidth=1, label="50 m criterion")
    if positive_error_exists:
        axis.set_yscale("log")
    else:
        axis.text(
            0.5, 0.5, "No feature-derived pose estimates available",
            transform=axis.transAxes, ha="center", va="center", fontsize=12,
        )
    axis.set_xticks(sites)
    axis.set_xlabel("Capture site")
    axis.set_ylabel("Horizontal radial position error [m]")
    axis.set_title(f"{resolution}: stage-by-stage localization error (paper Figure 24/26 style)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=4)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_site_table(path: Path, resolution: str, rows: list[dict[str, object]]) -> None:
    labels = (
        "Site", "DARCES status", "D err [m]", "RANSAC status", "R err [m]",
        "D→R", "MOGA status", "M err [m]", "R→M", "Best",
    )
    cells = []
    for row in rows:
        cells.append(
            [
                f"{int(row['site']):02d}",
                display(row["darces_status"]),
                display(row["darces_xy_error_m"]),
                display(row["ransac_status"]),
                display(row["ransac_xy_error_m"]),
                display(row["darces_to_ransac_result"]),
                display(row["moga_status"]),
                display(row["moga_xy_error_m"]),
                display(row["ransac_to_moga_result"]),
                display(row["best_available_stage"]),
            ]
        )
    figure, axis = plt.subplots(figsize=(17, max(8.5, 0.36 * len(cells) + 1.6)))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.22)
    for column in range(len(labels)):
        table[(0, column)].set_facecolor("#d9e2f3")
        table[(0, column)].set_text_props(weight="bold")
    result_columns = (5, 8)
    for row_index, row in enumerate(rows, start=1):
        for column, key in zip(
            result_columns,
            ("darces_to_ransac_result", "ransac_to_moga_result"),
            strict=True,
        ):
            result = row[key]
            color = {"improved": "#d9ead3", "worsened": "#f4cccc", "unchanged": "#eeeeee"}.get(result)
            if color:
                table[(row_index, column)].set_facecolor(color)
    axis.set_title(
        f"{resolution}: per-site DARCES → RANSAC → MOGA diagnostics",
        pad=14,
        weight="bold",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_component_errors(path: Path, resolution: str, rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    sites = np.asarray([int(row["site"]) for row in rows])
    components = (("x_m", "x"), ("y_m", "y"), ("z_m", "z"))
    for axis, (component, label) in zip(axes, components, strict=True):
        for stage in STAGES:
            values = []
            for row in rows:
                estimate = row[f"{stage}_{component}"]
                truth = row[f"truth_{component}"]
                values.append(np.nan if estimate is None or truth is None else abs(estimate - truth))
            axis.plot(
                sites + SITE_OFFSETS[stage],
                values,
                marker=MARKERS[stage],
                linestyle=LINE_STYLES[stage],
                linewidth=1.15,
                markersize=4,
                markerfacecolor="none" if stage == "darces" else COLORS[stage],
                markeredgewidth=1.2,
                color=COLORS[stage],
                label=STAGE_LABELS[stage],
            )
        axis.set_ylabel(f"|{label} error| [m]")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3)
    axes[0].set_title(f"{resolution}: position-component errors (paper Figure 18 style)")
    axes[-1].set_xticks(sites)
    axes[-1].set_xlabel("Capture site")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_summary_table(path: Path, summaries: list[dict[str, object]]) -> None:
    columns = ("resolution", "stage", "available", "median_xy_error_m", "mean_xy_error_m", "rmse_xy_m", "within_50m")
    labels = ("Resolution", "Stage", "Available", "Median [m]", "Mean [m]", "RMSE [m]", "≤50 m")
    cells = []
    for row in summaries:
        cells.append([
            display(row[column]) if column != "available" else f"{row['available']}/{row['total_sites']}"
            for column in columns
        ])
    figure_height = max(3.0, 0.42 * len(cells) + 1.5)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for column in range(len(labels)):
        table[(0, column)].set_facecolor("#d9e2f3")
        table[(0, column)].set_text_props(weight="bold")
    axis.set_title("Localization performance summary", pad=16, weight="bold")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_arguments()
    resolutions = (
        load_pipeline_config().available_resolutions
        if args.resolution == "all"
        else (args.resolution,)
    )
    all_rows: list[dict[str, object]] = []
    for resolution in resolutions:
        rows = diagnostic_rows(resolution)
        all_rows.extend(rows)
        if not args.no_plots and rows:
            plot_root = load_resolution_config(resolution).plots_path / "diagnostics"
            plot_paper_style_errors(plot_root / "pipeline_radial_error.png", resolution, rows)
            plot_component_errors(plot_root / "pipeline_component_errors.png", resolution, rows)
            plot_site_table(plot_root / "pipeline_site_table.png", resolution, rows)
    if not all_rows:
        raise ValueError("No diagnostic result files were found")
    summaries = summary_rows(all_rows)
    output_root = args.output_root.resolve()
    write_csv(output_root / "pipeline_diagnostics_all_sites.csv", all_rows)
    write_markdown(output_root / "pipeline_diagnostics_all_sites.md", all_rows)
    write_csv(output_root / "pipeline_summary.csv", summaries)
    if not args.no_plots:
        plot_summary_table(output_root / "pipeline_summary_table.png", summaries)
    print("Pipeline diagnostics")
    print("--------------------")
    print(f"Resolutions: {', '.join(resolutions)}")
    print(f"Site rows:   {len(all_rows)}")
    print(f"Full CSV:    {output_root / 'pipeline_diagnostics_all_sites.csv'}")
    print(f"Markdown:    {output_root / 'pipeline_diagnostics_all_sites.md'}")
    print(f"Summary CSV: {output_root / 'pipeline_summary.csv'}")
    if not args.no_plots:
        print(f"Summary PNG: {output_root / 'pipeline_summary_table.png'}")


if __name__ == "__main__":
    main()

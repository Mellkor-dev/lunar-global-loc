#!/usr/bin/env python3
"""Compare localization using 1.5 m-source and 0.25 m-source coarse DEMs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = (("5m", "5m_refined"), ("10m", "10m_refined"))
STAGES = ("darces", "ransac", "moga")
SOURCE_LABELS = ("1.5 m source", "0.25 m source")
SOURCE_COLORS = ("#555555", "#1565c0")


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def finite_errors(payload: dict[str, object]) -> dict[int, float]:
    result: dict[int, float] = {}
    for site in payload["sites"]:
        value = site.get("xy_error_m")
        if isinstance(value, (int, float)) and np.isfinite(value):
            result[int(site["site"])] = float(value)
    return result


def summary(errors: dict[int, float]) -> dict[str, object]:
    values = np.asarray(list(errors.values()), dtype=np.float64)
    return {
        "available_pose_count": len(values),
        "median_xy_error_m": float(np.median(values)) if len(values) else None,
        "maximum_xy_error_m": float(values.max()) if len(values) else None,
        "within_10m": int(np.count_nonzero(values <= 10.0)),
        "within_50m": int(np.count_nonzero(values <= 50.0)),
        "over_100m": int(np.count_nonzero(values > 100.0)),
    }


def plot_final_accuracy(path: Path) -> None:
    """Plot paired final MOGA errors and absolute error-range counts."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": (2.2, 1.0)},
    )
    error_bins = (0.0, 10.0, 50.0, 100.0, np.inf)
    error_labels = ("0–10 m", "10–50 m", "50–100 m", ">100 m")
    bin_colors = ("#2e7d32", "#9ccc65", "#f9a825", "#c62828")

    for column, (baseline, refined) in enumerate(COMPARISONS):
        profiles = (baseline, refined)
        errors_by_source: list[dict[int, float]] = []
        for profile in profiles:
            payload = load_json(
                PROJECT_ROOT
                / "results"
                / f"{profile}_px"
                / "moga_all_sites.json"
            )
            errors_by_source.append(finite_errors(payload))

        axis = axes[0, column]
        common_sites = sorted(errors_by_source[0].keys() & errors_by_source[1].keys())
        sites = np.asarray(common_sites, dtype=np.int64)
        for index, (errors, label, color) in enumerate(
            zip(errors_by_source, SOURCE_LABELS, SOURCE_COLORS, strict=True)
        ):
            values = np.asarray([errors[site] for site in common_sites])
            offset = -0.12 if index == 0 else 0.12
            axis.plot(
                sites + offset,
                values,
                marker="o" if index == 0 else "s",
                linestyle="none",
                markersize=5.5,
                color=color,
                label=f"{label} — median {np.median(values):.1f} m",
            )
        axis.axhline(50.0, color="#ef6c00", linestyle="--", linewidth=1.4)
        axis.text(
            28.4,
            50.0,
            "50 m",
            color="#ef6c00",
            va="center",
            ha="left",
            fontsize=9,
        )
        axis.set_yscale("log")
        axis.set_xlim(0.5, 29.2)
        axis.set_ylim(0.5, 2000.0)
        axis.set_xticks(np.arange(1, 29))
        axis.set_title(f"{baseline.removesuffix('m')} m final localization error")
        axis.set_xlabel("Site")
        axis.set_ylabel("Horizontal position error [m]")
        axis.grid(alpha=0.22, which="both")
        axis.legend(loc="upper left", fontsize=9)

        count_axis = axes[1, column]
        bottoms = np.zeros(2, dtype=np.int64)
        for bin_index, (low, high, label, color) in enumerate(
            zip(error_bins[:-1], error_bins[1:], error_labels, bin_colors, strict=True)
        ):
            counts = []
            for errors in errors_by_source:
                values = np.asarray(list(errors.values()), dtype=np.float64)
                if np.isinf(high):
                    count = np.count_nonzero(values > low)
                elif low == 0.0:
                    count = np.count_nonzero((values >= low) & (values <= high))
                else:
                    count = np.count_nonzero((values > low) & (values <= high))
                counts.append(int(count))
            bars = count_axis.bar(
                np.arange(2),
                counts,
                bottom=bottoms,
                color=color,
                width=0.62,
                label=label,
            )
            for bar, count, bottom in zip(bars, counts, bottoms, strict=True):
                if count:
                    count_axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom + count / 2,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if bin_index == 3 else "black",
                    )
            bottoms += np.asarray(counts)
        count_axis.set_xticks(np.arange(2), SOURCE_LABELS)
        count_axis.set_ylabel("Number of sites")
        count_axis.set_ylim(0, max(bottoms) + 1)
        count_axis.set_title("Absolute error-range counts")
        count_axis.grid(axis="y", alpha=0.2)
        if column == 1:
            count_axis.legend(loc="upper right", fontsize=9)

    figure.suptitle("Localization accuracy by coarse-DEM source", fontsize=17)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    output_directory = PROJECT_ROOT / "results" / "dem_source_comparison"
    output_directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"comparisons": {}}
    rows: list[dict[str, object]] = []

    for baseline, refined in COMPARISONS:
        baseline_dem = np.asarray(
            np.load(
                PROJECT_ROOT / "DEM" / f"{baseline}_px" / f"orbital_dem_{baseline}.npy",
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        refined_dem = np.asarray(
            np.load(
                PROJECT_ROOT / "DEM" / f"{refined}_px" / f"orbital_dem_{refined}.npy",
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        residual = refined_dem - baseline_dem
        comparison: dict[str, object] = {
            "baseline_profile": baseline,
            "refined_source_profile": refined,
            "dem_difference": {
                "mean_m": float(residual.mean()),
                "mae_m": float(np.abs(residual).mean()),
                "rmse_m": float(np.sqrt(np.mean(residual**2))),
                "absolute_p95_m": float(np.percentile(np.abs(residual), 95)),
                "absolute_max_m": float(np.abs(residual).max()),
            },
            "stages": {},
        }
        for stage in STAGES:
            baseline_payload = load_json(
                PROJECT_ROOT / "results" / f"{baseline}_px" / f"{stage}_all_sites.json"
            )
            refined_payload = load_json(
                PROJECT_ROOT / "results" / f"{refined}_px" / f"{stage}_all_sites.json"
            )
            baseline_errors = finite_errors(baseline_payload)
            refined_errors = finite_errors(refined_payload)
            common_sites = sorted(baseline_errors.keys() & refined_errors.keys())
            deltas = np.asarray(
                [refined_errors[site] - baseline_errors[site] for site in common_sites],
                dtype=np.float64,
            )
            stage_report = {
                "baseline": summary(baseline_errors),
                "refined_source": summary(refined_errors),
                "paired_common_site_count": len(common_sites),
                "improved_site_count": int(np.count_nonzero(deltas < -0.01)),
                "unchanged_site_count": int(np.count_nonzero(np.abs(deltas) <= 0.01)),
                "worsened_site_count": int(np.count_nonzero(deltas > 0.01)),
                "median_error_change_m": float(np.median(deltas)) if len(deltas) else None,
            }
            comparison["stages"][stage] = stage_report
            for site in common_sites:
                rows.append(
                    {
                        "resolution_m": float(baseline.removesuffix("m")),
                        "stage": stage,
                        "site": site,
                        "baseline_xy_error_m": baseline_errors[site],
                        "refined_source_xy_error_m": refined_errors[site],
                        "error_change_m": refined_errors[site] - baseline_errors[site],
                    }
                )
        report["comparisons"][baseline] = comparison

    json_path = output_directory / "dem_source_comparison.json"
    csv_path = output_directory / "dem_source_site_comparison.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_path = output_directory / "dem_source_localization_accuracy.png"
    plot_final_accuracy(plot_path)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()

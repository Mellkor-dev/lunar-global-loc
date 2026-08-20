#!/usr/bin/env python3
"""Relate localization error to DARCES hypotheses and feature rejection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "factor_analysis"
PLOTS = ROOT / "plots" / "factor_analysis"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT,
                        help="Root containing results/ and, if available, local_maps/")
    parser.add_argument("--output-root", type=Path,
                        help="Destination root; defaults to the experiment root")
    parser.add_argument("--label", default="Apollo 17", help="Environment label used in plot titles")
    return parser.parse_args()


def finite(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def load_stage(profile: str, stage: str) -> dict[int, dict]:
    path = ROOT / "results" / f"{profile}_px" / f"{stage}_all_sites.json"
    if not path.is_file():
        return {}
    return {int(row["site"]): row for row in json.loads(path.read_text())["sites"]}


def feature_statistics(profile: str, site: int, selected: set[int]) -> tuple[dict, list[dict]]:
    path = ROOT / "local_maps" / f"{profile}_px" / "features" / f"local_craters_site_{site:02d}.npz"
    if not path.is_file():
        return {}, []
    with np.load(path, allow_pickle=False) as data:
        count = len(data["x_m"])
        angle = np.abs(np.degrees(data["elevation_angle_rad"]))
        sigma_z = np.sqrt(data["local_covariances"][:, 2, 2])
        radius = np.hypot(data["x_m"], data["y_m"])
        feature_rows = [
            {"profile": profile, "site": site, "feature": i,
             "selected": int(i in selected), "abs_elevation_angle_deg": float(angle[i]),
             "sigma_z_m": float(sigma_z[i]), "range_m": float(radius[i])}
            for i in range(count)
        ]
        return {
            "feature_count_npz": count,
            "median_sigma_z_m": float(np.median(sigma_z)) if count else np.nan,
            "median_abs_elevation_angle_deg": float(np.median(angle)) if count else np.nan,
            "median_feature_range_m": float(np.median(radius)) if count else np.nan,
        }, feature_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    global ROOT, OUT, PLOTS
    args = parse_arguments()
    ROOT = args.experiment_root.resolve()
    destination = args.output_root.resolve() if args.output_root else ROOT
    OUT = destination / "results" / "factor_analysis"
    PLOTS = destination / "plots" / "factor_analysis"
    profiles = sorted(p.parent.name.removesuffix("_px") for p in ROOT.glob("results/*_px/darces_all_sites.json"))
    site_rows, crater_rows = [], []
    for profile in profiles:
        darces, ransac, moga = (load_stage(profile, x) for x in ("darces", "ransac", "moga"))
        for site, row in darces.items():
            selected = set(map(int, row.get("correspondence_local_indices", [])))
            fstats, features = feature_statistics(profile, site, selected)
            for feature in features:
                feature["darces_status"] = row.get("status", "unknown")
            crater_rows.extend(features)
            record = {"profile": profile, "site": site, **row, **fstats}
            record["selected_fraction"] = len(selected) / row.get("local_feature_count", 1) if row.get("local_feature_count", 0) else 0
            record["terrain_mae_m"] = -row["fitness"] if finite(row.get("fitness")) else np.nan
            record["log10_hypotheses"] = np.log10(max(1, row.get("evaluated_hypothesis_count", 0)))
            for name, stage in (("ransac", ransac), ("moga", moga)):
                other = stage.get(site, {})
                record[f"{name}_xy_error_m"] = other.get("xy_error_m", np.nan)
                record[f"{name}_delta_error_m"] = (other.get("xy_error_m", np.nan) - row.get("xy_error_m", np.nan)) if finite(other.get("xy_error_m")) and finite(row.get("xy_error_m")) else np.nan
            site_rows.append(record)

    write_csv(OUT / "site_factors.csv", site_rows)
    write_csv(OUT / "local_crater_selection.csv", crater_rows)
    solved = [r for r in site_rows if r.get("status") == "solution" and finite(r.get("xy_error_m"))]

    factors = ["local_feature_count", "evaluated_hypothesis_count", "cluster_size", "correspondence_count",
               "selected_fraction", "terrain_mae_m", "overlap", "correspondence_xy_rmse_m",
               "correspondence_z_rmse_m", "heading_error_deg", "median_sigma_z_m", "median_feature_range_m"]
    correlations = []
    for factor in factors:
        pairs = [(float(r[factor]), float(r["xy_error_m"])) for r in solved if finite(r.get(factor))]
        rho, p = spearmanr(*zip(*pairs)) if len(pairs) >= 3 else (np.nan, np.nan)
        correlations.append({"factor": factor, "n": len(pairs), "spearman_rho": rho, "p_value": p})
    write_csv(OUT / "darces_error_correlations.csv", correlations)

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    corrs = sorted(correlations, key=lambda r: abs(r["spearman_rho"]) if finite(r["spearman_rho"]) else -1)
    colors = ["#b2182b" if r["spearman_rho"] > 0 else "#2166ac" for r in corrs]
    ax.barh([r["factor"].replace("_", " ") for r in corrs], [r["spearman_rho"] for r in corrs], color=colors)
    ax.axvline(0, color="black", lw=.8); ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman correlation with DARCES XY error")
    ax.set_title(f"DARCES error associations across solved cases (n={len(solved)})")
    fig.tight_layout(); fig.savefig(PLOTS / "01_darces_error_correlations.png", dpi=200); plt.close(fig)

    # A decision-oriented view of hypothesis clustering. Individual hypothesis
    # coordinates are not persisted, so cluster_size is the available support
    # measure for the selected pose cluster.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    good = [r for r in solved if r["xy_error_m"] <= 10]
    bad = [r for r in solved if r["xy_error_m"] > 10]
    for group, color, name, marker in ((good, "#1b9e77", "Correct (≤10 m)", "o"),
                                        (bad, "#d95f02", "False (>10 m)", "X")):
        sizes = [22 + 12 * np.log10(max(1, r["evaluated_hypothesis_count"])) for r in group]
        axes[0].scatter([r["terrain_mae_m"] for r in group], [r["cluster_size"] for r in group],
                        s=sizes, c=color, marker=marker, alpha=.68, edgecolors="white", linewidths=.35, label=name)
    axes[0].axvline(.30, color="black", ls="--", lw=1, label="Exploratory MAE gate")
    axes[0].set(xlabel="Selected terrain MAE (m)", ylabel="Selected pose-cluster size",
                title="Pose-cluster support and terrain quality")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)

    mae_bins = np.array([0, .2, .3, .5, 1.0, np.inf])
    cluster_bins = np.array([2, 3, 4, 5, 6, np.inf])
    matrix = np.full((len(cluster_bins)-1, len(mae_bins)-1), np.nan)
    counts = np.zeros_like(matrix)
    for iy in range(matrix.shape[0]):
        for ix in range(matrix.shape[1]):
            cell = [r for r in solved if mae_bins[ix] <= r["terrain_mae_m"] < mae_bins[ix+1]
                    and cluster_bins[iy] <= r["cluster_size"] < cluster_bins[iy+1]]
            counts[iy, ix] = len(cell)
            if cell: matrix[iy, ix] = np.mean([r["xy_error_m"] <= 10 for r in cell])
    im = axes[1].imshow(matrix, origin="lower", vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    for iy in range(matrix.shape[0]):
        for ix in range(matrix.shape[1]):
            if counts[iy, ix]: axes[1].text(ix, iy, f"{matrix[iy,ix]*100:.0f}%\nn={int(counts[iy,ix])}", ha="center", va="center", fontsize=8)
    axes[1].set_xticks(range(5), ["<.20", ".20–.30", ".30–.50", ".50–1", ">1"])
    axes[1].set_yticks(range(5), ["2", "3", "4", "5", "6+"])
    axes[1].set(xlabel="Terrain MAE bin (m)", ylabel="Cluster size", title="Fraction of poses within 10 m")
    fig.colorbar(im, ax=axes[1], fraction=.046, label="Correct fraction")

    labels, good_counts, false_counts, failed_counts = [], [], [], []
    for profile in profiles:
        rows = [r for r in site_rows if r["profile"] == profile]
        labels.append(profile.replace("_refined", " R"))
        good_counts.append(sum(r.get("status") == "solution" and finite(r.get("xy_error_m")) and r["xy_error_m"] <= 10 for r in rows))
        false_counts.append(sum(r.get("status") == "solution" and finite(r.get("xy_error_m")) and r["xy_error_m"] > 10 for r in rows))
        failed_counts.append(sum(r.get("status") != "solution" for r in rows))
    x = np.arange(len(labels))
    axes[2].bar(x, good_counts, color="#1b9e77", label="Correct")
    axes[2].bar(x, false_counts, bottom=good_counts, color="#d95f02", label="False solution")
    axes[2].bar(x, failed_counts, bottom=np.array(good_counts)+false_counts, color="#999999", label="No solution")
    axes[2].set_xticks(x, labels, rotation=45, ha="right")
    axes[2].set(ylabel="Number of sites", title="Outcome by DEM profile")
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=.2)
    fig.suptitle(f"{args.label}: what DARCES hypothesis clustering actually predicts")
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(PLOTS / "02_darces_hypothesis_diagnostics.png", dpi=200); plt.close(fig)

    # Binned line relationship: connecting individual sites would imply an
    # ordering that does not exist. Quantile bins show the central trend while
    # retaining the large dispersion through an interquartile band.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    hypothesis_values = np.asarray([r["evaluated_hypothesis_count"] for r in solved], dtype=float)
    error_values = np.asarray([r["xy_error_m"] for r in solved], dtype=float)
    if len(solved):
        edges = np.unique(np.quantile(hypothesis_values, np.linspace(0, 1, min(6, len(solved) + 1))))
        centers, medians, lower, upper, ns = [], [], [], [], []
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (hypothesis_values >= left) & ((hypothesis_values <= right) if index == len(edges)-2 else (hypothesis_values < right))
            values = error_values[mask]
            hypotheses = hypothesis_values[mask]
            if len(values):
                centers.append(float(np.median(hypotheses)))
                medians.append(float(np.median(values)))
                lower.append(float(np.quantile(values, .25)))
                upper.append(float(np.quantile(values, .75)))
                ns.append(int(len(values)))
        centers, medians, lower, upper = map(np.asarray, (centers, medians, lower, upper))
        ax.fill_between(centers, lower, upper, color="#80b1d3", alpha=.28, label="Interquartile range")
        ax.plot(centers, medians, color="#1f78b4", marker="o", lw=2.2, label="Median XY error")
        for x_value, y_value, count in zip(centers, medians, ns):
            ax.annotate(f"n={count}", (x_value, y_value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Evaluated DARCES hypotheses per site (log scale)")
    ax.set_ylabel("DARCES XY localization error (m, log scale)")
    ax.set_title(f"{args.label}: hypothesis generation versus localization error")
    ax.grid(which="both", alpha=.22); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "05_hypothesis_count_vs_error_line.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # Compare features within solved sites only. Including all features from
    # failed sites as "rejected" would confound feature properties with site
    # failure and resolution.
    solved_craters = [r for r in crater_rows if r["darces_status"] == "solution"]
    selected = [r for r in solved_craters if r["selected"]]
    rejected = [r for r in solved_craters if not r["selected"]]
    metrics = [("sigma_z_m", "Vertical uncertainty σz (m)"), ("abs_elevation_angle_deg", "|Elevation angle| (deg)"), ("range_m", "Feature range (m)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    tests = []
    for ax, (field, label) in zip(axes, metrics):
        a, b = np.array([r[field] for r in selected]), np.array([r[field] for r in rejected])
        _, p = mannwhitneyu(a, b, alternative="two-sided") if len(a) and len(b) else (np.nan, np.nan)
        tests.append({"metric": field, "selected_n": len(a), "rejected_n": len(b), "selected_median": np.median(a), "rejected_median": np.median(b), "p_value": p})
        ax.boxplot([a, b], labels=["Selected", "Rejected"], showfliers=False)
        ax.set_ylabel(label); ax.set_title(f"p={p:.2g}"); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Local craters used in final DARCES correspondences vs all others")
    fig.tight_layout(); fig.savefig(PLOTS / "03_local_crater_selection.png", dpi=200); plt.close(fig)
    write_csv(OUT / "local_crater_selection_tests.csv", tests)

    stage_rows = []
    for r in solved:
        for stage in ("ransac", "moga"):
            delta = r.get(f"{stage}_delta_error_m")
            if finite(delta): stage_rows.append((stage, delta))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    values = [[d for s, d in stage_rows if s == stage] for stage in ("ransac", "moga")]
    ax.boxplot(values, labels=["RANSAC − DARCES", "MOGA − DARCES"], showfliers=True)
    ax.axhline(0, color="black", lw=1); ax.set_ylabel("Change in XY error (m); negative is improvement")
    ax.set_title("Downstream refinement effect on DARCES solutions"); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(PLOTS / "04_pipeline_stage_error_change.png", dpi=200); plt.close(fig)

    summary = {"profiles": profiles, "site_cases": len(site_rows), "solved_cases": len(solved),
               "selected_craters": len(selected), "rejected_or_unused_craters": len(rejected),
               "correlations": correlations, "crater_tests": tests}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

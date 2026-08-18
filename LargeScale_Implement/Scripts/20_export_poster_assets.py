#!/usr/bin/env python3
"""Export the scientific poster graphics as PowerPoint-friendly SVG files.

Matplotlib plots retain vector text, axes, contours, markers, and annotations.
DEM/hillshade layers remain embedded rasters because the underlying DEM is a
raster.  The paper architecture, logo, and header background are preserved at
their original embedded resolution inside SVG containers.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import load_pipeline_config, load_resolution_config
from site_selection import selected_sites_for_config
from traversal_presentation import (
    generate_localization_prediction_contour_map,
    select_best_pose_estimates,
)


DEFAULT_POSTER = Path("/home/soumyadeep/Downloads/GIP poster 3.pptx")
DEFAULT_EXPERIMENT = (
    REPOSITORY_ROOT / "experiments" / "apollo17_crater_D15_eps0p05"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poster", type=Path, default=DEFAULT_POSTER)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "plots" / "poster_svg",
    )
    return parser.parse_args()


def load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_site(path: Path, site: int) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    for record in payload.get("sites", []):
        if int(record.get("site", -1)) == site:
            return record
    raise KeyError(f"Site {site:02d} is absent from {path}")


def svg_raster_wrapper(
    destination: Path,
    image_bytes: bytes,
    width: int,
    height: int,
) -> None:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n'
        f'  <image width="{width}" height="{height}" '
        f'href="data:image/png;base64,{encoded}"/>\n'
        '</svg>\n'
    )
    destination.write_text(svg, encoding="utf-8")


def png_dimensions(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Expected PNG data")
    return (
        int.from_bytes(data[16:20], "big"),
        int.from_bytes(data[20:24], "big"),
    )


def export_embedded_assets(poster: Path, output: Path) -> None:
    names = {
        "image1.png": "01_header_background.svg",
        "image4.png": "04_pipeline_architecture.svg",
        "image7.png": "07_false_alias_site04.svg",
    }
    with ZipFile(poster) as archive:
        for media_name, output_name in names.items():
            data = archive.read(f"ppt/media/{media_name}")
            width, height = png_dimensions(data)
            svg_raster_wrapper(output / output_name, data, width, height)

        logo_data = archive.read("ppt/media/image2.png")
        logo = Image.open(BytesIO(logo_data)).convert("RGBA")
        alpha = np.asarray(logo.getchannel("A"))
        occupied_columns = np.flatnonzero(np.any(alpha > 0, axis=0))
        if occupied_columns.size < 2:
            raise ValueError("Combined GIST/MPIL logo has no visible content")
        gaps = np.diff(occupied_columns)
        gap_index = int(np.argmax(gaps))
        if int(gaps[gap_index]) < 8:
            raise ValueError("Could not locate the gap between GIST and MPIL")
        split_x = int(
            (occupied_columns[gap_index] + occupied_columns[gap_index + 1]) // 2
        )

        def cropped_png(image: Image.Image) -> tuple[bytes, int, int]:
            content = image.getbbox()
            if content is None:
                raise ValueError("Logo crop has no visible content")
            padding = 2
            left = max(0, content[0] - padding)
            top = max(0, content[1] - padding)
            right = min(image.width, content[2] + padding)
            bottom = min(image.height, content[3] + padding)
            cropped = image.crop((left, top, right, bottom))
            stream = BytesIO()
            cropped.save(stream, format="PNG", optimize=True)
            return stream.getvalue(), cropped.width, cropped.height

        logo_outputs = (
            (logo.crop((0, 0, split_x, logo.height)), "02_gist_logo.svg"),
            (
                logo.crop((split_x, 0, logo.width, logo.height)),
                "02_mpil_logo.svg",
            ),
        )
        for logo_crop, output_name in logo_outputs:
            data, width, height = cropped_png(logo_crop)
            svg_raster_wrapper(output / output_name, data, width, height)

    # Superseded by the two independently placeable logo assets above.
    (output / "02_gist_mpil_logo.svg").unlink(missing_ok=True)


def catalogue_xyz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.column_stack((data["x_m"], data["y_m"], data["z_m"]))


def raster_extent(config) -> tuple[float, float, float, float]:
    x, y = config.orbital_raster.coordinates()
    half = config.orbital_raster.resolution_m / 2.0
    return (x[0] - half, x[-1] + half, y[-1] - half, y[0] + half)


def export_global_features(config, destination: Path) -> None:
    dem = np.asarray(np.load(config.orbital_dem_path, allow_pickle=False))
    features = catalogue_xyz(config.global_features_path)
    figure, axis = plt.subplots(figsize=(9.2, 7.5), layout="constrained")
    image = axis.imshow(
        dem, cmap="terrain", origin="upper", extent=raster_extent(config),
        rasterized=True,
    )
    axis.scatter(
        features[:, 0], features[:, 1], marker="x", color="red",
        s=28, linewidths=1.2, label="Detected crater",
    )
    axis.legend(
        loc="center left", fontsize=10, framealpha=0.88,
        borderpad=0.45, handletextpad=0.55,
    )
    axis.set_title(
        "Apollo 17 5m/px DEM craters\n"
        f"n=3, D=15m, No.={len(features)}",
        fontsize=18, fontweight="bold", pad=9,
    )
    axis.set_xlabel("Map x / east [m]", fontsize=14)
    axis.set_ylabel("Map y / north [m]", fontsize=14)
    axis.tick_params(axis="both", labelsize=11)
    axis.set_aspect("equal")
    colorbar = figure.colorbar(
        image, ax=axis, fraction=0.026, pad=0.018, shrink=0.55, aspect=26,
    )
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label("Elevation [m]", fontsize=11, labelpad=6)
    figure.savefig(destination, bbox_inches="tight", pad_inches=0.04,
                   facecolor="white")
    plt.close(figure)


def export_local_features(
    config,
    darces_site: dict[str, object],
    destination: Path,
) -> None:
    grid_path = config.gridded_maps_path / "grid_site_06.npz"
    feature_path = config.local_features_path / "local_craters_site_06.npz"
    with np.load(grid_path, allow_pickle=False) as grid:
        elevation = np.asarray(grid["elevation"])
        valid = np.asarray(grid["valid_mask"], dtype=bool)
        x_centers = np.asarray(grid["x_centers_m"])
        y_centers = np.asarray(grid["y_centers_m"])
    features = catalogue_xyz(feature_path)
    passed = {int(index) for index in darces_site["correspondence_local_indices"]}
    half = config.orbital_raster.resolution_m / 2.0
    extent = (
        x_centers[0] - half, x_centers[-1] + half,
        y_centers[-1] - half, y_centers[0] + half,
    )
    figure, axis = plt.subplots(figsize=(9.2, 7.5), layout="constrained")
    image = axis.imshow(
        np.ma.masked_where(~valid, elevation), origin="upper", extent=extent,
        cmap="terrain", rasterized=True,
    )
    rejected = np.asarray([index not in passed for index in range(len(features))])
    accepted = ~rejected
    axis.scatter(
        features[rejected, 0], features[rejected, 1], marker="x",
        color="#d62728", s=34, linewidths=1.5,
        label=f"No DARCES match ({int(rejected.sum())})",
    )
    axis.scatter(
        features[accepted, 0], features[accepted, 1], marker="x",
        color="#1565a8", s=46, linewidths=2.0,
        label=f"Passed to RANSAC ({int(accepted.sum())})",
    )
    for index, (x, y) in enumerate(features[:, :2]):
        color = "#0d4f82" if index in passed else "#aa1d1d"
        horizontal = 4 if x <= 0 else -4
        axis.annotate(
            f"L{index + 1:02d}", (x, y), xytext=(horizontal, 5),
            textcoords="offset points",
            ha="left" if horizontal > 0 else "right", va="bottom",
            fontsize=8.2, fontweight="bold", color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72,
                  "pad": 0.6},
        )
    axis.scatter(0.0, 0.0, marker="^", color="black", s=45, label="Rover")
    axis.legend(loc="upper right", fontsize=10, framealpha=0.90,
                borderpad=0.45, handletextpad=0.55)
    axis.set_title(
        "Apollo 17 Site 06 local craters — gridded to 5m/px\n"
        f"n=3, D=15m, No.={len(features)}",
        fontsize=18, fontweight="bold", pad=9,
    )
    axis.set_xlabel("Rover-local x [m]", fontsize=14)
    axis.set_ylabel("Rover-local y [m]", fontsize=14)
    axis.tick_params(axis="both", labelsize=11)
    axis.set_aspect("equal")
    colorbar = figure.colorbar(
        image, ax=axis, fraction=0.026, pad=0.018, shrink=0.55, aspect=26,
    )
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label("Leveled elevation [m]", fontsize=11, labelpad=6)
    figure.savefig(destination, bbox_inches="tight", pad_inches=0.04,
                   facecolor="white")
    plt.close(figure)


def export_ransac(
    config,
    ransac_site: dict[str, object],
    destination: Path,
) -> None:
    local = catalogue_xyz(config.local_features_path / "local_craters_site_06.npz")
    global_ = catalogue_xyz(config.global_features_path)
    result = ransac_site["ransac"]
    darces = ransac_site["darces"]
    local_indices = np.asarray(result["inlier_local_indices"], dtype=np.int64)
    global_indices = np.asarray(result["inlier_global_indices"], dtype=np.int64)
    rotation = np.asarray(result["rotation"], dtype=np.float64)
    translation = np.asarray(result["translation_xy_m"], dtype=np.float64)
    projected = local[local_indices, :2] @ rotation.T + translation
    targets = global_[global_indices, :2]

    figure, axis = plt.subplots(figsize=(9.2, 7.5), layout="constrained")
    for index in range(len(projected)):
        axis.plot(
            [projected[index, 0], targets[index, 0]],
            [projected[index, 1], targets[index, 1]],
            color="#388e3c", linewidth=2.4, alpha=0.94,
            solid_capstyle="round", zorder=2,
            label="RANSAC inlier" if index == 0 else None,
        )
    axis.scatter(
        targets[:, 0], targets[:, 1], marker="o", facecolors="none",
        edgecolors="black", linewidths=2.0, s=145, zorder=4,
        label="Matched global crater",
    )
    axis.scatter(
        projected[:, 0], projected[:, 1], marker="x", color="#1565c0",
        s=145, linewidths=2.7, zorder=5, label="Transformed local crater",
    )
    for local_index, (x, y) in zip(local_indices, projected, strict=True):
        axis.annotate(
            f"L{local_index + 1:02d}", (x, y), xytext=(5, -8),
            textcoords="offset points", color="#0d4f82",
            fontsize=8.5, fontweight="bold", zorder=7,
        )
    axis.scatter(
        translation[0], translation[1], marker="^", color="#6a1b9a",
        edgecolors="white", linewidths=1.0, s=190, zorder=6,
        label="RANSAC pose",
    )
    axis.scatter(
        darces["estimated_x_m"], darces["estimated_y_m"], marker="+",
        color="#ef6c00", linewidths=2.8, s=190, zorder=7,
        label="DARCES pose",
    )
    axis.scatter(
        darces["truth_x_m"], darces["truth_y_m"], marker="*",
        color="#00838f", edgecolors="white", linewidths=0.9,
        s=220, zorder=8, label="Truth (evaluation only)",
    )
    axis.set_title(
        "Apollo 17 Site 06 5m/px RANSAC correspondences\n"
        f"DARCES matches={result['input_correspondence_count']} · "
        f"RANSAC inliers={result['inlier_count']}/"
        f"{result['input_correspondence_count']} · outliers={result['outlier_count']}",
        fontsize=18, fontweight="bold", pad=9,
    )
    axis.set_xlabel("Map x / east [m]", fontsize=14)
    axis.set_ylabel("Map y / north [m]", fontsize=14)
    axis.tick_params(axis="both", labelsize=11)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=9.4, framealpha=0.92,
                borderpad=0.50, handletextpad=0.55, labelspacing=0.35)
    figure.savefig(destination, bbox_inches="tight", pad_inches=0.04,
                   facecolor="white")
    plt.close(figure)


def export_summary(experiment: Path, destination: Path) -> None:
    module = load_script(
        PROJECT_ROOT / "Scripts" / "14_results_presentation.py",
        "poster_results_presentation",
    )
    module.make_figure(
        destination,
        destination.with_suffix(".csv"),
        600,
        experiment / "results",
    )


def export_localization_map(config, experiment: Path, destination: Path) -> None:
    module = load_script(
        PROJECT_ROOT / "Scripts" / "10_plot_traverse_map.py",
        "poster_traverse_map",
    )
    selected = selected_sites_for_config(config)
    sites, poses = module.load_odometry(
        config.captures_path / "odom_scans", selected,
    )
    stage_estimates = {
        stage: module.load_estimates(
            experiment / "results" / "5m_px" / f"{stage}_all_sites.json"
        )
        for stage in ("darces", "ransac", "moga")
    }
    best, _counts = select_best_pose_estimates(sites, poses, stage_estimates)
    dem = np.asarray(np.load(config.orbital_dem_path, allow_pickle=False))
    generate_localization_prediction_contour_map(
        config=config,
        sites=sites,
        poses=poses,
        estimates=best,
        dem=dem,
        output_path=destination,
        contour_interval_m=2.0,
        error_label_threshold_m=50.0,
        padding_m=70.0,
    )


def write_readme(path: Path) -> None:
    path.write_text(
        """# Poster SVG assets

The scientific plots were exported from the archived Apollo 17 crater
experiment (`apollo17_crater_D15_eps0p05`) so their values match the poster.

- Global/local DEM and contour figures are hybrid SVGs: labels, axes, markers,
  and contours are vector; the DEM itself remains a raster by definition.
- The resolution summary and RANSAC correspondence figure are vector plots.
- The header, separate GIST and MPIL logos, paper architecture, and false-alias
  composite preserve the exact PNGs embedded in the supplied PowerPoint inside
  lossless SVG wrappers.
- Insert the SVGs with PowerPoint's **Insert > Pictures** command and keep image
  compression disabled.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_arguments()
    poster = args.poster.resolve()
    experiment = args.experiment.resolve()
    output = args.output_directory.resolve()
    if not poster.is_file():
        raise FileNotFoundError(poster)
    if not (experiment / "results").is_dir():
        raise FileNotFoundError(experiment / "results")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none"})

    config = load_resolution_config("5m")
    darces_site = read_site(
        experiment / "results" / "5m_px" / "darces_all_sites.json", 6,
    )
    ransac_site = read_site(
        experiment / "results" / "5m_px" / "ransac_all_sites.json", 6,
    )

    export_embedded_assets(poster, output)
    export_global_features(config, output / "03_global_craters_5m.svg")
    export_localization_map(
        config, experiment, output / "05_localization_predictions_5m.svg",
    )
    export_summary(experiment, output / "06_resolution_results.svg")
    export_local_features(
        config, darces_site, output / "08_local_craters_site06.svg",
    )
    export_ransac(config, ransac_site, output / "09_ransac_site06.svg")
    write_readme(output / "README.md")

    print("Poster SVG export")
    print("-----------------")
    for path in sorted(output.glob("*.svg")):
        print(path)


if __name__ == "__main__":
    main()

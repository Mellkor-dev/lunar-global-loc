#!/usr/bin/env python3
"""Generate synchronized orbital and truth feature catalogues."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.dilation_detector import detect_craters, detect_peaks
from pipeline_config import PipelineConfig, RasterConfig, load_pipeline_config


CONFIG = load_pipeline_config()
PLOT_DIRECTORY = CONFIG.plots_path
GLOBAL_PREVIEW_PATH = PLOT_DIRECTORY / "global_features_preview.png"
TRUTH_PREVIEW_PATH = PLOT_DIRECTORY / "truth_features_preview.png"


def _load_dem(path: Path, raster: RasterConfig, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{name} DEM not found: {path}")
    dem = np.load(path, allow_pickle=False)
    if dem.shape != raster.shape:
        raise ValueError(
            f"{name} DEM shape {dem.shape} differs from configured "
            f"{raster.shape}"
        )
    if not np.isfinite(dem).all():
        raise ValueError(f"{name} DEM contains NaN or infinite elevations")
    return dem


def _detect(
    elevation: np.ndarray,
    *,
    kind: str,
    radius_cells: int,
    flatness_threshold_m: float,
) -> np.ndarray:
    detector = detect_craters if kind == "crater" else detect_peaks
    features = detector(
        elevation,
        n=radius_cells,
        flatness_eps=flatness_threshold_m,
        exclude_border=True,
    )
    features = np.asarray(features, dtype=np.int64)
    if features.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != 2:
        raise RuntimeError(
            f"Feature detector returned invalid shape {features.shape}"
        )
    return features


def _save_catalogue(
    path: Path,
    indices: np.ndarray,
    xyz: np.ndarray,
    *,
    config: PipelineConfig,
    resolution_m: float,
    radius_cells: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row=indices[:, 0],
        column=indices[:, 1],
        x_m=xyz[:, 0],
        y_m=xyz[:, 1],
        z_m=xyz[:, 2],
        feature_kind=np.asarray(config.features.kind),
        detection_radius_cells=np.int64(radius_cells),
        detection_distance_m=np.float64(config.features.distance_m),
        flatness_eps_m=np.float64(
            config.features.flatness_threshold_m
        ),
        resolution_m=np.float64(resolution_m),
        config_path=np.asarray(str(config.config_path)),
    )


def _save_preview(
    path: Path,
    elevation: np.ndarray,
    raster: RasterConfig,
    xyz: np.ndarray,
    *,
    title: str,
    radius_cells: int,
    config: PipelineConfig,
) -> None:
    x_centers, y_centers = raster.coordinates()
    half_cell = raster.resolution_m / 2.0
    extent = (
        x_centers[0] - half_cell,
        x_centers[-1] + half_cell,
        y_centers[-1] - half_cell,
        y_centers[0] + half_cell,
    )
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(
        elevation,
        cmap="terrain",
        origin="upper",
        extent=extent,
    )
    if len(xyz):
        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            marker="x",
            color="red",
            s=28,
            linewidths=1.2,
            label=f"Detected {config.features.kind}",
        )
        axis.legend()
    axis.set_title(
        f"{title}\n"
        f"kind={config.features.kind}, n={radius_cells}, "
        f"D={config.features.distance_m:g} m, N={len(xyz)}"
    )
    axis.set_xlabel("Map x / east [m]")
    axis.set_ylabel("Map y / north [m]")
    axis.set_aspect("equal")
    figure.colorbar(image, ax=axis, label="Elevation [m]")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    config = CONFIG
    orbital_dem = _load_dem(
        config.orbital_dem_path,
        config.orbital_raster,
        "Orbital",
    )
    truth_dem = _load_dem(
        config.truth_dem_path,
        config.truth_raster,
        "Truth",
    )

    orbital_radius = config.features.radius_for_resolution(
        config.orbital_raster.resolution_m
    )
    truth_radius = config.features.radius_for_resolution(
        config.truth_raster.resolution_m
    )
    orbital_indices = _detect(
        orbital_dem,
        kind=config.features.kind,
        radius_cells=orbital_radius,
        flatness_threshold_m=config.features.flatness_threshold_m,
    )
    truth_indices = _detect(
        truth_dem,
        kind=config.features.kind,
        radius_cells=truth_radius,
        flatness_threshold_m=config.features.flatness_threshold_m,
    )
    orbital_xyz = config.orbital_raster.indices_to_xyz(
        orbital_indices,
        orbital_dem,
    )
    truth_xyz = config.truth_raster.indices_to_xyz(
        truth_indices,
        truth_dem,
    )

    _save_catalogue(
        config.global_features_path,
        orbital_indices,
        orbital_xyz,
        config=config,
        resolution_m=config.orbital_raster.resolution_m,
        radius_cells=orbital_radius,
    )
    _save_catalogue(
        config.truth_features_path,
        truth_indices,
        truth_xyz,
        config=config,
        resolution_m=config.truth_raster.resolution_m,
        radius_cells=truth_radius,
    )

    PLOT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _save_preview(
        GLOBAL_PREVIEW_PATH,
        orbital_dem,
        config.orbital_raster,
        orbital_xyz,
        title="Apollo 17 orbital DEM features",
        radius_cells=orbital_radius,
        config=config,
    )
    _save_preview(
        TRUTH_PREVIEW_PATH,
        truth_dem,
        config.truth_raster,
        truth_xyz,
        title="Apollo 17 truth DEM features",
        radius_cells=truth_radius,
        config=config,
    )

    print("Synchronized feature catalogues")
    print("-------------------------------")
    print(
        f"Definition: {config.features.kind}, "
        f"D={config.features.distance_m:g} m, "
        f"flatness={config.features.flatness_threshold_m:g} m"
    )
    print(
        f"Orbital: n={orbital_radius}, count={len(orbital_indices)}, "
        f"path={config.global_features_path}"
    )
    print(
        f"Truth:   n={truth_radius}, count={len(truth_indices)}, "
        f"path={config.truth_features_path}"
    )


if __name__ == "__main__":
    main()

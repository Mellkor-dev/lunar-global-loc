#!/usr/bin/env python3
"""Generate feature catalogues for configured DEM resolutions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.dilation_detector import detect_craters, detect_peaks
from features import dilation_detector
from pipeline_config import (
    FeatureDetectionTarget,
    PipelineConfig,
    RasterConfig,
    load_pipeline_config,
)


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
    kind: str,
    resolution_m: float,
    radius_cells: int,
    distance_m: float,
    flatness_threshold_m: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        row=indices[:, 0],
        column=indices[:, 1],
        x_m=xyz[:, 0],
        y_m=xyz[:, 1],
        z_m=xyz[:, 2],
        feature_kind=np.asarray(kind),
        detection_radius_cells=np.int64(radius_cells),
        detection_distance_m=np.float64(distance_m),
        flatness_eps_m=np.float64(flatness_threshold_m),
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
    kind: str,
    distance_m: float,
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
            label=f"Detected {kind}",
        )
        axis.legend()
    axis.set_title(
        f"{title}\n"
        f"kind={kind}, n={radius_cells}, "
        f"D={distance_m:g} m, N={len(xyz)}"
    )
    axis.set_xlabel("Map x / east [m]")
    axis.set_ylabel("Map y / north [m]")
    axis.set_aspect("equal")
    figure.colorbar(image, ax=axis, label="Elevation [m]")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _parse_arguments(config: PipelineConfig) -> argparse.Namespace:
    choices = sorted(config.feature_detection_targets)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        choices=choices + ["all"],
        default="all",
        help="Raster resolution profile to process (default: all)",
    )
    return parser.parse_args()


def _process_target(config: PipelineConfig, target: FeatureDetectionTarget) -> int:
    print(
        f"{target.name}: loading {target.dem_path} "
        f"({target.raster.shape[0]}x{target.raster.shape[1]})",
        flush=True,
    )
    dem = _load_dem(target.dem_path, target.raster, target.name)
    backend = "OpenCV" if dilation_detector.cv2 is not None else "SciPy"
    print(
        f"{target.name}: detecting {config.features.kind}s with "
        f"n={target.radius_cells} using {backend} morphology...",
        flush=True,
    )
    started = time.perf_counter()
    indices = _detect(
        dem,
        kind=config.features.kind,
        radius_cells=target.radius_cells,
        flatness_threshold_m=target.flatness_threshold_m,
    )
    elapsed = time.perf_counter() - started
    print(
        f"{target.name}: detection finished in {elapsed:.1f} s; "
        f"writing {len(indices)} features...",
        flush=True,
    )
    xyz = target.raster.indices_to_xyz(indices, dem)
    _save_catalogue(
        target.output_path,
        indices,
        xyz,
        config=config,
        kind=config.features.kind,
        resolution_m=target.raster.resolution_m,
        radius_cells=target.radius_cells,
        distance_m=target.distance_m,
        flatness_threshold_m=target.flatness_threshold_m,
    )
    target.preview_path.parent.mkdir(parents=True, exist_ok=True)
    _save_preview(
        target.preview_path,
        dem,
        target.raster,
        xyz,
        title=(
            f"{config.config_path.stem.replace('_', ' ').title()} "
            f"{target.name} {target.catalogue_role} DEM features"
        ),
        radius_cells=target.radius_cells,
        kind=config.features.kind,
        distance_m=target.distance_m,
    )
    print(
        f"{target.name}: kind={config.features.kind}, "
        f"n={target.radius_cells}, D={target.distance_m:g} m, "
        f"flatness={target.flatness_threshold_m:g} m, count={len(indices)}"
    )
    print(f"  catalogue: {target.output_path}")
    print(f"  preview:   {target.preview_path}")
    return len(indices)


def main() -> None:
    config = load_pipeline_config()
    arguments = _parse_arguments(config)
    names = (
        sorted(config.feature_detection_targets)
        if arguments.resolution == "all"
        else [arguments.resolution]
    )
    print("Feature detection")
    print("-----------------")
    for name in names:
        _process_target(config, config.feature_detection_targets[name])


if __name__ == "__main__":
    main()

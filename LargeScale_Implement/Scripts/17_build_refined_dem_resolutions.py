#!/usr/bin/env python3
"""Build configured test DEMs directly from the refined 0.25 m DEM."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import load_resolution_config


SOURCE_PROFILE = "0p25m"
TARGET_PROFILES = ("0p5m", "1m", "2m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_block_average(source: np.ndarray, factor: int) -> np.ndarray:
    """Area-average square, exactly divisible blocks without interpolation."""
    if source.ndim != 2:
        raise ValueError(f"Source DEM must be 2-D, got {source.shape}")
    rows, columns = source.shape
    if rows % factor or columns % factor:
        raise ValueError(
            f"Source shape {source.shape} is not divisible by block factor {factor}"
        )
    reshaped = source.reshape(rows // factor, factor, columns // factor, factor)
    return np.asarray(reshaped.mean(axis=(1, 3), dtype=np.float64), dtype=np.float32)


def build_target(
    source: np.ndarray,
    source_path: Path,
    source_metadata_path: Path,
    profile: str,
) -> None:
    source_config = load_resolution_config(SOURCE_PROFILE)
    target_config = load_resolution_config(profile)
    source_resolution_m = source_config.orbital_raster.resolution_m
    target_resolution_m = target_config.orbital_raster.resolution_m
    ratio = target_resolution_m / source_resolution_m
    factor = int(round(ratio))
    if not np.isclose(ratio, factor):
        raise ValueError(
            f"{profile}: {target_resolution_m:g} m is not an integer multiple "
            f"of {source_resolution_m:g} m"
        )

    target = exact_block_average(source, factor)
    expected_shape = target_config.orbital_raster.shape
    if target.shape != expected_shape:
        raise RuntimeError(
            f"{profile}: generated shape {target.shape}, expected {expected_shape}"
        )
    if not np.isfinite(target).all():
        raise RuntimeError(f"{profile}: generated DEM contains invalid elevations")

    output_path = target_config.orbital_dem_path
    mask_path = target_config.orbital_mask_path
    metadata_path = target_config.orbital_metadata_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, target)
    np.save(mask_path, np.ones(target.shape, dtype=bool))

    extent_x_m = target.shape[1] * target_resolution_m
    extent_y_m = target.shape[0] * target_resolution_m
    metadata = {
        "name": f"apollo17_refined_prior_{profile}",
        "description": (
            f"Exact {factor}x{factor} block-area average of the refined "
            f"Apollo 17 0.25 m DEM at {target_resolution_m:g} m/cell."
        ),
        "source": {
            "dem_path": str(source_path.relative_to(PROJECT_ROOT)),
            "metadata_path": str(source_metadata_path.relative_to(PROJECT_ROOT)),
            "shape": [int(value) for value in source.shape],
            "resolution_m": source_resolution_m,
            "dem_sha256": sha256(source_path),
            "metadata_sha256": (
                sha256(source_metadata_path) if source_metadata_path.is_file() else None
            ),
        },
        "target": {
            "dem_path": str(output_path.relative_to(PROJECT_ROOT)),
            "valid_mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
            "shape": [int(value) for value in target.shape],
            "resolution_m": target_resolution_m,
            "cell_edge_extent_m": [extent_x_m, extent_y_m],
            "resampling": "exact_block_area_average",
            "block_factor": factor,
            "first_cell_center_m": {
                "x": target_config.orbital_raster.first_x_center_m,
                "y": target_config.orbital_raster.first_y_center_m,
            },
            "array_orientation": "north_up_rows_descend_in_map_y",
        },
        "elevation_statistics_m": {
            "minimum": float(target.min()),
            "maximum": float(target.max()),
            "mean": float(target.mean(dtype=np.float64)),
            "standard_deviation": float(target.std(dtype=np.float64)),
            "relief": float(target.max() - target.min()),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

    print(f"{profile}: {source.shape} -> {target.shape} ({factor}x{factor} blocks)")
    print(f"  DEM:      {output_path}")
    print(f"  mask:     {mask_path}")
    print(f"  metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        choices=(*TARGET_PROFILES, "all"),
        default="all",
        help="Target profile to build (default: both)",
    )
    args = parser.parse_args()

    source_config = load_resolution_config(SOURCE_PROFILE)
    source_path = source_config.orbital_dem_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Refined source DEM does not exist: {source_path}")
    source = np.asarray(np.load(source_path, allow_pickle=False))
    if source.shape != source_config.orbital_raster.shape:
        raise ValueError(
            f"Source shape {source.shape} does not match configured "
            f"{source_config.orbital_raster.shape}"
        )
    if not np.isfinite(source).all():
        raise ValueError("Refined source DEM contains invalid elevations")

    profiles = TARGET_PROFILES if args.resolution == "all" else (args.resolution,)
    print("Refined DEM downsampling")
    print("--------------------------")
    print(f"Source: {source_path} ({source.shape[0]}x{source.shape[1]})")
    for profile in profiles:
        build_target(
            source,
            source_path,
            source_config.orbital_metadata_path,
            profile,
        )


if __name__ == "__main__":
    main()

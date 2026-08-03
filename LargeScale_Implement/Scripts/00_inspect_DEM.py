#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import yaml
import hashlib

from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import load_pipeline_config

# DEM feature imports for global digital elevation model processing

def file_hash(file_path: str) -> str:
    """Compute the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            sha256.update(block)
    return sha256.hexdigest()

def load_metadata(metadata_path: str) -> dict:
    """Load metadata from a YAML file."""
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)
    return metadata

def load_dem(dem_path: str) -> np.ndarray:
    """Load a DEM from a .npy file."""
    return np.load(dem_path)

def print_dem_statistics(dem: np.ndarray, resolution_m: float | None) -> None:
    """Print basic DEM geometry and elevation statistics."""
    if dem.ndim != 2:
        raise ValueError(f"Expected a 2-D DEM, got shape {dem.shape}")

    finite = np.isfinite(dem)
    finite_count = int(finite.sum())

    if finite_count == 0:
        raise ValueError("DEM contains no finite elevation values")

    rows, cols = dem.shape

    print("\nDEM array")
    print("---------")
    print(f"shape                 : {dem.shape}")
    print(f"dtype                 : {dem.dtype}")
    print(f"finite cells          : {finite_count}/{dem.size}")
    print(f"finite fraction       : {finite.mean():.8f}")
    print(f"minimum elevation     : {np.nanmin(dem):.6f} m")
    print(f"maximum elevation     : {np.nanmax(dem):.6f} m")
    print(f"mean elevation        : {np.nanmean(dem):.6f} m")
    print(f"standard deviation    : {np.nanstd(dem):.6f} m")
    print(
        "total relief          : "
        f"{np.nanmax(dem) - np.nanmin(dem):.6f} m"
    )

    print("\nEdge statistics")
    print("---------------")
    print(f"first row mean        : {np.nanmean(dem[0, :]):.6f} m")
    print(f"last row mean         : {np.nanmean(dem[-1, :]):.6f} m")
    print(f"first column mean     : {np.nanmean(dem[:, 0]):.6f} m")
    print(f"last column mean      : {np.nanmean(dem[:, -1]):.6f} m")
    print(f"center elevation      : {dem[rows // 2, cols // 2]:.6f} m")
    print(f"upper-left elevation  : {dem[0, 0]:.6f} m")
    print(f"upper-right elevation : {dem[0, -1]:.6f} m")
    print(f"lower-left elevation  : {dem[-1, 0]:.6f} m")
    print(f"lower-right elevation : {dem[-1, -1]:.6f} m")

    if resolution_m is not None:
        cell_coverage_x = cols * resolution_m
        cell_coverage_y = rows * resolution_m

        center_span_x = (cols - 1) * resolution_m
        center_span_y = (rows - 1) * resolution_m

        print("\nNominal metric dimensions")
        print("-------------------------")
        print(f"resolution             : {resolution_m:.9f} m/cell")
        print(
            "cell-edge coverage     : "
            f"{cell_coverage_x:.3f} x {cell_coverage_y:.3f} m"
        )
        print(
            "cell-center span       : "
            f"{center_span_x:.3f} x {center_span_y:.3f} m"
        )
        
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Output preview image (default: configured plot directory)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_pipeline_config()
    dem_path = config.truth_dem_path
    metadata_path = config.truth_metadata_path
    preview_path = args.preview or config.plots_path / "truth_dem_preview.png"

    if not dem_path.exists():
        print(f"DEM file not found: {dem_path}")
        sys.exit(1)

    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        sys.exit(1)

    dem = load_dem(dem_path)
    metadata = load_metadata(metadata_path)
    dem_hash = file_hash(dem_path)

    print(f"DEM file hash (SHA256): {dem_hash}")
    print_dem_statistics(dem, config.truth_raster.resolution_m)

    # Save a preview image of the DEM
    plt.figure(figsize=(8, 6))
    plt.imshow(dem, cmap="terrain", origin="lower")
    plt.colorbar(label="Elevation (m)")
    plt.title("Digital Elevation Model Preview")
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(preview_path, dpi=200)
    print(f"Saved DEM preview image: {preview_path}")
    
if __name__ == "__main__":
    main()

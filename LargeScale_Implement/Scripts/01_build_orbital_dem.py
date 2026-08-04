#!/usr/bin/env python3
"""Create an exact 5 m/cell orbital DEM from the 1.5 m truth DEM.

The source DEM covers 2001 m x 2001 m using 1334 x 1334 cells.
The output uses a centered 2000 m x 2000 m region with 400 x 400
cells at exactly 5 m/cell.

Resampling is area-weighted averaging, not interpolation. This
suppresses terrain smaller than the orbital-map resolution and is
appropriate for creating a controlled coarse localization prior.
"""

from __future__ import annotations

import hashlib
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_config import add_resolution_argument, load_resolution_config


def sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML metadata."""
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping in {path}, got {type(data).__name__}"
        )

    return data


def build_overlap_matrix(
    source_count: int,
    source_resolution_m: float,
    target_count: int,
    target_resolution_m: float,
    target_offset_m: float,
) -> csr_matrix:
    """Build a sparse 1-D area-overlap resampling matrix.

    Each target row contains normalized overlap lengths with the source
    cells. Therefore, each target row should sum to one.
    """
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []

    source_extent_m = source_count * source_resolution_m

    for target_index in range(target_count):
        target_left = (
            target_offset_m + target_index * target_resolution_m
        )
        target_right = target_left + target_resolution_m

        if target_left < 0.0 or target_right > source_extent_m:
            raise ValueError(
                "Target cell extends outside source coverage: "
                f"[{target_left}, {target_right}] versus "
                f"[0, {source_extent_m}]"
            )

        first_source = max(
            0,
            int(np.floor(target_left / source_resolution_m)),
        )
        last_source = min(
            source_count - 1,
            int(np.ceil(target_right / source_resolution_m)) - 1,
        )

        for source_index in range(first_source, last_source + 1):
            source_left = source_index * source_resolution_m
            source_right = source_left + source_resolution_m

            overlap = max(
                0.0,
                min(target_right, source_right)
                - max(target_left, source_left),
            )

            if overlap <= 0.0:
                continue

            rows.append(target_index)
            columns.append(source_index)
            values.append(overlap / target_resolution_m)

    matrix = csr_matrix(
        (values, (rows, columns)),
        shape=(target_count, source_count),
        dtype=np.float64,
    )

    row_sums = np.asarray(matrix.sum(axis=1)).ravel()

    if not np.allclose(row_sums, 1.0, atol=1e-12):
        raise RuntimeError(
            "Area-overlap matrix is not normalized. "
            f"Row-sum range: {row_sums.min()} to {row_sums.max()}"
        )

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_resolution_argument(parser)
    args = parser.parse_args()
    config = load_resolution_config(args.resolution)

    SOURCE_DEM_PATH = config.truth_dem_path
    SOURCE_METADATA_PATH = config.truth_metadata_path
    OUTPUT_DEM_PATH = config.orbital_dem_path
    OUTPUT_MASK_PATH = config.orbital_mask_path
    OUTPUT_METADATA_PATH = config.orbital_metadata_path
    SOURCE_RESOLUTION_M = config.truth_raster.resolution_m
    TARGET_RESOLUTION_M = config.orbital_raster.resolution_m
    TARGET_EXTENT_M = config.orbital_raster.shape[0] * TARGET_RESOLUTION_M

    if SOURCE_DEM_PATH.resolve() == OUTPUT_DEM_PATH.resolve():
        raise ValueError("Source and target DEM paths must differ")
    if not SOURCE_DEM_PATH.is_file():
        raise FileNotFoundError(
            f"Source DEM does not exist: {SOURCE_DEM_PATH}"
        )

    source_dem = np.load(SOURCE_DEM_PATH)

    if source_dem.ndim != 2:
        raise ValueError(
            f"Expected a 2-D DEM, received shape {source_dem.shape}"
        )

    if not np.isfinite(source_dem).all():
        raise ValueError(
            "This first resampling implementation requires a fully "
            "finite source DEM."
        )

    source_rows, source_columns = source_dem.shape

    source_height_m = source_rows * SOURCE_RESOLUTION_M
    source_width_m = source_columns * SOURCE_RESOLUTION_M

    expected_source_height_m = (
        config.truth_raster.shape[0] * SOURCE_RESOLUTION_M
    )
    expected_source_width_m = (
        config.truth_raster.shape[1] * SOURCE_RESOLUTION_M
    )
    if not np.isclose(source_height_m, expected_source_height_m):
        raise ValueError(
            f"Unexpected source height: {source_height_m} m"
        )

    if not np.isclose(source_width_m, expected_source_width_m):
        raise ValueError(
            f"Unexpected source width: {source_width_m} m"
        )

    target_rows = int(round(TARGET_EXTENT_M / TARGET_RESOLUTION_M))
    target_columns = int(round(TARGET_EXTENT_M / TARGET_RESOLUTION_M))

    if target_rows * TARGET_RESOLUTION_M != TARGET_EXTENT_M:
        raise ValueError("Target vertical extent is not exactly divisible")

    if target_columns * TARGET_RESOLUTION_M != TARGET_EXTENT_M:
        raise ValueError("Target horizontal extent is not exactly divisible")

    vertical_crop_m = source_height_m - TARGET_EXTENT_M
    horizontal_crop_m = source_width_m - TARGET_EXTENT_M

    top_offset_m = vertical_crop_m / 2.0
    left_offset_m = horizontal_crop_m / 2.0

    row_weights = build_overlap_matrix(
        source_count=source_rows,
        source_resolution_m=SOURCE_RESOLUTION_M,
        target_count=target_rows,
        target_resolution_m=TARGET_RESOLUTION_M,
        target_offset_m=top_offset_m,
    )

    column_weights = build_overlap_matrix(
        source_count=source_columns,
        source_resolution_m=SOURCE_RESOLUTION_M,
        target_count=target_columns,
        target_resolution_m=TARGET_RESOLUTION_M,
        target_offset_m=left_offset_m,
    )

    # Separable 2-D area averaging:
    #
    # target = W_row @ source @ W_column.T
    #
    # The second multiplication is written in transposed form so both
    # sparse matrices multiply from the left.
    intermediate = row_weights @ source_dem.astype(np.float64)
    target_dem = (column_weights @ intermediate.T).T

    target_dem = np.asarray(target_dem, dtype=np.float32)
    target_valid_mask = np.isfinite(target_dem)

    if target_dem.shape != (target_rows, target_columns):
        raise RuntimeError(
            f"Unexpected target shape: {target_dem.shape}"
        )

    if not target_valid_mask.all():
        raise RuntimeError("Output DEM unexpectedly contains invalid cells")

    OUTPUT_DEM_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DEM_PATH, target_dem)
    np.save(OUTPUT_MASK_PATH, target_valid_mask)

    source_metadata = load_yaml(SOURCE_METADATA_PATH)

    metadata = {
        "name": "apollo17_orbital_prior_5m",
        "description": (
            "Area-averaged 5 m localization prior generated from the "
            "1.5 m OmniLRS Apollo 17 simulation-truth DEM."
        ),
        "source": {
            "dem_path": str(SOURCE_DEM_PATH.relative_to(ROOT)),
            "metadata_path": str(
                SOURCE_METADATA_PATH.relative_to(ROOT)
            ),
            "shape": [int(source_rows), int(source_columns)],
            "resolution_m": SOURCE_RESOLUTION_M,
            "cell_edge_extent_m": [
                float(source_width_m),
                float(source_height_m),
            ],
            "dem_sha256": sha256(SOURCE_DEM_PATH),
            "metadata_sha256": (
                sha256(SOURCE_METADATA_PATH)
                if SOURCE_METADATA_PATH.is_file()
                else None
            ),
        },
        "target": {
            "dem_path": str(OUTPUT_DEM_PATH.relative_to(ROOT)),
            "valid_mask_path": str(
                OUTPUT_MASK_PATH.relative_to(ROOT)
            ),
            "shape": [int(target_rows), int(target_columns)],
            "resolution_m": TARGET_RESOLUTION_M,
            "cell_edge_extent_m": [
                TARGET_EXTENT_M,
                TARGET_EXTENT_M,
            ],
            "resampling": "area_weighted_average",
            "array_orientation": "preserved_from_source",
            "source_array_edge_bounds_m": {
                "column_min": float(left_offset_m),
                "column_max": float(
                    left_offset_m + TARGET_EXTENT_M
                ),
                "row_min": float(top_offset_m),
                "row_max": float(
                    top_offset_m + TARGET_EXTENT_M
                ),
            },
            "first_cell_center_in_source_array_m": {
                "column": float(
                    left_offset_m + TARGET_RESOLUTION_M / 2.0
                ),
                "row": float(
                    top_offset_m + TARGET_RESOLUTION_M / 2.0
                ),
            },
            "last_cell_center_in_source_array_m": {
                "column": float(
                    left_offset_m
                    + TARGET_EXTENT_M
                    - TARGET_RESOLUTION_M / 2.0
                ),
                "row": float(
                    top_offset_m
                    + TARGET_EXTENT_M
                    - TARGET_RESOLUTION_M / 2.0
                ),
            },
            "cropped_border_m": {
                "left": float(left_offset_m),
                "right": float(horizontal_crop_m - left_offset_m),
                "top": float(top_offset_m),
                "bottom": float(vertical_crop_m - top_offset_m),
            },
        },
        "elevation_statistics_m": {
            "minimum": float(np.min(target_dem)),
            "maximum": float(np.max(target_dem)),
            "mean": float(np.mean(target_dem)),
            "standard_deviation": float(np.std(target_dem)),
            "relief": float(np.max(target_dem) - np.min(target_dem)),
        },
        "coordinate_status": {
            "world_origin_verified": False,
            "row_world_direction_verified": False,
            "column_world_direction_verified": False,
            "note": (
                "This product preserves source array orientation. "
                "World-coordinate metadata must be populated after "
                "verifying the source DEM georeferencing."
            ),
        },
        "original_source_metadata": source_metadata,
    }

    with OUTPUT_METADATA_PATH.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            metadata,
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    print("5 m orbital DEM created")
    print("------------------------")
    print(f"source shape          : {source_dem.shape}")
    print(f"source resolution     : {SOURCE_RESOLUTION_M:.6f} m")
    print(
        f"source coverage       : "
        f"{source_width_m:.3f} x {source_height_m:.3f} m"
    )
    print(f"target shape          : {target_dem.shape}")
    print(f"target resolution     : {TARGET_RESOLUTION_M:.6f} m")
    print(
        f"target coverage       : "
        f"{TARGET_EXTENT_M:.3f} x {TARGET_EXTENT_M:.3f} m"
    )
    print(
        f"centered border crop  : "
        f"{left_offset_m:.3f} m on each horizontal side, "
        f"{top_offset_m:.3f} m on each vertical side"
    )
    print(f"minimum elevation     : {target_dem.min():.6f} m")
    print(f"maximum elevation     : {target_dem.max():.6f} m")
    print(f"mean elevation        : {target_dem.mean():.6f} m")
    print(f"standard deviation    : {target_dem.std():.6f} m")
    print(
        f"total relief          : "
        f"{target_dem.max() - target_dem.min():.6f} m"
    )
    print()
    print(f"DEM written to        : {OUTPUT_DEM_PATH}")
    print(f"mask written to       : {OUTPUT_MASK_PATH}")
    print(f"metadata written to   : {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    main()

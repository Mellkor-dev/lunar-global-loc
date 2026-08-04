"""Validated shared configuration and raster-coordinate utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "apollo17_5m.yaml"


@dataclass(frozen=True)
class RasterConfig:
    resolution_m: float
    shape: tuple[int, int]
    first_x_center_m: float
    first_y_center_m: float

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return x centers eastward and y centers north-to-south."""
        rows, columns = self.shape
        x = self.first_x_center_m + np.arange(columns) * self.resolution_m
        y = self.first_y_center_m - np.arange(rows) * self.resolution_m
        return x, y

    def indices_to_xyz(
        self,
        row_column: np.ndarray,
        elevation: np.ndarray,
    ) -> np.ndarray:
        indices = np.asarray(row_column)
        if indices.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError(f"Indices must have shape (N, 2), got {indices.shape}")

        indices = indices.astype(np.int64, copy=False)
        rows = indices[:, 0]
        columns = indices[:, 1]
        if (
            np.any(rows < 0)
            or np.any(rows >= self.shape[0])
            or np.any(columns < 0)
            or np.any(columns >= self.shape[1])
        ):
            raise IndexError("Feature indices fall outside the configured raster")

        return np.column_stack(
            (
                self.first_x_center_m + columns * self.resolution_m,
                self.first_y_center_m - rows * self.resolution_m,
                elevation[rows, columns],
            )
        ).astype(np.float64)


@dataclass(frozen=True)
class FeatureDetectionConfig:
    kind: str
    radius_cells: int
    distance_m: float
    flatness_threshold_m: float
    local_min_valid_fraction: float

    def radius_for_resolution(self, resolution_m: float) -> int:
        radius = int(round(self.distance_m / resolution_m))
        if not np.isclose(radius * resolution_m, self.distance_m):
            raise ValueError(
                f"Detection distance {self.distance_m} m is not an integer "
                f"multiple of resolution {resolution_m} m"
            )
        return radius


@dataclass(frozen=True)
class FeatureDetectionTarget:
    """One raster and its resolution-specific detector parameters."""

    name: str
    dem_path: Path
    metadata_path: Path
    output_path: Path
    preview_path: Path
    raster: RasterConfig
    radius_cells: int
    distance_m: float
    flatness_threshold_m: float
    catalogue_role: str


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    truth_dem_path: Path
    truth_metadata_path: Path
    orbital_dem_path: Path
    orbital_metadata_path: Path
    orbital_mask_path: Path
    global_features_path: Path
    truth_features_path: Path
    leveled_maps_path: Path
    gridded_maps_path: Path
    local_features_path: Path
    feature_uncertainty_path: Path
    dem_qa_path: Path
    feature_validation_path: Path
    plots_path: Path
    results_path: Path
    captures_path: Path
    capture_validation_path: Path
    truth_raster: RasterConfig
    orbital_raster: RasterConfig
    features: FeatureDetectionConfig
    feature_detection_targets: dict[str, FeatureDetectionTarget]
    intrinsic_horizontal_sigma_m: float
    intrinsic_vertical_sigma_m: float
    match_gate_fraction: float


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def _raster_config(data: dict[str, Any], name: str) -> RasterConfig:
    shape = tuple(int(value) for value in data["shape"])
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError(f"{name}.shape must contain two positive integers")
    resolution = float(data["resolution_m"])
    first = _mapping(data["first_cell_center_m"], f"{name}.first_cell_center_m")
    if resolution <= 0.0:
        raise ValueError(f"{name}.resolution_m must be positive")
    return RasterConfig(
        resolution_m=resolution,
        shape=(shape[0], shape[1]),
        first_x_center_m=float(first["x"]),
        first_y_center_m=float(first["y"]),
    )


def load_pipeline_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> PipelineConfig:
    """Load and validate the shared Phase 1 configuration."""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw = _mapping(raw, "root")
    paths = _mapping(raw["paths"], "paths")
    feature_data = _mapping(raw["feature_detection"], "feature_detection")
    uncertainty = _mapping(
        raw["global_feature_uncertainty"],
        "global_feature_uncertainty",
    )

    truth_raster = _raster_config(
        _mapping(raw["truth_dem"], "truth_dem"),
        "truth_dem",
    )
    orbital_raster = _raster_config(
        _mapping(raw["orbital_dem"], "orbital_dem"),
        "orbital_dem",
    )
    features = FeatureDetectionConfig(
        kind=str(feature_data["feature_kind"]).lower(),
        radius_cells=int(feature_data["cell_radius"]),
        distance_m=float(feature_data["detection_distance_m"]),
        flatness_threshold_m=float(feature_data["flatness_threshold_m"]),
        local_min_valid_fraction=float(
            feature_data["local_min_valid_fraction"]
        ),
    )
    if features.kind not in {"peak", "crater"}:
        raise ValueError("feature_kind must be 'peak' or 'crater'")
    if features.radius_cells < 1:
        raise ValueError("feature_detection.cell_radius must be positive")
    if not np.isclose(
        features.radius_cells * orbital_raster.resolution_m,
        features.distance_m,
    ):
        raise ValueError(
            "Configured cell radius and physical detection distance disagree"
        )
    if features.flatness_threshold_m < 0.0:
        raise ValueError("flatness_threshold_m must be non-negative")
    if not 0.0 <= features.local_min_valid_fraction <= 1.0:
        raise ValueError("local_min_valid_fraction must lie in [0, 1]")

    target_data = _mapping(
        feature_data.get("resolutions", {}),
        "feature_detection.resolutions",
    )
    feature_detection_targets: dict[str, FeatureDetectionTarget] = {}
    for target_name, unvalidated_target in target_data.items():
        target = _mapping(
            unvalidated_target,
            f"feature_detection.resolutions.{target_name}",
        )
        target_raster = _raster_config(
            target,
            f"feature_detection.resolutions.{target_name}",
        )
        radius_cells = int(target["cell_radius"])
        distance_m = float(target["detection_distance_m"])
        if radius_cells < 1:
            raise ValueError(f"Detector radius for {target_name} must be positive")
        if not np.isclose(radius_cells * target_raster.resolution_m, distance_m):
            raise ValueError(
                f"Detector radius for {target_name} does not equal its "
                "physical detection distance"
            )
        flatness = float(target["flatness_threshold_m"])
        if flatness < 0.0:
            raise ValueError(
                f"Detector flatness threshold for {target_name} must be non-negative"
            )
        role = str(target.get("catalogue_role", "global")).lower()
        if role not in {"global", "truth"}:
            raise ValueError(
                f"catalogue_role for {target_name} must be 'global' or 'truth'"
            )
        feature_detection_targets[str(target_name)] = FeatureDetectionTarget(
            name=str(target_name),
            dem_path=PROJECT_ROOT / str(target["dem"]),
            metadata_path=PROJECT_ROOT / str(target["metadata"]),
            output_path=PROJECT_ROOT / str(target["output"]),
            preview_path=PROJECT_ROOT / str(target["preview"]),
            raster=target_raster,
            radius_cells=radius_cells,
            distance_m=distance_m,
            flatness_threshold_m=flatness,
            catalogue_role=role,
        )

    def resolved(key: str) -> Path:
        return PROJECT_ROOT / str(paths[key])

    return PipelineConfig(
        config_path=path,
        truth_dem_path=resolved("truth_dem"),
        truth_metadata_path=resolved("truth_metadata"),
        orbital_dem_path=resolved("orbital_dem"),
        orbital_metadata_path=resolved("orbital_metadata"),
        orbital_mask_path=resolved("orbital_valid_mask"),
        global_features_path=resolved("global_features"),
        truth_features_path=resolved("truth_features"),
        leveled_maps_path=resolved("leveled_local_maps"),
        gridded_maps_path=resolved("gridded_local_maps"),
        local_features_path=resolved("local_features"),
        feature_uncertainty_path=resolved("feature_uncertainty"),
        dem_qa_path=resolved("dem_qa"),
        feature_validation_path=resolved("feature_validation"),
        plots_path=resolved("plots"),
        results_path=resolved("results"),
        captures_path=resolved("captures"),
        capture_validation_path=resolved("capture_validation"),
        truth_raster=truth_raster,
        orbital_raster=orbital_raster,
        features=features,
        feature_detection_targets=feature_detection_targets,
        intrinsic_horizontal_sigma_m=float(
            uncertainty["intrinsic_horizontal_sigma_m"]
        ),
        intrinsic_vertical_sigma_m=float(
            uncertainty["intrinsic_vertical_sigma_m"]
        ),
        match_gate_fraction=float(
            uncertainty["match_gate_fraction_of_detection_distance"]
        ),
    )

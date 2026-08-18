"""Validated shared configuration and raster-coordinate utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "haworth.yaml"


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
    selectable: bool


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
    base_to_lidar_translation_m: tuple[float, float, float]
    site_selection_manifest_path: Path
    site_selection_maximum_sites: int
    site_selection_random_seed: int
    truth_source_label: str
    truth_source_profile: str

    @property
    def available_resolutions(self) -> tuple[str, ...]:
        """Return profiles from finest to coarsest physical cell size."""
        return tuple(
            name
            for name, _target in sorted(
                (
                    item
                    for item in self.feature_detection_targets.items()
                    if item[1].selectable
                ),
                key=lambda item: (item[1].raster.resolution_m, item[0]),
            )
        )

    def for_resolution(self, name: str) -> "PipelineConfig":
        """Return a pipeline view rooted in one resolution subdirectory."""
        if name not in self.available_resolutions:
            choices = ", ".join(self.available_resolutions)
            raise ValueError(f"Unknown resolution '{name}'; choose from {choices}")

        target = self.feature_detection_targets[name]
        truth_targets = [
            candidate
            for candidate in self.feature_detection_targets.values()
            if candidate.catalogue_role == "truth"
        ]
        if len(truth_targets) != 1:
            names = ", ".join(target.name for target in truth_targets) or "none"
            raise ValueError(
                f"Exactly one truth detector target is required; found {names}"
            )
        truth_target = truth_targets[0]
        directory_name = target.output_path.parent.name
        local_root = PROJECT_ROOT / "local_maps" / directory_name
        simulation_root = PROJECT_ROOT / "sim" / directory_name
        validation_root = target.output_path.parent / "validation"
        features = replace(
            self.features,
            radius_cells=target.radius_cells,
            distance_m=target.distance_m,
            flatness_threshold_m=target.flatness_threshold_m,
        )
        mask_name = f"orbital_valid_mask_{name}.npy"
        return replace(
            self,
            truth_dem_path=(
                truth_target.dem_path if truth_target else self.truth_dem_path
            ),
            truth_metadata_path=(
                truth_target.metadata_path
                if truth_target
                else self.truth_metadata_path
            ),
            truth_features_path=(
                truth_target.output_path
                if truth_target
                else self.truth_features_path
            ),
            truth_raster=(
                truth_target.raster if truth_target else self.truth_raster
            ),
            orbital_dem_path=target.dem_path,
            orbital_metadata_path=target.metadata_path,
            orbital_mask_path=target.output_path.parent / mask_name,
            global_features_path=target.output_path,
            leveled_maps_path=local_root / "leveled",
            gridded_maps_path=local_root / "gridded",
            local_features_path=local_root / "features",
            feature_uncertainty_path=(
                validation_root / "global_feature_uncertainty.json"
            ),
            dem_qa_path=target.output_path.parent / "qa",
            feature_validation_path=validation_root,
            plots_path=PROJECT_ROOT / "plots" / directory_name,
            results_path=PROJECT_ROOT / "results" / directory_name,
            captures_path=simulation_root,
            capture_validation_path=simulation_root / "validation",
            orbital_raster=target.raster,
            features=features,
        )


def add_resolution_argument(
    parser: argparse.ArgumentParser,
    *,
    include_all: bool = False,
) -> None:
    """Add the common pipeline resolution selector to a script parser."""
    config = load_pipeline_config()
    choices = list(config.available_resolutions)
    if include_all:
        choices.append("all")
    parser.add_argument(
        "--resolution",
        choices=choices,
        default="all" if include_all else "5m",
        help=(
            "Resolution workspace to use "
            f"(default: {'all' if include_all else '5m'})"
        ),
    )


def load_resolution_config(name: str) -> PipelineConfig:
    """Load configuration and select one resolution workspace."""
    return load_pipeline_config().for_resolution(name)


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
    sensor = _mapping(raw.get("sensor", {}), "sensor")
    base_to_lidar_translation = np.asarray(
        sensor.get("base_to_lidar_translation_m", [-0.15, 0.0, 0.415]),
        dtype=np.float64,
    )
    if (
        base_to_lidar_translation.shape != (3,)
        or not np.isfinite(base_to_lidar_translation).all()
    ):
        raise ValueError(
            "sensor.base_to_lidar_translation_m must contain three finite values"
        )
    site_selection = _mapping(raw.get("site_selection", {}), "site_selection")
    site_selection_maximum_sites = int(site_selection.get("maximum_sites", 50))
    site_selection_random_seed = int(site_selection.get("random_seed", 29))
    if site_selection_maximum_sites <= 0:
        raise ValueError("site_selection.maximum_sites must be positive")
    site_selection_manifest = str(
        site_selection.get("manifest", "sim/selected_sites.json")
    )

    truth_data = _mapping(raw["truth_dem"], "truth_dem")
    truth_source_label = str(truth_data.get("source_label", "")).strip()
    truth_source_profile = str(truth_data.get("source_profile", "")).strip()
    if not truth_source_label or not truth_source_profile:
        raise ValueError(
            "truth_dem.source_label and truth_dem.source_profile are required"
        )
    truth_raster = _raster_config(
        truth_data,
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
            selectable=bool(target.get("selectable", True)),
        )

    if truth_source_profile not in feature_detection_targets:
        raise ValueError(
            f"Truth source profile '{truth_source_profile}' has no detector target"
        )
    configured_truth_target = feature_detection_targets[truth_source_profile]
    if configured_truth_target.catalogue_role != "truth":
        raise ValueError(
            f"Truth source profile '{truth_source_profile}' must have "
            "catalogue_role: truth"
        )

    def resolved(key: str) -> Path:
        return PROJECT_ROOT / str(paths[key])

    if configured_truth_target.dem_path != resolved("truth_dem"):
        raise ValueError("Truth detector DEM does not match paths.truth_dem")
    if configured_truth_target.metadata_path != resolved("truth_metadata"):
        raise ValueError(
            "Truth detector metadata does not match paths.truth_metadata"
        )
    if configured_truth_target.output_path != resolved("truth_features"):
        raise ValueError(
            "Truth detector output does not match paths.truth_features"
        )
    if configured_truth_target.raster != truth_raster:
        raise ValueError("Truth detector raster does not match truth_dem raster")

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
        base_to_lidar_translation_m=tuple(
            float(value) for value in base_to_lidar_translation
        ),
        site_selection_manifest_path=PROJECT_ROOT / site_selection_manifest,
        site_selection_maximum_sites=site_selection_maximum_sites,
        site_selection_random_seed=site_selection_random_seed,
        truth_source_label=truth_source_label,
        truth_source_profile=truth_source_profile,
    )

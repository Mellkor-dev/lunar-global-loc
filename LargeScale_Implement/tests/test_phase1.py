"""Regression tests for shared Phase 1 configuration and feature geometry."""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.dilation_detector import detect_craters
from pipeline_config import load_pipeline_config


def test_feature_scale_is_consistent_across_rasters() -> None:
    config = load_pipeline_config()
    assert config.features.radius_for_resolution(
        config.orbital_raster.resolution_m
    ) == 3
    assert config.features.radius_for_resolution(
        config.truth_raster.resolution_m
    ) == 10


def test_resolution_specific_detector_profiles() -> None:
    config = load_pipeline_config()
    targets = config.feature_detection_targets
    assert set(targets) == {"0p25m", "1p5m", "5m"}
    assert targets["0p25m"].radius_cells == 60
    assert targets["1p5m"].radius_cells == 10
    assert targets["5m"].radius_cells == 3
    for target in targets.values():
        assert np.isclose(
            target.radius_cells * target.raster.resolution_m,
            target.distance_m,
        )


def test_resolution_workspace_paths_are_projected_consistently() -> None:
    config = load_pipeline_config().for_resolution("0p25m")
    assert config.orbital_raster.resolution_m == 0.25
    assert config.global_features_path.parent.name == "0p25m_px"
    assert config.gridded_maps_path.parts[-2:] == ("0p25m_px", "gridded")
    assert config.results_path.name == "0p25m_px"
    assert config.captures_path.name == "0p25m_px"
    assert config.features.radius_cells == 60
    assert config.truth_raster.resolution_m == 1.5


def test_truth_raster_coordinate_contract() -> None:
    config = load_pipeline_config()
    indices = np.array([[0, 0], [1333, 1333]])
    elevation = np.zeros(config.truth_raster.shape)
    xyz = config.truth_raster.indices_to_xyz(indices, elevation)
    assert np.allclose(xyz[0], (-999.75, 999.75, 0.0))
    assert np.allclose(xyz[1], (999.75, -999.75, 0.0))


def test_masked_crater_detection_and_border_exclusion() -> None:
    elevation = np.full((9, 9), 2.0)
    elevation[4, 4] = 0.0
    elevation[0, 0] = -1.0
    elevation[1, 8] = np.nan

    with_border = detect_craters(
        elevation,
        n=2,
        flatness_eps=0.1,
        min_valid_fraction=0.1,
    )
    without_border = detect_craters(
        elevation,
        n=2,
        flatness_eps=0.1,
        min_valid_fraction=0.1,
        exclude_border=True,
    )
    assert any(np.array_equal(item, (0, 0)) for item in with_border)
    assert np.array_equal(without_border, np.array([[4, 4]]))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q"]))

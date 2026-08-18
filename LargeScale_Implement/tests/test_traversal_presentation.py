"""Tests for the presentation traversal ordering and environment labels."""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import load_pipeline_config
from traversal_presentation import environment_display_name, spatial_route_order


def test_environment_name_matches_workspace_config() -> None:
    config = load_pipeline_config()
    expected = {
        "apollo17_5m.yaml": "Apollo 17",
        "apollo11.yaml": "Apollo 11",
        "haworth.yaml": "Haworth",
        "apollo14.yaml": "Apollo 14",
        "apollo15.yaml": "Apollo 15",
    }[config.config_path.name]
    assert environment_display_name(config) == expected


def test_spatial_order_is_deterministic_and_nearest_neighbour() -> None:
    sites = np.array([1, 2, 3, 4])
    xy = np.array(
        [
            [0.0, 0.0],
            [1.0, -1.0],
            [20.0, -2.0],
            [2.0, -1.5],
        ]
    )
    order = spatial_route_order(sites, xy)
    assert sites[order].tolist() == [1, 2, 4, 3]


def test_spatial_order_handles_one_site() -> None:
    order = spatial_route_order(np.array([8]), np.array([[3.0, 4.0]]))
    assert order.tolist() == [0]

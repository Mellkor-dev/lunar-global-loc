"""Tests for exact refined-DEM block downsampling."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "Scripts" / "17_build_refined_dem_resolutions.py"
SPEC = spec_from_file_location("build_refined_dem_resolutions", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_block_average() -> None:
    source = np.arange(64, dtype=np.float32).reshape(8, 8)
    result = MODULE.exact_block_average(source, 4)
    expected = np.array([[13.5, 17.5], [45.5, 49.5]], dtype=np.float32)
    assert result.dtype == np.float32
    assert np.array_equal(result, expected)


def test_exact_block_average_rejects_incompatible_shape() -> None:
    with np.testing.assert_raises(ValueError):
        MODULE.exact_block_average(np.zeros((7, 8)), 4)

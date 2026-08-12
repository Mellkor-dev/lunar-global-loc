"""Regression tests for the presentation-summary statistics."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "Scripts" / "14_results_presentation.py"
SPEC = spec_from_file_location("results_presentation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def result(site: int, error: float, x: float = 0.0) -> dict[str, object]:
    return {
        "site": site,
        "estimated_x_m": x,
        "estimated_y_m": 0.0,
        "truth_x_m": 0.0,
        "truth_y_m": 0.0,
        "xy_error_m": error,
    }


def test_oracle_best_uses_minimum_error_and_later_stage_for_ties() -> None:
    records = {
        "darces": [result(1, 5.0), result(2, 3.0)],
        "ransac": [result(1, 2.0), result(2, 3.0)],
        "moga": [result(1, 4.0), result(2, 3.0)],
    }

    selected = MODULE.select_oracle_best(records)

    assert [item["site"] for item in selected] == [1, 2]
    assert [item["selected_stage"] for item in selected] == ["ransac", "moga"]
    assert [item["xy_error_m"] for item in selected] == [2.0, 3.0]


def test_error_statistics_reports_percentiles_mean_and_population_std() -> None:
    stats = MODULE.error_statistics(np.asarray((1.0, 2.0, 3.0, 4.0, 10.0)))

    assert stats["solved_sites"] == 5
    assert stats["median_xy_error_m"] == 3.0
    assert stats["mean_xy_error_m"] == 4.0
    assert np.isclose(stats["std_xy_error_m"], np.std((1.0, 2.0, 3.0, 4.0, 10.0)))
    assert stats["p10_xy_error_m"] < stats["p25_xy_error_m"]
    assert stats["p75_xy_error_m"] < stats["p90_xy_error_m"]


def test_empty_statistics_are_explicitly_unavailable() -> None:
    stats = MODULE.error_statistics(np.empty(0))

    assert stats["solved_sites"] == 0
    assert stats["median_xy_error_m"] is None
    assert stats["std_xy_error_m"] is None

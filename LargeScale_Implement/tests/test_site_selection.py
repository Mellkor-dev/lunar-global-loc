"""Tests for reproducible dataset-level capture-site selection."""

from pathlib import Path
import json
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from site_selection import (
    complete_capture_sites,
    load_or_create_site_manifest,
    select_random_sites,
)


def make_capture_set(root: Path, sites: range) -> None:
    for directory in ("pointcloud_scans", "odom_scans", "transform_scan"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for site in sites:
        np.save(root / "pointcloud_scans" / f"scan_site_{site:03d}.npy", np.zeros((1, 3)))
        np.save(root / "odom_scans" / f"odom_site_{site:03d}.npy", np.zeros(7))
        np.savez(root / "transform_scan" / f"transform_site_{site:03d}.npz", T=np.eye(4))


def test_random_selection_is_unique_reproducible_and_not_a_prefix() -> None:
    first = select_random_sites(range(1, 101), maximum_sites=50, random_seed=29)
    second = select_random_sites(range(1, 101), maximum_sites=50, random_seed=29)
    assert first == second
    assert len(first) == len(set(first)) == 50
    assert first == sorted(first)
    assert first != list(range(1, 51))


def test_selection_uses_every_site_when_dataset_has_at_most_limit() -> None:
    assert select_random_sites(range(1, 29), maximum_sites=50, random_seed=29) == list(
        range(1, 29)
    )


def test_manifest_is_shared_and_regenerates_when_inventory_changes(tmp_path: Path) -> None:
    captures = tmp_path / "captures"
    manifest = tmp_path / "sim" / "selected_sites.json"
    make_capture_set(captures, range(1, 61))
    selected = load_or_create_site_manifest(
        captures, manifest, maximum_sites=50, random_seed=29
    )
    assert selected == load_or_create_site_manifest(
        captures, manifest, maximum_sites=50, random_seed=29
    )
    assert len(selected) == 50
    payload = json.loads(manifest.read_text())
    assert payload["selected_sites"] == selected
    assert payload["available_site_count"] == 60

    make_capture_set(captures, range(61, 62))
    updated = load_or_create_site_manifest(
        captures, manifest, maximum_sites=50, random_seed=29
    )
    assert len(updated) == 50
    assert set(updated).issubset(complete_capture_sites(captures))
    assert json.loads(manifest.read_text())["available_site_count"] == 61

"""Deterministic dataset-level selection of captured localization sites."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np


SITE_PATTERN = re.compile(r"site_(\d+)")
CAPTURE_PATTERNS = {
    "pointcloud_scans": "scan_site_*.npy",
    "odom_scans": "odom_site_*.npy",
    "transform_scan": "transform_site_*.npz",
}


def _site_number(path: Path) -> int:
    match = SITE_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot extract a site number from {path.name}")
    return int(match.group(1))


def complete_capture_sites(captures_path: Path) -> list[int]:
    """Return sites having synchronized cloud, odometry, and transform files."""
    captures_path = Path(captures_path)
    site_sets: list[set[int]] = []
    for directory_name, pattern in CAPTURE_PATTERNS.items():
        directory = captures_path / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Capture directory does not exist: {directory}")
        site_sets.append({_site_number(path) for path in directory.glob(pattern)})
    if not all(site_sets):
        raise FileNotFoundError(
            f"One or more capture directories under {captures_path} contain no data"
        )
    return sorted(set.intersection(*site_sets))


def select_random_sites(
    available_sites: Iterable[int],
    *,
    maximum_sites: int,
    random_seed: int,
) -> list[int]:
    """Choose up to ``maximum_sites`` unique IDs, returning them sorted."""
    available = np.asarray(sorted(set(int(site) for site in available_sites)))
    if maximum_sites <= 0:
        raise ValueError("maximum_sites must be positive")
    if len(available) == 0:
        raise ValueError("No complete capture sites are available")
    if len(available) <= maximum_sites:
        return available.astype(int).tolist()
    generator = np.random.default_rng(random_seed)
    return sorted(
        int(site)
        for site in generator.choice(available, size=maximum_sites, replace=False)
    )


def _available_digest(available_sites: Iterable[int]) -> str:
    encoded = ",".join(str(int(site)) for site in available_sites).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_or_create_site_manifest(
    captures_path: Path,
    manifest_path: Path,
    *,
    maximum_sites: int = 50,
    random_seed: int = 29,
    refresh: bool = False,
) -> list[int]:
    """Load a compatible selection manifest or atomically create a new one.

    The manifest is shared by every resolution. A changed capture inventory,
    selection limit, or seed invalidates it and deterministically regenerates
    the random sample.
    """
    captures_path = Path(captures_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    available = complete_capture_sites(captures_path)
    digest = _available_digest(available)

    if manifest_path.is_file() and not refresh:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            selected = [int(site) for site in payload["selected_sites"]]
            compatible = (
                int(payload["maximum_sites"]) == maximum_sites
                and int(payload["random_seed"]) == random_seed
                and str(payload["available_sites_sha256"]) == digest
                and int(payload["available_site_count"]) == len(available)
                and selected == sorted(set(selected))
                and len(selected) == min(maximum_sites, len(available))
                and set(selected).issubset(available)
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            compatible = False
        if compatible:
            return selected

    selected = select_random_sites(
        available,
        maximum_sites=maximum_sites,
        random_seed=random_seed,
    )
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "selection_method": "numpy_default_rng_without_replacement",
            "maximum_sites": maximum_sites,
            "random_seed": random_seed,
            "available_site_count": len(available),
            "available_sites_sha256": digest,
            "selected_site_count": len(selected),
            "selected_sites": selected,
            "captures_path_used": str(captures_path),
        },
    )
    return selected


def selected_sites_for_config(config, *, refresh: bool = False) -> list[int]:
    """Resolve the shared site selection described by a PipelineConfig."""
    return load_or_create_site_manifest(
        config.captures_path,
        config.site_selection_manifest_path,
        maximum_sites=config.site_selection_maximum_sites,
        random_seed=config.site_selection_random_seed,
        refresh=refresh,
    )


def filter_paths_to_sites(paths: Iterable[Path], selected_sites: Iterable[int]) -> list[Path]:
    selected = set(int(site) for site in selected_sites)
    return sorted(
        (Path(path) for path in paths if _site_number(Path(path)) in selected),
        key=_site_number,
    )


def remove_unselected_site_artifacts(
    directory: Path,
    pattern: str,
    selected_sites: Iterable[int],
) -> int:
    """Delete stale generated per-site files outside the active manifest."""
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    selected = set(int(site) for site in selected_sites)
    removed = 0
    for path in directory.glob(pattern):
        if _site_number(path) not in selected:
            path.unlink()
            removed += 1
    return removed

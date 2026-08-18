#!/usr/bin/env python3
"""Create or inspect the shared random capture-site selection manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_config import add_resolution_argument, load_resolution_config
from site_selection import complete_capture_sites, selected_sites_for_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_resolution_argument(parser)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate the manifest using the configured random seed",
    )
    args = parser.parse_args()
    config = load_resolution_config(args.resolution)
    available = complete_capture_sites(config.captures_path)
    selected = selected_sites_for_config(config, refresh=args.refresh)
    print("Capture site selection")
    print("----------------------")
    print(f"Available complete sites: {len(available)}")
    print(f"Selected sites:           {len(selected)}")
    print(f"Maximum sites:            {config.site_selection_maximum_sites}")
    print(f"Random seed:              {config.site_selection_random_seed}")
    print(f"Manifest:                 {config.site_selection_manifest_path}")
    print(f"Site IDs:                 {selected}")


if __name__ == "__main__":
    main()

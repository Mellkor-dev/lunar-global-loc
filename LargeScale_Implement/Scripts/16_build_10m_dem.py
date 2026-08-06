#!/usr/bin/env python3
"""Build the configured 10 m/cell orbital DEM from the native 1.5 m DEM."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    builder = Path(__file__).with_name("01_build_orbital_dem.py")
    forwarded = sys.argv[1:]
    if any(argument == "--resolution" or argument.startswith("--resolution=") for argument in forwarded):
        raise ValueError("16_build_10m_dem.py always builds the 10m profile")
    sys.argv = [str(builder), "--resolution", "10m", *forwarded]
    runpy.run_path(str(builder), run_name="__main__")


if __name__ == "__main__":
    main()

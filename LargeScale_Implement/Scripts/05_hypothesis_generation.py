#!/usr/bin/env python3


from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

DEM_PATH = PROJECT_ROOT / "DEM" / "orbital_dem_5m.npy"
PLOT_DIRECTORY = PROJECT_ROOT / "plots"

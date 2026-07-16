#!/usr/bin/env python3

from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
try:
    import yaml
except ImportError:
    yaml = None
    print("Warning: PyYAML is not installed. Falling back to simple YAML loader.")

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), ".."),
)

def _load_simple_yaml(path):
    data = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if value.startswith("[") and value.endswith("]"):
                items = [
                    item.strip()
                    for item in value[1:-1].split(",")
                    if item.strip()
                ]
                parsed = []
                for item in items:
                    if item.lower() in ("true", "false", "null", "none"):
                        parsed.append(
                            True
                            if item.lower() == "true"
                            else False
                            if item.lower() == "false"
                            else None
                        )
                    else:
                        try:
                            parsed.append(int(item))
                        except ValueError:
                            try:
                                parsed.append(float(item))
                            except ValueError:
                                parsed.append(item.strip('"').strip("'"))
                data[key] = parsed
            else:
                if value.lower() in ("true", "false", "null", "none"):
                    data[key] = (
                        True
                        if value.lower() == "true"
                        else False
                        if value.lower() == "false"
                        else None
                    )
                else:
                    try:
                        data[key] = int(value)
                    except ValueError:
                        try:
                            data[key] = float(value)
                        except ValueError:
                            data[key] = value.strip('"').strip("'")
    return data

from maps.global_dem import GlobalDEM
from features.dilation_detector import detect_peaks


# ---------------------------------------------------------------------
# Load source DEM and metadata
# ---------------------------------------------------------------------

dem_dir = Path(
    "/home/soumyadeep/OmniLRS/"
    "assets/Terrains/SouthPole/ldem_87s_5mpp"
)

dem_path = dem_dir / "dem.npy"
metadata_path = dem_dir / "dem.yaml"

if not dem_path.exists():
    raise FileNotFoundError(dem_path)

if not metadata_path.exists():
    raise FileNotFoundError(metadata_path)

# Memory-map the source file because the regional DEM may be large.
raw_dem = np.load(
    dem_path,
    mmap_mode="r",
)

if yaml is not None:
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
else:
    metadata = _load_simple_yaml(metadata_path)

lxy = 5

print("Raw DEM shape:", raw_dem.shape)
print("Resolution:", lxy, "m/pixel")


# ---------------------------------------------------------------------
# Match OmniLRS orientation
# ---------------------------------------------------------------------

# OmniLRS internal representation:
# axis 0 = x, axis 1 = y
dem_xy = np.flip(raw_dem.T, axis=1)

# Convert to the conventional representation used by your code:
# axis 0 / row = y
# axis 1 / col = x
dem = dem_xy.T

height, width = dem.shape

# Metric coordinate of the centre of pixel [0, 0].
origin_x = -(width * lxy) / 2.0 + lxy / 2.0
origin_y = -(height * lxy) / 2.0 + lxy / 2.0


# ---------------------------------------------------------------------
# Crop around largescale_200km starting position
# ---------------------------------------------------------------------

center_x = -7636.0
center_y = 10060.0

# 10 km × 10 km gives approximately 2000 × 2000 pixels.
patch_width_m = 10000.0

center_col = int(
    round((center_x - origin_x) / lxy)
)
center_row = int(
    round((center_y - origin_y) / lxy)
)

half_pixels = int(
    np.ceil(patch_width_m / (2.0 * lxy))
)

row0 = max(0, center_row - half_pixels)
row1 = min(
    height,
    center_row + half_pixels + 1,
)

col0 = max(0, center_col - half_pixels)
col1 = min(
    width,
    center_col + half_pixels + 1,
)

if row0 >= row1 or col0 >= col1:
    raise RuntimeError(
        "Requested patch lies outside the DEM. "
        f"Centre pixel=({center_row}, {center_col}), "
        f"DEM shape={dem.shape}"
    )

dem = np.asarray(
    dem[row0:row1, col0:col1],
    dtype=np.float32,
)

patch_origin_x = origin_x + col0 * lxy
patch_origin_y = origin_y + row0 * lxy

gdem = GlobalDEM(
    elevation=dem,
    lxy=lxy,
    origin_xy=(
        patch_origin_x,
        patch_origin_y,
    ),
)

print("Patch shape:", dem.shape)
print("Patch origin:", gdem.origin_xy)
print(
    "Elevation range:",
    float(np.nanmin(dem)),
    float(np.nanmax(dem)),
)


# ---------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------

desired_detection_scale_m = 60.0

n = max(
    1,
    int(round(
        desired_detection_scale_m / gdem.lxy
    )),
)

peaks = detect_peaks(
    dem,
    n=n,
    flatness_eps=0.5,
)

print(
    f"Detected {len(peaks)} peaks with n={n} "
    f"(D_detect={n * gdem.lxy:.1f} m)"
)

print(
    peaks[:10],
    "..." if len(peaks) > 10 else "",
)


# ---------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------

output_path = Path(
    "maps/data/largescale_global_features_preview.png"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

plt.figure(figsize=(10, 8))

plt.imshow(
    dem,
    cmap="terrain",
    origin="lower",
)

if len(peaks) > 0:
    plt.scatter(
        peaks[:, 1],
        peaks[:, 0],
        s=15,
        marker="x",
        label="Detected topographic peaks",
    )

plt.colorbar(label="Elevation [m]")

plt.title(
    "LargeScale 87°S global DEM features\n"
    f"{gdem.lxy:.1f} m/pixel, "
    f"n={n}, "
    f"{len(peaks)} peaks"
)

plt.xlabel("DEM column / x index")
plt.ylabel("DEM row / y index")

if len(peaks) > 0:
    plt.legend()

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight",
)

print(f"Saved: {output_path}")

plt.show()
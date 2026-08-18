# Lunar Global Localization

This workspace uses one persisted random sample of at most 50 synchronized
capture sites across every DEM resolution and localization stage. The selected
IDs are stored in `LargeScale_Implement/sim/selected_sites.json`; inspect or
intentionally regenerate them with `LargeScale_Implement/Scripts/20_select_sites.py`.

Experimental global localization for a lunar rover using terrain features
extracted from digital elevation models (DEMs) and local LiDAR scans. This
branch is configured for the Haworth south-pole OmniLRS dataset and detects crater
minima at multiple map resolutions, builds
local elevation maps, estimates feature uncertainty, and evaluates rover poses
with DARCES-style geometric matching.

The actively developed, resolution-aware implementation is in
[`LargeScale_Implement/`](LargeScale_Implement/). Curated DEMs, plots, and
machine-readable experiment results are versioned for reproducibility; raw
captures and intermediate local maps remain outside the repository.

## What is where

```text
.
├── LargeScale_Implement/       Haworth localization pipeline
│   ├── config/                 Shared experiment and resolution profiles
│   ├── DEM/                    DEMs, feature catalogues, QA, and validation
│   ├── Scripts/                Numbered pipeline entry points (00–18)
│   ├── features/               Crater/peak detector
│   ├── matching/               DARCES and RANSAC matching
│   ├── refinement/             MOGA pose refinement
│   ├── local_maps/             Leveled scans, grids, and local features
│   ├── sim/                    Captured point clouds, odometry, and transforms
│   ├── plots/                  Diagnostic figures grouped by resolution
│   ├── results/                Machine-readable localization results
│   ├── tests/                  Configuration, detector, and matcher tests
│   ├── pipeline_config.py      Validated configuration and path projection
│   └── DATA_LAYOUT.md          Resolution-directory conventions
├── requirements.txt           Reproducible Python dependencies
└── .github/                    CI, ownership, and contribution templates
```

Within the main implementation, generated and captured data are grouped by
resolution:

```text
LargeScale_Implement/
├── DEM/<resolution>_px/
├── local_maps/<resolution>_px/
├── plots/<resolution>_px/
├── results/<resolution>_px/
└── sim/<resolution>_px/
```

Resolution labels use `p` as the decimal separator. The configured Haworth
profiles, in increasing cell-size order, are `0p25m`, `0p5m`, `1m`, `2m`,
`5m`, and `10m`.

## Data layout

| Profile | Primary DEM | Shape | Current role |
| --- | --- | ---: | --- |
| `0p25m` | `DEM/0p25m_px/Haworth_sfs_dem_1m_v3_full_with_craters_0p25m.npy` | 1600 × 1600 | Crater-refined experiment DEM |
| `0p5m` | `DEM/0p5m_px/Haworth_sfs_dem_1m_v3_full_with_craters_0p5m.npy` | 800 × 800 | Aligned area-mean product |
| `1m` | `DEM/1m_px/Haworth_sfs_dem_1m_v3_full_with_craters_1m.npy` | 400 × 400 | Configured truth-source reference |
| `2m` | `DEM/2m_px/Haworth_sfs_dem_1m_v3_full_with_craters_2m.npy` | 200 × 200 | Aligned area-mean product |
| `5m` | `DEM/5m_px/Haworth_sfs_dem_1m_v3_full_with_craters_5m.npy` | 80 × 80 | Coarse localization prior |
| `10m` | `DEM/10m_px/Haworth_sfs_dem_1m_v3_full_with_craters_10m.npy` | 40 × 40 | Coarse localization prior |

Curated DEM products, plots, and result files are committed; large binary DEMs
and figures are stored through Git LFS. Raw captures under `sim/` and generated
intermediate maps under `local_maps/` remain external. Their manifests and
metadata should travel with the arrays so raster origin, shape, resolution, and
coordinate conventions remain verifiable.

## Installation

Python 3.10 or newer is recommended. Install the declared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

OpenCV is used to accelerate large circular morphology operations on the
8000 × 8000 DEM. The detector retains a SciPy fallback for environments where
OpenCV is unavailable. The ROS capture utilities under
`LargeScale_Implement/sim/` additionally require a ROS 2 environment with
`rclpy`, message packages, and `sensor_msgs_py`.

## Configuration and resolution selection

[`LargeScale_Implement/config/haworth.yaml`](LargeScale_Implement/config/haworth.yaml)
is the shared configuration authority. It contains the DEM, coordinate, OS1
sensor, detector, and output settings for all six resolutions under
`feature_detection.resolutions`.
The `1m` profile is explicitly labelled as the Haworth truth source; DEM
alignment and global-feature uncertainty use it for every selected profile.

The localization code is unchanged from the Apollo workspaces, but the
detector radii are Haworth-specific because this export's procedural crater
catalogue spans only 0.5--5 m diameter. The 0.25--2 m profiles use a 4 m
physical morphology scale; the 5 m profile uses its smallest representable
5 m scale. The 10 m raster retains fewer than three distinct crater minima in this
400 m crop, so it cannot form a DARCES control triangle.

Every numbered script accepts `--resolution`:

```bash
python3 LargeScale_Implement/Scripts/00_inspect_DEM.py --resolution 0p25m
python3 LargeScale_Implement/Scripts/03_feature_detector.py --resolution 0p5m
python3 LargeScale_Implement/Scripts/09_darces_all_sites.py --resolution 5m
python3 LargeScale_Implement/Scripts/11_ransac_all_sites.py --resolution 5m
python3 LargeScale_Implement/Scripts/12_moga_all_sites.py --resolution 5m
python3 LargeScale_Implement/Scripts/13_pipeline_diagnostics.py --resolution all
python3 LargeScale_Implement/Scripts/14_results_presentation.py
python3 LargeScale_Implement/Scripts/15_validate_dem_refinement.py
python3 LargeScale_Implement/Scripts/16_build_10m_dem.py
```

All scripts default to `5m`. The feature detector also accepts
`--resolution all`. Use `--help` on any script for its remaining controls.
Script 05 reserves `--resolution` for workspace selection and uses
`--grid-resolution` for an optional cell-size override.

Selecting a profile changes the whole workspace, not only the DEM. For example,
`--resolution 0p25m` selects:

```text
DEM/0p25m_px/
local_maps/0p25m_px/
plots/0p25m_px/
results/0p25m_px/
sim/0p25m_px/
```

## Main pipeline

For a complete Haworth run, use the checked parallel and resumable runner from
the workspace root:

```bash
./run_haworth_parallel_pipeline.sh
```

It logs every command under `pipeline_logs/`, validates the shared capture once,
uses six DARCES site processes by default, and runs localization at 0.25, 0.5,
1, 2, and 5 m. The 10 m profile is still
inspected, aligned, gridded, and locally characterized, but global localization
is skipped because its catalogue cannot form a three-feature DARCES control
triangle. To run only one resolution or resume at a later phase:

```bash
RESOLUTIONS="5m" ./run_haworth_parallel_pipeline.sh
START_PHASE=4 DARCES_WORKERS=6 ./run_haworth_parallel_pipeline.sh
```

Compatible completed aggregate results are skipped, and interrupted DARCES
runs resume their atomic per-site checkpoints. `run_haworth_pipeline.sh`
remains available as the single-process reference launcher.

Run commands from the repository root. Replace `5m` with another configured
profile when its required input data are available.

### 1. Inspect or build the DEM

```bash
python3 LargeScale_Implement/Scripts/00_inspect_DEM.py --resolution 5m
python3 LargeScale_Implement/Scripts/01_build_orbital_dem.py --resolution 5m
python3 LargeScale_Implement/Scripts/02_validate_dem_allignment.py --resolution 5m
```

Script 01 performs area-weighted downsampling from the configured 1 m truth reference.
It intentionally refuses to create a finer 0.25 m DEM from that coarser source.
The native 0.25 m DEM should be generated or acquired independently.

### 2. Detect and validate global features

```bash
python3 LargeScale_Implement/Scripts/03_feature_detector.py --resolution 5m
python3 LargeScale_Implement/Scripts/04_global_feature_uncertainty.py --resolution 5m
```

Detector radius, physical distance, flatness tolerance, DEM path, output path,
and preview path are resolution-specific configuration values. Feature
catalogues are compressed NumPy archives containing raster indices, map-frame
XYZ coordinates, feature type, resolution, and detector metadata.

### 3. Process local LiDAR scans

The native synchronized capture is stored under `sim/0p25m_px/`; the remaining
resolution workspaces link to that exact capture so every experiment sees the
same 316 frames. Each workspace exposes:

```text
pointcloud_scans/scan_site_001.npy
odom_scans/odom_site_001.npy
transform_scan/transform_site_001.npz
```

Then run:

```bash
python3 LargeScale_Implement/Scripts/05_local_scan_processing.py --resolution 5m
python3 LargeScale_Implement/Scripts/06_validate_capture.py --resolution 5m
python3 LargeScale_Implement/Scripts/07_local_feature_detection.py --resolution 5m
```

Script 05 transforms scans into gravity-leveled rover-local coordinates and
creates north-up elevation grids. Script 07 detects local crater features and
stores feature covariance estimates used by matching.

Generated local-feature previews are written beneath
`LargeScale_Implement/plots/<resolution>_px/local_features/`.

### 4. Run localization

```bash
# Focused Site 08 run
python3 LargeScale_Implement/Scripts/08_darces_run.py \
  --resolution 5m \
  --trials 100000

# All captured sites
python3 LargeScale_Implement/Scripts/09_darces_all_sites.py \
  --resolution 5m \
  --trials 100000 \
  --workers 6

# Covariance-aware exhaustive consensus refinement
python3 LargeScale_Implement/Scripts/11_ransac_all_sites.py \
  --resolution 5m

# Joint multi-frame landmark/pose refinement
python3 LargeScale_Implement/Scripts/12_moga_all_sites.py \
  --resolution 5m
```

`--workers` evaluates independent sites in separate processes without changing
the per-site seed or numerical DARCES settings. Each completed site is written
atomically beneath `results/<resolution>_px/darces_checkpoints/`; rerunning the
same command resumes compatible checkpoints. Use `--no-resume` for a deliberate
fresh run or `--sites 1 8 15` for a bounded subset. Worker detail logs are kept
beside the checkpoints.

Results are written to `LargeScale_Implement/results/<resolution>_px/` as JSON
and CSV files. RANSAC diagnostic plots are written below
`LargeScale_Implement/plots/<resolution>_px/ransac/`. DARCES uses local/global
feature geometry and a heading prior; absolute odometry position is withheld
from matching and used for evaluation. RANSAC preserves the DARCES estimate
when a consensus cannot be tested or does not produce a supported improvement.
MOGA jointly optimizes feature-derived 2-D poses, headings, and shared global
landmark positions with feature, landmark-prior, and heading-sensor residuals.
Sites without a DARCES/RANSAC pose remain unavailable, and false global aliases
remain visible. Odometry position is used only for post-localization error
evaluation and the truth overlay; it is never used to initialize or constrain
a pose. Its output and
traversal plot are written to `results/<resolution>_px/moga_all_sites.*` and
`plots/<resolution>_px/moga/`.

Generate a stage-by-stage diagnostic report after rerunning any localization
stage with:

```bash
python3 LargeScale_Implement/Scripts/13_pipeline_diagnostics.py --resolution all
```

The report joins every site across DARCES, RANSAC, and MOGA, records each pose
and error, and labels each transition as improved, worsened, unchanged, or not
comparable. Combined CSV/Markdown data and a summary table are placed in
`LargeScale_Implement/results/diagnostics/`; paper-style per-site radial and
component-error plots are placed in each resolution's `plots/.../diagnostics/`
directory.

For a concise presentation graphic instead of the full diagnostics, run
`Scripts/14_results_presentation.py`. It writes a single cross-resolution PNG
to `LargeScale_Implement/plots/results_summary/`.

The Haworth Drive package supplies aligned 0.25, 0.5, 1, 2, 5, and
10 m products. Each coarser array is an exact area-mean reduction of the native
0.25 m array, so no DEM-building step is needed before feature detection.

## Coordinate conventions

The main pipeline uses a north-up Cartesian map frame:

- `x`: east
- `y`: north
- `z`: up
- raster columns increase with `x`
- raster rows increase as `y` decreases
- row 0 is the northern edge
- positive yaw is counter-clockwise

The map origin is the OmniLRS Haworth canonical rover start at global DEM
coordinate `(-930 m, 2129 m)`. The configured local raster spans x = -120 to
280 m and y = -279 to 121 m. Raster-to-map conversion uses sample coordinates
defined in the Drive metadata and shared configuration. Preserve these
conventions when adding DEMs or local grids; silent offsets or row flips are a
common source of incorrect localization results.

## Tests

Run the focused configuration and geometry tests with:

```bash
python3 -m pytest LargeScale_Implement/tests/test_phase1.py -q
```

Run the complete main-implementation suite with:

```bash
python3 -m pytest LargeScale_Implement/tests -q
```

## Important notes

- Commit curated DEMs, plots, and machine-readable results for reproducibility;
  Git LFS handles the configured binary artifacts. Do not commit raw captures
  or generated intermediate local maps—keep those in external dataset storage
  and copy or link them into the ignored resolution workspaces when running
  experiments.
- Generated products should stay in their resolution workspace to avoid mixing
  incompatible cell sizes, feature radii, and coordinate vectors.
- A selected resolution does not manufacture missing captures or local maps;
  scripts report missing inputs when that workspace has not been populated.
- `.npz` feature and grid files carry resolution and coordinate metadata. Prefer
  reading those fields rather than reconstructing coordinates from filenames.
- See [`LargeScale_Implement/DATA_LAYOUT.md`](LargeScale_Implement/DATA_LAYOUT.md)
  for the concise data-layout contract.

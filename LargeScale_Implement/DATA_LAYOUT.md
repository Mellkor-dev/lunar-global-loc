# Resolution experiment data layout

## Shared capture-site selection

`sim/selected_sites.json` is the authoritative experiment-wide sample. When a
dataset contains more than 50 synchronized cloud/odometry/transform captures,
the pipeline selects 50 unique site IDs without replacement using the seed in
the active YAML configuration. Every DEM resolution and local/downstream stage
uses the same IDs. Run `Scripts/20_select_sites.py --resolution 0p25m
--refresh` only when intentionally choosing a new configured random trial.

All generated data is grouped by raster resolution. Directory labels use
`p` as the decimal separator so they remain shell-friendly (`0p5m_px`).

```text
DEM/
  0p25m_px/      High-resolution crater-refined experiment terrain
  0p5m_px/       Aligned area-mean DEM product
  1m_px/         Aligned area-mean DEM product
  2m_px/         Aligned area-mean DEM product
  5m_px/         Coarse DEM product, features, QA, and validation
  10m_px/        Coarse DEM product, features, QA, and validation
local_maps/
  <resolution>/  leveled/, gridded/, and features/
plots/
  <resolution>/  Diagnostic and evaluation figures
results/
  <resolution>/  DARCES/RANSAC/MOGA machine-readable results
sim/
  <resolution>/  odom_scans/, pointcloud_scans/, transform_scan/, validation/
```

`config/apollo11.yaml` is the authority for every input and generated artifact
used by the Apollo 11 experiment. Scripts must load paths through
`pipeline_config.load_pipeline_config()` rather than reconstructing them.
The configured 2 m/px profile is the truth source for DEM validation and
global-feature uncertainty at every selected resolution.

New experiments should use the same layout, with a dedicated configuration
selecting matching directories for every artifact class.

Feature detector inputs and parameters are listed under
`feature_detection.resolutions` in `config/apollo11.yaml`. Run one native
raster or every configured raster with:

```bash
python3 Scripts/03_feature_detector.py --resolution 0p25m
python3 Scripts/03_feature_detector.py --resolution 0p5m
python3 Scripts/03_feature_detector.py --resolution 5m
python3 Scripts/03_feature_detector.py --resolution all
```

Every executable in `Scripts/` accepts the same single-workspace selector:

```bash
python3 Scripts/00_inspect_DEM.py --resolution 0p25m
python3 Scripts/05_local_scan_processing.py --resolution 0p5m
python3 Scripts/07_local_feature_detection.py --resolution 0p5m
python3 Scripts/09_darces_all_sites.py --resolution 5m
python3 Scripts/11_ransac_all_sites.py --resolution 5m
python3 Scripts/12_moga_all_sites.py --resolution 5m
python3 Scripts/13_pipeline_diagnostics.py --resolution all
python3 Scripts/14_results_presentation.py
python3 Scripts/15_validate_dem_refinement.py
python3 Scripts/16_build_10m_dem.py
```

For a selected `<resolution>`, pipeline products are read from and written to
the matching `DEM/<resolution>_px`, `local_maps/<resolution>_px`,
`plots/<resolution>_px`, `results/<resolution>_px`, and
`sim/<resolution>_px` directories. Script 05 additionally accepts
`--grid-resolution` to override the profile's native cell size.

Script 10 preserves the two-panel `traversal/darces_traversal_map.png`
diagnostic and also writes the presentation-oriented grayscale contour map
`traversal/darces_prediction_contour_map.png`. Its companion
`darces_prediction_contour_map_site_mapping.csv` records the display-order to
raw-capture-site mapping used to draw a smooth spatial route without changing
localization identities or results.

The installed 0.5, 1, 2, 5, and 10 m DEMs are aligned non-overlapping area
means of the native 0.25 m terrain. All resolution workspaces link to the same
428 synchronized OS1 scan/odometry/transform triplets stored under
`sim/0p25m_px`.

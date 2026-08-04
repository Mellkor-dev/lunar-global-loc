# Resolution experiment data layout

All generated data is grouped by raster resolution. Directory labels use
`p` as the decimal separator so they remain shell-friendly (`1p5m_px`).

```text
DEM/
  1p5m_px/       Native truth DEM and truth feature catalogue
  5m_px/         Derived orbital prior, global features, QA, and validation
local_maps/
  <resolution>/  leveled/, gridded/, and features/
plots/
  <resolution>/  Diagnostic and evaluation figures
results/
  <resolution>/  DARCES/RANSAC/MOGA machine-readable results
sim/
  <resolution>/  odom_scans/, pointcloud_scans/, transform_scan/, validation/
```

`config/apollo17_5m.yaml` is the authority for every input and generated
artifact used by the 5 m experiment. Scripts must load paths through
`pipeline_config.load_pipeline_config()` rather than reconstructing them.

New experiments should use the same layout, for example `0p25m_px`,
`0p5m_px`, and `1p5m_px`, with a dedicated configuration selecting the
matching directories.

Feature detector inputs and parameters are listed under
`feature_detection.resolutions` in `config/apollo17_5m.yaml`. Run one native
raster or every configured raster with:

```bash
python3 Scripts/03_feature_detector.py --resolution 0p25m
python3 Scripts/03_feature_detector.py --resolution 1p5m
python3 Scripts/03_feature_detector.py --resolution 5m
python3 Scripts/03_feature_detector.py --resolution all
```

Every executable in `Scripts/` accepts the same single-workspace selector:

```bash
python3 Scripts/00_inspect_DEM.py --resolution 0p25m
python3 Scripts/05_local_scan_processing.py --resolution 1p5m
python3 Scripts/07_local_feature_detection.py --resolution 1p5m
python3 Scripts/09_darces_all_sites.py --resolution 5m
```

For a selected `<resolution>`, pipeline products are read from and written to
the matching `DEM/<resolution>_px`, `local_maps/<resolution>_px`,
`plots/<resolution>_px`, `results/<resolution>_px`, and
`sim/<resolution>_px` directories. Script 05 additionally accepts
`--grid-resolution` to override the profile's native cell size.

Script 01 only creates an equal- or lower-resolution DEM from the configured
1.5 m truth reference; it intentionally rejects an attempt to synthesize the
native 0.25 m DEM from a coarser source.

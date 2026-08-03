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

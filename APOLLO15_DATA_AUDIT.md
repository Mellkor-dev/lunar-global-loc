# Apollo 15 dataset audit

## Source package

- Google Drive file ID: `1R3Ew-_VWElsn32VrrAK3ZCGTxaHmxUVO`.
- Download size: 8.2 GiB; 9,095 ZIP members.
- SHA-256: `3801862cccbfbaed579b5575e23e940e08336d33b52d92143126870be2ce757e`.
- The complete source package is installed. No trajectory or frame subset was discarded.

## Capture inventory

- One recorded run: `east_plateau_safe` (`apollo15rille_east_plateau_safe_03`).
- 3,024 complete and synchronized scan/pose/transform triplets.
- Raw data: `LargeScale_Implement/sim/datasets/apollo15_01_global_dem`.
- Each scan is `262144 x 3`, `float32`, in the OS1 LiDAR frame.
- Each pose is `[x, y, z, qx, qy, qz, qw]`, `float64`.
- Each transform NPZ stores `T_global_dem_lidar` plus translation, quaternion,
  frame names, and timestamp.
- Pipeline site `N` maps to original frame index `N - 1`; the original six-digit
  frame IDs and metadata remain unchanged in the raw dataset.
- The active computational sample contains 100 sites drawn without replacement
  from all 3,024 complete sites using NumPy seed 42. The reproducible selection is
  stored in `LargeScale_Implement/sim/selected_sites.json`.

## DEM inventory and coordinates

- Native truth source: `NAC_DTM_APOLLO15_2m`, 2 m/px.
- Installed aligned products: 0.25, 0.5, 1, 2, 5, and 10 m/px.
- Every product covers the same 2 km x 2 km global-DEM region:
  `x=[-1701, 299] m`, `y=[201, 2201] m`.
- Raster convention: columns increase east; rows increase south; elevation is +Z.
- The coarse DEMs are aligned non-overlapping area means of the 0.25 m product.
- Procedural craters are included. Rock instances are not represented in the DEM.
- Canonical starting position: `(-701, 1201) m`.
- Geographic centre: `3.5963833333 deg E, 26.0641722222 deg N`.

## Vertical datum audit

Capture X/Y are already in `global_dem`, but capture Z uses the simulation-stage
datum. A robust surface comparison sampled 189,387 transformed returns across 96
frames against the native 2 m DEM:

- median `z_DEM - z_stage`: `-1899.82784 m`;
- median absolute deviation: `0.04183 m`;
- configured offset: `-1899.828 m`.

The configuration therefore evaluates elevation as
`z_DEM = z_stage - 1899.828 m`. Comparing the stored base poses and LiDAR
transforms recovers a stable base-to-LiDAR translation of
`(-0.150003, -0.000001, 0.413200) m` (maximum variation 0.117 mm and
0.000020 degrees). The operational config uses `(-0.15, 0, 0.4132) m`.

## Baseline experiment configuration

- Detector: terrain peaks, 10 m physical neighbourhood, 0.07 m minimum relief.
- Integer-grid neighbourhoods are 10 m at every configured resolution.
- Site cap: 100; selection seed: 42.
- DARCES defaults match the Apollo 11/17 baseline run script.
- The launcher retains the top eight DARCES hypotheses per site; all other DARCES gates use their baseline defaults.
- Apollo 15 has its own config, results, plots, local maps, and capture paths;
  no Apollo 11, Apollo 14, Apollo 17, or Haworth data is referenced.

## Result publication status

The initial peak/top-8 execution stopped during local feature processing after
partially regenerating DARCES outputs. Existing local plots and result files mix
that partial run with an older experiment, so they are intentionally excluded
from the first `apollo15` branch publication. They should be versioned only after
one uninterrupted six-resolution run regenerates DARCES, RANSAC, MOGA,
diagnostics, and presentation figures together.

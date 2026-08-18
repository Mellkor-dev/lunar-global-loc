# Haworth dataset audit

Source: public Google Drive folder `1V57Au2JKzx_hL-gGx7eqA80rT2JvEzrk`.
The import was audited before it replaced the inherited Apollo 11 artifacts in
this branch.

## Capture sequence

- Sequence: `haworth_02`
- Format: `lunar_global_loc_synchronized_v1`
- Frames: 316, contiguous source indices `000000` through `000315`
- Imported names: `scan_site_01.npy` through `scan_site_316.npy`, with matching
  `odom_site_*.npy` and `transform_site_*.npz`
- LiDAR: Ouster OS1 REV7, 128 channels, 20 Hz, 1024 azimuth samples
- Frames: target `odom`, base `base_link`, LiDAR `os1_lidar`
- Stored pose format: `[x, y, z, qx, qy, qz, qw]`
- Stored transform: homogeneous `T_target_lidar`
- Capture duration: approximately 1203.4 s
- Scan point count: 226,542 to 262,144 points per frame; 189 frames contain
  the full 262,144 returns
- Traversal length reported by capture validation: 318.858 m

Every pose is finite and has a unit quaternion. Every transform is a valid
homogeneous rigid transform, and the recovered base-to-LiDAR translation is
consistent with `[-0.15, 0.0, 0.4131984]` m to better than 0.1 mm. Sampled
point clouds are finite `N x 3` float32 arrays with no zero rows and reach
approximately 170 m.

## DEM products

The six DEMs cover the same 400 m by 400 m Haworth crop and use north-up
row-major rasters. The native source and configured truth profile is 1 m/px.

| Profile | Shape | Source relationship |
| --- | ---: | --- |
| 0.25 m | 1600 x 1600 | OmniLRS refined procedural product |
| 0.5 m | 800 x 800 | aligned area-mean reduction of 0.25 m |
| 1 m | 400 x 400 | native truth-source resolution |
| 2 m | 200 x 200 | aligned area-mean reduction of 0.25 m |
| 5 m | 80 x 80 | aligned area-mean reduction of 0.25 m |
| 10 m | 40 x 40 | aligned area-mean reduction of 0.25 m |

All arrays are finite float32 rasters. Recomputing each area-mean reduction
from the 0.25 m array agrees with the supplied products to within approximately
`6.1e-5` m. The configured map origin is the canonical rover start at global
DEM coordinates `(-930, 2129)` m. All 316 rover poses lie inside the crop.

Procedural craters are included; USD rock instances are not representable in
these height arrays and are not included.

## Resolution validation

Against the native 1 m truth source, the generated alignment reports give:

| Resolution | DEM RMSE | DEM MAE |
| ---: | ---: | ---: |
| 0.25 m | 0.092 m | 0.072 m |
| 0.5 m | 0.062 m | 0.048 m |
| 1 m | 0.000 m | 0.000 m |
| 2 m | 0.114 m | 0.090 m |
| 5 m | 0.437 m | 0.347 m |
| 10 m | 0.968 m | 0.766 m |

## Known limitation

The procedural crater metadata spans 0.5--5 m diameter, so Apollo's inherited
15--20 m morphology scale is not applicable. Haworth-specific physical scales
are configured in `config/haworth.yaml`. Catalogues through 5 m contain enough
features to form a global triangle. At 10 m, this 400 m crop retains only one
distinct crater minimum under defensible detector settings, so 10 m DARCES is
structurally underconstrained. The 10 m data remains installed for DEM and
sampling comparisons, but it should not be reported as a localization failure
equivalent to the finer profiles.

# Apollo 14 dataset audit

- Source Drive object SHA-256: `9db7044547e41e702413d8c158d6619c5aeceab0abd142140af1b5f82786f998`.
- Capture package: 2,673 synchronized OS1-128 frames across 11 recorded trajectories.
- Active trajectory: `curated_forward_approach` (`apollo14_cone_crater_blocks_forward_02`), 254 complete scan/pose/transform triplets.
- Active experiment sample: 100 unique sites selected without replacement using NumPy `default_rng(seed=42)`.
- Capture format: 262,144 × 3 float32 LiDAR points per nominal full scan; pose `[x,y,z,qx,qy,qz,qw]`; transform key `T` is 4×4 `T_global_dem_lidar`.
- Native truth source: `NAC_DTM_APOLLO14_2m`, 2 m/px.
- DEM products: aligned 0.25, 0.5, 1, 2, 5, and 10 m/px rasters over global x `[-300,1700]` m and y `[-6600,-4600]` m.
- Downsampling: aligned non-overlapping area means from the 0.25 m procedural-crater terrain.
- Procedural craters are included; rocks are USD instances and are not represented in the elevation arrays.
- Sensor-frame audit: captures store `T_global_dem_lidar`, so Apollo 14 correctly uses a zero base-to-LiDAR extrinsic (unlike the Husky base-pose datasets).
- Coordinate handling: native global DEM XY is retained end-to-end; absolute position is read only for evaluation, while heading may be used by the matcher gate.
- Vertical datum: robust alignment of 680,537 curated-forward surface returns established `z_DEM = z_stage - 1006.809 m`. The configured offset is applied only to truth-pose Z evaluation; matching geometry, XY, and heading are unchanged.
- Final detector configuration: terrain peaks with a 10 m physical support radius at every resolution (`n=40,20,10,5,2,1`) and `flatness_threshold_m=0.10`. The committed plots/results correspond to this configuration.

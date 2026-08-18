# Apollo 14 Lunar Global Localization

This branch is the isolated Apollo 14 workspace for the LiDAR-to-orbital-DEM localization pipeline. It uses the curated forward OS1-128 trajectory and six aligned DEM resolutions.

## Experiment

- Active trajectory: `curated_forward_approach` / `apollo14_cone_crater_blocks_forward_02`
- Available synchronized sites: 254
- Evaluated sites: random 100 without replacement, seed 42
- DEM resolutions: 0.25, 0.5, 1, 2, 5, and 10 m/px
- Truth source: native Apollo 14 2 m/px DEM
- Matching stages: DARCES, correspondence RANSAC, and independent single-frame 2.5-D MOGA

See [`APOLLO14_DATA_AUDIT.md`](APOLLO14_DATA_AUDIT.md) for source and coordinate validation, and [`LargeScale_Implement/DATA_LAYOUT.md`](LargeScale_Implement/DATA_LAYOUT.md) for the directory contract.

## Run

```bash
cd ~/apollo14
source .venv/bin/activate
DARCES_WORKERS=6 ./run_apollo14_parallel_pipeline.sh
```

The launcher is resumable with `START_PHASE=1..8`, can restrict profiles with `RESOLUTIONS="5m 10m"`, and accepts `FORCE_DARCES=1` to recompute matching checkpoints.

## Tests

```bash
python3 -m pytest LargeScale_Implement/tests -q
```

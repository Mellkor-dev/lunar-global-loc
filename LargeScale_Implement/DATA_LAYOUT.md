# Apollo 14 resolution experiment data layout

The active configuration is `config/apollo14.yaml`. The native 2 m/px Apollo 14 DEM is the truth source; aligned 0.25, 0.5, 1, 2, 5, and 10 m/px products are evaluated independently.

```text
DEM/<resolution>_px/          DEM, global features, QA, uncertainty
local_maps/<resolution>_px/   leveled/, gridded/, features/
plots/<resolution>_px/        diagnostic and presentation figures
results/<resolution>_px/      DARCES, RANSAC, MOGA outputs
sim/0p25m_px/                 authoritative synchronized capture links
sim/<other resolution>_px/    links to the same authoritative captures
sim/selected_sites.json       shared random-100 site sample (seed 42)
sim/datasets/apollo14_01/     curated-forward raw source files and metadata
```

Only `run_id=curated_forward_approach` / `source_sequence=apollo14_cone_crater_blocks_forward_02` is exposed to the pipeline. Its 254 complete frames map sequentially to sites 1–254. Every resolution and every stage consumes the same 100 selected sites.

Run the complete resumable pipeline from the repository root:

```bash
DARCES_WORKERS=6 ./run_apollo14_parallel_pipeline.sh
```

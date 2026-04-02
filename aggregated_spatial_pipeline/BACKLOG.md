# Aggregated Spatial Pipeline Backlog

## Deferred Architecture Cleanup

- Make `street-pattern` fully parquet-first and remove the temporary `GeoJSON` compatibility export used only for `mask`-based reads.
  - Current state: joint pipeline stores shared roads in parquet and creates a one-time adjacent `roads_drive_osmnx_street_pattern.geojson` only because the downstream street-pattern loader still relies on `geopandas.read_file(..., mask=...)`.
  - Why deferred: this is not blocking the main joint pipeline flow and is treated as cleanup rather than product work.

- Finish polishing the per-repository runtime split.
  - Current state: bootstrap now creates dedicated envs for `blocksnet`, `connectpt`, `floor-predictor`, `segregation-by-design-experiments`, and submodule `iduedu-fork`; `run_joint` already calls the heavy sibling stages through their own runtimes.
  - Remaining work: add the same runtime-resolver discipline to every ad hoc script/notebook entrypoint and tighten Windows path handling for non-bash launchers.
  - Why deferred: the main production pipeline is already isolated enough to avoid cross-repo dependency conflicts; the rest is cleanup and ergonomics.

## Deferred PT Bridge Follow-Up

- Extend the new `iduedu -> ConnectPT` stop bridge from ingestion to downstream route optimization / regeneration workflows.
  - Current state: `run_joint` now builds `iduedu` first, `connectpt` reuses modality stops derived from intermodal graph nodes, aggregates them with `connectpt` logic, and stores `raw_stop_id -> aggregated_stop_id` mapping artifacts per modality.
  - Remaining work: use the saved mapping to project optimized or regenerated routes back onto the richer `iduedu` stop layer when route editing/generation is introduced.
  - Why deferred: the ingestion bridge is now in place; the next step depends on route-optimization product decisions rather than collection correctness.

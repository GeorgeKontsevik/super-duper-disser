# Aggregated Spatial Pipeline Backlog

## Deferred Architecture Cleanup

- Make `street-pattern` fully parquet-first and remove the temporary `GeoJSON` compatibility export used only for `mask`-based reads.
  - Current state: joint pipeline stores shared roads in parquet and creates a one-time adjacent `roads_drive_osmnx_street_pattern.geojson` only because the downstream street-pattern loader still relies on `geopandas.read_file(..., mask=...)`.
  - Why deferred: this is not blocking the main joint pipeline flow and is treated as cleanup rather than product work.

- Split runtimes per repository/toolchain instead of keeping most execution in the shared root `.venv`.
  - Desired state: separate isolated environments for at least `aggregated_spatial_pipeline`, `blocksnet`, `connectpt`, `segregation-by-design-experiments`, and intermodal `iduedu 1.2.1`.
  - Current state: only intermodal graph building is isolated in `.venv-iduedu121`; the rest still shares the main root environment.
  - Why deferred: current setup works for the active pipeline flow, but runtime isolation is still an architectural cleanup task.

## Deferred PT Bridge Follow-Up

- Extend the new `iduedu -> ConnectPT` stop bridge from ingestion to downstream route optimization / regeneration workflows.
  - Current state: `run_joint` now builds `iduedu` first, `connectpt` reuses modality stops derived from intermodal graph nodes, aggregates them with `connectpt` logic, and stores `raw_stop_id -> aggregated_stop_id` mapping artifacts per modality.
  - Remaining work: use the saved mapping to project optimized or regenerated routes back onto the richer `iduedu` stop layer when route editing/generation is introduced.
  - Why deferred: the ingestion bridge is now in place; the next step depends on route-optimization product decisions rather than collection correctness.

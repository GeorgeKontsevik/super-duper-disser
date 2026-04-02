# Aggregated Spatial Pipeline Backlog

## Deferred Architecture Cleanup

- Make `street-pattern` fully parquet-first and remove the temporary `GeoJSON` compatibility export used only for `mask`-based reads.
  - Current state: joint pipeline stores shared roads in parquet and creates a one-time adjacent `roads_drive_osmnx_street_pattern.geojson` only because the downstream street-pattern loader still relies on `geopandas.read_file(..., mask=...)`.
  - Why deferred: this is not blocking the main joint pipeline flow and is treated as cleanup rather than product work.

- Split runtimes per repository/toolchain instead of keeping most execution in the shared root `.venv`.
  - Desired state: separate isolated environments for at least `aggregated_spatial_pipeline`, `blocksnet`, `connectpt`, `segregation-by-design-experiments`, and intermodal `iduedu 1.2.1`.
  - Current state: only intermodal graph building is isolated in `.venv-iduedu121`; the rest still shares the main root environment.
  - Why deferred: current setup works for the active pipeline flow, but runtime isolation is still an architectural cleanup task.

## Deferred PT Bridge Integration

- Integrate the tested `iduedu -> simplified ConnectPT stop` bridge into the main pipeline after prototype validation.
  - Current prototype state: `run_pt_bridge_iduedu_connectpt_prototype.py` extracts raw `iduedu` PT nodes, simplifies them with ConnectPT stop aggregation, saves `raw_iduedu_node_id -> simplified_stop_id` mapping, and renders QA previews on a small city.
  - Integration direction: keep `iduedu` as the intermodal accessibility graph, keep `connectpt` as the simplified stop graph for route generation/optimization, and connect them via explicit mapping artifacts rather than forcing both into one graph representation.
  - Why deferred: we first want to validate the simplification rules, mapping quality, and QA visuals on small-city cases before wiring this into the production pipeline.

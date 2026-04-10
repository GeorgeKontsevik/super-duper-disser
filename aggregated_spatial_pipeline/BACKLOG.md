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

## Deferred Cross-City Coverage Ops

- Add a fast coverage precheck utility for city eligibility before cross-city experiments.
  - Current state: city coverage screening exists inside `service_accessibility_street_pattern` run flow, but ad hoc standalone checks with full geometry `clip/union` over all cities are too slow for quick triage.
  - Remaining work: provide a lightweight cached/precomputed CLI report (or manifest-based summary) that returns per-city coverage and exclusion candidates in seconds.
  - Why deferred: not blocking experiment correctness, but needed for faster operational triage and rerun planning.

## Deferred Solver Evolution

- Rework placement genetic stage to a more controllable optimization backend (`Optuna` or another Python GA framework) with support for custom inner evaluation method.
  - Current state: genetic placement is implemented with in-repo custom GA and can be hard to tune/extend when injecting custom scoring or inner solving logic.
  - Remaining work: evaluate migration path (keep solver contract stable), prototype Optuna-based search loop, and compare against a maintained GA framework alternative before choosing final backend.
  - Why deferred: current flow is usable for experiments, but solver R&D and framework migration should be done explicitly as a separate task.

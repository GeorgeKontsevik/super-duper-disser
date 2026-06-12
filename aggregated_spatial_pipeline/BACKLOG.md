# Aggregated Spatial Pipeline Backlog

## Deferred Polyclinic PT-Improvement Substitution Experiment

- Test whether targeted PT improvements can reduce the number of additional `polyclinic` facilities needed to reach target coverage.
  - Core thesis: `street pattern` is the diagnostic layer for locating where transport improvement can substitute for part of service placement.
  - Placement-only target90 is only the baseline counterfactual, not the main experiment.
  - Current state: city-level pattern diagnostics now compare demand, existing service capacity, PT stops, PT-route length, coverage, unmet demand, first-mile failures, and PT-segment failures by street-pattern context.
  - Remaining work: connect these diagnostics to solver-selected and near-selected candidate blocks, then test intervention scenarios around those blocks.
  - Candidate scenarios: speed up selected links/corridors, add a route, replace one route with another, or improve stop connectivity around candidate blocks.
  - Main outcome: compare `additional_polyclinics_needed` before and after PT intervention under the same target coverage.
  - Important constraint: do not force route diversity with fallback generation; if route generation returns duplicate or weak alternatives, store and report that result honestly.
  - Why deferred: this requires an explicit experimental design on top of the current descriptive pattern-system layer and usable target90 candidate/placement outputs.

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

## Deferred Flood-Depth Input

- Add a real flood-depth source or hydraulic depth model for equatorial road fragility experiments.
  - Current state: Copernicus GFM flood extent can be fetched and used as a binary weekly flood proxy, but the project has no water-depth raster in meters for the March-May exact runs.
  - Remaining work: choose and document a valid depth source/model, ingest it under `equatorial/data/raw/flood_depth/<ISO3>/`, and verify resulting `flood_depth_week_*_max_m` columns against the produced overlay.
  - Why deferred: flood extent is not a substitute for flood depth, so filling depth columns from GFM would be analytically misleading.

## Deferred ConnectPT Dataset Expansion

- Expand the real-morph ConnectPT route-generator training dataset beyond the current 6 gravity-usable cities.
  - Current state: `connectpt_dataset_prep` owns the gravity-only dataset build, structure analysis, and synthetic-demand comparison; the current dataset has 480 gravity samples and skips 4 cities with missing blocks.
  - Remaining work: follow [connectpt_dataset_prep/TODO.md](/Users/gk/Code/super-duper-disser/connectpt_dataset_prep/TODO.md) to add more eligible cities, preserve the no-fallback rule, and rebalance morphology coverage before clean retraining.
  - Why deferred: the current dataset is usable for a clean baseline, but more city coverage is needed before treating training conclusions as stable.

## Deferred Solver Evolution

- Rework placement genetic stage to a more controllable optimization backend (`Optuna` or another Python GA framework) with support for custom inner evaluation method.
  - Current state: genetic placement is implemented with in-repo custom GA and can be hard to tune/extend when injecting custom scoring or inner solving logic.
  - Remaining work: evaluate migration path (keep solver contract stable), prototype Optuna-based search loop, and compare against a maintained GA framework alternative before choosing final backend.
  - Why deferred: current flow is usable for experiments, but solver R&D and framework migration should be done explicitly as a separate task.

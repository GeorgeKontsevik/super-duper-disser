# City-Level Polyclinic Coverage Plan

Цель:

- подготовить code path для проверки city-level question:
  - как `street pattern` и простые `PT descriptors` связаны с `PT + walk` coverage до `polyclinic`
  - и сколько `additional polyclinics` нужно, чтобы поднять coverage до target

## Scope

Phase 1:

- только `19 active19 cities`
- только `polyclinic`
- только `PT + walk`
- главный improvement mode = `additional polyclinics`

Phase 2:

- optional `additional PT routes`
- optional extension to `new17/new5/old23`

## Planned Outputs

Новый sub-track должен писать:

1. `city_level_inputs.csv`
- one row per city
- baseline coverage
- street-pattern shares
- PT descriptors
- solver baseline summary

2. `city_level_additional_services.csv`
- one row per city per target coverage
- additional polyclinics needed

3. `city_level_local_pattern_summary.csv`
- one row per city x local pattern
- local demand / local unmet / local difficulty

4. plots
- coverage vs morphology
- additional services needed vs morphology
- within-city pattern plots

## File Strategy

Likely code location:

- extend:
  - `/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/run_experiments.py`

Or, if the file starts getting too wide:

- create:
  - `/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/run_city_level_experiments.py`

Tests:

- extend:
  - `/Users/gk/Code/super-duper-disser/tests/test_polyclinic_access_components.py`

## Execution Order

### Step 1. Build city-level baseline table

Use:

- existing polyclinic diagnostics
- `service_access_diagnostics`
- `solver_inputs/polyclinic/summary.json`
- street-pattern summaries

Need:

- one row per city
- baseline coverage
- demand total
- accessibility gap
- capacity gap

### Step 2. Build city-level street-pattern mix

Use:

- `street_pattern/<city>/predicted_cells.csv`

Need:

- city-wide shares of:
  - `Irregular Grid`
  - `Loops & Lollipops`
  - `Regular Grid`
  - `Warped Parallel`
  - `Broken Grid`
  - `Sparse`

### Step 3. Build simple PT-descriptor table

Use the simplest stable sources first.

Candidates:

- `services_to_pt_top3/_run_report.tsv`
- `residential_to_pt_top3/_run_report.tsv`
- `intermodal_graph_iduedu/graph.pkl`

Need:

- route count
- stop count
- maybe graph node / edge counts

Do not block phase 1 on corridor-morphology PT overlays.

### Step 4. Define service-addition algorithm

For each city:

- start from baseline `blocks_solver.parquet`
- compute current coverage target gap
- iteratively add synthetic `polyclinic` capacity at selected blocks
- stop when target coverage is reached

Targets:

- `0.70`
- `0.80`
- `0.90`

The exact placement heuristic must be fixed explicitly before coding:

- candidate blocks
- service capacity assumption
- whether new services can be placed only in unmet blocks or anywhere
- whether `Loops & Lollipops` restrictions are respected

### Step 5. Build within-city local-pattern difficulty layer

Need:

- local unmet share by `street_pattern_top1_class`
- local additional-services burden by class

### Step 6. Add plots and summaries

Need:

- cross-city comparisons
- local-pattern breakdowns

### Step 7. Extension layer

Only after phase 1:

- `additional PT routes needed`
- `new17/new5/old23` baseline-only extension

## Open Decisions Before Coding

These must be fixed before service-addition code is written:

1. `coverage` should be taken from:
- house-level diagnostics
- or block-level solver provision

2. one new `polyclinic` means:
- fixed capacity equal to current city mean
- or another explicit capacity rule

3. candidate placement space:
- any living block
- or only blocks with unmet demand
- or only blocks allowed by street-pattern rules

4. improvement scoring:
- minimize number of services only
- or number of services plus morphology-aware penalty

## Default Decision For Now

Unless revised, use:

- coverage from existing house-level diagnostics
- service-addition difficulty from `pipeline_2/solver_inputs/polyclinic`
- fixed capacity rule matching current solver setup as closely as possible
- active19 only

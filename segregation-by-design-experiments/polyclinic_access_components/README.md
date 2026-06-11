# Polyclinic Access Components

Этот `README` нужен как handoff для следующего агента, который не знает ни этот подпроект, ни структуру большого репозитория.

Если нужно быстро войти в контекст, читать в таком порядке:

1. этот файл
2. [RESEARCH_SCHEME.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/RESEARCH_SCHEME.md)
3. [CITY_LEVEL_RESEARCH_SCHEME.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/CITY_LEVEL_RESEARCH_SCHEME.md)
4. [CITY_LEVEL_VERIFICATION_STATUS.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/CITY_LEVEL_VERIFICATION_STATUS.md)

## What This Subproject Is

Это отдельный research track внутри диссер-проекта про `PT + walk` доступность до `polyclinic`.

Здесь сейчас есть две связанные, но разные ветки:

1. `house-to-service failure decomposition`
- для каждого дома разбирается, какой именно компонент маршрута ломает доступность
- потом это связывается со `street pattern`

2. `city-level accessibility / improvement difficulty`
- на уровне городов проверяется, как городская сеть связана с текущим `coverage`
- и сколько новых `polyclinic` нужно, чтобы дотянуться до target, сейчас главный target это `0.9`

Важно не смешивать эти ветки.
Первая ветка про типы провала на уровне маршрутов/домов.
Вторая ветка про cross-city слой и placement difficulty.

## Repo Context

Рабочий репозиторий:

- `/Users/gk/Code/super-duper-disser`

Папка подпроекта:

- `/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components`

Этот подпроект не автономный.
Он опирается на уже собранные city bundles из `aggregated_spatial_pipeline`.

Основные upstream данные лежат в:

- `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs`
- `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_new17_access_20260610/joint_inputs_merged`
- `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_old23_access_20260609/joint_inputs_merged`

## Main Research Logic

### Branch A. Failure Decomposition

Для дома и его ближайшей `polyclinic` маршрут разбирается на части:

- direct walk
- home -> stop
- PT segment
- stop -> service
- walks sum
- total route

Текущая рабочая схема типов провала:

- `ok_walk`
- `ok_pt_only`
- `failed_access_gt_threshold`
- `failed_in_vehicle_gt_threshold`
- `failed_egress_gt_threshold`
- `failed_both_walks_gt_threshold`
- `failed_multi_component_gt_threshold`
- `failed_total_gt_threshold_no_single_component_gt_threshold`

По этой ветке главный narrative сейчас такой:

- `first-mile` и `PT-path` failures важнее, чем `last-mile`
- `PT path morphology` надо смотреть отдельно от morphology around home/service
- `street pattern` использовать диагностически, а не как normatively “good/bad”

Подробная схема:

- [RESEARCH_SCHEME.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/RESEARCH_SCHEME.md)

### Branch B. City-Level Track

Главный вопрос:

- как то, как устроена сеть города, связано с текущей `PT + walk` доступностью `polyclinic`
- и насколько трудно эту доступность улучшить

Сейчас operationalization difficulty такая:

- `additional_polyclinics_needed_to_0_9`

Важно:

- это означает `reach or exceed 0.9`
- это не `full coverage`
- это не “ровно 0.900”

Подробная схема:

- [CITY_LEVEL_RESEARCH_SCHEME.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/CITY_LEVEL_RESEARCH_SCHEME.md)

## Files You Actually Need

### Core Code

- [run_experiments.py](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/run_experiments.py)
  - main script for route/component decomposition and street-pattern plots
  - entry point: `main()`
  - important functions:
    - `_add_component_flags`
    - `_build_requested_summary`
    - `_build_single_component_pattern_summaries`
    - `_build_pt_path_pattern_raw`
    - `_render_combined_single_component_patterns_png`
    - `_render_four_panel_pattern_png`

- [city_level.py](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/city_level.py)
  - main script for city registry, baseline coverage, target-0.9 placement, placement-lift
  - entry point: `main()`
  - important functions:
    - `build_default_city_registry`
    - `build_city_level_research_dataset`
    - `scale_unmet_demand_to_target_provision`
    - `run_targeted_placement_for_city`
    - `build_city_level_target90_dataset`
    - `transfer_street_pattern_cells_to_blocks`
    - `build_city_target90_pattern_lift_rows`
    - `build_target90_pattern_lift_detail`
    - `render_target90_overview_png`
    - `render_target90_pattern_lift_png`

### Tests

- [tests/test_polyclinic_access_components.py](/Users/gk/Code/super-duper-disser/tests/test_polyclinic_access_components.py)
  - `PolyclinicAccessComponentsTests`
  - `PolyclinicCityLevelRegistryTests`

Run:

```bash
cd /Users/gk/Code/super-duper-disser
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python tests/test_polyclinic_access_components.py
```

Latest checked status when this README was written:

- `40 tests`
- `OK`

### Research Docs

- [RESEARCH_SCHEME.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/RESEARCH_SCHEME.md)
  - route-level interpretation and hypotheses

- [CITY_LEVEL_RESEARCH_SCHEME.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/CITY_LEVEL_RESEARCH_SCHEME.md)
  - city-level sample, data sources, target-0.9 logic, placement-lift logic

- [CITY_LEVEL_VERIFICATION_STATUS.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/CITY_LEVEL_VERIFICATION_STATUS.md)
  - what was actually verified in artifacts and what is still unstable

- [CITY_LEVEL_IMPLEMENTATION_PLAN.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/CITY_LEVEL_IMPLEMENTATION_PLAN.md)
  - old plan/history, useful mostly as context

## Input Data Map

### Route-Level / House-Level Inputs

The route/component branch uses already prepared diagnostics and city bundles.

Important sources:

- house-to-service diagnostics parquet
- per-city `intermodal_graph_iduedu/graph.pkl`
- per-city `street_pattern/<city>/predicted_cells.geojson`

The script already knows where to look for them through the project constants.
Do not invent a parallel data path unless the upstream bundle layout really changed.

### City-Level Inputs

For each city, `city_level.py` uses:

- `pipeline_2/solver_inputs/polyclinic/summary.json`
- `pipeline_2/solver_inputs/polyclinic/blocks_solver.parquet`
- `pipeline_2/solver_inputs/polyclinic/adj_matrix_time_min.parquet`
- `derived_layers/blocks_clipped.parquet`
- `analysis_territory/buffer.parquet`
- `street_pattern/<city>/predicted_cells.geojson`

The city registry is built from a deduped sample of `36` cities:

- `19` from active19
- `16` from new17
- `1` from old23

Priority:

- `active19 > new17 > old23 > new5`

## Important Implementation Details

### 1. Target 0.9 Is Scaled Before Placement

`city_level.py` does not send the full original unmet demand into the exact placement solver.

Instead it:

- reads `demand_without + demand_left`
- scales that unmet mass down to what corresponds to `0.9` total provision
- then runs exact placement on this reduced target

Function:

- `scale_unmet_demand_to_target_provision`

Interpretation:

- `additional_polyclinics_needed_to_0_9` means minimum count to reach or exceed `0.9`
- not minimum count for full accessibility coverage

### 2. Placement Mode Is Explicit

Current placement setup is:

- exact / non-genetic by default
- `prefer_existing=False`
- `allow_existing_expansion=False`
- `capacity_mode="fixed_mean"`
- no fallback route for “making it work”

This was done intentionally.

### 3. Placement Lift Uses Block-Level Pattern, Not City-Wide Share

The crude idea

- `share_pattern_in_city -> additional_polyclinics_needed`

was considered too weak, because it confounds “pattern is common in city” with “solver prefers this context”.

The stronger current layer is:

- baseline A: `share_pattern_in_city_blocks`
- baseline B: `share_pattern_in_solver_candidate_blocks`
- selected: `share_pattern_among_selected_new_polyclinics`

Main metric:

- `placement_lift_vs_candidates = selected_share - candidate_share`

This is the right metric if the question is:

- which street-pattern contexts solver over-selects relative to their baseline availability among candidate blocks

### 4. Street Pattern Must Be Transferred To Blocks

Do not assume `blocks_solver.parquet` always already carries street pattern.
In many cities it does not.

Current correct path:

- `street_pattern/<city>/predicted_cells.geojson`
- transfer to `derived_layers/blocks_clipped.parquet`
- then join to solver block ids

Current join logic:

- solver block key is `name`
- it matches the string form of the `blocks_clipped` index

Function:

- `transfer_street_pattern_cells_to_blocks`

## Outputs And Where To Look

### Branch A. Route / Component Outputs

Main output directory:

- `/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs`

Key files:

- [polyclinic_home_access_components.parquet](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/polyclinic_home_access_components.parquet)
  - main row-level derived table

- [polyclinic_requested_summary_overall.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/polyclinic_requested_summary_overall.csv)
- [polyclinic_requested_summary_overall.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/polyclinic_requested_summary_overall.png)
- [polyclinic_requested_summary_by_city.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/polyclinic_requested_summary_by_city.csv)
- [polyclinic_requested_summary_by_city.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/polyclinic_requested_summary_by_city.png)

Single-component street-pattern outputs:

- `/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/single_component_patterns`

Most useful files there:

- [single_component_patterns_combined.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/single_component_patterns/single_component_patterns_combined.png)
- [pt_segment_not_ok.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/single_component_patterns/pt_segment_not_ok.png)
- [all_polyclinic_pt_paths.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/single_component_patterns/all_polyclinic_pt_paths.png)
- [all_polyclinic_pt_paths_vs_all_city_street_pattern_by_city.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/single_component_patterns/all_polyclinic_pt_paths_vs_all_city_street_pattern_by_city.png)

### Branch B. City-Level Outputs

City-level output directory:

- `/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level`

Baseline cross-city outputs:

- [city_registry.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_registry.csv)
- [city_registry_verified.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_registry_verified.csv)
- [city_baseline_coverage.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_baseline_coverage.csv)
- [city_research_question_dataset.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_research_question_dataset.csv)
- [city_research_question_association_summary.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_research_question_association_summary.csv)

Ready-subset target-0.9 outputs:

- [city_target90_partial_ready_dataset.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_target90_partial_ready_dataset.csv)
- [city_target90_partial_ready_association_summary.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_target90_partial_ready_association_summary.csv)
- [city_target90_partial_ready_overview.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_target90_partial_ready_overview.png)

Ready-subset placement-lift outputs:

- [city_target90_pattern_lift_detail_partial_ready.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_target90_pattern_lift_detail_partial_ready.csv)
- [city_target90_pattern_lift_overall_partial_ready.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_target90_pattern_lift_overall_partial_ready.csv)
- [city_target90_pattern_lift_partial_ready.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_target90_pattern_lift_partial_ready.png)

## Current Verified Status

### What Was Actually Checked

- unit tests for both route-level and city-level helpers
- `target90` partial-ready CSV outputs
- `target90` partial-ready PNG previews
- placement-lift PNG preview
- sums in placement-lift shares:
  - per-city `city_share = 1`
  - per-city `candidate_share = 1`
  - per-city `selected_share = 1`
  - overall sums also equal `1`

### Ready Cities For Target 0.9

At the time of writing, confirmed completed `summary_after.json` exist for `10` cities:

- `bergen_norway`
- `bologna_italy`
- `brno_czechia`
- `krakow_poland`
- `marseille_france`
- `turku_finland`
- `hai_duong_h_i_d_ng_vietnam`
- `jaynagar_bih_r_india`
- `kananga_kasa_central_congo_kinshasa`
- `montes_claros_minas_gerais_brazil`

Anything phrased as “partial ready” uses only this subset.

### Current Placement-Lift Signal On Ready Subset

Relative to candidate baseline, current aggregated signal is:

- over-selected: `Loops & Lollipops` `+0.061`
- over-selected: `Warped Parallel` `+0.034`
- under-selected: `Regular Grid` `-0.067`
- under-selected: `Irregular Grid` `-0.050`

This is not yet a final full-sample claim.
But it is already the correct metric type.

## How To Re-Run

### Route / Component Branch

```bash
cd /Users/gk/Code/super-duper-disser
./.venv/bin/python segregation-by-design-experiments/polyclinic_access_components/run_experiments.py
```

### City-Level Branch

```bash
cd /Users/gk/Code/super-duper-disser
MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python segregation-by-design-experiments/polyclinic_access_components/city_level.py
```

Note:

- long exact-placement runs are expensive
- do not trust `script finished successfully` alone
- inspect generated CSV/PNG artifacts directly

## Practical Warnings For The Next Agent

1. Do not claim the full `36`-city target-0.9 layer is done unless you checked actual `summary_after.json` artifacts.
2. Do not rely on city-wide `share_pattern_in_city` alone when the question is about solver selection.
3. Do not assume `street pattern` already exists in solver block tables.
4. Do not interpret `accessibility_gap_share` as a stable difficulty metric without checking whether it matches actual poor coverage.
5. Do not confuse `partial ready` files with full-sample outputs.
6. Do not revert to fallback placement logic just to force a result.

## If You Need To Continue The Research

Most likely next steps:

1. finish `target 0.9` exact placement for the remaining cities in the 36-city registry
2. rerun the city-level output writer so full-sample `target90` outputs replace the current partial-ready layer
3. rerun the placement-lift layer on the full completed sample
4. only then make stronger claims about which street-pattern contexts are over-selected by solver
5. if the next question is specifically about `Take 3`, add visual corridor inspection for selected PT-path cases rather than only aggregate bars

## Minimal Orientation Summary

If you remember only five things, remember these:

1. `run_experiments.py` is the route/component branch.
2. `city_level.py` is the cross-city / placement branch.
3. `target 0.9` currently means `reach or exceed 0.9` after scaling unmet demand.
4. the strong new metric is `placement_lift_vs_candidates`, not crude city-wide pattern share.
5. always inspect the actual PNG/CSV outputs in `outputs/` before making claims.

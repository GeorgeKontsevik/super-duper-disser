# Route Generation + Service Placement Tracker

## Question

Do we need to consider route generation and service placement together, or are route generation / connectivity improvements enough on their own?

Working wording for the section:

> Joint route generation and service placement: when does transport connectivity solve the access gap, and when does the remaining gap require new or reassigned service capacity?

## Scope

In scope:

- Polyclinic route-strategy canvases with before/after access diagnostics.
- Route-generation outputs and selected route-improvement options.
- Service-placement outputs under the same experiments.
- Comparisons between:
  - placement only,
  - general connectivity / route generation,
  - existing-service expansion,
  - candidate-service placement,
  - candidate-or-existing service placement.
- Optional secondary SPB case only if it uses the richer road + generated route + service comparison.

Out of scope:

- Equatorial experiments.
- Arctic experiments.
- Street-pattern as a separate topic.

If `street_pattern_*` fields appear in summaries, treat them as technical leftovers unless the current run explicitly used them as designed analytical inputs.

## Primary Figure Candidate

Current selected image:

- Bologna full route-strategy canvas:
  `/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/overnight_route_strategy_batch_20260613_routes3_finalcanvas/_final_full_canvases/bologna_italy.png`

This figure is for the combined question: generated routes plus service placement, not just accessibility mapping.

## Primary Dataset

Batch root:

`/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/overnight_route_strategy_batch_20260613_routes3_finalcanvas`

Final canvases currently identified:

- `bologna_italy.png`
- `huainan_anhui_china.png`
- `krakow_poland.png`
- `marseille_france.png`
- `temuco_araucan_a_chile.png`

Likely narrative anchors:

- Bologna: current selected figure.
- Marseille: earlier selected figure with similar full canvas structure.

## What To Extract Next

For each candidate city, inspect:

- `route_count_selection_manifest.json`
- `gap_diagnostics/strategy_gap_summary.csv`
- `gap_diagnostics/strategy_route_stage_summary.csv`
- `combined_block_access_capacity_triage.csv`
- `recomputed_access_components/*summary.csv`

Fields to collect into one comparison table:

- city
- selected / best strategy
- requested routes
- actual routes
- new service count
- selected existing service count
- capacity added total
- demand target total
- `demand_without_after_total`
- `demand_left_after_total`
- `provision_total_after`
- `route_unserved_pct`
- route cost
- after-routes access gap

## Interpretation Discipline

Do not claim that joint route + service planning is necessary until the route-only / connectivity-only outcomes are compared against service-placement outcomes.

The expected distinction to test:

- Route generation can improve reachability and reduce transport-related access failures.
- Service placement or expansion is still needed when the remaining gap is driven by missing capacity, missing nearby services, or residual demand after route improvement.
- The useful result is the comparison, not just the existence of a visually better route.

## Narrative Order

Use this order in the write-up and experiments.

1. Road-first comparison, close to earlier work:
   - Say that the first setup is still somewhat abstract.
   - Then compare against a more conventional intervention: add the planned road, recompute access, and generate routes under the basic connectivity objective.
   - The route-generation goal here is pragmatic: improve connectivity so that fewer additional services are needed.
   - This is the "as in pre-existing work" baseline: infrastructure/network improvement first, service need recalculated afterward.

2. Joint route generation and optimal placement:
   - Then ask the actual next question: how should routes be generated when service placement is also optimized?
   - Run the SPB experiments in the same logic as the Marseille/Bologna/European route-strategy canvases.
   - Compare route/connectivity-only strategies against service-placement-aware strategies.
   - This is the section that answers whether route generation and service placement should be considered together.

## Optional SPB Secondary Case

Use SPB only as a secondary illustration if it supports road + route + service comparison.

SPB experiment root:

`/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_oktyabrskaya_large`

Useful SPB images:

- `experiment_report/visual_intervention_layers.png`
- `experiment_report/full_intervention_grid_matrix.png`
- `experiment_report/full_intervention_grid_services.png`
- `experiment_report/road_landuse_services_sequence.png`
- `planned_road_mixed_bundle/preview_png/all_together/pt_route_generator_bus.png`
- `planned_road_mixed_bundle/preview_png/all_together/accessibility_mean_time_delta_map_bus_generated.png`

Do not use the standalone accessibility map as the main evidence for this section.

SPB road-intervention requirement:

- The baseline must represent the state before the Telmana / Dalnevostochny-to-Oktyabrskaya road connection exists.
- The planned road must enter only as a modeled intervention.
- The planned-road figure should therefore show the added road as the intervention layer, not a road already present in the baseline graph.
- Current produced baseline needs re-checking before use: near the planned alignment, the baseline OSM drive layer contains road edges named `улица Тельмана`. If those edges represent the same connection, the SPB road experiment must be regenerated with that corridor removed from baseline and added back only in the planned-road scenario.

SPB land-use intervention context:

- The changed quarter is the development block between Telmana Street and Novoselov Street in the project/masterplan context.
- Ignore the current produced `quarter_index=229` run for this purpose: it is from the old scenario and is not the target quarter.
- Seed point for the new target quarter:
  - user-provided coordinate: `59.886550, 30.457582`
  - interpret as `lat, lon`
  - GIS/WGS84 coordinate: `lon=30.457582, lat=59.886550`
- In the current block layer this point falls inside block index `29`; use this only as the candidate target block for the new run, not as evidence from the old scenario.
- The new run must use this newly specified quarter / polygon from the Telmana-Novoselov development block.
- The scenario should be regenerated after the new quarter is fixed; do not reuse the old `mixed_use_industrial_229_polyclinic` metrics as evidence.
- Project-scale sanity target for the new quarter:
  - planned services: `1` polyclinic
  - planned population: about `12000` residents
  - use these project values directly for this case; do not use the SM imputer as the population source for the Telmana-Novoselov block.
- Working value for the new scenario narrative:
  - population: `12000` residents, from the project/masterplan
  - service: `1` planned polyclinic, from the project/masterplan
- Treat Telmana / Novoselov street names as project-context labels unless verified in the baseline OSM road layer.

SPB Telmana 2x2 experiment (superseded diagnostic):

- Output root:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620`
- Status:
  - keep only as discarded diagnostic;
  - do not use as evidence for the final Telmana case because the road intervention was too broad / not the selected short connector touching block `29`.
- Setup summary:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_project_setup_summary.json`
- Comparison summary:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_project_2x2_summary.json`
- Scenarios:
  - without Telmana road: `without_telmana_project_bundle`
  - with Telmana/planned road: `with_telmana_project_bundle`
- Shared project inputs:
  - target block: `29`
  - project population: `12000`
  - project service: `1` polyclinic, capacity `800`
- Result before optimal placement:
  - without Telmana: `demand_without=2500`, `provision_total=0.7748`
  - with Telmana: `demand_without=2054`, `provision_total=0.8149`
  - Telmana effect: `-446` unmet/out-of-reach demand, `+0.0402` provision
- Result after exact optimal placement:
  - without Telmana: `new_count=14`, `demand_without_after=122`, `provision_after=0.9890`
  - with Telmana: `new_count=14`, `demand_without_after=122`, `provision_after=0.9890`
  - interpretation: Telmana helps before placement, but under this exact placement setup it does not reduce the number of additional polyclinics.

SPB Telmana route-strategy experiment (superseded diagnostic):

- Output root:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_route_strategy_search_20260620`
- Status:
  - keep only as discarded diagnostic;
  - do not use as final evidence because it belongs to the superseded broad-road setup.
- Consolidated summary:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_route_strategy_summary.json`
- Quick route-edge previews:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_route_strategy_search_20260620/_route_edge_previews`
- Best SPB route preview:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_route_strategy_search_20260620/_route_edge_previews/with_telmana_project_bundle__existing_service__routes_3.png`
- Best SPB placement preview:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_route_strategy_search_20260620/with_telmana_project_bundle/existing_service/routes_3/preview_png/lp_polyclinic_placement_changes.png`
- Setup:
  - service: `polyclinic`
  - generated routes requested: `3`
  - new-service capacity: `800`
  - street-pattern route targeting: disabled / excluded from interpretation
  - route generator objective weights: `demand_time=0.3`, `route_time=0.3`, `median_connectivity=0.3`, `street_pattern=0.0`
- Road-first / basic-connectivity result:
  - without Telmana + `general_connectivity`: `new_count=12`, access gap after routes `2205`, after-placement residual `47`, provision after placement `0.9958`
  - with Telmana + `general_connectivity`: `new_count=12`, access gap after routes `1694`, after-placement residual `47`, provision after placement `0.9958`
  - interpretation: the road improves route-stage access gap for the basic connectivity route, but in this setup it still does not reduce the number of new polyclinics beyond the route-only baseline.
- With Telmana, route-target comparison:
  - baseline/no routes: `new_count=14`, target demand `2054`, after-placement residual `122`, provision `0.9890`
  - `general_connectivity`: `new_count=12`, target demand `1694`, after-placement residual `47`, provision `0.9958`
  - `candidate_service`: `new_count=13`, target demand `1665`, after-placement residual `47`, provision `0.9958`
  - `candidate_or_existing_service`: `new_count=13`, target demand `2013`, after-placement residual `122`, provision `0.9890`
  - `existing_service`: `new_count=9`, target demand `1160`, after-placement residual `47`, provision `0.9958`
  - `placement_assignment`: route generation skipped because the installed placement root had no assignment links, so target OD was zero; do not use it as generated-route evidence for this run.
- Current SPB route-strategy interpretation:
  - In this local case, simply adding Telmana and optimizing placement still leaves `14` new polyclinics.
  - Adding 3 basic-connectivity routes reduces the need to `12` new polyclinics.
  - Service-aware generation toward existing services reduces it further to `9` new polyclinics.
  - This supports the section ordering: road/connectivity first, then show that the route target/service-planning coupling changes the result.

SPB Telmana simplified one-route experiment (superseded diagnostic):

- Output root:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_one_route_existing_service_20260620`
- Status:
  - keep only as discarded diagnostic;
  - do not use as final evidence because it belongs to the superseded broad-road setup.
- Scenario:
  - bundle: `with_telmana_project_bundle`
  - route target: `existing_service`
  - requested routes: `1`
  - actual routes: `1`
  - route length by generator stops: `[10]`
  - mapped route length: `6.70 km`
  - mapped route time: `20.1 min`
  - mapped road/intermodal segments: `9`
  - generated intermodal graph edges: `18` directed bus edges (`9` segments in both directions)
  - route-to-intermodal mapping distance: median `0.0 m`, max `62.6 m`
- Result:
  - before route generation: access gap `2054`, placement `new_count=14`
  - after one generated route: access gap `1522`
  - after placement on the one-route matrix: `new_count=12`, residual `demand_without_after=122`, provision `0.9890`
- Mapped-road preview:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_one_route_existing_service_20260620/with_telmana_project_bundle/existing_service/routes_1/mapped_route_preview/with_telmana_existing_service_one_route_graph_mapped.png`

SPB Telmana short-connector clean 4x2 experiment:

- Output root:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_connector_clean_4x2_20260620`
- Selected connector:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_clean_4x2_20260620/visual_scenario_maps_square_telmana_corrected/telmana_connector_touching_block29_selected.parquet`
- Connector meaning:
  - only the short Telmana-side segment touching the block `29`;
  - not the full Telmana street and not the right-side segment;
  - connects the road sides immediately around the redevelopment quarter.
- Setup summary:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_connector_clean_4x2_20260620/telmana_connector_clean_4x2_setup_summary.json`
- Result table:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_connector_clean_4x2_20260620/connector_clean_4x2_summary_indexed.md`
- Square map matrix:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_connector_clean_4x2_20260620/visual_scenario_maps_square_connector/connector_clean_4x2_square_maps_matrix.png`
- Revised non-repeating visual matrix:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_connector_clean_4x2_20260620/visual_scenario_maps_square_connector_v2/telmana_connector_clean_visual_matrix_v2.png`
- Revised visual matrix renderer:
  `/Users/gk/Code/super-duper-disser/scripts/render_telmana_connector_visual_matrix_v2.py`
- Scenarios:
  - `01_current`: current placement, no project demand/service, no connector.
  - `02_current_plus_project`: current + project demand/service, no connector.
  - `03_current_plus_project_plus_connector`: current + project demand/service + selected short connector.
  - `04_current_plus_connector`: current + selected short connector.
- Project patch:
  - block `29`;
  - population `12000` from project/masterplan;
  - one planned polyclinic with capacity `800`;
  - do not use the SM imputer population for this case.
- No-route results:
  - `01_current`: access gap `2609`, new polyclinics `14`, residual after placement `82`.
  - `02_current_plus_project`: access gap `2500`, new polyclinics `14`, residual `122`.
  - `03_current_plus_project_plus_connector`: access gap `2500`, new polyclinics `14`, residual `122`.
  - `04_current_plus_connector`: access gap `2609`, new polyclinics `14`, residual `82`.
- One-route results:
  - `01_current + route`: access gap after route `2113`, new polyclinics `12`, route `5.88 km / 17.7 min`.
  - `02_current_plus_project + route`: access gap after route `2058`, new polyclinics `13`, route `6.89 km / 20.7 min`.
  - `03_current_plus_project_plus_connector + route`: access gap after route `2058`, new polyclinics `13`, route `6.89 km / 20.7 min`.
  - `04_current_plus_connector + route`: access gap after route `2113`, new polyclinics `12`, route `5.88 km / 17.7 min`.
- Current interpretation:
  - The selected short connector has no measurable effect in the current access/placement setup: paired scenarios with and without it are identical.
  - One generated route changes the placement requirement:
    - without project demand/service: `14 -> 12` new polyclinics;
    - with project demand/service: `14 -> 13` new polyclinics.
  - The clean evidence for this section is therefore the route/service comparison, while the short road connector itself is neutral under the current graph and access metric.
- Visual layout note:
  - Existing PT routes and existing polyclinics are shown only once in the context row.
  - Scenario rows show placement-only new services, one generated route, route + placement new services, and final unmet demand.
  - All unmet-demand panels use one common scale (`0..276` unmet demand per block), including the baseline context panel.
- Interpretation:
  - This is the cleaner SPB illustration if the section needs only one added route.
  - It is less strong than the 3-route strategy result (`new_count=9`), but visually and methodologically simpler.
  - The route is not just a generated stop sequence: it is appended into the intermodal graph as bus edges on mapped road/intermodal segments.

SPB Telmana one-route scenario comparison:

- Output root:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_one_route_scenario_comparison_20260620`
- Status:
  superseded by the clean 4x2 matrix below; do not use this older comparison as the current SPB result because it mixed route-target strategies with scenario definitions.
- Clean comparison table:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_one_route_scenario_comparison_20260620/telmana_one_route_comparable_summary_no_diagnostics.csv`
- All rows, including diagnostics:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_one_route_scenario_comparison_20260620/telmana_one_route_comparable_summary.csv`
- Mapped-route previews:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_project_20260620/telmana_one_route_scenario_comparison_20260620/_mapped_route_previews`
- Best comparable one-route scenario:
  - with Telmana + `existing_service`
  - `actual_routes=1`
  - mapped length `6.70 km`
  - mapped time `20.1 min`

SPB Telmana clean 4x2 one-route matrix:

- Output root:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_clean_4x2_20260620`
- Setup summary:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_clean_4x2_20260620/clean_4x2_setup_summary.json`
- Result table:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_clean_4x2_20260620/clean_4x2_summary.csv`
- Markdown summary:
  `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_spb_telmana_clean_4x2_20260620/clean_4x2_summary.md`
- Clean scenario definitions:
  - `01_current`: current placement, no project demand/service, no Telmana road.
  - `02_current_plus_project`: current placement + project demand/service, no Telmana road.
  - `03_current_plus_project_plus_road`: current placement + project demand/service + Telmana road.
  - `04_current_plus_road`: current placement + Telmana road, no project demand/service.
- Same four scenarios are repeated with exactly one generated bus route:
  - route target strategy: `existing_service`
  - requested routes: `1`
  - actual routes: `1`
  - street-pattern targeting: disabled and excluded from interpretation.
- Setup verification:
  - no-road scenarios remove Telmana from baseline road layers and intermodal graph: `19` road-layer rows and `32` graph edges removed.
  - road scenarios contain Telmana/planned-road graph edges: `32` Telmana graph-edge rows and `134` synthetic planned-road edge rows.
  - project block is block `29`; population is set directly to `12000`; project polyclinic capacity is `800`.
- Results:
  - current, no route: access gap `2609`, new services `14`.
  - current, +1 route: access gap `2113`, new services `12`, route `5.88 km / 17.7 min`.
  - project, no road, no route: access gap `2500`, new services `14`.
  - project, no road, +1 route: access gap `2058`, new services `13`, route `6.89 km / 20.7 min`.
  - project + Telmana road, no route: access gap `2054`, new services `14`.
  - project + Telmana road, +1 route: access gap `1522`, new services `12`, route `6.70 km / 20.1 min`.
  - Telmana road only, no route: access gap `2174`, new services `13`.
  - Telmana road only, +1 route: access gap `1607`, new services `10`, route `6.70 km / 20.1 min`.
- Interpretation:
  - The clean comparison is scenario-first; route-target strategy is fixed, so rows are not mixed with `candidate_service`.
  - Telmana road helps before placement by lowering the access gap.
  - One generated route helps further by lowering the post-route demand target and reducing required new services in all four scenario pairs.
  - mapped road/intermodal segments `9`
  - after-routes access gap `1522`
  - placement result `new_count=12`, capacity added `9600`, residual `demand_without_after=122`, provision `0.9890`
- Best comparable without-road scenario:
  - without Telmana + `existing_service`
  - `actual_routes=1`
  - mapped length `6.89 km`
  - mapped time `20.7 min`
  - after-routes access gap `2058`
  - placement result `new_count=13`, capacity added `10400`, residual `demand_without_after=122`, provision `0.9890`
- Route-only/basic-connectivity one-route result:
  - without Telmana + `general_connectivity`: `new_count=14`, access gap after route `2500`
  - with Telmana + `general_connectivity`: `new_count=14`, access gap after route `2054`
  - interpretation: one basic-connectivity route does not reduce service count; service-aware targeting is needed even in the one-route simplified story.
- Diagnostic note:
  - `placement_assignment` now generates routes after copying assignment links, but in this runner it starts from after-placement residual links and is not comparable for service-count minimization.
  - It is retained only as a diagnostic row; do not use its `new_count=0` as an analytical result.

## Checks

- [ ] Confirm Bologna final canvas path and visual content.
- [ ] Extract Bologna manifest values.
- [ ] Compare Bologna route/connectivity strategies against service-placement strategies.
- [ ] Repeat the same extraction for Marseille.
- [ ] Decide whether Bologna or Marseille is the main figure.
- [ ] Decide whether SPB is useful as a secondary example.
- [ ] For SPB, verify that Telmana / planned-road corridor is absent from baseline and present only after the road intervention.
- [ ] For SPB, verify block index `29` from seed point `30.457582,59.886550` as the new target quarter before rerunning the land-use scenario.
- [x] Use project-defined `~12000` residents and `1` polyclinic for the Telmana-Novoselov block.
- [x] Run SPB Telmana 2x2 experiment: without/with Telmana road x without/with exact optimal placement.
- [x] Run SPB road-first route experiment with `general_connectivity`, 3 generated routes, without/with Telmana.
- [x] Run SPB Telmana route-target strategy comparison with 3 generated routes.
- [x] Run simplified SPB Telmana one-route experiment and verify route is mapped into the intermodal graph.
- [x] Run simplified one-route scenario comparison across road states and route-target strategies.
- [ ] Verify final selected PNGs directly before using them.
- [ ] Confirm no equatorial, arctic, or separate street-pattern material enters this section.

## Current Working Claim

Open hypothesis:

Route generation and connectivity improvements may reduce access failures, but the section needs to show whether they close the full service gap or whether joint consideration with service placement is required.

This remains a hypothesis until the manifests, summaries, and final PNGs are checked city by city.

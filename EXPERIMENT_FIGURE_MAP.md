# Experiment Figure Map

This file tracks where the dissertation experiment figures live in the repository and, when confirmed, which script or output artifact produced them.

Source PDF:

- [КонцевикГИ_КИРПИЧ.pdf](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/КонцевикГИ_КИРПИЧ.pdf)

Extracted experiment pages:

- [tmp/pdfs/kirpich_extracted/experiment_pages](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages)

Key extracted page snapshots:

- [exp_equatorial_weekly_dynamics_p080.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_equatorial_weekly_dynamics_p080.png)
- [exp_equatorial_correlations_p081.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_equatorial_correlations_p081.png)
- [exp_arctic_multilayer_may_aug_p084.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_arctic_multilayer_may_aug_p084.png)
- [exp_city_access_diagnostics_20cities_p090.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_city_access_diagnostics_20cities_p090.png)
- [exp_intermodal_input_layers_p094.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_intermodal_input_layers_p094.png)
- [exp_genetic_algorithm_results_p095.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_genetic_algorithm_results_p095.png)
- [exp_telmana_input_layers_p097.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_telmana_input_layers_p097.png)
- [exp_telmana_scenarios_many_services_p098.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_telmana_scenarios_many_services_p098.png)
- [exp_telmana_scenarios_fewer_services_p099.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_telmana_scenarios_fewer_services_p099.png)
- [exp_route_generation_strategies_p101.png](/Users/gk/Code/super-duper-disser/tmp/pdfs/kirpich_extracted/experiment_pages/exp_route_generation_strategies_p101.png)

## Thesis Figure Includes

Main chapter include locations:

- [chapter4.tex](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/Dissertation/chapter4.tex:66)
- [chapter4_optimal_placement_local.tex](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/Dissertation/chapter4_optimal_placement_local.tex:58)

## Figure Map

### 1. Equatorial weekly accessibility + precipitation grid

Thesis image files:

- [lbr_weekly_accessibility_impact_heatmap.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/lbr_weekly_accessibility_impact_heatmap.png)
- [lbr_precip_grid_week_2024_08_19.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/lbr_precip_grid_week_2024_08_19.png)

Confirmed source paths:

- Heatmap renderer: [render_weekly_astar_accessibility_heatmaps.py](/Users/gk/Code/super-duper-disser/equatorial/scripts/render_weekly_astar_accessibility_heatmaps.py:473)
- Heatmap artifact: [LBR_weekly_accessibility_impact_heatmap.png](/Users/gk/Code/super-duper-disser/equatorial/outputs/astar_accessibility_weekly/cluster_connected_allclusters_10small_3large_3ports_3airports_delta_minutes_heatmaps/LBR_weekly_accessibility_impact_heatmap.png)
- Precip renderer: [render_lbr_precip_grid_figure.py](/Users/gk/Code/super-duper-disser/equatorial/scripts/render_lbr_precip_grid_figure.py:28)

Confidence:

- `lbr_precip_grid_week_2024_08_19.png`: exact
- `lbr_weekly_accessibility_impact_heatmap.png`: exact output match

### 2. Equatorial temporal rain burden + Spearman figure

Thesis image files:

- [temporal_rain_burden_top12_ru.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/temporal_rain_burden_top12_ru.png)
- [crop_spearman_transposed_ru.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/crop_spearman_transposed_ru.png)

Likely source scripts:

- [render_paper_country_mechanism_full.py](/Users/gk/Code/super-duper-disser/equatorial/scripts/render_paper_country_mechanism_full.py:965)
- [render_paper_country_mechanism_full.py](/Users/gk/Code/super-duper-disser/equatorial/scripts/render_paper_country_mechanism_full.py:986)
- [render_paper_rainfall_mechanism_experiment.py](/Users/gk/Code/super-duper-disser/equatorial/scripts/render_paper_rainfall_mechanism_experiment.py:450)

Notes:

- The country-mechanism pipeline clearly computes crop burden, temporal burden, and Spearman-style summaries.
- Exact filename handoff into the thesis copies has not been fully pinned down yet.

Confidence:

- probable

### 3. Arctic multilayer May / Aug

Thesis image files:

- [yanao_kras_multilayer_may.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/arctic/yanao_kras_multilayer_may.png)
- [yanao_kras_multilayer_aug.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/arctic/yanao_kras_multilayer_aug.png)

Confirmed repo artifacts:

- [multilayer_network_yanao_kras_May.png](/Users/gk/Code/super-duper-disser/arctic_access/plots/multilayer/multilayer_network_yanao_kras_May.png)
- [multilayer_network_yanao_kras_Aug.png](/Users/gk/Code/super-duper-disser/arctic_access/plots/multilayer/multilayer_network_yanao_kras_Aug.png)

Confidence:

- exact output family, thesis appears to be copied from these arctic outputs

### 4. 20-city service accessibility diagnostics

Thesis image file:

- [polyclinics_access_diagnostics_ru.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/access_diagnostics/polyclinics_access_diagnostics_ru.png)

Confirmed source paths:

- Renderer: [render_service_access_diagnostics_service_sheets.py](/Users/gk/Code/super-duper-disser/scripts/render_service_access_diagnostics_service_sheets.py:15)
- Default output dir config: [render_service_access_diagnostics_service_sheets.py](/Users/gk/Code/super-duper-disser/scripts/render_service_access_diagnostics_service_sheets.py:256)
- Output artifact: [02_polikliniki_access_diagnostics_ru.png](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/maps_by_service_ru/02_polikliniki_access_diagnostics_ru.png)

Confidence:

- exact output match

### 5. Tikhevich GA fitness + accessibility delta distribution

Thesis image files:

- [tikhevich_ga_fitness.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/optimal_local/tikhevich_ga_fitness.png)
- [tikhevich_accessibility_delta_distribution.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/optimal_local/tikhevich_accessibility_delta_distribution.png)

Confirmed generation mechanism:

- Genetic convergence rendering is implemented in [run_pipeline2_prepare_solver_inputs.py](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/pipeline/run_pipeline2_prepare_solver_inputs.py:1544)
- Example output artifact: [genetic_fitness_convergence_polyclinic.png](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/bergen_norway/preview_png/all_together/genetic_fitness_convergence_polyclinic.png)

Notes:

- The exact Tikhevich experiment output folder for the thesis copies is not yet pinned down.
- The pipeline mechanism for the fitness plot is confirmed.
- The delta-distribution thesis file is present, but its exact generating artifact has not yet been traced.

Confidence:

- partial

### 6. Telmana connector context + scenario matrices

Thesis image files:

- [telmana_connector_context_row.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/optimal_local/telmana_connector_context_row.png)
- [telmana_connector_scenarios_worse_new_services.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/optimal_local/telmana_connector_scenarios_worse_new_services.png)
- [telmana_connector_scenarios_better_new_services.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/optimal_local/telmana_connector_scenarios_better_new_services.png)

Confirmed source script:

- [render_telmana_connector_visual_matrix_v2.py](/Users/gk/Code/super-duper-disser/scripts/render_telmana_connector_visual_matrix_v2.py:614)
- [render_telmana_connector_visual_matrix_v2.py](/Users/gk/Code/super-duper-disser/scripts/render_telmana_connector_visual_matrix_v2.py:636)

Confidence:

- exact

### 7. Route generation strategies `7x3`

Thesis image file:

- [polyclinic_route_strategy_7x3_ru.png](/Users/gk/Code/super-duper-disser/itmo-phd-thesis-template-en/images/ch4/optimal_local/polyclinic_route_strategy_7x3_ru.png)

Confirmed source family:

- Tracker: [ROUTE_SERVICE_JOINT_TASK_TRACKER.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/ROUTE_SERVICE_JOINT_TASK_TRACKER.md:39)
- Batch collector: [collect_overnight_city_figures.py](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/collect_overnight_city_figures.py:1)
- Batch root: [overnight_route_strategy_batch_20260613_routes3_finalcanvas](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/overnight_route_strategy_batch_20260613_routes3_finalcanvas)
- Final full canvases: [_final_full_canvases](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/overnight_route_strategy_batch_20260613_routes3_finalcanvas/_final_full_canvases)
- Regenerated 5x3 grid: [polyclinic_5x3_round_maps_ru_gray_biglegend.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/overnight_route_strategy_batch_20260613_routes3_finalcanvas/_scenario_grid_5x3_regenerated_20260620/polyclinic_5x3_round_maps_ru_gray_biglegend.png)

Notes:

- The thesis `7x3` file is almost certainly assembled from this route-strategy batch family.
- The exact one-step script that copied the final thesis PNG has not yet been identified.

Confidence:

- strong probable

## Status Summary

Exact or near-exact source mapping confirmed for:

- equatorial Liberia precip figure
- equatorial Liberia weekly accessibility heatmap
- arctic multilayer outputs
- city accessibility diagnostics
- Telmana connector matrices
- route-strategy batch family

Still worth tracing more precisely later:

- `temporal_rain_burden_top12_ru.png`
- `crop_spearman_transposed_ru.png`
- `tikhevich_ga_fitness.png` thesis copy root
- `tikhevich_accessibility_delta_distribution.png` generating output
- exact thesis assembly step for `polyclinic_route_strategy_7x3_ru.png`

## Reference Context For Redesign

Visual references previously identified for later redesign work:

- [Tyn Studio urban heat intelligence](/Users/gk/Code/super-duper-disser)  
  External reference discussed in chat: <https://tynstudio.com/blog/urban-heat-intelligence.html>
- MIT Senseable overall project visuals  
  External reference discussed in chat: <https://senseable.mit.edu>
- MIT Senseable Cooling Path  
  External reference discussed in chat: <https://senseable.mit.edu/cooling-path/>

When redesign starts, the strongest candidates for rework in those directions are:

- Telmana connector matrices
- route-strategy canvases
- arctic multilayer figure
- equatorial weekly accessibility surface

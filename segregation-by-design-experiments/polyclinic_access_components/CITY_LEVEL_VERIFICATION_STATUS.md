# City-Level Verification Status

Этот файл фиксирует, что уже реально проверено кодом для city-level `polyclinic` question.

## Built Layer

Собран воспроизводимый city-level слой:

- sample: `36` городов
- outcome 1: `coverage`
- auxiliary outcome: `accessibility_gap_share` из `pipeline_2/solver_inputs/polyclinic/summary.json`
- explanatory layer A: city-wide `street pattern` mix
- explanatory layer B: простые city-level PT descriptors

Дополнительно собран ready-subset placement-lift слой для `target 0.9`:

- baseline city blocks: `share_pattern_in_city_blocks`
- baseline candidate blocks: `share_pattern_in_solver_candidate_blocks`
- selected blocks: `share_pattern_among_selected_new_polyclinics`
- main metric: `placement_lift_vs_candidates = selected_share - candidate_share`

Для этого слоя `street pattern` не берется из solver parquet напрямую.
Он переносится так:

- `street_pattern/<city>/predicted_cells.geojson`
- `-> derived_layers/blocks_clipped.parquet`
- `-> join to solver blocks by block index/name`

Для placement target `0.9` operationalization сейчас такая:

- solver не идет по полному исходному `unmet demand`
- перед запуском exact placement исходный `demand_without + demand_left` масштабируется до уровня,
  который соответствует `0.9` total provision
- после этого solver решает уже эту уменьшенную target-задачу

Важно:

- из-за дискретных новых объектов и fixed capacity итоговый `provision_after` обычно получается `>= 0.9`, а не ровно `0.900`
- поэтому текущая метрика читается как
  `how many new polyclinics are needed to reach or exceed 0.9`
- это не то же самое, что
  `how many are needed for full coverage`
- и не то же самое, что
  `minimum continuous increase to exactly 0.9`

Артефакты:

- [city_research_question_dataset.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_research_question_dataset.csv)
- [city_research_question_association_summary.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_research_question_association_summary.csv)
- [city_target90_pattern_lift_detail_partial_ready.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_target90_pattern_lift_detail_partial_ready.csv)
- [city_target90_pattern_lift_overall_partial_ready.csv](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_target90_pattern_lift_overall_partial_ready.csv)
- [city_target90_pattern_lift_partial_ready.png](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/city_level/city_target90_pattern_lift_partial_ready.png)

## Ready Subset Status

По фактическим `summary_after.json` сейчас подтвержден ready subset из `10` городов:

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

Именно на нем сейчас честно посчитан placement-lift слой.

## What Looks Stable In Placement Lift

На aggregated ready subset пока виден такой pattern относительно candidate baseline:

- over-selected: `Loops & Lollipops` (`+0.061`)
- over-selected: `Warped Parallel` (`+0.034`)
- under-selected: `Regular Grid` (`-0.067`)
- under-selected: `Irregular Grid` (`-0.050`)

Это еще не final claim для full sample.
Но это уже правильный тип метрики:

- не `какой паттерн просто чаще встречается в городе`
- а `какой паттерн solver выбирает сверх его фоновой доли среди candidate blocks`

## What Looks Stable

Для `coverage` уже есть внятный cross-city signal.

Самые сильные monotonic associations по модулю:

- `share_warped_parallel`: `rho = -0.736`
- `share_irregular_grid`: `rho = 0.696`
- `pt_route_count`: `rho = 0.681`
- `pt_stop_count`: `rho = 0.643`
- `pt_modality_count`: `rho = 0.495`

Это пока не causal claim.
Это только проверка, что у `research question` есть observable signal в текущем sample.

## What Is Not Stable Yet

`accessibility_gap_share` пока нельзя считать хорошей operational metric для части question про
`how easy or hard it is to improve`.

Причина:

- у части городов очень низкий `coverage`, но почти нулевой `accessibility_gap_share`
- значит текущий `solver summary gap` не совпадает с тем, что нам нужно как improvement-difficulty outcome

Следовательно:

- baseline question `network form -> current accessibility` уже можно тестировать
- difficulty question `network form -> how hard to improve` еще требует отдельной явной метрики

## Next Required Outcome

Следующий основной outcome надо считать явно:

- `additional_polyclinics_needed_to_reach_target_coverage`

Suggested targets:

- `0.70`
- `0.80`
- `0.90`

Только после этого можно честно тестировать вторую половину question.

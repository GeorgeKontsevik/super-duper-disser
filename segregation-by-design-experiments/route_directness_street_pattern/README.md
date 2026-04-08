# Route Directness Vs Street Pattern

Этот подпроект берет **существующие PT-маршруты** из `intermodal_graph_iduedu/graph.pkl` и смотрит, как их геометрия и route-time burden соотносятся со `street pattern` вдоль corridor.

Ключевая идея тут специально привязана к `connectpt`:

- `route_time_min` в этом эксперименте — прямой аналог `total_route_time` / `route_cost` компоненты в генераторе;
- `branch_time_min` и `branch_ratio` — это route-level penalties за ветвистость и непрямоту, которые логично можно держать как отдельный explanatory слой или добавлять в cost-function дальше.

То есть это не “произвольный detour ради detour”, а проверка, связана ли городская морфология с тем, насколько дорогой и непрямой уже существующая PT-сеть получается.

## Где лежит

- код: [route_directness_street_pattern](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/route_directness_street_pattern)
- раннер: [run_experiments.py](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/route_directness_street_pattern/run_experiments.py)
- outputs: [outputs](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/route_directness_street_pattern/outputs)

## Как запускать

```bash
cd /Users/gk/Code/super-duper-disser

PYTHONPATH=/Users/gk/Code/super-duper-disser \
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/.cache/mpl-route-directness-street-pattern \
.venv/bin/python segregation-by-design-experiments/route_directness_street_pattern/run_experiments.py \
  --cities warsaw_poland berlin_germany
```

Если нужен весь доступный набор городов:

```bash
cd /Users/gk/Code/super-duper-disser

PYTHONPATH=/Users/gk/Code/super-duper-disser \
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/.cache/mpl-route-directness-street-pattern \
.venv/bin/python segregation-by-design-experiments/route_directness_street_pattern/run_experiments.py \
  --cities all
```

## Что сохраняется

Для каждого города:

- `prepared/route_variants.parquet`
- `stats/route_pattern_correlations.csv`
- `stats/summary.json`
- `preview_png/01_route_branch_ratio_by_corridor_class.png`
- `preview_png/02_route_time_per_terminal_km_by_corridor_class.png`
- `preview_png/03_route_variants_branch_ratio_map.png`

Cross-city:

- `outputs/_cross_city/route_variants_all_cities.parquet`
- `outputs/_cross_city/stats/pooled_model_summary.csv`
- `outputs/_cross_city/stats/pooled_model_coefficients.csv`
- `outputs/_cross_city/preview_png/10_pooled_route_time_coefficients.png`
- `outputs/_cross_city/preview_png/11_pooled_branch_ratio_coefficients.png`
- `outputs/_cross_city/preview_png/12_branch_ratio_by_city.png`

# Pipeline 2: Joint Service Placement + ConnectPT

Этот README фиксирует рабочий запуск пайплайна для одного города в режиме:

- доразмещение сервисов через `solver_flp`;
- построение transport-target для `ConnectPT` из результата placement;
- генерация одного или нескольких маршрутов;
- встройка новых маршрутов обратно в intermodal graph без удаления существующей PT-сети;
- пересчет accessibility;
- повторный пересчет service provision после маршрутов.

Документ описывает именно текущую реализованную механику в коде, а не желаемую абстрактную схему.

Отдельный note с архитектурными соображениями и ограничениями лежит здесь:

- [README_PIPELINE2_JOINT_SERVICE_PT_NOTES.md](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/README_PIPELINE2_JOINT_SERVICE_PT_NOTES.md)

## Что Считает Пайплайн

Текущая связка работает так:

1. `run_pipeline2_prepare_solver_inputs` готовит `pipeline_2/solver_inputs/<service>/blocks_solver.parquet`.
2. Если включен placement, тот же скрипт запускает `solver_flp` и пишет:
   - `pipeline_2/placement_exact/.../blocks_solver_after.parquet`
   - или `pipeline_2/placement_exact_genetic/.../blocks_solver_after.parquet`
   - плюс `assignment_links_after.csv`, где видно, какой клиентский block к какому facility block привязан.
3. `run_pipeline2_accessibility_first --use-placement-outputs` берет after-placement blocks и остаточный `demand_without_after`.
4. Из `assignment_links_after.csv` строится service-aware stop-to-stop OD:
   - origin = stop ближайший к client block;
   - destination = stop ближайший к facility block;
   - weight = остаточный unmet accessibility demand этого клиента.
5. `ConnectPT` получает эту OD через `--od-matrix-path`.
6. Сгенерированный маршрут встраивается в `intermodal_graph_iduedu/graph.pkl` как `existing + generated`.
   Старые ребра выбранной модальности по умолчанию не удаляются.
7. На merged graph пересчитывается accessibility matrix.
8. Provision пересчитывается уже на after-placement capacities, то есть с учетом `optimized_capacity_total`.

## Что Важно Понимать

- Это уже рабочий мост `solver -> ConnectPT -> graph -> accessibility -> provision`.
- `ConnectPT` получает не "генетически улучшенную матрицу", а явную service-target OD.
- Внутренняя cost function `ConnectPT` при этом остается своей собственной.
- Поэтому пайплайн функционально связанный, но не является строгим joint optimizer в математическом смысле.

## Где Лежит Город

Ожидается папка города вида:

```bash
aggregated_spatial_pipeline/outputs/<some_batch>/joint_inputs/<city_slug>
```

Пример:

```bash
CITY="/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/old/joint_inputs/vienna_austria"
```

## Полный Запуск На Один Город

Из корня репозитория:

```bash
cd /Users/gk/Code/super-duper-disser
CITY="/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/old/joint_inputs/vienna_austria"
```

### Шаг 1. Подготовка Solver Inputs И Placement

Пример exact placement с генетикой и разрешением расширять существующие сервисы:

```bash
PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
  --joint-input-dir "$CITY" \
  --services hospital polyclinic school kindergarten \
  --placement-exact \
  --placement-genetic \
  --placement-allow-existing-expansion \
  --placement-prefer-existing \
  --placement-capacity-mode fixed_mean
```

Если генетика не нужна:

```bash
PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
  --joint-input-dir "$CITY" \
  --services hospital polyclinic school kindergarten \
  --placement-exact \
  --placement-allow-existing-expansion \
  --placement-prefer-existing \
  --placement-capacity-mode fixed_mean
```

Что появляется после шага:

- baseline:
  - `pipeline_2/solver_inputs/<service>/blocks_solver.parquet`
- after-placement:
  - `pipeline_2/placement_exact_genetic/<service>/blocks_solver_after.parquet`
  - `pipeline_2/placement_exact_genetic/<service>/assignment_links_after.csv`
  - `pipeline_2/placement_exact_genetic/<service>/summary_after.json`

Если генетика выключена, вместо `placement_exact_genetic` будет `placement_exact`.

### Шаг 2. Route Generation + Accessibility Recompute + Provision Recompute

Полный шаг поверх after-placement outputs:

```bash
PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
  --joint-input-dir "$CITY" \
  --services hospital polyclinic school kindergarten \
  --use-placement-outputs \
  --modality bus \
  --n-routes 1
```

Что делает этот вызов:

- загружает `blocks_solver_after.parquet`;
- строит `service_target_od/<modality>_service_target_od.csv`;
- запускает `ConnectPT` на этой OD;
- встраивает generated routes в intermodal graph без замены существующей модальности;
- пересчитывает accessibility;
- пересчитывает provision после маршрутов;
- сохраняет общий manifest.

### Шаг 2a. Явно Указать Placement Root

Если надо зафиксировать, откуда именно брать after-placement outputs:

```bash
PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
  --joint-input-dir "$CITY" \
  --services polyclinic \
  --use-placement-outputs \
  --placement-root-name placement_exact_genetic \
  --modality bus \
  --n-routes 1
```

Допустимые типичные значения:

- `placement_exact_genetic`
- `placement_exact`

Если `--placement-root-name` не задан, скрипт пытается взять:

1. `placement_exact_genetic`
2. `placement_exact`

## Запуск На Несколько Городов

Ниже команды, которыми удобно гонять те же шаги по набору городов.

### Batch По Активным 19 Городам

Если города лежат в:

```bash
/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/<city_slug>
```

то placement + route step можно крутить так:

```bash
cd /Users/gk/Code/super-duper-disser

for CITY in /Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/*; do
  echo "=== $(basename "$CITY") ==="

  PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
    --joint-input-dir "$CITY" \
    --services hospital polyclinic school kindergarten \
    --placement-exact \
    --placement-genetic \
    --placement-allow-existing-expansion \
    --placement-prefer-existing \
    --placement-capacity-mode fixed_mean || break

  PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
    --joint-input-dir "$CITY" \
    --services hospital polyclinic school kindergarten \
    --use-placement-outputs \
    --modality bus \
    --n-routes 1 || break
done
```

### Batch По Произвольному Набору Путей

Если города лежат в разных родительских папках:

```bash
cd /Users/gk/Code/super-duper-disser

CITIES=(
  "/abs/path/city_1"
  "/abs/path/city_2"
  "/abs/path/city_3"
)

for CITY in "${CITIES[@]}"; do
  echo "=== $(basename "$CITY") ==="

  PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
    --joint-input-dir "$CITY" \
    --services hospital polyclinic school kindergarten \
    --placement-exact \
    --placement-genetic \
    --placement-allow-existing-expansion \
    --placement-prefer-existing \
    --placement-capacity-mode fixed_mean || break

  PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
    --joint-input-dir "$CITY" \
    --services hospital polyclinic school kindergarten \
    --use-placement-outputs \
    --modality bus \
    --n-routes 1 || break
done
```

### Batch Только Для Recompute Provision По Уже Готовым Route Results

Если `connectpt_routes_generator/<modality>/summary.json` уже лежит в каждом городе и нужно быстро пересчитать финальную обеспеченность без повторного route generation:

```bash
cd /Users/gk/Code/super-duper-disser

for CITY in /Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/*; do
  echo "=== $(basename "$CITY") ==="

  PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
    --joint-input-dir "$CITY" \
    --services hospital polyclinic school kindergarten \
    --use-placement-outputs \
    --modality bus \
    --no-generate-routes \
    --recompute-provision || break
done
```

### Batch Только Для Transport-Side Debug

Если нужно отдельно погонять `ConnectPT` по уже сохраненной внешней OD для нескольких городов:

```bash
cd /Users/gk/Code/super-duper-disser

for CITY in /Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/*; do
  echo "=== $(basename "$CITY") ==="

  PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl-connectpt-batch \
    connectpt/.venv/bin/python -m aggregated_spatial_pipeline.connectpt_data_pipeline.run_route_generator_external \
    --joint-input-dir "$CITY" \
    --modality bus \
    --od-matrix-path "$CITY/pipeline_2/accessibility_first/service_target_od/bus_service_target_od.csv" \
    --n-routes 1 || break
done
```

### Что Полезно Проверять После Каждого Города В Batch

- `pipeline_2/placement_exact_genetic/<service>/summary_after.json`
- `pipeline_2/accessibility_first/service_target_od/bus_service_target_od_summary.json`
- `connectpt_routes_generator/bus/summary.json`
- `pipeline_2/accessibility_first/provision_after_routes/<service>/summary_after_routes.json`
- `preview_png/all_together`

## Запуск По Шагам Отдельно

Ниже минимальный разрез по этапам, если нужно итерироваться без полного прогона заново.

### Только Placement

```bash
PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
  --joint-input-dir "$CITY" \
  --services polyclinic \
  --placement-exact \
  --placement-genetic
```

Проверять:

- `pipeline_2/placement_exact_genetic/polyclinic/summary_after.json`
- `pipeline_2/placement_exact_genetic/polyclinic/assignment_links_after.csv`
- `preview_png/all_together/*placement*`

### Только Построение Route Target И Полный Route Step

```bash
PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
  --joint-input-dir "$CITY" \
  --services polyclinic \
  --use-placement-outputs \
  --placement-root-name placement_exact_genetic \
  --modality bus \
  --n-routes 1
```

Проверять:

- `pipeline_2/accessibility_first/service_target_od/bus_service_target_od.csv`
- `pipeline_2/accessibility_first/service_target_od/bus_service_target_od_summary.json`
- `connectpt_routes_generator/bus/summary.json`
- `connectpt_routes_generator/bus/intermodal_replaced/graph.pkl`
- `connectpt_routes_generator/bus/accessibility_recomputed/adj_matrix_time_min_union.parquet`

### Только Provision Recompute По Уже Готовому Route Summary

Если маршрут уже был сгенерен и accessibility уже пересчитана, можно не гонять `ConnectPT` заново:

```bash
PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
  --joint-input-dir "$CITY" \
  --services polyclinic \
  --use-placement-outputs \
  --placement-root-name placement_exact_genetic \
  --modality bus \
  --n-routes 1 \
  --no-generate-routes \
  --recompute-provision
```

По умолчанию возьмется:

```bash
$CITY/connectpt_routes_generator/bus/summary.json
```

Можно указать summary явно:

```bash
PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
  --joint-input-dir "$CITY" \
  --services polyclinic \
  --use-placement-outputs \
  --placement-root-name placement_exact_genetic \
  --modality bus \
  --no-generate-routes \
  --route-summary-path "$CITY/connectpt_routes_generator/bus/summary.json"
```

### Только Прямой Запуск ConnectPT С Уже Готовой Внешней OD

Иногда удобно дебажить transport-side без всего `accessibility_first`:

```bash
PYTHONPATH=$PWD MPLCONFIGDIR=/tmp/mpl-connectpt-debug \
  connectpt/.venv/bin/python -m aggregated_spatial_pipeline.connectpt_data_pipeline.run_route_generator_external \
  --joint-input-dir "$CITY" \
  --modality bus \
  --od-matrix-path "$CITY/pipeline_2/accessibility_first/service_target_od/bus_service_target_od.csv" \
  --n-routes 1 \
  --min-route-len 2 \
  --max-route-len 8 \
  --output-dir /tmp/connectpt_debug_vienna
```

Важно:

- без `--replace-existing-modality-routes` граф будет additive;
- то есть существующие `bus` ребра сохранятся, новые добавятся поверх.

Если зачем-то нужен старый destructive режим:

```bash
... run_route_generator_external \
  ... \
  --replace-in-intermodal \
  --replace-existing-modality-routes
```

## Главные Артефакты После Полного Прогона

### Placement

- `pipeline_2/placement_exact_genetic/<service>/blocks_solver_after.parquet`
- `pipeline_2/placement_exact_genetic/<service>/assignment_links_after.csv`
- `pipeline_2/placement_exact_genetic/<service>/summary_after.json`

### Accessibility-First

- `pipeline_2/accessibility_first/manifest_accessibility_first.json`
- `pipeline_2/accessibility_first/service_target_od/bus_service_target_od.csv`
- `pipeline_2/accessibility_first/service_target_od/bus_service_target_od_summary.json`
- `pipeline_2/accessibility_first/provision_after_routes/<service>/blocks_solver_after_routes.parquet`
- `pipeline_2/accessibility_first/provision_after_routes/<service>/summary_after_routes.json`
- `pipeline_2/accessibility_first/suffering_summary_after_routes.json`

### ConnectPT

- `connectpt_routes_generator/bus/summary.json`
- `connectpt_routes_generator/bus/intermodal_replaced/graph.pkl`
- `connectpt_routes_generator/bus/intermodal_replaced/bus_generated_route_edges.parquet`
- `connectpt_routes_generator/bus/accessibility_recomputed/adj_matrix_time_min_union.parquet`

### PNG

- `preview_png/all_together/pt_route_generator_bus.png`
- `preview_png/all_together/pt_route_generator_bus_with_existing.png`
- `preview_png/all_together/pt_route_generator_bus_generated_only.png`
- `preview_png/all_together/accessibility_mean_time_map_bus_before.png`
- `preview_png/all_together/accessibility_mean_time_map_bus_after.png`
- `preview_png/all_together/accessibility_mean_time_map_bus_generated.png`
- `preview_png/all_together/accessibility_mean_time_delta_map_bus_generated.png`
- `preview_png/all_together/lp_<service>_provision_before_placement.png`
- `preview_png/all_together/lp_<service>_provision_after_placement.png`
- `preview_png/all_together/lp_<service>_provision_delta_after_placement.png`
- `preview_png/all_together/lp_<service>_placement_changes.png`
- `preview_png/all_together/lp_<service>_provision_before_routes.png`
- `preview_png/all_together/lp_<service>_provision_after_routes.png`
- `preview_png/all_together/lp_<service>_provision_delta_after_routes.png`

## Как Проверять, Что Все Передалось Нормально

### 1. Проверить, что route target реально ненулевой

Смотреть:

- `pipeline_2/accessibility_first/service_target_od/bus_service_target_od_summary.json`

Ключевые поля:

- `positive_pairs`
- `target_weight_total`
- `services[].used_links`

Если `target_weight_total = 0`, route generation будет пропущен как неинформативный.

### 2. Проверить, что ConnectPT реально использовал внешний target

Смотреть:

- `connectpt_routes_generator/bus/summary.json`

Ключевые поля:

- `od_source` должно быть `external`
- `od_matrix_input_path` должен указывать на `service_target_od`

### 3. Проверить, что существующая PT-сеть не удалена

Смотреть:

- `connectpt_routes_generator/bus/summary.json`

Ключевые поля:

- `intermodal_replacement.merge_mode` должно быть `append_to_existing_modality_routes`
- `intermodal_replacement.removed_modality_edges` должно быть `0`
- `intermodal_replacement.generated_modality_edges` должно быть больше `0`

### 4. Проверить, что accessibility действительно пересчитана

Смотреть:

- `connectpt_routes_generator/bus/accessibility_recomputed/adj_matrix_time_min_union.parquet`
- `preview_png/all_together/accessibility_mean_time_map_bus_before.png`
- `preview_png/all_together/accessibility_mean_time_map_bus_after.png`
- `preview_png/all_together/accessibility_mean_time_delta_map_bus_generated.png`
- `preview_png/all_together/accessibility_mean_time_map_bus_generated.png`

### 5. Проверить, что provision считается уже после placement и после route

Смотреть:

- `pipeline_2/accessibility_first/provision_after_routes/<service>/summary_after_routes.json`

Ключевые поля:

- `accessibility_gap_before`
- `accessibility_gap_after`
- `accessibility_gap_delta`

Также смотреть:

- `pipeline_2/accessibility_first/suffering_summary_after_routes.json`
- `preview_png/all_together/lp_<service>_provision_before_placement.png`
- `preview_png/all_together/lp_<service>_provision_after_placement.png`
- `preview_png/all_together/lp_<service>_provision_delta_after_placement.png`
- `preview_png/all_together/lp_<service>_provision_before_routes.png`
- `preview_png/all_together/lp_<service>_provision_after_routes.png`
- `preview_png/all_together/lp_<service>_provision_delta_after_routes.png`
- `preview_png/all_together/lp_<service>_placement_changes.png`

## Практические Замечания

### Про Runtime

Для `ConnectPT` route generation скрипт сначала ищет:

1. `connectpt/.venv/bin/python`
2. `.venv/bin/python`

Это сделано потому, что route generator может требовать пакеты, которых нет в корневой `.venv`.

### Про Route Length Policy

По умолчанию `run_pipeline2_accessibility_first` пытается согласовать длину новых маршрутов с существующими:

- `--align-route-len-to-existing-mean-max`

Из-за этого реальные bounds могут отличаться от запрошенных.
Это видно в:

- `connectpt_routes_generator/bus/summary.json`
- поле `pipeline_route_length_policy`

### Про Небольшой Эффект На Маршруте

То, что пайплайн отработал корректно, не означает, что маршрут обязательно даст большой выигрыш.

Небольшой или почти нулевой эффект может означать:

- слабый target;
- неудачный single-route candidate;
- слишком жесткие route-length bounds;
- то, что внутренняя cost function `ConnectPT` пока еще не очень хорошо совпадает с service-loss objective.

Это уже вопрос качества оптимизации, а не вопрос работоспособности пайплайна.

## Проверенный Smoke Case

Рабочий smoke был прогнан на:

```bash
/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/old/joint_inputs/vienna_austria
```

Там было подтверждено:

- service-aware OD ненулевая;
- `ConnectPT` использовал `od_source=external`;
- generated route был встроен в intermodal graph без удаления существующих bus-ребер;
- accessibility была пересчитана;
- provision after routes был пересчитан на after-placement capacities.

## Короткий Чеклист Перед Доверием К Результату

- открыл `summary_after.json` у placement;
- открыл `bus_service_target_od_summary.json`;
- открыл `connectpt_routes_generator/bus/summary.json`;
- убедился, что `od_source=external`;
- убедился, что `merge_mode=append_to_existing_modality_routes`;
- убедился, что `adj_matrix_time_min_union.parquet` реально записан;
- открыл `summary_after_routes.json`;
- проверил PNG в `preview_png/all_together`.

# Aggregated Spatial Pipeline

Минимальный каркас для сборки единого пайплайна поверх нескольких пространственных разбиений.

Идея простая:

- есть несколько слоев пространственных единиц;
- между слоями есть `crosswalk` по пересечениям;
- перенос атрибутов между слоями делается по словарю правил агрегации/дезагрегации;
- сценарии описываются отдельно и работают уже на согласованном аналитическом слое.

## Интеграционный принцип

Этот слой специально задуман как `non-invasive`.

То есть:

- мы не правим исходные репозитории и модули ради интеграции;
- мы не меняем их внутренние форматы без необходимости;
- вся адаптация делается локально в этом пайплайне;
- любые преобразования данных, crosswalk и агрегации живут здесь, рядом с интеграционным кодом.

Практически это означает:

- `street-pattern-classifier` используется как источник морфологических результатов;
- `connectpt` используется как источник транспортных графов, OD и travel times;
- `blocksnet` используется как источник block geometry, land use, building aggregation и базовых block features;
- `floor-predictor` используется как источник предсказания `RES/NON_RES` и этажности зданий;
- `solver_flp` используется как источник логики service placement;
- `arctic_access` используется как источник климатических, устойчивостных и агломерационных расчетов.

Этот каркас подстраивается под существующие выходы модулей, а не заставляет их перестраиваться под себя.

Еще один важный принцип:

- интеграция строится не между абстрактными файлами и не между случайными тулзами;
- интеграция строится между осмысленными моделями изменения города.

В этой логике каждый основной репозиторий рассматривается как отдельная модель городской трансформации:

- изменения использования земли;
- изменения ожидаемой застройки;
- изменения морфологического и топологического контекста;
- изменения OD;
- изменения транспортной сети;
- изменения сервисного обеспечения;
- изменения связности и устойчивости под климатическими и сезонными воздействиями.

Поэтому bridge в этом пайплайне должен отвечать не только на вопрос:

- `как переложить данные из A в B`

но и на вопрос:

- `какой именно сигнал городской трансформации модель A передает модели B`.

Из этого следует еще два практических принципа:

- bridges должны быть не общими, а смысловыми;
- bridge должен отвечать не только на вопрос `как перелить файл`, но прежде всего на вопрос `какой смысловой сигнал один модуль передает другому`.

Иными словами, bridge описывает:

- что именно меняется в представлении города;
- в каком виде это изменение передается дальше;
- как следующий модуль должен это изменение интерпретировать.

## Роли Репозиториев

Ниже зафиксировано смысловое назначение основных модулей, чтобы интеграционный слой строился вокруг их реальной исследовательской роли, а не только вокруг форматов файлов.

### `sm_imputation`

Это блок про ожидаемую застройку при изменении использования земли.

Смысл:

- меняется `land use`;
- при этом должна меняться ожидаемая форма застройки;
- значит, нужно оценить, какой тип или характер застройки соответствует заданному переходу использования земли.

В этой логике `sm_imputation` используется как модель:

- `land-use transition -> expected built form`

При этом сам класс перехода может быть:

- задан извне;
- либо предсказан в другом модуле.

### `street-pattern-classifier`

Это блок про топологическую неоднородность города.

Смысл:

- разные части города различаются по структуре уличной сети;
- эти различия важны для транспортного планирования;
- разные модальности могут быть более или менее уместны в разных морфологических контекстах.

В этой логике `street-pattern-classifier` используется как модель:

- `urban network topology -> street-pattern descriptors`

Здесь под дескрипторами понимаются:

- класс паттерна;
- вероятности классов;
- морфологические или топологические признаки;
- при необходимости priors для пригодности модальности.

### `connectpt`

Это блок про транспортную реакцию на изменившееся состояние города.

Смысл:

- город меняется;
- меняется распределение функций, населения и спроса;
- меняется `OD`;
- значит, нужно адаптировать транспортную систему.

В этой логике `connectpt` используется как модель:

- `changed city state -> changed OD -> transport redesign`

Не принципиально, идет ли речь о генерации маршрутов с нуля или об оптимизации существующей сети. Главное, что это модуль транспортной адаптации.

### `pattern x mode` analysis

Связка между `street pattern` и использованием модальностей пока живет в отдельных ноутбуках, но концептуально это самостоятельный интеграционный блок.

Смысл:

- для автобуса некоторые паттерны могут быть менее предпочтительны;
- для веломаршрутов некоторые паттерны могут быть более предпочтительны;
- значит, транспортный генератор должен учитывать morphology-aware penalties или preferences.

В этой логике связующий блок используется как модель:

- `street pattern -> mode-specific cost modifiers`

### `solver_flp`

Это блок про модель обеспеченности сервисами, а в расширенном режиме — про оптимизацию размещения.

Смысл:

- при заданном размещении сервисов и заданной транспортной доступности нужно оценить обеспеченность спроса;
- поскольку доступность до сервисов считается по транспортной сети, изменение связности напрямую влияет на результат оценки обеспеченности;
- в расширенном режиме тот же блок может использоваться и для оптимизации размещения сервисов.

В этой логике `solver_flp` используется как модель:

- `demand + accessibility structure -> provision / coverage assessment`

Практически это означает:

- в базовом режиме `solver_flp` используется для оценки того, насколько текущая сеть и текущее размещение сервисов покрывают спрос;
- поэтому он естественно получает на вход результат транспортной доступности, вычисленной через `connectpt` или другой транспортный слой;
- в расширенном режиме он может переходить к задаче переразмещения или добавления сервисов;
- концептуально он также может использоваться для постановки вопроса: можно ли сохранить или улучшить обеспеченность при меньшем числе сервисов за счет улучшения связности.

### `arctic_access`

Это блок про агломерационный, климатический и устойчивостный контекст.

Смысл:

- транспортные и сервисные сети зависят не только от внутригородской структуры, но и от внешних ограничений;
- климатические факторы могут менять достижимость, связность и устойчивость;
- отдельные части сети могут сезонно деградировать или временно выпадать из связности;
- на более крупном масштабе город может рассматриваться как агрегированный узел.

В этой логике `arctic_access` используется как модель:

- `hazards / climate / seasonal degradation -> resilience-aware network state`

При этом `arctic_access` важен не только как описание риска, но и как постановка адаптационной задачи:

- если сеть сезонно меняется или частично распадается;
- и из-за этого часть узлов оказывается ближе к изоляции;
- то нужно понять, как изменить связность сети и где при необходимости добавить минимум сервисов;
- так, чтобы с учетом сезонной транспортной доступности никто не оставался критически изолирован.

В этом смысле арктический уровень связан и с `solver_flp`, и с `connectpt`:

- `solver_flp` отвечает на вопрос, где сервисы нужно добавить или перераспределить минимальным образом;
- `connectpt` отвечает на вопрос, какие маршруты или связи нужно добавить, чтобы компенсировать сезонную потерю доступности;
- `arctic_access` задает сами сезонные ограничения, шоки и сценарии распада связности.

## Типы Контекста

Для интеграционной логики полезно различать два типа контекстных модулей:

- `active context` — модуль, который меняет саму `accessibility matrix` или сетевое состояние, а значит напрямую меняет результат расчета обеспеченности;
- `descriptive context` — модуль, который не меняет матрицу доступности, а только размечает, группирует, стратифицирует или объясняет результаты.

Практически это означает:

- `arctic_access` в сезонно-климатическом режиме — это `active context`, потому что он меняет достижимость между узлами;
- `street-pattern` в текущем виде — это в основном `descriptive context`, потому что он пока в основном добавляет морфологическую интерпретацию и стратификацию результатов;
- один и тот же модуль в будущем может перейти из `descriptive` в `active`, если начнет напрямую менять costs, penalties или саму матрицу достижимости.

## Общая Цепочка

Если смотреть на все модули вместе, они образуют не набор разрозненных утилит, а последовательность смысловых преобразований:

1. `land-use transition`
2. `expected built form` через `sm_imputation`
3. `building living/non-living + floors` через `floor-predictor`
4. `street-pattern / topology characterization`
5. `OD shift`
6. `transport redesign` через `connectpt`
7. `mode-pattern compatibility adjustments`
8. `service placement and connectivity co-optimization`
9. `accessibility and resilience outcomes`

Именно вокруг этой цепочки и должен строиться интеграционный пайплайн.

## Что лежит в папке

- `config/layers.json` — словарь пространственных слоев;
- `config/crosswalks.json` — связи между слоями;
- `config/transfer_rules.json` — правила переноса атрибутов;
- `config/scenarios.json` — минимальный набор сценариев;
- `config/policy.json` — интеграционные ограничения;
- `spec.py` — маленький загрузчик и валидатор спецификации.
- `pipeline/` — новый интеграционный слой crosswalk/aggregation.
- `connectpt_data_pipeline/` — отдельный OSM ingest для transport inputs по городу.
- `blocksnet_data_pipeline/` — отдельный pipeline нарезки кварталов через BlocksNet `dev`.
- `floor_predictor_pipeline/` — отдельная обвязка для sibling-repo `floor-predictor`.

## Базовая логика масштаба

Сейчас каркас предполагает такую цепочку:

- `street_grid -> quarters -> cities`
- `climate_grid -> quarters -> cities`

## ConnectPT OSM Ingest

Отдельно добавлен новый, изолированный пайплайн сборки данных для `connectpt`.

Принцип:

- исходный `connectpt` код не меняется;
- новый код живет отдельно в `aggregated_spatial_pipeline/connectpt_data_pipeline/`;
- на вход подается `--place`;
- дальше по этому городу через OSM собираются:
  - boundary;
  - aggregated stops по модальностям;
  - линии;
  - линии после projection stops on roads;
  - граф stop-to-stop;
  - `time_matrix.csv`.

Точная команда:

```bash
PYTHONPATH=/Users/gk/Code/super-duper-disser \
/Users/gk/Code/super-duper-disser/.venv/bin/python -m aggregated_spatial_pipeline.connectpt_data_pipeline.run \
  --place 'Tianjin, China' \
  --modalities bus tram \
  --output-dir /Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/connectpt_osm/tianjin_china
```

Что лежит в результате:

- `boundary.geojson`
- `centre.geojson`
- `<modality>/aggregated_stops.geojson`
- `<modality>/lines.geojson`
- `<modality>/projected_lines.geojson`
- `<modality>/projected_stops.geojson`
- `<modality>/graph_nodes.geojson`
- `<modality>/graph_edges.geojson`
- `<modality>/time_matrix.csv`
- `manifest.json`

## BlocksNet Dev Ingest

`blocks` теперь вынесены в отдельный pipeline и отдельную среду.

Почему так:

- `BlocksNet` нужен с `dev`-ветки;
- под него удобнее держать отдельную `uv`-venv;
- это не смешивает зависимости `BlocksNet` с основной интеграционной средой.

Сначала клонируем `blocksnet` рядом с остальными репозиториями:

```bash
git clone https://github.com/aimclub/blocksnet.git \
  /Users/gk/Code/super-duper-disser/blocksnet
```

Потом поднимаем отдельную среду из локального checkout:

```bash
/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/blocksnet_data_pipeline/setup_blocksnet_dev_env.sh
```

При необходимости можно переопределить путь:

```bash
BLOCKSNET_DIR=/Users/gk/Code/somewhere/blocksnet \
/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/blocksnet_data_pipeline/setup_blocksnet_dev_env.sh
```

Запуск нарезки кварталов:

```bash
PYTHONPATH=/Users/gk/Code/super-duper-disser \
/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/.venv-blocksnet-dev/bin/python \
  -m aggregated_spatial_pipeline.blocksnet_data_pipeline.run \
  --place 'Tianjin, China' \
  --output-dir /Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/blocksnet/tianjin_china
```

Что сохраняется:

- `boundary.geojson`
- `water.geojson`
- `roads.geojson`
- `railways.geojson`
- `land_use.geojson`
- `buildings.geojson`
- `blocks.geojson`
- `manifest.json`

Что дополнительно появляется в `blocks.geojson` уже сейчас:

- `living_area_proxy`
- `population_proxy`
- `density_proxy`

Базовая временная формула:

- `population_proxy = living_area_proxy / 20`
- если `living_area` не агрегировалась, то `living_area_proxy = build_floor_area * 0.8`

Это временный baseline, пока не подключен более аккуратный расчёт через внешний population source или более точную building-level модель.

## floor-predictor

`floor-predictor` добавлен в схему как ещё один sibling-repo рядом с остальными зависимостями.

Его роль:

- предсказание `RES / NON_RES`
- предсказание этажности жилых зданий

Клонирование рядом с остальными репозиториями:

```bash
git clone https://github.com/GeorgeKontsevik/floor-predictor.git \
  /Users/gk/Code/super-duper-disser/floor-predictor
```

Отдельная среда:

```bash
/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/floor_predictor_pipeline/setup_floor_predictor_env.sh
```

При необходимости путь можно переопределить:

```bash
FLOOR_PREDICTOR_DIR=/Users/gk/Code/somewhere/floor-predictor \
/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/floor_predictor_pipeline/setup_floor_predictor_env.sh
```

При этом:

- `quarters` — основной аналитический уровень внутри города;
- `street_grid` и `climate_grid` — технические или внешние слои;
- `cities` — уровень агломерации, где город выступает как супер-узел.

## Pipeline_2: Сбор Данных И Подготовка Для Solver

Для второго пайплайна добавлен отдельный скрипт:

- `aggregated_spatial_pipeline/pipeline/run_pipeline2_prepare_solver_inputs.py`

Он запускается поверх уже готового city bundle из `pipeline_1` и делает шаги строго в таком порядке:

1. `data_collection`: скачать raw OSM сервисы внутри той же analysis territory (`buffer.parquet`).
2. `capacity_aggregation`: агрегировать сервисные capacity по blocks.
3. `matrix_build`: посчитать матрицу доступности между блоками через native shortest-path расчёт по intermodal graph.
4. `solver_prep`: сохранить solver-ready таблицы по каждому сервису (`demand`, `demand_within`, `demand_without`, `capacity`, `capacity_left`, `provision`).
5. `accessibility_first` (новый шаг): собрать multi-service рейтинг «страдающих» blocks, сгенерировать небольшое число ConnectPT маршрутов и пересчитать provision после новых маршрутов.

Теги, которые используются на шаге `data_collection`:

- `hospital`: `{"amenity": "hospital"}`, `{"healthcare": "hospital"}`
- `polyclinic`: `{"amenity": "clinic"}`, `{"healthcare": ["clinic", "centre"]}`
- `school`: `{"amenity": "school"}`

Пример запуска:

```bash
PYTHONPATH=/Users/gk/Code/super-duper-disser \
/Users/gk/Code/super-duper-disser/.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
  --joint-input-dir /Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/joint_inputs/barcelona_spain \
  --no-cache \
  --osm-timeout-s 180 \
  --overpass-url https://overpass-api.de/api/interpreter
```

Что сохраняется:

- `pipeline_2/services_raw/<service>.parquet`
- `pipeline_2/prepared/units_union.parquet`
- `pipeline_2/prepared/adj_matrix_time_min_union.parquet`
- `pipeline_2/solver_inputs/<service>/blocks_solver.parquet`
- `pipeline_2/solver_inputs/<service>/adj_matrix_time_min.parquet`
- `pipeline_2/solver_inputs/<service>/provision_links.csv`
- `pipeline_2/manifest_prepare_solver_inputs.json`

### Pipeline_2 Accessibility-First Step

Отдельный шаг:

- `aggregated_spatial_pipeline/pipeline/run_pipeline2_accessibility_first.py`

Логика:

1. Собрать combined `suffering_blocks` по 3 сервисам:
   - `accessibility_gap` = `demand_without`
   - `capacity_gap` = `demand_left`
2. Выбрать наиболее страдающие blocks.
3. Сгенерировать небольшое число ConnectPT маршрутов (например, `n_routes=2`).
4. Пересчитать provision после новых маршрутов на обновлённой матрице доступности.

Пример запуска:

```bash
PYTHONPATH=/Users/gk/Code/super-duper-disser \
/Users/gk/Code/super-duper-disser/.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
  --joint-input-dir /Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/joint_inputs/vienna_austria \
  --services hospital polyclinic school \
  --modality bus \
  --n-routes 2
```

Артефакты шага:

- `pipeline_2/accessibility_first/suffering_blocks_baseline.parquet`
- `pipeline_2/accessibility_first/top_blocks_access_gap_baseline.csv`
- `pipeline_2/accessibility_first/top_blocks_total_gap_baseline.csv`
- `pipeline_2/accessibility_first/provision_after_routes/<service>/blocks_solver_after_routes.parquet`
- `pipeline_2/accessibility_first/manifest_accessibility_first.json`

## Главные сущности

### 1. Layers

Слой описывает тип пространственных единиц:

- кварталы;
- регулярную сетку;
- гексагоны;
- климатические ячейки;
- города.

### 2. Crosswalks

`Crosswalk` хранит способ перехода между двумя несовпадающими разбиениями.

Минимальные поля:

- `intersection_area`
- `source_share`
- `target_share`

При необходимости можно добавлять специальные веса:

- `population_weight`
- `builtup_weight`
- `network_weight`

### 3. Transfer rules

Каждый атрибут переносится по своему правилу.

Примеры:

- `population_total` — сумма;
- `temperature_mean` — взвешенное среднее;
- `street_pattern_probs` — взвешенное среднее по вероятностям;
- `street_pattern_class` — majority vote;
- `service_capacity_total` — сумма.

### 4. Scenarios

Минимальный набор сценариев:

- `baseline`
- `service_only`
- `transport_only`
- `joint_optimization`
- `shock_resilience`

## Как использовать

Посмотреть, что конфиг согласован:

```bash
python -m aggregated_spatial_pipeline.spec
```

## Что добавлять дальше

Следующий слой реализации логично строить так:

1. загрузка исходных GeoDataFrame для каждого слоя;
2. генерация `crosswalk` через spatial overlay;
3. применение `transfer_rules`;
4. сборка целевого слоя `quarters`;
5. агрегация `quarters -> cities`;
6. прогон сценариев.

## Новый интеграционный код

Вся реализация лежит отдельно в подпапке `aggregated_spatial_pipeline/pipeline/`.

Это важно:

- исходные модули не трогаются;
- адаптация и агрегация живут только здесь;
- входом являются уже подготовленные spatial layers;
- выходом являются crosswalk, harmonized layers и сценарные слои.

Сейчас в `pipeline/` есть:

- `io.py` — загрузка и сохранение spatial layers;
- `crosswalks.py` — построение spatial overlay crosswalk;
- `transfers.py` — применение transfer rules;
- `scenarios.py` — baseline и сценарный раннер;
- `run.py` — CLI для запуска пайплайна.

## Запуск пайплайна

Базовый запуск:

```bash
PYTHONPATH=/Users/gk/Code/super-duper-disser \
/Users/gk/Code/super-duper-disser/.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run \
  --quarters /path/to/quarters.geojson \
  --street-grid /path/to/street_grid.geojson \
  --climate-grid /path/to/climate_grid.geojson \
  --cities /path/to/cities.geojson \
  --output-dir /path/to/aggregated_outputs
```

Что появляется в `output-dir`:

- `crosswalks.gpkg`
- `manifest.json`
- `<scenario_id>/quarters.geojson`
- `<scenario_id>/cities.geojson`
- `<scenario_id>/metadata.json`

Неподдержанные пока сценарные операции вроде `service_optimization`, `transport_optimization` и `network_shock` не ломают запуск:

- baseline и harmonization считаются;
- в `metadata.json` такие операции помечаются как `pending_operations`;
- это оставляет место для следующего шага интеграции без вмешательства в upstream-код.

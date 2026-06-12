# City-Level Polyclinic Coverage Research Scheme

Этот файл фиксирует отдельный research track поверх `polyclinic_access_components`.

Главный вопрос:

- как то, как устроена городская сеть, связано с текущей `PT + walk` доступностью `polyclinic`
- и насколько легко или трудно эту доступность улучшить

Placement-only baseline:

- `additional polyclinics needed`

Это не финальный ответ на research question.
Это counterfactual:

- сколько объектов нужно, если транспорт не менять

Главная проверяемая мера для гипотезы:

- `additional PT routes needed` или route-level PT improvement
- изменение `additional polyclinics needed` после route-level PT intervention

## Main Thesis To Preserve

Главная исследовательская мысль этого track:

- `street pattern` нужен не для ранжирования паттернов как хороших или плохих
- он нужен как diagnostic layer, который помогает понять, где targeted transport improvement может заменить часть нового service placement
- центральный эксперимент не `добавить поликлиники`, а `может ли добавление / замена маршрута или ускорение коридора уменьшить число нужных новых поликлиник`

Рабочая гипотеза:

- если доступность `polyclinic` можно улучшить за счет небольшого изменения PT в конкретных местах, то это должно проявляться не только в city-wide coverage
- это должно проявляться в сочетании:
  - где живет спрос
  - где уже стоят сервисы
  - где solver выбирает или почти выбирает новые кварталы для размещения
  - какие `street-pattern contexts` вокруг этих кварталов
  - как эти contexts связаны с PT stops, PT routes, first-mile failures и PT-segment failures

То есть главный вопрос не `какой street pattern лучше`.

Главный вопрос:

- в каких морфологических и транспортных условиях можно поднять `PT + walk` accessibility не только добавлением новых `polyclinic`, но и targeted PT intervention:
  - ускорить существующий link / corridor
  - добавить маршрут
  - заменить один маршрут другим
  - улучшить связность вокруг solver-selected или high-probability candidate blocks

Проверяемая planning idea:

- если targeted PT improvement вокруг выбранных / вероятных кварталов поднимает coverage до target с меньшим числом новых объектов, то street-pattern layer дает практический ответ:
  `где транспортное улучшение может быть substitute for service placement`
- placement-only target90 нужен только как baseline для сравнения:
  `до route intervention` vs `после route intervention`

Эту мысль надо держать выше descriptive results про отдельные pattern shares, lifts и correlations.

## Core Outcome

Под `обеспеченностью` здесь понимается:

- доля жилых домов, для которых `polyclinic` попадает в заданный `PT + walk` time threshold

Базовая формула:

- `coverage = share of residential homes with accessible polyclinic under PT + walk threshold`

## Main Explanatory Side

Два соседних explanatory слоя:

1. `street pattern`
- city-wide street-pattern mix
- local street-pattern context

2. `PT network`
- простые дескрипторы уровня города
- например число маршрутов, число остановок, route density, stop density

Здесь `street pattern` и `PT network` не смешиваются в один индекс.
Они идут рядом.

## Core Sample

### Main Working Sample

По текущему audit основной deduped large sample для этого question:

- `36` городов

Он состоит из:

- `19` active19 cities
- `16` eligible cities from `new17`
- `1` eligible unique city from `old23`

`new5` отдельно в основной sample не нужен:

- все его города уже покрыты `new17`

### Source Priority And Deduplication

Дедупликация делается так:

- `active19` > `new17` > `old23` > `new5`

То есть:

- если город уже есть в `active19`, берем `active19`
- если нет, но есть в `new17`, берем `new17`
- если нет, но есть в `old23`, берем `old23`
- `new5` служит только как промежуточный источник, но не как отдельный слой в final sample

### Sample Composition

#### Active19

Источник:

- `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs`

Список:

- `bergen_norway`
- `bologna_italy`
- `bristol_united_kingdom`
- `brno_czechia`
- `coimbra_portugal`
- `debrecen_hungary`
- `dresden_germany`
- `freiburg_im_breisgau_germany`
- `gothenburg_sweden`
- `graz_austria`
- `innsbruck_austria`
- `krakow_poland`
- `linz_austria`
- `lyon_france`
- `marseille_france`
- `porto_portugal`
- `turin_italy`
- `turku_finland`
- `zaragoza_spain`

Статус:

- у них есть diagnostics по `polyclinic`
- у них есть city bundles с `street pattern`
- у них есть `pipeline_2/solver_inputs/polyclinic`
- у них есть базовые PT-related layers

#### New17 Eligible Additions

Источник:

- `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_new17_access_20260610/joint_inputs_merged`

В основной sample входят:

- `adelaide_south_australia_australia`
- `amsterdam_netherlands`
- `arequipa_peru`
- `delft_netherlands`
- `hai_duong_h_i_d_ng_vietnam`
- `huainan_anhui_china`
- `jaynagar_bih_r_india`
- `kakogawacho_honmachi_hy_go_japan`
- `kananga_kasa_central_congo_kinshasa`
- `maracay_aragua_venezuela`
- `montes_claros_minas_gerais_brazil`
- `naihati_west_bengal_india`
- `narayanganj_dhaka_bangladesh`
- `spring_valley_nevada_united_states`
- `temuco_araucan_a_chile`
- `vologda_russia`

Исключен:

- `nouakchott_nouakchott_ouest_mauritania`

Причина исключения:

- по текущему audit у него нет подтвержденного `solver_inputs/polyclinic` слоя, нужного для `additional polyclinics needed`

#### Old23 Eligible Unique Addition

Источник:

- `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_old23_access_20260609/joint_inputs_merged`

В основной sample добавляется только:

- `vienna_austria`

Почему только она:

- остальные eligible old23 cities либо дублируют `new17`, либо не имеют подтвержденного `solver_inputs/polyclinic` слоя

### Practical Limitation

Готовые after-placement outputs для `additional polyclinics needed` сейчас есть только у части sample.

Подтвержденные готовые `target 0.9` outputs сейчас есть у:

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

Значит:

- interim placement analyses можно честно делать на ready subset
- full-sample placement layer все еще надо досчитывать новым кодом для всего `36`-city sample

## Verified Data Sources

### Baseline Accessibility

Главный baseline source:

- `/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/_all_home_to_service_access_diagnostics.parquet`

Также уже есть derived polyclinic layer:

- `/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs/polyclinic_home_access_components.parquet`

### Street Pattern

City-level street-pattern sources:

- `joint_inputs/<city>/street_pattern/<city>_summary.json`
- `joint_inputs/<city>/street_pattern/<city>/predicted_cells.csv`
- `joint_inputs/<city>/street_pattern/<city>/predicted_cells.geojson`

### Solver Inputs For Service Addition

Per-city baseline solver data:

- `joint_inputs/<city>/pipeline_2/solver_inputs/polyclinic/summary.json`
- `joint_inputs/<city>/pipeline_2/solver_inputs/polyclinic/blocks_solver.parquet`
- `joint_inputs/<city>/pipeline_2/solver_inputs/polyclinic/adj_matrix_time_min.parquet`
- `joint_inputs/<city>/pipeline_2/solver_inputs/polyclinic/provision_links.csv`

Important:

- `blocks_solver.parquet` already contains per-block demand / capacity / unmet fields
- но `street pattern` в solver layers есть не везде
- для placement-lift слоя block-level street pattern надо собирать отдельно:
  `street_pattern/<city>/predicted_cells.geojson -> blocks_clipped -> solver block name`

## Placement Lift Layer

Для вопроса

- `какие street patterns solver выбирает сверх их фоновой доли, когда надо добить 0.9`

используется отдельный слой:

- baseline A: `share_pattern_in_city_blocks`
- baseline B: `share_pattern_in_solver_candidate_blocks`
- target: `share_pattern_among_selected_new_polyclinics`

Главная рабочая метрика:

- `placement_lift_vs_candidates = selected_share - candidate_share`

Дополнительно сохраняются:

- `placement_lift_vs_city = selected_share - city_share`
- `placement_ratio_vs_candidates = selected_share / candidate_share`
- `placement_ratio_vs_city = selected_share / city_share`

Это сильнее, чем грубая cross-city связь вида

- `share_irregular_grid in city -> additional polyclinics needed`

потому что здесь мы смотрим не на фон паттерна в городе, а на то, какие block contexts solver реально добирает сверх baseline.

### PT Descriptors

Primary PT-descriptor candidates:

- `services_to_pt_top3/_run_report.tsv`
- `residential_to_pt_top3/_run_report.tsv`
- `joint_inputs/<city>/intermodal_graph_iduedu/graph.pkl`

Optional corridor-morphology layer:

- `joint_inputs/<city>/pt_street_pattern_dependency/route_stats.csv`
- `joint_inputs/<city>/pt_street_pattern_dependency/route_class_length.csv`

This layer is useful, but not required for the city-level baseline.

## Main Outcomes

### Outcome 1: Baseline Coverage

Per city:

- доля домов, у которых `polyclinic` доступна по `PT + walk` threshold

Operationally:

- берется из existing access diagnostics
- переводится в city-level `coverage`

### Outcome 2: Additional Polyclinics Needed

Per city:

- сколько дополнительных `polyclinics` нужно, чтобы поднять coverage до target

Targets:

- `0.70`
- `0.80`
- `0.90`

Это главный difficulty metric.

### Important Operational Note For `0.90`

Для `0.90` placement metric считается не как

- solver на полном исходном `unmet demand` до `100%`

а как отдельный target run:

- исходный `demand_without + demand_left` масштабируется до уровня,
  который соответствует `0.90` total provision
- exact placement затем решается уже на этом scaled target

Поэтому интерпретация метрики такая:

- `new polyclinics needed to reach or exceed 0.90`

а не:

- `new polyclinics needed for full coverage`
- и не `exactly 0.900 without overshoot`

Overshoot допустим:

- из-за дискретных новых объектов и fixed capacity итоговый `provision_after`
  может быть заметно выше `0.90`

### Outcome 3: Optional PT Improvement Difficulty

Secondary layer:

- насколько route-level PT improvements меняют coverage

Этот outcome идет после service-addition layer.

## Main Tests

### Test 1: City-Level Baseline

Проверяем:

- какие города имеют higher / lower `baseline coverage`
- как это связано с `street pattern mix`
- как это связано с простыми `PT descriptors`

### Test 2: City-Level Service-Addition Difficulty

Проверяем:

- каким городам нужно больше / меньше `additional polyclinics`
- как это связано с `street pattern mix`
- как это связано с простыми `PT descriptors`

### Test 3: Within-City Mechanism

Проверяем:

- какие local street-pattern contexts чаще under-served
- в каких contexts service addition improves coverage faster or slower

### Test 4: Optional PT Improvement Layer

Проверяем:

- как сильно route-level PT improvement changes coverage
- и различается ли это across different city morphologies

Этот слой secondary.

## Main Claim Shape

Не:

- `какой street pattern лучший вообще`

А:

- `urban street-pattern structure is associated both with current PT + walk polyclinic coverage and with the number of additional polyclinics required to reach a target coverage level`

И отдельно:

- `PT descriptors help explain part of this variation, but do not replace the morphology layer`

## What We Should Not Claim

- что один street pattern универсально “лучший”
- что PT network alone все объясняет
- что новые `new17/new5/old23` уже входят в тот же полный analytical bundle

## Practical Decision

По умолчанию дальше идем так:

1. core sample = `19 active19 cities`
2. main difficulty metric = `additional polyclinics needed`
3. PT layer = рядом, но не вместо morphology
4. `new17/new5/old23` пока только extension layer, не core intersection

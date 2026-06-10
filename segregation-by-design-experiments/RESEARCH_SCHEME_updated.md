# Polyclinic Street-Pattern Research Scheme

Этот файл фиксирует рабочую схему подпроекта. По умолчанию дальше опираемся именно на нее.

## Core Question

Как `street pattern` связан с разными механизмами провала доступности до `polyclinic`.

Единица анализа:
- дом `->` ближайшая `polyclinic`

Outcome:
- не только `ok / not ok`
- а `type of failure`

## Scope Clarification

Этот подпроект является `threshold-based` и `schedule-free`.

Важно:
- `PT segment` не включает `timetables`, `frequency`, `headway`, `waiting time` или `timetable coordination`
- `PT-too-long` означает, что `PT-network path cost` / длина пути по текущему PT-графу превышает заданный threshold
- результаты не доказывают глобальную `route inefficiency`
- результаты показывают, что при текущей геометрии `PT network` часть OD-пар `home -> polyclinic` не проходит threshold

Рабочая формулировка:

`PT-too-long = threshold-insufficient PT-network path under the current schedule-free PT graph.`

## Current Priority

### Primary Priority 1

`Take 3. PT-path morphology matters separately from local morphology`

Главный фокус подпроекта:
- для `PT-based accessibility` важно не только, где расположен дом и где расположен сервис
- важно, по каким `street-pattern contexts` проходит сам `PT path`

### Primary Priority 2

`Take 5. Use street pattern diagnostically, not normatively`

Второй главный фокус:
- `street pattern` используется как diagnostic layer
- не для ранжирования паттернов как “хороших” или “плохих” самих по себе

### Supporting Layer

- `H1 / first-mile`
- `H2 / last-mile`
- `H4 / distributed failure`

Они остаются в проекте, но не задают главный narrative.

## Analytical Blocks

### Block A. Accessibility Decomposition

Сначала фиксируем, как именно распадается доступность:

- `ok walk only`
- `ok walk + PT`
- `not ok: home -> stop`
- `not ok: PT segment`
- `not ok: stop -> service`
- `not ok: walks sum`
- `not ok: total > T, no single component > T`

Это базовый descriptive layer.

### Block B. Pedestrian Morphology

Потом проверяем, связан ли `street pattern` именно с пешеходными провалами.

#### H1. Home-side street pattern hypothesis

`home-side street pattern` систематически связан с `first-mile failure`, то есть с кейсами `home -> stop not ok`.

Статус:
- supporting result

#### H2. Service-side street pattern hypothesis

`service-side street pattern` систематически связан с `last-mile failure`, то есть с кейсами `stop -> service not ok`.

Статус:
- supporting result

### Block C. PT-Path Morphology

Потом отдельно смотрим не дом и не сервис, а сам `PT path`.

#### H3. PT-path morphology hypothesis

Для кейсов `PT-too-long` важна морфология тех `street-pattern contexts`, по которым проходит сам `PT path`, а не только morphology around origin/destination.

Сравнения:
- `PT-too-long paths`
- `all polyclinic PT paths`
- `all city street pattern baseline`

Статус:
- главный результат

Интерпретация:
- это не claim про то, что маршруты “неоптимальны” вообще
- это claim про то, что текущая `PT-network geometry` дает threshold-insufficient paths для части OD-пар
- route interventions нужно трактовать как counterfactual tests: снизит ли новый / измененный маршрут `PT-network path cost` ниже `T`

### Block D. Distributed Failure

Потом отдельно берем сложный case, где ни одна отдельная компонента не превышает порог, но весь маршрут все равно выходит за threshold.

#### H4. Distributed network-effect hypothesis

Для кейсов `total > T`, `each component <= T`, `walks sum <= T` street pattern на `PT path` отражает более распределенный network effect, а не single-component bottleneck.

Статус:
- secondary / exploratory result

## Writing Order

1. `Accessibility decomposition`
2. `PT-path failures vs PT baseline`
3. `PT-path failures vs all city street-pattern baseline`
4. `Home-side pedestrian failures`
5. `Service-side pedestrian failures`
6. `Distributed total-failure case`
7. `Interpretation through street-pattern literature`

## Claims We Can Make

- разные `street-pattern contexts` связаны с разными `types of accessibility failure`
- `first-mile` failures и `PT-path` failures нельзя смешивать в один общий result про “bad accessibility”
- `PT-path morphology` дает отдельный explanatory layer поверх `home/service morphology`

## Claims We Should Not Make

- что `egress` является центральным механизмом
- что `grid` точно не limiter вообще
- что `Warped Parallel` у нас явно лучший PT pattern

## Intervention Logic: Route vs Service Placement

Этот блок нужен, чтобы Codex не смешивал разные типы policy interventions.

### Route intervention candidates

Route interventions релевантны для:

- `not ok: PT segment`
- `not ok: total > T, no single component > T`, если снижение `PT path cost` может перевести OD-пару ниже `T`

Интерпретация:
- текущие PT-пути являются `threshold-insufficient` для части OD-пар
- это не доказывает, что маршруты глобально неоптимальны
- это означает, что надо тестировать counterfactual route geometry / PT-network scenarios

Возможные counterfactuals:
- добавить более прямую PT-связь между stop clusters
- изменить трассировку маршрута на более прямой corridor
- добавить cross-link между PT corridors
- уменьшить PT path circuity
- улучшить связку между origin-side stop clusters и service-side stop clusters

Что не утверждаем:
- не говорим про `frequency`, `headway`, `waiting time`
- не говорим про `timetable coordination`
- не называем текущие маршруты “неоптимальными” без отдельного optimization proof

Рабочая формулировка:

`Route interventions are tested as counterfactual reductions in schedule-free PT-network path cost.`

### Service-placement candidates

Service placement interventions релевантны для:

- `not ok: stop -> service`
- `not ok: total > T, no single component > T`, если добавление / перенос сервиса может перевести OD-пару ниже `T`

Интерпретация:
- обычный location-allocation может хорошо выглядеть по aggregate coverage, но проваливать decomposed PT accessibility
- сервис может быть расположен “правильно” относительно demand field, но неудачно относительно PT stops и local network geometry
- candidate service sites нужно оценивать не только по demand coverage, но и по decomposed PT path

Candidate-site evaluation должна включать:
- `home -> stop`
- `PT segment`
- `stop -> service`
- `total <= T`
- component-specific thresholds, если они используются

Рабочая формулировка:

`Service placement should be optimized against decomposed PT-network accessibility, not only aggregate demand coverage.`

### Relation to Irregular Grid

`Irregular Grid` в текущих графиках является dominant exposure context.

Важно:
- высокая доля `Irregular Grid` не является causal proof, что этот pattern “плохой”
- его надо трактовать как основной spatial context, где нужно тестировать route и service-placement counterfactuals
- механизм нужно диагностировать отдельно: `PT segment too long`, `total accumulation`, `first-mile`, `last-mile`, или `service-side micro-location`

Рабочая формулировка:

`Irregular Grid is treated as the dominant exposure context, not as a normative or causal explanation of failure.`


## Planning Implications

### Take 1. Placement alone is not enough

Если цель в `PT-based accessibility to polyclinics`, то одного размещения сервиса недостаточно в средах, где слабые `home -> stop` или `stop -> service` связи сжимают реальный catchment.

Смысл:
- сервис можно поставить “правильно” по карте
- но фактическая доступность останется плохой
- если local access geometry ломает `first mile` или `last mile`

Рабочая формулировка:

`Service placement alone is insufficient in areas with poor first-mile or last-mile stop accessibility.`

### Take 2. Different failures imply different interventions

Разные типы провала требуют разных решений.

- `home -> stop`:
  home-side pedestrian connectivity problem
- `stop -> service`:
  service-side access problem
- `PT segment`:
  PT-path / network structure problem
- `total > T, each <= T, walks sum <= T`:
  distributed network-effect problem

Следствие:
- нельзя лечить все случаи одной policy
- нужен `failure-type-specific planning logic`

### Take 3. PT-path morphology matters separately from local morphology

Для PT accessibility важно не только, где расположен дом и где расположен сервис, но и через какие `street-pattern contexts` проходит сам `PT path`.

Смысл:
- origin/destination context может выглядеть приемлемо
- но проблема может сидеть в morphology of the transport corridor

Статус:
- primary planning implication

### Take 4. Weak last-mile result is still informative

То, что `stop -> service` failures редки, само по себе является результатом.

Для `polyclinic` в этом наборе городов главный bottleneck:
- не `last mile`
- а `first mile` и `PT segment`

Но:
- если `stop -> service` failures все же возникают, их можно использовать как diagnostic для service-side micro-location
- last-mile result не должен быть центральным narrative
- но он важен для обсуждения service placement и выбора candidate sites относительно PT stops

Следствие:
- `egress` не надо artificially раздувать в narrative
- но его можно использовать как supporting evidence для того, что service placement нужно проверять через decomposed PT accessibility

### Take 5. Use street pattern diagnostically, not normatively

Текущий результат не доказывает, что какой-то один pattern “лучший вообще”.

Он показывает:
- какие `morphology contexts` связаны с какими `failure modes`
- где placement likely matters most
- где connectivity improvements likely matter most
- где проблема, вероятно, сидит в `PT structure`

Статус:
- primary interpretation principle

### Main Planning Takeaway

Для повышения `PT-based accessibility to polyclinics` нужно не просто размещать сервисы, а различать:

- места, где ограничитель — local pedestrian access to stops
- места, где ограничитель — morphology of the PT path
- места, где размещение сервиса само по себе не решает проблему without connectivity improvements

## Current Interpretation

- `H1`: поддерживается
- `H2`: слабый supporting result
- `H3`: поддерживается лучше всего
- `H4`: частично поддерживается, но слабее `H3`

## How To Test Take 3

### Goal

Проверить, что `PT-path morphology` дает отдельный explanatory layer и не сводится к:
- `home-side morphology`
- `service-side morphology`
- общему city-wide `street-pattern mix`

### Test 1. Compare PT-problem paths against PT baseline

Сравнение:
- `PT-too-long paths`
- `all polyclinic PT paths`

Сильный сигнал:
- если mix по `street_pattern_class` в `PT-too-long` заметно отличается от общего PT baseline

Текущая реализация:
- `Graph 3c`

### Test 2. Compare PT baseline against city-wide street-pattern baseline

Сравнение:
- `all polyclinic PT paths`
- `all city street pattern`

Сильный сигнал:
- если PT paths проходят не просто по “среднему паттерну города”, а по своему corridor-specific mix

Текущая реализация:
- `Graph 5`

### Test 2b. Overrepresentation ratios

Shares сами по себе недостаточны, потому что некоторые patterns могут быть большими просто из-за baseline exposure.

Нужно считать ratios глобально и по городам:

1. `exposure_ratio = all_polyclinic_PT_path_share / all_city_street_pattern_share`
2. `PT_failure_ratio = PT_too_long_path_share / all_polyclinic_PT_path_share`
3. `distributed_failure_ratio = distributed_total_failure_path_share / all_polyclinic_PT_path_share`
4. `service_side_ratio = stop_to_service_not_ok_share / service_side_baseline_share`, если доступен service-side baseline

Смысл:
- отличать “pattern часто встречается в проблемах” от “pattern overrepresented in failures”
- не превращать high share `Irregular Grid` в causal claim
- проверять, где street pattern действительно дает отдельный diagnostic signal

### Test 3. Visual corridor inspection

Нужно делать визуальную аналитику по выбранным кейсам:
- маршруты, у которых основная доля времени проходит по `Irregular Grid`
- маршруты, у которых основная доля времени проходит по `Loops & Lollipops`
- маршруты с высокой долей `Warped Parallel`

Нужно отрисовывать:
- дом
- сервис
- PT path
- stop locations
- overlay street-pattern cells

Цель:
- проверить, что агрегированные доли действительно соответствуют осмысленным corridor geometries, а не артефакту overlay

### Test 4. Dominant-pattern route subsets

Нужно собрать subsets маршрутов по dominant class:
- `dominant Irregular Grid PT paths`
- `dominant Loops & Lollipops PT paths`
- `dominant Regular Grid PT paths`

Потом сравнить:
- route length
- route time
- долю `PT-too-long`

Это уже шаг от visual analytics к более формальному empirical check.

### Recommended Order

1. `Graph 3c` vs `Graph 5`
2. overrepresentation ratios
3. visual corridor inspection
4. dominant-pattern route subsets

## One-Line Thesis

`Street pattern` is used here primarily as a diagnostic description of `PT-path morphology`, with supporting evidence from `first-mile pedestrian failures` and weaker evidence for other failure mechanisms.

# Polyclinic Street-Pattern Research Scheme

Этот файл фиксирует рабочую схему подпроекта. По умолчанию дальше опираемся именно на нее.

## Core Question

Как `street pattern` связан с разными механизмами провала доступности до `polyclinic`.

Единица анализа:
- дом `->` ближайшая `polyclinic`

Outcome:
- не только `ok / not ok`
- а `type of failure`

## Main Research Thesis

Главная мысль, которую нельзя терять:

- `street pattern` нужен как способ найти, где транспортное улучшение может заменить часть нового размещения сервисов
- это не ranking паттернов и не доказательство, что один pattern универсально лучше другого
- placement-only solver output не является финальным экспериментом; это baseline, относительно которого проверяется эффект route intervention

Основная гипотеза:

- если в нужных местах немного улучшить PT accessibility, то для достижения того же coverage может понадобиться меньше новых `polyclinic`
- улучшение может быть задано как speed-up corridor/link, добавление маршрута или замена одного маршрута другим
- `street pattern` должен подсказать, где такие intervention candidates искать:
  - вокруг кварталов, которые solver выбирает для новых объектов
  - вокруг high-probability / near-selected candidate blocks
  - в местах, где спрос, сервисы, PT stops/routes и observed failures расходятся между собой

### Why Route Interventions, Not Road Rebuilding

В задаче изменения городской сети этот подпроект намеренно моделирует intervention через `PT routes`, а не через перестройку дорожной сети.

Причина:
- изменение road network обычно является тяжелым, медленным и капиталоемким вмешательством
- часть предыдущей литературы обсуждает network improvement именно как изменение roads / street links / morphology
- для реального planning experiment это часто слишком жесткий counterfactual

Маршруты ОТ являются более pragmatic / agile intervention:
- их можно добавить, перенаправить, усилить или заменить быстрее, чем перестроить улицу
- они ближе к реально управляемому инструменту транспортного планирования
- они позволяют тестировать targeted connectivity improvement вокруг solver-selected или high-probability candidate blocks
- они напрямую проверяют тезис `transport improvement can substitute for part of new service placement`

Поэтому route-generation experiment здесь не является технической деталью.
Это содержательный выбор: мы проверяем, может ли более гибкое изменение PT network дать тот же planning effect, который в более жестких постановках пытались бы получить через изменение street/road network.

Ремарка для writing:
- этот аргумент надо связать с логикой `demand-supply change`
- см. статью с Васей: изменение доступности надо интерпретировать не только как network engineering, но и как перераспределение отношения спроса и предложения
- route intervention меняет effective supply catchment существующих / потенциальных `polyclinic`
- поэтому результат `меньше новых объектов при том же target coverage` следует описывать как эффект изменения связности между demand blocks и service supply, а не просто как улучшение маршрутов само по себе

Поэтому descriptive layers ниже нужны не сами по себе.
Они должны вести к вопросу:

- `где именно PT/walk network improvement может снизить additional polyclinics needed`
- `какие selected / near-selected blocks лучше не закрывать новым объектом, а подключать новым или замененным маршрутом`

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

Следствие:
- `egress` не надо artificially раздувать в narrative

### Take 5. Use street pattern diagnostically, not normatively

Текущий результат не доказывает, что какой-то один pattern “лучший вообще”.

Он показывает:
- какие `morphology contexts` связаны с какими `failure modes`
- где placement likely matters most
- где connectivity improvements likely matter most
- где проблема, вероятно, сидит в `PT structure`

Статус:
- primary interpretation principle

### Take 6. Street pattern as PT-improvement substitution guide

Главное planning use-case:

- найти места, где targeted PT improvement может снизить `additional polyclinics needed`

Смысл:

- solver-selected или near-selected кварталы показывают, где сервисное размещение пытается закрыть deficit
- street-pattern + PT diagnostics должны показать, что именно ограничивает доступность вокруг этих кварталов
- если limiter транспортный, то надо тестировать не только новый объект, но и PT intervention:
  - ускорить corridor
  - добавить маршрут
  - заменить маршрут
  - улучшить связность с остановками

Рабочая формулировка:

`Street pattern is used to identify where targeted PT improvements can substitute for part of the required new service placement.`

### Main Planning Takeaway

Для повышения `PT-based accessibility to polyclinics` нужно не просто размещать сервисы, а различать:

- места, где ограничитель — local pedestrian access to stops
- места, где ограничитель — morphology of the PT path
- места, где размещение сервиса само по себе не решает проблему without connectivity improvements
- места, где транспортное улучшение может уменьшить число новых объектов, нужных для target coverage

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
2. visual corridor inspection
3. dominant-pattern route subsets

## One-Line Thesis

`Street pattern` is used here primarily as a diagnostic description of `PT-path morphology`, with supporting evidence from `first-mile pedestrian failures` and weaker evidence for other failure mechanisms.

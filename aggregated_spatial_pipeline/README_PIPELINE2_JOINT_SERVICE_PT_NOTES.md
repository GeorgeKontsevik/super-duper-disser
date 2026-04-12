# Notes: Full Joint Service Placement + PT Pipeline

Этот файл не про команды запуска, а про соображения, решения и ограничения вокруг полного пайплайна

- `service placement`
- `transport target construction`
- `ConnectPT route generation`
- `graph integration`
- `accessibility recompute`
- `provision recompute`

Идея документа простая: выгрузить из головы в репозиторий, что именно мы имели в виду, где уже рабочая реализация, а где остаются смысловые и методические нюансы.

Практический README по запуску лежит отдельно:

- [README_PIPELINE2_JOINT_SERVICE_PT.md](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/README_PIPELINE2_JOINT_SERVICE_PT.md)

## Главная Идея

Нужный нам полный сценарий был не просто:

- посчитать placement;
- потом отдельно посчитать маршрут;
- потом отдельно посмотреть что-нибудь еще.

Нужна была именно связанная цепочка, где каждый следующий шаг реально потребляет смысловой результат предыдущего:

1. `solver_flp` решает, где открыть/расширить сервисы;
2. из этого решения извлекается, между какими частями города еще плохо добираются до тех сервисов, которые placement предполагает использовать;
3. этот сигнал превращается в transport target;
4. `ConnectPT` пытается предложить маршрут, который улучшает именно эту связность;
5. маршрут встраивается обратно в городской intermodal graph;
6. accessibility пересчитывается на этом обновленном графе;
7. provision пересчитывается уже на новой accessibility и уже с after-placement capacities.

Ключевая мысль тут в том, что bridge должен передавать не файл, а городской смысл:

- где именно у нас остался accessibility gap;
- к каким facilities этот gap относится;
- и какая транспортная связность должна улучшиться, чтобы этот gap сократился.

## Что Раньше Было Не Так

До этой сшивки было два почти независимых контура.

### 1. Placement Был Сам По Себе

`run_pipeline2_prepare_solver_inputs` мог считать exact placement, в том числе genetic placement, и сохранял after-placement outputs.

Но дальше этот результат не становился transport target для `ConnectPT`.

### 2. ConnectPT Был Сам По Себе

`run_pipeline2_accessibility_first` мог считать suffering blocks и запускать route generator, но `ConnectPT` строил свою OD из population / density / diversity / land use.

То есть транспортная задача была общей городской транспортной задачей, а не service-aware задачей.

### 3. Был Критичный Баг С Встройкой Маршрута

При пересчете intermodal graph удалялись старые ребра выбранной модальности и оставались только generated routes.

То есть сценарий был не:

- `existing bus + new bus route`

а по сути:

- `all other modes + only generated bus routes`

Для нашего кейса это было концептуально неправильно.

## Главные Архитектурные Решения

Ниже то, что было решено и уже отражено в коде.

### 1. Не Использовать Genetic Matrix Как Конечную Транспортную Цель

Внутренняя генетика в placement помогает solver-у через гипотетические улучшения матрицы достижимости.

Но мы не стали передавать эту "улучшенную матрицу" напрямую в `ConnectPT`.

Почему:

- она не является реальным маршрутом;
- она не объясняет, какие именно transport links нужно улучшить;
- это слишком непрозрачный bridge между двумя задачами;
- такую матрицу потом трудно интерпретировать как physical transport intervention.

Вместо этого выбрано другое решение:

- брать реальный after-placement assignment;
- брать остаточный unmet accessibility demand;
- превращать это в явную target OD.

Это проще интерпретировать и проще дебажить.

### 2. Bridge Должен Идти Через Assignment Links

Самый важный смысловой выход placement-а для транспорта не просто:

- `вот новые сервисы`

а:

- `вот от каких client blocks к каким facility blocks placement предполагает обслуживание`.

Поэтому в bridge используются:

- `blocks_solver_after.parquet`
- `assignment_links_after.csv`

Это дает связку:

- demand side
- assigned facility side
- remaining access problem side

Именно это уже можно переводить в transport target.

### 3. Transport Target Должен Быть Service-Aware OD

Было выбрано такое представление:

- origin = ближайший PT stop к client block;
- destination = ближайший PT stop к facility block;
- weight = remaining unmet accessibility demand этого client block.

Если у клиента несколько assignment links, вес делится между ними.

Это не идеальная транспортная модель, но это хороший и понятный first bridge:

- он читаем;
- он прозрачен;
- по нему можно смотреть конкретные пары;
- он дает возможность в явном виде подать `ConnectPT` целевую OD.

### 4. Generated Route Должен Добавляться Поверх Existing PT

Очень важное решение:

- старую сеть модальности сохраняем;
- generated routes добавляем поверх;
- recompute делаем на `existing + generated`.

Это соответствует смыслу реального сценария:

- мы не заменяем всю автобусную сеть одним новым маршрутом;
- мы добавляем intervention в уже существующую систему.

### 5. Provision После Маршрута Надо Считать На After-Placement Capacities

Еще один важный смысловой момент:

- если placement открыл или расширил сервисы,
- то recompute provision после маршрута нельзя делать на старых capacities.

Иначе получается логическая ошибка:

- транспорт уже считает на after-route graph,
- а сервисная часть как будто живет в до-placement мире.

Поэтому после маршрута provision должен считаться на:

- `optimized_capacity_total`

а не на baseline `capacity`.

## Что Сейчас Уже Работает

На текущий момент уже реализовано и проверено следующее.

### Solver -> ConnectPT Bridge

`run_pipeline2_accessibility_first --use-placement-outputs`:

- читает after-placement blocks;
- читает `assignment_links_after.csv`;
- строит `service_target_od/<modality>_service_target_od.csv`;
- передает ее в `ConnectPT` через `--od-matrix-path`.

### Additive Graph Integration

`run_route_generator_external` по умолчанию:

- не удаляет existing routes выбранной модальности;
- добавляет generated route поверх;
- помечает новые ребра как `is_generated=True`.

### Accessibility Recompute

После встройки generated route:

- пересчитывается `adj_matrix_time_min_union.parquet`;
- сохраняется preview с generated accessibility.

### Provision Recompute

После route step:

- provision пересчитывается на after-placement capacities;
- сохраняются `blocks_solver_after_routes.parquet` и `summary_after_routes.json`.

### Reuse Existing Route Summary

Для быстрых итераций можно:

- не гонять `ConnectPT` заново;
- а подхватывать уже готовый `connectpt_routes_generator/<modality>/summary.json`
- и отдельно делать `provision recompute`.

Это полезно для анализа и дебага.

## Что Уже Подтверждено На Реальном Городе

Smoke-проверка была сделана на:

- `vienna_austria`

Там было подтверждено:

- service-aware OD действительно строится и может быть ненулевой;
- `ConnectPT` реально берет `od_source=external`;
- generated route реально встраивается в graph как additive merge;
- старые `bus` edges не исчезают;
- accessibility matrix реально пересчитывается;
- provision after routes реально считается после placement и после route.

Важно: то, что эффект маршрута на provision оказался маленьким, не отменяет того, что сам pipeline уже функционально сшит.

## Почему Эффект Маршрута Может Быть Маленьким

Это отдельно важно проговорить, чтобы потом не путать "пайплайн не работает" с "маршрут не дал большого эффекта".

Небольшой эффект может означать:

- target OD слишком sparse;
- один маршрут недостаточен;
- route-length bounds слишком жесткие;
- cost function `ConnectPT` плохо совпадает с service objective;
- nearest-stop mapping слишком грубо переводит block-to-block проблему в stop-to-stop;
- улучшение реально помогает только части пар, а не всей проблемной зоне.

То есть маленький прирост не обязательно означает баг.
Это может быть уже содержательный результат.

## Важное Ограничение Текущей Постановки

Хотя pipeline теперь связан, это еще не "полный joint optimizer" в строгом смысле.

Почему:

- placement и transport пока все равно оптимизируются разными моделями;
- `ConnectPT` не минимизирует service-gap напрямую;
- bridge идет через OD-приближение, а не через единый общий objective;
- route generation не знает всей service-loss функции, а получает только proxy target.

Поэтому корректнее думать о текущей схеме как о:

- `sequential coupled optimization`

а не как о:

- `single end-to-end mathematically joint optimization`.

Это нормально для практического первого рабочего контура.

## Почему Sequential Coupling Все Равно Имеет Смысл

Даже без общего objective эта схема полезна, потому что:

- placement выдает осмысленное service-side решение;
- transport получает целевой сигнал не из абстрактной mobility demand, а из service accessibility problem;
- recompute происходит на реальном обновленном graph;
- финальная provision-метрика уже честно отражает combined intervention.

Это уже намного лучше, чем независимый placement и независимый route generation.

## Как Лучше Интерпретировать Генетику В Placement

Важно не переинтерпретировать genetic mode.

`placement-genetic` здесь нужно понимать прежде всего как:

- внутренний эвристический способ найти лучшее размещение сервисов при гипотетических локальных улучшениях доступности.

Но не как:

- готовый транспортный план,
- готовый маршрут,
- или готовую улучшенную физическую сеть.

Поэтому bridge должен идти не через "генетически улучшенную матрицу как истину", а через:

- after-placement assignments;
- оставшийся unmet demand;
- и explicit service-aware OD.

## Где Сейчас Наиболее Слабое Место

Самое слабое место теперь уже не в plumbing, а в качестве transport objective.

То есть:

- мост от placement к transport уже есть;
- graph integration уже правильная;
- recompute уже правильный;
- а вот качество маршрута относительно service problem может быть слабым.

Причина в том, что `ConnectPT` остается general-purpose route generator с собственной cost function.

Значит, если хотеть большего эффекта, скорее всего надо улучшать не plumbing, а одно из:

- target OD construction;
- weighting scheme;
- route length policy;
- число маршрутов;
- смешивание service-target с baseline OD;
- или сам objective route generator.

## Идея Про Смешанную Целевую OD

Одно из естественных направлений дальше:

- не кормить `ConnectPT` только service-target OD,
- а смешивать ее с baseline transport OD.

Например:

- `OD_final = (1 - lambda) * OD_baseline + lambda * OD_service_target`

Зачем это может быть нужно:

- маршруты станут менее "узко специальными";
- может стать проще получить физически более естественные route candidates;
- при этом service problem все равно останется в objective.

Это пока идея, не реализованная в коде.

## Идея Про Улучшение Block -> Stop Mapping

Сейчас block-to-stop mapping простой:

- block representative point
- nearest stop

Это нормально для первого рабочего контура, но у этого есть ограничения:

- несколько block problems могут collapse-нуться в один stop pair;
- nearest stop не всегда соответствует реальному access path;
- крупные блоки и сложные пересадки могут искажать transport target.

Если когда-нибудь понадобится точнее:

- можно переходить к многоточечному mapping;
- или к распределению веса по нескольким ближайшим stops;
- или к walk-time-aware mapping.

Пока это не обязательная проблема, но про нее важно помнить.

## Почему Важно Смотреть Не Только На Total Gap

На route experiments важно смотреть не только на:

- суммарный `accessibility_gap_delta`

но и на:

- какие конкретно blocks улучшились;
- какие ухудшились;
- поменялись ли top suffering blocks;
- насколько route действительно проходит около целевых stop pairs;
- не появилось ли перераспределение проблемы вместо ее общего решения.

Один и тот же `delta` может скрывать очень разные пространственные эффекты.

## Почему Нужны PNG И Прямой Осмотр Артефактов

Для этого pipeline особенно опасно верить только логам.

Надо смотреть:

- `summary_after.json`
- `bus_service_target_od_summary.json`
- `connectpt_routes_generator/.../summary.json`
- `summary_after_routes.json`
- PNG в `preview_png/all_together`

Потому что здесь легко получить сценарий, который:

- технически отработал,
- но смыслово делает не то, что ожидалось.

## Что Стоит Помнить Перед Большими Прогонами

Перед серьезным batch-run полезно держать в голове следующее.

### 1. Сначала Проверять На Одном Городе

Если меняется хоть что-то из:

- target construction;
- route objective;
- graph integration;
- recompute logic;

сначала лучше смотреть один город и реальные артефакты.

### 2. Не Считать "Успешный Exit Code" Доказательством Правильности

В этом пайплайне это особенно неверно.

Нужна проверка:

- external OD реально использована;
- merge mode правильный;
- matrix реально пересчитана;
- after-route provision реально считается на after-placement state.

### 3. Не Смешивать Интерпретации

Важно различать:

- pipeline functional correctness;
- quality of transport solution;
- quality of service outcome.

Это три разные вещи.

### 4. Лучше Явно Фиксировать Режим Эксперимента

Например:

- exact / non-genetic
- exact / genetic
- allow existing expansion или нет
- 1 route / multiple routes
- pure service-target OD / mixed OD

Иначе потом трудно интерпретировать результаты.

## Текущее Рабочее Ментальное Представление

Самый полезный способ мыслить про текущую схему:

- placement решает, где сервисная система должна быть;
- transport решает, как немного помочь ей связностью;
- graph recompute проверяет, что это значит для реальной достижимости;
- final provision recompute измеряет combined effect.

Это и есть текущий смысл полного pipeline.

## Если Когда-нибудь Делать Следующий Уровень

Следующий уровень развития, если он понадобится, скорее всего будет не в plumbing, а в optimization logic:

- более сильный transport objective под service loss;
- mixed OD;
- multi-route scenario;
- direct scoring по final provision delta;
- или iterative loop:
  - placement
  - route
  - recompute
  - placement again

Но это уже следующий исследовательский слой, а не обязательная часть текущего рабочего контура.

# Street Pattern Experiments

## Быстрый запуск

Из корня репозитория:

```bash
cd /Users/gk/Code/super-duper-disser && \
PYTHONPATH=/Users/gk/Code/super-duper-disser \
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/.cache/mpl-street-pattern \
.venv/bin/python segregation-by-design-experiments/run_street_pattern_city.py \
  --place "Portland, Oregon, USA" \
  --buffer-m 5000 \
  --grid-step 500 \
  --road-source auto \
  --map-coloring multivariate \
  --device cpu \
  --output aggregated_spatial_pipeline/outputs/joint_inputs_storyline_b5000_g500/portland_oregon_usa/street_pattern_b5000_g500/portland_oregon_usa_summary.json
```

Серийный прогон по storyline-списку (единые `buffer=5000`, `step=500`):

```bash
cd /Users/gk/Code/super-duper-disser && \
XDG_CACHE_HOME=/Users/gk/Code/super-duper-disser/.cache/xdg-storyline \
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/.cache/mpl-street-pattern-b5000-g500 \
bash segregation-by-design-experiments/run_storyline_street_pattern_uniform_b5000_g500.sh
```

## North Star

Это главный концептуальный ориентир для всех экспериментов в этой папке:

- Города исторически отличаются по street pattern.
- Эти паттерны не mode-neutral.
- Из-за этого доступность по кварталам системно различается.
- Значит часть transport inequality буквально встроена в urban form.
- Именно это и есть `segregated by design`.

И связанный с этим второй тезис:

- `provision and accessibility cannot be understood without context`
- потому что сами услуги и транспорт работают не в пустоте, а внутри уже заданной морфологической среды.

Все новые эксперименты, картинки и формулировки здесь стоит проверять относительно этого north star. Если шаг уводит слишком далеко в техническую механику и перестает работать на этот тезис, его лучше считать вторичным или выносить в appendix/mechanism layer.

Скрипт [run_street_pattern_city.py](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/run_street_pattern_city.py) умеет:

- брать центр города через `osmapi` из OSM relation;
- автоматически выбирать источник дорог: локальный country roads файл или OSM;
- классифицировать ячейки уличного паттерна;
- сохранять результаты по папкам города;
- по `--year` тянуть исторический snapshot OSM;
- по `--compare-year` сравнивать два года и сохранять карту изменений.

Визуализация `map_<map_coloring>.png` поддерживает режимы через `--map-coloring`:

- `multivariate` (по умолчанию): цвет каждой ячейки — взвешенная смесь базовых цветов всех классов по вероятностям модели;
- `top1`: классическая окраска по одному победившему (`top-1`) классу.
- `vba`: каноничный PySAL value-by-alpha choropleth с отдельной `VBA`-легендой.

Для пересчета "с нуля" можно добавить `--no-cache` (отключает использование OSMnx-кэша в этом запуске).

Для single-city запуска используется отдельный pickle-кэш (`roads/subgraphs/dataset/predictions`) в:

- `--cache-dir` (по умолчанию `outputs/cache/city_single`).

Этот кэш не зависит от `--map-coloring`, поэтому можно переключать визуализацию (`top1`/`multivariate`/`vba`) без повторной классификации.

По умолчанию используется `--road-source auto`:

- `Canada` -> локальный Canada roads GeoPackage, если он есть;
- `USA` / `US` / `United States` -> локальный USA roads GeoPackage, если он есть;
- остальные страны -> скачивание дорог из OSM.

## Команды

Текущий OSM-граф для одного города:

```bash
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/cache/mpl \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/.venv/bin/python \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/run_street_pattern_city.py \
  --place 'Tianjin, China' \
  --device cpu \
  --output /Users/gk/Code/super-duper-disser/segregation-by-design-experiments/outputs/tianjin_predictions.json
```

Автоматический запуск для страны с локальным roads-файлом:

```bash
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/cache/mpl \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/.venv/bin/python \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/run_street_pattern_city.py \
  --place 'Montreal, Canada' \
  --road-source auto \
  --device cpu \
  --output /Users/gk/Code/super-duper-disser/segregation-by-design-experiments/outputs/montreal_auto.json
```

Тот же запуск с явным выбором режима карты:

```bash
# Классический top-1
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/cache/mpl \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/.venv/bin/python \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/run_street_pattern_city.py \
  --place 'Montreal, Canada' \
  --road-source auto \
  --map-coloring top1 \
  --device cpu \
  --output /Users/gk/Code/super-duper-disser/segregation-by-design-experiments/outputs/montreal_top1.json

# Новый multivariate
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/cache/mpl \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/.venv/bin/python \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/run_street_pattern_city.py \
  --place 'Montreal, Canada' \
  --road-source auto \
  --map-coloring multivariate \
  --device cpu \
  --output /Users/gk/Code/super-duper-disser/segregation-by-design-experiments/outputs/montreal_multivariate.json

# Каноничный value-by-alpha (PySAL)
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/cache/mpl \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/.venv/bin/python \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/run_street_pattern_city.py \
  --place 'Montreal, Canada' \
  --road-source auto \
  --map-coloring vba \
  --device cpu \
  --output /Users/gk/Code/super-duper-disser/segregation-by-design-experiments/outputs/montreal_vba.json
```

Полный пересчет без кэша:

```bash
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/cache/mpl \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/.venv/bin/python \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/run_street_pattern_city.py \
  --place 'Montreal, Canada' \
  --road-source auto \
  --map-coloring multivariate \
  --no-cache \
  --device cpu \
  --output /Users/gk/Code/super-duper-disser/segregation-by-design-experiments/outputs/montreal_multivariate_nocache.json
```

Исторический snapshot OSM за выбранный год:

```bash
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/cache/mpl \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/.venv/bin/python \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/run_street_pattern_city.py \
  --place 'Tianjin, China' \
  --year 2020 \
  --device cpu \
  --output /Users/gk/Code/super-duper-disser/segregation-by-design-experiments/outputs/tianjin_2020_predictions.json
```

Сравнение двух лет с картой изменений:

```bash
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/cache/mpl \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/.venv/bin/python \
/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/run_street_pattern_city.py \
  --place 'Tianjin, China' \
  --year 2020 \
  --compare-year 2024 \
  --device cpu \
  --output /Users/gk/Code/super-duper-disser/segregation-by-design-experiments/outputs/tianjin_2020_vs_2024.json
```

## Что сохраняется

Для обычного запуска:

- `outputs/<city_slug>/summary.json`
- `outputs/<city_slug>/roads.geojson`
- `outputs/<city_slug>/buffer.geojson`
- `outputs/<city_slug>/centre.geojson`
- `outputs/<city_slug>/predicted_cells.geojson`
- `outputs/<city_slug>/predicted_cells.csv`
- `outputs/<city_slug>/map_<map_coloring>.png`

В `predicted_cells.geojson` / `predicted_cells.csv` дополнительно сохраняются:

- `top2_class_name`, `top2_probability`
- `top3_class_name`, `top3_probability`
- `top3_signature`
- `multivariate_color`

Для запуска с `--year`:

- `outputs/<city_slug>/year_<year>/...`

Для запуска с `--year` и `--compare-year`:

- `outputs/<city_slug>/year_<year>/...`
- `outputs/<city_slug>/year_<compare_year>/...`
- `outputs/<city_slug>/comparison_<year>_vs_<compare_year>/summary.json`
- `outputs/<city_slug>/comparison_<year>_vs_<compare_year>/comparison_cells.geojson`
- `outputs/<city_slug>/comparison_<year>_vs_<compare_year>/comparison_cells.csv`
- `outputs/<city_slug>/comparison_<year>_vs_<compare_year>/<city_slug>_<year>_vs_<compare_year>.gpkg`
- `outputs/<city_slug>/comparison_<year>_vs_<compare_year>/map_changed.png`
- `outputs/<city_slug>/comparison_<year>_vs_<compare_year>/map_change_reason.png`
- `outputs/<city_slug>/comparison_<year>_vs_<compare_year>/roads_<year>.geojson`
- `outputs/<city_slug>/comparison_<year>_vs_<compare_year>/roads_<compare_year>.geojson`

В `comparison_cells.geojson` есть:

- классы по каждому году;
- `class_changed`;
- `change_reason`;
- изменение уверенности модели;
- изменение длины дорог и числа сегментов по ячейке;
- поле `why` с кратким текстовым объяснением.

В `.gpkg` для сравнения сохраняются QGIS-совместимые слои:

- `class_changed`
- `class_<year>`
- `class_<compare_year>`
- `graph_<year>`
- `graph_<compare_year>`

# Service And Accessibility Vs Street Pattern

## North Star

Этот подпроект читается не как набор отдельных корреляций, а как часть общей истории `segregated by design`:

- Города исторически отличаются по street pattern.
- Эти паттерны не mode-neutral.
- Из-за этого доступность по кварталам системно различается.
- Значит часть transport inequality буквально встроена в urban form.

И здесь особенно важен второй framing:

- `provision and accessibility cannot be understood without context`
- потому что сами услуги и транспорт работают не в пустоте, а внутри уже заданной морфологической среды.

Поэтому центральный вопрос этого подпроекта:

- как morphology и context меняют то, во что превращается provision с точки зрения access.

Отдельно важно:

- здесь `accessibility` не должна читаться только как среднее время "до всех";
- для социальной части главный смысловой слой — это доступность **до конкретной социалки**:
  - `school`
  - `polyclinic`
  - `hospital`
- то есть `assigned_service_time_*`, а не только global mean accessibility.

Этот подпроект запускает отдельные exploratory-эксперименты поверх уже собранных городских bundle'ов из `aggregated_spatial_pipeline/outputs/joint_inputs/<city_slug>`.

Сейчас он делает по каждому городу отдельно:

- перенос street-pattern probabilities и dominant class с grid cells на `blocks`;
- догрузку сервисов `hospital`, `polyclinic`, `school` по той же OSM/capacity-логике, что и `pipeline_2`;
- агрегацию `connectpt` stops по `blocks` для доступных модальностей (`bus`, `tram`, `trolleybus`, если есть в bundle);
- пересчет block-to-block accessibility mean time с текущего `intermodal_graph_iduedu/graph.pkl`;
- обычные корреляции `service/accessibility vs street-pattern probabilities`;
- обычные корреляции `stops vs street-pattern probabilities`;
- пространственные bivariate Moran correlations `service/accessibility vs street-pattern probabilities`;
- пространственные bivariate Moran correlations `stops vs street-pattern probabilities`;
- group summaries и PNG по dominant street-pattern class.

Если запускается `2+` города сразу, раннер дополнительно делает общий cross-city pilot:

- distribution tests по street-pattern probabilities между городами;
- variance decomposition `between-city vs within-city`;
- модели `city only`, `street-pattern only`, `city + street-pattern`;
- between/within decomposition для response vs street-pattern;
- отдельные cross-city PNG и CSV.

## Где лежит

- код: [service_accessibility_street_pattern](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/service_accessibility_street_pattern)
- раннер: [run_experiments.py](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/service_accessibility_street_pattern/run_experiments.py)
- outputs: [outputs](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/service_accessibility_street_pattern/outputs)

## Как запускать

Из корня репы:

```bash
cd /Users/gk/Code/super-duper-disser

PYTHONPATH=/Users/gk/Code/super-duper-disser \
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/.cache/mpl-sbd-service-accessibility \
.venv/bin/python segregation-by-design-experiments/service_accessibility_street_pattern/run_experiments.py \
  --cities warsaw_poland berlin_germany
```

Если нужен только один город:

```bash
cd /Users/gk/Code/super-duper-disser

PYTHONPATH=/Users/gk/Code/super-duper-disser \
MPLCONFIGDIR=/Users/gk/Code/super-duper-disser/.cache/mpl-sbd-service-accessibility \
.venv/bin/python segregation-by-design-experiments/service_accessibility_street_pattern/run_experiments.py \
  --cities warsaw_poland
```

## Что смотреть

Для каждого города появляется папка:

- [outputs/warsaw_poland](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/service_accessibility_street_pattern/outputs/warsaw_poland)
- [outputs/berlin_germany](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/service_accessibility_street_pattern/outputs/berlin_germany)

Если городов несколько, появляется еще:

- [outputs/_cross_city](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/service_accessibility_street_pattern/outputs/_cross_city)
- [outputs/_transport_pattern_story](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/service_accessibility_street_pattern/outputs/_transport_pattern_story)
- [outputs/_social_access_pattern_story](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/service_accessibility_street_pattern/outputs/_social_access_pattern_story)

Главные файлы там:

- `prepared/blocks_experiment.parquet` — одна строка на `block` со street-pattern, сервисами и accessibility;
- `stats/service_correlations.csv` — обычные Spearman correlations по сервисам;
- `stats/service_spatial_bivariate_moran.csv` — пространственные bivariate Moran correlations по сервисам;
- `stats/stop_correlations.csv` — обычные Spearman correlations по остановкам;
- `stats/stop_spatial_bivariate_moran.csv` — пространственные bivariate Moran correlations по остановкам;
- `stats/accessibility_correlations.csv` — обычные correlations для accessibility;
- `stats/accessibility_spatial_bivariate_moran.csv` — spatial correlations для accessibility;
- `stats/dominant_class_summary.csv` — средние/доли по dominant street-pattern class;
- `preview_png/*.png` — карты и summary plots.

Главные cross-city файлы:

- `stats/distribution_tests.csv` — различия распределений street-pattern feature между городами;
- `stats/street_pattern_variance_decomposition.csv` — between-city vs within-city variance;
- `stats/response_model_summary.csv` — `city / pattern / city+pattern` fit summary;
- `stats/response_model_full_coefficients.csv` — коэффициенты моделей;
- `stats/between_within_models.csv` — between/within decomposition по response и feature;
- `preview_png/10_street_pattern_feature_distributions_by_city.png`
- `preview_png/11_dominant_class_composition_by_city.png`
- `preview_png/12_street_pattern_variance_decomposition.png`
- `preview_png/13_city_vs_pattern_model_fit.png`
- `preview_png/14_city_effect_reduction_after_pattern.png`
- `preview_png/15_between_city_effects_heatmap.png`
- `preview_png/16_within_city_effects_heatmap.png`

Для destination-specific story по социальной доступности:

- `_social_access_pattern_story/stats/response_availability.csv`
- `_social_access_pattern_story/stats/fixed_effect_model_summary.csv`
- `_social_access_pattern_story/stats/fixed_effect_coefficients.csv`
- `_social_access_pattern_story/preview_png/*_school_*`
- `_social_access_pattern_story/preview_png/*_polyclinic_*`
- `_social_access_pattern_story/preview_png/*_hospital_*`

Дополнительно раннер рисует локальные bivariate Moran maps для strongest street-pattern feature по каждому response:

- `moran_local_service_has_*.png`
- `moran_local_service_capacity_*.png`
- `moran_local_stop_has_*.png`
- `moran_local_stop_count_*.png`
- `moran_local_accessibility_time_mean_pt.png`

## Как читать эксперимент

Идея здесь не в том, чтобы доказать причинность, а в том, чтобы на одном и том же городе показать:

- как presence/capacity сервисов связаны с разными street-pattern classes;
- как accessibility distribution различается между `blocks`, где доминируют разные street-pattern classes;
- и есть ли пространственная связь не только по обычной корреляции, но и по bivariate Moran.

Практически это читается так:

1. Смотри `01_street_pattern_dominant_class.png`.
2. Смотри `04_accessibility_by_dominant_class.png`.
3. Смотри `02_service_presence_by_dominant_class.png` и `03_service_capacity_by_dominant_class.png`.
4. Смотри `03b_stop_presence_by_dominant_class.png` и `03c_stop_count_by_dominant_class.png`.
4. Потом смотри heatmap'ы Spearman и bivariate Moran.

## Замечания

- Этот раннер сам докачивает сервисы, если их еще нет в output папке подпроекта.
- Если сервисная точка не пересекает ни один квартал по `intersects`, она дополнительно привязывается к ближайшему кварталу (`nearest polygon`) и включается в `service_count_*` / `service_capacity_*`.
- Accessibility пересчитывается заново с `intermodal_graph_iduedu/graph.pkl`, потому что в phase-1 bundle `accessibility_time_mean` может быть пустым или нулевым.
- Для service-метрик используются все `blocks`.
- Для accessibility используются только `blocks` с `population > 0`.

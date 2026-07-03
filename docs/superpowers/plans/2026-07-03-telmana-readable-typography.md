# Telmana Readable Typography Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Перерендерить три иллюстрации квартала Тельмана с Arial и крупным читаемым текстом.

**Architecture:** Сохранить существующий односценарный скрипт и его расчётные входы. Изменить только глобальный шрифт, размеры текстовых элементов, интервалы карточек и пространство под легенду, затем перегенерировать и визуально проверить PNG.

**Tech Stack:** Python, Matplotlib, GeoPandas, Pillow.

---

### Task 1: Проверка типографики

**Files:**
- Create: `tests/test_telmana_render_typography.py`
- Modify: `scripts/render_telmana_connector_visual_matrix_v2.py`

- [ ] **Step 1: Написать падающий тест**

Проверить, что скрипт задаёт `font.family = Arial`, использует карточный шрифт не меньше 14 pt и легенду не меньше 12 pt.

- [ ] **Step 2: Запустить тест и подтвердить падение**

Run: `pytest -q tests/test_telmana_render_typography.py`
Expected: FAIL на текущих размерах 6.7–8 pt и отсутствии Arial.

- [ ] **Step 3: Внести минимальную правку**

Задать Arial через `plt.rcParams`, увеличить размеры всего текста, карточек, шкалы и легенды; убрать `fontfamily="serif"`; скорректировать вертикальные позиции текста карточки и нижнее поле легенды.

- [ ] **Step 4: Запустить тест повторно**

Run: `pytest -q tests/test_telmana_render_typography.py`
Expected: PASS.

### Task 2: Рендер и проверка артефактов

**Files:**
- Regenerate: `aggregated_spatial_pipeline/outputs/experiments_spb_telmana_connector_clean_4x2_20260620/visual_scenario_maps_square_connector_v2/*.png`
- Update: `itmo-phd-thesis-template-en/images/ch4/optimal_local/telmana_connector_*.png`

- [ ] **Step 1: Запустить скрипт**

Run: `python scripts/render_telmana_connector_visual_matrix_v2.py`
Expected: три пути PNG без traceback.

- [ ] **Step 2: Проверить результаты программно**

Проверить Pillow, что три PNG открываются и имеют ненулевой размер; проверить восемь записей `scenario_metrics` в манифесте.

- [ ] **Step 3: Проверить результаты визуально**

Открыть все три PNG и убедиться, что Arial применяется, карточки читаемы, текст не пересекается и не обрезан.

- [ ] **Step 4: Обновить копии диссертации**

Скопировать три проверенных PNG в `itmo-phd-thesis-template-en/images/ch4/optimal_local/` с существующими именами.

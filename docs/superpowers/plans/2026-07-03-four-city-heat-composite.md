# Four-city Heat Composite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one high-resolution 4×2 heat-experiment figure with large typography, two shared legends, and service points in every lower panel.

**Architecture:** Add a focused composite renderer beside the existing single-city renderer and reuse its data-loading and map-layer helpers. Keep all eight panels in one Matplotlib figure so shared legends and typography are native rather than raster post-processing.

**Tech Stack:** Python, Matplotlib, GeoPandas, pandas, NetworkX, pytest

---

### Task 1: Define and test shared composite semantics

**Files:**
- Create: `tests/test_render_four_city_heat_composite.py`
- Create: `scripts/render_four_city_heat_composite.py`

- [ ] **Step 1: Write the failing tests**

Test that city order is Gothenburg, Hrodna, Graz, Innsbruck; the top and bottom legend builders return the expected categories; and the bottom legend contains the selected service label.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_render_four_city_heat_composite.py`

Expected: collection fails because `scripts.render_four_city_heat_composite` does not exist.

- [ ] **Step 3: Add the minimal constants and legend builders**

Create the renderer with `CITY_ORDER`, type-size constants, `top_legend_handles()`, and `bottom_legend_handles(service)` using the existing UTCI labels/colors and the existing five delta-bin colors.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest -q tests/test_render_four_city_heat_composite.py`

Expected: all tests pass.

### Task 2: Render the native 4×2 figure

**Files:**
- Modify: `scripts/render_four_city_heat_composite.py`
- Modify: `tests/test_render_four_city_heat_composite.py`

- [ ] **Step 1: Write a failing layout test**

Test a lightweight figure factory and assert it creates eight map axes plus two dedicated shared-legend axes, with no per-panel legends.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_render_four_city_heat_composite.py`

Expected: failure because the figure factory is missing.

- [ ] **Step 3: Implement the composite renderer**

Reuse `_load_utci_edges`, `_load_context_layers`, `_background_edges`, `_walk_subgraph`, `_living_points`, and existing project paths. For each city, load the boundary, UTCI edges, water, buildings, walk network, living-building deltas, and selected service parquet. Draw the four UTCI panels above four delta panels, overlay representative service points with a black star marker, create only the two shared legends, and save `four_city_heat_composite_<service>.png` beneath the heat experiment output root.

- [ ] **Step 4: Run unit and syntax checks**

Run: `pytest -q tests/test_render_four_city_heat_composite.py && python -m py_compile scripts/render_four_city_heat_composite.py`

Expected: all tests pass and compilation exits 0.

### Task 3: Generate and inspect the artifact

**Files:**
- Output: `thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/four_city_heat_composite_polyclinic.png`

- [ ] **Step 1: Run the renderer**

Run: `python scripts/render_four_city_heat_composite.py --service polyclinic`

Expected: exit 0 and a printed output path.

- [ ] **Step 2: Verify the artifact structurally**

Open the PNG with Pillow and confirm nonzero dimensions and a file size consistent with a high-resolution map. Inspect logged per-city counts for UTCI edges, delta buildings, and services; all must be positive.

- [ ] **Step 3: Verify the artifact visually**

Render/view the final PNG directly. Confirm correct city order, eight populated panels, substantially enlarged titles and legend text, exactly one legend between/under each row, visible service stars in each lower panel, unclipped labels, and visible contextual layers.

- [ ] **Step 4: Commit the tested implementation**

Stage only the renderer, its test, and the plan; preserve unrelated worktree changes.

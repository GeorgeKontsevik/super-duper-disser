# Liberia Two-City Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a 4:3 Liberia accessibility heatmap with only small-city and large-city destinations, five crops, a shared fitted color scale, and a labeled weekly X axis.

**Architecture:** Add one focused renderer that imports the existing data access, aggregation, class colors, labels, and `plot_heatmap` function. Query the existing Liberia scenario, select the two city destination types, and create a compact Matplotlib layout without changing the general-purpose renderer.

**Tech Stack:** Python, pandas, psycopg, Matplotlib

---

### Task 1: Render the compact Liberia figure

**Files:**
- Create: `equatorial/scripts/render_lbr_two_city_heatmap.py`
- Create: `equatorial/outputs/astar_accessibility_weekly/lbr_two_city_heatmap/LBR_two_city_accessibility_heatmap_4x3.png`

- [ ] **Step 1: Verify the expected output is absent or has the wrong dimensions**

Use Pillow to assert that the target PNG exists at the selected 4:3 pixel dimensions. Expected before implementation: assertion failure.

- [ ] **Step 2: Implement the minimal renderer**

Import existing helpers from `render_weekly_astar_accessibility_heatmaps.py`; fetch Liberia for scenario `weekly_sum_penalty_v1` and origin scope `cluster_connected_allclusters_10small_3large_3ports_3airports`; render only `city_5_100k` and `city_100k_plus` on a 12×9 inch figure at 200 dpi. Keep all available crops, share X, show monthly week-start labels and `начало недели` on the bottom axis, and reserve a fitted right column for the shared six-class colorbar.

- [ ] **Step 3: Run the renderer**

Run: `uv run --project equatorial python equatorial/scripts/render_lbr_two_city_heatmap.py`

Expected: the output path and summary counts are printed; exit status is zero.

- [ ] **Step 4: Verify the produced artifact**

Assert the PNG is 2400×1800 pixels, inspect it directly, and confirm two panels, five crop rows, visible week labels/X title, and an unclipped colorbar.

- [ ] **Step 5: Check touched code**

Run `python -m py_compile` for the renderer and `git diff --check`. Inspect the output file rather than treating process success as correctness.

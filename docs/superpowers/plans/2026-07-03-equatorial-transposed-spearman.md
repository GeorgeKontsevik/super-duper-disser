# Equatorial Transposed Spearman Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the crop Spearman matrix as a readable 5×3 heatmap on a 4:3 canvas, with the sign of the unpaved-road coefficient inverted.

**Architecture:** Add a focused renderer that reads the saved `crop_stat_summary.csv` artifact. It selects the three existing rho/support column pairs, negates only the unpaved-road rho column, masks unsupported cells, and renders the transposed crop-by-factor matrix.

**Tech Stack:** Python, pandas, NumPy, Matplotlib

---

### Task 1: Render the transposed matrix

**Files:**
- Create: `equatorial/scripts/render_crop_spearman_transposed_ru.py`
- Create: `itmo-phd-thesis-template-en/images/ch4/crop_spearman_transposed_4x3_ru.png`

- [ ] **Step 1: Verify the target output is absent**

Assert that the target 2400×1800 PNG exists. Expected before implementation: failure.

- [ ] **Step 2: Implement the CSV-backed renderer**

Read the structural experiment `crop_stat_summary.csv`, order the five crops, build the 5×3 rho matrix, negate `rho_actual_unpaved_time_share`, and mask cells whose matching `_supported` field is false. Render with `RdBu_r`, fixed limits −1…1, horizontal multiline factor labels, numeric annotations, gray unsupported cells, and a fitted `ρ Спирмена` colorbar.

- [ ] **Step 3: Run and verify**

Run the renderer, assert a 2400×1800 output, and assert the displayed supported values round to `0.71, 0.38, 0.77, 0.52, 0.76, 0.38, 0.74, 0.50` in crop-row order.

- [ ] **Step 4: Inspect the PNG**

Open the output and confirm 5 crop rows, 3 factor columns, eight annotated cells, gray unsupported cells, readable labels, and an unclipped colorbar.

- [ ] **Step 5: Verify code quality**

Run `python -m py_compile` and `git diff --check`.

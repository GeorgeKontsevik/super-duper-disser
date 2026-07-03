# Equatorial Square Composite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dense 1:1 PNG from the five existing Russian analytical panels.

**Architecture:** A small Pillow compositor reads the already rendered high-resolution source figures, crops the required panels by stable source-relative boxes, fits them into a 2400×2400 layout, and writes one presentation-ready PNG. It does not recompute or modify analytical values.

**Tech Stack:** Python 3, Pillow, existing PNG artifacts.

---

### Task 1: Add the square compositor

**Files:**
- Create: `scripts/compose_equatorial_square_results.py`
- Output: `itmo-phd-thesis-template-en/images/ch4/equatorial_results_square.png`

- [ ] **Step 1: Validate the five logical inputs**

The script must fail clearly when any of the precipitation/degradation panel,
LBR heatmap, temporal plot, or Spearman plot sources are absent.

- [ ] **Step 2: Extract content panels**

Split the precipitation/degradation source into two map panels. Crop the first
two destination heatmaps and their shared color scale from the LBR heatmap.
Trim only outer white margins from the temporal and Spearman plots.

- [ ] **Step 3: Compose the square**

Create a 2400×2400 white canvas. Use a 45%/55% vertical split, 28 px outer
margin, and 18 px gaps. Place the two maps beside one another in the left 40%
of the upper group and the destination heatmaps in the remaining 60%. Place
the temporal panel in the left 60% of the lower group and Spearman in the
remaining 40%. Preserve aspect ratio and align each panel to its cell.

- [ ] **Step 4: Run and verify**

Run:

```bash
.venv/bin/python scripts/compose_equatorial_square_results.py
```

Expected: exit code 0 and a 2400×2400 RGB PNG. Open the PNG and verify that all
five panels, titles, legends, and color scales are visible without overlap.

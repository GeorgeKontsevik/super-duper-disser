# Liberia Maps Portrait Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-render the paired Liberia precipitation and road-degradation maps on a 3:4 portrait canvas with larger maps and smaller scales.

**Architecture:** Change only the Matplotlib layout in the existing `render_precip_grid` function. Preserve all database queries, masking, road classification, colors, values, manifest fields, and output destinations.

**Tech Stack:** Python, GeoPandas, Matplotlib, PostgreSQL/PostGIS

---

### Task 1: Recompose the paired map figure

**Files:**
- Modify: `equatorial/scripts/render_lbr_precip_grid_figure.py`
- Regenerate: `equatorial/outputs/astar_accessibility_weekly/paper_lbr_precip_grid/lbr_precip_grid_week_2024_08_19.png`
- Regenerate: `itmo-phd-thesis-template-en/images/ch4/lbr_precip_grid_week_2024_08_19.png`
- Verify: `equatorial/outputs/astar_accessibility_weekly/paper_lbr_precip_grid/manifest.json`

- [ ] **Step 1: Verify the current output is not 3:4**

Use Pillow to assert the existing output is 1620×2160 pixels. Expected before implementation: failure.

- [ ] **Step 2: Implement the portrait layout**

Set the figure to 9×12 inches at the existing 180 dpi. Place precipitation above road degradation. Allocate nearly full-width map axes and compact horizontal colorbar axes directly below each map. Retain equal geographic aspect and existing symbology.

- [ ] **Step 3: Re-render from the project database**

Start the PostgreSQL 18 data cluster temporarily and run `uv run python equatorial/scripts/render_lbr_precip_grid_figure.py --week-start 2024-08-19`. Stop the cluster after rendering.

- [ ] **Step 4: Verify artifacts**

Assert both PNG copies are 1620×2160. Read `manifest.json` and confirm week `2024-08-19`, positive ERA5 point count, positive road count, and output paths. Open the PNG and inspect map size, order, scale fit, clipping, and whitespace.

- [ ] **Step 5: Verify code quality**

Run `python -m py_compile` and `git diff --check`; inspect the exact touched diff.

# Arctic 4:3 Sankey Panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-render the four selected Arctic Sankey diagrams as 4:3 PNGs and assemble a 4:3 review sheet.

**Architecture:** Extend the existing renderer with an explicit list of selected region/service pairs. Reuse the current pickle inputs, Russian labels, and `create_clean_sankey`; write only the four square outputs plus one contact sheet.

**Tech Stack:** Python, Plotly/Kaleido, Pillow

---

### Task 1: Add selected square rendering

**Files:**
- Modify: `scripts/render_arctic_sankey_panels_ru.py`

- [ ] **Step 1: Record the expected outputs before implementation**

Run a Python assertion that expects the four `*_square.png` files and the contact sheet at their intended dimensions. Expected result: failure because the files do not exist yet.

- [ ] **Step 2: Implement the minimal square render path**

Render each selected pair through `create_clean_sankey` and call `figure.write_image(..., width=1200, height=900, scale=2)`. Assemble the rendered files in a `2 × 2` Pillow contact sheet without resizing them out of 4:3 proportions.

- [ ] **Step 3: Run the renderer**

Run: `python scripts/render_arctic_sankey_panels_ru.py`

Expected: four square image paths and one contact-sheet path are printed without errors.

- [ ] **Step 4: Verify dimensions and contents**

Use Pillow assertions to confirm each individual output is `2400 × 1800` physical pixels (Plotly scale 2) and the contact sheet is `4800 × 3600`. Open the contact sheet and inspect titles, labels, flows, clipping, and the intended panel selection.

- [ ] **Step 5: Review the diff**

Run `git diff --check` and inspect `git diff -- scripts/render_arctic_sankey_panels_ru.py` so unrelated user changes remain untouched.

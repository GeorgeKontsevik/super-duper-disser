# Tikhevich Russian Map Pair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose the two original Tikhevich QGIS maps into one Russian-language thesis figure.

**Architecture:** A single Pillow script crops and scales the two source JPEGs, covers the English legend areas, and draws Russian legends and panel captions. A small integration test runs the renderer and verifies the PNG contract.

**Tech Stack:** Python 3, Pillow, unittest

---

### Task 1: Define and implement the renderer

**Files:**
- Create: `tests/test_render_tikhevich_russian_map_pair.py`
- Create: `scripts/render_tikhevich_russian_map_pair.py`

- [ ] Write a failing test that runs the missing script and expects an RGB PNG sized 2400 × 1500.
- [ ] Run the test and confirm it fails because the script is missing.
- [ ] Implement the Pillow composition using the two approved source images and exact Russian labels from the design.
- [ ] Run the test and confirm it passes.

### Task 2: Verify the artifact

**Files:**
- Inspect: `itmo-phd-thesis-template-en/images/ch4/optimal_local/tikhevich_russian_map_pair.png`

- [ ] Compile the script, run `git diff --check`, and inspect PNG dimensions and mode.
- [ ] Open the PNG and confirm both maps are complete, Russian text is readable, English legends are absent, and color intervals match the source.

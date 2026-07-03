# Heat Route Map Style Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the approved endpoint, PT-line, and context styling to every heat route-map renderer and regenerate verified PNGs.

**Architecture:** Add a small shared set of Matplotlib style constants in the existing heat route renderer, then import those values in sibling route renderers. Keep route reconstruction and data loading unchanged; only plotting and legends change.

**Tech Stack:** Python, Matplotlib, GeoPandas, pytest, Pillow

---

### Task 1: Lock the shared style contract

**Files:**
- Create: `tests/test_heat_route_map_style.py`
- Modify: `scripts/render_heat_service_city_pairs_and_routes.py`

- [ ] **Step 1: Write a failing test**

Create a test importing the shared colors, star marker, marker sizes, and PT dash tuple and asserting: green home star, blue service star, burgundy baseline PT, green heat PT, both endpoint sizes at least 100, and `(0, (2, 1.5))` frequent dashes.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_heat_route_map_style.py`
Expected: FAIL because the shared style constants do not exist.

- [ ] **Step 3: Add the minimum constants and use them in plotting and legends**

Define the approved values once in `render_heat_service_city_pairs_and_routes.py`; replace literal endpoint and PT styles in its heat route-map functions.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_heat_route_map_style.py`
Expected: PASS.

### Task 2: Apply the contract to sibling heat route maps

**Files:**
- Modify: `scripts/render_super_changed_route_overlays.py`
- Modify: `scripts/render_debrecen_heat_story_maps.py`
- Modify: `scripts/render_mode_switch_route_maps.py`
- Test: `tests/test_heat_route_map_style.py`

- [ ] **Step 1: Extend the test with source-level renderer checks**

Assert each renderer imports the shared style contract and does not retain orange home or black service marker literals in its route plotting function.

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_heat_route_map_style.py`
Expected: FAIL on the sibling renderers.

- [ ] **Step 3: Replace local marker/PT literals and strengthen context layers**

Import and use the shared constants. Use green/blue stars, frequent dashed PT, slightly darker and more opaque street/building/water context, and matching legends.

- [ ] **Step 4: Verify GREEN and compile**

Run: `pytest -q tests/test_heat_route_map_style.py && python -m py_compile scripts/render_heat_service_city_pairs_and_routes.py scripts/render_super_changed_route_overlays.py scripts/render_debrecen_heat_story_maps.py scripts/render_mode_switch_route_maps.py`
Expected: PASS with no compile errors.

### Task 3: Regenerate and inspect outputs

**Files:**
- Modify: `thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/**/*.png`

- [ ] **Step 1: Run the existing render commands for all available city heat route maps**

Use each renderer's CLI with its existing city/service set; do not alter data inputs.

- [ ] **Step 2: Inspect artifacts programmatically**

Open the output PNGs with Pillow, confirm non-zero dimensions, recent modification times, and non-empty files.

- [ ] **Step 3: Inspect representative PNGs visually**

Open at least one output from every affected renderer and confirm green home stars, blue service stars, burgundy/green frequent-dashed PT lines, clearer context, and accurate legends.

- [ ] **Step 4: Run final verification**

Run the focused pytest and compile command again after rendering. Expected: PASS.

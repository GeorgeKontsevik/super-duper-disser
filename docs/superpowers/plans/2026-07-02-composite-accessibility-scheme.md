# Composite Accessibility Scheme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and visually verify one Russian-language PNG explaining composite accessibility, external impacts, interventions, and the four dissertation experiments.

**Architecture:** Add one standalone Matplotlib renderer following the repository's existing dissertation-figure scripts. The script contains the fixed labels and geometry, writes directly to the dissertation images directory, and performs basic output assertions.

**Tech Stack:** Python, Matplotlib, pathlib

---

### Task 1: Render the scheme

**Files:**
- Create: `scripts/render_composite_accessibility_scheme.py`
- Create: `itmo-phd-thesis-template-en/Dissertation/composite_accessibility_external_environment.png`

- [ ] **Step 1: Implement one standalone renderer**

Create a wide horizontal figure with four stacked regions: external environment, composite path, intervention controls, and experiment cards. Reuse Matplotlib only; do not add dependencies.

- [ ] **Step 2: Add a minimal self-check**

After saving, assert that the PNG exists and is larger than 50 KB so an empty or failed render cannot be reported as complete.

- [ ] **Step 3: Run the renderer**

Run:

```bash
python scripts/render_composite_accessibility_scheme.py
```

Expected: prints the output path and image dimensions with exit code 0.

- [ ] **Step 4: Inspect the PNG directly**

Open the generated file at original resolution and verify that all labels are readable, arrows avoid text, and the four experimental blocks are distinct.

- [ ] **Step 5: Run source and artifact checks**

Run:

```bash
python -m py_compile scripts/render_composite_accessibility_scheme.py
test -s itmo-phd-thesis-template-en/Dissertation/composite_accessibility_external_environment.png
```

Expected: exit code 0 for both checks.

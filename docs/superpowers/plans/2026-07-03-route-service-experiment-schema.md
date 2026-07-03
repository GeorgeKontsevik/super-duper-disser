# Route Service Experiment Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a four-panel Russian-language PNG explaining the route-versus-service-placement experiment in the palette and style of the existing network decomposition figure.

**Architecture:** Add one static Pillow renderer with explicit coordinates and small drawing helpers. Add one standard-library test that runs the renderer and validates the produced PNG dimensions and representative palette colors; final acceptance also includes direct visual inspection.

**Tech Stack:** Python 3, Pillow, `unittest`, existing macOS Arial fonts

---

### Task 1: Define the render contract

**Files:**
- Create: `tests/test_render_route_service_experiment_schema.py`
- Create later: `scripts/render_route_service_experiment_schema.py`

- [ ] **Step 1: Write the failing integration test**

```python
import subprocess
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_route_service_experiment_schema.py"
OUTPUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "route_service_experiment_schema.png"


class RouteServiceExperimentSchemaTest(unittest.TestCase):
    def test_renderer_writes_expected_png(self):
        OUTPUT.unlink(missing_ok=True)
        subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)

        with Image.open(OUTPUT) as image:
            self.assertEqual(image.size, (2580, 980))
            self.assertEqual(image.mode, "RGB")
            colors = {rgb for _, rgb in image.getcolors(maxcolors=image.width * image.height)}

        for expected in {
            (255, 212, 212),  # demand fill
            (210, 238, 210),  # service fill
            (165, 105, 190),  # route
            (0, 140, 137),    # synthetic connection
        }:
            self.assertIn(expected, colors)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.test_render_route_service_experiment_schema -v`

Expected: `ERROR` because `scripts/render_route_service_experiment_schema.py` does not exist.

### Task 2: Implement the static renderer

**Files:**
- Create: `scripts/render_route_service_experiment_schema.py`
- Test: `tests/test_render_route_service_experiment_schema.py`

- [ ] **Step 1: Add the minimal renderer**

Implement a Pillow script that:

```python
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "route_service_experiment_schema.png"
W, H = 2580, 980
```

It must define only the drawing helpers needed by the static composition: `font`, `panel`, `node`, `edge`, `dashed_line`, `metric`, `draw_graph`, and `legend_item`. Draw the approved four panels with one shared graph topology, symbolic metrics `N₀`, `N₁`, `Nдор`, `Nмарш`, `Nобщ`, and `Nсерв`, and the exact palette from `scripts/render_part1_network_decomposition.py`.

- [ ] **Step 2: Run the test and verify GREEN**

Run: `python -m unittest tests.test_render_route_service_experiment_schema -v`

Expected: one passing test and a newly written `route_service_experiment_schema.png`.

- [ ] **Step 3: Check syntax and diff hygiene**

Run: `python -m py_compile scripts/render_route_service_experiment_schema.py tests/test_render_route_service_experiment_schema.py && git diff --check`

Expected: exit code 0 with no output.

### Task 3: Inspect the actual artifact

**Files:**
- Inspect: `itmo-phd-thesis-template-en/Dissertation/route_service_experiment_schema.png`

- [ ] **Step 1: Inspect file metadata**

Run:

```bash
python -c 'from PIL import Image; from pathlib import Path; p=Path("itmo-phd-thesis-template-en/Dissertation/route_service_experiment_schema.png"); im=Image.open(p); print(p.stat().st_size, im.size, im.mode)'
```

Expected: nonzero byte count, `(2580, 980)`, and `RGB`.

- [ ] **Step 2: Open the final PNG and visually verify it**

Inspect the image at original detail. Confirm all four panels, complete labels, shared topology, visible route/road/synthetic-link distinctions, complete legend, and no overlaps or clipping. If any defect is visible, adjust coordinates or font sizes and repeat Tasks 2 Step 2 through Task 3 Step 2.

- [ ] **Step 3: Run final verification**

Run:

```bash
python -m unittest tests.test_render_route_service_experiment_schema -v && python -m py_compile scripts/render_route_service_experiment_schema.py tests/test_render_route_service_experiment_schema.py && git diff --check
```

Expected: one passing test, successful compilation, and no diff errors.

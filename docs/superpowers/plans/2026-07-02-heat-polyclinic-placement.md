# Heat-Aware Polyclinic Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare baseline vs heat-aware optimal polyclinic placement across the existing batch of city bundles and measure whether heat increases required new placements or expansion.

**Architecture:** Reuse the existing placement pipeline instead of adding a new solver path. Baseline uses the existing city bundles under `aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs`, while heat uses the already prepared heat-weighted graph copies under `thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/heat_joint_inputs`. The same `run_pipeline2_prepare_solver_inputs` placement step runs on both roots, then a compact comparison script reads the produced `summary_after.json` and `blocks_solver_after.parquet` artifacts.

**Tech Stack:** Python, existing `aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs`, parquet/json outputs, shell batch execution.

---

### Task 1: Verify existing heat-ready inputs and baseline placement artifacts

**Files:**
- Inspect: `aggregated_spatial_pipeline/pipeline/run_pipeline2_prepare_solver_inputs.py`
- Inspect: `aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/*`
- Inspect: `thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/heat_joint_inputs/*`

- [ ] **Step 1: Verify that both roots contain matching city bundles**

Run:
```bash
cd /Users/gk/Code/super-duper-disser
python3 - <<'PY'
from pathlib import Path
base = Path('aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs')
heat = Path('thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/heat_joint_inputs')
base_c = sorted(p.name for p in base.iterdir() if p.is_dir())
heat_c = sorted(p.name for p in heat.iterdir() if p.is_dir())
print(len(base_c), len(heat_c), base_c == heat_c)
PY
```
Expected: same city count and `True` for the name comparison.

- [ ] **Step 2: Verify the placement CLI flags we need already exist**

Run:
```bash
cd /Users/gk/Code/super-duper-disser
rg -n "placement-exact|placement-allow-existing-expansion|placement-prefer-existing|placement-capacity-mode" aggregated_spatial_pipeline/pipeline/run_pipeline2_prepare_solver_inputs.py
```
Expected: the exact placement flags are present; no new CLI implementation needed.

### Task 2: Run baseline and heat-aware polyclinic placement

**Files:**
- Reuse: `aggregated_spatial_pipeline/pipeline/run_pipeline2_prepare_solver_inputs.py`
- Create output in-place under each city bundle's `pipeline_2/placement_exact*`

- [ ] **Step 1: Run baseline placement across all cities for `polyclinic`**

Run:
```bash
cd /Users/gk/Code/super-duper-disser
for CITY in aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/*; do
  echo "=== $(basename "$CITY") baseline polyclinic ==="
  PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
    --joint-input-dir "$CITY" \
    --services polyclinic \
    --placement-exact \
    --placement-allow-existing-expansion \
    --placement-prefer-existing \
    --placement-capacity-mode fixed_mean || break
done
```
Expected: each city writes `pipeline_2/placement_exact/polyclinic/summary_after.json` and `blocks_solver_after.parquet`.

- [ ] **Step 2: Run heat-aware placement across all cities for `polyclinic`**

Run:
```bash
cd /Users/gk/Code/super-duper-disser
for CITY in thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/heat_joint_inputs/*; do
  echo "=== $(basename "$CITY") heat polyclinic ==="
  PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
    --joint-input-dir "$CITY" \
    --services polyclinic \
    --placement-exact \
    --placement-allow-existing-expansion \
    --placement-prefer-existing \
    --placement-capacity-mode fixed_mean || break
done
```
Expected: heat city bundles also write `pipeline_2/placement_exact/polyclinic/summary_after.json` and `blocks_solver_after.parquet`.

### Task 3: Build a compact baseline-vs-heat comparison table

**Files:**
- Create: `scripts/compare_heat_vs_baseline_polyclinic_placement.py`
- Output: `thermal_access_pilot/outputs/heat_polyclinic_placement_comparison_hottest/summary.csv`

- [ ] **Step 1: Write the comparison script**

The script should, for each city, read:
- baseline root: `aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/<city>/pipeline_2/placement_exact/polyclinic/summary_after.json`
- heat root: `thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/heat_joint_inputs/<city>/pipeline_2/placement_exact/polyclinic/summary_after.json`
- baseline and heat `blocks_solver_after.parquet`

It should emit per city:
- `new_facilities_baseline`
- `new_facilities_heat`
- `delta_new_facilities`
- `optimized_capacity_total_baseline`
- `optimized_capacity_total_heat`
- `delta_optimized_capacity_total`
- `demand_without_after_total_baseline`
- `demand_without_after_total_heat`
- `delta_demand_without_after_total`
- `demand_left_after_total_baseline`
- `demand_left_after_total_heat`
- `delta_demand_left_after_total`

- [ ] **Step 2: Run the comparison script**

Run:
```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=$PWD .venv/bin/python scripts/compare_heat_vs_baseline_polyclinic_placement.py
```
Expected: `thermal_access_pilot/outputs/heat_polyclinic_placement_comparison_hottest/summary.csv` exists and has one row per city.

- [ ] **Step 3: Inspect the summary and call out cities where heat increases required placement**

Run:
```bash
cd /Users/gk/Code/super-duper-disser
python3 - <<'PY'
import pandas as pd
p = 'thermal_access_pilot/outputs/heat_polyclinic_placement_comparison_hottest/summary.csv'
df = pd.read_csv(p)
print(df[['city','delta_new_facilities','delta_optimized_capacity_total','delta_demand_without_after_total','delta_demand_left_after_total']].sort_values(['delta_new_facilities','delta_optimized_capacity_total'], ascending=False).head(30).to_string(index=False))
PY
```
Expected: a ranked table showing whether heat increases placement requirements.

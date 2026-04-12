# super-duper-disser

## Setup From Scratch

```bash
cd /Users/gk/Code
git clone --recurse-submodules https://github.com/GeorgeKontsevik/super-duper-disser.git super-duper-disser
cd /Users/gk/Code/super-duper-disser
chmod +x scripts/bootstrap_fresh_machine.sh
./scripts/bootstrap_fresh_machine.sh
```

What the script does:
- installs `uv` if missing
- tries to install `python3`, `curl`, and `git` automatically via available package manager (`brew`, `apt`, `dnf`, `yum`, `pacman`, `zypper`, `winget`, `choco`)
- initializes git submodules
- creates root orchestration env: `.venv`
- creates dedicated per-repo envs:
  - `blocksnet/.venv`
  - `connectpt/.venv` with forked `iduedu` available for preprocessing imports
  - `floor-predictor/.venv`
  - `segregation-by-design-experiments/.venv`
  - `iduedu-fork/.venv` from forked `GeorgeKontsevik/IduEdu`

## Run One City

```bash
cd /Users/gk/Code/super-duper-disser
PLACE="Saint Petersburg, Russia"
PYTHONPATH=/Users/gk/Code/super-duper-disser .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_joint --place "$PLACE" --buffer-m 5000 --street-grid-step 500
PYTHONPATH=/Users/gk/Code/super-duper-disser blocksnet/.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs --place "$PLACE"
PYTHONPATH=/Users/gk/Code/super-duper-disser .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters --place "$PLACE"
```

## Run Batch

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser MPLCONFIGDIR=/tmp/mpl-super-duper-disser \
.venv/bin/python scripts/run_random_50_cities_pipeline.py \
  --cities-file simplemaps_worldcities_basicv1/worldcities.csv \
  --min-population 800000 \
  --sample-size 30 \
  --seed 42 \
  --buffer-m 10000 \
  --street-grid-step 500 \
  --pt-subway-stop-buffer-m 0 \
  --pt-dependency-top-routes 30 \
  --services hospital polyclinic school kindergarten \
  --output-root aggregated_spatial_pipeline/outputs/batch_runs/random50_pop800k_10km
```

This runner now supports:
- on-the-fly population filtering via `--min-population`
- automatic skip for already completed cities in the same `--output-root`
  (`joint/<slug>/manifest_joint.json` and `joint_inputs/<slug>/pipeline_2/manifest_prepare_solver_inputs.json`)

Batch runner:
- [scripts/run_random_50_cities_pipeline.py](/Users/gk/Code/super-duper-disser/scripts/run_random_50_cities_pipeline.py)

Force full rebuild for all cities:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser MPLCONFIGDIR=/tmp/mpl-super-duper-disser \
.venv/bin/python scripts/run_random_50_cities_pipeline.py \
  --cities-file simplemaps_worldcities_basicv1/worldcities.csv \
  --min-population 800000 \
  --sample-size 30 \
  --seed 42 \
  --buffer-m 10000 \
  --street-grid-step 500 \
  --pt-subway-stop-buffer-m 0 \
  --pt-dependency-top-routes 30 \
  --services hospital polyclinic school kindergarten \
  --output-root aggregated_spatial_pipeline/outputs/batch_runs/random50_pop800k_10km \
  --no-cache
```

Retry only failed cities from an existing batch summary:

```bash
cd /Users/gk/Code/super-duper-disser
.venv/bin/python - <<'PY'
import json, os, subprocess
from pathlib import Path

summary = Path("aggregated_spatial_pipeline/outputs/batch_runs/random50_pop800k_10km/summary.json")
data = json.loads(summary.read_text(encoding="utf-8"))

env = dict(os.environ)
env["PYTHONPATH"] = f"{Path.cwd()}:{env.get('PYTHONPATH','')}".rstrip(":")
env.setdefault("MPLCONFIGDIR", "/tmp/mpl-super-duper-disser")

for row in data.get("results", []):
    if row.get("status") != "failed":
        continue

    slug = row.get("slug", "unknown")
    print(f"\n==> retry failed city: {slug}")

    joint_manifest = Path(row["joint_output_dir"]) / "manifest_joint.json"
    p2_manifest = Path(row["joint_input_dir"]) / "pipeline_2" / "manifest_prepare_solver_inputs.json"

    try:
        if not joint_manifest.exists():
            subprocess.run(row["commands"]["run_joint"], check=True, env=env)
        else:
            print("  skip run_joint (already has manifest_joint.json)")

        if not p2_manifest.exists():
            subprocess.run(row["commands"]["run_pipeline2_prepare_solver_inputs"], check=True, env=env)
        else:
            print("  skip pipeline2 (already has manifest_prepare_solver_inputs.json)")

        row["status"] = "ok_after_retry"
        row.pop("error", None)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)

summary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nupdated summary: {summary}")
PY
```

Audit all accumulated outputs and classify which city bundles are complete, resumable, or just experimental/non-bundle artifacts:

```bash
cd /Users/gk/Code/super-duper-disser
.venv/bin/python scripts/audit_outputs_status.py \
  --only-problematic \
  --print-cities \
  --write-json /tmp/sdd_outputs_audit.json \
  --write-tsv /tmp/sdd_outputs_audit.tsv
```

This audit distinguishes:
- complete city bundles with `pipeline_2`
- phase-1-complete bundles that can go straight to `pipeline_2`
- resumable partial bundles (`early / mid / late`)
- `non_bundle_layout` experimental roots that are not full city bundles

## Shared Visualization Tool

One project-level visualization tool now owns the default preview canvas and base map styling:

- [aggregated_spatial_pipeline/visualization/map_canvas.py](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/visualization/map_canvas.py)
- [aggregated_spatial_pipeline/visualization/__init__.py](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/visualization/__init__.py)

Use it instead of adding custom per-module matplotlib setup when a step already has:
- a boundary or circle canvas
- one or more `GeoDataFrame` layers
- a default background layer such as blocks
- a preview PNG output

Core helpers provided by the tool:
- `normalize_preview_gdf(...)`
- `clip_to_preview_boundary(...)`
- `apply_preview_canvas(...)`
- `legend_bottom(...)`
- `footer_text(...)`
- `save_preview_figure(...)`

Already wired into:
- [aggregated_spatial_pipeline/pipeline/run_joint.py](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/pipeline/run_joint.py)
- [aggregated_spatial_pipeline/pipeline/run_pipeline2_prepare_solver_inputs.py](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/pipeline/run_pipeline2_prepare_solver_inputs.py)
- [aggregated_spatial_pipeline/pipeline/run_sm_imputation_external.py](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/pipeline/run_sm_imputation_external.py)

Quick check after changes:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPYCACHEPREFIX=/Users/gk/Code/super-duper-disser/.cache/pyc python3 -m py_compile \
  aggregated_spatial_pipeline/visualization/__init__.py \
  aggregated_spatial_pipeline/visualization/map_canvas.py \
  aggregated_spatial_pipeline/pipeline/run_joint.py \
  aggregated_spatial_pipeline/pipeline/run_pipeline2_prepare_solver_inputs.py \
  aggregated_spatial_pipeline/pipeline/run_sm_imputation_external.py
```

If a preview changes visually, check the rendered PNGs in:
- [aggregated_spatial_pipeline/outputs/joint_inputs](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/joint_inputs)
- `.../preview_png/all_together/`
- `.../preview_png/stages/<stage>/`

## Shared Runtime Config

One shared runtime module now owns the default logger format and workspace-local cache settings:

- [aggregated_spatial_pipeline/runtime_config.py](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/runtime_config.py)

Use it instead of per-file `LOG_FORMAT`, ad-hoc `logger.remove()` setup, or `/tmp`-based `MPLCONFIGDIR` defaults.

Core helpers:
- `configure_logger(...)`
- `repo_cache_dir(...)`
- `repo_mplconfigdir(...)`
- `ensure_repo_mplconfigdir(...)`

## Standalone Steps

Each major step can now be invoked directly for a city bundle or territory.

Phase 1 only:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_joint \
  --place "Tartu, Estonia" \
  --buffer-m 1000 \
  --street-grid-step 300 \
  --collect-only \
  --no-cache
```

Floor enrichment only:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
floor-predictor/.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_floor_predictor_external \
  --place "Tartu, Estonia"
```

Intermodal graph only:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
iduedu-fork/.venv/bin/python -m aggregated_spatial_pipeline.intermodal_graph_data_pipeline.run_bundle_external \
  --place "Tartu, Estonia"
```

BlocksNet bundle only:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
blocksnet/.venv/bin/python -m aggregated_spatial_pipeline.blocksnet_data_pipeline.run_bundle_external \
  --place "Tartu, Estonia"
```

ConnectPT bundle only:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
connectpt/.venv/bin/python -m aggregated_spatial_pipeline.connectpt_data_pipeline.run_bundle_external \
  --place "Tartu, Estonia" \
  --modalities bus tram trolleybus subway
```

SM imputation only:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
sm_imputation/.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_sm_imputation_external \
  --place "Tartu, Estonia"
```

Solver inputs and accessibility:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
blocksnet/.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
  --place tartu_estonia \
  --placement-exact
```

Street-pattern transfer to blocks:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters \
  --place tartu_estonia
```

ConnectPT route generation on existing city bundle:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
connectpt/.venv/bin/python -m aggregated_spatial_pipeline.connectpt_data_pipeline.run_route_generator_external \
  --place "Tartu, Estonia" \
  --modality bus \
  --replace-in-intermodal \
  --recompute-accessibility
```

Batch phase 1 for many cities:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser \
.venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_phase1_batch \
  --regions europe usa australia_oceania africa asia \
  --limit-per-region 25 \
  --buffer-m 1000 \
  --street-grid-step 300 \
  --no-cache
```

Regional city lists live here:
- [aggregated_spatial_pipeline/config/phase1_city_batches.json](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/config/phase1_city_batches.json)

## Temporary Heuristics And Fallbacks

The root repo contains a few explicitly temporary guardrails. If you add a new workaround,
document it here as well.

Current temporary behavior:
- `aggregated_spatial_pipeline/connectpt_data_pipeline/pipeline.py` keeps the standard intermodal-to-connectpt stop bridge distance at `IDUEDU_CONNECTPT_BRIDGE_DISTANCE_M = 30.0`.

## Next Steps

1. Extend the solver to support changing capacities of existing services.
2. Combine that capacity-change scenario with the genetic solver workflow.
3. Integrate `sm-imputer` into the main pipeline.
4. Recompute accessibility, provision, and optimization outputs after `sm-imputer`.
5. Build a new bus graph in `connectpt`.
6. Recompute accessibility on top of the new `connectpt` bus graph.
7. Add `connectpt` optimization over proposed PT links together with service optimization.
8. Move from city-level runs to agglomeration-level runs with `arctic_access` to account for seasonal effects.
9. Repeat the same agglomeration-level flow for Africa to account for climate and external-environment effects.

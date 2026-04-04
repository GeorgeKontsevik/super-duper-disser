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
./run_all_cities.sh 2>&1 | tee run_all_cities.log
```

City list:
- [cities_small_compare.txt](/Users/gk/Code/super-duper-disser/cities_small_compare.txt)

Batch runner:
- [run_all_cities.sh](/Users/gk/Code/super-duper-disser/run_all_cities.sh)

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

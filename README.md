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

## Temporary Heuristics And Fallbacks

The root repo contains a few explicitly temporary guardrails. If you add a new workaround,
document it here as well.

Current temporary behavior:
- `aggregated_spatial_pipeline/connectpt_data_pipeline/pipeline.py` keeps the standard intermodal-to-connectpt stop bridge distance at `IDUEDU_CONNECTPT_BRIDGE_DISTANCE_M = 30.0`.
- The same file currently contains a temporary sparse-stop fallback for `tram` and `trolleybus`:
  - `INTERMODAL_SPARSE_STOP_FALLBACK_MODALITIES = {Modality.TRAM, Modality.TROLLEYBUS}`
  - `MIN_RELIABLE_INTERMODAL_AGGREGATED_STOPS = 10`
- Meaning:
  - if intermodal/iduedu-derived stops for `tram` or `trolleybus` are present but look suspiciously sparse after aggregation,
    the pipeline is allowed to retry stop collection via direct OSM/connectpt stop extraction and keep the richer result.
- Why this exists:
  - on some cities, intermodal graph nodes are currently rich enough for `bus` but visibly incomplete for `tram`/`trolleybus`;
    for example, Vologda trolleybus stops from the intermodal graph were much sparser than direct OSM-derived stops.
- This is a stopgap, not the desired final architecture:
  - the intended long-term fix is to make intermodal/iduedu provide complete modality stop layers,
    or to replace this heuristic with an explicit, documented source-selection policy.

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

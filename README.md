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
- creates `.venv`
- installs main project dependencies
- installs local editable packages: `blocksnet`, `connectpt`, `floor-predictor`
- creates dedicated `.venv-iduedu121` for intermodal graph building from forked `GeorgeKontsevik/IduEdu`

## Run One City

```bash
cd /Users/gk/Code/super-duper-disser
PLACE="Saint Petersburg, Russia"
PYTHONPATH=/Users/gk/Code/super-duper-disser .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_joint --place "$PLACE" --buffer-m 5000 --street-grid-step 500
PYTHONPATH=/Users/gk/Code/super-duper-disser .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs --place "$PLACE"
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

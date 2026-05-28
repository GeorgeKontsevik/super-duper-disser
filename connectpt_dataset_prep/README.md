# ConnectPT Dataset Prep

Small module for the current route-generator training-dataset work:

- build a real-network, street-pattern-aware ConnectPT dataset;
- enforce `gravity` demand only unless `synthetic` is requested explicitly;
- analyze sampled graph / demand / street-pattern structure;
- compare the sampled gravity demand against the original ConnectPT synthetic-demand generator.

The current working dataset is:

```text
connectpt/datasets/real_morph_no_bergen_loop25_bus50
```

Current verified state:

```text
sample_count: 560
demand_source: gravity only
successful cities: 7
Bergen: excluded from training, reserved for evaluation
min Loops & Lollipops per 50-node sample: 13
skipped_no_gravity_od: 4
skipped_insufficient_focus_class_nodes: 1
```

## Commands

Run status:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser:/Users/gk/Code/super-duper-disser/connectpt \
connectpt/.venv/bin/python -m connectpt_dataset_prep.cli status
```

Rebuild current dataset from the tracked candidate city set:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser:/Users/gk/Code/super-duper-disser/connectpt \
connectpt/.venv/bin/python -m connectpt_dataset_prep.cli build-current
```

By default `build-current` now oversamples loop-heavy connected crops:

```text
min_focus_class_share: 0.25
min_focus_class_nodes: 13 for 50-node samples
focus class: Loops & Lollipops
```

Set `--min-focus-class-share 0` only for an unconstrained random-BFS baseline.

Regenerate structure analysis:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser:/Users/gk/Code/super-duper-disser/connectpt \
connectpt/.venv/bin/python -m connectpt_dataset_prep.cli analyze
```

Compare sampled gravity demand against the original synthetic reference:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser:/Users/gk/Code/super-duper-disser/connectpt \
connectpt/.venv/bin/python -m connectpt_dataset_prep.cli compare-synthetic
```

Collect targeted non-European top-up city bundles:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser:/Users/gk/Code/super-duper-disser/connectpt \
connectpt/.venv/bin/python -m connectpt_dataset_prep.cli collect-topup \
  --target-usable-cities 6
```

Full current flow:

```bash
cd /Users/gk/Code/super-duper-disser
PYTHONPATH=/Users/gk/Code/super-duper-disser:/Users/gk/Code/super-duper-disser/connectpt \
connectpt/.venv/bin/python -m connectpt_dataset_prep.cli all
```

## Outputs

Dataset manifest:

- `connectpt/datasets/real_morph_no_bergen_loop25_bus50/manifest.json`

Structure analysis:

- `connectpt/datasets/real_morph_no_bergen_loop25_bus50/analysis/summary.json`
- `connectpt/datasets/real_morph_no_bergen_loop25_bus50/analysis/*.png`

Synthetic comparison:

- `connectpt/datasets/real_morph_no_bergen_loop25_bus50/analysis/demand_synthetic_vs_real/summary.json`
- `connectpt/datasets/real_morph_no_bergen_loop25_bus50/analysis/demand_synthetic_vs_real/02_network_distance_demand.png`

Top-up collection:

- `aggregated_spatial_pipeline/outputs/batch_runs/connectpt_dataset_geo_topup_20260429/summary.json`
- `aggregated_spatial_pipeline/outputs/batch_runs/connectpt_dataset_geo_topup_20260429/joint_inputs/<city>/`

## Rules

- No silent demand fallback. `auto` means gravity if usable, otherwise skip.
- Do not mix fallback synthetic demand into a real-network training dataset.
- Current training samples require at least 25% `Loops & Lollipops` nodes unless explicitly disabled.
- If `processed/collated.pt` exists and the raw dataset is rebuilt, remove `processed/` before processing.
- Treat script completion as insufficient; inspect manifest, summary CSV/JSON, and key PNGs.

# Dataset Prep TODO

## Immediate

- Add more gravity-usable loop-covered city bundles until the training set has at least 10 successful cities.
- Current top-up target regions:
  - Asia: existing `huainan`, plus `Da Nang`, `Chiang Mai`;
  - Americas: existing `arequipa`, plus `Portland`, `Pittsburgh`, `Curitiba`, `Valparaiso`;
  - Russia: `Kazan`, `Yekaterinburg`;
  - Australia: existing `adelaide`, plus `Canberra`, `Hobart`.
- Keep the current rule: cities without usable `blocks` / gravity OD are skipped, not filled with synthetic demand.
- For every new candidate city, record:
  - `connectpt_osm/bus/graph.pkl` exists and has at least 50 nodes;
  - street-pattern cells exist and match most stops;
  - `derived_layers/blocks_clipped.parquet` or equivalent blocks layer exists;
  - gravity OD total demand is positive;
  - 50-node slices produce non-zero OD without fallback.

## City Sampling

- Prefer small / medium cities with varied morphologies:
  - loop-heavy suburbs;
  - mostly irregular grids;
  - regular-grid cores;
  - warped-parallel layouts;
  - sparse / broken-grid cases if they exist in labels.
- Keep Bergen out of training; use it as the held-out evaluation city.
- Current builder oversamples 50-node crops with at least 25% `Loops & Lollipops`
  nodes, so inspect city coverage after rebuild; low-loop cities may contribute
  few or zero samples.
- Track per-city `loops_node_share`, graph density, activity gini, and total demand before training.
- Target a wider but controlled range of demand concentration:
  - keep gravity OD;
  - include sparse-ish OD cases if blocks produce them naturally;
  - avoid artificial OD densification.

## Dataset Balancing

- After adding cities, rerun:
  - `python -m connectpt_dataset_prep.cli build-current`
  - `python -m connectpt_dataset_prep.cli analyze`
  - `python -m connectpt_dataset_prep.cli compare-synthetic`
- Check whether new samples reduce these current biases:
  - only 7 successful gravity + loop-heavy training cities;
  - Loops & Lollipops is intentionally oversampled, so check generalization on held-out Bergen;
  - Broken Grid and Sparse are nearly absent;
  - OD is still almost fully positive in most 50-node slices.
- If city count grows enough, create a balanced sample selector instead of equal `samples_per_city`.

## Synthetic Comparison

- If the original `datasets/mixed_50/raw_graphs_1000.pkl` is recovered, rerun comparison with:

```bash
python -m connectpt_dataset_prep.cli compare-synthetic \
  --synthetic-pickle /path/to/datasets/mixed_50/raw_graphs_1000.pkl
```

- Until then, the comparison uses the original ConnectPT synthetic generator as a reference.
- Keep reporting both:
  - raw demand scale;
  - within-sample normalized demand shape by network-distance deciles.

## Before Training

- Confirm `manifest.json` has `sample_demand_sources == {"gravity": N}`.
- Confirm every sample in `manifest["samples"]` has `demand_source == "gravity"`.
- Inspect `analysis/summary.json` and key PNG previews.
- Only then start a clean training run; old weights trained on fallback synthetic demand should be treated as contaminated.

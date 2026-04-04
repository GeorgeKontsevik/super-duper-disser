# Working Rules

This file captures repo-specific working rules that should survive chat history and be followed by any new agent working in this repository.

## General

- Do not treat `script finished successfully` as `result is correct`.
- After important runs, inspect output artifacts directly:
  - manifests / summaries
  - key parquet outputs
  - final PNG previews
  - obvious sanity of counts, coverage, probabilities, and rendered layers
- Prefer checking real produced artifacts over assuming correctness from logs.
- In pipeline/data-collection code, prefer existing project or submodule functionality over bespoke helpers.
- Before adding a new helper or workaround, check whether `blocksnet`, `connectpt`, `iduedu-fork`, `solver_flp`, or existing pipeline modules already provide the needed behavior.
- If an existing implementation is close enough, reuse or adapt it instead of introducing a parallel custom path.

## Backlog

- If work is explicitly deferred, record it in:
  - [aggregated_spatial_pipeline/BACKLOG.md](/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/BACKLOG.md)
- Do not leave deferred architectural or product decisions only in chat.

## Logging

- Keep logs neutral when semantics were not actually designed.
- Do not describe default/spec-driven scenario passes as meaningful analytical scenarios unless that logic was truly implemented.
- Prefer short, readable log lines over full absolute-path dumps when basename is enough.
- Avoid silent long phases when practical; add heartbeat logs around expensive steps.

## Outputs And Previews

- Keep city preview PNGs in one place:
  - `aggregated_spatial_pipeline/outputs/joint_inputs/<city>/preview_png/all_together`
- Final previews matter; verify they were actually written at the end of the run.
- When preview style or rendering logic changes, confirm the new PNGs were really regenerated.
- Do not assume a layer is visible just because the render step completed; visually verify when relevant.

## Solver

- For solver experiments, be explicit about mode:
  - exact / non-genetic
  - genetic
  - preference for expanding existing services vs opening new ones
- Do not silently rely on implicit behavior when a flag can make intent explicit.
- When using current pipeline_2 outputs, remember that practical unmet demand may live in `demand_left`, not only `demand_without`.

## PT / Data Integration

- Treat `iduedu` as the primary source for intermodal PT stops when that bundle exists.
- Reuse already collected local layers instead of redownloading equivalent OSM data when possible.
- Preserve and inspect mapping artifacts when simplifying or bridging stop layers.

## Practical Review Habit

- If something feels suspicious, open the file, check the numbers, inspect the picture.
- Prefer one concrete verification over a long explanation.

## Communication

- Be direct and concrete.
- Do not claim something was implemented, designed, or analytically meaningful unless it really was.
- When a run is still in progress, say what stage it is on and what was actually verified already.
- Prefer short progress updates during longer work rather than long speculative explanations.

## Before Push

- Check the touched code paths actually compile or run.
- Verify the produced outputs relevant to the change, not just the code diff.
- If the change affects rendering, inspect the resulting PNGs.
- If the change affects manifests or parquet outputs, inspect at least the key summary fields and row counts.
- Push only after confirming the result is not obviously broken.

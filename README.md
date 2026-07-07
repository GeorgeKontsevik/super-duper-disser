# super-duper-disser

Orchestrates the dissertation spatial pipeline and submodules.

## Scheme

```mermaid
flowchart LR
    A[Inputs] --> B[Run: aggregated_spatial_pipeline/pipeline/run_joint.py]
    B --> C[Checked outputs]
    C --> D[Paper / thesis use]
```

## Main Result

![Main result](itmo-phd-thesis-template-en/images/ch4/optimal_local/polyclinic_route_strategy_4x3_ru.png)

## Run

Entrypoint: `aggregated_spatial_pipeline/pipeline/run_joint.py`

Human:

```bash
PYTHONPATH=$PWD .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_joint --place "Saint Petersburg, Russia" --buffer-m 5000 --street-grid-step 500
```

Agent:

Check submodules, run one small city, inspect manifests and preview PNGs.

## Publication

See `itmo-phd-thesis-template-en/thesis-itmo.pdf` and thesis publication PDFs.

## Next Steps / Heuristics

Keep core here only: bridge, pipeline, pipeline scripts. Experiments live as submodules.

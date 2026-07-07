# super-duper-disser

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style={0}&logo=SciPy&logoColor=white)
![tqdm](https://img.shields.io/badge/tqdm-FFC107.svg?style={0}&logo=tqdm&logoColor=black)

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

super-duper-disser is a Python-based research pipeline for orchestration of spatial accessibility and dissertation-oriented analysis workflows. It is aimed at developers and researchers working with geospatial data, transport/accessibility studies, and thesis or paper figure generation. The repository provides an aggregated spatial pipeline with command-line entry points for running scenario-based processing and joint pipeline execution, along with supporting scripts and notebooks for related workflows. For runnable steps and expected inputs and outputs, see Getting Started.

---

## Installation

**Prerequisites:** requires Python >=3.11

Install super-duper-disser using one of the following methods:

**Build from source:**

1. Clone the super-duper-disser repository:
```sh
git clone https://github.com/GeorgeKontsevik/super-duper-disser
```

2. Navigate to the project directory:
```sh
cd super-duper-disser
```

3. Install the project dependencies:

```sh
pip install -r requirements.txt
```

---

## Getting Started

Prerequisites:
- Python 3.11 or newer
- The project dependencies from `pyproject.toml`

1. Create and activate a Python environment for the repository.
2. Install the project dependencies.
3. Prepare the input layers expected by the aggregated pipeline:
   - `quarters`
   - `street-grid`
   - `climate-grid`
   - `cities`
4. Run the pipeline from the repository root:

```bash
python -m aggregated_spatial_pipeline.pipeline.run --quarters PATH --street-grid PATH --climate-grid PATH --cities PATH --output-dir PATH
```

5. Inspect the generated output directory. The run writes `crosswalks.gpkg`, a `manifest.json`, and per-scenario folders containing `quarters.geojson`, `cities.geojson`, and `metadata.json`.

---

## Architecture

The repository is organized as a dissertation-oriented geospatial analysis monorepo with a thin root layer and several focused pipeline packages. The main orchestration lives in `aggregated_spatial_pipeline`, which loads a JSON-based `PipelineSpec`, validates crosswalks and transfer rules, and then runs scenario transformations over shared geospatial layers.

- `aggregated_spatial_pipeline/pipeline/run.py` is the simplest entry point: it loads input layers, builds crosswalks between them, runs scenarios, and writes per-scenario GeoJSON outputs plus a manifest.
- `aggregated_spatial_pipeline/pipeline/scenarios.py` applies scenario operations in order, including copying from parent scenarios and attribute transfers through crosswalks.
- `aggregated_spatial_pipeline/pipeline/io.py` handles layer loading/saving and ensures basic geospatial validity such as non-empty data, geometry, and CRS.
- `aggregated_spatial_pipeline/spec.py` provides the configuration-driven structure for layers, crosswalks, transfer rules, scenarios, and policy.
- `aggregated_spatial_pipeline/pipeline/run_joint.py` extends the same orchestration pattern with data collection and external runtime invocation, using dedicated Python executables for some upstream components.
- Supporting packages such as `blocksnet`, `connectpt`, `floor-predictor`, `bridge`, and `intermodal_graph_data_pipeline` appear to provide external or auxiliary geospatial/modeling components, while `scripts/` and `tests/` contain runnable experiments, figure-generation utilities, and validation coverage.

Overall, the architecture is configuration-driven and modular: shared geospatial layers flow through crosswalk construction into scenario-specific transformations, with outputs written as reproducible geodata artifacts for downstream analysis and visualization.

---

## Examples

Examples of how this should work and how it should be used are available [here](https://github.com/GeorgeKontsevik/super-duper-disser/tree/main/notebook.ipynb).

---

## Documentation

A detailed super-duper-disser description is available [here](https://github.com/GeorgeKontsevik/super-duper-disser/tree/main/docs).

---

## Contributing

- **[Report Issues](https://github.com/GeorgeKontsevik/super-duper-disser/issues)**: Submit bugs found or log feature requests for the project.

- **[Submit Pull Requests](https://github.com/GeorgeKontsevik/super-duper-disser/tree/main/CONTRIBUTING.md)**: To learn more about making a contribution to super-duper-disser.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    GeorgeKontsevik (2026). super-duper-disser repository [Computer software]. https://github.com/GeorgeKontsevik/super-duper-disser

### BibTeX format:

    @misc{super-duper-disser,

        author = {GeorgeKontsevik},

        title = {super-duper-disser repository},

        year = {2026},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/GeorgeKontsevik/super-duper-disser}},

        url = {https://github.com/GeorgeKontsevik/super-duper-disser}

    }

---

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.pandana_bridge import build_units_matrix_pandana_external
from aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs import (
    _plot_accessibility_previews,
)
from aggregated_spatial_pipeline.runtime_config import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute accessibility matrix and preview for a provided graph pickle.")
    parser.add_argument("--units-path", required=True)
    parser.add_argument("--graph-pickle-path", required=True)
    parser.add_argument("--boundary-path", required=True)
    parser.add_argument("--matrix-output-path", required=True)
    parser.add_argument("--preview-output-dir", required=True)
    return parser.parse_args()


def _configure_logging() -> None:
    configure_logger("[connectpt-access]")


def _log(message: str) -> None:
    logger.bind(tag="[connectpt-access]").info(message)


def main() -> None:
    args = parse_args()
    _configure_logging()

    units_path = Path(args.units_path).resolve()
    graph_pickle_path = Path(args.graph_pickle_path).resolve()
    boundary_path = Path(args.boundary_path).resolve()
    matrix_output_path = Path(args.matrix_output_path).resolve()
    preview_output_dir = Path(args.preview_output_dir).resolve()

    units = gpd.read_parquet(units_path)
    boundary = gpd.read_parquet(boundary_path)
    with graph_pickle_path.open("rb") as fh:
        graph = pickle.load(fh)
    graph_crs = graph.graph.get("crs")
    if graph_crs is not None and not isinstance(graph_crs, int):
        if hasattr(graph_crs, "to_epsg") and graph_crs.to_epsg() is not None:
            graph.graph["crs"] = int(graph_crs.to_epsg())

    _log(
        f"Recomputing accessibility matrix: units={len(units)}, "
        f"graph_nodes={graph.number_of_nodes()}, graph_edges={graph.number_of_edges()}"
    )
    matrix = build_units_matrix_pandana_external(
        units_path=units_path,
        graph_pickle_path=graph_pickle_path,
        output_path=matrix_output_path,
        weight_key="time_min",
    )
    matrix_output_path.parent.mkdir(parents=True, exist_ok=True)
    previews = _plot_accessibility_previews(
        units,
        matrix,
        preview_output_dir,
        boundary=boundary,
        use_cache=False,
    )
    _log(f"Saved matrix: {matrix_output_path}")
    for name, path in previews.items():
        _log(f"Saved preview [{name}]: {path}")


if __name__ == "__main__":
    main()

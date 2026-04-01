from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata

from .crosswalks import build_crosswalk
from .transfers import apply_transfer_rule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline_3 preparation step: transfer street-pattern class probabilities "
            "from street-grid cells to quarters using area-overlap shares."
        )
    )
    parser.add_argument(
        "--joint-input-dir",
        required=True,
        help=(
            "Path to pipeline_1 city bundle, e.g. "
            "/.../aggregated_spatial_pipeline/outputs/joint_inputs/barcelona_spain"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: <joint-input-dir>/pipeline_3",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild the street-pattern-to-quarters transfer even if cached outputs exist.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    logger.info(f"[pipeline_3_street_pattern] {message}")


def _warn(message: str) -> None:
    logger.warning(f"[pipeline_3_street_pattern] {message}")


def main() -> None:
    args = parse_args()
    city_dir = Path(args.joint_input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else city_dir / "pipeline_3"
    prepared_dir = output_dir / "prepared"
    manifest_path = output_dir / "manifest_street_pattern_to_quarters.json"
    transferred_path = prepared_dir / "quarters_with_street_pattern_probs.parquet"
    crosswalk_path = prepared_dir / "crosswalk_street_grid_to_quarters.parquet"

    _log(f"Using city bundle: {city_dir}")
    if args.no_cache:
        _warn("Cache mode: disabled (--no-cache). Street-pattern transfer will be rebuilt.")
    else:
        _log("Cache mode: enabled")

    if transferred_path.exists() and manifest_path.exists() and (not args.no_cache):
        _log(f"Using cached street-pattern-to-quarters transfer: {transferred_path}")
        _log(f"Done. Manifest: {manifest_path}")
        return

    street_grid_path = city_dir / "derived_layers" / "street_grid_buffered.parquet"
    quarters_path = city_dir / "derived_layers" / "quarters_clipped.parquet"
    if not street_grid_path.exists():
        raise FileNotFoundError(f"Missing street grid layer: {street_grid_path}")
    if not quarters_path.exists():
        raise FileNotFoundError(f"Missing quarters layer: {quarters_path}")

    _log("Loading street grid and quarters layers...")
    street = read_geodata(street_grid_path).reset_index(drop=True)
    quarters = read_geodata(quarters_path).reset_index(drop=True)
    if street.empty:
        raise ValueError(f"Street grid is empty: {street_grid_path}")
    if quarters.empty:
        raise ValueError(f"Quarters are empty: {quarters_path}")

    street["street_grid_id"] = street.index.astype(str)
    quarters["quarters_id"] = quarters.index.astype(str)

    _log(
        f"Building crosswalk street_grid -> quarters "
        f"(street cells={len(street)}, quarters={len(quarters)})..."
    )
    crosswalk = build_crosswalk(street, quarters, "street_grid", "quarters")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepare_geodata_for_parquet(crosswalk).to_parquet(crosswalk_path)
    _log(f"Crosswalk ready: {crosswalk_path} ({len(crosswalk)} intersections)")

    _log(
        "Transferring street-pattern probability distribution to quarters "
        "using target_share as overlap weight..."
    )
    transferred = apply_transfer_rule(
        source_gdf=street,
        target_gdf=quarters,
        crosswalk_gdf=crosswalk,
        source_layer="street_grid",
        target_layer="quarters",
        attribute="street_pattern_prob_area_shares",
        aggregation_method="weighted_sum",
        weight_field="target_share",
    )
    prob_columns = [column for column in transferred.columns if column.startswith("prob_")]
    if not prob_columns:
        raise RuntimeError("Street-pattern transfer produced no probability columns on quarters.")

    prepare_geodata_for_parquet(transferred).to_parquet(transferred_path)
    row_sum = transferred[prob_columns].sum(axis=1)
    manifest = {
        "pipeline": "pipeline_3_street_pattern_to_quarters",
        "city_bundle": str(city_dir),
        "street_grid": str(street_grid_path),
        "quarters": str(quarters_path),
        "crosswalk": str(crosswalk_path),
        "output": str(transferred_path),
        "probability_columns": prob_columns,
        "crosswalk_intersections": int(len(crosswalk)),
        "street_cells": int(len(street)),
        "quarters_count": int(len(quarters)),
        "row_probability_sum_stats": {
            "min": float(row_sum.min()),
            "mean": float(row_sum.mean()),
            "median": float(row_sum.median()),
            "max": float(row_sum.max()),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        f"Transfer ready: {transferred_path} "
        f"(prob cols={len(prob_columns)}, mean covered probability mass={row_sum.mean():.4f})"
    )
    _log(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

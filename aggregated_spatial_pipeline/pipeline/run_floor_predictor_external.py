from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.runtime_config import configure_logger
from aggregated_spatial_pipeline.visualization import (
    apply_preview_canvas,
    footer_text,
    legend_bottom,
    normalize_preview_gdf,
    save_preview_figure,
)


ROOT = Path(__file__).resolve().parents[2]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run floor predictor preprocessing in dedicated floor-predictor runtime.")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--place")
    parser.add_argument("--joint-input-dir")
    parser.add_argument("--buildings-path")
    parser.add_argument("--land-use-path")
    parser.add_argument("--roads-path")
    parser.add_argument("--amenities-path")
    parser.add_argument("--output-path")
    parser.add_argument("--summary-path")
    parser.add_argument("--boundary-path")
    parser.add_argument("--preview-dir")
    parser.add_argument("--is-living-model-path")
    parser.add_argument("--overpass-url", default=None)
    parser.add_argument("--osm-timeout-s", type=int, default=180)
    parser.add_argument("--floor-ignore-missing-below-pct", type=float, default=2.0)
    parser.add_argument("--is-living-only", action="store_true")
    args = parser.parse_args()
    if not any([args.place, args.joint_input_dir, args.buildings_path]):
        parser.error("Provide --place or --joint-input-dir or explicit --buildings-path/--land-use-path/--output-path.")
    return args


def _configure_logging() -> None:
    configure_logger("[floor-predictor]")


def _resolve_city_dir(args: argparse.Namespace) -> Path | None:
    if args.joint_input_dir:
        return Path(args.joint_input_dir).resolve()
    if args.place:
        return (ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs" / _slugify(str(args.place))).resolve()
    return None


def _resolve_floor_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path | None, Path | None, Path | None]:
    city_dir = _resolve_city_dir(args)

    if args.buildings_path:
        buildings_path = Path(args.buildings_path).resolve()
    elif city_dir is not None:
        buildings_path = (city_dir / "blocksnet_raw_osm" / "buildings.parquet").resolve()
    else:
        raise ValueError("Could not resolve buildings path.")

    if args.land_use_path:
        land_use_path = Path(args.land_use_path).resolve()
    elif city_dir is not None:
        land_use_path = (city_dir / "blocksnet_raw_osm" / "land_use.parquet").resolve()
    else:
        raise ValueError("Could not resolve land-use path.")

    if args.output_path:
        output_path = Path(args.output_path).resolve()
    elif city_dir is not None:
        output_path = (city_dir / "derived_layers" / "buildings_floor_enriched.parquet").resolve()
    else:
        raise ValueError("Could not resolve output path.")

    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
    else:
        summary_path = output_path.with_name("buildings_floor_enriched_summary.json")

    if args.boundary_path:
        boundary_path = Path(args.boundary_path).resolve()
    elif city_dir is not None:
        candidate = city_dir / "analysis_territory" / "buffer.parquet"
        boundary_path = candidate.resolve() if candidate.exists() else None
    else:
        boundary_path = None

    if args.preview_dir:
        preview_dir = Path(args.preview_dir).resolve()
    elif city_dir is not None:
        preview_dir = (city_dir / "preview_png" / "all_together").resolve()
    else:
        preview_dir = None

    return buildings_path, land_use_path, output_path, summary_path, boundary_path, preview_dir


def _read_optional_geodata(path: Path | None) -> gpd.GeoDataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        gdf = read_geodata(path)
    except Exception:
        return None
    return gdf if gdf is not None and not gdf.empty else None


def _save_floor_previews(
    *,
    source_buildings: gpd.GeoDataFrame,
    enriched_buildings: gpd.GeoDataFrame,
    metrics: dict,
    boundary_path: Path | None,
    preview_dir: Path | None,
) -> dict[str, str]:
    if preview_dir is None:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    preview_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    boundary = _read_optional_geodata(boundary_path)
    if boundary is None:
        boundary = enriched_buildings[["geometry"]].copy()

    boundary_norm = normalize_preview_gdf(boundary, target_crs="EPSG:3857")
    source_norm = normalize_preview_gdf(source_buildings, boundary_norm, target_crs="EPSG:3857")
    enriched_norm = normalize_preview_gdf(enriched_buildings, boundary_norm, target_crs="EPSG:3857")
    if enriched_norm is None or enriched_norm.empty:
        return outputs

    def _numeric_col(frame: gpd.GeoDataFrame, column: str) -> pd.Series:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")

    before_is_living = _numeric_col(source_buildings, "is_living").reindex(enriched_buildings.index)
    before_storey = _numeric_col(source_buildings, "storey").reindex(enriched_buildings.index)
    after_is_living = _numeric_col(enriched_buildings, "is_living")
    after_storey = _numeric_col(enriched_buildings, "storey")

    target_mask = (
        before_is_living.isna()
        | before_storey.isna()
        | before_storey.le(0)
    ).fillna(False)
    resolved_mask = target_mask & after_is_living.notna() & after_storey.notna() & after_storey.gt(0)
    unresolved_mask = target_mask & ~resolved_mask

    status = enriched_norm.copy()
    status["floor_status"] = "usable building record"
    status.loc[resolved_mask.reindex(status.index).fillna(False), "floor_status"] = "resolved by floor step"
    status.loc[unresolved_mask.reindex(status.index).fillna(False), "floor_status"] = "still missing after floor step"
    groups = [
        ("usable building record", "#bfdbfe"),
        ("resolved by floor step", "#f59e0b"),
        ("still missing after floor step", "#dc2626"),
    ]

    def _save(fig, stem: str) -> None:
        out = preview_dir / f"{stem}.png"
        save_preview_figure(fig, out)
        plt.close(fig)
        outputs[stem] = str(out)
        logger.info("Saved preview: {}", out.name)

    fig, ax = plt.subplots(figsize=(12, 12))
    legend_handles: list = []
    for label, color in groups:
        part = status[status["floor_status"] == label]
        if part.empty:
            continue
        part.plot(ax=ax, color=color, linewidth=0.05, edgecolor="#d1d5db", alpha=0.92)
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=label))
    apply_preview_canvas(fig, ax, boundary_norm, title="Floor Enrichment Status", min_pad=100.0)
    legend_handles.append(Line2D([0], [0], color="#111111", linewidth=2, label="analysis boundary"))
    legend_bottom(ax, legend_handles)
    footer_text(
        fig,
        [
            f"rows={int(len(enriched_buildings))}, targeted={int(target_mask.sum())}, resolved={int(resolved_mask.sum())}",
            f"is_living missing {metrics.get('is_living_missing_before')} -> {metrics.get('is_living_missing_after')}, storey missing {metrics.get('storey_missing_before_model')} -> {metrics.get('storey_missing_after_model')}",
            (
                "storey missing after split: "
                f"living={metrics.get('storey_missing_after_model_living')}, "
                f"non-living={metrics.get('storey_missing_after_model_non_living')} (often OK), "
                f"unknown_living={metrics.get('storey_missing_after_model_unknown_living')}"
            ),
        ],
    )
    ax.set_axis_off()
    _save(fig, "buildings_floor_enrichment_status")

    storey_plot = enriched_norm.copy()
    storey_plot["storey"] = pd.to_numeric(storey_plot.get("storey"), errors="coerce")
    storey_plot = storey_plot[storey_plot["storey"].notna() & storey_plot["storey"].gt(0)].copy()
    if not storey_plot.empty:
        fig, ax = plt.subplots(figsize=(12, 12))
        storey_plot.plot(
            ax=ax,
            column="storey",
            cmap="YlGnBu",
            linewidth=0.05,
            edgecolor="#d1d5db",
            alpha=0.92,
            legend=True,
            legend_kwds={"shrink": 0.72, "label": "storey"},
        )
        apply_preview_canvas(fig, ax, boundary_norm, title="Buildings Final Storey", min_pad=100.0)
        ax.set_axis_off()
        _save(fig, "buildings_floor_storey_final")

    return outputs


def main() -> None:
    _configure_logging()
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    city_dir = _resolve_city_dir(args)
    buildings_path, land_use_path, output_path, summary_path, boundary_path, preview_dir = _resolve_floor_paths(args)
    if args.roads_path:
        roads_path = Path(args.roads_path).resolve()
    elif city_dir is not None:
        roads_path = (city_dir / "derived_layers" / "roads_drive_osmnx.parquet").resolve()
    else:
        roads_path = None
    if args.amenities_path:
        amenities_path = Path(args.amenities_path).resolve()
    elif city_dir is not None:
        amenities_path = (city_dir / "blocksnet_raw_osm" / "amenities.parquet").resolve()
    else:
        amenities_path = None
    logger.info(
        "Starting dedicated floor preprocessing: buildings={}, land_use={}, output={}",
        buildings_path.name,
        land_use_path.name,
        output_path.name,
    )

    from aggregated_spatial_pipeline.pipeline.run_joint import _run_floor_predictor_preprocessing

    _configure_logging()

    metrics = _run_floor_predictor_preprocessing(
        repo_root=repo_root,
        buildings_path=buildings_path,
        land_use_path=land_use_path,
        output_path=output_path,
        local_roads_path=roads_path,
        local_amenities_path=amenities_path,
        is_living_model_path=str(args.is_living_model_path) if args.is_living_model_path else None,
        overpass_url=str(args.overpass_url) if args.overpass_url else None,
        osm_timeout_s=int(args.osm_timeout_s),
        floor_ignore_missing_below_pct=float(args.floor_ignore_missing_below_pct),
        is_living_only=bool(args.is_living_only),
    )

    source_buildings = read_geodata(buildings_path)
    enriched_buildings = read_geodata(output_path)
    preview_outputs = _save_floor_previews(
        source_buildings=source_buildings,
        enriched_buildings=enriched_buildings,
        metrics=metrics,
        boundary_path=boundary_path,
        preview_dir=preview_dir,
    )
    summary = {
        **metrics,
        "repo_root": str(repo_root),
        "buildings_path": str(buildings_path),
        "land_use_path": str(land_use_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "boundary_path": None if boundary_path is None else str(boundary_path),
        "preview_dir": None if preview_dir is None else str(preview_dir),
        "preview_outputs": preview_outputs,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

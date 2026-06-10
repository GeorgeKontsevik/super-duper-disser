from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.runtime_paths import intermodal_python
from aggregated_spatial_pipeline.pipeline.run_joint import (
    _clip_street_grid_to_buffer,
    _ensure_street_grid_from_repo,
    _save_collection_previews,
    _slugify,
)
from aggregated_spatial_pipeline.runtime_config import configure_logger, repo_mplconfigdir


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_ROOT = ROOT / "aggregated_spatial_pipeline" / "outputs" / "historical_osm_2000_2025"
DEFAULT_OUTPUT_ROOT = ROOT / "aggregated_spatial_pipeline" / "outputs" / "historical_osm_analysis_2000_2025"
DEFAULT_YEARS = list(range(2000, 2026, 5))
RAW_LAYER_FILES = {
    "water": "water.parquet",
    "roads": "roads.parquet",
    "railways": "railways.parquet",
    "land_use": "land_use.parquet",
    "buildings": "buildings.parquet",
    "amenities": "amenities_floor_context.parquet",
    "services_pipeline2_raw": "services_pipeline2_raw.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build year-by-year historical OSM analysis bundles: building living/non-living "
            "classification, street-pattern classification, PT graph, service accessibility, and maps."
        )
    )
    parser.add_argument("--historical-root", default=str(DEFAULT_HISTORICAL_ROOT))
    parser.add_argument("--city-names", nargs="+", help="City folder names under --historical-root.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--services", nargs="+", default=["hospital", "polyclinic", "school", "kindergarten"])
    parser.add_argument(
        "--building-classifier",
        choices=("floor-predictor",),
        default="floor-predictor",
        help=(
            "How to produce derived_layers/buildings_floor_enriched.parquet. "
            "Historical runs must use the existing imputer/floor-predictor path, not OSM tag rules."
        ),
    )
    parser.add_argument("--street-grid-step", type=float, default=500.0)
    parser.add_argument("--street-min-road-count", type=int, default=5)
    parser.add_argument("--street-min-total-road-length", type=float, default=500.0)
    parser.add_argument("--osm-timeout-s", type=int, default=240)
    parser.add_argument("--overpass-url", default=None)
    parser.add_argument("--matrix-engine", choices=("auto", "pandana", "native"), default="pandana")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def _configure_logging() -> None:
    configure_logger("[historical-analysis]")


def _run_command(command: list[str], *, mplconfig_name: str | None = None) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT)
    if mplconfig_name:
        env.setdefault("MPLCONFIGDIR", repo_mplconfigdir(mplconfig_name, root=ROOT))
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def _city_names(historical_root: Path, args: argparse.Namespace) -> list[str]:
    if args.city_names:
        names = list(args.city_names)
    else:
        names = sorted(path.name for path in historical_root.iterdir() if path.is_dir())
    if args.limit is not None:
        names = names[: int(args.limit)]
    return names


def _snapshot_iso(year: int) -> str:
    return f"{int(year):04d}-12-31T23:59:59Z"


def _copy_geodata(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _prepare_city_year_bundle(
    *,
    historical_root: Path,
    output_root: Path,
    city: str,
    year: int,
) -> tuple[Path, dict[str, Path]]:
    src_root = historical_root / city
    year_src = src_root / str(year)
    if not year_src.exists():
        raise FileNotFoundError(f"Missing downloaded snapshot: {year_src}")

    city_dir = output_root / "joint_inputs" / f"{city}_{year}"
    analysis_dir = city_dir / "analysis_territory"
    raw_dir = city_dir / "blocksnet_raw_osm"
    derived_dir = city_dir / "derived_layers"
    preview_dir = city_dir / "preview_png" / "all_together"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    boundary_src = src_root / "boundary.parquet"
    _copy_geodata(boundary_src, analysis_dir / "buffer.parquet")
    _copy_geodata(boundary_src, analysis_dir / "buffer_collection.parquet")
    _copy_geodata(boundary_src, raw_dir / "boundary.parquet")

    raw_files: dict[str, Path] = {
        "boundary": raw_dir / "boundary.parquet",
    }
    for key, filename in RAW_LAYER_FILES.items():
        src = year_src / filename
        dst_name = "amenities.parquet" if key == "amenities" else filename
        dst = raw_dir / dst_name
        _copy_geodata(src, dst)
        raw_files[key] = dst

    raw_manifest = {
        "city": city,
        "year": int(year),
        "source": str(year_src),
        "files": {key: str(path) for key, path in raw_files.items()},
    }
    (raw_dir / "manifest.json").write_text(json.dumps(raw_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return city_dir, raw_files


def _read_count(path: Path) -> int:
    try:
        return int(len(read_geodata(path)))
    except Exception:
        return 0


def _write_roads_derived(city_dir: Path, raw_files: dict[str, Path]) -> Path:
    roads = read_geodata(raw_files["roads"])
    out = city_dir / "derived_layers" / "roads_drive_osmnx.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    roads.to_parquet(out)
    return out


def _run_floor(city_dir: Path, raw_files: dict[str, Path], args: argparse.Namespace) -> Path:
    output_path = city_dir / "derived_layers" / "buildings_floor_enriched.parquet"
    if output_path.exists() and not args.no_cache:
        logger.info("Using cached building living/non-living classification: {}", output_path.name)
        return output_path
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_floor_predictor_external",
        "--repo-root",
        str(ROOT),
        "--buildings-path",
        str(raw_files["buildings"]),
        "--land-use-path",
        str(raw_files["land_use"]),
        "--roads-path",
        str(raw_files["roads"]),
        "--amenities-path",
        str(raw_files["amenities"]),
        "--output-path",
        str(output_path),
        "--boundary-path",
        str(city_dir / "analysis_territory" / "buffer.parquet"),
        "--preview-dir",
        str(city_dir / "preview_png" / "all_together"),
        "--osm-timeout-s",
        str(int(args.osm_timeout_s)),
    ]
    if args.overpass_url:
        command.extend(["--overpass-url", str(args.overpass_url)])
    _run_command(command, mplconfig_name="mpl-historical-floor")
    return output_path


def _run_blocks(city_dir: Path, raw_files: dict[str, Path], buildings_path: Path, args: argparse.Namespace) -> Path:
    output_dir = city_dir / "blocksnet"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.no_cache:
        logger.info("Using cached blocksnet bundle: {}", manifest_path.name)
    else:
        command = [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "aggregated_spatial_pipeline.blocksnet_data_pipeline.run_bundle_external",
            "--joint-input-dir",
            str(city_dir),
            "--output-dir",
            str(output_dir),
            "--boundary-path",
            str(city_dir / "analysis_territory" / "buffer_collection.parquet"),
            "--prefetched-layers-json",
            json.dumps({key: str(path) for key, path in raw_files.items()}, ensure_ascii=False),
            "--buildings-override-path",
            str(buildings_path),
        ]
        _run_command(command, mplconfig_name="mpl-historical-blocksnet")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blocks_path = Path(manifest["files"]["blocks"]).resolve()
    clipped_path = city_dir / "derived_layers" / "blocks_clipped.parquet"
    blocks = read_geodata(blocks_path)
    boundary = read_geodata(city_dir / "analysis_territory" / "buffer.parquet")
    clipped = blocks.clip(boundary.to_crs(blocks.crs) if blocks.crs != boundary.crs else boundary)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].reset_index(drop=True)
    clipped.to_parquet(clipped_path)
    return clipped_path


def _run_intermodal(city_dir: Path, city: str, year: int, args: argparse.Namespace) -> Path:
    output_dir = city_dir / "intermodal_graph_iduedu"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.no_cache:
        logger.info("Using cached historical intermodal graph: {}", manifest_path.name)
        return output_dir / "graph.pkl"
    command = [
        str(intermodal_python(ROOT)),
        "-m",
        "aggregated_spatial_pipeline.intermodal_graph_data_pipeline.build_bundle_external",
        "--place",
        city,
        "--boundary-path",
        str(city_dir / "analysis_territory" / "buffer.parquet"),
        "--output-dir",
        str(output_dir),
        "--overpass-date",
        _snapshot_iso(year),
        "--osm-timeout-s",
        str(int(args.osm_timeout_s)),
    ]
    if args.overpass_url:
        command.extend(["--overpass-url", str(args.overpass_url)])
    _run_command(command, mplconfig_name="mpl-historical-iduedu")
    return output_dir / "graph.pkl"


def _run_street_pattern(city_dir: Path, city: str, roads_path: Path, args: argparse.Namespace) -> Path:
    street_grid_source_path, _, rebuilt = _ensure_street_grid_from_repo(
        place=city_dir.name,
        repo_root=ROOT,
        data_root=city_dir,
        no_cache=bool(args.no_cache),
        buffer_m=0.0,
        grid_step=float(args.street_grid_step),
        min_road_count=int(args.street_min_road_count),
        min_total_road_length=float(args.street_min_total_road_length),
        boundary_path=city_dir / "analysis_territory" / "buffer.parquet",
        roads_path=roads_path,
    )
    clipped_path = city_dir / "derived_layers" / "street_grid_buffered.parquet"
    if args.no_cache or rebuilt or not clipped_path.exists():
        _clip_street_grid_to_buffer(
            street_grid_path=street_grid_source_path,
            buffer_path=city_dir / "analysis_territory" / "buffer.parquet",
            output_path=clipped_path,
        )
    return clipped_path


def _run_connectpt_preview(city_dir: Path, roads_path: Path, buildings_path: Path, args: argparse.Namespace) -> None:
    output_dir = city_dir / "connectpt_osm"
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.no_cache:
        return
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.connectpt_data_pipeline.run_bundle_external",
        "--joint-input-dir",
        str(city_dir),
        "--modalities",
        "bus",
        "tram",
        "trolleybus",
        "--output-dir",
        str(output_dir),
        "--boundary-path",
        str(city_dir / "analysis_territory" / "buffer.parquet"),
        "--drive-roads-path",
        str(roads_path),
        "--buildings-path",
        str(buildings_path),
        "--intermodal-nodes-path",
        str(city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet"),
    ]
    _run_command(command, mplconfig_name="mpl-historical-connectpt")


def _run_pipeline2(city_dir: Path, args: argparse.Namespace) -> None:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs",
        "--joint-input-dir",
        str(city_dir),
        "--services",
        *args.services,
        "--matrix-engine",
        str(args.matrix_engine),
    ]
    if args.no_cache:
        command.append("--no-cache")
    _run_command(command, mplconfig_name="mpl-historical-pipeline2")


def _run_pipeline3(city_dir: Path, args: argparse.Namespace) -> None:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters",
        "--joint-input-dir",
        str(city_dir),
    ]
    if args.no_cache:
        command.append("--no-cache")
    _run_command(command, mplconfig_name="mpl-historical-pipeline3")


def _run_pt_street_pattern(city_dir: Path, street_grid_path: Path, args: argparse.Namespace) -> None:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_pt_street_pattern_dependency",
        "--joint-input-dir",
        str(city_dir),
        "--street-pattern-cells",
        str(street_grid_path),
        "--pt-types",
        "bus",
        "tram",
        "trolleybus",
        "subway",
    ]
    if args.no_cache:
        command.append("--no-cache")
    _run_command(command, mplconfig_name="mpl-historical-pt-street")


def _refresh_previews(city_dir: Path, raw_files: dict[str, Path], args: argparse.Namespace) -> None:
    _save_collection_previews(
        data_root=city_dir,
        buffer_path=city_dir / "analysis_territory" / "buffer.parquet",
        raw_files=raw_files,
        connectpt_manifest_path=city_dir / "connectpt_osm" / "manifest.json",
        intermodal_manifest_path=city_dir / "intermodal_graph_iduedu" / "manifest.json",
        blocks_manifest_path=city_dir / "blocksnet" / "manifest.json",
        buffered_blocks_path=city_dir / "derived_layers" / "blocks_clipped.parquet",
        street_grid_path=city_dir / "derived_layers" / "street_grid_buffered.parquet",
        floor_enriched_path=city_dir / "derived_layers" / "buildings_floor_enriched.parquet",
        floor_metrics=None,
        stage_label="historical_year_complete",
    )


def _summarize_city_year(city_dir: Path, city: str, year: int, elapsed_s: float) -> dict:
    row = {
        "city": city,
        "year": int(year),
        "city_dir": str(city_dir),
        "elapsed_s": round(float(elapsed_s), 1),
        "roads": _read_count(city_dir / "blocksnet_raw_osm" / "roads.parquet"),
        "buildings_raw": _read_count(city_dir / "blocksnet_raw_osm" / "buildings.parquet"),
        "buildings_classified": _read_count(city_dir / "derived_layers" / "buildings_floor_enriched.parquet"),
        "services_raw": _read_count(city_dir / "blocksnet_raw_osm" / "services_pipeline2_raw.parquet"),
        "blocks": _read_count(city_dir / "derived_layers" / "blocks_clipped.parquet"),
        "street_cells": _read_count(city_dir / "derived_layers" / "street_grid_buffered.parquet"),
    }
    for service in ("hospital", "polyclinic", "school", "kindergarten"):
        summary_path = city_dir / "pipeline_2" / "solver_inputs" / service / "summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            row[f"{service}_demand_without_total"] = float(data.get("demand_without_total", 0.0))
            row[f"{service}_provision_total"] = float(data.get("provision_total", 0.0))
    intermodal_manifest = city_dir / "intermodal_graph_iduedu" / "manifest.json"
    if intermodal_manifest.exists():
        data = json.loads(intermodal_manifest.read_text(encoding="utf-8"))
        stats = data.get("stats", {})
        row["pt_graph_nodes"] = int(stats.get("node_count", 0))
        row["pt_graph_edges"] = int(stats.get("edge_count", 0))
    return row


def main() -> None:
    _configure_logging()
    args = parse_args()
    historical_root = Path(args.historical_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[dict] = []
    cities = _city_names(historical_root, args)
    logger.info("Queued historical analysis: cities={}, years={}", cities, args.years)

    for city in cities:
        for year in sorted({int(year) for year in args.years}):
            started = time.time()
            try:
                logger.info("Year bundle start: {}/{}", city, year)
                city_dir, raw_files = _prepare_city_year_bundle(
                    historical_root=historical_root,
                    output_root=output_root,
                    city=city,
                    year=year,
                )
                roads_path = _write_roads_derived(city_dir, raw_files)
                buildings_path = _run_floor(city_dir, raw_files, args)
                _run_blocks(city_dir, raw_files, buildings_path, args)
                _run_intermodal(city_dir, city, year, args)
                street_grid_path = _run_street_pattern(city_dir, city, roads_path, args)
                _run_connectpt_preview(city_dir, roads_path, buildings_path, args)
                _run_pipeline2(city_dir, args)
                _run_pipeline3(city_dir, args)
                _run_pt_street_pattern(city_dir, street_grid_path, args)
                _refresh_previews(city_dir, raw_files, args)
                rows.append(_summarize_city_year(city_dir, city, year, time.time() - started))
                logger.info("Year bundle done: {}/{} elapsed={:.1f}s", city, year, time.time() - started)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Year bundle failed: {}/{} error={}", city, year, exc)
                failures.append({"city": city, "year": int(year), "error": str(exc)})

            summary = pd.DataFrame(rows)
            summary_path = output_root / "historical_osm_analysis_summary.csv"
            summary.to_csv(summary_path, index=False)
            manifest = {
                "historical_root": str(historical_root),
                "output_root": str(output_root),
                "years": sorted({int(year) for year in args.years}),
                "cities": cities,
                "summary_csv": str(summary_path),
                "failures": failures,
            }
            (output_root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print(output_root / "historical_osm_analysis_summary.csv")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.pipeline.run_historical_osm_analysis import (
    DEFAULT_HISTORICAL_ROOT,
    DEFAULT_YEARS,
    ROOT,
    _read_count,
    _run_intermodal,
    _run_street_pattern,
    _run_command,
    _snapshot_iso,
    _write_roads_derived,
)
from aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs import SERVICE_SPECS
from aggregated_spatial_pipeline.runtime_config import configure_logger


DEFAULT_ACCESS_OUTPUT_ROOT = ROOT / "aggregated_spatial_pipeline" / "outputs" / "historical_osm_access_experiments_2000_2025"
ACCESS_RAW_LAYER_FILES = {
    "roads": "roads.parquet",
    "land_use": "land_use.parquet",
    "buildings": "buildings.parquet",
    "amenities": "amenities_floor_context.parquet",
    "services_pipeline2_raw": "services_pipeline2_raw.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run historical OSM snapshots through the latest home/service/PT access experiments "
            "without BlocksNet solver/provision calculations."
        )
    )
    parser.add_argument("--historical-root", default=str(DEFAULT_HISTORICAL_ROOT))
    parser.add_argument("--city-names", nargs="+", help="City folder names under --historical-root.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--output-root", default=str(DEFAULT_ACCESS_OUTPUT_ROOT))
    parser.add_argument("--services", nargs="+", default=["hospital", "polyclinic", "school", "kindergarten"])
    parser.add_argument("--street-grid-step", type=float, default=500.0)
    parser.add_argument("--street-min-road-count", type=int, default=5)
    parser.add_argument("--street-min-total-road-length", type=float, default=500.0)
    parser.add_argument("--osm-timeout-s", type=int, default=240)
    parser.add_argument("--overpass-url", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--skip-experiment-stages",
        action="store_true",
        help="Only prepare year bundles; do not run residential/service/PT experiment scripts.",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    configure_logger("[historical-access]")


def _city_names(historical_root: Path, args: argparse.Namespace) -> list[str]:
    if args.city_names:
        names = list(args.city_names)
    else:
        names = sorted(path.name for path in historical_root.iterdir() if path.is_dir())
    if args.limit is not None:
        names = names[: int(args.limit)]
    return names


def _copy_geodata(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _prepare_access_city_year_bundle(
    *,
    historical_root: Path,
    output_root: Path,
    city: str,
    year: int,
) -> tuple[Path, dict[str, Path]]:
    year_src = historical_root / city / str(year)
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

    boundary_src = historical_root / city / "boundary.parquet"
    _copy_geodata(boundary_src, analysis_dir / "buffer.parquet")
    _copy_geodata(boundary_src, analysis_dir / "buffer_collection.parquet")
    _copy_geodata(boundary_src, raw_dir / "boundary.parquet")

    raw_files: dict[str, Path] = {"boundary": raw_dir / "boundary.parquet"}
    for key, filename in ACCESS_RAW_LAYER_FILES.items():
        src = year_src / filename
        dst_name = "amenities.parquet" if key == "amenities" else filename
        dst = raw_dir / dst_name
        _copy_geodata(src, dst)
        raw_files[key] = dst

    raw_manifest = {
        "city": city,
        "year": int(year),
        "source": str(year_src),
        "mode": "latest_access_experiments",
        "required_layers": list(ACCESS_RAW_LAYER_FILES),
        "files": {key: str(path) for key, path in raw_files.items()},
    }
    (raw_dir / "manifest.json").write_text(json.dumps(raw_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return city_dir, raw_files


def _tag_match(value, expected) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_tag_match(item, expected) for item in value)
    try:
        if pd.isna(value):
            return False
    except ValueError:
        return False
    actual = str(value).strip().lower()
    if isinstance(expected, list):
        return actual in {str(item).strip().lower() for item in expected}
    if expected is True:
        return actual not in {"", "false", "no", "0", "nan", "none"}
    return actual == str(expected).strip().lower()


def _matches_service(row: pd.Series, tag_groups: list[dict]) -> bool:
    for group in tag_groups:
        if all(column in row.index and _tag_match(row.get(column), expected) for column, expected in group.items()):
            return True
    return False


def _write_services_raw(city_dir: Path, raw_files: dict[str, Path], services: list[str]) -> dict:
    services_dir = city_dir / "pipeline_2" / "services_raw"
    services_dir.mkdir(parents=True, exist_ok=True)
    raw = read_geodata(raw_files["services_pipeline2_raw"])
    if raw.crs is None:
        raw = raw.set_crs(4326)

    summary: dict[str, dict] = {}
    for service in services:
        spec = SERVICE_SPECS.get(service)
        if spec is None:
            raise ValueError(f"Unsupported service for historical access experiment: {service}")
        if raw.empty:
            selected = gpd.GeoDataFrame(raw.copy(), geometry="geometry", crs=raw.crs)
        else:
            mask = raw.apply(lambda row: _matches_service(row, spec.tags), axis=1)
            selected = raw.loc[mask].copy()
        selected = selected[selected.geometry.notna() & ~selected.geometry.is_empty].reset_index(drop=True)
        selected.to_parquet(services_dir / f"{service}.parquet")
        summary[service] = {
            "features": int(len(selected)),
            "path": str(services_dir / f"{service}.parquet"),
        }

    manifest = {"services": services, "summary": summary}
    (services_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write_compat_boundary(city_dir: Path) -> None:
    # The latest diagnostics renderer reads boundary from city_dir/blocksnet/boundary.parquet.
    # Keep only this compatibility artifact; no BlocksNet block/provision calculation is run.
    boundary = read_geodata(city_dir / "analysis_territory" / "buffer.parquet")
    compat_dir = city_dir / "blocksnet"
    compat_dir.mkdir(parents=True, exist_ok=True)
    boundary.to_parquet(compat_dir / "boundary.parquet")


def _run_is_living_imputer(city_dir: Path, raw_files: dict[str, Path], args: argparse.Namespace) -> Path:
    output_path = city_dir / "derived_layers" / "buildings_is_living_enriched.parquet"
    summary_path = output_path.with_name("buildings_is_living_enriched_summary.json")
    if output_path.exists() and summary_path.exists() and not args.no_cache:
        logger.info("Using cached imputed is_living layer: {}", output_path.name)
        return output_path

    # Historical downloads may come from older cached snapshots where is_living
    # was derived from tags. Drop it so the existing imputer is the only source
    # of living/non-living classification.
    imputer_input = city_dir / "derived_layers" / "buildings_for_is_living_imputer.parquet"
    buildings = read_geodata(raw_files["buildings"]).copy()
    buildings = buildings.drop(
        columns=[
            "is_living",
            "is_living_source",
            "is_living_rule_source",
            "is_living_restored",
        ],
        errors="ignore",
    )
    imputer_input.parent.mkdir(parents=True, exist_ok=True)
    buildings.to_parquet(imputer_input)

    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_floor_predictor_external",
        "--repo-root",
        str(ROOT),
        "--buildings-path",
        str(imputer_input),
        "--land-use-path",
        str(raw_files["land_use"]),
        "--roads-path",
        str(raw_files["roads"]),
        "--amenities-path",
        str(raw_files["amenities"]),
        "--output-path",
        str(output_path),
        "--summary-path",
        str(summary_path),
        "--boundary-path",
        str(city_dir / "analysis_territory" / "buffer.parquet"),
        "--osm-timeout-s",
        str(int(args.osm_timeout_s)),
        "--is-living-only",
    ]
    if args.overpass_url:
        command.extend(["--overpass-url", str(args.overpass_url)])
    _run_command(command, mplconfig_name="mpl-historical-is-living")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not bool(summary.get("is_living_only")):
        raise RuntimeError(f"is_living imputer did not run in is_living_only mode: {summary_path}")
    if int(summary.get("is_living_restored_count") or 0) <= 0 and int(summary.get("is_living_original_count") or 0) > 0:
        raise RuntimeError(
            "Unexpected original is_living values remained before imputation; "
            f"check stripped input: {imputer_input}"
        )
    return output_path


def _prepare_year_bundle(
    *,
    historical_root: Path,
    output_root: Path,
    city: str,
    year: int,
    args: argparse.Namespace,
) -> dict:
    started = time.time()
    city_dir, raw_files = _prepare_access_city_year_bundle(
        historical_root=historical_root,
        output_root=output_root,
        city=city,
        year=year,
    )
    roads_path = _write_roads_derived(city_dir, raw_files)
    buildings_path = _run_is_living_imputer(city_dir, raw_files, args)
    services_summary = _write_services_raw(city_dir, raw_files, list(args.services))
    _write_compat_boundary(city_dir)
    _run_intermodal(city_dir, city, year, args)
    street_grid_path = _run_street_pattern(city_dir, city, roads_path, args)

    return {
        "city": city,
        "year": int(year),
        "bundle": city_dir.name,
        "city_dir": str(city_dir),
        "snapshot_date": _snapshot_iso(year),
        "elapsed_s": round(time.time() - started, 1),
        "roads": _read_count(raw_files["roads"]),
        "buildings_raw": _read_count(raw_files["buildings"]),
        "buildings_living": int(
            pd.to_numeric(read_geodata(buildings_path).get("is_living"), errors="coerce").fillna(0).gt(0).sum()
        ),
        "services": services_summary,
        "street_cells": _read_count(street_grid_path),
        "pt_nodes": _read_count(city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet"),
        "pt_edges": int(len(pd.read_parquet(city_dir / "intermodal_graph_iduedu" / "graph_edges.parquet"))),
        "status": "ok",
    }


def _run_stage(script_name: str, args: list[str], *, mplconfig_name: str) -> None:
    command = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / script_name), *args]
    _run_command(command, mplconfig_name=mplconfig_name)


def _run_experiment_stages(output_root: Path, bundles: list[str], services: list[str]) -> dict:
    joint_inputs_root = output_root / "joint_inputs"
    experiments_root = output_root / "experiments"
    walk_root = experiments_root / "residential_to_services_top1"
    pt_ge_root = experiments_root / "residential_to_services_pt_top1_walk15plus"
    pt_lt_root = experiments_root / "residential_to_services_pt_top1_walk_lt15"
    homes_pt_root = experiments_root / "residential_to_pt_top3"
    services_pt_root = experiments_root / "services_to_pt_top3"
    diagnostics_root = experiments_root / "service_access_diagnostics"
    pattern_tables_root = diagnostics_root / "pattern_tables"

    city_args = ["--cities", *bundles]
    stage_specs = [
        (
            "residential_to_services_top1",
            "run_residential_to_services_top1.py",
            ["--joint-inputs-root", str(joint_inputs_root), "--out-root", str(walk_root), *city_args, "--services", *services],
        ),
        (
            "residential_to_services_pt_top1_walk15plus",
            "run_residential_to_services_pt_top1.py",
            [
                "--joint-inputs-root",
                str(joint_inputs_root),
                "--walk-root",
                str(walk_root),
                "--out-root",
                str(pt_ge_root),
                *city_args,
                "--services",
                *services,
                "--min-walk-min",
                "15",
            ],
        ),
        (
            "residential_to_services_pt_top1_walk_lt15",
            "run_residential_to_services_pt_top1.py",
            [
                "--joint-inputs-root",
                str(joint_inputs_root),
                "--walk-root",
                str(walk_root),
                "--out-root",
                str(pt_lt_root),
                *city_args,
                "--services",
                *services,
                "--min-walk-min",
                "0",
                "--max-walk-min-exclusive",
                "15",
            ],
        ),
        (
            "residential_to_pt_top3",
            "run_residential_to_pt_top3.py",
            ["--joint-inputs-root", str(joint_inputs_root), "--out-root", str(homes_pt_root), *city_args],
        ),
        (
            "services_to_pt_top3",
            "run_services_to_pt_top3.py",
            ["--joint-inputs-root", str(joint_inputs_root), "--out-root", str(services_pt_root), *city_args, "--services", *services],
        ),
        (
            "service_access_diagnostics",
            "classify_service_access_failures.py",
            [
                "--walk-root",
                str(walk_root),
                "--pt-walk-lt-root",
                str(pt_lt_root),
                "--pt-walk-ge-root",
                str(pt_ge_root),
                "--joint-inputs-root",
                str(joint_inputs_root),
                "--out-root",
                str(diagnostics_root),
                *city_args,
            ],
        ),
        (
            "service_access_pattern_tables",
            "render_service_access_diagnostics_pattern_tables.py",
            [
                "--input",
                str(diagnostics_root / "_all_home_to_service_access_diagnostics.parquet"),
                "--out-root",
                str(pattern_tables_root),
                "--services",
                *services,
            ],
        ),
    ]

    results = []
    for stage_name, script_name, stage_args in stage_specs:
        logger.info("Experiment stage start: {}", stage_name)
        started = time.time()
        _run_stage(script_name, stage_args, mplconfig_name=f"mpl-historical-access-{stage_name}")
        results.append({"stage": stage_name, "elapsed_s": round(time.time() - started, 1), "status": "ok"})
        logger.info("Experiment stage done: {} elapsed={:.1f}s", stage_name, time.time() - started)

    return {
        "experiments_root": str(experiments_root),
        "walk_root": str(walk_root),
        "pt_walk_ge_root": str(pt_ge_root),
        "pt_walk_lt_root": str(pt_lt_root),
        "homes_pt_root": str(homes_pt_root),
        "services_pt_root": str(services_pt_root),
        "diagnostics_root": str(diagnostics_root),
        "pattern_tables_root": str(pattern_tables_root),
        "stages": results,
    }


def main() -> None:
    _configure_logging()
    args = parse_args()
    historical_root = Path(args.historical_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cities = _city_names(historical_root, args)
    years = list(dict.fromkeys(int(year) for year in args.years))
    logger.info("Queued historical access experiments: cities={}, years={}", cities, years)

    rows: list[dict] = []
    failures: list[dict] = []
    bundles: list[str] = []
    for city in cities:
        for year in years:
            try:
                logger.info("Historical access bundle start: {}/{}", city, year)
                row = _prepare_year_bundle(
                    historical_root=historical_root,
                    output_root=output_root,
                    city=city,
                    year=year,
                    args=args,
                )
                rows.append(row)
                bundles.append(str(row["bundle"]))
                logger.info("Historical access bundle done: {}/{} elapsed={}s", city, year, row["elapsed_s"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Historical access bundle failed: {}/{} error={}", city, year, exc)
                failures.append({"city": city, "year": int(year), "error": str(exc)})

            bundle_summary = pd.DataFrame([{k: v for k, v in row.items() if k != "services"} for row in rows])
            bundle_summary_path = output_root / "historical_access_bundle_summary.csv"
            bundle_summary.to_csv(bundle_summary_path, index=False)
            manifest = {
                "historical_root": str(historical_root),
                "output_root": str(output_root),
                "years": years,
                "cities": cities,
                "bundles": bundles,
                "bundle_summary_csv": str(bundle_summary_path),
                "failures": failures,
            }
            (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    experiment_manifest = None
    if bundles and not args.skip_experiment_stages:
        experiment_manifest = _run_experiment_stages(output_root, bundles, list(args.services))

    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["experiment_stages"] = experiment_manifest
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()

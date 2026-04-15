#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl-service-accessibility-rerun"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.blocksnet_data_pipeline.pipeline import (
    _clip_to_boundary,
    _get_floor_amenities_raw,
    _get_pipeline2_services_raw,
    _keep_geometry_types,
    _save_geodata,
)
from aggregated_spatial_pipeline.pipeline.run_joint import _build_buffered_quarters


DEFAULT_JOINT_INPUT_ROOT = ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs"
DEFAULT_EXPERIMENT_OUTPUT_ROOT = (
    ROOT / "segregation-by-design-experiments" / "service_accessibility_street_pattern" / "outputs"
)
DEFAULT_SERVICES = ("hospital", "polyclinic", "school")
DEFAULT_CLEAR_PREVIEW_OLDER_THAN_S = 3600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild late-stage building->quarters inputs for the cities already used by "
            "service_accessibility_street_pattern, then rerun the accessibility experiment."
        )
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["all"],
        help="City slugs to process, or 'all' to reuse the slugs already present in the experiment outputs.",
    )
    parser.add_argument(
        "--joint-input-root",
        type=Path,
        default=DEFAULT_JOINT_INPUT_ROOT,
        help="Root with prepared aggregated_spatial_pipeline joint_inputs city bundles.",
    )
    parser.add_argument(
        "--experiment-output-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUT_ROOT,
        help="service_accessibility_street_pattern outputs root.",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        default=list(DEFAULT_SERVICES),
        help="Services to pass to service_accessibility_street_pattern.",
    )
    parser.add_argument(
        "--skip-service-accessibility",
        action="store_true",
        help="Only rebuild floor + blocks/quarters, without rerunning service accessibility.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Pass --local-only to service_accessibility_street_pattern to reuse existing raw_services cache.",
    )
    parser.add_argument(
        "--skip-raw-backfill",
        action="store_true",
        help="Do not backfill missing blocksnet_raw_osm amenities/services layers before floor rebuild.",
    )
    parser.add_argument(
        "--force-raw-backfill",
        action="store_true",
        help="Refresh raw amenities/services layers even if they already exist.",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=None,
        help="Optional cap for smoke runs.",
    )
    parser.add_argument(
        "--clear-preview-older-than-s",
        type=int,
        default=DEFAULT_CLEAR_PREVIEW_OLDER_THAN_S,
        help="Delete only preview PNGs older than this many seconds before rerun (default: 3600).",
    )
    return parser.parse_args()


def _discover_experiment_slugs(experiment_output_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in experiment_output_root.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )


def _resolve_slugs(args: argparse.Namespace) -> list[str]:
    requested = [slug.strip() for slug in args.cities if str(slug).strip()]
    if requested == ["all"]:
        slugs = _discover_experiment_slugs(args.experiment_output_root)
    else:
        slugs = requested
    if args.max_cities is not None:
        slugs = slugs[: int(args.max_cities)]
    return slugs


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl-service-accessibility-rerun"))
    return env


def _run(command: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(command, check=True, cwd=str(ROOT), env=env)


def _clear_pngs(directory: Path, *, older_than_s: int) -> int:
    if not directory.exists():
        return 0
    now = time.time()
    removed = 0
    for path in directory.glob("*.png"):
        try:
            age_s = now - path.stat().st_mtime
        except FileNotFoundError:
            continue
        if age_s < max(0, int(older_than_s)):
            continue
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def _assert_city_ready(city_dir: Path) -> None:
    required = [
        city_dir / "blocksnet_raw_osm" / "buildings.parquet",
        city_dir / "blocksnet_raw_osm" / "land_use.parquet",
        city_dir / "derived_layers" / "roads_drive_osmnx.parquet",
        city_dir / "analysis_territory" / "buffer.parquet",
        city_dir / "analysis_territory" / "buffer_collection.parquet",
        city_dir / "intermodal_graph_iduedu" / "graph.pkl",
        city_dir / "street_pattern" / city_dir.name / "predicted_cells.geojson",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"City bundle is missing required inputs:\n{joined}")


def _rewrite_derived_manifest(city_dir: Path, *, blocks_path: Path, blocks_processed_path: Path, floor_output_path: Path) -> None:
    manifest_path = city_dir / "derived_layers" / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    files = manifest.setdefault("files", {})
    files["blocks"] = str(blocks_path.resolve())
    files["blocks_processed"] = str(blocks_processed_path.resolve())
    files["buildings_floor_enriched"] = str(floor_output_path.resolve())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_city_preview_pngs(
    slug: str,
    *,
    city_dir: Path,
    experiment_output_root: Path,
    older_than_s: int,
) -> dict[str, int]:
    removed_floor_all = _clear_pngs(city_dir / "preview_png" / "all_together", older_than_s=older_than_s)
    removed_floor_stage = _clear_pngs(city_dir / "preview_png" / "stages" / "floor_enrichment", older_than_s=older_than_s)
    removed_experiment = _clear_pngs(experiment_output_root / slug / "preview_png", older_than_s=older_than_s)
    if removed_floor_all or removed_floor_stage or removed_experiment:
        print(
            "  cleared old preview PNGs: "
            f"floor_all={removed_floor_all}, "
            f"floor_stage={removed_floor_stage}, "
            f"service_accessibility={removed_experiment}"
        )
    return {
        "removed_floor_all_png": removed_floor_all,
        "removed_floor_stage_png": removed_floor_stage,
        "removed_service_accessibility_png": removed_experiment,
    }


def _rewrite_raw_manifest(
    city_dir: Path,
    *,
    amenities_path: Path | None = None,
    services_pipeline2_raw_path: Path | None = None,
    amenities_count: int | None = None,
    services_count: int | None = None,
) -> None:
    manifest_path = city_dir / "blocksnet_raw_osm" / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    files = manifest.setdefault("files", {})
    counts = manifest.setdefault("counts", {})
    if amenities_path is not None:
        files["amenities"] = str(amenities_path.resolve())
    if services_pipeline2_raw_path is not None:
        files["services_pipeline2_raw"] = str(services_pipeline2_raw_path.resolve())
    if amenities_count is not None:
        counts["amenities_floor_context"] = int(amenities_count)
    if services_count is not None:
        counts["services_pipeline2_raw"] = int(services_count)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_raw_boundary_path(city_dir: Path) -> Path:
    for candidate in (
        city_dir / "analysis_territory" / "buffer_collection.parquet",
        city_dir / "analysis_territory" / "buffer.parquet",
        city_dir / "blocksnet_raw_osm" / "boundary.parquet",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve boundary source for raw backfill: {city_dir}")


def _backfill_raw_context_layers(
    slug: str,
    *,
    city_dir: Path,
    skip_raw_backfill: bool,
    force_raw_backfill: bool,
) -> dict[str, object]:
    raw_dir = city_dir / "blocksnet_raw_osm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    amenities_path = raw_dir / "amenities.parquet"
    services_path = raw_dir / "services_pipeline2_raw.parquet"

    need_amenities = force_raw_backfill or not amenities_path.exists()
    need_services = force_raw_backfill or not services_path.exists()
    if skip_raw_backfill or (not need_amenities and not need_services):
        return {
            "amenities_backfilled": False,
            "services_backfilled": False,
            "amenities_count": None,
            "services_count": None,
        }

    boundary_path = _resolve_raw_boundary_path(city_dir)
    boundary_gdf = read_geodata(boundary_path)
    if boundary_gdf.empty:
        raise ValueError(f"Boundary is empty for raw backfill: {boundary_path}")
    boundary_geom = boundary_gdf.union_all()

    result = {
        "amenities_backfilled": False,
        "services_backfilled": False,
        "amenities_count": None,
        "services_count": None,
    }

    if need_amenities:
        started = time.time()
        amenities = _get_floor_amenities_raw(boundary_geom)
        amenities = _clip_to_boundary(amenities, boundary_geom)
        amenities = _keep_geometry_types(
            amenities,
            {"Polygon", "MultiPolygon"},
            layer_name="amenities_floor_context",
        )
        _save_geodata(amenities, amenities_path)
        _rewrite_raw_manifest(
            city_dir,
            amenities_path=amenities_path,
            amenities_count=len(amenities),
        )
        result["amenities_backfilled"] = True
        result["amenities_count"] = int(len(amenities))
        print(
            f"  raw backfill: amenities.parquet={len(amenities)} "
            f"({time.time() - started:.1f}s)"
        )

    if need_services:
        started = time.time()
        services = _get_pipeline2_services_raw(boundary_geom)
        services = _clip_to_boundary(services, boundary_geom)
        services = _keep_geometry_types(
            services,
            {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"},
            layer_name="services_pipeline2_raw",
        )
        _save_geodata(services, services_path)
        _rewrite_raw_manifest(
            city_dir,
            services_pipeline2_raw_path=services_path,
            services_count=len(services),
        )
        result["services_backfilled"] = True
        result["services_count"] = int(len(services))
        print(
            f"  raw backfill: services_pipeline2_raw.parquet={len(services)} "
            f"({time.time() - started:.1f}s)"
        )

    return result


def _rebuild_floor_and_quarters(
    slug: str,
    *,
    joint_input_root: Path,
    experiment_output_root: Path,
    env: dict[str, str],
    skip_raw_backfill: bool,
    force_raw_backfill: bool,
    clear_preview_older_than_s: int,
) -> dict[str, object]:
    city_dir = (joint_input_root / slug).resolve()
    if not city_dir.exists():
        raise FileNotFoundError(f"City bundle not found: {city_dir}")
    _assert_city_ready(city_dir)
    preview_cleanup = _clear_city_preview_pngs(
        slug,
        city_dir=city_dir,
        experiment_output_root=experiment_output_root,
        older_than_s=clear_preview_older_than_s,
    )

    raw_backfill = _backfill_raw_context_layers(
        slug,
        city_dir=city_dir,
        skip_raw_backfill=skip_raw_backfill,
        force_raw_backfill=force_raw_backfill,
    )

    floor_output_path = city_dir / "derived_layers" / "buildings_floor_enriched.parquet"
    amenities_path = city_dir / "blocksnet_raw_osm" / "amenities.parquet"
    floor_rebuilt = False
    floor_reused_existing = False
    if amenities_path.exists():
        floor_cmd = [
            str(ROOT / "floor-predictor" / ".venv" / "bin" / "python"),
            "-m",
            "aggregated_spatial_pipeline.pipeline.run_floor_predictor_external",
            "--joint-input-dir",
            str(city_dir),
            "--osm-timeout-s",
            "60",
            "--floor-ignore-missing-below-pct",
            "2.0",
        ]
        _run(floor_cmd, env=env)
        floor_rebuilt = True
    elif floor_output_path.exists():
        floor_reused_existing = True
        print(f"  reuse existing floor output (no amenities raw layer): {slug}")
    else:
        raise FileNotFoundError(
            f"Cannot rebuild floor stage for {slug}: amenities.parquet is missing and no existing "
            f"buildings_floor_enriched.parquet was found at {floor_output_path}"
        )

    blocks_cmd = [
        str(ROOT / "blocksnet" / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.blocksnet_data_pipeline.run_bundle_external",
        "--joint-input-dir",
        str(city_dir),
    ]
    _run(blocks_cmd, env=env)

    blocks_processed_path = city_dir / "blocksnet" / "blocks.parquet"
    buffer_path = city_dir / "analysis_territory" / "buffer.parquet"
    derived_dir = city_dir / "derived_layers"
    derived_dir.mkdir(parents=True, exist_ok=True)
    blocks_clipped_path = derived_dir / "blocks_clipped.parquet"
    quarters_clipped_path = derived_dir / "quarters_clipped.parquet"
    clipped_path, clipped_count = _build_buffered_quarters(
        blocks_path=blocks_processed_path,
        buffer_path=buffer_path,
        output_path=blocks_clipped_path,
    )
    shutil.copy2(clipped_path, quarters_clipped_path)

    _rewrite_derived_manifest(
        city_dir,
        blocks_path=clipped_path,
        blocks_processed_path=blocks_processed_path,
        floor_output_path=floor_output_path,
    )

    clipped = read_geodata(quarters_clipped_path)
    population_total = 0.0
    if "population_total" in clipped.columns:
        population_total = float(clipped["population_total"].fillna(0.0).sum())
    elif "population_proxy" in clipped.columns:
        population_total = float(clipped["population_proxy"].fillna(0.0).sum())
    return {
        "slug": slug,
        "quarters_count": int(len(clipped)),
        "population_total": population_total,
        "floor_output_path": str(floor_output_path),
        "quarters_path": str(quarters_clipped_path),
        "floor_rebuilt": floor_rebuilt,
        "floor_reused_existing": floor_reused_existing,
        **preview_cleanup,
        **raw_backfill,
    }


def _rerun_service_accessibility(
    slugs: list[str],
    *,
    experiment_output_root: Path,
    services: list[str],
    local_only: bool,
    env: dict[str, str],
    clear_preview_older_than_s: int,
) -> None:
    cross_city_removed = _clear_pngs(
        experiment_output_root / "_cross_city" / "preview_png",
        older_than_s=clear_preview_older_than_s,
    )
    if cross_city_removed:
        print(f"[late-rerun] cleared old cross-city preview PNGs: {cross_city_removed}")
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        "segregation-by-design-experiments/service_accessibility_street_pattern/run_experiments.py",
        "--output-root",
        str(experiment_output_root),
        "--cities",
        *slugs,
        "--services",
        *services,
        "--no-cache",
    ]
    if local_only:
        cmd.append("--local-only")
    _run(cmd, env=env)


def main() -> None:
    args = parse_args()
    joint_input_root = args.joint_input_root.resolve()
    experiment_output_root = args.experiment_output_root.resolve()
    slugs = _resolve_slugs(args)
    if not slugs:
        raise SystemExit("No city slugs resolved.")

    env = _env()
    print(f"[late-rerun] cities={len(slugs)}")
    print(f"[late-rerun] joint_input_root={joint_input_root}")
    print(f"[late-rerun] experiment_output_root={experiment_output_root}")

    rebuilt: list[dict[str, object]] = []
    for idx, slug in enumerate(slugs, start=1):
        print(f"[{idx}/{len(slugs)}] rebuild late stage for {slug}")
        metrics = _rebuild_floor_and_quarters(
            slug,
            joint_input_root=joint_input_root,
            experiment_output_root=experiment_output_root,
            env=env,
            skip_raw_backfill=bool(args.skip_raw_backfill),
            force_raw_backfill=bool(args.force_raw_backfill),
            clear_preview_older_than_s=int(args.clear_preview_older_than_s),
        )
        rebuilt.append(metrics)
        print(
            f"  quarters={metrics['quarters_count']} population_total={metrics['population_total']:.1f}"
        )

    if args.skip_service_accessibility:
        print("[late-rerun] skip service accessibility rerun by flag.")
    else:
        print("[late-rerun] rerunning service_accessibility_street_pattern ...")
        _rerun_service_accessibility(
            slugs,
            experiment_output_root=experiment_output_root,
            services=list(args.services),
            local_only=bool(args.local_only),
            env=env,
            clear_preview_older_than_s=int(args.clear_preview_older_than_s),
        )
        print("[late-rerun] service accessibility rerun finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

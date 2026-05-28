#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregated_spatial_pipeline.blocksnet_data_pipeline.pipeline import slugify_place  # noqa: E402
from aggregated_spatial_pipeline.geodata_io import read_geodata  # noqa: E402
from aggregated_spatial_pipeline.pipeline.run_joint import (  # noqa: E402
    _analysis_buffer_matches,
    _clip_street_grid_to_buffer,
    _configure_osm_requests,
    _ensure_shared_drive_roads,
    _ensure_street_grid_from_repo,
    _resolve_analysis_buffer_from_osm,
)
from aggregated_spatial_pipeline.runtime_paths import connectpt_python, repo_root  # noqa: E402


DEFAULT_CITIES_CSV = ROOT / "simplemaps_worldcities_basicv1" / "worldcities.csv"


@dataclass(frozen=True)
class CityCandidate:
    slug: str
    place: str
    city: str
    admin: str
    country: str
    iso3: str
    population: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly collect a light real-city seed set: analysis boundary, simplified roads, "
            "street-pattern cells, and ConnectPT graph bundle. Heavy blocks/floor/services are skipped."
        )
    )
    parser.add_argument("--cities-csv", default=str(DEFAULT_CITIES_CSV))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--target-usable-cities",
        type=int,
        default=None,
        help="Stop after this many cities have a usable ConnectPT graph; sample-size is then the candidate pool.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-population", type=float, default=100_000)
    parser.add_argument("--max-population", type=float, default=900_000)
    parser.add_argument("--max-per-country", type=int, default=2)
    parser.add_argument("--buffer-m", type=float, default=10_000.0)
    parser.add_argument("--street-grid-step", type=float, default=500.0)
    parser.add_argument("--street-min-road-count", type=int, default=5)
    parser.add_argument("--street-min-total-road-length", type=float, default=500.0)
    parser.add_argument("--modalities", nargs="+", default=["bus"])
    parser.add_argument("--min-connectpt-graph-nodes", type=int, default=1)
    parser.add_argument("--min-connectpt-graph-edges", type=int, default=1)
    parser.add_argument("--speed-kmh", type=float, default=20.0)
    parser.add_argument("--osm-timeout-s", type=float, default=60.0)
    parser.add_argument("--overpass-url", default=None)
    parser.add_argument("--max-cities", type=int, default=None, help="Cap execution after sampling; useful for smoke runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[random50-light] {message}", flush=True)


def _parse_float(value: object) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except Exception:
        return None


def _load_candidates(path: Path, min_population: float, max_population: float) -> list[CityCandidate]:
    candidates: list[CityCandidate] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pop = _parse_float(row.get("population"))
            if pop is None or pop < min_population or pop > max_population:
                continue
            city = str(row.get("city_ascii") or row.get("city") or "").strip()
            admin = str(row.get("admin_name") or "").strip()
            country = str(row.get("country") or "").strip()
            iso3 = str(row.get("iso3") or "").strip()
            if not city or not country:
                continue
            parts = [part for part in (city, admin, country) if part]
            place = ", ".join(parts)
            candidates.append(
                CityCandidate(
                    slug=slugify_place(place),
                    place=place,
                    city=city,
                    admin=admin,
                    country=country,
                    iso3=iso3,
                    population=float(pop),
                )
            )
    return candidates


def _sample_across_countries(
    candidates: list[CityCandidate],
    *,
    sample_size: int,
    seed: int,
    max_per_country: int,
) -> list[CityCandidate]:
    rng = random.Random(seed)
    by_country: dict[str, list[CityCandidate]] = {}
    for candidate in candidates:
        by_country.setdefault(candidate.iso3 or candidate.country, []).append(candidate)
    for values in by_country.values():
        rng.shuffle(values)

    country_keys = list(by_country)
    rng.shuffle(country_keys)
    selected: list[CityCandidate] = []
    per_country: dict[str, int] = {}
    while len(selected) < sample_size:
        progressed = False
        for key in country_keys:
            if len(selected) >= sample_size:
                break
            if per_country.get(key, 0) >= max_per_country:
                continue
            bucket = by_country[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            per_country[key] = per_country.get(key, 0) + 1
            progressed = True
        if not progressed:
            break
    if len(selected) < sample_size:
        remaining = [candidate for values in by_country.values() for candidate in values]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])
    return selected[:sample_size]


def _resolve_output_root(override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (ROOT / "aggregated_spatial_pipeline" / "outputs" / "batch_runs" / f"random50_light_connectpt_{stamp}").resolve()


def _copy_or_build_collection_buffer(analysis_buffer_path: Path, collection_buffer_path: Path, buffer_m: float, no_cache: bool) -> None:
    if collection_buffer_path.exists() and not no_cache and _analysis_buffer_matches(collection_buffer_path, buffer_m):
        return
    collection_buffer_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(analysis_buffer_path, collection_buffer_path)


def _run_connectpt_bundle(
    *,
    place: str,
    output_dir: Path,
    modalities: list[str],
    boundary_path: Path,
    roads_path: Path,
    speed_kmh: float,
) -> dict:
    py = connectpt_python(repo_root())
    if not py.exists():
        raise FileNotFoundError(f"ConnectPT runtime was not found: {py}")
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl-random50-light-connectpt"))
    command = [
        str(py),
        "-m",
        "aggregated_spatial_pipeline.connectpt_data_pipeline.run_bundle_external",
        "--place",
        place,
        "--modalities",
        *modalities,
        "--output-dir",
        str(output_dir),
        "--speed-kmh",
        str(float(speed_kmh)),
        "--boundary-path",
        str(boundary_path),
        "--drive-roads-path",
        str(roads_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    stdout = (completed.stdout or "").strip()
    try:
        return json.loads(stdout.splitlines()[-1]) if stdout else {}
    except Exception:
        return {"stdout_tail": stdout[-4000:]}


def _graph_counts(connectpt_dir: Path, modalities: list[str]) -> dict[str, dict]:
    counts: dict[str, dict] = {}
    manifest_path = connectpt_dir / "manifest.json"
    if not manifest_path.exists():
        return counts
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("modalities", []):
        if item.get("modality") in modalities:
            counts[str(item.get("modality"))] = {
                "graph_node_count": item.get("graph_node_count"),
                "graph_edge_count": item.get("graph_edge_count"),
                "projected_stop_count": item.get("projected_stop_count"),
                "stop_source": item.get("stop_source"),
            }
    return counts


def _has_usable_graph(counts_by_modality: dict[str, dict], *, min_nodes: int, min_edges: int) -> bool:
    for counts in counts_by_modality.values():
        nodes = int(pd.to_numeric(counts.get("graph_node_count"), errors="coerce") or 0)
        edges = int(pd.to_numeric(counts.get("graph_edge_count"), errors="coerce") or 0)
        if nodes >= int(min_nodes) and edges >= int(min_edges):
            return True
    return False


def main() -> None:
    args = parse_args()
    _configure_osm_requests(float(args.osm_timeout_s), overpass_url=args.overpass_url)

    cities_csv = Path(args.cities_csv).resolve()
    output_root = _resolve_output_root(args.output_root)
    joint_inputs_root = output_root / "joint_inputs"
    output_root.mkdir(parents=True, exist_ok=True)
    joint_inputs_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    sample_path = output_root / "sampled_cities.tsv"

    candidates = _load_candidates(
        cities_csv,
        min_population=float(args.min_population),
        max_population=float(args.max_population),
    )
    sampled = _sample_across_countries(
        candidates,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        max_per_country=int(args.max_per_country),
    )
    if args.max_cities is not None:
        sampled = sampled[: int(args.max_cities)]
    sample_path.write_text(
        "\n".join(
            f"{item.slug}\t{item.place}\t{item.country}\t{int(item.population)}"
            for item in sampled
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "cities_csv": str(cities_csv),
        "candidate_count": int(len(candidates)),
        "sample_size_requested": int(args.sample_size),
        "sample_size_effective": int(len(sampled)),
        "target_usable_cities": int(args.target_usable_cities) if args.target_usable_cities is not None else None,
        "seed": int(args.seed),
        "population_range": [float(args.min_population), float(args.max_population)],
        "max_per_country": int(args.max_per_country),
        "buffer_m": float(args.buffer_m),
        "street_grid_step": float(args.street_grid_step),
        "modalities": list(args.modalities),
        "min_connectpt_graph_nodes": int(args.min_connectpt_graph_nodes),
        "min_connectpt_graph_edges": int(args.min_connectpt_graph_edges),
        "output_root": str(output_root),
        "joint_inputs_root": str(joint_inputs_root),
        "sample_file": str(sample_path),
        "dry_run": bool(args.dry_run),
        "results": [],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _log(f"sampled={len(sampled)} from candidates={len(candidates)} -> {sample_path}")
    if args.dry_run:
        _log(f"dry-run summary -> {summary_path}")
        return

    for idx, item in enumerate(sampled, start=1):
        started = time.time()
        city_dir = joint_inputs_root / item.slug
        analysis_dir = city_dir / "analysis_territory"
        derived_dir = city_dir / "derived_layers"
        connectpt_dir = city_dir / "connectpt_osm"
        analysis_buffer_path = analysis_dir / "buffer.parquet"
        collection_buffer_path = analysis_dir / "buffer_collection.parquet"
        roads_path = derived_dir / "roads_drive_osmnx.parquet"
        clipped_street_grid_path = derived_dir / "street_grid_buffered.parquet"
        row = {
            "index": idx,
            "slug": item.slug,
            "place": item.place,
            "country": item.country,
            "population": int(item.population),
            "city_dir": str(city_dir),
            "status": "ok",
            "elapsed_s": None,
            "error": None,
            "roads_count": None,
            "street_cells_count": None,
            "connectpt_graph_counts": {},
        }
        _log(f"[{idx}/{len(sampled)}] {item.place}")
        try:
            if args.no_cache or not analysis_buffer_path.exists() or not _analysis_buffer_matches(analysis_buffer_path, float(args.buffer_m)):
                _resolve_analysis_buffer_from_osm(
                    place=item.place,
                    buffer_m=float(args.buffer_m),
                    output_path=analysis_buffer_path,
                    city_centers_csv=cities_csv,
                )
            _copy_or_build_collection_buffer(
                analysis_buffer_path,
                collection_buffer_path,
                float(args.buffer_m),
                bool(args.no_cache),
            )
            roads_path, roads_count, _ = _ensure_shared_drive_roads(
                buffer_path=collection_buffer_path,
                output_path=roads_path,
                no_cache=bool(args.no_cache),
            )
            row["roads_count"] = int(roads_count)

            street_grid_source_path, _, street_rebuilt = _ensure_street_grid_from_repo(
                place=item.place,
                repo_root=ROOT,
                data_root=city_dir,
                no_cache=bool(args.no_cache),
                buffer_m=float(args.buffer_m),
                grid_step=float(args.street_grid_step),
                min_road_count=int(args.street_min_road_count),
                min_total_road_length=float(args.street_min_total_road_length),
                boundary_path=collection_buffer_path,
                roads_path=roads_path,
            )
            if args.no_cache or street_rebuilt or not clipped_street_grid_path.exists():
                _, cells_count = _clip_street_grid_to_buffer(
                    street_grid_path=street_grid_source_path,
                    buffer_path=analysis_buffer_path,
                    output_path=clipped_street_grid_path,
                )
            else:
                cells_count = len(read_geodata(clipped_street_grid_path))
            row["street_cells_count"] = int(cells_count)

            if args.no_cache or not (connectpt_dir / "manifest.json").exists():
                _run_connectpt_bundle(
                    place=item.place,
                    output_dir=connectpt_dir,
                    modalities=list(args.modalities),
                    boundary_path=collection_buffer_path,
                    roads_path=roads_path,
                    speed_kmh=float(args.speed_kmh),
                )
            row["connectpt_graph_counts"] = _graph_counts(connectpt_dir, list(args.modalities))
            has_graph = _has_usable_graph(
                row["connectpt_graph_counts"],
                min_nodes=int(args.min_connectpt_graph_nodes),
                min_edges=int(args.min_connectpt_graph_edges),
            )
            if not has_graph:
                row["status"] = "no_connectpt_graph"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = str(exc)
            _log(f"FAIL {item.slug}: {exc}")
            if args.fail_fast:
                row["elapsed_s"] = round(time.time() - started, 1)
                summary["results"].append(row)
                summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                raise
        finally:
            row["elapsed_s"] = round(time.time() - started, 1)
            summary["results"].append(row)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            _log(f"{item.slug}: {row['status']} elapsed={row['elapsed_s']}s graphs={row['connectpt_graph_counts']}")
            if args.target_usable_cities is not None:
                usable_count = sum(
                    _has_usable_graph(
                        result.get("connectpt_graph_counts") or {},
                        min_nodes=int(args.min_connectpt_graph_nodes),
                        min_edges=int(args.min_connectpt_graph_edges),
                    )
                    for result in summary["results"]
                )
                if usable_count >= int(args.target_usable_cities):
                    _log(f"target usable cities reached: {usable_count}/{int(args.target_usable_cities)}")
                    break

    _log(f"done -> {summary_path}")


if __name__ == "__main__":
    main()

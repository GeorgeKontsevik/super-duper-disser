#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregated_spatial_pipeline.blocksnet_data_pipeline.pipeline import slugify_place  # noqa: E402


DEFAULT_CITIES_CSV = ROOT / "simplemaps_worldcities_basicv1" / "worldcities.csv"
DEFAULT_SERVICES = ["hospital", "polyclinic", "school", "kindergarten"]
DEFAULT_MODALITIES = ["bus", "tram", "trolleybus"]
DEFAULT_OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
PREFERRED_ISO3 = {
    "AUS",
    "AUT",
    "BEL",
    "BRA",
    "CAN",
    "CHE",
    "CHL",
    "COL",
    "CRI",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GBR",
    "GRC",
    "GTM",
    "HKG",
    "HRV",
    "HUN",
    "IDN",
    "IRL",
    "ISR",
    "ITA",
    "JPN",
    "KOR",
    "LTU",
    "LVA",
    "MEX",
    "MYS",
    "NLD",
    "NOR",
    "NZL",
    "PER",
    "POL",
    "PRT",
    "ROU",
    "SGP",
    "SVK",
    "SVN",
    "SWE",
    "THA",
    "TUR",
    "TWN",
    "URY",
    "USA",
    "VNM",
    "ZAF",
}


class CityCandidate(NamedTuple):
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
            "Collect a full phase-1 joint-input batch for 1.5M-5M population cities and run "
            "walk/PT accessibility plus street-pattern diagnostics on the resulting bundle."
        )
    )
    parser.add_argument("--cities-csv", type=Path, default=DEFAULT_CITIES_CSV)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--target-successes", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-population", type=float, default=1_500_000.0)
    parser.add_argument("--max-population", type=float, default=5_000_000.0)
    parser.add_argument("--max-per-country", type=int, default=2)
    parser.add_argument("--buffer-m", type=float, default=10_000.0)
    parser.add_argument("--street-grid-step", type=float, default=500.0)
    parser.add_argument("--osm-timeout-s", type=float, default=600.0)
    parser.add_argument("--floor-ignore-missing-below-pct", type=float, default=0.0)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--services", nargs="+", default=DEFAULT_SERVICES)
    parser.add_argument("--overpass-urls", nargs="+", default=DEFAULT_OVERPASS_URLS)
    parser.add_argument("--collection-max-retries", type=int, default=4)
    parser.add_argument("--collection-attempt-timeout-s", type=float, default=2400.0)
    parser.add_argument(
        "--exclude-root",
        action="append",
        type=Path,
        default=[
            ROOT / "aggregated_spatial_pipeline" / "outputs" / "active_19_good_cities_20260412" / "joint_inputs",
        ],
        help="Existing joint_inputs roots whose city slugs should be excluded from sampling.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[pop-band-full-access] {message}", flush=True)


def _parse_float(value: object) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except Exception:
        return None


def _resolve_output_root(override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT
        / "aggregated_spatial_pipeline"
        / "outputs"
        / "batch_runs"
        / f"population_band_full_accessibility_{stamp}"
    ).resolve()


def _env(mpl_suffix: str) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = [str(ROOT), str(ROOT / "connectpt")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / mpl_suffix))
    return env


def _load_candidates(path: Path, *, min_population: float, max_population: float) -> list[CityCandidate]:
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
            place = ", ".join(part for part in (city, admin, country) if part)
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


def _gather_existing_slugs(roots: list[Path]) -> set[str]:
    existing: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for city_dir in root.iterdir():
            if city_dir.is_dir() and not city_dir.name.startswith("_"):
                existing.add(city_dir.name)
    return existing


def _sample_across_countries(
    candidates: list[CityCandidate],
    *,
    sample_size: int,
    seed: int,
    max_per_country: int,
    exclude_slugs: set[str],
) -> list[CityCandidate]:
    filtered = [candidate for candidate in candidates if candidate.slug not in exclude_slugs]
    preferred = [candidate for candidate in filtered if candidate.iso3 in PREFERRED_ISO3]
    nonpreferred = [candidate for candidate in filtered if candidate.iso3 not in PREFERRED_ISO3]
    preferred_selected = _round_robin_sample(
        preferred,
        sample_size=sample_size,
        seed=seed,
        max_per_country=max_per_country,
    )
    if len(preferred_selected) >= sample_size:
        return preferred_selected[:sample_size]
    remaining_needed = sample_size - len(preferred_selected)
    nonpreferred_selected = _round_robin_sample(
        nonpreferred,
        sample_size=remaining_needed,
        seed=seed + 1,
        max_per_country=max_per_country,
    )
    return [*preferred_selected, *nonpreferred_selected][:sample_size]


def _round_robin_sample(
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
        remainder = [candidate for values in by_country.values() for candidate in values]
        remainder.sort(key=lambda item: item.population, reverse=True)
        for candidate in remainder:
            if len(selected) >= sample_size:
                break
            if candidate.slug not in {item.slug for item in selected}:
                selected.append(candidate)
    return selected[:sample_size]


def _prioritize_candidates_for_collection(candidates: list[CityCandidate]) -> list[CityCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            0 if item.iso3 in PREFERRED_ISO3 else 1,
            -float(item.population),
            item.place,
        ),
    )


def _read_sample_file(path: Path) -> list[CityCandidate]:
    sampled: list[CityCandidate] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sampled.append(
                CityCandidate(
                    slug=str(row["slug"]),
                    place=str(row["place"]),
                    city=str(row["city"]),
                    admin=str(row["admin"]),
                    country=str(row["country"]),
                    iso3=str(row["iso3"]),
                    population=float(row["population"]),
                )
            )
    return sampled


def _write_sample_file(path: Path, sampled: list[CityCandidate]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["slug", "place", "city", "admin", "country", "iso3", "population"],
        )
        writer.writeheader()
        for item in sampled:
            writer.writerow(item._asdict())


def _is_retryable_run_joint_failure(message: str) -> bool:
    text = str(message or "").lower()
    markers = [
        "read timed out",
        "timed out",
        "connectionerror",
        "readtimeouterror",
        "httpsconnectionpool",
        "remote end closed connection",
        "temporarily unavailable",
        "proxyerror",
    ]
    return any(marker in text for marker in markers)


def _build_collection_command(
    *,
    place: str,
    data_dir: Path,
    output_dir: Path,
    buffer_m: float,
    street_grid_step: float,
    osm_timeout_s: float,
    modalities: list[str],
    floor_ignore_missing_below_pct: float,
    overpass_url: str | None,
    no_cache: bool,
) -> list[str]:
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(py)
    cmd = [
        str(py),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_joint",
        "--place",
        place,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--collect-only",
        "--buffer-m",
        str(float(buffer_m)),
        "--street-grid-step",
        str(float(street_grid_step)),
        "--osm-timeout-s",
        str(float(osm_timeout_s)),
        "--floor-ignore-missing-below-pct",
        str(float(floor_ignore_missing_below_pct)),
        "--modalities",
        *modalities,
    ]
    if overpass_url:
        cmd.extend(["--overpass-url", str(overpass_url)])
    if no_cache:
        cmd.append("--no-cache")
    return cmd


def _build_python_stage_command(*, script_name: str, args: list[str]) -> list[str]:
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(py)
    return [str(py), str(ROOT / "scripts" / script_name), *args]


def _collection_complete(city_dir: Path) -> tuple[bool, dict[str, int]]:
    buildings_path = city_dir / "derived_layers" / "buildings_floor_enriched.parquet"
    graph_nodes_path = city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet"
    graph_edges_path = city_dir / "intermodal_graph_iduedu" / "graph_edges.parquet"
    graph_pickle_path = city_dir / "intermodal_graph_iduedu" / "graph.pkl"
    street_path = city_dir / "street_pattern" / city_dir.name / "predicted_cells.geojson"
    details = {
        "buildings_exists": int(buildings_path.exists()),
        "buildings_has_is_living": 0,
        "graph_nodes_exists": int(graph_nodes_path.exists()),
        "graph_edges_exists": int(graph_edges_path.exists()),
        "graph_pickle_exists": int(graph_pickle_path.exists()),
        "street_pattern_exists": int(street_path.exists()),
        "services_present": 0,
    }
    if buildings_path.exists():
        try:
            details["buildings_has_is_living"] = int("is_living" in pd.read_parquet(buildings_path, columns=None).columns)
        except Exception:
            details["buildings_has_is_living"] = 0
    service_count = 0
    for service in DEFAULT_SERVICES:
        if (city_dir / "pipeline_2" / "services_raw" / f"{service}.parquet").exists():
            service_count += 1
    details["services_present"] = service_count
    ok = (
        details["buildings_exists"] == 1
        and details["buildings_has_is_living"] == 1
        and details["graph_nodes_exists"] == 1
        and details["graph_edges_exists"] == 1
        and details["graph_pickle_exists"] == 1
        and details["street_pattern_exists"] == 1
        and details["services_present"] == len(DEFAULT_SERVICES)
    )
    return ok, details


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _exception_message(exc: BaseException) -> str:
    parts = [str(exc)]
    if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
        for extra in (_coerce_text(getattr(exc, "output", "")), _coerce_text(getattr(exc, "stderr", ""))):
            text = extra.strip()
            if text:
                parts.append(text)
    return "\n".join(part for part in parts if part)


def _run_command(command: list[str], *, env: dict[str, str], timeout_s: float | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    stdout = completed.stdout or ""
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, output=stdout[-4000:])


def _load_stage_report(report_path: Path) -> pd.DataFrame:
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    return pd.read_csv(report_path, sep="\t")


def _stage_error_cities(report: pd.DataFrame) -> list[str]:
    if "status" not in report.columns or "city" not in report.columns:
        return []
    bad = report[report["status"].astype(str).eq("error")]
    return sorted(set(bad["city"].astype(str)))


def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_stage_with_single_retry(
    *,
    stage_name: str,
    command: list[str],
    report_path: Path | None,
    env: dict[str, str],
) -> dict[str, object]:
    started = time.time()
    _run_command(command, env=env)
    rerun_cities: list[str] = []
    if report_path is not None and report_path.exists():
        report = _load_stage_report(report_path)
        rerun_cities = _stage_error_cities(report)
        if rerun_cities:
            retry_command = command + ["--cities", *rerun_cities]
            _log(f"{stage_name}: rerun {len(rerun_cities)} error cities")
            _run_command(retry_command, env=env)
            report = _load_stage_report(report_path)
            rerun_cities = _stage_error_cities(report)
    return {
        "stage": stage_name,
        "elapsed_s": round(time.time() - started, 1),
        "remaining_error_cities": rerun_cities,
        "report_path": str(report_path) if report_path is not None else None,
        "status": "ok" if not rerun_cities else "partial_error",
    }


def main() -> None:
    args = parse_args()
    output_root = _resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    joint_inputs_root = output_root / "joint_inputs"
    joint_root = output_root / "joint"
    experiments_root = output_root / "experiments"
    summary_path = output_root / "summary.json"
    sample_path = output_root / "sampled_cities.tsv"
    joint_inputs_root.mkdir(parents=True, exist_ok=True)
    joint_root.mkdir(parents=True, exist_ok=True)
    experiments_root.mkdir(parents=True, exist_ok=True)

    if sample_path.exists():
        sampled = _read_sample_file(sample_path)
    else:
        candidates = _load_candidates(
            args.cities_csv.resolve(),
            min_population=float(args.min_population),
            max_population=float(args.max_population),
        )
        exclude_roots = [root.resolve() for root in args.exclude_root] + [joint_inputs_root]
        exclude_slugs = _gather_existing_slugs(exclude_roots)
        sampled = _sample_across_countries(
            _prioritize_candidates_for_collection(candidates),
            sample_size=int(args.candidate_pool),
            seed=int(args.seed),
            max_per_country=int(args.max_per_country),
            exclude_slugs=exclude_slugs,
        )
        _write_sample_file(sample_path, sampled)

    summary: dict[str, object] = {
        "output_root": str(output_root),
        "joint_inputs_root": str(joint_inputs_root),
        "joint_root": str(joint_root),
        "experiments_root": str(experiments_root),
        "sample_file": str(sample_path),
        "target_successes": int(args.target_successes),
        "candidate_pool": int(args.candidate_pool),
        "population_range": [float(args.min_population), float(args.max_population)],
        "sampled_count": int(len(sampled)),
        "sampled_places": [item._asdict() for item in sampled],
        "collection": [],
        "stage_runs": [],
    }
    _write_summary(summary_path, summary)

    if args.dry_run:
        _log(f"dry-run summary -> {summary_path}")
        return

    collection_env = _env("mpl-pop-band-collect")
    successful_cities: list[str] = []
    for index, item in enumerate(sampled, start=1):
        if len(successful_cities) >= int(args.target_successes):
            break
        city_dir = joint_inputs_root / item.slug
        city_joint_dir = joint_root / item.slug
        city_dir.mkdir(parents=True, exist_ok=True)
        city_joint_dir.mkdir(parents=True, exist_ok=True)

        complete, checks = _collection_complete(city_dir)
        row: dict[str, object] = {
            "index": index,
            "slug": item.slug,
            "place": item.place,
            "population": int(item.population),
            "status": "cached_complete" if complete else "pending",
            "checks": checks,
            "attempts": [],
        }
        if complete:
            successful_cities.append(item.slug)
            summary["collection"].append(row)
            _write_summary(summary_path, summary)
            _log(f"[{index}/{len(sampled)}] {item.slug}: cached_complete")
            continue

        started = time.time()
        for attempt in range(1, int(args.collection_max_retries) + 1):
            overpass_url = args.overpass_urls[(attempt - 1) % len(args.overpass_urls)]
            command = _build_collection_command(
                place=item.place,
                data_dir=city_dir,
                output_dir=city_joint_dir,
                buffer_m=float(args.buffer_m),
                street_grid_step=float(args.street_grid_step),
                osm_timeout_s=float(args.osm_timeout_s),
                modalities=list(args.modalities),
                floor_ignore_missing_below_pct=float(args.floor_ignore_missing_below_pct),
                overpass_url=overpass_url,
                no_cache=bool(args.no_cache),
            )
            attempt_row = {
                "attempt": attempt,
                "overpass_url": overpass_url,
                "status": "ok",
                "error": None,
            }
            try:
                _log(f"[{index}/{len(sampled)}] {item.place} attempt={attempt} endpoint={overpass_url}")
                _run_command(command, env=collection_env, timeout_s=float(args.collection_attempt_timeout_s))
                complete, checks = _collection_complete(city_dir)
                attempt_row["checks"] = checks
                if complete:
                    row["status"] = "ok"
                    row["checks"] = checks
                    break
                attempt_row["status"] = "incomplete_artifacts"
                row["status"] = "incomplete_artifacts"
                row["checks"] = checks
            except Exception as exc:  # noqa: BLE001
                message = _exception_message(exc)
                attempt_row["status"] = "failed"
                attempt_row["error"] = message
                row["status"] = "failed"
                row["error"] = message
                if not _is_retryable_run_joint_failure(message) and attempt < int(args.collection_max_retries):
                    row["attempts"].append(attempt_row)
                    break
            finally:
                row["attempts"].append(attempt_row)

            if row.get("status") == "ok":
                break

        row["elapsed_s"] = round(time.time() - started, 1)
        summary["collection"].append(row)
        _write_summary(summary_path, summary)
        _log(f"{item.slug}: {row['status']} elapsed={row['elapsed_s']}s")
        if row.get("status") == "ok":
            successful_cities.append(item.slug)
        elif args.fail_fast:
            raise SystemExit(f"collection failed for {item.slug}")

    if len(successful_cities) < int(args.target_successes):
        raise SystemExit(
            f"Only {len(successful_cities)} cities collected successfully; target={int(args.target_successes)}. "
            f"See {summary_path}."
        )

    selected_cities = successful_cities[: int(args.target_successes)]
    summary["successful_cities"] = selected_cities
    _write_summary(summary_path, summary)

    walk_root = experiments_root / "residential_to_services_top1"
    pt_ge_root = experiments_root / "residential_to_services_pt_top1_walk15plus"
    pt_lt_root = experiments_root / "residential_to_services_pt_top1_walk_lt15"
    homes_pt_root = experiments_root / "residential_to_pt_top3"
    services_pt_root = experiments_root / "services_to_pt_top3"
    diagnostics_root = experiments_root / "service_access_diagnostics"
    pattern_tables_root = diagnostics_root / "pattern_tables"

    shared_city_args = ["--cities", *selected_cities]
    stage_specs = [
        {
            "name": "residential_to_services_top1",
            "command": _build_python_stage_command(
                script_name="run_residential_to_services_top1.py",
                args=[
                    "--joint-inputs-root",
                    str(joint_inputs_root),
                    "--out-root",
                    str(walk_root),
                    *shared_city_args,
                    "--services",
                    *args.services,
                ],
            ),
            "report_path": walk_root / "_run_report.tsv",
            "env": _env("mpl-pop-band-walk"),
        },
        {
            "name": "residential_to_services_pt_top1_walk15plus",
            "command": _build_python_stage_command(
                script_name="run_residential_to_services_pt_top1.py",
                args=[
                    "--joint-inputs-root",
                    str(joint_inputs_root),
                    "--walk-root",
                    str(walk_root),
                    "--out-root",
                    str(pt_ge_root),
                    *shared_city_args,
                    "--services",
                    *args.services,
                    "--min-walk-min",
                    "15",
                ],
            ),
            "report_path": pt_ge_root / "_run_report.tsv",
            "env": _env("mpl-pop-band-pt-ge"),
        },
        {
            "name": "residential_to_services_pt_top1_walk_lt15",
            "command": _build_python_stage_command(
                script_name="run_residential_to_services_pt_top1.py",
                args=[
                    "--joint-inputs-root",
                    str(joint_inputs_root),
                    "--walk-root",
                    str(walk_root),
                    "--out-root",
                    str(pt_lt_root),
                    *shared_city_args,
                    "--services",
                    *args.services,
                    "--min-walk-min",
                    "0",
                    "--max-walk-min-exclusive",
                    "15",
                ],
            ),
            "report_path": pt_lt_root / "_run_report.tsv",
            "env": _env("mpl-pop-band-pt-lt"),
        },
        {
            "name": "residential_to_pt_top3",
            "command": _build_python_stage_command(
                script_name="run_residential_to_pt_top3.py",
                args=[
                    "--joint-inputs-root",
                    str(joint_inputs_root),
                    "--out-root",
                    str(homes_pt_root),
                    *shared_city_args,
                ],
            ),
            "report_path": homes_pt_root / "_run_report.tsv",
            "env": _env("mpl-pop-band-homes-pt"),
        },
        {
            "name": "services_to_pt_top3",
            "command": _build_python_stage_command(
                script_name="run_services_to_pt_top3.py",
                args=[
                    "--joint-inputs-root",
                    str(joint_inputs_root),
                    "--out-root",
                    str(services_pt_root),
                    *shared_city_args,
                    "--services",
                    *args.services,
                ],
            ),
            "report_path": services_pt_root / "_run_report.tsv",
            "env": _env("mpl-pop-band-services-pt"),
        },
        {
            "name": "service_access_diagnostics",
            "command": _build_python_stage_command(
                script_name="classify_service_access_failures.py",
                args=[
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
                    *shared_city_args,
                ],
            ),
            "report_path": None,
            "env": _env("mpl-pop-band-diagnostics"),
        },
        {
            "name": "service_access_pattern_tables",
            "command": _build_python_stage_command(
                script_name="render_service_access_diagnostics_pattern_tables.py",
                args=[
                    "--input",
                    str(diagnostics_root / "_all_home_to_service_access_diagnostics.parquet"),
                    "--out-root",
                    str(pattern_tables_root),
                    "--services",
                    *args.services,
                ],
            ),
            "report_path": None,
            "env": _env("mpl-pop-band-pattern-tables"),
        },
    ]

    for stage in stage_specs:
        _log(f"stage -> {stage['name']}")
        stage_result = _run_stage_with_single_retry(
            stage_name=str(stage["name"]),
            command=list(stage["command"]),
            report_path=stage["report_path"],
            env=stage["env"],
        )
        summary["stage_runs"].append(stage_result)
        _write_summary(summary_path, summary)
        if stage["name"] == "service_access_diagnostics":
            parquet_path = diagnostics_root / "_all_home_to_service_access_diagnostics.parquet"
            summary_csv = diagnostics_root / "_all_home_to_service_access_diagnostics_summary.csv"
            if not parquet_path.exists() or not summary_csv.exists():
                raise FileNotFoundError("service_access_diagnostics outputs are missing")
        if stage["name"] == "service_access_pattern_tables":
            combined_png = pattern_tables_root / "all_services_home_street_pattern_label_share_heatmaps.png"
            if not combined_png.exists():
                raise FileNotFoundError(combined_png)

    summary["status"] = "ok"
    _write_summary(summary_path, summary)
    _log(f"done -> {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITIES_FILE = (
    ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "joint_inputs_new_r2_cities_fresh"
    / "new_cities_places_14.tsv"
)


@dataclass(frozen=True)
class CityItem:
    slug: str
    place: str


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random cities from a place list and run pipeline_1 + pipeline_2 "
            "(baseline provision only, no exact/genetic optimization) into a brand-new output root."
        )
    )
    parser.add_argument(
        "--cities-file",
        default=str(DEFAULT_CITIES_FILE),
        help="Path to city list (TSV preferred: slug<TAB>place; or place per line).",
    )
    parser.add_argument("--sample-size", type=int, default=50, help="Number of random cities to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--min-population",
        type=float,
        default=None,
        help=(
            "Optional minimum city population filter applied on the fly when the input file "
            "contains population values (for example, SimpleMaps worldcities.csv)."
        ),
    )
    parser.add_argument("--buffer-m", type=float, default=10000.0, help="Territory buffer in meters (default: 10km).")
    parser.add_argument("--street-grid-step", type=float, default=500.0)
    parser.add_argument("--pt-subway-stop-buffer-m", type=float, default=0.0)
    parser.add_argument("--pt-dependency-top-routes", type=int, default=30)
    parser.add_argument("--services", nargs="+", default=["hospital", "polyclinic", "school", "kindergarten"])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cities", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "New output root for this experiment. "
            "Default: aggregated_spatial_pipeline/outputs/batch_runs/random50_<timestamp>"
        ),
    )
    return parser.parse_args()


def _load_city_items(path: Path) -> list[CityItem]:
    if not path.exists():
        raise FileNotFoundError(f"City list file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_city_items_from_csv(path)

    return _load_city_items_from_text(path)


def _parse_population(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except Exception:
        return None


def _load_city_items_from_text(path: Path) -> list[CityItem]:
    items: list[CityItem] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            slug_raw, place_raw = line.split("\t", 1)
            slug = _slugify(slug_raw)
            place = place_raw.strip()
        else:
            place = line
            slug = _slugify(place)
        if not place:
            continue
        items.append(CityItem(slug=slug, place=place))
    return _deduplicate_city_items(items)


def _load_city_items_from_csv(path: Path) -> list[CityItem]:
    items: list[CityItem] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name) for name in (reader.fieldnames or [])]
        has_population = "population" in fieldnames
        has_place = "place" in fieldnames
        for row in reader:
            place = ""
            if has_place:
                place = str(row.get("place") or "").strip()
            if not place:
                city = str(row.get("city_ascii") or row.get("city") or "").strip()
                admin = str(row.get("admin_name") or "").strip()
                country = str(row.get("country") or "").strip()
                parts = [p for p in (city, admin, country) if p]
                place = ", ".join(parts)
            if not place:
                continue
            slug = _slugify(str(row.get("slug") or place))
            item = CityItem(slug=slug, place=place)
            if has_population:
                pop = _parse_population(row.get("population"))
                if pop is not None:
                    # preserve for filtering by temporarily encoding in slug marker-free map below
                    item = CityItem(slug=f"{item.slug}__pop_{int(pop)}", place=item.place)
            items.append(item)

    return _deduplicate_city_items(items)


def _deduplicate_city_items(items: list[CityItem]) -> list[CityItem]:
    uniq: list[CityItem] = []
    seen: set[str] = set()
    for item in items:
        slug = item.slug
        if "__pop_" in slug:
            slug = slug.split("__pop_", 1)[0]
            item = CityItem(slug=slug, place=item.place)
        if slug in seen:
            continue
        seen.add(slug)
        uniq.append(item)
    return uniq


def _load_population_lookup(path: Path) -> dict[str, float]:
    if path.suffix.lower() != ".csv":
        return {}
    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name) for name in (reader.fieldnames or [])]
        if "population" not in fieldnames:
            return {}
        for row in reader:
            pop = _parse_population(row.get("population"))
            if pop is None:
                continue
            place = str(row.get("place") or "").strip()
            if not place:
                city = str(row.get("city_ascii") or row.get("city") or "").strip()
                admin = str(row.get("admin_name") or "").strip()
                country = str(row.get("country") or "").strip()
                parts = [p for p in (city, admin, country) if p]
                place = ", ".join(parts)
            if place:
                result[_slugify(place)] = pop
    return result


def _resolve_output_root(override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT / "aggregated_spatial_pipeline" / "outputs" / "batch_runs" / f"random50_{ts}"
    ).resolve()


def _choose_pipeline2_python() -> Path:
    candidate = ROOT / "blocksnet" / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    fallback = ROOT / ".venv" / "bin" / "python"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find python runtime for pipeline_2 (checked blocksnet/.venv and .venv).")


def _run_command(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    subprocess.run(command, check=True, cwd=str(cwd), env=env)


def main() -> None:
    args = parse_args()
    cities_file = Path(args.cities_file).resolve()
    output_root = _resolve_output_root(args.output_root)
    input_root = output_root / "joint_inputs"
    joint_root = output_root / "joint"
    summary_path = output_root / "summary.json"
    sample_path = output_root / "sampled_cities.tsv"
    output_root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(parents=True, exist_ok=True)
    joint_root.mkdir(parents=True, exist_ok=True)

    cities = _load_city_items(cities_file)
    if args.min_population is not None:
        population_lookup = _load_population_lookup(cities_file)
        if not population_lookup:
            raise ValueError(
                f"--min-population={float(args.min_population)} was requested, "
                f"but no population column was found in {cities_file}. "
                "Use a source file with population data (e.g., SimpleMaps worldcities.csv)."
            )
        min_pop = float(args.min_population)
        cities = [item for item in cities if float(population_lookup.get(item.slug, -1.0)) >= min_pop]
        print(f"[batch] population filter applied: min_population={int(min_pop)} -> {len(cities)} candidate cities")
    if len(cities) < int(args.sample_size):
        raise ValueError(
            f"Requested sample-size={int(args.sample_size)}, but only {len(cities)} cities are available in {cities_file}"
        )

    rng = random.Random(int(args.seed))
    sampled = rng.sample(cities, int(args.sample_size))
    if args.max_cities is not None:
        sampled = sampled[: int(args.max_cities)]

    sample_lines = [f"{item.slug}\t{item.place}" for item in sampled]
    sample_path.write_text("\n".join(sample_lines) + "\n", encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT)
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-super-duper-disser")

    py_joint = ROOT / ".venv" / "bin" / "python"
    if not py_joint.exists():
        raise FileNotFoundError(f"run_joint python runtime not found: {py_joint}")
    py_pipeline2 = _choose_pipeline2_python()

    results: list[dict] = []
    print(f"[batch] sampled={len(sampled)} cities from {cities_file}")
    print(f"[batch] output_root={output_root}")
    print(f"[batch] sample_file={sample_path}")

    for idx, item in enumerate(sampled, start=1):
        city_input_dir = input_root / item.slug
        city_joint_dir = joint_root / item.slug
        city_input_dir.mkdir(parents=True, exist_ok=True)
        city_joint_dir.mkdir(parents=True, exist_ok=True)

        joint_cmd = [
            str(py_joint),
            "-m",
            "aggregated_spatial_pipeline.pipeline.run_joint",
            "--place",
            item.place,
            "--data-dir",
            str(city_input_dir),
            "--output-dir",
            str(city_joint_dir),
            "--buffer-m",
            str(float(args.buffer_m)),
            "--street-grid-step",
            str(float(args.street_grid_step)),
            "--modalities",
            "bus",
            "tram",
            "trolleybus",
            "--pt-subway-stop-buffer-m",
            str(float(args.pt_subway_stop_buffer_m)),
            "--pt-dependency-top-routes",
            str(int(args.pt_dependency_top_routes)),
        ]
        if args.no_cache:
            joint_cmd.append("--no-cache")

        pipeline2_cmd = [
            str(py_pipeline2),
            "-m",
            "aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs",
            "--joint-input-dir",
            str(city_input_dir),
            "--services",
            *list(args.services),
            "--no-placement-genetic",
        ]
        if args.no_cache:
            pipeline2_cmd.append("--no-cache")

        print(f"[{idx}/{len(sampled)}] {item.slug} :: {item.place}")
        print(f"  joint-data-dir={city_input_dir}")
        started = time.time()
        row = {
            "slug": item.slug,
            "place": item.place,
            "buffer_m": float(args.buffer_m),
            "joint_input_dir": str(city_input_dir),
            "joint_output_dir": str(city_joint_dir),
            "status": "ok",
            "elapsed_s": None,
            "commands": {
                "run_joint": joint_cmd,
                "run_pipeline2_prepare_solver_inputs": pipeline2_cmd,
            },
            "service_preview_png": str(city_input_dir / "preview_png" / "all_together" / "29_services_raw_all_categories.png"),
            "service_preview_exists": False,
        }

        try:
            if args.dry_run:
                print("  dry-run: commands prepared, not executed.")
            else:
                _run_command(joint_cmd, env=env, cwd=ROOT)
                _run_command(pipeline2_cmd, env=env, cwd=ROOT)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = str(exc)
            if args.fail_fast:
                row["elapsed_s"] = round(time.time() - started, 1)
                results.append(row)
                summary = {
                    "cities_file": str(cities_file),
                    "sample_size_requested": int(args.sample_size),
                    "sample_size_effective": len(sampled),
                    "seed": int(args.seed),
                    "buffer_m": float(args.buffer_m),
                    "street_grid_step": float(args.street_grid_step),
                    "pt_subway_stop_buffer_m": float(args.pt_subway_stop_buffer_m),
                    "services": list(args.services),
                    "output_root": str(output_root),
                    "sample_file": str(sample_path),
                    "results": results,
                }
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                raise
        finally:
            row["elapsed_s"] = round(time.time() - started, 1)
            row["service_preview_exists"] = Path(row["service_preview_png"]).exists()
            results.append(row)
            summary = {
                "cities_file": str(cities_file),
                "sample_size_requested": int(args.sample_size),
                "sample_size_effective": len(sampled),
                "seed": int(args.seed),
                "buffer_m": float(args.buffer_m),
                "street_grid_step": float(args.street_grid_step),
                "pt_subway_stop_buffer_m": float(args.pt_subway_stop_buffer_m),
                "services": list(args.services),
                "output_root": str(output_root),
                "sample_file": str(sample_path),
                "results": results,
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[batch] done -> {summary_path}")


if __name__ == "__main__":
    main()

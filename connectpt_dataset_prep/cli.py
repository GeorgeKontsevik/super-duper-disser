from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from .city_sets import CURRENT_CANDIDATE_CITY_DIRS, DEFAULT_DATASET_DIR, ROOT, TOPUP_CITY_PLACES


BUILD_SCRIPT = ROOT / "scripts" / "build_real_connectpt_morph_dataset.py"
ANALYZE_SCRIPT = ROOT / "scripts" / "analyze_real_morph_dataset.py"
COMPARE_SCRIPT = ROOT / "scripts" / "compare_synthetic_real_demand.py"
DEFAULT_TOPUP_OUTPUT_ROOT = (
    ROOT / "aggregated_spatial_pipeline" / "outputs" / "batch_runs" / "connectpt_dataset_geo_topup_20260429"
)


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "city"


def _env(mpl_cache_name: str) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = [str(ROOT), str(ROOT / "connectpt")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / mpl_cache_name))
    return env


def _run(cmd: list[str], *, mpl_cache_name: str) -> None:
    print(f"[connectpt-dataset-prep] run: {Path(cmd[1]).name}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=_env(mpl_cache_name), check=True)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_city_dir(city_dir: Path, *, modality: str = "bus", target_nodes: int = 50) -> dict:
    graph_path = city_dir / "connectpt_osm" / modality / "graph.pkl"
    blocks_path = city_dir / "derived_layers" / "blocks_clipped.parquet"
    if not blocks_path.exists():
        fallback = city_dir / "derived_layers" / "blocks_sm_imputed.parquet"
        blocks_path = fallback if fallback.exists() else blocks_path
    street_path = city_dir / "derived_layers" / "street_grid_buffered.parquet"
    if not street_path.exists():
        fallback = city_dir / "derived_layers" / "street_grid_clipped.parquet"
        street_path = fallback if fallback.exists() else street_path

    graph_nodes = 0
    graph_edges = 0
    graph_error = None
    if graph_path.exists():
        try:
            with graph_path.open("rb") as fh:
                graph = pickle.load(fh)
            graph_nodes = int(graph.number_of_nodes())
            graph_edges = int(graph.number_of_edges())
        except Exception as exc:  # noqa: BLE001
            graph_error = str(exc)

    blocks_exists = blocks_path.exists()
    street_exists = street_path.exists()
    graph_ok = graph_nodes >= int(target_nodes) and graph_edges > 0
    return {
        "city_dir": str(city_dir),
        "blocks_path": str(blocks_path) if blocks_exists else None,
        "street_path": str(street_path) if street_exists else None,
        "graph_path": str(graph_path) if graph_path.exists() else None,
        "blocks_exists": bool(blocks_exists),
        "street_exists": bool(street_exists),
        "graph_nodes": int(graph_nodes),
        "graph_edges": int(graph_edges),
        "graph_error": graph_error,
        "usable_for_training": bool(blocks_exists and street_exists and graph_ok),
    }


def build_current(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    if args.process and not args.keep_processed:
        processed_dir = dataset_dir / "processed"
        if processed_dir.exists():
            print("[connectpt-dataset-prep] removing stale processed/collated.pt", flush=True)
            shutil.rmtree(processed_dir)

    cmd = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--output-dir",
        str(dataset_dir),
        "--modality",
        args.modality,
        "--graph-source",
        args.graph_source,
        "--target-nodes",
        str(args.target_nodes),
        "--samples-per-city",
        str(args.samples_per_city),
        "--max-samples",
        str(args.max_samples),
        "--seed",
        str(args.seed),
        "--demand-source",
        args.demand_source,
        "--min-focus-class-share",
        str(args.min_focus_class_share),
        "--sample-search-attempts",
        str(args.sample_search_attempts),
    ]
    if args.process:
        cmd.append("--process")
    for city_dir in CURRENT_CANDIDATE_CITY_DIRS:
        cmd.extend(["--city-dir", str(city_dir)])
    _run(cmd, mpl_cache_name="mpl-connectpt-dataset-prep")


def analyze(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--dataset-dir",
        str(args.dataset_dir.resolve()),
    ]
    _run(cmd, mpl_cache_name="mpl-connectpt-dataset-analysis")


def compare(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--real-dataset-dir",
        str(args.dataset_dir.resolve()),
        "--seed",
        str(args.seed),
    ]
    if args.synthetic_pickle is not None:
        cmd.extend(["--synthetic-pickle", str(args.synthetic_pickle.resolve())])
    _run(cmd, mpl_cache_name="mpl-connectpt-demand-compare")


def run_all(args: argparse.Namespace) -> None:
    build_current(args)
    analyze(args)
    compare(args)


def collect_topup(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    joint_inputs_root = output_root / "joint_inputs"
    joint_root = output_root / "joint"
    summary_path = output_root / "summary.json"
    output_root.mkdir(parents=True, exist_ok=True)
    joint_inputs_root.mkdir(parents=True, exist_ok=True)
    joint_root.mkdir(parents=True, exist_ok=True)

    places = list(TOPUP_CITY_PLACES)
    if args.max_cities is not None:
        places = places[: int(args.max_cities)]

    env = _env("mpl-connectpt-dataset-topup")
    py_joint = ROOT / ".venv" / "bin" / "python"
    if not py_joint.exists():
        raise FileNotFoundError(py_joint)

    results: list[dict] = []
    for index, place in enumerate(places, start=1):
        slug = _slugify(place)
        city_input_dir = joint_inputs_root / slug
        city_joint_dir = joint_root / slug
        city_input_dir.mkdir(parents=True, exist_ok=True)
        city_joint_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(py_joint),
            "-m",
            "aggregated_spatial_pipeline.pipeline.run_joint",
            "--place",
            place,
            "--data-dir",
            str(city_input_dir),
            "--output-dir",
            str(city_joint_dir),
            "--collect-only",
            "--buffer-m",
            str(float(args.buffer_m)),
            "--street-grid-step",
            str(float(args.street_grid_step)),
            "--modalities",
            args.modality,
        ]
        if args.no_cache:
            command.append("--no-cache")

        row = {
            "index": int(index),
            "place": place,
            "slug": slug,
            "city_dir": str(city_input_dir),
            "joint_output_dir": str(city_joint_dir),
            "status": "ok",
            "elapsed_s": None,
            "command": command,
            "validation": {},
        }
        started = time.time()
        print(f"[connectpt-dataset-prep] topup [{index}/{len(places)}] {place}", flush=True)
        try:
            manifest_path = city_joint_dir / "manifest_joint.json"
            if args.dry_run:
                row["status"] = "dry_run"
            elif manifest_path.exists() and not args.no_cache:
                print("[connectpt-dataset-prep] using cached run_joint output", flush=True)
            else:
                subprocess.run(command, cwd=ROOT, env=env, check=True)
            row["validation"] = _validate_city_dir(
                city_input_dir,
                modality=args.modality,
                target_nodes=int(args.target_nodes),
            )
            if not row["validation"].get("usable_for_training"):
                row["status"] = "not_usable"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = str(exc)
            if args.fail_fast:
                raise
        finally:
            row["elapsed_s"] = round(time.time() - started, 1)
            results.append(row)
            usable_count = sum(1 for item in results if (item.get("validation") or {}).get("usable_for_training"))
            summary = {
                "output_root": str(output_root),
                "joint_inputs_root": str(joint_inputs_root),
                "places": places,
                "buffer_m": float(args.buffer_m),
                "street_grid_step": float(args.street_grid_step),
                "modality": args.modality,
                "target_nodes": int(args.target_nodes),
                "usable_count": int(usable_count),
                "results": results,
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[connectpt-dataset-prep] {slug}: {row['status']} "
                f"usable={bool((row.get('validation') or {}).get('usable_for_training'))} "
                f"elapsed={row['elapsed_s']}s",
                flush=True,
            )
            if args.target_usable_cities is not None and usable_count >= int(args.target_usable_cities):
                print(f"[connectpt-dataset-prep] target usable reached: {usable_count}", flush=True)
                break

    print(f"[connectpt-dataset-prep] topup summary: {summary_path}", flush=True)


def status(args: argparse.Namespace) -> None:
    dataset_dir = args.dataset_dir.resolve()
    manifest = _read_json(dataset_dir / "manifest.json")
    analysis = _read_json(dataset_dir / "analysis" / "summary.json")
    comparison = _read_json(dataset_dir / "analysis" / "demand_synthetic_vs_real" / "summary.json")

    print(f"dataset: {dataset_dir}")
    if not manifest:
        print("manifest: missing")
        return

    print(f"samples: {manifest.get('sample_count')}")
    print(f"demand sources: {manifest.get('sample_demand_sources')}")
    city_statuses = Counter(str(row.get("status", "unknown")) for row in manifest.get("cities", []))
    print(f"city statuses: {dict(sorted(city_statuses.items()))}")
    skipped = [
        f"{row.get('city')} ({row.get('error')})"
        for row in manifest.get("cities", [])
        if int(row.get("samples") or 0) == 0
    ]
    if skipped:
        print("skipped cities:")
        for item in skipped:
            print(f"  - {item}")

    if analysis:
        print(f"analysis demand sources: {analysis.get('demand_source_counts')}")
        print(f"analysis pngs: {len(analysis.get('png_checks', {}))}")

    if comparison:
        print("synthetic comparison:")
        for row in comparison.get("summary_by_dataset", []):
            print(
                "  - {dataset}: samples={samples}, od_gini={od_demand_gini_median:.3f}, "
                "weighted_network_pct={demand_weighted_network_percentile_median:.3f}".format(**row)
            )

    topup_summary = _read_json(args.topup_output_root.resolve() / "summary.json")
    if topup_summary:
        print(f"topup usable: {topup_summary.get('usable_count')} / {len(topup_summary.get('results', []))}")
        for row in topup_summary.get("results", []):
            validation = row.get("validation") or {}
            print(
                "  - {slug}: {status}, usable={usable}, nodes={nodes}, blocks={blocks}, street={street}".format(
                    slug=row.get("slug"),
                    status=row.get("status"),
                    usable=validation.get("usable_for_training"),
                    nodes=validation.get("graph_nodes"),
                    blocks=validation.get("blocks_exists"),
                    street=validation.get("street_exists"),
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConnectPT real-morph dataset preparation utilities.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--topup-output-root", type=Path, default=DEFAULT_TOPUP_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260429)

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-current", help="Build the current gravity-only sampled dataset.")
    build_parser.add_argument("--modality", default="bus")
    build_parser.add_argument("--graph-source", choices=("auto", "connectpt", "roads"), default="auto")
    build_parser.add_argument("--target-nodes", type=int, default=50)
    build_parser.add_argument("--samples-per-city", type=int, default=80)
    build_parser.add_argument("--max-samples", type=int, default=800)
    build_parser.add_argument("--demand-source", choices=("auto", "gravity", "synthetic"), default="auto")
    build_parser.add_argument("--min-focus-class-share", type=float, default=0.25)
    build_parser.add_argument("--sample-search-attempts", type=int, default=500)
    build_parser.add_argument("--process", action=argparse.BooleanOptionalAction, default=True)
    build_parser.add_argument("--keep-processed", action="store_true")
    build_parser.set_defaults(func=build_current)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze dataset structure and write PNG/CSV outputs.")
    analyze_parser.set_defaults(func=analyze)

    compare_parser = subparsers.add_parser("compare-synthetic", help="Compare sampled demand with synthetic reference.")
    compare_parser.add_argument("--synthetic-pickle", type=Path, default=None)
    compare_parser.set_defaults(func=compare)

    all_parser = subparsers.add_parser("all", help="Build, analyze, and compare in sequence.")
    all_parser.add_argument("--modality", default="bus")
    all_parser.add_argument("--graph-source", choices=("auto", "connectpt", "roads"), default="auto")
    all_parser.add_argument("--target-nodes", type=int, default=50)
    all_parser.add_argument("--samples-per-city", type=int, default=80)
    all_parser.add_argument("--max-samples", type=int, default=800)
    all_parser.add_argument("--demand-source", choices=("auto", "gravity", "synthetic"), default="auto")
    all_parser.add_argument("--min-focus-class-share", type=float, default=0.25)
    all_parser.add_argument("--sample-search-attempts", type=int, default=500)
    all_parser.add_argument("--process", action=argparse.BooleanOptionalAction, default=True)
    all_parser.add_argument("--keep-processed", action="store_true")
    all_parser.add_argument("--synthetic-pickle", type=Path, default=None)
    all_parser.set_defaults(func=run_all)

    status_parser = subparsers.add_parser("status", help="Print current dataset and analysis status.")
    status_parser.set_defaults(func=status)

    topup_parser = subparsers.add_parser("collect-topup", help="Collect targeted non-European city bundles.")
    topup_parser.add_argument("--output-root", type=Path, default=DEFAULT_TOPUP_OUTPUT_ROOT)
    topup_parser.add_argument("--modality", default="bus")
    topup_parser.add_argument("--target-nodes", type=int, default=50)
    topup_parser.add_argument("--target-usable-cities", type=int, default=None)
    topup_parser.add_argument("--max-cities", type=int, default=None)
    topup_parser.add_argument("--buffer-m", type=float, default=10_000.0)
    topup_parser.add_argument("--street-grid-step", type=float, default=500.0)
    topup_parser.add_argument("--no-cache", action="store_true")
    topup_parser.add_argument("--dry-run", action="store_true")
    topup_parser.add_argument("--fail-fast", action="store_true")
    topup_parser.set_defaults(func=collect_topup)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

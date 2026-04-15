#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _score_city(city_dir: Path) -> tuple[int, int, int, int, int]:
    """
    Higher is better.
    1) model+amenities+roads
    2) model
    3) has buildings_floor_enriched.parquet
    4) has summary
    5) has manifest
    """
    derived = city_dir / "derived_layers"
    summary_path = derived / "buildings_floor_enriched_summary.json"
    manifest_path = derived / "manifest.json"
    enriched_path = derived / "buildings_floor_enriched.parquet"

    summary = _load_json(summary_path) or {}
    method = summary.get("is_living_restore_method")
    restored = int(summary.get("is_living_restored_count") or 0)
    amenities_ctx = bool(summary.get("floor_context_amenities_input_path"))
    roads_ctx = bool(summary.get("floor_context_roads_input_path"))

    model = int(method == "osm_living_model" and restored > 0)
    model_ctx = int(model == 1 and amenities_ctx and roads_ctx)

    return (
        model_ctx,
        model,
        int(enriched_path.exists()),
        int(summary_path.exists()),
        int(manifest_path.exists()),
    )


def _discover_city_dirs(src_root: Path) -> list[Path]:
    cities: list[Path] = []
    for ji in src_root.rglob("joint_inputs"):
        if not ji.is_dir():
            continue
        for c in ji.iterdir():
            if c.is_dir():
                cities.append(c)
    return sorted(cities)


def build_unique(src_root: Path, unique_root: Path) -> Path:
    city_dirs = _discover_city_dirs(src_root)
    by_slug: dict[str, list[Path]] = {}
    for c in city_dirs:
        by_slug.setdefault(c.name, []).append(c)

    if unique_root.exists():
        shutil.rmtree(unique_root)
    unique_root.mkdir(parents=True, exist_ok=True)

    report_path = unique_root / "_unique_selection_report.tsv"
    lines = [
        "city\tselected_source\tnum_candidates\tselected_score\tselected_method\tselected_restored\tselected_amenities_ctx\tselected_roads_ctx"
    ]

    for city, candidates in sorted(by_slug.items()):
        ranked = sorted(candidates, key=lambda p: (_score_city(p), str(p)), reverse=True)
        best = ranked[0]
        score = _score_city(best)

        summary = _load_json(best / "derived_layers" / "buildings_floor_enriched_summary.json") or {}
        method = summary.get("is_living_restore_method", "")
        restored = int(summary.get("is_living_restored_count") or 0)
        amenities_ctx = bool(summary.get("floor_context_amenities_input_path"))
        roads_ctx = bool(summary.get("floor_context_roads_input_path"))

        dst = unique_root / city
        shutil.copytree(best, dst)

        lines.append(
            "\t".join(
                [
                    city,
                    str(best),
                    str(len(candidates)),
                    ",".join(map(str, score)),
                    str(method),
                    str(restored),
                    str(amenities_ctx),
                    str(roads_ctx),
                ]
            )
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _needs_rerun(city_dir: Path) -> tuple[bool, str]:
    derived = city_dir / "derived_layers"
    summary_path = derived / "buildings_floor_enriched_summary.json"
    enriched = derived / "buildings_floor_enriched.parquet"

    if not enriched.exists():
        return True, "missing_buildings_floor_enriched"
    if not summary_path.exists():
        return True, "missing_summary"

    summary = _load_json(summary_path) or {}
    method = summary.get("is_living_restore_method")
    restored = int(summary.get("is_living_restored_count") or 0)
    amenities_ctx = bool(summary.get("floor_context_amenities_input_path"))
    roads_ctx = bool(summary.get("floor_context_roads_input_path"))

    if method == "osm_living_model" and restored > 0 and amenities_ctx and roads_ctx:
        return False, "ok_model_with_context"
    return True, f"method={method}|restored={restored}|amenities={amenities_ctx}|roads={roads_ctx}"


def rebuild_for_remaining(unique_root: Path, dry_run: bool, limit: int | None) -> Path:
    city_dirs = sorted([p for p in unique_root.iterdir() if p.is_dir()])

    queue: list[tuple[Path, str]] = []
    status_lines = ["city\tstatus\treason\tsummary_path"]
    for c in city_dirs:
        need, reason = _needs_rerun(c)
        if need:
            queue.append((c, reason))
        else:
            status_lines.append(
                f"{c.name}\tskipped\t{reason}\t{c / 'derived_layers' / 'buildings_floor_enriched_summary.json'}"
            )

    if limit is not None:
        queue = queue[:limit]

    for c, reason in queue:
        cmd = [
            sys.executable,
            "-m",
            "aggregated_spatial_pipeline.pipeline.run_floor_predictor_external",
            "--joint-input-dir",
            str(c),
            "--floor-ignore-missing-below-pct",
            "0",
        ]
        if dry_run:
            status_lines.append(
                f"{c.name}\tdry_run\t{reason}\t{c / 'derived_layers' / 'buildings_floor_enriched_summary.json'}"
            )
            continue

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            status = "rerun_ok"
        else:
            status = "rerun_failed"
        status_lines.append(
            f"{c.name}\t{status}\t{reason}\t{c / 'derived_layers' / 'buildings_floor_enriched_summary.json'}"
        )

    out = unique_root / "_rerun_status.tsv"
    out.write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare unique city folder and rerun floor/is_living restoration for remaining cities."
    )
    parser.add_argument(
        "--src-root",
        default="aggregated_spatial_pipeline/outputs/cities_with_street_grid_and_routes_20260412",
        help="Source root with experiment hierarchy and joint_inputs dirs.",
    )
    parser.add_argument(
        "--unique-root",
        default="aggregated_spatial_pipeline/outputs/cities_with_street_grid_and_routes_unique_20260412",
        help="Destination root with one folder per unique city.",
    )
    parser.add_argument("--skip-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    unique_root = Path(args.unique_root).resolve()

    report = build_unique(src_root, unique_root)
    print(f"unique_ready: {unique_root}")
    print(f"selection_report: {report}")

    if not args.skip_rerun:
        status = rebuild_for_remaining(unique_root, dry_run=bool(args.dry_run), limit=args.limit)
        print(f"rerun_status: {status}")


if __name__ == "__main__":
    main()

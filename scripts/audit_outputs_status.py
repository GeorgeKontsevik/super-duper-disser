#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit accumulated pipeline outputs across standalone city bundles, "
            "experimental joint_inputs roots, and batch runs."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=OUTPUTS_ROOT,
        help="Root directory with aggregated_spatial_pipeline outputs.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        default=None,
        help="Optional path for full JSON audit export.",
    )
    parser.add_argument(
        "--write-tsv",
        type=Path,
        default=None,
        help="Optional path for flat TSV audit export.",
    )
    parser.add_argument(
        "--print-cities",
        action="store_true",
        help="Print per-city rows in addition to per-root summary.",
    )
    parser.add_argument(
        "--only-problematic",
        action="store_true",
        help="When printing cities, show only non-complete rows.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_sampled_places(sample_path: Path) -> dict[str, str]:
    sampled: dict[str, str] = {}
    if not sample_path.exists():
        return sampled
    with sample_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) == 1:
                sampled[row[0].strip()] = ""
            else:
                sampled[row[0].strip()] = row[1].strip()
    return sampled


def _bool_flag(value: bool) -> str:
    return "yes" if value else "no"


def _street_pattern_summary_exists(city_dir: Path) -> bool:
    if not city_dir.exists():
        return False
    street_dir = city_dir / "street_pattern"
    if not street_dir.exists():
        return False
    return any(street_dir.rglob("*_summary.json"))


def _alternate_street_pattern_summary_exists(city_dir: Path) -> bool:
    if not city_dir.exists():
        return False
    for path in city_dir.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith("street_pattern"):
            continue
        if any(path.rglob("*_summary.json")):
            return True
    return False


def _preview_count(city_dir: Path) -> int:
    if not city_dir.exists():
        return 0
    preview_dir = city_dir / "preview_png" / "all_together"
    if not preview_dir.exists():
        return 0
    return sum(1 for path in preview_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")


def _city_signals(city_dir: Path, *, joint_manifest_path: Path | None) -> dict[str, Any]:
    city_dir_exists = city_dir.exists()
    analysis_territory_exists = (city_dir / "analysis_territory").exists()
    blocksnet_manifest_exists = (city_dir / "blocksnet_raw_osm" / "manifest.json").exists()
    intermodal_manifest_exists = (city_dir / "intermodal_graph_iduedu" / "manifest.json").exists()
    connectpt_manifest_exists = (city_dir / "connectpt_osm" / "manifest.json").exists()
    pt_street_manifest_exists = (city_dir / "pt_street_pattern_dependency" / "manifest.json").exists()
    street_pattern_summary_exists = _street_pattern_summary_exists(city_dir)
    derived_manifest_exists = (city_dir / "derived_layers" / "manifest.json").exists()
    pipeline2_manifest_exists = (city_dir / "pipeline_2" / "manifest_prepare_solver_inputs.json").exists()
    joint_manifest_exists = bool(joint_manifest_path and joint_manifest_path.exists())
    preview_png_count = _preview_count(city_dir)
    services_preview_exists = (city_dir / "preview_png" / "all_together" / "29_services_raw_all_categories.png").exists()
    manifest_collection_exists = (city_dir / "manifest_collection.json").exists()
    alternate_street_pattern_summary_exists = _alternate_street_pattern_summary_exists(city_dir)
    has_any_entries = city_dir_exists and any(city_dir.iterdir())

    if pipeline2_manifest_exists:
        category = "complete_pipeline2"
        suitability = "ready_for_solver_outputs"
        next_step = "-"
    elif joint_manifest_exists and derived_manifest_exists:
        category = "complete_phase1"
        suitability = "ready_for_pipeline2"
        next_step = "run_pipeline2_prepare_solver_inputs"
    elif derived_manifest_exists:
        category = "phase1_ready_missing_joint_manifest"
        suitability = "usable_but_manifest_repair_needed"
        next_step = "rerun run_joint to restore manifest_joint"
    elif pt_street_manifest_exists or street_pattern_summary_exists:
        category = "partial_phase1_late"
        suitability = "resume_phase1_late"
        next_step = "rerun run_joint"
    elif connectpt_manifest_exists or intermodal_manifest_exists:
        category = "partial_phase1_mid"
        suitability = "resume_phase1_mid"
        next_step = "rerun run_joint"
    elif blocksnet_manifest_exists or analysis_territory_exists:
        category = "partial_phase1_early"
        suitability = "resume_phase1_early"
        next_step = "rerun run_joint"
    elif manifest_collection_exists or alternate_street_pattern_summary_exists or has_any_entries:
        category = "non_bundle_layout"
        suitability = "experiment_specific_only"
        next_step = "inspect experiment-specific artifacts"
    else:
        category = "empty_or_unknown"
        suitability = "not_ready"
        next_step = "inspect manually"

    return {
        "analysis_territory_exists": analysis_territory_exists,
        "blocksnet_manifest_exists": blocksnet_manifest_exists,
        "intermodal_manifest_exists": intermodal_manifest_exists,
        "connectpt_manifest_exists": connectpt_manifest_exists,
        "pt_street_pattern_manifest_exists": pt_street_manifest_exists,
        "street_pattern_summary_exists": street_pattern_summary_exists,
        "derived_manifest_exists": derived_manifest_exists,
        "joint_manifest_exists": joint_manifest_exists,
        "pipeline2_manifest_exists": pipeline2_manifest_exists,
        "preview_png_count": preview_png_count,
        "services_preview_exists": services_preview_exists,
        "manifest_collection_exists": manifest_collection_exists,
        "alternate_street_pattern_summary_exists": alternate_street_pattern_summary_exists,
        "category": category,
        "suitability": suitability,
        "next_step": next_step,
    }


def _root_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record["category"] for record in records)
    return dict(sorted(counts.items()))


def _batch_records(batch_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    input_root = batch_root / "joint_inputs"
    joint_root = batch_root / "joint"
    summary_path = batch_root / "summary.json"
    sample_path = batch_root / "sampled_cities.tsv"

    summary = _read_json(summary_path) or {}
    sampled_places = _load_sampled_places(sample_path)
    rows_by_slug = {str(row.get("slug")): row for row in summary.get("results", []) if row.get("slug")}
    sampled_slugs = list(sampled_places)
    all_slugs = sampled_slugs[:]
    for slug in rows_by_slug:
        if slug not in sampled_places:
            all_slugs.append(slug)

    for slug in all_slugs:
        place = sampled_places.get(slug, "")
        row = rows_by_slug.get(slug, {})
        city_dir = input_root / slug
        joint_manifest_path = joint_root / slug / "manifest_joint.json"
        signals = _city_signals(city_dir, joint_manifest_path=joint_manifest_path)

        batch_summary_status = row.get("status")
        if not batch_summary_status:
            batch_summary_status = "not_started_in_summary"

        record = {
            "source_type": "batch",
            "source_root": str(batch_root.relative_to(OUTPUTS_ROOT)),
            "batch_name": batch_root.name,
            "slug": slug,
            "place": place or str(row.get("place") or ""),
            "city_dir": str(city_dir),
            "joint_manifest_path": str(joint_manifest_path),
            "batch_summary_status": batch_summary_status,
            "batch_summary_error": str(row.get("error") or ""),
            "sampled_in_batch": slug in sampled_places,
            **signals,
        }

        if batch_summary_status == "failed" and record["category"] in {"complete_phase1", "complete_pipeline2"}:
            record["category"] = f"{record['category']}_after_retry"
            if record["category"] == "complete_pipeline2_after_retry":
                record["suitability"] = "ready_for_solver_outputs"
                record["next_step"] = "-"

        if batch_summary_status == "not_started_in_summary" and record["category"] == "empty_or_unknown":
            record["category"] = "not_started"
            record["suitability"] = "not_started"
            record["next_step"] = "run batch or run_joint"

        records.append(record)
    return records


def _standalone_records(input_root: Path, *, joint_root: Path | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for city_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        joint_manifest_path = None
        if joint_root is not None:
            joint_manifest_path = joint_root / city_dir.name / "manifest_joint.json"
        signals = _city_signals(city_dir, joint_manifest_path=joint_manifest_path)
        records.append(
            {
                "source_type": "standalone",
                "source_root": str(input_root.relative_to(OUTPUTS_ROOT)),
                "batch_name": "",
                "slug": city_dir.name,
                "place": "",
                "city_dir": str(city_dir),
                "joint_manifest_path": str(joint_manifest_path) if joint_manifest_path else "",
                "batch_summary_status": "",
                "batch_summary_error": "",
                "sampled_in_batch": False,
                **signals,
            }
        )
    return records


def _collect_all_records(outputs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    batch_runs_root = outputs_root / "batch_runs"
    if batch_runs_root.exists():
        for batch_root in sorted(path for path in batch_runs_root.iterdir() if path.is_dir()):
            records.extend(_batch_records(batch_root))

    standalone_joint_inputs = outputs_root / "joint_inputs"
    if standalone_joint_inputs.exists():
        records.extend(_standalone_records(standalone_joint_inputs, joint_root=outputs_root / "joint"))

    for input_root in sorted(path for path in outputs_root.iterdir() if path.is_dir() and path.name.startswith("joint_inputs_")):
        records.extend(_standalone_records(input_root, joint_root=None))
    return records


def _build_report(records: list[dict[str, Any]], outputs_root: Path) -> dict[str, Any]:
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_root[record["source_root"]].append(record)

    roots_summary = {}
    for root_name, root_records in sorted(by_root.items()):
        roots_summary[root_name] = {
            "cities": len(root_records),
            "categories": _root_summary(root_records),
            "batch_statuses": dict(sorted(Counter(r["batch_summary_status"] for r in root_records if r["batch_summary_status"]).items())),
            "ready_for_pipeline2": sum(r["category"] in {"complete_phase1", "complete_phase1_after_retry"} for r in root_records),
            "ready_for_solver_outputs": sum(r["category"] in {"complete_pipeline2", "complete_pipeline2_after_retry"} for r in root_records),
            "problematic": sum(
                r["category"]
                in {
                    "partial_phase1_early",
                    "partial_phase1_mid",
                    "partial_phase1_late",
                    "phase1_ready_missing_joint_manifest",
                    "non_bundle_layout",
                    "empty_or_unknown",
                }
                or r["batch_summary_status"] == "failed"
                for r in root_records
            ),
        }

    global_categories = dict(sorted(Counter(record["category"] for record in records).items()))
    global_batch_statuses = dict(sorted(Counter(record["batch_summary_status"] for record in records if record["batch_summary_status"]).items()))
    return {
        "outputs_root": str(outputs_root),
        "total_city_records": len(records),
        "global_categories": global_categories,
        "global_batch_statuses": global_batch_statuses,
        "roots": roots_summary,
        "records": records,
    }


def _write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_tsv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_type",
        "source_root",
        "batch_name",
        "slug",
        "place",
        "category",
        "suitability",
        "next_step",
        "batch_summary_status",
        "batch_summary_error",
        "analysis_territory_exists",
        "blocksnet_manifest_exists",
        "intermodal_manifest_exists",
        "connectpt_manifest_exists",
        "pt_street_pattern_manifest_exists",
        "street_pattern_summary_exists",
        "derived_manifest_exists",
        "joint_manifest_exists",
        "pipeline2_manifest_exists",
        "preview_png_count",
        "services_preview_exists",
        "manifest_collection_exists",
        "alternate_street_pattern_summary_exists",
        "city_dir",
        "joint_manifest_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _print_report(report: dict[str, Any], *, print_cities: bool, only_problematic: bool) -> None:
    print(f"[audit] outputs_root={report['outputs_root']}")
    print(f"[audit] total_city_records={report['total_city_records']}")
    print(f"[audit] global_categories={json.dumps(report['global_categories'], ensure_ascii=False, sort_keys=True)}")
    if report["global_batch_statuses"]:
        print(f"[audit] global_batch_statuses={json.dumps(report['global_batch_statuses'], ensure_ascii=False, sort_keys=True)}")
    print()
    print("[audit] per-root summary")
    for root_name, root_summary in report["roots"].items():
        print(
            f"- {root_name}: cities={root_summary['cities']}, "
            f"categories={json.dumps(root_summary['categories'], ensure_ascii=False, sort_keys=True)}, "
            f"batch_statuses={json.dumps(root_summary['batch_statuses'], ensure_ascii=False, sort_keys=True)}"
        )

    if not print_cities:
        return

    print()
    print("[audit] city rows")
    for record in sorted(report["records"], key=lambda row: (row["source_root"], row["slug"])):
        if only_problematic and record["category"] in {"complete_phase1", "complete_phase1_after_retry", "complete_pipeline2", "complete_pipeline2_after_retry"}:
            continue
        print(
            "\t".join(
                [
                    record["source_root"],
                    record["slug"],
                    record["category"],
                    record["suitability"],
                    record["next_step"],
                    record["batch_summary_status"] or "-",
                    f"derived={_bool_flag(record['derived_manifest_exists'])}",
                    f"joint={_bool_flag(record['joint_manifest_exists'])}",
                    f"p2={_bool_flag(record['pipeline2_manifest_exists'])}",
                    f"intermodal={_bool_flag(record['intermodal_manifest_exists'])}",
                    f"connectpt={_bool_flag(record['connectpt_manifest_exists'])}",
                    f"street={_bool_flag(record['street_pattern_summary_exists'])}",
                    f"nonbundle={_bool_flag(record['manifest_collection_exists'] or record['alternate_street_pattern_summary_exists'])}",
                ]
            )
        )


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root.resolve()
    records = _collect_all_records(outputs_root)
    report = _build_report(records, outputs_root)
    _print_report(report, print_cities=bool(args.print_cities), only_problematic=bool(args.only_problematic))

    if args.write_json is not None:
        _write_json(report, args.write_json.resolve())
        print(f"\n[audit] wrote json: {args.write_json.resolve()}")
    if args.write_tsv is not None:
        _write_tsv(records, args.write_tsv.resolve())
        print(f"[audit] wrote tsv: {args.write_tsv.resolve()}")


if __name__ == "__main__":
    main()

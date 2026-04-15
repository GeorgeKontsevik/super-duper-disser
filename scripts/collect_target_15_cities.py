#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

TARGET_CITY_KEYS = {
    "uppsala": ["uppsala"],
    "tartu": ["tartu"],
    "olomouc": ["olomouc"],
    "kaunas": ["kaunas"],
    "coimbra": ["coimbra"],
    "innsbruck": ["innsbruck"],
    "bergen": ["bergen"],
    "miskolc": ["miskolc"],
    "debrecen": ["debrecen"],
    "graz": ["graz"],
    "brno": ["brno"],
    "freiburg_im_breisgau": ["freiburg", "breisgau"],
    "linz": ["linz"],
    "trieste": ["trieste"],
    "pecs": ["pecs", "p_cs"],
}


def _norm(s: str) -> str:
    s = s.lower().replace("é", "e").replace("è", "e").replace("ö", "o").replace("ü", "u")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _score_city(city_dir: Path) -> tuple[int, int, int, int, int]:
    derived = city_dir / "derived_layers"
    summary = _load_json(derived / "buildings_floor_enriched_summary.json") or {}
    method = summary.get("is_living_restore_method")
    restored = int(summary.get("is_living_restored_count") or 0)
    amenities_ctx = bool(summary.get("floor_context_amenities_input_path"))
    roads_ctx = bool(summary.get("floor_context_roads_input_path"))
    model = int(method == "osm_living_model" and restored > 0)
    model_ctx = int(model == 1 and amenities_ctx and roads_ctx)
    return (
        model_ctx,
        model,
        int((derived / "buildings_floor_enriched.parquet").exists()),
        int((derived / "buildings_floor_enriched_summary.json").exists()),
        int((city_dir / "connectpt_osm").exists()),
    )


def _discover_city_dirs(outputs_root: Path) -> list[Path]:
    out: list[Path] = []
    for ji in outputs_root.rglob("joint_inputs"):
        if not ji.is_dir():
            continue
        for c in ji.iterdir():
            if c.is_dir():
                out.append(c)
    return sorted(out)


def _match_key(city_slug: str) -> str | None:
    n = _norm(city_slug)
    for key, tokens in TARGET_CITY_KEYS.items():
        if all(tok in n for tok in tokens):
            return key
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect fixed list of 15 target cities into one folder.")
    parser.add_argument(
        "--outputs-root",
        default="aggregated_spatial_pipeline/outputs",
        help="Root to scan for */joint_inputs/<city>",
    )
    parser.add_argument(
        "--dest-root",
        default="aggregated_spatial_pipeline/outputs/target_15_cities_full_collection_20260412",
        help="Destination folder for collected cities.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    dest_root = Path(args.dest_root).resolve()

    candidates_by_key: dict[str, list[Path]] = {k: [] for k in TARGET_CITY_KEYS}
    for city_dir in _discover_city_dirs(outputs_root):
        key = _match_key(city_dir.name)
        if key is not None:
            candidates_by_key[key].append(city_dir)

    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    report = [
        "target_key\tstatus\tselected_city_slug\tselected_source\tnum_candidates\tselected_score\tselected_method\tselected_restored"
    ]

    for key in sorted(TARGET_CITY_KEYS):
        cand = candidates_by_key.get(key, [])
        if not cand:
            report.append(f"{key}\tnot_found\t\t\t0\t\t\t")
            continue

        ranked = sorted(cand, key=lambda p: (_score_city(p), str(p)), reverse=True)
        best = ranked[0]
        score = _score_city(best)
        summary = _load_json(best / "derived_layers" / "buildings_floor_enriched_summary.json") or {}
        method = summary.get("is_living_restore_method", "")
        restored = int(summary.get("is_living_restored_count") or 0)

        city_slug = best.name
        dst = dest_root / city_slug
        shutil.copytree(best, dst)

        report.append(
            "\t".join(
                [
                    key,
                    "collected",
                    city_slug,
                    str(best),
                    str(len(cand)),
                    ",".join(map(str, score)),
                    str(method),
                    str(restored),
                ]
            )
        )

    report_path = dest_root / "_collection_report.tsv"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"collection_ready: {dest_root}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()

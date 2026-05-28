from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONNECTPT_ROOT = ROOT / "connectpt"
if str(CONNECTPT_ROOT) not in sys.path:
    sys.path.insert(0, str(CONNECTPT_ROOT))

from aggregated_spatial_pipeline.pipeline.run_pt_street_pattern_dependency import (  # noqa: E402
    _overlay_pt_with_street_pattern,
    _pick_class_column,
)

CITY_DIR = ROOT / "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/bergen_norway"
OD_MATRIX = (
    ROOT
    / "yana_experiments/bare_od_route_generation/bergen_norway/"
    "bus_existing_count_meanmax_033_n17_len9_25/bus_od_matrix.csv"
)
FOCUS_CLASS = "Loops & Lollipops"
SWEEP_ROOT = ROOT / "yana_experiments/street_pattern_route_comparison/bergen_norway/cost_sweep_lollipop_20260429"
GEN_ROOT = SWEEP_ROOT / "generated"
CMP_ROOT = SWEEP_ROOT / "comparison"
LOG_ROOT = SWEEP_ROOT / "logs"
TABLE_PATH = SWEEP_ROOT / "metrics_all_variants.csv"
CONFIG_PATH = SWEEP_ROOT / "variant_configs.json"
BEST_PATH = SWEEP_ROOT / "best_variant.json"


def _variant(name: str, **overrides: float | int | str) -> dict:
    config = {
        "n_routes": 17,
        "min_route_len": 9,
        "max_route_len": 25,
        "n_samples": 50,
        "demand_time_weight": 0.0,
        "route_time_weight": 0.0,
        "median_connectivity_weight": 0.6,
        "street_pattern_weight": 0.0,
        "focus_class_weight": 0.0,
        "focus_class_presence_weight": 0.0,
        "focus_class_presence_threshold": 0.0,
        "focus_class_distribution_weight": 0.0,
        "street_pattern_diversity_weight": 0.0,
        "street_pattern_target_distribution_weight": 0.0,
        "street_pattern_target_focus_multiplier": 4.0,
        "route_overlap_weight": 0.0,
        "focus_class_overlap_weight": 0.4,
    }
    config.update(overrides)
    config["variant"] = name
    return config


def _variants() -> list[dict]:
    return [
        _variant("v01_conn060_focusoverlap040"),
        _variant("v02_conn065_focusoverlap050", median_connectivity_weight=0.65, focus_class_overlap_weight=0.50),
        _variant("v03_conn070_focusoverlap060", median_connectivity_weight=0.70, focus_class_overlap_weight=0.60),
        _variant("v04_conn080_focusoverlap060", median_connectivity_weight=0.80, focus_class_overlap_weight=0.60),
        _variant("v05_conn060_focusshare020_fo040", focus_class_weight=0.20),
        _variant("v06_conn060_focusshare040_fo040", focus_class_weight=0.40),
        _variant(
            "v07_conn070_focusshare030_fo050",
            median_connectivity_weight=0.70,
            focus_class_weight=0.30,
            focus_class_overlap_weight=0.50,
        ),
        _variant("v08_conn060_focuspresence020_fo040", focus_class_presence_weight=0.20),
        _variant("v09_conn060_focuspresence040_fo040", focus_class_presence_weight=0.40),
        _variant(
            "v10_conn060_focuspresence030_thr015_fo040",
            focus_class_presence_weight=0.30,
            focus_class_presence_threshold=0.15,
        ),
        _variant(
            "v11_conn070_focuspresence030_thr015_fo040",
            median_connectivity_weight=0.70,
            focus_class_presence_weight=0.30,
            focus_class_presence_threshold=0.15,
        ),
        _variant("v12_conn060_focusdist020_fo040", focus_class_distribution_weight=0.20),
        _variant("v13_conn060_focusdist050_fo040", focus_class_distribution_weight=0.50),
        _variant(
            "v14_conn070_focusdist030_fo050",
            median_connectivity_weight=0.70,
            focus_class_distribution_weight=0.30,
            focus_class_overlap_weight=0.50,
        ),
        _variant(
            "v15_conn060_targetdist020_mult4_fo040",
            street_pattern_target_distribution_weight=0.20,
            street_pattern_target_focus_multiplier=4.0,
        ),
        _variant(
            "v16_conn060_targetdist050_mult6_fo040",
            street_pattern_target_distribution_weight=0.50,
            street_pattern_target_focus_multiplier=6.0,
        ),
        _variant(
            "v17_conn070_targetdist030_mult8_fo050",
            median_connectivity_weight=0.70,
            street_pattern_target_distribution_weight=0.30,
            street_pattern_target_focus_multiplier=8.0,
            focus_class_overlap_weight=0.50,
        ),
        _variant("v18_conn060_routeoverlap030_fo040", route_overlap_weight=0.30),
        _variant("v19_conn060_routeoverlap060_fo040", route_overlap_weight=0.60),
        _variant(
            "v20_conn070_routeoverlap040_fo060",
            median_connectivity_weight=0.70,
            route_overlap_weight=0.40,
            focus_class_overlap_weight=0.60,
        ),
        _variant("v21_conn060_div050_fo040", street_pattern_diversity_weight=0.50),
        _variant("v22_conn060_div100_fo040", street_pattern_diversity_weight=1.00),
        _variant(
            "v23_conn070_div050_routeoverlap030_fo050",
            median_connectivity_weight=0.70,
            street_pattern_diversity_weight=0.50,
            route_overlap_weight=0.30,
            focus_class_overlap_weight=0.50,
        ),
        _variant("v24_d10_r10_conn60_fo040", demand_time_weight=0.10, route_time_weight=0.10),
        _variant("v25_d20_r10_conn60_fo040", demand_time_weight=0.20, route_time_weight=0.10),
        _variant(
            "v26_d10_r20_conn60_fo050",
            demand_time_weight=0.10,
            route_time_weight=0.20,
            focus_class_overlap_weight=0.50,
        ),
        _variant(
            "v27_d10_r10_conn50_focusshare020_fo040",
            demand_time_weight=0.10,
            route_time_weight=0.10,
            median_connectivity_weight=0.50,
            focus_class_weight=0.20,
        ),
        _variant(
            "v28_conn050_focusshare030_presence020_fo040",
            median_connectivity_weight=0.50,
            focus_class_weight=0.30,
            focus_class_presence_weight=0.20,
        ),
        _variant(
            "v29_conn050_targetdist030_routeoverlap030_fo040",
            median_connectivity_weight=0.50,
            street_pattern_target_distribution_weight=0.30,
            route_overlap_weight=0.30,
        ),
        _variant("v30_conn060_classcount020_fo040", street_pattern_weight=0.20),
    ]


FIELDNAMES = [
    "variant",
    "status",
    "is_best",
    "score",
    "demand_time_weight",
    "route_time_weight",
    "median_connectivity_weight",
    "street_pattern_weight",
    "focus_class_weight",
    "focus_class_presence_weight",
    "focus_class_presence_threshold",
    "focus_class_distribution_weight",
    "street_pattern_diversity_weight",
    "street_pattern_target_distribution_weight",
    "street_pattern_target_focus_multiplier",
    "route_overlap_weight",
    "focus_class_overlap_weight",
    "n_samples",
    "min_route_len",
    "max_route_len",
    "cost",
    "att",
    "unserved_demand_pct",
    "median_connectivity",
    "unique_route_count",
    "route_count",
    "route_len_min",
    "route_len_mean",
    "route_len_max",
    "generator_focus_stop_share",
    "generator_focus_presence_share",
    "generator_route_overlap_duplicate_edge_share",
    "generator_focus_overlap_duplicate_edge_share",
    "generator_composition_diversity",
    "generated_total_route_km",
    "generated_focus_length_share",
    "generated_mean_focus_route_share",
    "generated_median_focus_route_share",
    "generated_dominant_focus_route_share",
    "generated_edge_duplicate_use_share",
    "generated_edge_multi_route_share",
    "generated_focus_edge_duplicate_use_share",
    "generated_focus_edge_multi_route_share",
    "existing_final_od_coverage_share",
    "generated_final_od_coverage_share",
    "coverage_loss_abs",
    "coverage_loss_rel",
    "generated_final_covered_pairs",
    "existing_final_covered_pairs",
    "generated_summary",
    "comparison_summary",
    "log_path",
    "command",
]


def _write_rows(rows: list[dict]) -> None:
    completed = [r for r in rows if r.get("status") == "ok" and math.isfinite(float(r.get("score", float("inf"))))]
    best_variant = min(completed, key=lambda r: float(r["score"]))["variant"] if completed else None
    for row in rows:
        row["is_best"] = row.get("variant") == best_variant

    tmp = TABLE_PATH.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    tmp.replace(TABLE_PATH)

    if best_variant:
        best = next(row for row in rows if row.get("variant") == best_variant)
        BEST_PATH.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")


def _edge_duplicate_metrics(edges: gpd.GeoDataFrame, focus_keys: set[str]) -> dict:
    if edges.empty:
        return {
            "edge_duplicate_use_share": 0.0,
            "edge_multi_route_share": 0.0,
            "focus_edge_duplicate_use_share": 0.0,
            "focus_edge_multi_route_share": 0.0,
        }

    u = pd.to_numeric(edges["u"], errors="coerce")
    v = pd.to_numeric(edges["v"], errors="coerce")
    out = edges.copy()
    out["edge_key"] = np.minimum(u, v).astype("Int64").astype(str) + ":" + np.maximum(u, v).astype("Int64").astype(str)
    counts = out.groupby("edge_key")["route_label"].nunique()
    total_uses = float(counts.sum())
    duplicate_use_share = float((counts - 1).clip(lower=0).sum() / total_uses) if total_uses > 0 else 0.0
    multi_route_share = float((counts > 1).sum() / len(counts)) if len(counts) else 0.0

    focus_counts = counts[counts.index.isin(focus_keys)]
    focus_uses = float(focus_counts.sum())
    focus_duplicate_use_share = (
        float((focus_counts - 1).clip(lower=0).sum() / focus_uses) if focus_uses > 0 else 0.0
    )
    focus_multi_route_share = float((focus_counts > 1).sum() / len(focus_counts)) if len(focus_counts) else 0.0

    return {
        "edge_duplicate_use_share": duplicate_use_share,
        "edge_multi_route_share": multi_route_share,
        "focus_edge_duplicate_use_share": focus_duplicate_use_share,
        "focus_edge_multi_route_share": focus_multi_route_share,
    }


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(CONNECTPT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLCONFIGDIR"] = str(ROOT / ".cache/mpl-yana-cost-sweep")
    return env


def _score(*, focus_length_share: float, edge_dup: float, focus_edge_dup: float, dominant_focus_share: float, coverage_loss: float, unserved: float) -> float:
    return (
        focus_length_share
        + 0.55 * edge_dup
        + 0.75 * focus_edge_dup
        + 0.35 * dominant_focus_share
        + 6.0 * max(0.0, coverage_loss - 0.05)
        + 0.8 * max(0.0, unserved - 0.03)
    )


def _commands(config: dict, gen_dir: Path, cmp_dir: Path) -> tuple[list[str], list[str]]:
    generator = [
        sys.executable,
        "aggregated_spatial_pipeline/connectpt_data_pipeline/run_route_generator_external.py",
        "--joint-input-dir",
        str(CITY_DIR),
        "--output-dir",
        str(gen_dir),
        "--shared-preview-dir",
        str(gen_dir / "shared_preview"),
        "--modality",
        "bus",
        "--od-matrix-path",
        str(OD_MATRIX),
        "--n-routes",
        str(config["n_routes"]),
        "--min-route-len",
        str(config["min_route_len"]),
        "--max-route-len",
        str(config["max_route_len"]),
        "--n-samples",
        str(config["n_samples"]),
        "--demand-time-weight",
        str(config["demand_time_weight"]),
        "--route-time-weight",
        str(config["route_time_weight"]),
        "--median-connectivity-weight",
        str(config["median_connectivity_weight"]),
        "--street-pattern-weight",
        str(config["street_pattern_weight"]),
        "--focus-class-name",
        FOCUS_CLASS,
        "--focus-class-weight",
        str(config["focus_class_weight"]),
        "--focus-class-presence-weight",
        str(config["focus_class_presence_weight"]),
        "--focus-class-presence-threshold",
        str(config["focus_class_presence_threshold"]),
        "--focus-class-distribution-weight",
        str(config["focus_class_distribution_weight"]),
        "--street-pattern-diversity-weight",
        str(config["street_pattern_diversity_weight"]),
        "--street-pattern-target-distribution-weight",
        str(config["street_pattern_target_distribution_weight"]),
        "--street-pattern-target-focus-multiplier",
        str(config["street_pattern_target_focus_multiplier"]),
        "--route-overlap-weight",
        str(config["route_overlap_weight"]),
        "--focus-class-overlap-weight",
        str(config["focus_class_overlap_weight"]),
    ]
    comparison = [
        sys.executable,
        "scripts/run_yana_street_pattern_route_comparison.py",
        "--city-dir",
        str(CITY_DIR),
        "--generated-summary",
        str(gen_dir / "summary.json"),
        "--output-dir",
        str(cmp_dir),
        "--modality",
        "bus",
        "--route-count",
        str(config["n_routes"]),
        "--existing-route-policy",
        "all",
        "--focus-class",
        FOCUS_CLASS,
    ]
    return generator, comparison


def _load_existing_rows() -> dict[str, dict]:
    if not TABLE_PATH.exists():
        return {}
    try:
        rows = pd.read_csv(TABLE_PATH).to_dict(orient="records")
    except Exception:
        return {}
    return {str(row.get("variant")): row for row in rows if row.get("variant")}


def _collect_metrics(config: dict, gen_dir: Path, cmp_dir: Path, log_path: Path, command: list[str], cells: gpd.GeoDataFrame, class_col: str) -> dict:
    summary = json.loads((gen_dir / "summary.json").read_text(encoding="utf-8"))
    route_lengths = [float(value) for value in summary.get("route_lengths", [])]
    focus = pd.read_csv(cmp_dir / "stats/scenario_street_pattern_focus_metrics.csv")
    generated_focus = focus[focus["scenario"].astype(str).eq("generated")].iloc[0]
    coverage = pd.read_csv(cmp_dir / "stats/od_coverage_by_route_count.csv")
    final_coverage = coverage.sort_values("route_count").groupby("scenario", as_index=False).tail(1)
    existing_coverage = final_coverage[final_coverage["scenario"].astype(str).eq("existing")].iloc[0]
    generated_coverage = final_coverage[final_coverage["scenario"].astype(str).eq("generated")].iloc[0]

    generated_edges = gpd.read_parquet(cmp_dir / "stats/generated_route_edges.parquet")
    overlay, _ = _overlay_pt_with_street_pattern(generated_edges, cells, class_col=class_col)
    if overlay.empty:
        focus_keys: set[str] = set()
    else:
        focus_edge_ids = set(
            pd.to_numeric(
                overlay.loc[overlay["street_pattern_class"].astype(str).eq(FOCUS_CLASS), "edge_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )
        focus_edges = generated_edges[generated_edges["edge_id"].isin(focus_edge_ids)]
        u = pd.to_numeric(focus_edges["u"], errors="coerce")
        v = pd.to_numeric(focus_edges["v"], errors="coerce")
        focus_keys = set(
            (
                np.minimum(u, v).astype("Int64").astype(str)
                + ":"
                + np.maximum(u, v).astype("Int64").astype(str)
            ).dropna()
        )
    duplicates = _edge_duplicate_metrics(generated_edges, focus_keys)

    existing_share = float(existing_coverage["covered_share"])
    generated_share = float(generated_coverage["covered_share"])
    coverage_loss = existing_share - generated_share
    unserved = float(summary.get("unserved_demand_pct") or 0.0)
    focus_length_share = float(generated_focus["focus_class_share_weighted"])
    edge_dup = float(duplicates["edge_duplicate_use_share"])
    focus_edge_dup = float(duplicates["focus_edge_duplicate_use_share"])
    dominant_focus_share = float(generated_focus["dominant_focus_class_route_share"])

    row = dict(config)
    row.update(
        {
            "status": "ok",
            "score": _score(
                focus_length_share=focus_length_share,
                edge_dup=edge_dup,
                focus_edge_dup=focus_edge_dup,
                dominant_focus_share=dominant_focus_share,
                coverage_loss=coverage_loss,
                unserved=unserved,
            ),
            "cost": summary.get("cost"),
            "att": summary.get("att"),
            "unserved_demand_pct": summary.get("unserved_demand_pct"),
            "median_connectivity": summary.get("median_connectivity"),
            "unique_route_count": summary.get("unique_route_count"),
            "route_count": summary.get("route_count"),
            "route_len_min": min(route_lengths) if route_lengths else "",
            "route_len_mean": sum(route_lengths) / len(route_lengths) if route_lengths else "",
            "route_len_max": max(route_lengths) if route_lengths else "",
            "generator_focus_stop_share": summary.get("street_pattern_focus_class_share"),
            "generator_focus_presence_share": summary.get("street_pattern_focus_class_presence_share"),
            "generator_route_overlap_duplicate_edge_share": summary.get("route_overlap_duplicate_edge_share"),
            "generator_focus_overlap_duplicate_edge_share": summary.get("route_focus_overlap_duplicate_edge_share"),
            "generator_composition_diversity": summary.get("street_pattern_composition_diversity"),
            "generated_total_route_km": float(generated_focus["route_total_m"]) / 1000.0,
            "generated_focus_length_share": focus_length_share,
            "generated_mean_focus_route_share": float(generated_focus["mean_focus_class_share"]),
            "generated_median_focus_route_share": float(generated_focus["median_focus_class_share"]),
            "generated_dominant_focus_route_share": dominant_focus_share,
            "generated_edge_duplicate_use_share": edge_dup,
            "generated_edge_multi_route_share": float(duplicates["edge_multi_route_share"]),
            "generated_focus_edge_duplicate_use_share": focus_edge_dup,
            "generated_focus_edge_multi_route_share": float(duplicates["focus_edge_multi_route_share"]),
            "existing_final_od_coverage_share": existing_share,
            "generated_final_od_coverage_share": generated_share,
            "coverage_loss_abs": coverage_loss,
            "coverage_loss_rel": coverage_loss / existing_share if existing_share > 0 else 0.0,
            "generated_final_covered_pairs": int(generated_coverage["covered_pairs"]),
            "existing_final_covered_pairs": int(existing_coverage["covered_pairs"]),
            "generated_summary": str(gen_dir / "summary.json"),
            "comparison_summary": str(cmp_dir / "summary.json"),
            "log_path": str(log_path),
            "command": " ".join(command),
        }
    )
    return row


def main() -> None:
    for path in [GEN_ROOT, CMP_ROOT, LOG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    variants = _variants()
    CONFIG_PATH.write_text(json.dumps(variants, indent=2, ensure_ascii=False), encoding="utf-8")
    rows_by_variant = _load_existing_rows()
    cells = gpd.read_file(CITY_DIR / "street_pattern/bergen_norway/predicted_cells.geojson")
    class_col = _pick_class_column(cells, None)

    for index, config in enumerate(variants, 1):
        name = str(config["variant"])
        if rows_by_variant.get(name, {}).get("status") == "ok":
            print(f"[{index:02d}/{len(variants):02d}] skip existing ok {name}", flush=True)
            continue

        gen_dir = GEN_ROOT / name
        cmp_dir = CMP_ROOT / name
        gen_dir.mkdir(parents=True, exist_ok=True)
        cmp_dir.mkdir(parents=True, exist_ok=True)
        log_path = LOG_ROOT / f"{name}.log"
        generator, comparison = _commands(config, gen_dir, cmp_dir)
        row = dict(config)
        row.update({"status": "failed", "is_best": False, "score": float("inf"), "log_path": str(log_path), "command": " ".join(generator)})

        if (gen_dir / "summary.json").exists() and (cmp_dir / "summary.json").exists():
            try:
                row = _collect_metrics(config, gen_dir, cmp_dir, log_path, generator, cells, class_col)
                rows_by_variant[name] = row
                ordered_rows = [rows_by_variant[item["variant"]] for item in variants if item["variant"] in rows_by_variant]
                _write_rows(ordered_rows)
                print(f"[{index:02d}/{len(variants):02d}] collected existing {name}: score={row['score']:.4f}", flush=True)
                continue
            except Exception as exc:
                print(f"[{index:02d}/{len(variants):02d}] existing outputs incomplete for {name}: {exc}", flush=True)

        print(f"[{index:02d}/{len(variants):02d}] running {name}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(generator) + "\n")
                result = subprocess.run(generator, cwd=ROOT, env=_env(), stdout=log, stderr=subprocess.STDOUT, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"generator failed rc={result.returncode}")
                log.write("\n$ " + " ".join(comparison) + "\n")
                result = subprocess.run(comparison, cwd=ROOT, env=_env(), stdout=log, stderr=subprocess.STDOUT, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"comparison failed rc={result.returncode}")
            row = _collect_metrics(config, gen_dir, cmp_dir, log_path, generator, cells, class_col)
            print(
                f"[{index:02d}/{len(variants):02d}] ok {name}: score={row['score']:.4f}, "
                f"loops_len={row['generated_focus_length_share']:.3f}, "
                f"edge_dup={row['generated_edge_duplicate_use_share']:.3f}, "
                f"focus_dup={row['generated_focus_edge_duplicate_use_share']:.3f}, "
                f"od={row['generated_final_od_coverage_share']:.3f}, "
                f"loss={row['coverage_loss_abs']:.3f}",
                flush=True,
            )
        except Exception as exc:
            row["status"] = f"failed: {exc}"
            print(f"[{index:02d}/{len(variants):02d}] FAILED {name}: {exc}", flush=True)

        rows_by_variant[name] = row
        ordered_rows = [rows_by_variant[item["variant"]] for item in variants if item["variant"] in rows_by_variant]
        _write_rows(ordered_rows)

    rows = [rows_by_variant[item["variant"]] for item in variants if item["variant"] in rows_by_variant]
    _write_rows(rows)
    completed = [row for row in rows if row.get("status") == "ok"]
    print(f"done: {len(completed)}/{len(variants)} ok", flush=True)
    if completed:
        best = min(completed, key=lambda row: float(row["score"]))
        print(f"best: {best['variant']} score={best['score']} table={TABLE_PATH}", flush=True)


if __name__ == "__main__":
    main()

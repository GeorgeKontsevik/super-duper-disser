#!/usr/bin/env python3
"""Watch training checkpoints and evaluate them on the Bergen 17-route task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITY_DIR = ROOT / "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/bergen_norway"
DEFAULT_OD = (
    ROOT
    / "yana_experiments/bare_od_route_generation/bergen_norway/"
    / "bus_existing_count_meanmax_033_n17_len9_25/bus_od_matrix.csv"
)
DEFAULT_BASELINE_COMPARE = (
    ROOT
    / "yana_experiments/street_pattern_route_comparison/bergen_norway/"
    / "bus_conn060_loopoverlap040_n17_len9_25"
)
DEFAULT_BASELINE_SUMMARY = (
    ROOT
    / "yana_experiments/bare_od_route_generation/bergen_norway/"
    / "bus_conn060_loopoverlap040_n17_len9_25/summary.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "yana_experiments/real_morph_training_eval/bergen_norway"
PYTHON = ROOT / "connectpt/.venv/bin/python"
CHECKPOINT_RE = re.compile(r"iter(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--city-dir", type=Path, default=DEFAULT_CITY_DIR)
    parser.add_argument("--od-matrix-path", type=Path, default=DEFAULT_OD)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--baseline-compare-dir", type=Path, default=DEFAULT_BASELINE_COMPARE)
    parser.add_argument("--run-suffix", default="conn060_loopoverlap040_n17_len9_25")
    parser.add_argument("--comparison-label-prefix", default="trained iter")
    parser.add_argument("--demand-time-weight", type=float, default=0.0)
    parser.add_argument("--route-time-weight", type=float, default=0.0)
    parser.add_argument("--median-connectivity-weight", type=float, default=0.6)
    parser.add_argument("--street-pattern-weight", type=float, default=0.0)
    parser.add_argument("--focus-class-weight", type=float, default=0.0)
    parser.add_argument("--focus-class-presence-weight", type=float, default=0.0)
    parser.add_argument("--focus-class-distribution-weight", type=float, default=0.0)
    parser.add_argument("--street-pattern-diversity-weight", type=float, default=0.0)
    parser.add_argument("--street-pattern-target-distribution-weight", type=float, default=0.0)
    parser.add_argument("--street-pattern-target-focus-multiplier", type=float, default=4.0)
    parser.add_argument("--route-overlap-weight", type=float, default=0.0)
    parser.add_argument("--focus-class-overlap-weight", type=float, default=0.4)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--stop-after-iter", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(ROOT), str(ROOT / "connectpt")]
    current = env.get("PYTHONPATH")
    if current:
        paths.append(current)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["MPLCONFIGDIR"] = str(ROOT / ".cache/mpl-real-morph-eval")
    return env


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, env=_env(), check=True)


def _iter_from_checkpoint(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    if not match:
        raise ValueError(f"Not an iteration checkpoint: {path}")
    return int(match.group(1))


def _checkpoint_paths(checkpoint_dir: Path) -> list[Path]:
    return sorted(
        checkpoint_dir.glob("iter*.pt"),
        key=lambda path: _iter_from_checkpoint(path),
    )


def _run_dir(args: argparse.Namespace, iteration: int) -> Path:
    return args.output_root / f"ckpt_iter{iteration:03d}_{args.run_suffix}"


def _evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path) -> Path:
    iteration = _iter_from_checkpoint(checkpoint)
    run_dir = _run_dir(args, iteration)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"[watch] eval iter{iteration}: {checkpoint}", flush=True)
        _run(
            [
                str(PYTHON),
                "-m",
                "aggregated_spatial_pipeline.connectpt_data_pipeline.run_route_generator_external",
                "--joint-input-dir",
                str(args.city_dir),
                "--modality",
                "bus",
                "--od-matrix-path",
                str(args.od_matrix_path),
                "--output-dir",
                str(run_dir),
                "--n-routes",
                "17",
                "--min-route-len",
                "9",
                "--max-route-len",
                "25",
                "--demand-time-weight",
                str(args.demand_time_weight),
                "--route-time-weight",
                str(args.route_time_weight),
                "--median-connectivity-weight",
                str(args.median_connectivity_weight),
                "--street-pattern-weight",
                str(args.street_pattern_weight),
                "--focus-class-weight",
                str(args.focus_class_weight),
                "--focus-class-presence-weight",
                str(args.focus_class_presence_weight),
                "--focus-class-distribution-weight",
                str(args.focus_class_distribution_weight),
                "--street-pattern-diversity-weight",
                str(args.street_pattern_diversity_weight),
                "--street-pattern-target-distribution-weight",
                str(args.street_pattern_target_distribution_weight),
                "--street-pattern-target-focus-multiplier",
                str(args.street_pattern_target_focus_multiplier),
                "--route-overlap-weight",
                str(args.route_overlap_weight),
                "--focus-class-overlap-weight",
                str(args.focus_class_overlap_weight),
                "--focus-class-name",
                "Loops & Lollipops",
                "--weights-path",
                str(checkpoint),
            ]
        )

    comparison_dir = run_dir / "street_pattern_comparison"
    comparison_summary = comparison_dir / "summary.json"
    if not comparison_summary.exists():
        print(f"[watch] street-pattern compare iter{iteration}", flush=True)
        _run(
            [
                str(PYTHON),
                "scripts/run_yana_street_pattern_route_comparison.py",
                "--city-dir",
                str(args.city_dir),
                "--generated-summary",
                str(summary_path),
                "--output-dir",
                str(comparison_dir),
                "--modality",
                "bus",
                "--od-matrix-path",
                str(args.od_matrix_path),
                "--route-count",
                "17",
                "--focus-class",
                "Loops & Lollipops",
            ]
        )

    rendered_dir = run_dir / "preview_png_vs_baseline"
    rendered_key = rendered_dir / "10_route_length_and_stop_mix_by_street_pattern.png"
    if not rendered_key.exists():
        print(f"[watch] render iter{iteration}", flush=True)
        _run(
            [
                str(PYTHON),
                "scripts/render_yana_bergen_segregated_style.py",
                "--baseline-run-dir",
                str(args.baseline_compare_dir),
                "--comparison-run-dir",
                str(comparison_dir),
                "--comparison-label",
                f"{args.comparison_label_prefix} {iteration}",
                "--output-dir",
                str(rendered_dir),
            ]
        )

    preview_dir = run_dir / "preview_png"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for png in rendered_dir.glob("*.png"):
        shutil.copy2(png, preview_dir / png.name)

    live_dir = args.output_root / "live_preview"
    live_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "05_existing_vs_generated_vs_street_pattern_routes.png",
        "10_route_length_and_stop_mix_by_street_pattern.png",
    ):
        src = preview_dir / name
        if src.exists():
            shutil.copy2(src, live_dir / f"latest_{name}")

    return summary_path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _progress_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline = _read_json(args.baseline_summary)
    rows.append(_summary_row(args, "baseline", baseline, None))
    pattern = f"ckpt_iter*_{args.run_suffix}/summary.json"
    for summary_path in sorted(args.output_root.glob(pattern)):
        match = re.search(r"ckpt_iter(\d+)_", str(summary_path))
        iteration = int(match.group(1)) if match else -1
        rows.append(_summary_row(args, f"iter{iteration}", _read_json(summary_path), iteration))
    return rows


def _metric_score(args: argparse.Namespace, summary: dict) -> float:
    return (
        float(args.median_connectivity_weight) * float(summary.get("median_connectivity") or 0.0)
        + float(args.route_overlap_weight) * float(summary.get("route_overlap_duplicate_edge_share") or 0.0)
        + float(args.focus_class_overlap_weight)
        * float(summary.get("route_focus_overlap_duplicate_edge_share") or 0.0)
        + float(args.focus_class_weight) * float(summary.get("street_pattern_focus_class_share") or 0.0)
    )


def _summary_row(args: argparse.Namespace, name: str, summary: dict, iteration: int | None) -> dict[str, object]:
    return {
        "name": name,
        "iteration": "" if iteration is None else iteration,
        "cost": summary.get("cost"),
        "metric_score": _metric_score(args, summary),
        "median_connectivity": summary.get("median_connectivity"),
        "loop_duplicate_edge_share": summary.get("route_focus_overlap_duplicate_edge_share"),
        "all_duplicate_edge_share": summary.get("route_overlap_duplicate_edge_share"),
        "loop_length_share": summary.get("street_pattern_focus_class_share"),
        "att": summary.get("att"),
        "unserved_demand_pct": summary.get("unserved_demand_pct"),
        "unique_route_count": summary.get("unique_route_count"),
        "route_count": summary.get("route_count"),
    }


def _write_progress(args: argparse.Namespace) -> None:
    rows = _progress_rows(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "progress.csv"
    json_path = args.output_root / "progress.json"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = rows[-1]
    print(
        "[watch] latest "
        f"{latest['name']}: cost={float(latest['cost']):.4f}, "
        f"conn={float(latest['median_connectivity']):.4f}, "
        f"loop_dup={float(latest['loop_duplicate_edge_share']):.4f}, "
        f"loop_share={float(latest['loop_length_share']):.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_root = args.output_root.resolve()
    done: set[int] = set()
    while True:
        checkpoints = _checkpoint_paths(args.checkpoint_dir)
        for checkpoint in checkpoints:
            iteration = _iter_from_checkpoint(checkpoint)
            if iteration in done:
                continue
            _evaluate_checkpoint(args, checkpoint)
            done.add(iteration)
            _write_progress(args)
        if args.once:
            break
        if args.stop_after_iter >= 0 and any(iteration >= args.stop_after_iter for iteration in done):
            break
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()

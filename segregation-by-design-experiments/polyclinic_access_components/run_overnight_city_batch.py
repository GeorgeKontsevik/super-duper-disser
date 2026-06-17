"""Run the route/service substitution experiment for eligible cities overnight."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = Path(
    "segregation-by-design-experiments/polyclinic_access_components/outputs/"
    "route_strategy_service_reduction_20260612/eligible_city_inventory.csv"
)
DEFAULT_OUT_ROOT = Path(
    "segregation-by-design-experiments/polyclinic_access_components/outputs/"
    "overnight_route_strategy_batch_20260613"
)
DIAGNOSTICS_ROOTS = [
    Path("aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics"),
    Path("aggregated_spatial_pipeline/outputs/experiments_new5_access_20260609/service_access_diagnostics"),
    Path("aggregated_spatial_pipeline/outputs/experiments_old23_access_20260609/service_access_diagnostics"),
    Path("aggregated_spatial_pipeline/outputs/experiments_old23_access_20260609_pilot/service_access_diagnostics"),
]
DEFAULT_PATTERN_ROOT = Path(
    "aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_accessibility_street_pattern"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _city_size(city_dir: Path, service: str) -> float:
    summary = city_dir / "pipeline_2" / "solver_inputs" / service / "summary.json"
    if summary.exists():
        payload = _read_json(summary)
        return float(payload.get("demand_total") or 0.0)
    return 0.0


def _diagnostics_root_for_city(city: str) -> Path | None:
    for root in DIAGNOSTICS_ROOTS:
        if (root / city / "home_to_service_access_diagnostics.parquet").exists():
            return root
    return None


def _eligible_cities(inventory: Path) -> list[dict[str, str]]:
    df = pd.read_csv(inventory)
    bus_ready = df.get("connectpt_bus_graph", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    work = df[(df["ready_core"]) & (~df["duplicate_later_source"]) & bus_ready].copy()
    source_roots = {
        "active19": Path("aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"),
        "new17": Path("aggregated_spatial_pipeline/outputs/experiments_new17_access_20260610/joint_inputs_merged"),
        "old23": Path("aggregated_spatial_pipeline/outputs/experiments_old23_access_20260609/joint_inputs_merged"),
        "new5": Path("aggregated_spatial_pipeline/outputs/experiments_new5_access_20260609/joint_inputs_merged"),
    }
    rows: list[dict[str, str]] = []
    for rec in work.to_dict("records"):
        root = source_roots[str(rec["source"])]
        city_dir = root / str(rec["city"])
        diagnostics_root = _diagnostics_root_for_city(str(rec["city"]))
        if diagnostics_root is None:
            continue
        rows.append(
            {
                "source": str(rec["source"]),
                "city": str(rec["city"]),
                "city_dir": str(city_dir),
                "diagnostics_root": str(diagnostics_root),
                "pattern_root": str(DEFAULT_PATTERN_ROOT),
                "size": str(_city_size(city_dir, "polyclinic")),
            }
        )
    return sorted(rows, key=lambda r: (float(r["size"]), r["city"]))


def _write_status(out_root: Path, rows: list[dict[str, object]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "_batch_status.tsv", sep="\t", index=False)
    (out_root / "_batch_status.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cap_thread_env(env: dict[str, str], max_threads: int) -> dict[str, str]:
    capped = dict(env)
    max_threads = max(1, int(max_threads))
    thread_env_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    requested = capped.get("CONNECTPT_MAX_THREADS")
    if requested:
        try:
            max_threads = max(1, min(max_threads, int(requested)))
        except ValueError:
            pass
    for name in thread_env_vars:
        value = capped.get(name)
        if value:
            try:
                max_threads = max(1, min(max_threads, int(value)))
            except ValueError:
                pass
    for name in thread_env_vars:
        capped[name] = str(max_threads)
    capped["CONNECTPT_MAX_THREADS"] = str(max_threads)
    return capped


def _run_city(
    *,
    city: str,
    city_dir: Path,
    out_root: Path,
    service: str,
    modality: str,
    max_routes: int,
    capacity: float,
    diagnostics_root: Path,
    pattern_root: Path,
    env: dict[str, str],
    render: bool,
) -> dict[str, object]:
    city_out = out_root / city
    log_path = city_out / "run.log"
    city_out.mkdir(parents=True, exist_ok=True)
    strategies = [
        "placement_assignment",
        "candidate_service",
        "candidate_or_existing_service",
        "existing_service",
        "general_connectivity",
    ]
    cmd = [
        sys.executable,
        "segregation-by-design-experiments/polyclinic_access_components/run_min_route_count_search.py",
        "--city-dir",
        str(city_dir),
        "--service",
        service,
        "--modality",
        modality,
        "--strategies",
        *strategies,
        "--min-routes",
        str(int(max_routes)),
        "--max-routes",
        str(int(max_routes)),
        "--capacity",
        str(float(capacity)),
        "--connectpt-max-threads",
        env.get("CONNECTPT_MAX_THREADS", "4"),
        "--placement-root-name",
        "placement_exact_target90_cap800_batch",
        "--out-root",
        str(out_root),
    ]
    start = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"START {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        recompute_cmd = [
            sys.executable,
            "segregation-by-design-experiments/polyclinic_access_components/recompute_route_strategy_access_components.py",
            "--city-dir",
            str(city_dir),
            "--experiment-city-dir",
            str(city_out),
            "--service",
            service,
            "--modality",
            modality,
            "--n-routes",
            str(int(max_routes)),
            "--strategies",
            *strategies,
            "--placement-root-name",
            "placement_exact_target90_cap800_batch",
            "--connectpt-max-threads",
            env.get("CONNECTPT_MAX_THREADS", "4"),
        ]
        log.write(f"RECOMPUTE {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(recompute_cmd)}\n")
        log.flush()
        subprocess.run(recompute_cmd, cwd=str(REPO_ROOT), env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        for png in city_out.rglob("*.png"):
            png.unlink()
        final_canvas = None
        if render:
            final_dir = out_root / "_final_canvases"
            final_dir.mkdir(parents=True, exist_ok=True)
            final_canvas = final_dir / f"{city}.png"
            render_cmd = [
                sys.executable,
                "segregation-by-design-experiments/polyclinic_access_components/render_route_strategy_gap_diagnostics.py",
                "--city",
                city,
                "--service",
                service,
                "--joint-root",
                str(city_dir.parent),
                "--experiment-root",
                str(out_root),
                "--diagnostics-root",
                str(diagnostics_root),
                "--pattern-root",
                str(pattern_root),
                "--only-final-canvas",
                "--final-canvas-out",
                str(final_canvas),
            ]
            log.write(f"RENDER {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(render_cmd)}\n")
            log.flush()
            subprocess.run(render_cmd, cwd=str(REPO_ROOT), env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            if not final_canvas.exists():
                raise FileNotFoundError(final_canvas)
        log.write(f"END {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    manifest_path = city_out / "route_count_selection_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    best = manifest.get("best") or {}
    return {
        "city": city,
        "status": "success",
        "seconds": round(time.time() - start, 1),
        "best_strategy": best.get("strategy"),
        "best_requested_routes": best.get("requested_routes"),
        "best_actual_routes": best.get("actual_routes"),
        "best_new_count": best.get("new_count"),
        "final_canvas": str(final_canvas) if final_canvas is not None else "",
        "out_dir": str(city_out),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--service", default="polyclinic")
    parser.add_argument("--modality", default="bus")
    parser.add_argument("--max-routes", type=int, default=3)
    parser.add_argument("--capacity", type=float, default=800.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--connectpt-max-threads", type=int, default=4)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    env = _cap_thread_env(os.environ, int(args.connectpt_max_threads))
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cities = _eligible_cities(args.inventory)
    if args.cities:
        wanted = set(args.cities)
        cities = [row for row in cities if row["city"] in wanted]

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "_city_order.json").write_text(json.dumps(cities, ensure_ascii=False, indent=2), encoding="utf-8")

    status_rows: list[dict[str, object]] = []
    _write_status(args.out_root, status_rows)
    for idx, row in enumerate(cities, start=1):
        city = row["city"]
        city_dir = Path(row["city_dir"]).resolve()
        base = {
            "idx": idx,
            "total": len(cities),
            "source": row["source"],
            "city": city,
            "size": float(row["size"]),
            "city_dir": str(city_dir),
        }
        if args.skip_existing and (args.out_root / city / "route_count_selection_manifest.json").exists():
            status_rows.append(
                {
                    **base,
                    "attempt": 0,
                    "status": "skipped_existing",
                    "out_dir": str((args.out_root / city).resolve()),
                }
            )
            _write_status(args.out_root, status_rows)
            continue
        attempt = 0
        while True:
            attempt += 1
            started = {**base, "attempt": attempt, "status": "running", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            status_rows.append(started)
            _write_status(args.out_root, status_rows)
            try:
                result = _run_city(
                    city=city,
                    city_dir=city_dir,
                    out_root=args.out_root.resolve(),
                    service=str(args.service),
                    modality=str(args.modality),
                    max_routes=int(args.max_routes),
                    capacity=float(args.capacity),
                    diagnostics_root=Path(row["diagnostics_root"]).resolve(),
                    pattern_root=Path(row["pattern_root"]).resolve(),
                    env=env,
                    render=not bool(args.skip_render),
                )
                status_rows[-1] = {**base, "attempt": attempt, **result}
                _write_status(args.out_root, status_rows)
                break
            except Exception as exc:  # pragma: no cover
                error_path = args.out_root / city / f"error_attempt_{attempt}.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                status_rows[-1] = {
                    **base,
                    "attempt": attempt,
                    "status": "failed",
                    "error": str(exc),
                    "error_path": str(error_path),
                }
                _write_status(args.out_root, status_rows)
                if attempt > int(args.max_retries):
                    break


if __name__ == "__main__":
    main()

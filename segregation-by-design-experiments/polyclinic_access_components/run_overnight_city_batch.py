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


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _city_size(city_dir: Path, service: str) -> float:
    summary = city_dir / "pipeline_2" / "solver_inputs" / service / "summary.json"
    if summary.exists():
        payload = _read_json(summary)
        return float(payload.get("demand_total") or 0.0)
    return 0.0


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
        rows.append(
            {
                "source": str(rec["source"]),
                "city": str(rec["city"]),
                "city_dir": str(city_dir),
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


def _run_city(
    *,
    city: str,
    city_dir: Path,
    out_root: Path,
    service: str,
    modality: str,
    max_routes: int,
    capacity: float,
    env: dict[str, str],
) -> dict[str, object]:
    city_out = out_root / city
    log_path = city_out / "run.log"
    city_out.mkdir(parents=True, exist_ok=True)
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
        "placement_assignment",
        "candidate_service",
        "candidate_or_existing_service",
        "existing_service",
        "general_connectivity",
        "--min-routes",
        "0",
        "--max-routes",
        str(int(max_routes)),
        "--capacity",
        str(float(capacity)),
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
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()

    env = os.environ.copy()
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
                    env=env,
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

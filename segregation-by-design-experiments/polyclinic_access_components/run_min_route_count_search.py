"""Search the minimum generated route count needed for service reduction.

For each route-target strategy and each route count in a range, this runner:
1. runs the existing accessibility-first route generation/recompute step;
2. runs the existing exact placement solver on the resulting accessibility;
3. selects the lexicographic best option: fewer new services first, fewer routes second.

The zero-route candidate is evaluated without ConnectPT route generation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs import _run_exact_placement_for_service


DEFAULT_CITY_DIR = Path(
    "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/brno_czechia"
)
DEFAULT_OUT_ROOT = Path(
    "segregation-by-design-experiments/polyclinic_access_components/outputs/min_route_count_search_20260612"
)


def _repo_root() -> Path:
    return REPO_ROOT


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return str(dst)


def _run_command(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


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


def _service_matrix_path(city_dir: Path, service: str) -> Path:
    return city_dir / "pipeline_2" / "solver_inputs" / service / "adj_matrix_time_min.parquet"


def _service_blocks_path(city_dir: Path, service: str) -> Path:
    return city_dir / "pipeline_2" / "solver_inputs" / service / "blocks_solver.parquet"


def _run_accessibility_first(
    *,
    city_dir: Path,
    service: str,
    modality: str,
    n_routes: int,
    strategy: str,
    placement_root_name: str | None,
    street_pattern_aware: bool,
    repo_root: Path,
    env: dict[str, str],
) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first",
        "--joint-input-dir",
        str(city_dir),
        "--services",
        service,
        "--modality",
        modality,
        "--n-routes",
        str(int(n_routes)),
        "--route-target-strategy",
        strategy,
        "--recompute-provision",
        "--no-recompute-provision-only-access-problem-services",
    ]
    if strategy == "placement_assignment":
        cmd.append("--use-placement-outputs")
        if placement_root_name:
            cmd.extend(["--placement-root-name", placement_root_name])
    if street_pattern_aware:
        cmd.append("--street-pattern-aware-route-target")
    max_threads = env.get("CONNECTPT_MAX_THREADS")
    if max_threads:
        cmd.extend(["--connectpt-max-threads", str(max_threads)])
    _run_command(cmd, cwd=repo_root, env=env)
    manifest_path = city_dir / "pipeline_2" / "accessibility_first" / "manifest_accessibility_first.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return _read_json(manifest_path)


def _snapshot_accessibility_outputs(
    city_dir: Path,
    modality: str,
    out_dir: Path,
    *,
    copy_route_outputs: bool = True,
) -> dict[str, str | None]:
    route_root = city_dir / "connectpt_routes_generator" / modality
    intermodal_snapshot = out_dir / "snapshots" / "intermodal_replaced"
    copied = {
        "manifest": _copy_if_exists(
            city_dir / "pipeline_2" / "accessibility_first" / "manifest_accessibility_first.json",
            out_dir / "accessibility_first" / "manifest_accessibility_first.json",
        ),
        "provision_after_routes": _copy_if_exists(
            city_dir / "pipeline_2" / "accessibility_first" / "provision_after_routes",
            out_dir / "accessibility_first" / "provision_after_routes",
        ),
        "connectpt_summary": None,
        "connectpt_generated_routes": None,
        "intermodal_replaced": None,
        "accessibility_recomputed": None,
    }
    if copy_route_outputs:
        copied["connectpt_summary"] = _copy_if_exists(
            route_root / "summary.json",
            out_dir / f"connectpt_{modality}_summary.json",
        )
        copied["connectpt_generated_routes"] = _copy_if_exists(
            route_root / "routes.geojson",
            out_dir / f"connectpt_{modality}_routes.geojson",
        )
        copied["intermodal_replaced"] = _copy_if_exists(
            route_root / "intermodal_replaced",
            intermodal_snapshot,
        )
        copied["accessibility_recomputed"] = _copy_if_exists(
            route_root / "accessibility_recomputed",
            out_dir / "snapshots" / "accessibility_recomputed",
        )
        copied["graph_nodes"] = _copy_if_exists(
            city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet",
            intermodal_snapshot / "graph_nodes.parquet",
        )
        copied["graph_edges_source"] = _copy_if_exists(
            city_dir / "intermodal_graph_iduedu" / "graph_edges.parquet",
            intermodal_snapshot / "graph_edges_source.parquet",
        )
    else:
        copied["graph_nodes"] = None
        copied["graph_edges_source"] = None
    return copied


def _install_placement_root_for_route_targets(
    *,
    city_dir: Path,
    service: str,
    baseline_dir: Path,
    placement_root_name: str,
) -> None:
    src = baseline_dir / "placement"
    if not src.exists():
        raise FileNotFoundError(src)
    dst = city_dir / "pipeline_2" / placement_root_name / service
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "blocks_solver_after.parquet",
        "summary_after.json",
        "selected_sites.parquet",
        "selected_sites.geojson",
        "assignment_links_after.csv",
        "provision_links_after.csv",
    ]:
        _copy_if_exists(src / name, dst / name)


def _run_placement_candidate(
    *,
    city_dir: Path,
    service: str,
    blocks_path: Path,
    matrix_path: Path,
    out_dir: Path,
    capacity: float,
    use_cache: bool,
) -> dict:
    if not blocks_path.exists():
        raise FileNotFoundError(blocks_path)
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)
    blocks = gpd.read_parquet(blocks_path)
    matrix = pd.read_parquet(matrix_path)
    preview_dir = out_dir / "preview_png"
    placement_dir = out_dir / "placement"
    result = _run_exact_placement_for_service(
        blocks,
        matrix,
        service,
        placement_dir,
        preview_dir=preview_dir,
        blocks_ref=blocks,
        boundary=blocks,
        use_genetic=False,
        progress=False,
        prefer_existing=False,
        allow_existing_expansion=False,
        capacity_mode="fixed_mean",
        fixed_new_capacity_override=float(capacity),
        use_cache=use_cache,
    )
    summary_path = Path(result["summary_after"])
    summary = _read_json(summary_path)
    summary["summary_after"] = str(summary_path)
    return summary


def _required_path(value: str | None, *, label: str) -> Path:
    if not value:
        raise FileNotFoundError(f"Missing {label} path in accessibility-first manifest")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _candidate_from_summary(
    *,
    city: str,
    strategy: str,
    requested_routes: int,
    actual_routes: int,
    placement_summary: dict,
    route_manifest: dict | None,
    out_dir: Path,
) -> dict:
    route_generation = (route_manifest or {}).get("route_generation") or {}
    after_routes = (route_manifest or {}).get("after_routes") or {}
    return {
        "city": city,
        "strategy": strategy,
        "requested_routes": int(requested_routes),
        "actual_routes": int(actual_routes),
        "new_count": int(placement_summary.get("new_count", -1)),
        "selected_count": int(placement_summary.get("selected_count", -1)),
        "capacity_added_total": float(placement_summary.get("capacity_added_total", 0.0)),
        "demand_target_total": float(placement_summary.get("demand_target_total", 0.0)),
        "demand_without_after_total": float(placement_summary.get("demand_without_after_total", 0.0)),
        "demand_left_after_total": float(placement_summary.get("demand_left_after_total", 0.0)),
        "provision_total_after": float(placement_summary.get("provision_total_after", 0.0)),
        "route_unserved_pct": route_generation.get("unserved_demand_pct"),
        "route_cost": route_generation.get("cost"),
        "after_routes_access_gap": after_routes.get("total_access_gap"),
        "candidate_dir": str(out_dir),
        "placement_summary": str(out_dir / "placement" / "summary_after.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-dir", type=Path, default=DEFAULT_CITY_DIR)
    parser.add_argument("--service", default="polyclinic")
    parser.add_argument("--modality", default="bus")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["candidate_service", "candidate_or_existing_service", "existing_service", "general_connectivity"],
    )
    parser.add_argument("--min-routes", type=int, default=0)
    parser.add_argument("--max-routes", type=int, default=3)
    parser.add_argument("--capacity", type=float, default=800.0)
    parser.add_argument("--placement-root-name", default="placement_exact_target90_cap800")
    parser.add_argument("--street-pattern-aware-route-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--connectpt-max-threads", type=int, default=4)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    repo_root = _repo_root()
    city_dir = args.city_dir.resolve()
    service = str(args.service)
    out_root = (args.out_root / city_dir.name).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    env = _cap_thread_env(os.environ, int(args.connectpt_max_threads))
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

    rows: list[dict] = []
    baseline_blocks_path = _service_blocks_path(city_dir, service)
    baseline_matrix_path = _service_matrix_path(city_dir, service)
    baseline_dir = out_root / "_baseline_no_new_routes"
    baseline_summary = _run_placement_candidate(
        city_dir=city_dir,
        service=service,
        blocks_path=baseline_blocks_path,
        matrix_path=baseline_matrix_path,
        out_dir=baseline_dir,
        capacity=float(args.capacity),
        use_cache=not args.no_cache,
    )
    if args.placement_root_name:
        _install_placement_root_for_route_targets(
            city_dir=city_dir,
            service=service,
            baseline_dir=baseline_dir,
            placement_root_name=str(args.placement_root_name),
        )
    rows.append(
        _candidate_from_summary(
            city=city_dir.name,
            strategy="baseline_no_routes",
            requested_routes=0,
            actual_routes=0,
            placement_summary=baseline_summary,
            route_manifest=None,
            out_dir=baseline_dir,
        )
    )

    for strategy in args.strategies:
        for n_routes in range(max(1, int(args.min_routes)), int(args.max_routes) + 1):
            candidate_dir = out_root / str(strategy) / f"routes_{n_routes}"
            route_manifest = _run_accessibility_first(
                city_dir=city_dir,
                service=service,
                modality=str(args.modality),
                n_routes=n_routes,
                strategy=str(strategy),
                placement_root_name=str(args.placement_root_name) if args.placement_root_name else None,
                street_pattern_aware=bool(args.street_pattern_aware_route_target),
                repo_root=repo_root,
                env=env,
            )
            route_generation = route_manifest.get("route_generation") or {}
            route_skipped = bool(route_generation.get("skipped"))
            snapshots = _snapshot_accessibility_outputs(
                city_dir,
                str(args.modality),
                candidate_dir,
                copy_route_outputs=not route_skipped,
            )
            actual_routes = 0 if route_skipped else int(route_generation.get("route_count") or n_routes)
            if route_skipped:
                matrix_path = baseline_matrix_path
                blocks_path = baseline_blocks_path
            else:
                recomputed = route_generation.get("recomputed_accessibility") or {}
                matrix_path = _required_path(recomputed.get("matrix_path"), label="recomputed matrix")
                provision_after_routes = route_manifest.get("provision_after_routes") or {}
                service_files = (provision_after_routes.get(service) or {}).get("files") or {}
                blocks_path = _required_path(service_files.get("blocks_after_routes"), label="blocks_after_routes")
            placement_summary = _run_placement_candidate(
                city_dir=city_dir,
                service=service,
                blocks_path=blocks_path,
                matrix_path=matrix_path,
                out_dir=candidate_dir,
                capacity=float(args.capacity),
                use_cache=not args.no_cache,
            )
            row = _candidate_from_summary(
                city=city_dir.name,
                strategy=str(strategy),
                requested_routes=n_routes,
                actual_routes=actual_routes,
                placement_summary=placement_summary,
                route_manifest=route_manifest,
                out_dir=candidate_dir,
            )
            row.update({f"snapshot_{key}": value for key, value in snapshots.items()})
            rows.append(row)

            candidate_manifest = {
                "candidate": row,
                "route_manifest": snapshots.get("manifest"),
                "snapshots": snapshots,
            }
            (candidate_dir / "candidate_manifest.json").write_text(
                json.dumps(candidate_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["new_count", "actual_routes", "requested_routes", "strategy"]).reset_index(drop=True)
    best = summary.iloc[0].to_dict()
    summary_path = out_root / "route_count_selection_summary.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "city": city_dir.name,
        "service": service,
        "capacity": float(args.capacity),
        "min_routes": int(args.min_routes),
        "max_routes": int(args.max_routes),
        "objective": "minimize new_count, then minimize actual_routes, then requested_routes",
        "best": best,
        "summary_csv": str(summary_path),
    }
    (out_root / "route_count_selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_root": str(out_root), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

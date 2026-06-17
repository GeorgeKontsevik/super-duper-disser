"""Recompute home-to-service access components after each route strategy.

The route-generation pipeline writes updated ConnectPT graph/matrix outputs to a
shared city directory. This runner reruns one strategy at a time, snapshots the
updated graph immediately, and then computes building-level access diagnostics
from that snapshot so later strategies cannot overwrite the evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CITY_DIR = Path(
    "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/brno_czechia"
)
DEFAULT_EXPERIMENT_CITY_DIR = Path(
    "segregation-by-design-experiments/polyclinic_access_components/outputs/"
    "route_strategy_service_reduction_20260612/brno_czechia"
)
DEFAULT_STRATEGIES = [
    "placement_assignment",
    "general_connectivity",
    "existing_service",
    "candidate_service",
    "candidate_or_existing_service",
]


def _copy_path(src: Path, dst: Path) -> str | None:
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


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _run_accessibility_first(
    *,
    city_dir: Path,
    strategy: str,
    service: str,
    modality: str,
    n_routes: int,
    placement_root_name: str,
    street_pattern_aware: bool,
    connectpt_max_threads: int,
    env: dict[str, str],
) -> None:
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
        "--connectpt-max-threads",
        str(int(connectpt_max_threads)),
    ]
    if strategy == "placement_assignment":
        cmd.extend(["--use-placement-outputs", "--placement-root-name", placement_root_name])
    if street_pattern_aware:
        cmd.append("--street-pattern-aware-route-target")
    _run(cmd, cwd=REPO_ROOT, env=env)


def _snapshot_after_route_outputs(city_dir: Path, strategy_dir: Path, modality: str) -> dict[str, str | None]:
    route_root = city_dir / "connectpt_routes_generator" / modality
    snapshot_root = strategy_dir / "snapshots"
    intermodal_snapshot = snapshot_root / "intermodal_replaced"
    copied = {
        "accessibility_first_manifest": _copy_path(
            city_dir / "pipeline_2/accessibility_first/manifest_accessibility_first.json",
            snapshot_root / "accessibility_first/manifest_accessibility_first.json",
        ),
        "provision_after_routes": _copy_path(
            city_dir / "pipeline_2/accessibility_first/provision_after_routes",
            snapshot_root / "accessibility_first/provision_after_routes",
        ),
        "connectpt_summary": _copy_path(route_root / "summary.json", snapshot_root / f"connectpt_{modality}_summary.json"),
        "connectpt_routes": _copy_path(route_root / "routes.geojson", snapshot_root / f"connectpt_{modality}_routes.geojson"),
        "intermodal_replaced": _copy_path(route_root / "intermodal_replaced", intermodal_snapshot),
        "accessibility_recomputed": _copy_path(route_root / "accessibility_recomputed", snapshot_root / "accessibility_recomputed"),
    }
    # intermodal_replaced keeps graph.pkl but not the node table needed for
    # building/service snapping; route generation does not renumber graph nodes.
    copied["graph_nodes"] = _copy_path(
        city_dir / "intermodal_graph_iduedu/graph_nodes.parquet",
        intermodal_snapshot / "graph_nodes.parquet",
    )
    copied["graph_edges_source"] = _copy_path(
        city_dir / "intermodal_graph_iduedu/graph_edges.parquet",
        intermodal_snapshot / "graph_edges_source.parquet",
    )
    return copied


def _recompute_building_pt_components(
    *,
    city_dir: Path,
    strategy: str,
    intermodal_dir: Path,
    service: str,
    pt_root: Path,
    diagnostics_root: Path,
    env: dict[str, str],
) -> None:
    _run(
        [
            sys.executable,
            "scripts/run_residential_to_services_pt_top1.py",
            "--joint-inputs-root",
            str(city_dir.parent),
            "--cities",
            city_dir.name,
            "--services",
            service,
            "--min-walk-min",
            "15",
            "--out-root",
            str(pt_root / strategy),
            "--intermodal-dir",
            str(intermodal_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    empty_lt_root = diagnostics_root / "_empty_pt_walk_lt15"
    empty_lt_root.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "scripts/classify_service_access_failures.py",
            "--joint-inputs-root",
            str(city_dir.parent),
            "--cities",
            city_dir.name,
            "--pt-walk-lt-root",
            str(empty_lt_root),
            "--pt-walk-ge-root",
            str(pt_root / strategy),
            "--out-root",
            str(diagnostics_root / strategy),
        ],
        cwd=REPO_ROOT,
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-dir", type=Path, default=DEFAULT_CITY_DIR)
    parser.add_argument("--experiment-city-dir", type=Path, default=DEFAULT_EXPERIMENT_CITY_DIR)
    parser.add_argument("--service", default="polyclinic")
    parser.add_argument("--modality", default="bus")
    parser.add_argument("--n-routes", type=int, default=3)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--placement-root-name", default="placement_exact_target90_cap800")
    parser.add_argument("--street-pattern-aware-route-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--connectpt-max-threads", type=int, default=4)
    args = parser.parse_args()

    city_dir = args.city_dir.resolve()
    experiment_city_dir = args.experiment_city_dir.resolve()
    diagnostics_root = experiment_city_dir / "gap_diagnostics" / "recomputed_access_components"
    pt_root = experiment_city_dir / "gap_diagnostics" / "recomputed_pt_top1_walk15plus"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    pt_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    rows: list[dict[str, object]] = []
    for strategy in args.strategies:
        strategy_dir = diagnostics_root / strategy
        print(f"== strategy {strategy}: rerun routes and snapshot ==", flush=True)
        _run_accessibility_first(
            city_dir=city_dir,
            strategy=strategy,
            service=str(args.service),
            modality=str(args.modality),
            n_routes=int(args.n_routes),
            placement_root_name=str(args.placement_root_name),
            street_pattern_aware=bool(args.street_pattern_aware_route_target),
            connectpt_max_threads=int(args.connectpt_max_threads),
            env=env,
        )
        copied = _snapshot_after_route_outputs(city_dir, strategy_dir, str(args.modality))
        intermodal_dir = strategy_dir / "snapshots/intermodal_replaced"
        _recompute_building_pt_components(
            city_dir=city_dir,
            strategy=strategy,
            intermodal_dir=intermodal_dir,
            service=str(args.service),
            pt_root=pt_root,
            diagnostics_root=diagnostics_root,
            env=env,
        )
        diag_path = diagnostics_root / strategy / city_dir.name / "home_to_service_access_diagnostics.parquet"
        rows.append(
            {
                "strategy": strategy,
                "diagnostics_path": str(diag_path),
                "diagnostics_exists": diag_path.exists(),
                **{f"snapshot_{key}": value for key, value in copied.items()},
            }
        )

    manifest = {
        "city": city_dir.name,
        "service": str(args.service),
        "modality": str(args.modality),
        "n_routes": int(args.n_routes),
        "strategies": rows,
    }
    manifest_path = diagnostics_root / "recompute_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "strategies": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

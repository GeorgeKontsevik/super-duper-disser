from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.runtime_config import configure_logger, repo_mplconfigdir
from aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs import (
    SUPPORTED_SERVICES,
    PROVISION_ENGINE_NAME,
    _plot_service_lp_preview,
    _run_arctic_lp_provision,
)


def _configure_logging() -> None:
    configure_logger("[pipeline_2_accessibility_first]")


def _log(message: str) -> None:
    logger.bind(tag="[pipeline_2_accessibility_first]").info(message)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def _resolve_city_dir(args: argparse.Namespace) -> Path:
    if args.joint_input_dir:
        return Path(args.joint_input_dir).resolve()
    if args.place:
        slug = _slugify(str(args.place))
        return (
            Path(__file__).resolve().parents[2]
            / "aggregated_spatial_pipeline"
            / "outputs"
            / "joint_inputs"
            / slug
        ).resolve()
    raise ValueError("Provide either --joint-input-dir or --place.")


def _ensure_services(services: list[str]) -> list[str]:
    normalized = [str(s).strip().lower() for s in services]
    unknown = [s for s in normalized if s not in SUPPORTED_SERVICES]
    if unknown:
        raise ValueError(f"Unsupported services: {', '.join(unknown)}")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline_2 accessibility-first step: aggregate multi-service suffering blocks, "
            "generate a small set of ConnectPT routes, and recompute provision on updated accessibility."
        )
    )
    parser.add_argument("--joint-input-dir", default=None)
    parser.add_argument("--place", default=None)
    parser.add_argument("--services", nargs="+", default=list(SUPPORTED_SERVICES))
    parser.add_argument("--modality", default="bus")
    parser.add_argument("--n-routes", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--generate-routes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--align-route-len-to-existing-mean-max",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For ConnectPT route generation, align route-length bounds to existing routes: "
            "min >= ceil(mean existing stops), max <= max existing stops."
        ),
    )
    parser.add_argument("--replace-in-intermodal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recompute-accessibility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recompute-provision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--recompute-provision-only-access-problem-services",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recompute provision after routes only for services where baseline accessibility "
            "gap (demand_without) is positive."
        ),
    )
    parser.add_argument("--min-route-len", type=int, default=2)
    parser.add_argument("--max-route-len", type=int, default=8)
    parser.add_argument("--demand-time-weight", type=float, default=0.33)
    parser.add_argument("--route-time-weight", type=float, default=0.33)
    parser.add_argument("--median-connectivity-weight", type=float, default=0.33)
    return parser.parse_args()


def _service_solver_path(city_dir: Path, service: str) -> Path:
    return city_dir / "pipeline_2" / "solver_inputs" / service / "blocks_solver.parquet"


def _load_baseline_solver_blocks(city_dir: Path, services: list[str]) -> dict[str, gpd.GeoDataFrame]:
    loaded: dict[str, gpd.GeoDataFrame] = {}
    for service in services:
        path = _service_solver_path(city_dir, service)
        if not path.exists():
            raise FileNotFoundError(f"Missing solver output for service [{service}]: {path}")
        gdf = gpd.read_parquet(path)
        if gdf.empty:
            raise ValueError(f"Empty solver output for service [{service}]: {path}")
        loaded[service] = gdf
    return loaded


def _normalize_block_id_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out.index = out.index.astype(str)
    return out


def _build_suffering_frame(service_blocks: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    merged: gpd.GeoDataFrame | None = None
    for service, gdf in service_blocks.items():
        work = _normalize_block_id_index(gdf)
        work[f"{service}_access_gap"] = pd.to_numeric(work.get("demand_without", 0.0), errors="coerce").fillna(0.0)
        work[f"{service}_capacity_gap"] = pd.to_numeric(work.get("demand_left", 0.0), errors="coerce").fillna(0.0)
        work[f"{service}_demand"] = pd.to_numeric(work.get("demand", 0.0), errors="coerce").fillna(0.0)
        cols = ["geometry", f"{service}_access_gap", f"{service}_capacity_gap", f"{service}_demand"]
        part = work[cols].copy()
        if merged is None:
            merged = part
        else:
            merged = merged.join(part.drop(columns=["geometry"]), how="outer")
            if "geometry" in merged.columns and merged["geometry"].isna().any():
                fill_geom = part["geometry"]
                merged["geometry"] = merged["geometry"].where(merged["geometry"].notna(), fill_geom)

    if merged is None:
        raise RuntimeError("No service blocks were loaded.")

    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=next(iter(service_blocks.values())).crs)
    access_cols = [c for c in merged.columns if c.endswith("_access_gap")]
    capacity_cols = [c for c in merged.columns if c.endswith("_capacity_gap")]
    merged["total_access_gap"] = merged[access_cols].sum(axis=1).fillna(0.0)
    merged["total_capacity_gap"] = merged[capacity_cols].sum(axis=1).fillna(0.0)
    merged["total_gap"] = merged["total_access_gap"] + merged["total_capacity_gap"]
    merged["services_with_gap"] = (
        (merged[access_cols].fillna(0.0) > 0).sum(axis=1)
        + (merged[capacity_cols].fillna(0.0) > 0).sum(axis=1)
    )
    merged["gap_type"] = np.where(
        (merged["total_access_gap"] > 0) & (merged["total_capacity_gap"] > 0),
        "both_gaps",
        np.where(
            merged["total_access_gap"] > 0,
            "accessibility_only",
            np.where(merged["total_capacity_gap"] > 0, "capacity_only", "no_gap"),
        ),
    )
    merged["block_id"] = merged.index.astype(str)
    return merged


def _build_service_gap_summary(suffering: gpd.GeoDataFrame, services: list[str]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for service in services:
        access = pd.to_numeric(suffering.get(f"{service}_access_gap", 0.0), errors="coerce").fillna(0.0)
        capacity = pd.to_numeric(suffering.get(f"{service}_capacity_gap", 0.0), errors="coerce").fillna(0.0)
        demand = pd.to_numeric(suffering.get(f"{service}_demand", 0.0), errors="coerce").fillna(0.0)
        demand_total = float(demand.sum())
        summary[service] = {
            "demand_total": demand_total,
            "accessibility_gap_total": float(access.sum()),
            "capacity_gap_total": float(capacity.sum()),
            "accessibility_gap_pct": float((access.sum() / demand_total * 100.0) if demand_total > 0 else 0.0),
            "capacity_gap_pct": float((capacity.sum() / demand_total * 100.0) if demand_total > 0 else 0.0),
        }
    return summary


def _save_suffering_outputs(
    suffering: gpd.GeoDataFrame,
    *,
    output_root: Path,
    services: list[str],
    top_k: int,
    label: str,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    suffering_path = output_root / f"suffering_blocks_{label}.parquet"
    csv_path = output_root / f"suffering_blocks_{label}.csv"
    top_access_path = output_root / f"top_blocks_access_gap_{label}.csv"
    top_total_path = output_root / f"top_blocks_total_gap_{label}.csv"
    summary_path = output_root / f"suffering_summary_{label}.json"

    suffering_to_save = suffering.copy()
    suffering_to_save.to_parquet(suffering_path)
    pd.DataFrame(suffering_to_save.drop(columns="geometry", errors="ignore")).to_csv(csv_path, index=False)

    top_access = suffering_to_save[suffering_to_save["total_access_gap"] > 0].sort_values("total_access_gap", ascending=False).head(int(top_k))
    top_total = suffering_to_save[suffering_to_save["total_gap"] > 0].sort_values("total_gap", ascending=False).head(int(top_k))
    pd.DataFrame(top_access.drop(columns="geometry", errors="ignore")).to_csv(top_access_path, index=False)
    pd.DataFrame(top_total.drop(columns="geometry", errors="ignore")).to_csv(top_total_path, index=False)

    summary = {
        "label": label,
        "blocks_total": int(len(suffering_to_save)),
        "blocks_with_any_gap": int((pd.to_numeric(suffering_to_save["total_gap"], errors="coerce").fillna(0.0) > 0).sum()),
        "total_access_gap": float(pd.to_numeric(suffering_to_save["total_access_gap"], errors="coerce").fillna(0.0).sum()),
        "total_capacity_gap": float(pd.to_numeric(suffering_to_save["total_capacity_gap"], errors="coerce").fillna(0.0).sum()),
        "service_gaps": _build_service_gap_summary(suffering_to_save, services),
        "top_access_block_ids": top_access["block_id"].astype(str).tolist(),
        "top_total_block_ids": top_total["block_id"].astype(str).tolist(),
        "files": {
            "suffering_parquet": str(suffering_path),
            "suffering_csv": str(csv_path),
            "top_access_csv": str(top_access_path),
            "top_total_csv": str(top_total_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["files"]["summary_json"] = str(summary_path)
    return summary


def _run_route_generator(
    *,
    repo_root: Path,
    city_dir: Path,
    args: argparse.Namespace,
) -> dict:
    python_path = repo_root / ".venv" / "bin" / "python"
    if not python_path.exists():
        raise FileNotFoundError(f"Missing project runtime: {python_path}")

    min_route_len = int(args.min_route_len)
    max_route_len = int(args.max_route_len)
    route_len_policy: dict | None = None

    if args.align_route_len_to_existing_mean_max:
        stats = _compute_existing_route_stop_stats(city_dir=city_dir, modality=str(args.modality))
        if stats is not None:
            req_min = max(2, min_route_len)
            req_max = max(req_min, max_route_len)
            mean_floor_min = max(2, int(np.ceil(float(stats["mean_stops"]) - 1e-9)))
            existing_max = max(2, int(stats["max_stops"]))
            eff_min = max(req_min, mean_floor_min)
            eff_max = min(req_max, existing_max)
            conflict = False
            if eff_max < eff_min:
                conflict = True
                if eff_min <= existing_max:
                    eff_max = eff_min
                else:
                    eff_min = existing_max
                    eff_max = existing_max

            min_route_len = int(eff_min)
            max_route_len = int(eff_max)
            route_len_policy = {
                "enabled": True,
                "requested_min_route_len": int(req_min),
                "requested_max_route_len": int(req_max),
                "existing_route_count": int(stats["route_count"]),
                "existing_mean_stops": float(stats["mean_stops"]),
                "existing_median_stops": float(stats["median_stops"]),
                "existing_min_stops": int(stats["min_stops"]),
                "existing_max_stops": int(stats["max_stops"]),
                "effective_min_route_len": int(min_route_len),
                "effective_max_route_len": int(max_route_len),
                "requested_bounds_conflicted_with_existing_stats": bool(conflict),
            }
            _log(
                "Route-length policy aligned to existing routes: "
                f"requested=[{req_min},{req_max}], "
                f"existing_mean={stats['mean_stops']:.2f}, existing_max={existing_max}, "
                f"effective=[{min_route_len},{max_route_len}]"
            )
        else:
            route_len_policy = {"enabled": True, "stats_available": False}
            _log("Route-length policy alignment requested, but existing route stats are unavailable; using requested bounds.")
    else:
        route_len_policy = {"enabled": False}

    cmd = [
        str(python_path),
        "-m",
        "aggregated_spatial_pipeline.connectpt_data_pipeline.run_route_generator_external",
        "--joint-input-dir",
        str(city_dir),
        "--modality",
        str(args.modality),
        "--n-routes",
        str(int(args.n_routes)),
        "--min-route-len",
        str(int(min_route_len)),
        "--max-route-len",
        str(int(max_route_len)),
        "--demand-time-weight",
        str(float(args.demand_time_weight)),
        "--route-time-weight",
        str(float(args.route_time_weight)),
        "--median-connectivity-weight",
        str(float(args.median_connectivity_weight)),
    ]
    if args.replace_in_intermodal:
        cmd.append("--replace-in-intermodal")
    if args.recompute_accessibility:
        cmd.append("--recompute-accessibility")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "connectpt")])
    env["MPLCONFIGDIR"] = repo_mplconfigdir("mpl-pipeline2-access-first", root=repo_root)
    _log(f"Running ConnectPT route generator: modality={args.modality}, n_routes={args.n_routes}")
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)

    summary_path = city_dir / "connectpt_routes_generator" / str(args.modality) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing route-generator summary after run: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pipeline_route_length_policy"] = route_len_policy
    return summary


def _compute_existing_route_stop_stats(city_dir: Path, modality: str) -> dict | None:
    intermodal_graph_path = city_dir / "intermodal_graph_iduedu" / "graph.pkl"
    if not intermodal_graph_path.exists():
        return None
    with intermodal_graph_path.open("rb") as fh:
        intermodal_graph = pickle.load(fh)

    route_nodes: dict[str, set[int]] = {}
    for u, v, _key, data in intermodal_graph.edges(keys=True, data=True):
        if str(data.get("type")) != str(modality):
            continue
        route_key = data.get("name") or data.get("route")
        if route_key is None:
            continue
        key = str(route_key)
        if key not in route_nodes:
            route_nodes[key] = set()
        route_nodes[key].add(int(u))
        route_nodes[key].add(int(v))

    if not route_nodes:
        return None

    counts = np.array([len(nodes_set) for nodes_set in route_nodes.values()], dtype=float)
    return {
        "route_count": int(len(counts)),
        "mean_stops": float(np.mean(counts)),
        "median_stops": float(np.median(counts)),
        "min_stops": int(np.min(counts)),
        "max_stops": int(np.max(counts)),
    }


def _recompute_provision_after_routes(
    *,
    city_dir: Path,
    services: list[str],
    route_summary: dict,
    output_root: Path,
) -> dict[str, dict]:
    rec = route_summary.get("recomputed_accessibility") or {}
    matrix_path = rec.get("matrix_path")
    if not matrix_path:
        raise ValueError("Route summary does not provide recomputed accessibility matrix path.")
    matrix = pd.read_parquet(Path(matrix_path))

    boundary_path = city_dir / "analysis_territory" / "buffer.parquet"
    boundary = gpd.read_parquet(boundary_path) if boundary_path.exists() else None
    blocks_ref_path = city_dir / "derived_layers" / "blocks_clipped.parquet"
    blocks_ref = gpd.read_parquet(blocks_ref_path) if blocks_ref_path.exists() else None
    preview_dir = city_dir / "preview_png" / "all_together"
    out: dict[str, dict] = {}

    for service in services:
        baseline_path = _service_solver_path(city_dir, service)
        baseline = _normalize_block_id_index(gpd.read_parquet(baseline_path))
        mx = matrix.copy()
        mx.index = mx.index.astype(str)
        mx.columns = mx.columns.astype(str)
        common_idx = [idx for idx in baseline.index if idx in mx.index]
        if not common_idx:
            raise ValueError(f"No common block ids for service [{service}] and recomputed matrix.")
        baseline_work = baseline.loc[common_idx].copy()
        sub_mx = mx.loc[common_idx, common_idx].copy()

        radius = float(pd.to_numeric(baseline_work.get("service_radius_min"), errors="coerce").dropna().iloc[0])
        demand_per_1000 = float(pd.to_numeric(baseline_work.get("service_demand_per_1000"), errors="coerce").dropna().iloc[0])
        recomputed, links = _run_arctic_lp_provision(
            blocks_df=baseline_work,
            accessibility_matrix=sub_mx,
            service=service,
            service_radius_min=radius,
            service_demand_per_1000=demand_per_1000,
        )

        service_out_dir = output_root / "provision_after_routes" / service
        service_out_dir.mkdir(parents=True, exist_ok=True)
        blocks_after_path = service_out_dir / "blocks_solver_after_routes.parquet"
        links_after_path = service_out_dir / "provision_links_after_routes.csv"
        summary_after_path = service_out_dir / "summary_after_routes.json"
        recomputed.to_parquet(blocks_after_path)
        links.to_csv(links_after_path, index=False)

        before_access = float(pd.to_numeric(baseline_work.get("demand_without", 0.0), errors="coerce").fillna(0.0).sum())
        before_capacity = float(pd.to_numeric(baseline_work.get("demand_left", 0.0), errors="coerce").fillna(0.0).sum())
        after_access = float(pd.to_numeric(recomputed.get("demand_without", 0.0), errors="coerce").fillna(0.0).sum())
        after_capacity = float(pd.to_numeric(recomputed.get("demand_left", 0.0), errors="coerce").fillna(0.0).sum())
        demand_total = float(pd.to_numeric(recomputed.get("demand", 0.0), errors="coerce").fillna(0.0).sum())
        summary = {
            "service": service,
            "provision_engine": PROVISION_ENGINE_NAME,
            "demand_total": demand_total,
            "accessibility_gap_before": before_access,
            "capacity_gap_before": before_capacity,
            "accessibility_gap_after": after_access,
            "capacity_gap_after": after_capacity,
            "accessibility_gap_delta": float(after_access - before_access),
            "capacity_gap_delta": float(after_capacity - before_capacity),
            "files": {
                "blocks_after_routes": str(blocks_after_path),
                "links_after_routes": str(links_after_path),
            },
        }
        summary_after_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["files"]["summary_after_routes"] = str(summary_after_path)

        preview_after_path = preview_dir / f"lp_{service}_provision_after_routes.png"
        _plot_service_lp_preview(recomputed, f"{service} after-routes", preview_after_path, blocks_ref=blocks_ref, boundary=boundary)
        summary["files"]["preview_after_routes"] = str(preview_after_path)
        out[service] = summary
    return out


def main() -> None:
    _configure_logging()
    args = parse_args()
    services = _ensure_services(args.services)
    city_dir = _resolve_city_dir(args)
    repo_root = Path(__file__).resolve().parents[2]
    output_root = city_dir / "pipeline_2" / "accessibility_first"
    output_root.mkdir(parents=True, exist_ok=True)

    _log(f"Starting accessibility-first step: city={city_dir.name}, services={services}")
    baseline_blocks = _load_baseline_solver_blocks(city_dir, services)
    suffering_baseline = _build_suffering_frame(baseline_blocks)
    baseline_summary = _save_suffering_outputs(
        suffering_baseline,
        output_root=output_root,
        services=services,
        top_k=int(args.top_k),
        label="baseline",
    )
    _log(
        "Baseline suffering prepared: "
        f"blocks_with_gap={baseline_summary['blocks_with_any_gap']}, "
        f"access_gap={baseline_summary['total_access_gap']:.1f}, "
        f"capacity_gap={baseline_summary['total_capacity_gap']:.1f}"
    )

    route_summary = None
    after_route_summaries: dict[str, dict] | None = None
    after_summary = None
    services_for_recompute: list[str] = list(services)
    skipped_services_no_access_problem: list[str] = []
    if args.recompute_provision_only_access_problem_services:
        service_gaps = baseline_summary.get("service_gaps") or {}
        services_for_recompute = [
            service
            for service in services
            if float((service_gaps.get(service) or {}).get("accessibility_gap_total", 0.0)) > 0.0
        ]
        skipped_services_no_access_problem = [service for service in services if service not in services_for_recompute]
        if services_for_recompute:
            _log(
                "Provision recompute will run only for services with accessibility gap > 0: "
                f"{services_for_recompute} (skipped={skipped_services_no_access_problem})"
            )
        else:
            _log(
                "No services with accessibility gap > 0 were found in baseline. "
                "Provision recompute is skipped."
            )

    if args.generate_routes:
        route_summary = _run_route_generator(repo_root=repo_root, city_dir=city_dir, args=args)
        if (
            args.recompute_provision
            and services_for_recompute
            and route_summary.get("recomputed_accessibility", {}).get("matrix_path")
        ):
            after_route_summaries = _recompute_provision_after_routes(
                city_dir=city_dir,
                services=services_for_recompute,
                route_summary=route_summary,
                output_root=output_root,
            )
            after_blocks = {
                service: gpd.read_parquet(Path(summary["files"]["blocks_after_routes"]))
                for service, summary in after_route_summaries.items()
            }
            suffering_after = _build_suffering_frame(after_blocks)
            after_summary = _save_suffering_outputs(
                suffering_after,
                output_root=output_root,
                services=services,
                top_k=int(args.top_k),
                label="after_routes",
            )
            _log(
                "After-routes suffering prepared: "
                f"blocks_with_gap={after_summary['blocks_with_any_gap']}, "
                f"access_gap={after_summary['total_access_gap']:.1f}, "
                f"capacity_gap={after_summary['total_capacity_gap']:.1f}"
            )

    manifest = {
        "city_dir": str(city_dir),
        "step": "pipeline_2_accessibility_first",
        "services": services,
        "provision_engine": PROVISION_ENGINE_NAME,
        "connectpt_modality": str(args.modality),
        "connectpt_n_routes": int(args.n_routes),
        "recompute_provision_only_access_problem_services": bool(args.recompute_provision_only_access_problem_services),
        "services_for_recompute": services_for_recompute,
        "services_skipped_no_access_problem": skipped_services_no_access_problem,
        "baseline": baseline_summary,
        "route_generation": route_summary,
        "provision_after_routes": after_route_summaries,
        "after_routes": after_summary,
    }
    manifest_path = output_root / "manifest_accessibility_first.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

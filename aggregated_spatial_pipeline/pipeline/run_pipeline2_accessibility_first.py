from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import Point

from aggregated_spatial_pipeline.runtime_config import configure_logger, repo_mplconfigdir
from aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs import (
    SUPPORTED_SERVICES,
    PROVISION_ENGINE_NAME,
    _plot_service_lp_preview,
    _run_arctic_lp_provision,
)
from aggregated_spatial_pipeline.visualization import apply_preview_canvas, footer_text, normalize_preview_gdf, save_preview_figure


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
    parser.add_argument(
        "--route-summary-path",
        default=None,
        help=(
            "Optional existing ConnectPT summary.json to reuse when route generation is disabled "
            "or when provision recompute should run from a previously generated route."
        ),
    )
    parser.add_argument(
        "--use-placement-outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use after-placement solver outputs as the baseline for suffering/provision recompute, "
            "and build a service-aware ConnectPT target OD from placement assignments."
        ),
    )
    parser.add_argument(
        "--placement-root-name",
        default=None,
        help=(
            "Optional placement output root under pipeline_2, e.g. placement_exact_genetic or placement_exact. "
            "When omitted, auto-resolve placement_exact_genetic first, then placement_exact."
        ),
    )
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
    parser.add_argument("--min-route-len", type=int, default=6)
    parser.add_argument("--max-route-len", type=int, default=10)
    parser.add_argument("--demand-time-weight", type=float, default=0.33)
    parser.add_argument("--route-time-weight", type=float, default=0.33)
    parser.add_argument("--median-connectivity-weight", type=float, default=0.33)
    return parser.parse_args()


def _service_solver_path(city_dir: Path, service: str) -> Path:
    return city_dir / "pipeline_2" / "solver_inputs" / service / "blocks_solver.parquet"


def _service_solver_provision_links_path(city_dir: Path, service: str) -> Path:
    return city_dir / "pipeline_2" / "solver_inputs" / service / "provision_links.csv"


def _service_placement_blocks_path(city_dir: Path, service: str, placement_root_name: str) -> Path:
    return city_dir / "pipeline_2" / placement_root_name / service / "blocks_solver_after.parquet"


def _service_placement_assignment_links_path(city_dir: Path, service: str, placement_root_name: str) -> Path:
    return city_dir / "pipeline_2" / placement_root_name / service / "assignment_links_after.csv"


def _default_route_summary_path(city_dir: Path, modality: str) -> Path:
    return city_dir / "connectpt_routes_generator" / modality / "summary.json"


def _joint_experiment_export_dir(
    *,
    city_dir: Path,
    services: list[str],
    modality: str,
    placement_root_name: str | None,
    n_routes: int,
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    outputs_root = repo_root / "aggregated_spatial_pipeline" / "outputs" / "joint_service_pt_experiments"
    service_part = "_".join(services)
    placement_part = placement_root_name or "baseline_solver_inputs"
    scenario_slug = f"{placement_part}__{modality}__routes_{int(n_routes)}__services_{service_part}"
    return outputs_root / city_dir.name / scenario_slug


def _resolve_placement_root_name(city_dir: Path, services: list[str], requested: str | None) -> str:
    candidates = [requested] if requested else ["placement_exact_genetic", "placement_exact"]
    for candidate in candidates:
        if candidate is None:
            continue
        root = city_dir / "pipeline_2" / candidate
        if all(_service_placement_blocks_path(city_dir, service, candidate).exists() for service in services):
            return candidate
    raise FileNotFoundError(
        "Could not resolve placement output root with after-placement solver files "
        f"for services={services}. requested={requested!r}"
    )


def _load_solver_blocks(
    city_dir: Path,
    services: list[str],
    *,
    use_placement_outputs: bool,
    placement_root_name: str | None,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, Path], str | None]:
    loaded: dict[str, gpd.GeoDataFrame] = {}
    paths: dict[str, Path] = {}
    resolved_placement_root = None
    if use_placement_outputs:
        resolved_placement_root = _resolve_placement_root_name(city_dir, services, placement_root_name)
    for service in services:
        if resolved_placement_root is not None:
            path = _service_placement_blocks_path(city_dir, service, resolved_placement_root)
        else:
            path = _service_solver_path(city_dir, service)
        if not path.exists():
            raise FileNotFoundError(f"Missing solver output for service [{service}]: {path}")
        gdf = gpd.read_parquet(path)
        if gdf.empty:
            raise ValueError(f"Empty solver output for service [{service}]: {path}")
        loaded[service] = gdf
        paths[service] = path
    return loaded, paths, resolved_placement_root


def _normalize_block_id_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out.index = out.index.astype(str)
    return out


def _extract_gap_series(gdf: gpd.GeoDataFrame, *candidate_cols: str) -> pd.Series:
    for column in candidate_cols:
        if column in gdf.columns:
            return pd.to_numeric(gdf.get(column, 0.0), errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=gdf.index, dtype="float64")


def _build_suffering_frame(service_blocks: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    merged: gpd.GeoDataFrame | None = None
    for service, gdf in service_blocks.items():
        work = _normalize_block_id_index(gdf)
        work[f"{service}_access_gap"] = _extract_gap_series(
            work,
            "demand_without_after_routes",
            "demand_without_after",
            "demand_without",
        )
        work[f"{service}_capacity_gap"] = _extract_gap_series(
            work,
            "demand_left_after_routes",
            "demand_left_after",
            "demand_left",
        )
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


def _plot_service_provision_delta_preview(
    before_blocks: gpd.GeoDataFrame,
    after_blocks: gpd.GeoDataFrame,
    service: str,
    out_path: Path,
    *,
    boundary: gpd.GeoDataFrame | None = None,
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    before = _normalize_block_id_index(before_blocks)
    after = _normalize_block_id_index(after_blocks)
    common_idx = [idx for idx in before.index if idx in after.index]
    if not common_idx:
        return None

    def _pick(gdf: gpd.GeoDataFrame, *cols: str) -> pd.Series:
        for col in cols:
            if col in gdf.columns:
                return pd.to_numeric(gdf[col], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=gdf.index, dtype="float64")

    before_vals = _pick(before, "provision_after", "provision", "provision_strong").reindex(common_idx).fillna(0.0)
    after_vals = _pick(after, "provision_after_routes", "provision", "provision_strong").reindex(common_idx).fillna(0.0)
    plot_gdf = after.loc[common_idx, ["geometry"]].copy()
    plot_gdf["provision_delta"] = after_vals - before_vals
    plot_gdf = plot_gdf[plot_gdf.geometry.notna() & ~plot_gdf.geometry.is_empty].copy()
    if plot_gdf.empty:
        return None

    boundary_plot = normalize_preview_gdf(boundary, target_crs="EPSG:3857")
    plot_gdf = normalize_preview_gdf(plot_gdf, boundary_plot, target_crs="EPSG:3857")
    vmax = float(np.nanmax(np.abs(pd.to_numeric(plot_gdf["provision_delta"], errors="coerce").to_numpy()))) if len(plot_gdf) else 0.0
    vmax = max(vmax, 0.05)

    fig, ax = plt.subplots(figsize=(12, 10))
    apply_preview_canvas(fig, ax, boundary_plot, title=f"{service}: provision delta after routes")
    plot_gdf.plot(
        ax=ax,
        column="provision_delta",
        cmap="RdYlGn",
        linewidth=0.05,
        edgecolor="#d1d5db",
        legend=True,
        vmin=-vmax,
        vmax=vmax,
        legend_kwds={"label": "provision delta, positive = better"},
        zorder=3,
    )
    ax.set_axis_off()
    footer_text(fig, [f"min={float(plot_gdf['provision_delta'].min()):.3f}, max={float(plot_gdf['provision_delta'].max()):.3f}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview_figure(fig, out_path)
    plt.close(fig)
    return str(out_path)


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
    od_matrix_path: Path | None = None,
) -> dict:
    runtime_candidates = [
        repo_root / "connectpt" / ".venv" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python",
    ]
    python_path = next((path for path in runtime_candidates if path.exists()), None)
    if python_path is None:
        raise FileNotFoundError(
            "Missing route-generator runtime. Checked: "
            + ", ".join(str(path) for path in runtime_candidates)
        )

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
    if od_matrix_path is not None:
        cmd.extend(["--od-matrix-path", str(od_matrix_path)])
    if args.replace_in_intermodal:
        cmd.append("--replace-in-intermodal")
    if args.recompute_accessibility:
        cmd.append("--recompute-accessibility")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), str(repo_root / "connectpt")])
    env["MPLCONFIGDIR"] = repo_mplconfigdir("mpl-pipeline2-access-first", root=repo_root)
    _log(
        f"Running ConnectPT route generator: modality={args.modality}, "
        f"n_routes={args.n_routes}, runtime={python_path}"
    )
    subprocess.run(cmd, check=True, cwd=str(repo_root), env=env)

    summary_path = city_dir / "connectpt_routes_generator" / str(args.modality) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing route-generator summary after run: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["pipeline_route_length_policy"] = route_len_policy
    return summary


def _build_stop_points_from_connectpt_graph(city_dir: Path, modality: str, crs) -> gpd.GeoDataFrame:
    graph_path = city_dir / "connectpt_osm" / modality / "graph.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing ConnectPT graph for modality={modality}: {graph_path}")
    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)

    records = []
    for seq_idx, node_id in enumerate(sorted(graph.nodes())):
        records.append(
            {
                "stop_idx": int(seq_idx),
                "graph_node_id": int(node_id),
                "geometry": Point(float(graph.nodes[node_id]["x"]), float(graph.nodes[node_id]["y"])),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def _map_blocks_to_stop_indices(blocks_gdf: gpd.GeoDataFrame, stop_points: gpd.GeoDataFrame) -> dict[str, int]:
    work = _normalize_block_id_index(blocks_gdf)
    points = work[["geometry"]].copy()
    points["geometry"] = points.geometry.representative_point()
    points = points[points.geometry.notna() & ~points.geometry.is_empty].copy()
    if points.empty:
        return {}
    if points.crs is not None and stop_points.crs is not None and points.crs != stop_points.crs:
        points = points.to_crs(stop_points.crs)
    joined = points.sjoin_nearest(stop_points[["stop_idx", "geometry"]], how="left", distance_col="distance_to_stop")
    mapping: dict[str, int] = {}
    for block_id, row in joined.iterrows():
        if pd.isna(row.get("stop_idx")):
            continue
        mapping[str(block_id)] = int(row["stop_idx"])
    return mapping


def _build_service_target_od_from_placement(
    *,
    city_dir: Path,
    services: list[str],
    modality: str,
    placement_root_name: str,
    output_root: Path,
    service_blocks: dict[str, gpd.GeoDataFrame],
) -> dict | None:
    if not services:
        return None
    first_blocks = next(iter(service_blocks.values()))
    stop_points = _build_stop_points_from_connectpt_graph(city_dir, modality, first_blocks.crs)
    if stop_points.empty:
        raise ValueError(f"No ConnectPT stops found for modality={modality}")

    od = pd.DataFrame(
        0.0,
        index=range(len(stop_points)),
        columns=range(len(stop_points)),
        dtype=float,
    )
    summary_rows: list[dict] = []
    total_weight = 0.0

    for service in services:
        blocks_after = _normalize_block_id_index(service_blocks[service])
        assignment_links_path = _service_placement_assignment_links_path(city_dir, service, placement_root_name)
        if not assignment_links_path.exists():
            _log(f"Placement assignment links are missing for service [{service}]; skipping target OD contribution.")
            continue
        assignment_links = pd.read_csv(assignment_links_path)
        if assignment_links.empty:
            _log(f"Placement assignment links are empty for service [{service}]; skipping target OD contribution.")
            continue

        unresolved = _extract_gap_series(blocks_after, "demand_without_after", "demand_without")
        unresolved.index = unresolved.index.astype(str)
        stop_by_block = _map_blocks_to_stop_indices(blocks_after, stop_points)
        client_counts = assignment_links["client_id"].astype(str).value_counts().to_dict()

        service_weight = 0.0
        used_links = 0
        skipped_missing_stops = 0
        skipped_zero_gap = 0
        for row in assignment_links.itertuples(index=False):
            client_id = str(getattr(row, "client_id"))
            facility_id = str(getattr(row, "facility_id"))
            remaining_gap = float(unresolved.get(client_id, 0.0))
            if remaining_gap <= 0.0:
                skipped_zero_gap += 1
                continue
            origin_stop = stop_by_block.get(client_id)
            destination_stop = stop_by_block.get(facility_id)
            if origin_stop is None or destination_stop is None:
                skipped_missing_stops += 1
                continue
            divisor = max(1, int(client_counts.get(client_id, 1)))
            weight = remaining_gap / float(divisor)
            od.loc[origin_stop, destination_stop] += weight
            service_weight += weight
            used_links += 1

        total_weight += service_weight
        summary_rows.append(
            {
                "service": service,
                "assignment_links_path": str(assignment_links_path),
                "used_links": int(used_links),
                "skipped_zero_gap": int(skipped_zero_gap),
                "skipped_missing_stops": int(skipped_missing_stops),
                "target_weight_total": float(service_weight),
            }
        )

    od_dir = output_root / "service_target_od"
    od_dir.mkdir(parents=True, exist_ok=True)
    od_path = od_dir / f"{modality}_service_target_od.csv"
    od.to_csv(od_path)
    summary = {
        "placement_root_name": placement_root_name,
        "modality": modality,
        "stop_count": int(len(stop_points)),
        "positive_pairs": int((od.to_numpy() > 0.0).sum()),
        "target_weight_total": float(total_weight),
        "files": {
            "od_matrix_csv": str(od_path),
        },
        "services": summary_rows,
    }
    summary_path = od_dir / f"{modality}_service_target_od_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["files"]["summary_json"] = str(summary_path)
    service_logs = ", ".join(
        (
            f"{row['service']}: weight={row['target_weight_total']:.1f}, "
            f"used_links={row['used_links']}, "
            f"zero_gap={row['skipped_zero_gap']}, "
            f"missing_stops={row['skipped_missing_stops']}"
        )
        for row in summary_rows
    )
    _log(
        f"Service-target OD [{modality}] built: "
        f"stops={summary['stop_count']}, "
        f"positive_pairs={summary['positive_pairs']}, "
        f"target_weight={summary['target_weight_total']:.1f}"
        + (f", per_service=({service_logs})" if service_logs else "")
    )
    return summary


def _build_service_target_od_from_links(
    *,
    city_dir: Path,
    services: list[str],
    modality: str,
    output_root: Path,
    service_blocks: dict[str, gpd.GeoDataFrame],
    links_path_by_service: dict[str, Path],
    unresolved_gap_cols: tuple[str, ...],
    source_label: str,
) -> dict | None:
    if not services:
        return None
    first_blocks = next(iter(service_blocks.values()))
    stop_points = _build_stop_points_from_connectpt_graph(city_dir, modality, first_blocks.crs)
    if stop_points.empty:
        raise ValueError(f"No ConnectPT stops found for modality={modality}")

    od = pd.DataFrame(
        0.0,
        index=range(len(stop_points)),
        columns=range(len(stop_points)),
        dtype=float,
    )
    summary_rows: list[dict] = []
    total_weight = 0.0

    for service in services:
        blocks = _normalize_block_id_index(service_blocks[service])
        links_path = links_path_by_service.get(service)
        if links_path is None or not links_path.exists():
            _log(f"Target links are missing for service [{service}]; skipping target OD contribution.")
            continue
        links_df = pd.read_csv(links_path)
        if links_df.empty:
            _log(f"Target links are empty for service [{service}]; skipping target OD contribution.")
            continue
        if not {"source", "target"}.issubset(links_df.columns):
            _log(
                f"Target links for service [{service}] do not contain source/target columns; "
                f"found={list(links_df.columns)}. Skipping target OD contribution."
            )
            continue

        unresolved = _extract_gap_series(blocks, *unresolved_gap_cols)
        unresolved.index = unresolved.index.astype(str)
        stop_by_block = _map_blocks_to_stop_indices(blocks, stop_points)
        client_counts = links_df["source"].astype(str).value_counts().to_dict()

        service_weight = 0.0
        used_links = 0
        skipped_missing_stops = 0
        skipped_zero_gap = 0
        for row in links_df.itertuples(index=False):
            client_id = str(getattr(row, "source"))
            facility_id = str(getattr(row, "target"))
            remaining_gap = float(unresolved.get(client_id, 0.0))
            if remaining_gap <= 0.0:
                skipped_zero_gap += 1
                continue
            origin_stop = stop_by_block.get(client_id)
            destination_stop = stop_by_block.get(facility_id)
            if origin_stop is None or destination_stop is None:
                skipped_missing_stops += 1
                continue
            divisor = max(1, int(client_counts.get(client_id, 1)))
            weight = remaining_gap / float(divisor)
            od.loc[origin_stop, destination_stop] += weight
            service_weight += weight
            used_links += 1

        total_weight += service_weight
        summary_rows.append(
            {
                "service": service,
                "links_path": str(links_path),
                "used_links": int(used_links),
                "skipped_zero_gap": int(skipped_zero_gap),
                "skipped_missing_stops": int(skipped_missing_stops),
                "target_weight_total": float(service_weight),
            }
        )

    od_dir = output_root / "service_target_od"
    od_dir.mkdir(parents=True, exist_ok=True)
    od_path = od_dir / f"{modality}_service_target_od.csv"
    od.to_csv(od_path)
    summary = {
        "source_label": source_label,
        "modality": modality,
        "stop_count": int(len(stop_points)),
        "positive_pairs": int((od.to_numpy() > 0.0).sum()),
        "target_weight_total": float(total_weight),
        "files": {
            "od_matrix_csv": str(od_path),
        },
        "services": summary_rows,
    }
    summary_path = od_dir / f"{modality}_service_target_od_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["files"]["summary_json"] = str(summary_path)
    service_logs = ", ".join(
        (
            f"{row['service']}: weight={row['target_weight_total']:.1f}, "
            f"used_links={row['used_links']}, "
            f"zero_gap={row['skipped_zero_gap']}, "
            f"missing_stops={row['skipped_missing_stops']}"
        )
        for row in summary_rows
    )
    _log(
        f"Service-target OD [{modality}] built from {source_label}: "
        f"stops={summary['stop_count']}, "
        f"positive_pairs={summary['positive_pairs']}, "
        f"target_weight={summary['target_weight_total']:.1f}"
        + (f", per_service=({service_logs})" if service_logs else "")
    )
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
    service_blocks: dict[str, gpd.GeoDataFrame],
    services: list[str],
    route_summary: dict,
    output_root: Path,
) -> dict[str, dict]:
    rec = route_summary.get("recomputed_accessibility") or {}
    matrix_path = rec.get("matrix_path")
    if not matrix_path:
        raise ValueError("Route summary does not provide recomputed accessibility matrix path.")
    matrix = pd.read_parquet(Path(matrix_path))

    city_dir = Path(route_summary.get("city_dir", ""))
    boundary_path = city_dir / "analysis_territory" / "buffer.parquet"
    boundary = gpd.read_parquet(boundary_path) if boundary_path.exists() else None
    blocks_ref_path = city_dir / "derived_layers" / "blocks_clipped.parquet"
    blocks_ref = gpd.read_parquet(blocks_ref_path) if blocks_ref_path.exists() else None
    preview_dir = city_dir / "preview_png" / "all_together"
    out: dict[str, dict] = {}

    for service in services:
        baseline = _normalize_block_id_index(service_blocks[service])
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
        reprovision_input = baseline_work[["name", "population", "demand", "geometry"]].copy()
        capacity_source = "optimized_capacity_total" if "optimized_capacity_total" in baseline_work.columns else "capacity"
        reprovision_input["capacity"] = pd.to_numeric(baseline_work.get(capacity_source, 0.0), errors="coerce").fillna(0.0)
        recomputed_raw, links = _run_arctic_lp_provision(
            blocks_df=gpd.GeoDataFrame(reprovision_input, geometry="geometry", crs=baseline_work.crs),
            accessibility_matrix=sub_mx,
            service=service,
            service_radius_min=radius,
            service_demand_per_1000=demand_per_1000,
        )
        recomputed = baseline_work.copy()
        for column in (
            "demand_within",
            "demand_without",
            "capacity_left",
            "provision",
            "demand_left",
            "capacity_within",
            "capacity_without",
            "provision_strong",
            "provision_weak",
        ):
            if column in recomputed_raw.columns:
                values = pd.to_numeric(recomputed_raw[column], errors="coerce").reindex(recomputed.index)
                recomputed[column] = values
                recomputed[f"{column}_after_routes"] = values

        service_out_dir = output_root / "provision_after_routes" / service
        service_out_dir.mkdir(parents=True, exist_ok=True)
        blocks_after_path = service_out_dir / "blocks_solver_after_routes.parquet"
        links_after_path = service_out_dir / "provision_links_after_routes.csv"
        summary_after_path = service_out_dir / "summary_after_routes.json"
        recomputed.to_parquet(blocks_after_path)
        links.to_csv(links_after_path, index=False)

        before_access = float(_extract_gap_series(baseline_work, "demand_without_after", "demand_without").sum())
        before_capacity = float(_extract_gap_series(baseline_work, "demand_left_after", "demand_left").sum())
        after_access = float(_extract_gap_series(recomputed, "demand_without_after_routes", "demand_without").sum())
        after_capacity = float(_extract_gap_series(recomputed, "demand_left_after_routes", "demand_left").sum())
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
        _log(
            f"Provision after routes [{service}]: "
            f"demand_total={demand_total:.1f}, "
            f"access_before={before_access:.1f}, "
            f"access_after={after_access:.1f}, "
            f"access_delta={after_access - before_access:.1f}, "
            f"capacity_before={before_capacity:.1f}, "
            f"capacity_after={after_capacity:.1f}, "
            f"capacity_delta={after_capacity - before_capacity:.1f}"
        )

        preview_after_path = preview_dir / f"lp_{service}_provision_after_routes.png"
        _plot_service_lp_preview(recomputed, f"{service} after-routes", preview_after_path, blocks_ref=blocks_ref, boundary=boundary)
        summary["files"]["preview_after_routes"] = str(preview_after_path)
        preview_before_path = preview_dir / f"lp_{service}_provision_before_routes.png"
        _plot_service_lp_preview(baseline_work, f"{service} before-routes", preview_before_path, blocks_ref=blocks_ref, boundary=boundary)
        summary["files"]["preview_before_routes"] = str(preview_before_path)
        preview_delta_path = preview_dir / f"lp_{service}_provision_delta_after_routes.png"
        delta_path = _plot_service_provision_delta_preview(
            baseline_work,
            recomputed,
            service,
            preview_delta_path,
            boundary=boundary,
        )
        if delta_path is not None:
            summary["files"]["preview_delta_after_routes"] = str(preview_delta_path)
        out[service] = summary
    return out


def _export_joint_experiment_gallery(
    *,
    city_dir: Path,
    services: list[str],
    modality: str,
    placement_root_name: str | None,
    n_routes: int,
    manifest_path: Path,
    route_summary: dict | None,
    service_target_summary: dict | None,
    after_route_summaries: dict[str, dict] | None,
) -> Path:
    export_dir = _joint_experiment_export_dir(
        city_dir=city_dir,
        services=services,
        modality=modality,
        placement_root_name=placement_root_name,
        n_routes=n_routes,
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = export_dir / "preview_png"
    previews_dir.mkdir(parents=True, exist_ok=True)

    all_together = city_dir / "preview_png" / "all_together"
    preview_names = [
        f"pt_route_generator_{modality}.png",
        f"pt_route_generator_{modality}_with_existing.png",
        f"pt_route_generator_{modality}_generated_only.png",
        f"accessibility_mean_time_map_{modality}_before.png",
        f"accessibility_mean_time_map_{modality}_after.png",
        f"accessibility_mean_time_map_{modality}_generated.png",
        f"accessibility_mean_time_delta_map_{modality}_generated.png",
    ]
    for service in services:
        preview_names.extend(
            [
                f"lp_{service}_provision_before_placement.png",
                f"lp_{service}_provision_after_placement.png",
                f"lp_{service}_provision_delta_after_placement.png",
                f"lp_{service}_placement_changes.png",
                f"lp_{service}_provision_before_routes.png",
                f"lp_{service}_provision_after_routes.png",
                f"lp_{service}_provision_delta_after_routes.png",
            ]
        )
    copied_previews = 0
    for name in preview_names:
        src = all_together / name
        if src.exists():
            shutil.copy2(src, previews_dir / name)
            copied_previews += 1

    shutil.copy2(manifest_path, export_dir / "manifest_accessibility_first.json")
    if service_target_summary is not None:
        summary_json = service_target_summary.get("files", {}).get("summary_json")
        od_csv = service_target_summary.get("files", {}).get("od_matrix_csv")
        if summary_json and Path(summary_json).exists():
            shutil.copy2(Path(summary_json), export_dir / "service_target_od_summary.json")
        if od_csv and Path(od_csv).exists():
            shutil.copy2(Path(od_csv), export_dir / f"{modality}_service_target_od.csv")
    if route_summary is not None and not route_summary.get("skipped"):
        route_summary_path = city_dir / "connectpt_routes_generator" / modality / "summary.json"
        if route_summary_path.exists():
            shutil.copy2(route_summary_path, export_dir / f"connectpt_{modality}_summary.json")
    if after_route_summaries:
        for service, summary in after_route_summaries.items():
            summary_json = summary.get("files", {}).get("summary_after_routes")
            if summary_json and Path(summary_json).exists():
                shutil.copy2(Path(summary_json), export_dir / f"{service}_summary_after_routes.json")

    export_manifest = {
        "city": city_dir.name,
        "scenario": export_dir.name,
        "services": services,
        "modality": modality,
        "placement_root_name": placement_root_name,
        "n_routes": int(n_routes),
        "previews_dir": str(previews_dir),
        "copied_previews_count": int(copied_previews),
        "source_manifest": str(manifest_path),
    }
    export_manifest_path = export_dir / "export_manifest.json"
    export_manifest_path.write_text(json.dumps(export_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(
        f"Exported joint experiment gallery: dir={export_dir}, "
        f"previews={copied_previews}, scenario={export_dir.name}"
    )
    return export_dir


def main() -> None:
    _configure_logging()
    args = parse_args()
    services = _ensure_services(args.services)
    city_dir = _resolve_city_dir(args)
    repo_root = Path(__file__).resolve().parents[2]
    output_root = city_dir / "pipeline_2" / "accessibility_first"
    output_root.mkdir(parents=True, exist_ok=True)

    _log(f"Starting accessibility-first step: city={city_dir.name}, services={services}")
    baseline_blocks, service_block_paths, resolved_placement_root = _load_solver_blocks(
        city_dir,
        services,
        use_placement_outputs=bool(args.use_placement_outputs),
        placement_root_name=args.placement_root_name,
    )
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
    service_target_summary = None
    service_target_od_path = None
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
        if args.use_placement_outputs:
            if resolved_placement_root is None:
                raise RuntimeError("Placement outputs were requested but no placement root was resolved.")
            service_target_summary = _build_service_target_od_from_placement(
                city_dir=city_dir,
                services=services,
                modality=str(args.modality),
                placement_root_name=resolved_placement_root,
                output_root=output_root,
                service_blocks=baseline_blocks,
            )
            if service_target_summary is not None:
                service_target_od_path = Path(service_target_summary["files"]["od_matrix_csv"])
                if float(service_target_summary.get("target_weight_total", 0.0)) <= 0.0:
                    _log("Service-aware target OD has zero total weight after placement. Route generation is skipped.")
                    route_summary = {
                        "skipped": True,
                        "reason": "zero_service_target_od",
                        "service_target_od": service_target_summary,
                    }
        else:
            links_path_by_service = {
                service: _service_solver_provision_links_path(city_dir, service)
                for service in services
            }
            service_target_summary = _build_service_target_od_from_links(
                city_dir=city_dir,
                services=services,
                modality=str(args.modality),
                output_root=output_root,
                service_blocks=baseline_blocks,
                links_path_by_service=links_path_by_service,
                unresolved_gap_cols=("demand_without",),
                source_label="baseline_provision_links",
            )
            if service_target_summary is not None:
                service_target_od_path = Path(service_target_summary["files"]["od_matrix_csv"])
                if float(service_target_summary.get("target_weight_total", 0.0)) <= 0.0:
                    _log("Baseline service-aware target OD has zero total weight. Route generation is skipped.")
                    route_summary = {
                        "skipped": True,
                        "reason": "zero_service_target_od",
                        "service_target_od": service_target_summary,
                    }

        if route_summary is None:
            route_summary = _run_route_generator(
                repo_root=repo_root,
                city_dir=city_dir,
                args=args,
                od_matrix_path=service_target_od_path,
            )
            _log(
                "ConnectPT route step finished: "
                f"routes={route_summary.get('route_count')}, "
                f"cost={route_summary.get('cost')}, "
                f"att={route_summary.get('att')}, "
                f"unserved_pct={route_summary.get('unserved_demand_pct')}, "
                f"demand_sum={route_summary.get('demand_sum')}, "
                f"demand_max={route_summary.get('demand_max')}"
            )
    elif args.recompute_provision and services_for_recompute:
        existing_summary_path = (
            Path(args.route_summary_path).resolve()
            if args.route_summary_path
            else _default_route_summary_path(city_dir, str(args.modality))
        )
        if not existing_summary_path.exists():
            raise FileNotFoundError(
                "Provision recompute requested without route generation, but no existing route summary was found: "
                f"{existing_summary_path}"
            )
        route_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        _log(f"Using existing ConnectPT route summary for provision recompute: {existing_summary_path}")

    if (
        args.recompute_provision
        and services_for_recompute
        and not route_summary.get("skipped")
            and route_summary.get("recomputed_accessibility", {}).get("matrix_path")
        ):
            after_route_summaries = _recompute_provision_after_routes(
                service_blocks=baseline_blocks,
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
        "service_block_source": ("placement_after" if args.use_placement_outputs else "baseline_solver_inputs"),
        "service_block_paths": {service: str(path) for service, path in service_block_paths.items()},
        "placement_root_name": resolved_placement_root,
        "recompute_provision_only_access_problem_services": bool(args.recompute_provision_only_access_problem_services),
        "services_for_recompute": services_for_recompute,
        "services_skipped_no_access_problem": skipped_services_no_access_problem,
        "baseline": baseline_summary,
        "service_target_od": service_target_summary,
        "route_generation": route_summary,
        "provision_after_routes": after_route_summaries,
        "after_routes": after_summary,
    }
    manifest_path = output_root / "manifest_accessibility_first.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _export_joint_experiment_gallery(
        city_dir=city_dir,
        services=services,
        modality=str(args.modality),
        placement_root_name=resolved_placement_root,
        n_routes=int(args.n_routes),
        manifest_path=manifest_path,
        route_summary=route_summary,
        service_target_summary=service_target_summary,
        after_route_summaries=after_route_summaries,
    )
    _log(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

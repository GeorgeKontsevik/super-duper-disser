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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from shapely.geometry import Point
from torch_geometric.loader import DataLoader

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet
from aggregated_spatial_pipeline.runtime_config import configure_logger, repo_mplconfigdir
from aggregated_spatial_pipeline.visualization import apply_preview_canvas, normalize_preview_gdf, save_preview_figure

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connectpt.preprocess.od import get_OD  # noqa: E402
from connectpt.routes_generator.citygraph_dataset import get_dataset_from_config  # noqa: E402
from connectpt.routes_generator.eval_route_generator import eval_model  # noqa: E402
from connectpt.routes_generator.torch_utils import dump_routes  # noqa: E402
from connectpt.routes_generator.utils import get_eval_cfg  # noqa: E402
import connectpt.routes_generator.utils as lrnu  # noqa: E402


LAND_USE_SHARE_COLUMNS = [
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
]
POPULATION_COLUMNS = [
    "population_total",
    "population",
    "population_proxy",
    "pop_total",
    "residents",
    "res_population",
]


def _configure_logging() -> None:
    configure_logger("[connectpt-routes]")


def _log(message: str) -> None:
    logger.bind(tag="[connectpt-routes]").info(message)


def slugify_place(place: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")
    return slug or "place"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ConnectPT pretrained route generator on a city bundle and save standalone previews."
    )
    parser.add_argument("--place")
    parser.add_argument("--joint-input-dir")
    parser.add_argument("--modality", default="bus")
    parser.add_argument("--blocks-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--boundary-path")
    parser.add_argument("--weights-path")
    parser.add_argument("--n-routes", type=int, default=6)
    parser.add_argument("--min-route-len", type=int, default=2)
    parser.add_argument("--max-route-len", type=int, default=8)
    parser.add_argument("--demand-time-weight", type=float, default=0.33)
    parser.add_argument("--route-time-weight", type=float, default=0.33)
    parser.add_argument("--median-connectivity-weight", type=float, default=0.33)
    parser.add_argument("--replace-in-intermodal", action="store_true")
    parser.add_argument("--recompute-accessibility", action="store_true")
    return parser.parse_args()


def _resolve_city_dir(args: argparse.Namespace) -> Path:
    if args.joint_input_dir:
        return Path(args.joint_input_dir).resolve()
    if args.place:
        return (ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs" / slugify_place(args.place)).resolve()
    raise ValueError("Either --place or --joint-input-dir must be provided.")


def _resolve_blocks_path(city_dir: Path, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    candidates = [
        city_dir / "derived_layers" / "blocks_clipped.parquet",
        city_dir / "derived_layers" / "blocks_sm_imputed.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve blocks source in {city_dir / 'derived_layers'}")


def _resolve_boundary_path(city_dir: Path, override: str | None) -> Path | None:
    if override:
        return Path(override).resolve()
    candidates = [
        city_dir / "analysis_territory" / "buffer.parquet",
        city_dir / "connectpt_osm" / "boundary.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_output_dir(city_dir: Path, modality: str, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    return (city_dir / "connectpt_routes_generator" / modality).resolve()


def _resolve_weights_path(override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    return (ROOT / "connectpt" / "examples" / "data" / "model_weights" / "inductive_random_graphs_weighted_connectivity.pt").resolve()


def _resolve_population(blocks: gpd.GeoDataFrame) -> pd.Series:
    for column in POPULATION_COLUMNS:
        if column in blocks.columns:
            return pd.to_numeric(blocks[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=blocks.index, dtype="float64")


def _compute_site_area_m2(blocks: gpd.GeoDataFrame) -> pd.Series:
    work = blocks[["geometry"]].copy()
    if work.crs is None:
        work = work.set_crs(4326)
    local_crs = work.estimate_utm_crs() or "EPSG:3857"
    local = work.to_crs(local_crs)
    return pd.to_numeric(local.geometry.area, errors="coerce").replace(0, np.nan)


def _normalize_land_use(value) -> str | None:
    if pd.isna(value):
        return None
    value_str = str(value)
    if value_str.startswith("LandUse."):
        return value_str
    return f"LandUse.{value_str.upper()}"


def _compute_diversity(blocks: gpd.GeoDataFrame) -> pd.Series:
    def _entropy(row: pd.Series) -> float:
        vals = np.array([float(row.get(c, 0) or 0) for c in LAND_USE_SHARE_COLUMNS], dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) <= 1:
            return 0.0
        vals = vals / vals.sum()
        return float((-(vals * np.log(vals)).sum()) / np.log(len(LAND_USE_SHARE_COLUMNS)))

    return blocks.apply(_entropy, axis=1)


def _prepare_blocks_for_demand(blocks_path: Path) -> gpd.GeoDataFrame:
    blocks = gpd.read_parquet(blocks_path).copy()
    blocks["population"] = _resolve_population(blocks)
    if "density_proxy" in blocks.columns:
        density = pd.to_numeric(blocks["density_proxy"], errors="coerce")
    else:
        site_area = _compute_site_area_m2(blocks)
        density = blocks["population"] / site_area
    blocks["density"] = pd.to_numeric(density, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    blocks["diversity"] = _compute_diversity(blocks)
    for col in LAND_USE_SHARE_COLUMNS:
        if col not in blocks.columns:
            blocks[col] = 0.0
    blocks["land_use"] = blocks.get("land_use", pd.Series([None] * len(blocks), index=blocks.index)).map(_normalize_land_use)
    return blocks


def _load_stop_graph(city_dir: Path, modality: str):
    graph_path = city_dir / "connectpt_osm" / modality / "graph.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph pickle for modality={modality}: {graph_path}")
    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    return graph


def _build_stops_gdf_from_graph(graph, crs) -> tuple[list[int], dict[int, int], gpd.GeoDataFrame]:
    nodes = sorted(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    records = []
    for node in nodes:
        x = float(graph.nodes[node]["x"])
        y = float(graph.nodes[node]["y"])
        records.append({"graph_node_id": int(node), "geometry": Point(x, y)})
    stops = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    return nodes, node_to_idx, stops


def _build_street_adj(graph, nodes: list[int], node_to_idx: dict[int, int]) -> torch.Tensor:
    n_nodes = len(nodes)
    adj = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    for u, v, data in graph.edges(data=True):
        time_min = data.get("time_min")
        if time_min is None:
            continue
        i = node_to_idx[u]
        j = node_to_idx[v]
        value = float(time_min)
        adj[i, j] = min(adj[i, j], value)
        adj[j, i] = min(adj[j, i], value)
    return torch.tensor(adj, dtype=torch.float32)


def _build_node_locs(graph, nodes: list[int]) -> torch.Tensor:
    return torch.tensor([[graph.nodes[node]["x"], graph.nodes[node]["y"]] for node in nodes], dtype=torch.float32)


def _compute_od_matrix(blocks: gpd.GeoDataFrame, stops: gpd.GeoDataFrame, graph) -> pd.DataFrame:
    demand_blocks = blocks[["population", "density", "diversity", "land_use", "geometry"]].copy()
    return get_OD(demand_blocks, stops.copy(), graph.to_directed(), blocks.crs)


def _extract_metric(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    if hasattr(value, "ndim") and getattr(value, "ndim", 0) > 0:
        return float(value[0].item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _route_sequences(routes_tensor: torch.Tensor) -> list[list[int]]:
    if routes_tensor.ndim == 3:
        routes_tensor = routes_tensor[0]
    sequences = []
    for route in routes_tensor:
        seq = [int(node) for node in route.tolist() if int(node) >= 0]
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _build_connectpt_to_intermodal_mapping(city_dir: Path, modality: str, connectpt_stops: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    intermodal_nodes = gpd.read_parquet(city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet")
    preferred = intermodal_nodes[intermodal_nodes["type"].astype(str).eq(modality)].copy()
    fallback = intermodal_nodes[intermodal_nodes["type"].astype(str).isin([modality, "platform"])].copy()
    candidates = preferred if not preferred.empty else fallback
    if candidates.empty:
        raise ValueError(f"No intermodal nodes available for modality={modality}")
    joined = connectpt_stops.sjoin_nearest(
        candidates[["index", "type", "geometry"]],
        how="left",
        distance_col="distance_to_intermodal_m",
    ).rename(columns={"index": "intermodal_node_id", "type": "intermodal_node_type"})
    if joined["intermodal_node_id"].isna().any():
        raise ValueError(f"Could not map all connectpt stops to intermodal nodes for modality={modality}")
    return joined


def _extract_connectpt_route_sequences(routes_tensor: torch.Tensor, connectpt_nodes: list[int]) -> list[list[int]]:
    sequences = []
    for route in _route_sequences(routes_tensor):
        seq = [int(connectpt_nodes[idx]) for idx in route if 0 <= int(idx) < len(connectpt_nodes)]
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences


def _replace_modality_edges_in_intermodal(
    *,
    city_dir: Path,
    modality: str,
    connectpt_graph,
    connectpt_nodes: list[int],
    connectpt_stops: gpd.GeoDataFrame,
    routes_tensor: torch.Tensor,
    output_dir: Path,
) -> dict:
    with (city_dir / "intermodal_graph_iduedu" / "graph.pkl").open("rb") as fh:
        intermodal_graph = pickle.load(fh)

    mapping = _build_connectpt_to_intermodal_mapping(city_dir, modality, connectpt_stops)
    node_map = {
        int(row["graph_node_id"]): int(row["intermodal_node_id"])
        for _, row in mapping.iterrows()
        if pd.notna(row["intermodal_node_id"])
    }

    edges_to_remove = [
        (u, v, key)
        for u, v, key, data in intermodal_graph.edges(keys=True, data=True)
        if str(data.get("type")) == modality
    ]
    for u, v, key in edges_to_remove:
        intermodal_graph.remove_edge(u, v, key)

    generated_sequences = _extract_connectpt_route_sequences(routes_tensor, connectpt_nodes)
    generated_edge_count = 0
    route_records: list[dict] = []
    for route_idx, route in enumerate(generated_sequences, start=1):
        route_name = f"generated_{modality}_{route_idx}"
        for cu, cv in zip(route[:-1], route[1:]):
            if cu not in node_map or cv not in node_map:
                continue
            if not connectpt_graph.has_edge(cu, cv):
                continue
            edge_data = connectpt_graph.get_edge_data(cu, cv)
            inter_u = node_map[cu]
            inter_v = node_map[cv]
            if inter_u == inter_v:
                continue
            attrs = {
                "route": route_name,
                "type": modality,
                "length_meter": float(edge_data.get("weight", edge_data.get("length_meter", 0.0)) or 0.0),
                "time_min": float(edge_data.get("time_min", 0.0) or 0.0),
                "name": f"Generated {modality} route {route_idx}",
                "geometry": edge_data.get("geometry"),
            }
            intermodal_graph.add_edge(inter_u, inter_v, **attrs)
            intermodal_graph.add_edge(inter_v, inter_u, **attrs)
            generated_edge_count += 2
            route_records.append(
                {
                    "route_name": route_name,
                    "connectpt_u": int(cu),
                    "connectpt_v": int(cv),
                    "intermodal_u": int(inter_u),
                    "intermodal_v": int(inter_v),
                    "time_min": attrs["time_min"],
                    "length_meter": attrs["length_meter"],
                }
            )

    replace_dir = output_dir / "intermodal_replaced"
    replace_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = replace_dir / f"{modality}_connectpt_to_intermodal_map.parquet"
    prepare_geodata_for_parquet(mapping).to_parquet(mapping_path)
    route_edges_path = replace_dir / f"{modality}_generated_route_edges.parquet"
    pd.DataFrame.from_records(route_records).to_parquet(route_edges_path)
    graph_out_path = replace_dir / "graph.pkl"
    _save_pickle(intermodal_graph, graph_out_path)

    return {
        "graph_path": graph_out_path,
        "mapping_path": mapping_path,
        "generated_edges_path": route_edges_path,
        "removed_modality_edges": len(edges_to_remove),
        "generated_modality_edges": generated_edge_count,
        "mapping_distance_max_m": float(pd.to_numeric(mapping["distance_to_intermodal_m"], errors="coerce").max()),
        "mapping_distance_median_m": float(pd.to_numeric(mapping["distance_to_intermodal_m"], errors="coerce").median()),
    }


def _recompute_accessibility_with_root_env(
    *,
    city_dir: Path,
    output_dir: Path,
    replaced_graph_path: Path,
    modality: str,
) -> dict:
    root_python = ROOT / ".venv" / "bin" / "python"
    if not root_python.exists():
        raise FileNotFoundError(f"Missing root runtime for accessibility recompute: {root_python}")

    matrix_output_path = output_dir / "accessibility_recomputed" / "adj_matrix_time_min_union.parquet"
    preview_output_dir = output_dir / "accessibility_recomputed" / "preview_png"
    units_path = city_dir / "pipeline_2" / "prepared" / "units_union.parquet"
    boundary_path = city_dir / "analysis_territory" / "buffer.parquet"

    command = [
        str(root_python),
        "-m",
        "aggregated_spatial_pipeline.connectpt_data_pipeline.run_accessibility_recompute_external",
        "--units-path",
        str(units_path),
        "--graph-pickle-path",
        str(replaced_graph_path),
        "--boundary-path",
        str(boundary_path),
        "--matrix-output-path",
        str(matrix_output_path),
        "--preview-output-dir",
        str(preview_output_dir),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["MPLCONFIGDIR"] = repo_mplconfigdir("mpl-connectpt-access", root=ROOT)
    subprocess.run(command, check=True, env=env, cwd=str(ROOT))

    local_preview_path = preview_output_dir / "30_accessibility_mean_time_map.png"
    shared_preview_path = city_dir / "preview_png" / "all_together" / f"accessibility_mean_time_map_{modality}_generated.png"
    if local_preview_path.exists():
        shared_preview_path.write_bytes(local_preview_path.read_bytes())
    return {
        "matrix_path": matrix_output_path,
        "local_preview_path": local_preview_path,
        "shared_preview_path": shared_preview_path,
    }


def _save_route_preview(
    *,
    graph,
    boundary: gpd.GeoDataFrame | None,
    routes_tensor: torch.Tensor,
    summary: dict,
    out_path: Path,
) -> None:
    boundary_plot = normalize_preview_gdf(boundary, target_crs="EPSG:3857")
    target_crs = getattr(boundary_plot, "crs", None) or "EPSG:3857"
    if boundary_plot is None or boundary_plot.empty:
        try:
            all_edge_geoms = [data.get("geometry") for _, _, data in graph.edges(data=True) if data.get("geometry") is not None]
            if all_edge_geoms:
                boundary_plot = normalize_preview_gdf(gpd.GeoDataFrame({"geometry": all_edge_geoms}, crs=target_crs), target_crs="EPSG:3857")
        except Exception:
            boundary_plot = None
    fig, ax = plt.subplots(figsize=(10, 10))
    apply_preview_canvas(fig, ax, boundary_plot, title=None, min_pad=120.0)

    edge_geoms = []
    for _, _, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is not None and not geom.is_empty:
            edge_geoms.append(geom)
    if edge_geoms:
        gpd.GeoSeries(edge_geoms, crs=target_crs).pipe(lambda s: s.to_crs("EPSG:3857") if getattr(s, "crs", None) else s).plot(
            ax=ax, color="#9aa0a6", linewidth=1.2, alpha=0.45, zorder=1
        )

    node_points = [Point(float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])) for n in sorted(graph.nodes())]
    gpd.GeoSeries(node_points, crs=target_crs).pipe(lambda s: s.to_crs("EPSG:3857") if getattr(s, "crs", None) else s).plot(
        ax=ax, color="#495057", markersize=18, alpha=0.9, zorder=2
    )

    palette = ["#e03131", "#1971c2", "#2f9e44", "#f08c00", "#7048e8", "#c2255c", "#0b7285", "#5c940d"]
    for idx, route in enumerate(_route_sequences(routes_tensor)):
        color = palette[idx % len(palette)]
        for u, v in zip(route[:-1], route[1:]):
            if graph.has_edge(u, v):
                geom = graph.get_edge_data(u, v).get("geometry")
                if geom is not None and not geom.is_empty:
                    gpd.GeoSeries([geom], crs=target_crs).pipe(lambda s: s.to_crs("EPSG:3857") if getattr(s, "crs", None) else s).plot(
                        ax=ax, color=color, linewidth=3.6, alpha=0.95, zorder=4
                    )
        xs = [float(graph.nodes[n]["x"]) for n in route]
        ys = [float(graph.nodes[n]["y"]) for n in route]
        ax.scatter(xs, ys, s=22, color=color, zorder=5)

    title = (
        f"ConnectPT Route Generator ({summary['modality']})\n"
        f"routes={summary['route_count']} | cost={summary['cost']:.3f} | "
        f"ATT={summary['att']:.3f} | unserved={summary['unserved_demand_pct']:.2f}%"
    )
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _configure_logging()

    city_dir = _resolve_city_dir(args)
    modality = str(args.modality).lower()
    output_dir = _resolve_output_dir(city_dir, modality, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview_png"
    shared_preview_dir = city_dir / "preview_png" / "all_together"
    shared_preview_dir.mkdir(parents=True, exist_ok=True)

    blocks_path = _resolve_blocks_path(city_dir, args.blocks_path)
    boundary_path = _resolve_boundary_path(city_dir, args.boundary_path)
    boundary = gpd.read_parquet(boundary_path) if boundary_path is not None and boundary_path.exists() else None
    weights_path = _resolve_weights_path(args.weights_path)
    graph = _load_stop_graph(city_dir, modality)

    _log(
        f"Preparing ConnectPT route-generator inputs: city={city_dir.name}, modality={modality}, blocks={blocks_path.name}"
    )
    blocks = _prepare_blocks_for_demand(blocks_path)
    nodes, node_to_idx, stops = _build_stops_gdf_from_graph(graph, blocks.crs)
    od_matrix = _compute_od_matrix(blocks, stops, graph)
    node_locs = _build_node_locs(graph, nodes)
    street_adj = _build_street_adj(graph, nodes, node_to_idx)
    demand = torch.tensor(od_matrix.reindex(index=range(len(nodes)), columns=range(len(nodes)), fill_value=0.0).to_numpy(), dtype=torch.float32)

    tensors = {"street_adj": street_adj, "demand": demand, "node_locs": node_locs}
    cfg_dir = str(ROOT / "connectpt" / "connectpt" / "routes_generator" / "cfg")
    params = {
        "dataset_name": "tensor",
        "n_routes": args.n_routes,
        "min_route_len": args.min_route_len,
        "max_route_len": args.max_route_len,
        "demand_time_weight": args.demand_time_weight,
        "route_time_weight": args.route_time_weight,
        "median_connectivity_weight": args.median_connectivity_weight,
        "run_name": f"{city_dir.name}_{modality}_bundle_lc",
        "model_weights": str(weights_path),
    }
    cfg = get_eval_cfg(cfg_dir=cfg_dir, base_cfg_name="eval_model_mumford", params=params)
    test_ds = get_dataset_from_config(cfg.eval.dataset, tensors=tensors)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)
    device, run_name, _, cost_obj, model = lrnu.process_standard_experiment_cfg(cfg, run_name_prefix="lc_", weights_required=True)
    _log(
        f"Running pretrained route generator on bundle data: nodes={len(nodes)}, blocks={len(blocks)}, n_routes={args.n_routes}"
    )
    _, _, metrics, routes = eval_model(
        model,
        test_dl,
        cfg.eval,
        cost_obj,
        n_samples=cfg.get("n_samples", 1),
        return_routes=True,
        silent=True,
        device=device,
    )

    dump_routes(run_name, routes.cpu(), out_dir=output_dir)
    od_path = output_dir / f"{modality}_od_matrix.csv"
    od_matrix.to_csv(od_path)

    summary = {
        "city_dir": str(city_dir),
        "modality": modality,
        "blocks_source": str(blocks_path),
        "boundary_source": str(boundary_path) if boundary_path is not None else None,
        "weights_path": str(weights_path),
        "graph_node_count": int(len(nodes)),
        "graph_edge_count": int(graph.number_of_edges()),
        "blocks_count": int(len(blocks)),
        "population_total": float(pd.to_numeric(blocks["population"], errors="coerce").fillna(0.0).sum()),
        "route_count": len(_route_sequences(routes)),
        "cost": _extract_metric(metrics, "cost"),
        "att": _extract_metric(metrics, "ATT"),
        "unserved_demand_pct": _extract_metric(metrics, "$d_{un}$"),
        "median_connectivity": _extract_metric(metrics, "median_connectivity"),
        "demand_sum": float(demand.sum().item()),
        "demand_max": float(demand.max().item()),
        "routes_shape": list(routes.shape),
        "routes_tensor": routes.cpu().tolist(),
        "files": {
            "od_matrix": str(od_path),
            "routes_pickle": str(output_dir / f"{run_name}_routes.pkl"),
        },
    }

    route_preview_path = preview_dir / f"pt_route_generator_{modality}.png"
    _save_route_preview(
        graph=graph,
        boundary=boundary.to_crs(blocks.crs) if boundary is not None and boundary.crs != blocks.crs else boundary,
        routes_tensor=routes.cpu(),
        summary=summary,
        out_path=route_preview_path,
    )
    shared_preview_path = shared_preview_dir / f"pt_route_generator_{modality}.png"
    shared_preview_path.write_bytes(route_preview_path.read_bytes())
    summary["files"]["route_preview"] = str(route_preview_path)
    summary["files"]["shared_route_preview"] = str(shared_preview_path)

    if args.replace_in_intermodal or args.recompute_accessibility:
        replace_summary = _replace_modality_edges_in_intermodal(
            city_dir=city_dir,
            modality=modality,
            connectpt_graph=graph,
            connectpt_nodes=nodes,
            connectpt_stops=stops,
            routes_tensor=routes.cpu(),
            output_dir=output_dir,
        )
        summary["intermodal_replacement"] = {
            "enabled": True,
            **{k: str(v) if isinstance(v, Path) else v for k, v in replace_summary.items()},
        }
        _log(
            f"Intermodal graph updated for modality={modality}: "
            f"removed={replace_summary['removed_modality_edges']}, "
            f"added={replace_summary['generated_modality_edges']}"
        )

        if args.recompute_accessibility:
            access_summary = _recompute_accessibility_with_root_env(
                city_dir=city_dir,
                output_dir=output_dir,
                replaced_graph_path=replace_summary["graph_path"],
                modality=modality,
            )
            summary["recomputed_accessibility"] = {
                "enabled": True,
                **{k: str(v) if isinstance(v, Path) else v for k, v in access_summary.items()},
            }
            _log(f"Recomputed accessibility matrix on generated {modality} routes")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _log(f"Saved route-generator summary: {summary_path}")
    _log(f"Saved route preview: {route_preview_path}")


if __name__ == "__main__":
    main()

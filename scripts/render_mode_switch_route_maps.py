#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_heat_service_city_pairs_and_routes import (
    CITY_RU,
    SERVICE_RU,
    _edge_geometry,
    _living_points,
    _load_graph,
    _load_nodes,
    _pick_pt_row,
    _plot_boundary,
    _walk_subgraph,
)


def _concat_existing(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _best_edge_data(graph: nx.MultiDiGraph | nx.Graph, u: int, v: int) -> dict:
    if isinstance(graph, nx.MultiDiGraph):
        bundle = graph.get_edge_data(u, v)
        if not bundle:
            raise KeyError(f"missing edge {u}->{v}")
        return min(bundle.values(), key=lambda d: float(d.get("time_min", float("inf"))))
    data = graph.get_edge_data(u, v)
    if not data:
        raise KeyError(f"missing edge {u}->{v}")
    return data


def _edge_keys_to_gdf(edge_keys: set[tuple[int, int]], graph: nx.MultiDiGraph | nx.Graph, nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geoms = []
    for u, v in sorted(edge_keys):
        try:
            geoms.append(_edge_geometry(graph, nodes_gdf, int(u), int(v)))
        except Exception:
            try:
                geoms.append(_edge_geometry(graph, nodes_gdf, int(v), int(u)))
            except Exception:
                continue
    return gpd.GeoDataFrame({"geometry": geoms}, geometry="geometry", crs=nodes_gdf.crs)


def _graph_edges_gdf(graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geoms = []
    for u, v in graph.edges():
        try:
            geoms.append(_edge_geometry(graph, nodes_gdf, int(u), int(v)))
        except Exception:
            continue
    return gpd.GeoDataFrame({"geometry": geoms}, geometry="geometry", crs=nodes_gdf.crs)


def _path_edge_sets(path_nodes: list[int], graph: nx.MultiDiGraph | nx.Graph) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    walk_edges: set[tuple[int, int]] = set()
    other_edges: set[tuple[int, int]] = set()
    for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False):
        data = _best_edge_data(graph, int(u), int(v))
        edge_key = (int(u), int(v))
        edge_key = edge_key if edge_key[0] <= edge_key[1] else (edge_key[1], edge_key[0])
        edge_type = "walk" if not isinstance(graph, nx.MultiDiGraph) else str(data.get("type", "")).lower()
        if edge_type == "walk":
            walk_edges.add(edge_key)
        else:
            other_edges.add(edge_key)
    return walk_edges, other_edges


def _render_map(
    boundary: gpd.GeoDataFrame,
    background_edges: gpd.GeoDataFrame,
    service_points: gpd.GeoDataFrame,
    changed_buildings: gpd.GeoDataFrame,
    out_path: Path,
    title: str,
    old_walk: gpd.GeoDataFrame | None = None,
    old_pt: gpd.GeoDataFrame | None = None,
    new_walk: gpd.GeoDataFrame | None = None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=320)
    _plot_boundary(ax, boundary)
    if not background_edges.empty:
        background_edges.plot(ax=ax, color="#dbe4f0", linewidth=0.35, alpha=0.75)
    if old_walk is not None and not old_walk.empty:
        old_walk.plot(ax=ax, color="#7f1734", linewidth=1.2, alpha=0.80)
    if old_pt is not None and not old_pt.empty:
        old_pt.plot(ax=ax, color="#2563eb", linewidth=1.4, alpha=0.88)
    if new_walk is not None and not new_walk.empty:
        new_walk.plot(ax=ax, color="#16a34a", linewidth=1.4, alpha=0.88)
    if not changed_buildings.empty:
        ax.scatter(
            changed_buildings.geometry.x,
            changed_buildings.geometry.y,
            s=18,
            c="#f97316",
            alpha=0.80,
            linewidths=0,
            zorder=4,
            rasterized=True,
        )
    service_points.plot(ax=ax, color="#111827", markersize=40, marker="*", alpha=0.92, zorder=5)
    handles = []
    if old_walk is not None:
        handles.append(Line2D([0], [0], color="#7f1734", lw=3, label="старый пеший access/egress"))
    if old_pt is not None:
        handles.append(Line2D([0], [0], color="#2563eb", lw=3, label="старый OT участок"))
    if new_walk is not None:
        handles.append(Line2D([0], [0], color="#16a34a", lw=3, label="новый пеший route"))
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#f97316", markeredgecolor="#f97316", markersize=10, label="изменившиеся дома"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#111827", markeredgecolor="#111827", markersize=12, label="сервисы"),
        ]
    )
    ax.set_title(title, fontsize=18)
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="debrecen_hungary")
    parser.add_argument("--service", default="polyclinic")
    parser.add_argument("--joint-inputs-root", type=Path, default=Path("aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"))
    parser.add_argument("--baseline-walk-root", type=Path, default=Path("aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_top1"))
    parser.add_argument("--baseline-pt-lt-root", type=Path, default=Path("aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_pt_top1_walk_lt15"))
    parser.add_argument("--baseline-pt-ge-root", type=Path, default=Path("aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_pt_top1_walk15plus"))
    parser.add_argument("--baseline-diag-root", type=Path, default=Path("thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/diag_baseline"))
    parser.add_argument("--heat-joint-inputs-root", type=Path, default=Path("thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/heat_joint_inputs"))
    parser.add_argument("--heat-walk-root", type=Path, default=Path("thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/walk_heat"))
    parser.add_argument("--heat-diag-root", type=Path, default=Path("thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/diag_heat"))
    parser.add_argument("--out-root", type=Path, default=Path("thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/debrecen_mode_switch_maps"))
    args = parser.parse_args()

    city = args.city
    service = args.service
    city_root = args.joint_inputs_root / city
    boundary = gpd.read_parquet(city_root / "blocksnet" / "boundary.parquet")
    buildings = _living_points(city_root)
    services = gpd.read_parquet(city_root / "pipeline_2" / "services_raw" / f"{service}.parquet")
    services["geometry"] = services.geometry.representative_point()
    metric_crs = boundary.estimate_utm_crs()
    if metric_crs is not None:
        boundary = boundary.to_crs(metric_crs)
        buildings = buildings.to_crs(metric_crs)
        services = services.to_crs(metric_crs)

    base_diag = pd.read_parquet(args.baseline_diag_root / city / "home_to_service_access_diagnostics.parquet")
    heat_diag = pd.read_parquet(args.heat_diag_root / city / "home_to_service_access_diagnostics.parquet")
    base_diag = base_diag.loc[base_diag["service_name"] == service].copy()
    heat_diag = heat_diag.loc[heat_diag["service_name"] == service].copy()
    merged = base_diag.merge(heat_diag, on=["building_idx", "service_name"], suffixes=("_baseline", "_heat"))
    merged["mode_baseline"] = merged.apply(
        lambda r: "walk" if float(r["walk_time_min_baseline"]) <= float(r["effective_pt_total_min_baseline"]) else "pt",
        axis=1,
    )
    merged["mode_heat"] = merged.apply(
        lambda r: "walk" if float(r["walk_time_min_heat"]) <= float(r["effective_pt_total_min_heat"]) else "pt",
        axis=1,
    )
    merged["delta_effective_min"] = merged.apply(
        lambda r: min(float(r["walk_time_min_heat"]), float(r["effective_pt_total_min_heat"]))
        - min(float(r["walk_time_min_baseline"]), float(r["effective_pt_total_min_baseline"])),
        axis=1,
    )
    switched = merged.loc[(merged["mode_baseline"] == "pt") & (merged["mode_heat"] == "walk")].copy()
    switched = switched.sort_values("delta_effective_min", ascending=False)

    base_walk = pd.read_parquet(args.baseline_walk_root / city / "residential_to_services_top1.parquet")
    heat_walk = pd.read_parquet(args.heat_walk_root / city / "residential_to_services_top1.parquet")
    base_walk = base_walk.loc[base_walk["service_name"] == service]
    heat_walk = heat_walk.loc[heat_walk["service_name"] == service]
    base_pt = _concat_existing(
        [
            args.baseline_pt_lt_root / city / "residential_to_services_pt_top1.parquet",
            args.baseline_pt_ge_root / city / "residential_to_services_pt_top1.parquet",
        ]
    )
    if not base_pt.empty:
        base_pt = base_pt.loc[base_pt["service_name"] == service]

    base_graph = _load_graph(str(args.joint_inputs_root / city / "intermodal_graph_iduedu" / "graph.pkl"))
    base_nodes = _load_nodes(str(args.joint_inputs_root / city / "intermodal_graph_iduedu" / "graph_nodes.parquet"))
    heat_graph = _load_graph(str(args.heat_joint_inputs_root / city / "intermodal_graph_iduedu" / "graph.pkl"))
    heat_nodes = _load_nodes(str(args.heat_joint_inputs_root / city / "intermodal_graph_iduedu" / "graph_nodes.parquet"))
    base_walk_graph = _walk_subgraph(base_graph)
    heat_walk_graph = _walk_subgraph(heat_graph)

    background_edges = _graph_edges_gdf(base_walk_graph, base_nodes).to_crs(boundary.crs)
    old_walk_edges: set[tuple[int, int]] = set()
    old_pt_edges: set[tuple[int, int]] = set()
    new_walk_edges: set[tuple[int, int]] = set()
    changed_building_ids: list[int] = []
    failures = 0

    for _, row in switched.iterrows():
        bidx = int(row["building_idx"])
        pt_row = _pick_pt_row(base_pt, bidx, service)
        walk_row = heat_walk.loc[heat_walk["building_idx"] == bidx]
        walk_row = walk_row.iloc[0] if not walk_row.empty else None
        if pt_row is None or walk_row is None:
            failures += 1
            continue
        try:
            base_path = nx.shortest_path(
                base_graph,
                int(pt_row["home_graph_node"]),
                int(pt_row["nearest_service_graph_node"]),
                weight="time_min",
            )
            heat_path = nx.shortest_path(
                heat_walk_graph,
                int(walk_row["home_graph_node"]),
                int(walk_row["service_graph_node"]),
                weight="time_min",
            )
        except Exception:
            failures += 1
            continue
        walk_edges, pt_edges = _path_edge_sets(base_path, base_graph)
        new_walk, _ = _path_edge_sets(heat_path, heat_walk_graph)
        old_walk_edges.update(walk_edges)
        old_pt_edges.update(pt_edges)
        new_walk_edges.update(new_walk)
        changed_building_ids.append(bidx)

    old_walk_gdf = _edge_keys_to_gdf(old_walk_edges, base_graph, base_nodes).to_crs(boundary.crs)
    old_pt_gdf = _edge_keys_to_gdf(old_pt_edges, base_graph, base_nodes).to_crs(boundary.crs)
    new_walk_gdf = _edge_keys_to_gdf(new_walk_edges, heat_walk_graph, heat_nodes).to_crs(boundary.crs)
    changed_buildings = buildings.loc[buildings["building_idx"].isin(changed_building_ids)].copy().to_crs(boundary.crs)

    city_label = CITY_RU.get(city, city)
    service_label = SERVICE_RU.get(service, service)
    mode_count = len(changed_building_ids)
    max_delta = float(switched["delta_effective_min"].max()) if not switched.empty else 0.0
    base_title = f"{city_label} — {service_label} — PT→walk changed buildings: {mode_count} (max Δ={max_delta:.2f} min)"

    _render_map(
        boundary=boundary,
        background_edges=background_edges,
        service_points=services.to_crs(boundary.crs),
        changed_buildings=changed_buildings,
        old_walk=old_walk_gdf,
        old_pt=old_pt_gdf,
        new_walk=new_walk_gdf,
        out_path=args.out_root / city / f"{service}_pt_to_walk_all_routes.png",
        title=base_title,
    )
    _render_map(
        boundary=boundary,
        background_edges=background_edges,
        service_points=services.to_crs(boundary.crs),
        changed_buildings=changed_buildings,
        old_walk=None,
        old_pt=None,
        new_walk=new_walk_gdf,
        out_path=args.out_root / city / f"{service}_pt_to_walk_new_walk_only.png",
        title=f"{city_label} — {service_label} — новые heat-пешеходные routes",
    )
    _render_map(
        boundary=boundary,
        background_edges=background_edges,
        service_points=services.to_crs(boundary.crs),
        changed_buildings=changed_buildings,
        old_walk=old_walk_gdf,
        old_pt=old_pt_gdf,
        new_walk=None,
        out_path=args.out_root / city / f"{service}_pt_to_walk_old_pt_only.png",
        title=f"{city_label} — {service_label} — старые PT routes",
    )

    print(
        {
            "city": city,
            "service": service,
            "mode_switch": "pt_to_walk",
            "changed_buildings": mode_count,
            "old_walk_edges": len(old_walk_edges),
            "old_pt_edges": len(old_pt_edges),
            "new_walk_edges": len(new_walk_edges),
            "failed_routes": failures,
        }
    )


if __name__ == "__main__":
    main()

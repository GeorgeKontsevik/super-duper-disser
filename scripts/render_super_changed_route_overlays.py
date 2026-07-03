#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.render_heat_service_city_pairs_and_routes import (
    BASELINE_PT_COLOR,
    CITY_RU,
    HEAT_PT_COLOR,
    HOME_COLOR,
    HOME_MARKER,
    HOME_MARKER_SIZE,
    PT_LINESTYLE,
    SERVICE_COLOR,
    SERVICE_MARKER,
    SERVICE_MARKER_SIZE,
    _edge_geometry,
    _living_points,
    _load_graph,
    _load_nodes,
    _pick_pt_row,
    _plot_boundary,
    _walk_subgraph,
)
from scripts.render_debrecen_heat_story_maps import _concat_existing


ROOT = Path(__file__).resolve().parents[1]
HEAT_ROOT = ROOT / "thermal_access_pilot/outputs/batch_service_access_hottest_summer2025"
BASE_EXPERIMENT_ROOT = ROOT / "aggregated_spatial_pipeline/outputs/experiments_active19_20260412"
JOINT_INPUTS = ROOT / "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"


def _path_gdf(path_nodes: list[int], graph: nx.MultiDiGraph | nx.Graph, nodes_gdf: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    geoms = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False):
        try:
            geoms.append(_edge_geometry(graph, nodes_gdf, int(u), int(v)))
        except Exception:
            continue
    if not geoms:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=nodes_gdf.crs).to_crs(crs)
    return gpd.GeoDataFrame({"geometry": geoms}, geometry="geometry", crs=nodes_gdf.crs).to_crs(crs)


def _split_path_gdfs(
    path_nodes: list[int],
    graph: nx.MultiDiGraph | nx.Graph,
    nodes_gdf: gpd.GeoDataFrame,
    crs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    walk_geoms = []
    pt_geoms = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False):
        try:
            geom = _edge_geometry(graph, nodes_gdf, int(u), int(v))
        except Exception:
            continue
        if isinstance(graph, nx.MultiDiGraph):
            edge_data = graph.get_edge_data(int(u), int(v))
            best = min(edge_data.values(), key=lambda d: float(d.get("time_min", np.inf)))
            edge_type = str(best.get("type", "")).lower()
            if edge_type == "walk":
                walk_geoms.append(geom)
            else:
                pt_geoms.append(geom)
        else:
            walk_geoms.append(geom)
    walk = gpd.GeoDataFrame({"geometry": walk_geoms}, geometry="geometry", crs=nodes_gdf.crs).to_crs(crs)
    pt = gpd.GeoDataFrame({"geometry": pt_geoms}, geometry="geometry", crs=nodes_gdf.crs).to_crs(crs)
    return walk, pt


def _background_edges(graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    geoms = []
    for u, v in graph.edges():
        try:
            geoms.append(_edge_geometry(graph, nodes_gdf, int(u), int(v)))
        except Exception:
            continue
    return gpd.GeoDataFrame({"geometry": geoms}, geometry="geometry", crs=nodes_gdf.crs).to_crs(crs)


def _route_edge_keys(path_nodes: list[int]) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False)}


def _route_signature(case: dict) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    sig: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for key in ["base_walk_gdf", "base_pt_gdf", "heat_walk_gdf", "heat_pt_gdf"]:
        gdf = case.get(key)
        if gdf is None or gdf.empty:
            continue
        for geom in gdf.geometry:
            coords = list(geom.coords)
            for a, b in zip(coords[:-1], coords[1:], strict=False):
                qa = (round(float(a[0]), 1), round(float(a[1]), 1))
                qb = (round(float(b[0]), 1), round(float(b[1]), 1))
                sig.add(tuple(sorted((qa, qb))))
    return sig


def _load_compare(city: str, service: str) -> pd.DataFrame:
    base = pd.read_parquet(HEAT_ROOT / "diag_baseline" / city / "home_to_service_access_diagnostics.parquet")
    heat = pd.read_parquet(HEAT_ROOT / "diag_heat" / city / "home_to_service_access_diagnostics.parquet")
    base = base.loc[base["service_name"] == service].copy()
    heat = heat.loc[heat["service_name"] == service].copy()
    merged = base.merge(heat, on=["building_idx", "service_name"], suffixes=("_baseline", "_heat"))
    merged["mode_baseline"] = np.where(
        merged["walk_time_min_baseline"] <= merged["effective_pt_total_min_baseline"], "walk", "pt"
    )
    merged["mode_heat"] = np.where(
        merged["walk_time_min_heat"] <= merged["effective_pt_total_min_heat"], "walk", "pt"
    )
    merged["effective_baseline_min"] = np.minimum(
        merged["walk_time_min_baseline"], merged["effective_pt_total_min_baseline"]
    )
    merged["effective_heat_min"] = np.minimum(
        merged["walk_time_min_heat"], merged["effective_pt_total_min_heat"]
    )
    merged["delta_min"] = merged["effective_heat_min"] - merged["effective_baseline_min"]
    return merged


def _build_cases(city: str, service: str, mode_filter: str | None = None) -> tuple[gpd.GeoDataFrame, list[dict], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    city_root = JOINT_INPUTS / city
    boundary = gpd.read_parquet(city_root / "blocksnet" / "boundary.parquet")
    buildings = _living_points(city_root)
    services = gpd.read_parquet(city_root / "pipeline_2" / "services_raw" / f"{service}.parquet")
    services["geometry"] = services.geometry.representative_point()
    metric_crs = boundary.estimate_utm_crs()
    if metric_crs is not None:
        boundary = boundary.to_crs(metric_crs)
        buildings = buildings.to_crs(metric_crs)
        services = services.to_crs(metric_crs)

    compare = _load_compare(city, service)
    base_graph = _load_graph(str(JOINT_INPUTS / city / "intermodal_graph_iduedu" / "graph.pkl"))
    heat_graph = _load_graph(str(HEAT_ROOT / "heat_joint_inputs" / city / "intermodal_graph_iduedu" / "graph.pkl"))
    base_nodes = _load_nodes(str(JOINT_INPUTS / city / "intermodal_graph_iduedu" / "graph_nodes.parquet"))
    heat_nodes = _load_nodes(str(HEAT_ROOT / "heat_joint_inputs" / city / "intermodal_graph_iduedu" / "graph_nodes.parquet"))
    base_walk_graph = _walk_subgraph(base_graph)
    heat_walk_graph = _walk_subgraph(heat_graph)
    services_pts = services.to_crs(boundary.crs)

    base_walk = pd.read_parquet(BASE_EXPERIMENT_ROOT / "residential_to_services_top1" / city / "residential_to_services_top1.parquet")
    heat_walk = pd.read_parquet(HEAT_ROOT / "walk_heat" / city / "residential_to_services_top1.parquet")
    base_walk = base_walk.loc[base_walk["service_name"] == service]
    heat_walk = heat_walk.loc[heat_walk["service_name"] == service]
    base_pt = _concat_existing([
        BASE_EXPERIMENT_ROOT / "residential_to_services_pt_top1_walk_lt15" / city / "residential_to_services_pt_top1.parquet",
        BASE_EXPERIMENT_ROOT / "residential_to_services_pt_top1_walk15plus" / city / "residential_to_services_pt_top1.parquet",
    ])
    heat_pt = _concat_existing([
        HEAT_ROOT / "pt_heat_lt" / city / "residential_to_services_pt_top1.parquet",
        HEAT_ROOT / "pt_heat_ge" / city / "residential_to_services_pt_top1.parquet",
    ])
    if not base_pt.empty:
        base_pt = base_pt.loc[base_pt["service_name"] == service]
    if not heat_pt.empty:
        heat_pt = heat_pt.loc[heat_pt["service_name"] == service]

    buildings_idx = buildings.set_index("building_idx")
    cases: list[dict] = []
    for _, row in compare.sort_values("delta_min", ascending=False).iterrows():
        bidx = int(row["building_idx"])
        base_walk_row = base_walk.loc[base_walk["building_idx"] == bidx]
        heat_walk_row = heat_walk.loc[heat_walk["building_idx"] == bidx]
        base_walk_row = base_walk_row.iloc[0] if not base_walk_row.empty else None
        heat_walk_row = heat_walk_row.iloc[0] if not heat_walk_row.empty else None
        base_pt_row = _pick_pt_row(base_pt, bidx, service)
        heat_pt_row = _pick_pt_row(heat_pt, bidx, service)
        try:
            if row["mode_baseline"] == "walk":
                base_path = nx.shortest_path(
                    base_walk_graph,
                    int(base_walk_row["home_graph_node"]),
                    int(base_walk_row["service_graph_node"]),
                    weight="time_min",
                )
                base_graph_use, base_nodes_use = base_walk_graph, base_nodes
            else:
                base_path = nx.shortest_path(
                    base_graph,
                    int(base_pt_row["home_graph_node"]),
                    int(base_pt_row["nearest_service_graph_node"]),
                    weight="time_min",
                )
                base_graph_use, base_nodes_use = base_graph, base_nodes
            if row["mode_heat"] == "walk":
                heat_path = nx.shortest_path(
                    heat_walk_graph,
                    int(heat_walk_row["home_graph_node"]),
                    int(heat_walk_row["service_graph_node"]),
                    weight="time_min",
                )
                heat_graph_use, heat_nodes_use = heat_walk_graph, heat_nodes
            else:
                heat_path = nx.shortest_path(
                    heat_graph,
                    int(heat_pt_row["home_graph_node"]),
                    int(heat_pt_row["nearest_service_graph_node"]),
                    weight="time_min",
                )
                heat_graph_use, heat_nodes_use = heat_graph, heat_nodes
        except Exception:
            continue
        base_edges = _route_edge_keys(base_path)
        heat_edges = _route_edge_keys(heat_path)
        union = len(base_edges | heat_edges)
        overlap = len(base_edges & heat_edges)
        dissim = 1.0 - (overlap / union if union else 1.0)
        mode_switch = int(row["mode_baseline"] != row["mode_heat"])
        if mode_filter == "pt_switch" and row["mode_baseline"] == row["mode_heat"]:
            continue
        if mode_filter == "pt_pt_changed" and not (row["mode_baseline"] == "pt" and row["mode_heat"] == "pt"):
            continue
        if mode_filter == "pt_involved" and not (row["mode_baseline"] == "pt" or row["mode_heat"] == "pt"):
            continue
        if dissim < 0.45 and mode_switch == 0:
            continue
        cases.append(
            {
                "building_idx": bidx,
                "delta_min": float(row["delta_min"]),
                "mode_baseline": row["mode_baseline"],
                "mode_heat": row["mode_heat"],
                "dissimilarity": dissim,
                "home": buildings_idx.loc[bidx].geometry,
                "base_gdf": _path_gdf(base_path, base_graph_use, base_nodes_use, boundary.crs),
                "heat_gdf": _path_gdf(heat_path, heat_graph_use, heat_nodes_use, boundary.crs),
                "base_walk_gdf": _split_path_gdfs(base_path, base_graph_use, base_nodes_use, boundary.crs)[0],
                "base_pt_gdf": _split_path_gdfs(base_path, base_graph_use, base_nodes_use, boundary.crs)[1],
                "heat_walk_gdf": _split_path_gdfs(heat_path, heat_graph_use, heat_nodes_use, boundary.crs)[0],
                "heat_pt_gdf": _split_path_gdfs(heat_path, heat_graph_use, heat_nodes_use, boundary.crs)[1],
            }
        )
    cases = sorted(cases, key=lambda x: (x["dissimilarity"], x["delta_min"]), reverse=True)
    dedup = []
    seen = set()
    kept_homes = []
    kept_sigs = []
    for case in cases:
        if case["building_idx"] in seen:
            continue
        home = case["home"]
        if any(home.distance(prev_home) < 250 for prev_home in kept_homes):
            continue
        sig = _route_signature(case)
        duplicate_pattern = False
        for prev_sig in kept_sigs:
            inter = len(sig & prev_sig)
            union = len(sig | prev_sig)
            if union and inter / union > 0.75:
                duplicate_pattern = True
                break
        if duplicate_pattern:
            continue
        seen.add(case["building_idx"])
        dedup.append(case)
        kept_homes.append(home)
        kept_sigs.append(sig)
    if mode_filter == "pt_pt_changed" and len(dedup) > 4:
        dedup = [dedup[0], dedup[4], dedup[2], dedup[3]]
    return boundary, dedup[:4], services_pts, _background_edges(base_walk_graph, base_nodes, boundary.crs)


def render_city(city: str, service: str, mode_filter: str | None = None) -> Path:
    boundary, cases, services_pts, bg = _build_cases(city, service, mode_filter=mode_filter)
    out_dir = HEAT_ROOT / "city_story_maps" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode_filter == "pt_switch":
        suffix = "04_super_changed_routes_overlay_ptswitch.png"
    elif mode_filter == "pt_pt_changed":
        suffix = "05_pt_to_pt_changed_routes_overlay.png"
    else:
        suffix = "04_super_changed_routes_overlay.png"
    out = out_dir / suffix
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=320)
    axes = axes.ravel()
    for ax, case in zip(axes, cases, strict=False):
        _plot_boundary(ax, boundary)
        if not bg.empty:
            bg.plot(ax=ax, color="#c5d0dc", linewidth=0.45, alpha=0.82, zorder=1)
        services_pts.plot(ax=ax, color=SERVICE_COLOR, markersize=SERVICE_MARKER_SIZE, marker=SERVICE_MARKER, alpha=0.98, zorder=2)
        ax.scatter([case["home"].x], [case["home"].y], s=HOME_MARKER_SIZE, c=HOME_COLOR, marker=HOME_MARKER, edgecolors="white", linewidths=1.1, zorder=3)
        if not case["base_walk_gdf"].empty:
            case["base_walk_gdf"].plot(ax=ax, color="#7f1d1d", linewidth=3.8, alpha=0.42, zorder=4)
        if not case["base_pt_gdf"].empty:
            case["base_pt_gdf"].plot(ax=ax, color=BASELINE_PT_COLOR, linewidth=4.2, alpha=0.82, linestyle=PT_LINESTYLE, zorder=5)
        if not case["heat_walk_gdf"].empty:
            case["heat_walk_gdf"].plot(ax=ax, color="#16a34a", linewidth=3.8, alpha=0.48, zorder=6)
        if not case["heat_pt_gdf"].empty:
            case["heat_pt_gdf"].plot(ax=ax, color=HEAT_PT_COLOR, linewidth=4.2, alpha=0.9, linestyle=PT_LINESTYLE, zorder=7)
        bounds_parts = [
            g
            for g in [
                case["base_walk_gdf"],
                case["base_pt_gdf"],
                case["heat_walk_gdf"],
                case["heat_pt_gdf"],
            ]
            if not g.empty
        ]
        if bounds_parts:
            all_bounds = np.array([g.total_bounds for g in bounds_parts])
            xmin, ymin = all_bounds[:, 0].min(), all_bounds[:, 1].min()
            xmax, ymax = all_bounds[:, 2].max(), all_bounds[:, 3].max()
            pad_x = (xmax - xmin) * 0.22 if xmax > xmin else 160
            pad_y = (ymax - ymin) * 0.22 if ymax > ymin else 160
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.set_title(
            f"дом {case['building_idx']} | Δ={case['delta_min']:.2f} мин | "
            f"{case['mode_baseline']} → {case['mode_heat']} | diff={case['dissimilarity']:.2f}",
            fontsize=16,
        )
    for ax in axes[len(cases):]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], color="#7f1d1d", lw=5, alpha=0.42, label="старый пеший участок"),
        Line2D([0], [0], color=BASELINE_PT_COLOR, lw=5, alpha=0.82, linestyle=PT_LINESTYLE, label="старый PT-участок"),
        Line2D([0], [0], color="#16a34a", lw=5, alpha=0.48, label="новый пеший участок"),
        Line2D([0], [0], color=HEAT_PT_COLOR, lw=5, alpha=0.9, linestyle=PT_LINESTYLE, label="новый PT-участок"),
        Line2D([0], [0], marker=HOME_MARKER, color="none", markerfacecolor=HOME_COLOR, markeredgecolor="white", markersize=15, label="дом"),
        Line2D([0], [0], marker=SERVICE_MARKER, color="none", markerfacecolor=SERVICE_COLOR, markeredgecolor=SERVICE_COLOR, markersize=17, label="целевой сервис"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=19, bbox_to_anchor=(0.5, 0.015))
    title = f"{CITY_RU.get(city, city)} — самые сильные изменения маршрутов (baseline vs heat)"
    if mode_filter == "pt_switch":
        title = f"{CITY_RU.get(city, city)} — самые сильные PT-переключения маршрута (baseline vs heat)"
    elif mode_filter == "pt_pt_changed":
        title = f"{CITY_RU.get(city, city)} — самые сильные изменения маршрутов с PT до и после"
    fig.suptitle(title, fontsize=24, y=0.93)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.09, hspace=0.10, wspace=0.06)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--service", default="polyclinic")
    parser.add_argument("--mode-filter", default=None, choices=[None, "pt_switch", "pt_involved", "pt_pt_changed"])
    args = parser.parse_args()
    for city in args.cities:
        out = render_city(city, args.service, mode_filter=args.mode_filter)
        print(out)


if __name__ == "__main__":
    main()

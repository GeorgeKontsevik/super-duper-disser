#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import copy
from matplotlib.lines import Line2D
from shapely import wkb
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_heat_service_city_pairs_and_routes import (  # noqa: E402
    CITY_RU,
    SERVICE_RU,
    _edge_geometry,
    _legend_handles,
    _living_points,
    _load_graph,
    _load_nodes,
    _pick_pt_row,
    _plot_boundary,
    _walk_subgraph,
)


UTCI_ROOT = ROOT / "thermal_access_pilot/outputs/batch_utci_links_2km_hottest_summer2025"
HEAT_ROOT = ROOT / "thermal_access_pilot/outputs/batch_service_access_hottest_summer2025"
BASE_EXPERIMENT_ROOT = ROOT / "aggregated_spatial_pipeline/outputs/experiments_active19_20260412"
JOINT_INPUTS = ROOT / "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"

UTCI_ORDER = [
    "<9°C cold stress",
    "9–26°C no thermal stress",
    "26–32°C moderate heat stress",
    "32–38°C strong heat stress",
    "38–46°C very strong heat stress",
    ">46°C extreme heat stress",
]
UTCI_COLORS = {
    "<9°C cold stress": "#2b83ba",
    "9–26°C no thermal stress": "#abdda4",
    "26–32°C moderate heat stress": "#ffffbf",
    "32–38°C strong heat stress": "#fdae61",
    "38–46°C very strong heat stress": "#f46d43",
    ">46°C extreme heat stress": "#a50026",
}
UTCI_LABELS_RU = {
    "<9°C cold stress": "< 9°C — холодовой стресс",
    "9–26°C no thermal stress": "9–26°C — без терм. стресса",
    "26–32°C moderate heat stress": "26–32°C — умеренный тепловой стресс",
    "32–38°C strong heat stress": "32–38°C — сильный тепловой стресс",
    "38–46°C very strong heat stress": "38–46°C — очень сильный тепловой стресс",
    ">46°C extreme heat stress": "> 46°C — экстремальный тепловой стресс",
}


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


def _load_diag_compare() -> pd.DataFrame:
    base = pd.read_parquet(HEAT_ROOT / "diag_baseline" / CITY / "home_to_service_access_diagnostics.parquet")
    heat = pd.read_parquet(HEAT_ROOT / "diag_heat" / CITY / "home_to_service_access_diagnostics.parquet")
    base = base.loc[base["service_name"] == SERVICE].copy()
    heat = heat.loc[heat["service_name"] == SERVICE].copy()
    merged = base.merge(heat, on=["building_idx", "service_name"], suffixes=("_baseline", "_heat"))
    merged["mode_baseline"] = np.where(
        merged["walk_time_min_baseline"] <= merged["effective_pt_total_min_baseline"],
        "walk",
        "pt",
    )
    merged["mode_heat"] = np.where(
        merged["walk_time_min_heat"] <= merged["effective_pt_total_min_heat"],
        "walk",
        "pt",
    )
    merged["effective_baseline_min"] = np.minimum(merged["walk_time_min_baseline"], merged["effective_pt_total_min_baseline"])
    merged["effective_heat_min"] = np.minimum(merged["walk_time_min_heat"], merged["effective_pt_total_min_heat"])
    merged["delta_min"] = merged["effective_heat_min"] - merged["effective_baseline_min"]
    return merged


def _background_edges(graph: nx.Graph, nodes_gdf: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    geoms = []
    for u, v in graph.edges():
        try:
            geoms.append(_edge_geometry(graph, nodes_gdf, int(u), int(v)))
        except Exception:
            continue
    return gpd.GeoDataFrame({"geometry": geoms}, geometry="geometry", crs=nodes_gdf.crs).to_crs(crs)


def _walk_graph_with_utci_factors(base_graph: nx.MultiDiGraph, utci_edges: gpd.GeoDataFrame) -> nx.Graph:
    walk_graph = _walk_subgraph(base_graph)
    pair_factor: dict[tuple[int, int], float] = {}
    for row in utci_edges.itertuples(index=False):
        key = tuple(sorted((int(row.u), int(row.v))))
        cand_len = float(getattr(row, "length_meter", np.nan))
        cand_factor = float(getattr(row, "cost_factor", 1.0))
        prev = pair_factor.get(key)
        if prev is None:
            pair_factor[key] = cand_factor
            continue
        # keep the more expensive factor if duplicates survived
        pair_factor[key] = max(prev, cand_factor)
    for u, v, data in walk_graph.edges(data=True):
        key = tuple(sorted((int(u), int(v))))
        factor = float(pair_factor.get(key, 1.0))
        data["baseline_time_min"] = float(data["time_min"])
        data["time_min"] = float(data["time_min"]) * factor
        data["cost_factor"] = factor
    return walk_graph


def _snap_points(points: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, id_col: str) -> pd.DataFrame:
    cols = [id_col, "geometry"]
    pts = points[cols].copy()
    snapped = gpd.sjoin_nearest(pts, nodes[["index", "geometry"]], how="left", distance_col="snap_m")
    return pd.DataFrame({id_col: snapped[id_col].astype(int), "node_id": snapped["index"].astype(int), "snap_m": snapped["snap_m"].astype(float)})


def _best_edge_data(graph: nx.MultiDiGraph | nx.Graph, u: int, v: int) -> dict:
    if isinstance(graph, nx.MultiDiGraph):
        bundle = graph.get_edge_data(u, v)
        return min(bundle.values(), key=lambda d: float(d.get("time_min", np.inf)))
    return graph.get_edge_data(u, v)


def _split_edges(path_nodes: list[int], graph: nx.MultiDiGraph | nx.Graph, nodes_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    walk_geoms = []
    pt_geoms = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False):
        geom = _edge_geometry(graph, nodes_gdf, int(u), int(v))
        if isinstance(graph, nx.MultiDiGraph):
            edge_type = str(_best_edge_data(graph, int(u), int(v)).get("type", "")).lower()
            if edge_type == "walk":
                walk_geoms.append(geom)
            else:
                pt_geoms.append(geom)
        else:
            walk_geoms.append(geom)
    walk = gpd.GeoDataFrame({"geometry": walk_geoms}, geometry="geometry", crs=nodes_gdf.crs)
    pt = gpd.GeoDataFrame({"geometry": pt_geoms}, geometry="geometry", crs=nodes_gdf.crs)
    return walk, pt


def render_utci_links(boundary: gpd.GeoDataFrame) -> Path:
    edges = _load_utci_edges(boundary.crs)
    fig = plt.figure(figsize=(12, 12.8), dpi=320)
    gs = fig.add_gridspec(2, 1, height_ratios=[18, 1.6], hspace=0.02)
    ax = fig.add_subplot(gs[0])
    lax = fig.add_subplot(gs[1])
    lax.axis("off")
    _plot_boundary(ax, boundary)
    for cls in UTCI_ORDER:
        sub = edges.loc[edges["utci_class"] == cls]
        if not sub.empty:
            sub.plot(ax=ax, color=UTCI_COLORS[cls], linewidth=1.1, alpha=0.95)
    minx, miny, maxx, maxy = edges.total_bounds
    pad_x = max((maxx - minx) * 0.005, 5)
    pad_y = max((maxy - miny) * 0.005, 5)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.margins(0)
    ax.set_title(f"{CITY_RU[CITY]} — пешеходные линки по группам UTCI", fontsize=18)
    handles = [Line2D([0], [0], color=UTCI_COLORS[c], lw=3, label=UTCI_LABELS_RU[c]) for c in UTCI_ORDER]
    lax.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04)
    out = OUT_DIR / "01_utci_links_by_class.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _load_utci_edges(boundary_crs):
    path = UTCI_ROOT / CITY / "tables" / "pedestrian_links_utci.parquet"
    edges = gpd.read_parquet(path)
    if isinstance(edges.geometry.iloc[0], (bytes, bytearray)):
        edges["geometry"] = edges["geometry"].apply(wkb.loads)
        edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=boundary_crs)
    else:
        edges = gpd.GeoDataFrame(edges, geometry="geometry")
        if edges.crs is None:
            edges = edges.set_crs(boundary_crs)
    edges = edges.to_crs(boundary_crs)
    if "utci_c" in edges.columns:
        utci = pd.to_numeric(edges["utci_c"], errors="coerce")
        edges = edges.loc[np.isfinite(utci)].copy()
    return edges


def _utci_bbox_boundary(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    utci_edges = _load_utci_edges(boundary.crs)
    minx, miny, maxx, maxy = utci_edges.total_bounds
    return gpd.GeoDataFrame({"geometry": [box(minx, miny, maxx, maxy)]}, geometry="geometry", crs=boundary.crs)


def _load_context_layers(boundary: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    city_root = JOINT_INPUTS / CITY
    water = gpd.read_parquet(city_root / "blocksnet" / "water.parquet").to_crs(boundary.crs)
    buildings = gpd.read_parquet(city_root / "blocksnet" / "buildings.parquet").to_crs(boundary.crs)
    if not boundary.empty:
        clip_geom = boundary.union_all()
        water = water.loc[water.geometry.notna() & ~water.geometry.is_empty & water.geometry.intersects(clip_geom)].copy()
        buildings = buildings.loc[
            buildings.geometry.notna() & ~buildings.geometry.is_empty & buildings.geometry.intersects(clip_geom)
        ].copy()
    return water, buildings


def render_utci_links_with_context(boundary: gpd.GeoDataFrame) -> Path:
    edges = _load_utci_edges(boundary.crs)
    water, buildings = _load_context_layers(boundary)
    fig = plt.figure(figsize=(12, 12.8), dpi=320)
    gs = fig.add_gridspec(2, 1, height_ratios=[18, 1.6], hspace=0.02)
    ax = fig.add_subplot(gs[0])
    lax = fig.add_subplot(gs[1])
    lax.axis("off")
    _plot_boundary(ax, boundary)
    if not water.empty:
        water.plot(ax=ax, color="#d8ecff", edgecolor="none", alpha=0.95, zorder=1)
    if not buildings.empty:
        buildings.plot(ax=ax, facecolor="#eeeeee", edgecolor="#d4d4d4", linewidth=0.10, alpha=0.80, zorder=2)
    for cls in UTCI_ORDER:
        sub = edges.loc[edges["utci_class"] == cls]
        if not sub.empty:
            sub.plot(ax=ax, color=UTCI_COLORS[cls], linewidth=1.1, alpha=0.98, zorder=3)
    minx, miny, maxx, maxy = edges.total_bounds
    pad_x = max((maxx - minx) * 0.005, 5)
    pad_y = max((maxy - miny) * 0.005, 5)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.margins(0)
    ax.set_title(f"{CITY_RU[CITY]} — пешеходные линки UTCI + вода и здания", fontsize=18)
    handles = [Line2D([0], [0], color=UTCI_COLORS[c], lw=3, label=UTCI_LABELS_RU[c]) for c in UTCI_ORDER]
    handles += [
        Line2D([0], [0], color="#d8ecff", lw=7, label="вода"),
        Line2D([0], [0], color="#bdbdbd", lw=7, label="здания"),
    ]
    lax.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.04)
    out = OUT_DIR / "01b_utci_links_with_water_buildings.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def render_delta_buildings(boundary: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, compare: pd.DataFrame) -> Path:
    pts = buildings.merge(compare[["building_idx", "delta_min"]], on="building_idx", how="left")
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=buildings.crs).to_crs(boundary.crs)
    base_graph = _load_graph(str(JOINT_INPUTS / CITY / "intermodal_graph_iduedu" / "graph.pkl"))
    base_nodes = _load_nodes(str(JOINT_INPUTS / CITY / "intermodal_graph_iduedu" / "graph_nodes.parquet"))
    bg = _background_edges(_walk_subgraph(base_graph), base_nodes, boundary.crs)
    water, _buildings_ctx = _load_context_layers(boundary)
    positive = pts.loc[pts["delta_min"].fillna(0) > 1e-9].copy()
    zeroish = pts.loc[~pts.index.isin(positive.index)].copy()
    bins = [0, 1, 2, 4, 6, float(max(positive["delta_min"].max(), 8))] if not positive.empty else [0, 1]
    colors = ["#fde68a", "#fbbf24", "#f97316", "#ef4444", "#991b1b"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=320)
    _plot_boundary(ax, boundary)
    if not water.empty:
        water.plot(ax=ax, color="#d8ecff", edgecolor="none", alpha=0.95, zorder=1)
    if not bg.empty:
        bg.plot(ax=ax, color="#cbd5e1", linewidth=0.45, alpha=0.75, zorder=2)
    if not zeroish.empty:
        ax.scatter(zeroish.geometry.x, zeroish.geometry.y, s=5, c="#cbd5e1", alpha=0.50, linewidths=0, rasterized=True, zorder=3)
    legend_handles = [
        Line2D([0], [0], color="#d8ecff", lw=7, label="вода"),
        Line2D([0], [0], color="#cbd5e1", lw=3, label="дороги"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#cbd5e1", markeredgecolor="none", markersize=8, label="0 / без увеличения"),
    ]
    if not positive.empty:
        labels = []
        for lo, hi in zip(bins[:-1], bins[1:], strict=False):
            labels.append((lo, hi))
        for (lo, hi), color in zip(labels, colors, strict=False):
            if hi == bins[-1]:
                mask = positive["delta_min"].between(lo, hi, inclusive="both")
                label = f"{lo:.0f}–{hi:.2f} мин"
            else:
                mask = (positive["delta_min"] > lo) & (positive["delta_min"] <= hi)
                label = f">{lo:.0f}–{hi:.0f} мин"
            sub = positive.loc[mask]
            if sub.empty:
                continue
            ax.scatter(sub.geometry.x, sub.geometry.y, s=7, c=color, alpha=0.82, linewidths=0, rasterized=True, zorder=4)
            legend_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none", markersize=8, label=label))
    ax.set_title(f"{CITY_RU[CITY]} — дома по увеличению времени heat vs baseline", fontsize=18)
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12)
    out = OUT_DIR / "02_buildings_delta_time_vs_baseline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def compute_same_run_walk_compare(boundary: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, services: gpd.GeoDataFrame) -> pd.DataFrame:
    base_graph = _load_graph(str(JOINT_INPUTS / CITY / "intermodal_graph_iduedu" / "graph.pkl"))
    nodes = _load_nodes(str(JOINT_INPUTS / CITY / "intermodal_graph_iduedu" / "graph_nodes.parquet")).to_crs(boundary.crs)
    utci_edges = _load_utci_edges(boundary.crs)
    base_walk_graph = _walk_subgraph(base_graph)
    heat_walk_graph = _walk_graph_with_utci_factors(base_graph, utci_edges)

    b = buildings.copy().to_crs(boundary.crs)
    if "building_idx" not in b.columns:
        b = b.reset_index(drop=True)
        b["building_idx"] = b.index.astype(int)
    b["geometry"] = b.geometry.representative_point()
    s = services.copy().to_crs(boundary.crs).reset_index(drop=True)
    s["service_idx"] = s.index.astype(int)
    s["geometry"] = s.geometry.representative_point()

    b_snap = _snap_points(b[["building_idx", "geometry"]], nodes, "building_idx")
    s_snap = _snap_points(s[["service_idx", "geometry"]], nodes, "service_idx")
    service_nodes = sorted(s_snap["node_id"].unique().tolist())

    base_lengths = nx.multi_source_dijkstra_path_length(base_walk_graph, service_nodes, weight="time_min")
    heat_lengths = nx.multi_source_dijkstra_path_length(heat_walk_graph, service_nodes, weight="time_min")

    out = b_snap.copy()
    out["baseline_min"] = out["node_id"].map(base_lengths)
    out["heat_min"] = out["node_id"].map(heat_lengths)
    out["delta_min"] = pd.to_numeric(out["heat_min"], errors="coerce") - pd.to_numeric(out["baseline_min"], errors="coerce")
    return out[["building_idx", "baseline_min", "heat_min", "delta_min"]]


def _choose_cases(compare: pd.DataFrame, base_graph: nx.MultiDiGraph, heat_graph: nx.MultiDiGraph, base_walk: pd.DataFrame, heat_walk: pd.DataFrame, base_pt: pd.DataFrame, heat_pt: pd.DataFrame) -> list[dict]:
    picked: list[dict] = []
    pt_case = None
    for _, row in compare.sort_values("delta_min", ascending=False).iterrows():
        bidx = int(row["building_idx"])
        base_walk_row = base_walk.loc[base_walk["building_idx"] == bidx]
        heat_walk_row = heat_walk.loc[heat_walk["building_idx"] == bidx]
        base_walk_row = base_walk_row.iloc[0] if not base_walk_row.empty else None
        heat_walk_row = heat_walk_row.iloc[0] if not heat_walk_row.empty else None
        base_pt_row = _pick_pt_row(base_pt, bidx, SERVICE)
        heat_pt_row = _pick_pt_row(heat_pt, bidx, SERVICE)
        try:
            if row["mode_baseline"] == "walk":
                base_path = nx.shortest_path(base_walk_graph, int(base_walk_row["home_graph_node"]), int(base_walk_row["service_graph_node"]), weight="time_min")
            else:
                base_path = nx.shortest_path(base_graph, int(base_pt_row["home_graph_node"]), int(base_pt_row["nearest_service_graph_node"]), weight="time_min")
            if row["mode_heat"] == "walk":
                heat_path = nx.shortest_path(heat_walk_graph, int(heat_walk_row["home_graph_node"]), int(heat_walk_row["service_graph_node"]), weight="time_min")
            else:
                heat_path = nx.shortest_path(heat_graph, int(heat_pt_row["home_graph_node"]), int(heat_pt_row["nearest_service_graph_node"]), weight="time_min")
        except Exception:
            continue
        changed = tuple(base_path) != tuple(heat_path) or row["mode_baseline"] != row["mode_heat"]
        if not changed:
            continue
        candidate = row.to_dict() | {
            "base_path": base_path,
            "heat_path": heat_path,
        }
        if row["mode_heat"] == "pt" and pt_case is None:
            pt_case = candidate
        if len(picked) < 3:
            picked.append(candidate)
        if len(picked) >= 3 and pt_case is not None:
            break
    if pt_case is not None and all(int(x["building_idx"]) != int(pt_case["building_idx"]) for x in picked):
        picked = picked[:2] + [pt_case]
    seen = set()
    out = []
    for row in picked:
        bidx = int(row["building_idx"])
        if bidx in seen:
            continue
        seen.add(bidx)
        out.append(row)
    return out[:3]


def render_top3_routes(boundary: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame, services: gpd.GeoDataFrame, compare: pd.DataFrame) -> Path:
    base_graph = _load_graph(str(JOINT_INPUTS / CITY / "intermodal_graph_iduedu" / "graph.pkl"))
    heat_graph = _load_graph(str(HEAT_ROOT / "heat_joint_inputs" / CITY / "intermodal_graph_iduedu" / "graph.pkl"))
    base_nodes = _load_nodes(str(JOINT_INPUTS / CITY / "intermodal_graph_iduedu" / "graph_nodes.parquet"))
    heat_nodes = _load_nodes(str(HEAT_ROOT / "heat_joint_inputs" / CITY / "intermodal_graph_iduedu" / "graph_nodes.parquet"))
    global base_walk_graph, heat_walk_graph
    base_walk_graph = _walk_subgraph(base_graph)
    heat_walk_graph = _walk_subgraph(heat_graph)

    base_walk = pd.read_parquet(BASE_EXPERIMENT_ROOT / "residential_to_services_top1" / CITY / "residential_to_services_top1.parquet")
    heat_walk = pd.read_parquet(HEAT_ROOT / "walk_heat" / CITY / "residential_to_services_top1.parquet")
    base_walk = base_walk.loc[base_walk["service_name"] == SERVICE]
    heat_walk = heat_walk.loc[heat_walk["service_name"] == SERVICE]
    base_pt = _concat_existing(
        [
            BASE_EXPERIMENT_ROOT / "residential_to_services_pt_top1_walk_lt15" / CITY / "residential_to_services_pt_top1.parquet",
            BASE_EXPERIMENT_ROOT / "residential_to_services_pt_top1_walk15plus" / CITY / "residential_to_services_pt_top1.parquet",
        ]
    )
    heat_pt = _concat_existing(
        [
            HEAT_ROOT / "pt_heat_lt" / CITY / "residential_to_services_pt_top1.parquet",
            HEAT_ROOT / "pt_heat_ge" / CITY / "residential_to_services_pt_top1.parquet",
        ]
    )
    if not base_pt.empty:
        base_pt = base_pt.loc[base_pt["service_name"] == SERVICE]
    if not heat_pt.empty:
        heat_pt = heat_pt.loc[heat_pt["service_name"] == SERVICE]

    compare = compare.loc[compare["building_idx"].isin(buildings["building_idx"])].copy()
    cases = _choose_cases(compare, base_graph, heat_graph, base_walk, heat_walk, base_pt, heat_pt)
    bg = _background_edges(base_walk_graph, base_nodes, boundary.crs)
    buildings_idx = buildings.set_index("building_idx")
    services_pts = services.to_crs(boundary.crs)

    fig, axes = plt.subplots(3, 2, figsize=(16, 20), dpi=320)
    for row_axes, case in zip(axes, cases, strict=False):
        old_ax, new_ax = row_axes
        if case["mode_baseline"] == "walk":
            old_walk, old_pt = _split_edges(case["base_path"], base_walk_graph, base_nodes)
        else:
            old_walk, old_pt = _split_edges(case["base_path"], base_graph, base_nodes)
        if case["mode_heat"] == "walk":
            new_walk, new_pt = _split_edges(case["heat_path"], heat_walk_graph, heat_nodes)
        else:
            new_walk, new_pt = _split_edges(case["heat_path"], heat_graph, heat_nodes)
        home = buildings_idx.loc[int(case["building_idx"])].geometry
        bounds_parts = []
        for g in [old_walk, old_pt, new_walk, new_pt]:
            if not g.empty:
                bounds_parts.append(g.to_crs(boundary.crs))
        xmin, ymin, xmax, ymax = boundary.total_bounds
        if bounds_parts:
            all_bounds = np.array([g.total_bounds for g in bounds_parts])
            xmin, ymin = all_bounds[:, 0].min(), all_bounds[:, 1].min()
            xmax, ymax = all_bounds[:, 2].max(), all_bounds[:, 3].max()
        pad_x = (xmax - xmin) * 0.25 if xmax > xmin else 200
        pad_y = (ymax - ymin) * 0.25 if ymax > ymin else 200
        for ax in [old_ax, new_ax]:
            _plot_boundary(ax, boundary)
            bg.plot(ax=ax, color="#dbe4f0", linewidth=0.35, alpha=0.7)
            services_pts.plot(ax=ax, color="#111827", markersize=50, marker="*", alpha=0.92, zorder=4)
            ax.scatter([home.x], [home.y], s=42, c="#f97316", linewidths=0, zorder=5)
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)
        if not old_walk.empty:
            old_walk.to_crs(boundary.crs).plot(ax=old_ax, color="#16a34a", linewidth=2.0, alpha=0.95)
        if not old_pt.empty:
            old_pt.to_crs(boundary.crs).plot(ax=old_ax, color="#2563eb", linewidth=2.0, alpha=0.95)
        if not new_walk.empty:
            new_walk.to_crs(boundary.crs).plot(ax=new_ax, color="#16a34a", linewidth=2.0, alpha=0.95)
        if not new_pt.empty:
            new_pt.to_crs(boundary.crs).plot(ax=new_ax, color="#2563eb", linewidth=2.0, alpha=0.95)
        old_ax.set_title(f"baseline | дом {int(case['building_idx'])} | Δ={float(case['delta_min']):.2f} мин\n{case['mode_baseline']}", fontsize=13)
        new_ax.set_title(f"heat | дом {int(case['building_idx'])}\n{case['mode_heat']}", fontsize=13)
    handles = [
        Line2D([0], [0], color="#16a34a", lw=3, label="пеший участок"),
        Line2D([0], [0], color="#2563eb", lw=3, label="OT участок"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f97316", markeredgecolor="#f97316", markersize=10, label="дом"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111827", markeredgecolor="#111827", markersize=12, label="сервис"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{CITY_RU[CITY]} — 3 дома с наибольшим изменением пути: baseline vs heat", fontsize=18, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.06, hspace=0.15, wspace=0.06)
    out = OUT_DIR / "03_top3_changed_routes_old_vs_new.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="debrecen_hungary")
    parser.add_argument("--service", default="polyclinic")
    args = parser.parse_args()
    global CITY, SERVICE, OUT_DIR
    CITY = args.city
    SERVICE = args.service
    OUT_DIR = HEAT_ROOT / "city_story_maps" / CITY
    city_root = JOINT_INPUTS / CITY
    boundary = gpd.read_parquet(city_root / "blocksnet" / "boundary.parquet")
    buildings = _living_points(city_root)
    services = gpd.read_parquet(city_root / "pipeline_2" / "services_raw" / f"{SERVICE}.parquet")
    services["geometry"] = services.geometry.representative_point()
    metric_crs = boundary.estimate_utm_crs()
    if metric_crs is not None:
        boundary = boundary.to_crs(metric_crs)
        buildings = buildings.to_crs(metric_crs)
        services = services.to_crs(metric_crs)
    utci_boundary = _utci_bbox_boundary(boundary)
    compare = _load_diag_compare()
    buildings = gpd.clip(buildings.to_crs(utci_boundary.crs), utci_boundary)
    services = gpd.clip(services.to_crs(utci_boundary.crs), utci_boundary)
    same_run_walk_compare = compute_same_run_walk_compare(utci_boundary, buildings, services)
    p1 = render_utci_links(utci_boundary)
    p1b = render_utci_links_with_context(utci_boundary)
    p2 = render_delta_buildings(utci_boundary, buildings, same_run_walk_compare)
    p3 = render_top3_routes(utci_boundary, buildings, services, compare)
    print({"utci_links": str(p1), "utci_links_context": str(p1b), "delta_buildings": str(p2), "top3_routes": str(p3)})


if __name__ == "__main__":
    main()

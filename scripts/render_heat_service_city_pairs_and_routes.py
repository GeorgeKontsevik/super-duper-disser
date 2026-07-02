#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString


SERVICES = ["school", "polyclinic"]
LABEL_ORDER = [
    "ok_walk",
    "ok_pt_only",
    "failed_no_pt_path",
    "failed_access_gt_threshold",
    "failed_egress_gt_threshold",
    "failed_access_egress_sum_gt_threshold",
    "failed_in_vehicle_gt_threshold",
    "failed_transfer_gt_threshold",
    "failed_multi_component_gt_threshold",
    "failed_total_gt_threshold_no_single_component_gt_threshold",
]
PLOT_ORDER = [
    "failed_no_pt_path",
    "failed_transfer_gt_threshold",
    "failed_access_gt_threshold",
    "failed_egress_gt_threshold",
    "failed_access_egress_sum_gt_threshold",
    "failed_in_vehicle_gt_threshold",
    "failed_multi_component_gt_threshold",
    "failed_total_gt_threshold_no_single_component_gt_threshold",
    "ok_pt_only",
    "ok_walk",
]
LABEL_COLORS = {
    "ok_walk": "#16a34a",
    "ok_pt_only": "#2563eb",
    "failed_no_pt_path": "#475569",
    "failed_access_gt_threshold": "#f59e0b",
    "failed_egress_gt_threshold": "#fb7185",
    "failed_access_egress_sum_gt_threshold": "#f97316",
    "failed_in_vehicle_gt_threshold": "#dc2626",
    "failed_transfer_gt_threshold": "#7c3aed",
    "failed_multi_component_gt_threshold": "#8b5cf6",
    "failed_total_gt_threshold_no_single_component_gt_threshold": "#6b7280",
}
LABEL_RU = {
    "ok_walk": "доступно пешком",
    "ok_pt_only": "доступно на ОТ",
    "failed_no_pt_path": "нет пути по ОТ",
    "failed_access_gt_threshold": "дом - остановка > 15 мин",
    "failed_egress_gt_threshold": "остановка - сервис > 15 мин",
    "failed_access_egress_sum_gt_threshold": "оба пеших участка > 15 мин",
    "failed_in_vehicle_gt_threshold": "поездка в ОТ > 15 мин",
    "failed_transfer_gt_threshold": "пересадки > 15 мин",
    "failed_multi_component_gt_threshold": "несколько компонент > 15 мин",
    "failed_total_gt_threshold_no_single_component_gt_threshold": "сумма > 15 мин, без доминирующей компоненты",
}
SERVICE_RU = {"school": "Школы", "polyclinic": "Поликлиники"}
CITY_RU = {
    "bergen_norway": "Берген, Норвегия",
    "bologna_italy": "Болонья, Италия",
    "bristol_united_kingdom": "Бристоль, Великобритания",
    "brno_czechia": "Брно, Чехия",
    "coimbra_portugal": "Коимбра, Португалия",
    "debrecen_hungary": "Дебрецен, Венгрия",
    "dresden_germany": "Дрезден, Германия",
    "freiburg_im_breisgau_germany": "Фрайбург, Германия",
    "gothenburg_sweden": "Гётеборг, Швеция",
    "graz_austria": "Грац, Австрия",
    "hrodna_belarus": "Гродно, Беларусь",
    "innsbruck_austria": "Инсбрук, Австрия",
    "kaliningrad_russia": "Калининград, Россия",
    "krakow_poland": "Краков, Польша",
    "linz_austria": "Линц, Австрия",
    "lyon_france": "Лион, Франция",
    "marseille_france": "Марсель, Франция",
    "novi_sad_serbia": "Нови-Сад, Сербия",
    "porto_portugal": "Порту, Португалия",
    "turin_italy": "Турин, Италия",
    "turku_finland": "Турку, Финляндия",
    "zaragoza_spain": "Сарагоса, Испания",
}
CITY_ORDER = [
    "bergen_norway",
    "bologna_italy",
    "bristol_united_kingdom",
    "brno_czechia",
    "coimbra_portugal",
    "debrecen_hungary",
    "dresden_germany",
    "freiburg_im_breisgau_germany",
    "gothenburg_sweden",
    "graz_austria",
    "hrodna_belarus",
    "innsbruck_austria",
    "kaliningrad_russia",
    "linz_austria",
    "lyon_france",
    "novi_sad_serbia",
    "porto_portugal",
    "turin_italy",
    "turku_finland",
    "zaragoza_spain",
]


@dataclass
class ScenarioTables:
    walk: pd.DataFrame
    pt: pd.DataFrame
    diag: pd.DataFrame


def _buildings_path(city_root: Path) -> Path:
    derived = city_root / "derived_layers"
    enriched = derived / "buildings_is_living_enriched.parquet"
    return enriched if enriched.exists() else derived / "buildings_floor_enriched.parquet"


def _living_points(city_root: Path) -> gpd.GeoDataFrame:
    buildings = gpd.read_parquet(_buildings_path(city_root))
    if "is_living" in buildings.columns:
        living_mask = pd.to_numeric(buildings["is_living"], errors="coerce").fillna(0).astype(float) > 0
        buildings = buildings.loc[living_mask].copy()
    buildings = buildings.reset_index(drop=False).rename(columns={"index": "building_idx"})
    buildings["geometry"] = buildings.geometry.representative_point()
    return buildings[["building_idx", "geometry"]]


def _legend_handles():
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LABEL_COLORS[label], markeredgecolor="none", markersize=7, label=LABEL_RU[label])
        for label in LABEL_ORDER
    ]


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _concat_pt(city: str, root_lt: Path, root_ge: Path) -> pd.DataFrame:
    frames = []
    for root in [root_lt, root_ge]:
        path = root / city / "residential_to_services_pt_top1.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_scenario(city: str, walk_root: Path, pt_lt_root: Path, pt_ge_root: Path, diag_root: Path) -> ScenarioTables:
    walk = _load_parquet(walk_root / city / "residential_to_services_top1.parquet")
    pt = _concat_pt(city, pt_lt_root, pt_ge_root)
    diag = _load_parquet(diag_root / city / "home_to_service_access_diagnostics.parquet")
    return ScenarioTables(walk=walk, pt=pt, diag=diag)


@lru_cache(maxsize=128)
def _load_graph(graph_path: str) -> nx.MultiDiGraph:
    with Path(graph_path).open("rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=128)
def _load_nodes(nodes_path: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(nodes_path)


def _walk_subgraph(graph: nx.MultiDiGraph) -> nx.Graph:
    out = nx.Graph()
    for node, data in graph.nodes(data=True):
        out.add_node(node, **data)
    for u, v, data in graph.edges(data=True):
        if str(data.get("type", "")).lower() != "walk":
            continue
        time_min = float(data.get("time_min", np.inf))
        geom = data.get("geometry")
        length_meter = float(data.get("length_meter", np.nan)) if data.get("length_meter") is not None else np.nan
        if out.has_edge(u, v):
            if time_min >= float(out[u][v]["time_min"]):
                continue
        out.add_edge(u, v, time_min=time_min, geometry=geom, length_meter=length_meter)
    return out


def _best_edge_data(graph: nx.MultiDiGraph | nx.Graph, u: int, v: int) -> dict:
    if isinstance(graph, nx.MultiDiGraph):
        bundle = graph.get_edge_data(u, v)
        if not bundle:
            raise KeyError(f"missing edge {u}->{v}")
        return min(bundle.values(), key=lambda d: float(d.get("time_min", np.inf)))
    data = graph.get_edge_data(u, v)
    if not data:
        raise KeyError(f"missing edge {u}->{v}")
    return data


def _edge_geometry(graph: nx.MultiDiGraph | nx.Graph, nodes_gdf: gpd.GeoDataFrame, u: int, v: int):
    data = _best_edge_data(graph, u, v)
    geom = data.get("geometry")
    if geom is not None and not geom.is_empty:
        return geom
    node_idx = nodes_gdf.set_index("index")
    pu = node_idx.loc[int(u)].geometry
    pv = node_idx.loc[int(v)].geometry
    return LineString([pu, pv])


def _path_to_gdf(path_nodes: list[int], graph: nx.MultiDiGraph | nx.Graph, nodes_gdf: gpd.GeoDataFrame):
    if len(path_nodes) < 2:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=nodes_gdf.crs)
    geoms = [_edge_geometry(graph, nodes_gdf, int(u), int(v)) for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False)]
    return gpd.GeoDataFrame({"geometry": geoms}, geometry="geometry", crs=nodes_gdf.crs)


def _path_edge_keys(path_nodes: list[int]) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False):
        a = int(u)
        b = int(v)
        keys.append((a, b) if a <= b else (b, a))
    return keys


def _edge_keys_to_gdf(edge_keys: set[tuple[int, int]], graph: nx.MultiDiGraph | nx.Graph, nodes_gdf: gpd.GeoDataFrame):
    if not edge_keys:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=nodes_gdf.crs)
    geoms = []
    for u, v in edge_keys:
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


def _plot_boundary(ax, boundary: gpd.GeoDataFrame) -> None:
    boundary.plot(ax=ax, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.95)
    minx, miny, maxx, maxy = boundary.total_bounds
    padx = (maxx - minx) * 0.02
    pady = (maxy - miny) * 0.02
    ax.set_xlim(minx - padx, maxx + padx)
    ax.set_ylim(miny - pady, maxy + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def render_city_service_pair(
    city: str,
    service: str,
    joint_inputs_root: Path,
    diag_base_root: Path,
    diag_heat_root: Path,
    out_path: Path,
) -> None:
    city_root = joint_inputs_root / city
    buildings = _living_points(city_root)
    boundary = gpd.read_parquet(city_root / "blocksnet" / "boundary.parquet")
    metric_crs = boundary.estimate_utm_crs()
    if metric_crs is not None:
        boundary = boundary.to_crs(metric_crs)
        buildings = buildings.to_crs(metric_crs)
    base = pd.read_parquet(diag_base_root / city / "home_to_service_access_diagnostics.parquet")
    heat = pd.read_parquet(diag_heat_root / city / "home_to_service_access_diagnostics.parquet")
    base = base.loc[base["service_name"] == service, ["building_idx", "access_diagnosis_label"]]
    heat = heat.loc[heat["service_name"] == service, ["building_idx", "access_diagnosis_label"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=220)
    for ax, title, sub in [
        (axes[0], "baseline", base),
        (axes[1], "heat", heat),
    ]:
        _plot_boundary(ax, boundary)
        pts = buildings.merge(sub, on="building_idx", how="left")
        pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=buildings.crs)
        for label in PLOT_ORDER:
            cur = pts.loc[pts["access_diagnosis_label"] == label]
            if cur.empty:
                continue
            ax.scatter(cur.geometry.x, cur.geometry.y, s=5, c=LABEL_COLORS[label], alpha=0.76, linewidths=0, rasterized=True)
        ax.set_title(f"{SERVICE_RU[service]} — {CITY_RU.get(city, city)} — {title}", fontsize=13)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.12, wspace=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _pick_pt_row(pt_df: pd.DataFrame, building_idx: int, service: str) -> pd.Series | None:
    sub = pt_df.loc[(pt_df["building_idx"] == building_idx) & (pt_df["service_name"] == service)]
    if sub.empty:
        return None
    return sub.sort_values(["walk_filter_min", "pt_total_decomposed_time_min"], ascending=[False, True]).iloc[0]


def _effective_time(row: pd.Series) -> float:
    return float(min(float(row["walk_time_min"]), float(row["effective_pt_total_min"])))


def _choose_mode(diag_row: pd.Series) -> str:
    walk_t = float(diag_row["walk_time_min"])
    pt_t = float(diag_row["effective_pt_total_min"])
    if walk_t <= pt_t:
        return "walk"
    return "pt"


def _reconstruct_route(
    mode: str,
    walk_row: pd.Series | None,
    pt_row: pd.Series | None,
    walk_graph: nx.Graph,
    full_graph: nx.MultiDiGraph,
) -> tuple[list[int], int | None, int | None]:
    if mode == "walk" and walk_row is not None:
        source = int(walk_row["home_graph_node"])
        target = int(walk_row["service_graph_node"])
        try:
            return nx.shortest_path(walk_graph, source, target, weight="time_min"), source, target
        except Exception:
            return [source, target], source, target
    if pt_row is not None:
        source = int(pt_row["home_graph_node"])
        target = int(pt_row["nearest_service_graph_node"])
        try:
            return nx.shortest_path(full_graph, source, target, weight="time_min"), source, target
        except Exception:
            return [source, target], source, target
    return [], None, None


def render_city_service_routes(
    city: str,
    service: str,
    joint_inputs_root: Path,
    heat_joint_inputs_root: Path,
    baseline: ScenarioTables,
    heat: ScenarioTables,
    out_path: Path,
    top_n: int | None = None,
) -> None:
    city_root = joint_inputs_root / city
    boundary = gpd.read_parquet(city_root / "blocksnet" / "boundary.parquet")
    buildings = _living_points(city_root)
    service_points = gpd.read_parquet(city_root / "pipeline_2" / "services_raw" / f"{service}.parquet")
    service_points["geometry"] = service_points.geometry.representative_point()
    metric_crs = boundary.estimate_utm_crs()
    if metric_crs is not None:
        boundary = boundary.to_crs(metric_crs)
        buildings = buildings.to_crs(metric_crs)
        service_points = service_points.to_crs(metric_crs)

    base_diag = baseline.diag.loc[baseline.diag["service_name"] == service].copy()
    heat_diag = heat.diag.loc[heat.diag["service_name"] == service].copy()
    merged = base_diag.merge(
        heat_diag,
        on=["building_idx", "service_name"],
        suffixes=("_baseline", "_heat"),
    )
    if merged.empty:
        return
    merged["effective_baseline"] = merged.apply(lambda r: min(float(r["walk_time_min_baseline"]), float(r["effective_pt_total_min_baseline"])), axis=1)
    merged["effective_heat"] = merged.apply(lambda r: min(float(r["walk_time_min_heat"]), float(r["effective_pt_total_min_heat"])), axis=1)
    merged["delta_effective_min"] = merged["effective_heat"] - merged["effective_baseline"]
    merged["mode_baseline"] = merged.apply(
        lambda r: "walk" if float(r["walk_time_min_baseline"]) <= float(r["effective_pt_total_min_baseline"]) else "pt",
        axis=1,
    )
    merged["mode_heat"] = merged.apply(
        lambda r: "walk" if float(r["walk_time_min_heat"]) <= float(r["effective_pt_total_min_heat"]) else "pt",
        axis=1,
    )

    base_walk = baseline.walk.loc[baseline.walk["service_name"] == service]
    heat_walk = heat.walk.loc[heat.walk["service_name"] == service]

    base_graph_path = str(joint_inputs_root / city / "intermodal_graph_iduedu" / "graph.pkl")
    heat_graph_path = str(heat_joint_inputs_root / city / "intermodal_graph_iduedu" / "graph.pkl")
    base_nodes_path = str(joint_inputs_root / city / "intermodal_graph_iduedu" / "graph_nodes.parquet")
    heat_nodes_path = str(heat_joint_inputs_root / city / "intermodal_graph_iduedu" / "graph_nodes.parquet")
    base_graph = _load_graph(base_graph_path)
    heat_graph = _load_graph(heat_graph_path)
    base_nodes = _load_nodes(base_nodes_path)
    heat_nodes = _load_nodes(heat_nodes_path)
    base_walk_graph = _walk_subgraph(base_graph)
    heat_walk_graph = _walk_subgraph(heat_graph)
    base_walk_edges_gdf = _graph_edges_gdf(base_walk_graph, base_nodes).to_crs(boundary.crs)

    changed_rows = []
    base_route_gdfs = []
    heat_route_gdfs = []
    for _, row in merged.sort_values("delta_effective_min", ascending=False).iterrows():
        bidx = int(row["building_idx"])
        walk_row_base = base_walk.loc[base_walk["building_idx"] == bidx]
        walk_row_heat = heat_walk.loc[heat_walk["building_idx"] == bidx]
        walk_row_base = walk_row_base.iloc[0] if not walk_row_base.empty else None
        walk_row_heat = walk_row_heat.iloc[0] if not walk_row_heat.empty else None
        pt_row_base = _pick_pt_row(baseline.pt, bidx, service)
        pt_row_heat = _pick_pt_row(heat.pt, bidx, service)
        base_path, _, _ = _reconstruct_route(str(row["mode_baseline"]), walk_row_base, pt_row_base, base_walk_graph, base_graph)
        heat_path, _, _ = _reconstruct_route(str(row["mode_heat"]), walk_row_heat, pt_row_heat, heat_walk_graph, heat_graph)
        if not base_path or not heat_path:
            continue
        path_changed = tuple(base_path) != tuple(heat_path) or row["mode_baseline"] != row["mode_heat"] or row["access_diagnosis_label_baseline"] != row["access_diagnosis_label_heat"]
        if not path_changed:
            continue
        changed_rows.append(row)
        base_route_gdfs.append(_path_to_gdf(base_path, base_graph if row["mode_baseline"] == "pt" else base_walk_graph, base_nodes))
        heat_route_gdfs.append(_path_to_gdf(heat_path, heat_graph if row["mode_heat"] == "pt" else heat_walk_graph, heat_nodes))
        if top_n is not None and len(changed_rows) >= top_n:
            break

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=220)
    panels = [
        (axes[0], "baseline changed routes", base_route_gdfs, "#7f1734"),
        (axes[1], "heat changed routes", heat_route_gdfs, "#2f9e44"),
        (axes[2], "overlay", None, None),
    ]
    service_points = service_points.to_crs(boundary.crs)
    buildings = buildings.to_crs(boundary.crs)
    for ax, title, route_gdfs, color in panels:
        _plot_boundary(ax, boundary)
        if not base_walk_edges_gdf.empty:
            base_walk_edges_gdf.plot(ax=ax, color="#cbd5e1", linewidth=0.35, alpha=0.55)
        if route_gdfs is not None:
            for gdf in route_gdfs:
                if not gdf.empty:
                    gdf.to_crs(boundary.crs).plot(ax=ax, color=color, linewidth=1.6, alpha=0.78)
        else:
            for gdf in base_route_gdfs:
                if not gdf.empty:
                    gdf.to_crs(boundary.crs).plot(ax=ax, color="#7f1734", linewidth=1.4, alpha=0.65)
            for gdf in heat_route_gdfs:
                if not gdf.empty:
                    gdf.to_crs(boundary.crs).plot(ax=ax, color="#2f9e44", linewidth=1.6, alpha=0.78)
        service_points.plot(ax=ax, color="#111827", markersize=22, marker="*", alpha=0.9)
        ax.set_title(title, fontsize=12)
    changed_count = len(changed_rows)
    title = f"{CITY_RU.get(city, city)} — {SERVICE_RU[service]} — changed routes ({changed_count})"
    if changed_rows:
        max_delta = float(max(r["delta_effective_min"] for r in changed_rows))
        title += f" (max Δ={max_delta:.2f} min)"
    handles = [
        Line2D([0], [0], color="#7f1734", linewidth=2.0, label="baseline route"),
        Line2D([0], [0], color="#2f9e44", linewidth=2.0, label="heat-aware route"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#111827", markeredgecolor="#111827", markersize=10, label="service points"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.10, wspace=0.03)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_city_service_replaced_links(
    city: str,
    service: str,
    joint_inputs_root: Path,
    heat_joint_inputs_root: Path,
    baseline: ScenarioTables,
    heat: ScenarioTables,
    out_path: Path,
    top_n: int | None = None,
) -> None:
    city_root = joint_inputs_root / city
    boundary = gpd.read_parquet(city_root / "blocksnet" / "boundary.parquet")
    metric_crs = boundary.estimate_utm_crs()
    if metric_crs is not None:
        boundary = boundary.to_crs(metric_crs)

    base_diag = baseline.diag.loc[baseline.diag["service_name"] == service].copy()
    heat_diag = heat.diag.loc[heat.diag["service_name"] == service].copy()
    merged = base_diag.merge(
        heat_diag,
        on=["building_idx", "service_name"],
        suffixes=("_baseline", "_heat"),
    )
    if merged.empty:
        return
    merged["effective_baseline"] = merged.apply(lambda r: min(float(r["walk_time_min_baseline"]), float(r["effective_pt_total_min_baseline"])), axis=1)
    merged["effective_heat"] = merged.apply(lambda r: min(float(r["walk_time_min_heat"]), float(r["effective_pt_total_min_heat"])), axis=1)
    merged["delta_effective_min"] = merged["effective_heat"] - merged["effective_baseline"]
    merged["mode_baseline"] = merged.apply(
        lambda r: "walk" if float(r["walk_time_min_baseline"]) <= float(r["effective_pt_total_min_baseline"]) else "pt",
        axis=1,
    )
    merged["mode_heat"] = merged.apply(
        lambda r: "walk" if float(r["walk_time_min_heat"]) <= float(r["effective_pt_total_min_heat"]) else "pt",
        axis=1,
    )

    base_walk = baseline.walk.loc[baseline.walk["service_name"] == service]
    heat_walk = heat.walk.loc[heat.walk["service_name"] == service]

    base_graph_path = str(joint_inputs_root / city / "intermodal_graph_iduedu" / "graph.pkl")
    heat_graph_path = str(heat_joint_inputs_root / city / "intermodal_graph_iduedu" / "graph.pkl")
    base_nodes_path = str(joint_inputs_root / city / "intermodal_graph_iduedu" / "graph_nodes.parquet")
    heat_nodes_path = str(heat_joint_inputs_root / city / "intermodal_graph_iduedu" / "graph_nodes.parquet")
    base_graph = _load_graph(base_graph_path)
    heat_graph = _load_graph(heat_graph_path)
    base_nodes = _load_nodes(base_nodes_path)
    heat_nodes = _load_nodes(heat_nodes_path)
    base_walk_graph = _walk_subgraph(base_graph)

    replaced_base_edges: set[tuple[int, int]] = set()
    replaced_heat_edges: set[tuple[int, int]] = set()
    changed_count = 0
    for _, row in merged.sort_values("delta_effective_min", ascending=False).iterrows():
        bidx = int(row["building_idx"])
        walk_row_base = base_walk.loc[base_walk["building_idx"] == bidx]
        walk_row_heat = heat_walk.loc[heat_walk["building_idx"] == bidx]
        walk_row_base = walk_row_base.iloc[0] if not walk_row_base.empty else None
        walk_row_heat = walk_row_heat.iloc[0] if not walk_row_heat.empty else None
        pt_row_base = _pick_pt_row(baseline.pt, bidx, service)
        pt_row_heat = _pick_pt_row(heat.pt, bidx, service)
        base_path, _, _ = _reconstruct_route(str(row["mode_baseline"]), walk_row_base, pt_row_base, base_walk_graph, base_graph)
        heat_path, _, _ = _reconstruct_route(str(row["mode_heat"]), walk_row_heat, pt_row_heat, _walk_subgraph(heat_graph), heat_graph)
        if not base_path or not heat_path:
            continue
        base_keys = set(_path_edge_keys(base_path))
        heat_keys = set(_path_edge_keys(heat_path))
        if not (base_keys ^ heat_keys):
            continue
        replaced_base_edges.update(base_keys - heat_keys)
        replaced_heat_edges.update(heat_keys - base_keys)
        changed_count += 1
        if top_n is not None and changed_count >= top_n:
            break

    base_diff_gdf = _edge_keys_to_gdf(replaced_base_edges, base_graph, base_nodes).to_crs(boundary.crs)
    heat_diff_gdf = _edge_keys_to_gdf(replaced_heat_edges, heat_graph, heat_nodes).to_crs(boundary.crs)
    base_walk_edges_gdf = _graph_edges_gdf(base_walk_graph, base_nodes).to_crs(boundary.crs)

    fig, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=260)
    _plot_boundary(ax, boundary)
    if not base_walk_edges_gdf.empty:
        base_walk_edges_gdf.plot(ax=ax, color="#d1d5db", linewidth=0.45, alpha=0.45)
    if not base_diff_gdf.empty:
        base_diff_gdf.plot(ax=ax, color="#7f1734", linewidth=2.4, alpha=0.95)
    if not heat_diff_gdf.empty:
        heat_diff_gdf.plot(ax=ax, color="#2f9e44", linewidth=2.4, alpha=0.95)
    ax.set_title(f"{CITY_RU.get(city, city)} — {SERVICE_RU[service]} — заменённые сегменты", fontsize=14)
    handles = [
        Line2D([0], [0], color="#7f1734", lw=3, label="было в baseline, исчезло в heat"),
        Line2D([0], [0], color="#2f9e44", lw=3, label="появилось в heat"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-inputs-root", type=Path, required=True)
    parser.add_argument("--heat-joint-inputs-root", type=Path, required=True)
    parser.add_argument("--baseline-walk-root", type=Path, required=True)
    parser.add_argument("--baseline-pt-lt-root", type=Path, required=True)
    parser.add_argument("--baseline-pt-ge-root", type=Path, required=True)
    parser.add_argument("--baseline-diag-root", type=Path, required=True)
    parser.add_argument("--heat-walk-root", type=Path, required=True)
    parser.add_argument("--heat-pt-lt-root", type=Path, required=True)
    parser.add_argument("--heat-pt-ge-root", type=Path, required=True)
    parser.add_argument("--heat-diag-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--services", nargs="*", default=SERVICES)
    parser.add_argument("--top-n-routes", type=int, default=None)
    args = parser.parse_args()

    city_map = {p.name: p for p in args.joint_inputs_root.iterdir() if p.is_dir() and not p.name.startswith("_")}
    wanted = args.cities or CITY_ORDER
    city_dirs = [city_map[name] for name in wanted if name in city_map]

    for city_dir in city_dirs:
        city = city_dir.name
        baseline = _load_scenario(city, args.baseline_walk_root, args.baseline_pt_lt_root, args.baseline_pt_ge_root, args.baseline_diag_root)
        heat = _load_scenario(city, args.heat_walk_root, args.heat_pt_lt_root, args.heat_pt_ge_root, args.heat_diag_root)
        for service in args.services:
            render_city_service_pair(
                city=city,
                service=service,
                joint_inputs_root=args.joint_inputs_root,
                diag_base_root=args.baseline_diag_root,
                diag_heat_root=args.heat_diag_root,
                out_path=args.out_root / "city_service_pairs" / city / f"{service}_baseline_vs_heat.png",
            )
            render_city_service_routes(
                city=city,
                service=service,
                joint_inputs_root=args.joint_inputs_root,
                heat_joint_inputs_root=args.heat_joint_inputs_root,
                baseline=baseline,
                heat=heat,
                out_path=args.out_root / "city_service_routes" / city / f"{service}_changed_routes.png",
                top_n=args.top_n_routes,
            )
            render_city_service_replaced_links(
                city=city,
                service=service,
                joint_inputs_root=args.joint_inputs_root,
                heat_joint_inputs_root=args.heat_joint_inputs_root,
                baseline=baseline,
                heat=heat,
                out_path=args.out_root / "city_service_route_diffs" / city / f"{service}_replaced_links_only.png",
                top_n=args.top_n_routes,
            )
            print(f"{city} {service}: ok")


def render_png_gallery(image_paths: list[Path], out_path: Path, title: str, cols: int = 4) -> None:
    if not image_paths:
        return
    opened = [Image.open(p).convert("RGB") for p in image_paths]
    thumb_w = 520
    thumb_h = 320
    pad = 24
    title_h = 56
    rows = math.ceil(len(opened) / cols)
    width = cols * thumb_w + (cols + 1) * pad
    height = title_h + rows * thumb_h + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 18), title, fill="#0f172a", font=font)
    for idx, (img, path) in enumerate(zip(opened, image_paths, strict=False)):
        r = idx // cols
        c = idx % cols
        x = pad + c * thumb_w
        y = title_h + pad + r * thumb_h
        thumb = img.copy()
        thumb.thumbnail((thumb_w - pad, thumb_h - 36))
        canvas.paste(thumb, (x, y + 20))
        draw.text((x + 4, y), CITY_RU.get(path.parent.name, path.parent.name), fill="#111827", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


if __name__ == "__main__":
    main()

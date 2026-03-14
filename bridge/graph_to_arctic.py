"""
Convert OSMnx/iduedu graphs to arctic format (transport_df, G for make_g / graph_to_city_model).

Blocks (кварталы) as nodes, drive + transit modalities, output in minutes.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

try:
    from iduedu import get_adj_matrix_gdf_to_gdf
except ImportError:
    get_adj_matrix_gdf_to_gdf = None

from .adapters import graph_to_bridge_format

DEFAULT_DRIVE_SPEED_KMH = 30.0
DEFAULT_TRANSIT_SPEED_KMH = 20.0
DEFAULT_WALK_SPEED_KMH = 5.0

ARCTIC_TRANSPORT_MODES = ["Aviation", "Regular road", "Winter road", "Water transport"]


def _ensure_blocks_have_name(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = blocks_gdf.copy()
    if "name" not in blocks.columns:
        blocks["name"] = [f"block_{i}" for i in range(len(blocks))]
    return blocks


def _gdf_from_blocks(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = _ensure_blocks_have_name(blocks_gdf)
    blocks.geometry = blocks.geometry.centroid
    return blocks


def _pts_for_iduedu(pts: gpd.GeoDataFrame, graph_crs_epsg: Optional[int] = 4326) -> gpd.GeoDataFrame:
    pts = pts.copy()
    valid = pts.geometry.notna() & ~pts.geometry.is_empty
    pts = pts[valid].copy()
    if pts.empty:
        return pts
    epsg = graph_crs_epsg if isinstance(graph_crs_epsg, int) else 4326
    target_crs = f"EPSG:{epsg}"
    if pts.crs and str(pts.crs) != target_crs:
        pts = pts.to_crs(target_crs)
    return pts


def _infer_weight_attr(G: nx.Graph) -> str:
    if not G.edges():
        return "weight"
    sample = next(iter(G.edges(data=True)))
    attrs = sample[2] or {}
    if "time_min" in attrs:
        return "time_min"
    if "length_m" in attrs or "length" in attrs:
        return "length_m" if "length_m" in attrs else "length"
    return "weight"


def _length_to_time_min(adj_matrix: pd.DataFrame, speed_kmh: float) -> pd.DataFrame:
    factor = 60.0 / (speed_kmh * 1000.0)
    return adj_matrix * factor


def _get_nearest_nodes(pts: gpd.GeoDataFrame, G: nx.Graph, crs) -> Dict:
    """Block index -> nearest OSM node."""
    pts_work = pts.to_crs(crs) if crs and str(pts.crs) != str(crs) else pts
    try:
        import osmnx as ox
        nearest = ox.distance.nearest_nodes(
            G, pts_work.geometry.x.values, pts_work.geometry.y.values
        )
        return dict(zip(pts_work.index, nearest))
    except Exception:
        # Fallback: find nearest by euclidean distance (nodes need x,y)
        name_to_node = {}
        for idx, row in pts_work.iterrows():
            x, y = row.geometry.x, row.geometry.y
            best, best_d = None, float("inf")
            for n, d in G.nodes(data=True):
                nx_ = d.get("x", d.get("lon"))
                ny_ = d.get("y", d.get("lat"))
                if nx_ is not None and ny_ is not None:
                    dist = (float(nx_) - x) ** 2 + (float(ny_) - y) ** 2
                    if dist < best_d:
                        best_d, best = dist, n
            name_to_node[idx] = best if best is not None else idx
        return name_to_node


def blocks_adjacency_from_graph(
    blocks_gdf: gpd.GeoDataFrame,
    G: nx.Graph,
    weight_attr: Optional[str] = None,
    speed_kmh: Optional[float] = None,
    use_iduedu: bool = True,
    direct_edges_only: bool = False,
) -> pd.DataFrame:
    blocks = _ensure_blocks_have_name(blocks_gdf)
    pts = _gdf_from_blocks(blocks)

    if weight_attr is None:
        weight_attr = _infer_weight_attr(G)
    is_time = weight_attr in ("time_min", "travel_time")
    if speed_kmh is None:
        speed_kmh = DEFAULT_DRIVE_SPEED_KMH

    crs = G.graph.get("crs") if hasattr(G, "graph") else None
    graph_epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else (crs or 4326)

    if direct_edges_only:
        # Только прямые рёбра: A-C добавляем только если путь не идёт через другое поселение
        pts_work = pts.to_crs(f"EPSG:{graph_epsg}") if pts.crs and str(pts.crs) != f"EPSG:{graph_epsg}" else pts
        pts_work = pts_work[pts_work.geometry.notna() & ~pts_work.geometry.is_empty]
        if pts_work.empty:
            adj = pd.DataFrame(np.full((len(pts), len(pts)), np.inf), index=pts.index, columns=pts.index)
            np.fill_diagonal(adj.values, 0)
        else:
            name_to_node = _get_nearest_nodes(pts_work, G, f"EPSG:{graph_epsg}")
            node_to_names = {}
            for idx, n in name_to_node.items():
                node_to_names.setdefault(n, []).append(idx)
            n = len(pts.index)
            adj = pd.DataFrame(np.full((n, n), np.inf), index=pts.index, columns=pts.index)
            np.fill_diagonal(adj.values, 0)
            for src in pts.index:
                u = name_to_node.get(src)
                if u is None or u not in G:
                    continue
                try:
                    lengths, paths = nx.single_source_dijkstra(G, u, weight=weight_attr)
                    for tgt in pts.index:
                        if src == tgt:
                            continue
                        v = name_to_node.get(tgt)
                        if v is None or v not in lengths or v not in paths:
                            continue
                        path_nodes = set(paths[v])
                        other_settlement_osm_nodes = {
                            name_to_node[k] for k in pts.index
                            if k not in (src, tgt) and k in name_to_node
                        }
                        if path_nodes & other_settlement_osm_nodes:
                            continue
                        adj.loc[src, tgt] = lengths[v]
                except (nx.NetworkXError, KeyError):
                    pass
    elif use_iduedu and get_adj_matrix_gdf_to_gdf is not None:
        graph_crs = G.graph.get("crs")
        epsg = graph_crs.to_epsg() if hasattr(graph_crs, "to_epsg") else (graph_crs or 4326)
        pts_work = _pts_for_iduedu(pts, graph_crs_epsg=epsg)
        if pts_work.empty:
            adj = pd.DataFrame(np.full((len(pts), len(pts)), np.inf), index=pts.index, columns=pts.index)
            np.fill_diagonal(adj.values, 0)
        else:
            adj = get_adj_matrix_gdf_to_gdf(pts_work, pts_work, G, weight=weight_attr, dtype=np.float64)
            if set(adj.index) != set(pts.index):
                adj = adj.reindex(index=pts.index, columns=pts.index, fill_value=np.inf)
                np.fill_diagonal(adj.values, 0)
    else:
        if use_iduedu and get_adj_matrix_gdf_to_gdf is None:
            raise ImportError("iduedu is required when use_iduedu=True")
        import osmnx as ox
        crs = G.graph.get("crs") if hasattr(G, "graph") else None
        pts_work = pts.to_crs(crs) if crs and pts.crs != crs else pts
        nearest = ox.distance.nearest_nodes(G, pts_work.geometry.x.values, pts_work.geometry.y.values)
        name_to_node = dict(zip(pts.index, nearest))
        n = len(pts.index)
        adj = pd.DataFrame(np.full((n, n), np.inf), index=pts.index, columns=pts.index)
        np.fill_diagonal(adj.values, 0)
        for i, src in enumerate(pts.index):
            u = name_to_node[src]
            try:
                lengths = nx.single_source_dijkstra_path_length(G, u, weight=weight_attr)
                for j, tgt in enumerate(pts.index):
                    if i == j:
                        continue
                    v = name_to_node[tgt]
                    if v in lengths:
                        adj.iloc[i, j] = lengths[v]
            except (nx.NetworkXError, KeyError):
                pass

    if not is_time and weight_attr in ("length", "length_m", "weight"):
        adj = _length_to_time_min(adj, speed_kmh)
    return adj


def adjacency_to_transport_df(
    adj_matrix: pd.DataFrame,
    modalities: Optional[List[str]] = None,
    index_to_name: Optional[Dict] = None,
    arctic_modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    if modalities is None:
        modalities = ["drive", "transit"]
    modality = modalities[0]
    idx2name = index_to_name or {}
    rows = []
    for i in adj_matrix.index:
        for j in adj_matrix.columns:
            if i == j:
                continue
            val = adj_matrix.loc[i, j]
            if np.isfinite(val) and val > 0:
                e1 = idx2name.get(i, i)
                e2 = idx2name.get(j, j)
                row = {"edge1": e1, "edge2": e2, modality: round(float(val), 2)}
                for m in modalities[1:]:
                    row[m] = 0.0
                rows.append(row)
    if not rows:
        out_cols = ["edge1", "edge2"] + (arctic_modes or modalities)
        return pd.DataFrame(columns=out_cols)
    df = pd.DataFrame(rows)
    for m in modalities[1:]:
        if m not in df.columns or df[m].eq(0).all():
            df[m] = df[modality]
    if arctic_modes:
        for m in arctic_modes:
            if m not in df.columns:
                df[m] = 0.0
    return df


def graph_to_arctic_format(
    blocks_gdf: gpd.GeoDataFrame,
    G_drive: nx.Graph,
    G_transit: Optional[nx.Graph] = None,
    service_name: str = "hospital",
    weight_attr_drive: Optional[str] = None,
    weight_attr_transit: Optional[str] = None,
    speed_drive_kmh: float = DEFAULT_DRIVE_SPEED_KMH,
    speed_transit_kmh: float = DEFAULT_TRANSIT_SPEED_KMH,
    modalities: Optional[List[str]] = None,
    mode_mapping: Optional[Dict[str, str]] = None,
    use_iduedu: bool = True,
    arctic_compatible: bool = True,
    direct_edges_only: bool = False,
) -> tuple[pd.DataFrame, nx.Graph]:
    if arctic_compatible:
        modalities = ["Regular road"]
        G_transit = None
    elif modalities is None:
        modalities = ["drive", "transit"] if G_transit is not None else ["drive"]

    blocks = _ensure_blocks_have_name(blocks_gdf)
    capacity_col = f"capacity_{service_name}"
    if capacity_col not in blocks.columns:
        blocks[capacity_col] = 0
    if "population" not in blocks.columns:
        blocks["population"] = 0

    G_drive = graph_to_bridge_format(
        G_drive, source="iduedu" if use_iduedu else "osmnx", speed_kmh=speed_drive_kmh
    )

    adj_drive = blocks_adjacency_from_graph(
        blocks, G_drive,
        weight_attr=weight_attr_drive,
        speed_kmh=speed_drive_kmh,
        use_iduedu=use_iduedu,
        direct_edges_only=direct_edges_only,
    )

    index_to_name = blocks["name"].to_dict()

    if arctic_compatible:
        transport_df = adjacency_to_transport_df(
            adj_drive,
            modalities=modalities,
            index_to_name=index_to_name,
            arctic_modes=ARCTIC_TRANSPORT_MODES,
        )
    elif G_transit is not None and "transit" in modalities:
        adj_transit = blocks_adjacency_from_graph(
            blocks, G_transit,
            weight_attr=weight_attr_transit,
            speed_kmh=speed_transit_kmh,
            use_iduedu=use_iduedu,
        )
        transport_rows = []
        for i in adj_drive.index:
            for j in adj_drive.columns:
                if i == j:
                    continue
                d_d = adj_drive.loc[i, j]
                d_t = adj_transit.loc[i, j] if i in adj_transit.index and j in adj_transit.columns else np.inf
                if np.isfinite(d_d) or np.isfinite(d_t):
                    transport_rows.append({
                        "edge1": index_to_name.get(i, i),
                        "edge2": index_to_name.get(j, j),
                        "drive": round(float(d_d), 2) if np.isfinite(d_d) else 0.0,
                        "transit": round(float(d_t), 2) if np.isfinite(d_t) else 0.0,
                    })
        transport_df = pd.DataFrame(transport_rows)
        if transport_rows:
            transport_df = transport_df[(transport_df["drive"] > 0) | (transport_df["transit"] > 0)]
    else:
        transport_df = adjacency_to_transport_df(
            adj_drive, modalities=modalities, index_to_name=index_to_name
        )

    if mode_mapping and not arctic_compatible:
        transport_df = transport_df.rename(columns={k: v for k, v in mode_mapping.items() if k in transport_df.columns})

    G_arctic = nx.Graph()
    G_arctic.graph["crs"] = blocks.crs.to_epsg() if blocks.crs else None

    skip_cols = {"geometry", "name", "id", "population"}
    for _, row in blocks.iterrows():
        name = row["name"]
        G_arctic.add_node(
            name,
            id=name,
            name=name,
            geometry=row.get("geometry"),
            population=row.get("population", 0),
            **{k: v for k, v in row.items() if k not in skip_cols},
        )

    modality_cols = [c for c in transport_df.columns if c not in ("edge1", "edge2")]
    for _, row in transport_df.iterrows():
        w = min(row.get(m, np.inf) for m in modality_cols if m in row and row[m] > 0)
        if np.isfinite(w) and w > 0:
            G_arctic.add_edge(row["edge1"], row["edge2"], weight=w)

    return transport_df, G_arctic


def settl_from_blocks(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    blocks = _ensure_blocks_have_name(blocks_gdf)
    union = blocks.geometry.union_all()
    return gpd.GeoDataFrame(
        {"name": ["_bridge"]},
        geometry=[union],
        crs=blocks.crs,
    )

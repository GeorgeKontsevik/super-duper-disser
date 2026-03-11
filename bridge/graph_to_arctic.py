"""
Convert OSMnx/iduedu graphs to arctic format (transport_df, G for make_g / graph_to_city_model).

Blocks (кварталы) as nodes, drive + transit modalities, output in minutes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx

try:
    from iduedu import get_adj_matrix_gdf_to_gdf
except ImportError:
    get_adj_matrix_gdf_to_gdf = None

DEFAULT_DRIVE_SPEED_KMH = 30.0
DEFAULT_TRANSIT_SPEED_KMH = 20.0
DEFAULT_WALK_SPEED_KMH = 5.0


def _ensure_blocks_have_name(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure blocks_gdf has 'name' column for arctic compatibility."""
    blocks = blocks_gdf.copy()
    if "name" not in blocks.columns:
        blocks["name"] = [f"block_{i}" for i in range(len(blocks))]
    return blocks


def _gdf_from_blocks(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create points GeoDataFrame from block centroids for iduedu."""
    blocks = _ensure_blocks_have_name(blocks_gdf)
    centroids = blocks.geometry.centroid
    pts = gpd.GeoDataFrame(
        geometry=centroids,
        index=blocks["name"].values,
        crs=blocks.crs,
    )
    return pts


def _infer_weight_attr(G: nx.Graph) -> str:
    """Infer which weight attribute to use (time_min preferred, else length)."""
    if not G.edges():
        return "weight"
    sample = next(iter(G.edges(data=True)))
    attrs = sample[2] or {}
    if "time_min" in attrs:
        return "time_min"
    if "length_m" in attrs or "length" in attrs:
        return "length_m" if "length_m" in attrs else "length"
    return "weight"


def _length_to_time_min(
    adj_matrix: pd.DataFrame,
    speed_kmh: float,
) -> pd.DataFrame:
    """Convert length (meters) matrix to time (minutes)."""
    # adj_matrix values in meters -> time_min = m / (speed_kmh * 1000/60)
    factor = 60.0 / (speed_kmh * 1000.0)  # min per meter
    return adj_matrix * factor


def blocks_adjacency_from_graph(
    blocks_gdf: gpd.GeoDataFrame,
    G: nx.Graph,
    weight_attr: Optional[str] = None,
    speed_kmh: Optional[float] = None,
    use_iduedu: bool = True,
) -> pd.DataFrame:
    """
    Compute block-to-block adjacency matrix (shortest path) via graph.

    Args:
        blocks_gdf: GeoDataFrame of blocks (polygons). Will get 'name' if missing.
        G: NetworkX graph from OSMnx or iduedu with edge weights.
        weight_attr: Edge attribute for weight ('time_min', 'length_m', 'weight').
            Auto-inferred if None.
        speed_kmh: If weight is length (m), convert to time using this speed.
        use_iduedu: Use iduedu.get_adj_matrix_gdf_to_gdf if available (faster).

    Returns:
        DataFrame (index=block names, columns=block names) with travel time in minutes.
    """
    blocks = _ensure_blocks_have_name(blocks_gdf)
    pts = _gdf_from_blocks(blocks)

    if weight_attr is None:
        weight_attr = _infer_weight_attr(G)

    is_time = weight_attr in ("time_min", "travel_time")
    if speed_kmh is None:
        speed_kmh = DEFAULT_DRIVE_SPEED_KMH

    if use_iduedu and get_adj_matrix_gdf_to_gdf is not None:
        adj = get_adj_matrix_gdf_to_gdf(
            pts, pts, G, weight=weight_attr, dtype=np.float64
        )
    else:
        if get_adj_matrix_gdf_to_gdf is None:
            raise ImportError("iduedu is required for blocks_adjacency_from_graph")
        import osmnx as ox
        crs = G.graph.get("crs") if hasattr(G, "graph") else None
        pts_work = pts.to_crs(crs) if crs and pts.crs != crs else pts
        nearest = ox.distance.nearest_nodes(
            G, pts_work.geometry.x.values, pts_work.geometry.y.values
        )
        name_to_node = dict(zip(pts.index, nearest))
        n = len(pts.index)
        adj = pd.DataFrame(
            np.full((n, n), np.inf),
            index=pts.index,
            columns=pts.index,
        )
        np.fill_diagonal(adj.values, 0)
        for i, src in enumerate(pts.index):
            u = name_to_node[src]
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G, u, weight=weight_attr
                )
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
) -> pd.DataFrame:
    """
    Convert adjacency matrix to transport_df format for make_g.

    Args:
        adj_matrix: Block-to-block travel time (minutes). Index/columns = block names.
        modalities: Column names for modalities. Default: ["drive", "transit"].

    Returns:
        DataFrame with edge1, edge2, {modality} columns. Values in minutes.
    """
    if modalities is None:
        modalities = ["drive", "transit"]

    # Use first modality for all edges (single-season: one weight per edge)
    modality = modalities[0]

    rows = []
    for i in adj_matrix.index:
        for j in adj_matrix.columns:
            if i == j:
                continue
            val = adj_matrix.loc[i, j]
            if np.isfinite(val) and val > 0:
                row = {"edge1": i, "edge2": j, modality: round(float(val), 2)}
                for m in modalities[1:]:
                    row[m] = 0.0
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["edge1", "edge2"] + modalities)

    df = pd.DataFrame(rows)
    # Fill other modalities with same value if only one modality used
    for m in modalities[1:]:
        if m not in df.columns or df[m].eq(0).all():
            df[m] = df[modality]
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
    use_iduedu: bool = True,
) -> tuple[pd.DataFrame, nx.Graph]:
    """
    Convert drive (and optionally transit) graph to arctic format.

    Returns transport_df and G_arctic for make_g / graph_to_city_model.
    All weights in minutes.

    Args:
        blocks_gdf: Blocks (кварталы) with geometry. Needs population, capacity_{service}.
        G_drive: Drive network graph.
        G_transit: Optional transit network graph. If None, only drive.
        service_name: Service type for capacity column.
        weight_attr_drive, weight_attr_transit: Edge weight attributes.
        speed_drive_kmh, speed_transit_kmh: For length→time conversion.
        modalities: Output modality names. Default: ["drive", "transit"].
        use_iduedu: Use iduedu for adjacency computation.

    Returns:
        (transport_df, G_arctic)
        - transport_df: edge1, edge2, drive, transit (minutes)
        - G_arctic: nx.Graph with nodes=blocks, edges from transport_df
    """
    if modalities is None:
        modalities = ["drive", "transit"] if G_transit is not None else ["drive"]

    blocks = _ensure_blocks_have_name(blocks_gdf)
    capacity_col = f"capacity_{service_name}"
    if capacity_col not in blocks.columns:
        blocks[capacity_col] = 0
    if "population" not in blocks.columns:
        blocks["population"] = 0

    adj_drive = blocks_adjacency_from_graph(
        blocks, G_drive,
        weight_attr=weight_attr_drive,
        speed_kmh=speed_drive_kmh,
        use_iduedu=use_iduedu,
    )

    if G_transit is not None and "transit" in modalities:
        adj_transit = blocks_adjacency_from_graph(
            blocks, G_transit,
            weight_attr=weight_attr_transit,
            speed_kmh=speed_transit_kmh,
            use_iduedu=use_iduedu,
        )
        # Merge: for each edge take min(drive, transit) per modality
        transport_rows = []
        for i in adj_drive.index:
            for j in adj_drive.columns:
                if i == j:
                    continue
                d_d = adj_drive.loc[i, j]
                d_t = adj_transit.loc[i, j] if i in adj_transit.index and j in adj_transit.columns else np.inf
                if np.isfinite(d_d) or np.isfinite(d_t):
                    row = {
                        "edge1": i, "edge2": j,
                        "drive": round(float(d_d), 2) if np.isfinite(d_d) else 0.0,
                        "transit": round(float(d_t), 2) if np.isfinite(d_t) else 0.0,
                    }
                    transport_rows.append(row)
        transport_df = pd.DataFrame(transport_rows)
        if transport_rows:
            transport_df = transport_df[(transport_df["drive"] > 0) | (transport_df["transit"] > 0)]
    else:
        transport_df = adjacency_to_transport_df(adj_drive, modalities=modalities)

    # Build G_arctic: simple graph with one weight per edge (min across modalities)
    G_arctic = nx.Graph()
    G_arctic.graph["crs"] = blocks.crs.to_epsg() if blocks.crs else None

    for _, row in blocks.iterrows():
        name = row["name"]
        G_arctic.add_node(
            name,
            id=name,
            name=name,
            geometry=row.get("geometry"),
            population=row.get("population", 0),
            **{k: v for k, v in row.items() if k not in ("geometry", "name")},
        )

    for _, row in transport_df.iterrows():
        w = min(
            row.get(m, np.inf) for m in modalities
            if m in row and row[m] > 0
        )
        if np.isfinite(w) and w > 0:
            G_arctic.add_edge(row["edge1"], row["edge2"], weight=w)

    return transport_df, G_arctic


def settl_from_blocks(blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create minimal settl GeoDataFrame from blocks for make_g.

    make_g expects settl for G.graph['crs']. This provides a compatible object.
    """
    blocks = _ensure_blocks_have_name(blocks_gdf)
    union = blocks.geometry.unary_union
    settl = gpd.GeoDataFrame(
        {"name": ["_bridge"]},
        geometry=[union],
        crs=blocks.crs,
    )
    return settl

"""
Adapters: bring OSMnx/iduedu graphs to unified format (time_min in minutes).

Does not modify existing libraries — adapts input to what bridge expects.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx

DEFAULT_DRIVE_SPEED_KMH = 30.0
DEFAULT_TRANSIT_SPEED_KMH = 20.0


def osmnx_graph_add_time_min(
    G: nx.Graph,
    speed_kmh: float = DEFAULT_DRIVE_SPEED_KMH,
    length_attr: str = "length",
) -> nx.Graph:
    """
    Add time_min (minutes) to OSMnx graph edges.

    OSMnx typically has 'length' in meters. Does not modify original.
    """
    G = G.copy()
    factor = 60.0 / (speed_kmh * 1000.0)
    for u, v, data in G.edges(data=True):
        length = data.get(length_attr, data.get("weight", 0))
        if length and length > 0:
            G[u][v]["time_min"] = length * factor
    return G


def ensure_graph_has_time_min(
    G: nx.Graph,
    weight_attr: Optional[str] = None,
    speed_kmh: float = DEFAULT_DRIVE_SPEED_KMH,
) -> nx.Graph:
    """
    Ensure graph has time_min on edges. Add from length if missing.

    Use before passing to blocks_adjacency_from_graph when graph has only length.
    """
    sample = next(iter(G.edges(data=True)), None)
    if not sample:
        return G
    attrs = sample[2] or {}
    if "time_min" in attrs:
        return G
    length_attr = weight_attr or ("length_m" if "length_m" in attrs else "length")
    return osmnx_graph_add_time_min(G, speed_kmh=speed_kmh, length_attr=length_attr)

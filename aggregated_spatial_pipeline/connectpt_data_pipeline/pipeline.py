from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import momepy as mp
import networkx as nx
import osmnx as ox

CONNECTPT_SRC = Path(__file__).resolve().parents[2] / "connectpt" / "connectpt"
if str(CONNECTPT_SRC) not in sys.path:
    sys.path.insert(0, str(CONNECTPT_SRC))

from preprocess.lines import get_lines
from preprocess.network import (
    build_time_matrix,
    roads_to_graph,
    stop_complete_then_prune,
)
from preprocess.projection import project_stops_on_roads
from preprocess.stops import get_agg_stops
from preprocess.types import Modality


DEFAULT_SPEED_KMH = 20.0


@dataclass(frozen=True)
class ModalityArtifacts:
    modality: str
    raw_stop_count: int
    projected_stop_count: int
    graph_node_count: int
    graph_edge_count: int
    files: dict[str, str]


def slugify_place(place: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")
    return slug or "place"


def parse_modalities(values: list[str]) -> list[Modality]:
    normalized = {value.strip().lower() for value in values if value.strip()}
    if not normalized:
        raise ValueError("At least one modality must be provided.")
    allowed = {modality.value: modality for modality in Modality}
    unknown = sorted(normalized - allowed.keys())
    if unknown:
        raise ValueError(f"Unsupported modalities: {', '.join(unknown)}")
    return [allowed[value] for value in sorted(normalized)]


def geocode_place_boundary(place: str) -> tuple[gpd.GeoDataFrame, object]:
    place_gdf = ox.geocode_to_gdf(place)
    if place_gdf.empty:
        raise ValueError(f"Could not geocode place: {place}")
    boundary = place_gdf.geometry.iloc[0]
    return place_gdf[["display_name", "geometry"]].copy(), boundary


def _ensure_length_meter(lines_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = lines_gdf.copy()
    if "length_meter" not in result.columns:
        result["length_meter"] = result.geometry.length
    else:
        missing = result["length_meter"].isna()
        result.loc[missing, "length_meter"] = result.loc[missing, "geometry"].length
    return result


def _largest_connected_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        raise ValueError("Simplified graph is empty.")
    if graph.number_of_edges() == 0:
        return graph.copy()
    largest_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_nodes).copy()


def _save_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def _save_time_matrix(matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix.tolist())


def _prepare_graph_outputs(graph: nx.Graph, modality: Modality) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes_gdf, edges_gdf = mp.nx_to_gdf(graph)
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()
    nodes["modality"] = modality.value
    edges["modality"] = modality.value
    if "geometry" in nodes.columns:
        nodes[["x_coord", "y_coord"]] = nodes.geometry.apply(lambda p: [round(p.x, 2), round(p.y, 2)]).tolist()
    return nodes, edges


def build_connectpt_osm_bundle(
    place: str,
    modalities: list[Modality],
    output_dir: str | Path,
    speed_kmh: float = DEFAULT_SPEED_KMH,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    place_gdf, boundary = geocode_place_boundary(place)
    boundary_path = output_path / "boundary.geojson"
    _save_geojson(place_gdf, boundary_path)

    centre_gdf = gpd.GeoDataFrame(
        [{"place": place, "geometry": boundary.centroid}],
        crs=place_gdf.crs,
    )
    centre_path = output_path / "centre.geojson"
    _save_geojson(centre_gdf, centre_path)

    stops_by_modality = get_agg_stops(boundary, modalities)
    lines_by_modality = get_lines(boundary, modalities)

    modality_artifacts: list[ModalityArtifacts] = []

    for modality in modalities:
        modality_dir = output_path / modality.value
        modality_dir.mkdir(parents=True, exist_ok=True)

        if modality not in stops_by_modality or modality not in lines_by_modality:
            modality_artifacts.append(
                ModalityArtifacts(
                    modality=modality.value,
                    raw_stop_count=0,
                    projected_stop_count=0,
                    graph_node_count=0,
                    graph_edge_count=0,
                    files={},
                )
            )
            continue

        agg_stops = stops_by_modality[modality].copy()
        lines = lines_by_modality[modality].copy()

        agg_stops_path = modality_dir / "aggregated_stops.geojson"
        lines_path = modality_dir / "lines.geojson"
        _save_geojson(agg_stops, agg_stops_path)
        _save_geojson(lines, lines_path)

        roads_with_stops, filtered_stops = project_stops_on_roads(lines, agg_stops)
        roads_with_stops = _ensure_length_meter(roads_with_stops)

        projected_lines_path = modality_dir / "projected_lines.geojson"
        projected_stops_path = modality_dir / "projected_stops.geojson"
        _save_geojson(roads_with_stops, projected_lines_path)
        _save_geojson(filtered_stops, projected_stops_path)

        roads_graph = roads_to_graph(roads_with_stops, filtered_stops)
        simplified_graph = stop_complete_then_prune(roads_graph, speed_kmh=speed_kmh)
        simplified_graph = _largest_connected_component(simplified_graph)

        graph_nodes, graph_edges = _prepare_graph_outputs(simplified_graph, modality)
        graph_nodes_path = modality_dir / "graph_nodes.geojson"
        graph_edges_path = modality_dir / "graph_edges.geojson"
        _save_geojson(graph_nodes, graph_nodes_path)
        _save_geojson(graph_edges, graph_edges_path)

        time_matrix = build_time_matrix(simplified_graph, attr="time_min")
        time_matrix_path = modality_dir / "time_matrix.csv"
        _save_time_matrix(time_matrix, time_matrix_path)

        modality_artifacts.append(
            ModalityArtifacts(
                modality=modality.value,
                raw_stop_count=len(agg_stops),
                projected_stop_count=len(filtered_stops),
                graph_node_count=len(graph_nodes),
                graph_edge_count=len(graph_edges),
                files={
                    "aggregated_stops": str(agg_stops_path),
                    "lines": str(lines_path),
                    "projected_lines": str(projected_lines_path),
                    "projected_stops": str(projected_stops_path),
                    "graph_nodes": str(graph_nodes_path),
                    "graph_edges": str(graph_edges_path),
                    "time_matrix": str(time_matrix_path),
                },
            )
        )

    manifest = {
        "place": place,
        "slug": slugify_place(place),
        "speed_kmh": speed_kmh,
        "boundary": str(boundary_path),
        "centre": str(centre_path),
        "modalities": [artifact.__dict__ for artifact in modality_artifacts],
    }

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

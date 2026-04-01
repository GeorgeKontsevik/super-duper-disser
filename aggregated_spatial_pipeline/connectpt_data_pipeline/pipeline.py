from __future__ import annotations

import csv
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import momepy as mp
import networkx as nx
import osmnx as ox
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata

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


def _save_geodata(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        prepare_geodata_for_parquet(gdf).to_parquet(path)
    elif path.suffix.lower() == ".gpkg":
        gdf.to_file(path, driver="GPKG")
    else:
        gdf.to_file(path, driver="GeoJSON")


def _clip_to_boundary(gdf: gpd.GeoDataFrame, boundary_geom, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    boundary_gdf = gpd.GeoDataFrame({"geometry": [boundary_geom]}, crs=crs)
    if gdf.crs != boundary_gdf.crs:
        boundary_gdf = boundary_gdf.to_crs(gdf.crs)
    clipped = gdf.clip(boundary_gdf)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].reset_index(drop=True)
    return clipped


def _save_time_matrix(matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix.tolist())


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    boundary_path: str | Path | None = None,
    drive_roads_path: str | Path | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if boundary_path is not None:
        place_gdf = read_geodata(Path(boundary_path))
        if place_gdf.empty:
            raise ValueError(f"Boundary override is empty: {boundary_path}")
        if "display_name" not in place_gdf.columns:
            place_gdf = place_gdf.copy()
            place_gdf["display_name"] = place
        boundary = place_gdf.union_all()
    else:
        place_gdf, boundary = geocode_place_boundary(place)
    boundary_path = output_path / "boundary.parquet"
    _save_geodata(place_gdf, boundary_path)

    centre_gdf = gpd.GeoDataFrame(
        [{"place": place, "geometry": boundary.centroid}],
        crs=place_gdf.crs,
    )
    centre_path = output_path / "centre.parquet"
    _save_geodata(centre_gdf, centre_path)

    stops_by_modality: dict[Modality, gpd.GeoDataFrame] = {}
    lines_by_modality: dict[Modality, gpd.GeoDataFrame] = {}
    preloaded_drive_lines: gpd.GeoDataFrame | None = None

    if drive_roads_path is not None and Modality.BUS in modalities:
        try:
            preloaded_drive_lines = read_geodata(Path(drive_roads_path))
            if preloaded_drive_lines.empty:
                logger.warning(
                    "ConnectPT preloaded drive roads are empty ({}); fallback to OSM download for bus lines.",
                    drive_roads_path,
                )
                preloaded_drive_lines = None
            else:
                preloaded_drive_lines = preloaded_drive_lines[
                    preloaded_drive_lines.geometry.notna() & ~preloaded_drive_lines.geometry.is_empty
                ].copy()
                preloaded_drive_lines = preloaded_drive_lines[
                    preloaded_drive_lines.geometry.geom_type.isin(["LineString", "MultiLineString"])
                ].copy()
                preloaded_drive_lines = gpd.GeoDataFrame(
                    preloaded_drive_lines[["geometry"]].copy(),
                    geometry="geometry",
                    crs=preloaded_drive_lines.crs,
                )
                logger.info(
                    "ConnectPT will reuse preloaded drive roads for bus lines: {} (features={})",
                    drive_roads_path,
                    len(preloaded_drive_lines),
                )
        except Exception as exc:
            logger.warning(
                "ConnectPT failed to read preloaded drive roads ({}): {}. "
                "Fallback to OSM download for bus lines.",
                drive_roads_path,
                exc,
            )
            preloaded_drive_lines = None

    for modality in modalities:
        try:
            stops_by_modality.update(get_agg_stops(boundary, [modality]))
        except Exception as exc:
            logger.warning(
                "ConnectPT stops collection skipped for modality '{}' due to OSM/processing error: {}",
                modality.value,
                exc,
            )

        try:
            lines_by_modality.update(
                get_lines(
                    boundary,
                    [modality],
                    preloaded_drive_lines=preloaded_drive_lines if modality == Modality.BUS else None,
                )
            )
        except Exception as exc:
            logger.warning(
                "ConnectPT lines collection skipped for modality '{}' due to OSM/processing error: {}",
                modality.value,
                exc,
            )

    modality_artifacts: list[ModalityArtifacts] = []

    for modality in modalities:
        modality_dir = output_path / modality.value
        modality_dir.mkdir(parents=True, exist_ok=True)

        if modality not in stops_by_modality or modality not in lines_by_modality:
            logger.warning(
                "ConnectPT modality '{}' has incomplete inputs (stops_present={}, lines_present={}); skipping.",
                modality.value,
                modality in stops_by_modality,
                modality in lines_by_modality,
            )
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
        agg_stops = _clip_to_boundary(agg_stops, boundary, crs=place_gdf.crs)
        lines = _clip_to_boundary(lines, boundary, crs=place_gdf.crs)

        agg_stops_path = modality_dir / "aggregated_stops.parquet"
        lines_path = modality_dir / "lines.parquet"
        _save_geodata(agg_stops, agg_stops_path)
        _save_geodata(lines, lines_path)

        roads_with_stops, filtered_stops = project_stops_on_roads(lines, agg_stops)
        roads_with_stops = _ensure_length_meter(roads_with_stops)
        roads_with_stops = _clip_to_boundary(roads_with_stops, boundary, crs=place_gdf.crs)
        filtered_stops = _clip_to_boundary(filtered_stops, boundary, crs=place_gdf.crs)

        projected_lines_path = modality_dir / "projected_lines.parquet"
        projected_stops_path = modality_dir / "projected_stops.parquet"
        _save_geodata(roads_with_stops, projected_lines_path)
        _save_geodata(filtered_stops, projected_stops_path)

        roads_graph = roads_to_graph(roads_with_stops, filtered_stops)
        simplified_graph = stop_complete_then_prune(roads_graph, speed_kmh=speed_kmh)
        simplified_graph = _largest_connected_component(simplified_graph)
        graph_pickle_path = modality_dir / "graph.pkl"
        _save_pickle(simplified_graph, graph_pickle_path)

        graph_nodes, graph_edges = _prepare_graph_outputs(simplified_graph, modality)
        graph_nodes = _clip_to_boundary(graph_nodes, boundary, crs=place_gdf.crs)
        graph_edges = _clip_to_boundary(graph_edges, boundary, crs=place_gdf.crs)
        graph_nodes_path = modality_dir / "graph_nodes.parquet"
        graph_edges_path = modality_dir / "graph_edges.parquet"
        _save_geodata(graph_nodes, graph_nodes_path)
        _save_geodata(graph_edges, graph_edges_path)

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
                    "graph_pickle": str(graph_pickle_path),
                    "time_matrix": str(time_matrix_path),
                },
            )
        )

    manifest = {
        "place": place,
        "slug": slugify_place(place),
        "boundary_source": str(Path(boundary_path).resolve()) if boundary_path is not None else "osmnx_geocode",
        "speed_kmh": speed_kmh,
        "boundary": str(boundary_path),
        "centre": str(centre_path),
        "modalities": [artifact.__dict__ for artifact in modality_artifacts],
    }

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

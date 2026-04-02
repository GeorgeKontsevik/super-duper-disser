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
import pandas as pd
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
from preprocess.stops import aggregate_stops, get_agg_stops
from preprocess.types import Modality


DEFAULT_SPEED_KMH = 20.0
IDUEDU_CONNECTPT_BRIDGE_DISTANCE_M = 30.0


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


def _save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _prepare_graph_outputs(graph: nx.Graph, modality: Modality) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes_gdf, edges_gdf = mp.nx_to_gdf(graph)
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()
    nodes["modality"] = modality.value
    edges["modality"] = modality.value
    if "geometry" in nodes.columns:
        nodes[["x_coord", "y_coord"]] = nodes.geometry.apply(lambda p: [round(p.x, 2), round(p.y, 2)]).tolist()
    return nodes, edges


def _extract_iduedu_connectpt_candidate_stops(
    intermodal_nodes_path: Path,
    modality: Modality,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes = read_geodata(intermodal_nodes_path)
    if nodes.empty:
        raise ValueError(f"Intermodal graph nodes are empty: {intermodal_nodes_path}")
    if "index" not in nodes.columns:
        raise ValueError(f"Intermodal graph nodes do not contain stable 'index': {intermodal_nodes_path}")

    node_type = nodes.get("type", pd.Series(index=nodes.index, dtype="object")).astype("string")
    stop_transport_type = nodes.get("stop_transport_type", pd.Series(index=nodes.index, dtype="object")).astype("string")
    source = nodes.get("source", pd.Series(index=nodes.index, dtype="object")).astype("string")
    name = nodes.get("name", pd.Series(index=nodes.index, dtype="object")).astype("string")
    ref = nodes.get("ref", pd.Series(index=nodes.index, dtype="object")).astype("string")

    # Keep raw modality stop nodes from the graph, plus only the extra tag-derived platforms
    # injected to mirror connectpt-compatible stop tags.
    modality_mask = node_type == modality.value
    extra_platform_mask = (
        (source == "osm_extra_stop_tag")
        & (stop_transport_type == modality.value)
    )
    raw = nodes[modality_mask | extra_platform_mask].copy()
    raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
    if raw.empty:
        raise ValueError(
            f"No intermodal stop candidates found for modality '{modality.value}' in {intermodal_nodes_path}"
        )

    raw["raw_stop_id"] = raw["index"].astype(str)
    raw["name"] = raw.get("name", pd.Series(index=raw.index, dtype="object")).astype("string")
    raw["ref"] = raw.get("ref", pd.Series(index=raw.index, dtype="object")).astype("string")
    raw["name"] = raw["name"].where(raw["name"].notna(), raw["ref"])
    raw["stop_origin"] = pd.Series("iduedu_graph_stop", index=raw.index, dtype="string")
    raw.loc[extra_platform_mask.loc[raw.index], "stop_origin"] = "iduedu_extra_stop_tag"
    raw["modality"] = modality.value
    raw["group_name"] = raw["raw_stop_id"]
    raw["original_stops"] = 1
    raw["original_ids"] = raw["raw_stop_id"].map(lambda value: [str(value)])
    raw_stops = raw[
        [
            "raw_stop_id",
            "index",
            "type",
            "source",
            "stop_transport_type",
            "stop_origin",
            "name",
            "modality",
            "group_name",
            "original_stops",
            "original_ids",
            "geometry",
        ]
    ].copy().reset_index(drop=True)

    aggregate_input = raw_stops[["geometry", "name"]].copy()
    aggregate_input["name"] = [
        None if pd.isna(value) else str(value)
        for value in aggregate_input["name"].tolist()
    ]
    aggregate_input.index = raw_stops["raw_stop_id"].astype(str)
    simplified = aggregate_stops(
        aggregate_input,
        distance_threshold=IDUEDU_CONNECTPT_BRIDGE_DISTANCE_M,
        progress_desc=f"Stops aggregation [{modality.value}]",
    ).reset_index(drop=True)
    simplified["modality"] = modality.value
    simplified["stop_source"] = "iduedu_bridge"
    simplified["aggregated_stop_id"] = [f"{modality.value}_{i}" for i in range(len(simplified))]
    return raw_stops, simplified


def _build_raw_to_aggregated_mapping(
    raw_stops: gpd.GeoDataFrame,
    aggregated_stops: gpd.GeoDataFrame,
    modality: Modality,
) -> gpd.GeoDataFrame:
    raw_lookup = raw_stops.set_index("raw_stop_id", drop=False)
    records: list[dict] = []
    for _, row in aggregated_stops.iterrows():
        aggregated_stop_id = str(row["aggregated_stop_id"])
        centroid = row.geometry
        for raw_id in row.get("original_ids", []):
            raw_id = str(raw_id)
            raw_row = raw_lookup.loc[raw_id]
            records.append(
                {
                    "modality": modality.value,
                    "aggregated_stop_id": aggregated_stop_id,
                    "raw_stop_id": raw_id,
                    "raw_node_index": str(raw_row.get("index")),
                    "stop_origin": raw_row.get("stop_origin"),
                    "raw_type": raw_row.get("type"),
                    "distance_to_aggregated_centroid_m": float(raw_row.geometry.distance(centroid)),
                    "geometry": raw_row.geometry,
                }
            )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=raw_stops.crs)


def build_connectpt_osm_bundle(
    place: str,
    modalities: list[Modality],
    output_dir: str | Path,
    speed_kmh: float = DEFAULT_SPEED_KMH,
    boundary_path: str | Path | None = None,
    drive_roads_path: str | Path | None = None,
    intermodal_nodes_path: str | Path | None = None,
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
    raw_stops_by_modality: dict[Modality, gpd.GeoDataFrame] = {}
    lines_by_modality: dict[Modality, gpd.GeoDataFrame] = {}
    preloaded_drive_lines: gpd.GeoDataFrame | None = None
    intermodal_nodes_file = Path(intermodal_nodes_path).resolve() if intermodal_nodes_path is not None else None

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
        if intermodal_nodes_file is not None and intermodal_nodes_file.exists():
            try:
                raw_stops, aggregated_stops = _extract_iduedu_connectpt_candidate_stops(intermodal_nodes_file, modality)
                raw_stops_by_modality[modality] = raw_stops
                stops_by_modality[modality] = aggregated_stops
                logger.info(
                    "ConnectPT modality '{}' will reuse intermodal graph stops (raw={}, aggregated={}, source={})",
                    modality.value,
                    len(raw_stops),
                    len(aggregated_stops),
                    intermodal_nodes_file.name,
                )
            except Exception as exc:
                logger.warning(
                    "ConnectPT failed to derive modality '{}' stops from intermodal graph ({}). "
                    "Falling back to direct OSM stop collection.",
                    modality.value,
                    exc,
                )

        if modality not in stops_by_modality:
            try:
                direct = get_agg_stops(boundary, [modality])
                if modality in direct:
                    stops_by_modality[modality] = direct[modality]
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
        raw_stops = raw_stops_by_modality.get(modality)
        agg_stops = _clip_to_boundary(agg_stops, boundary, crs=place_gdf.crs)
        lines = _clip_to_boundary(lines, boundary, crs=place_gdf.crs)
        if raw_stops is not None:
            raw_stops = _clip_to_boundary(raw_stops, boundary, crs=place_gdf.crs)

        agg_stops_path = modality_dir / "aggregated_stops.parquet"
        raw_stops_path = modality_dir / "raw_stops.parquet"
        mapping_path = modality_dir / "raw_to_aggregated_stop_map.parquet"
        lines_path = modality_dir / "lines.parquet"
        _save_geodata(agg_stops, agg_stops_path)
        if raw_stops is not None:
            _save_geodata(raw_stops, raw_stops_path)
            raw_to_aggregated = _build_raw_to_aggregated_mapping(raw_stops, agg_stops, modality)
            _save_geodata(raw_to_aggregated, mapping_path)
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
                    **({"raw_stops": str(raw_stops_path), "raw_to_aggregated_stop_map": str(mapping_path)} if raw_stops is not None else {}),
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
        "stop_source_policy": (
            "intermodal_iduedu_bridge_then_connectpt_fallback"
            if intermodal_nodes_file is not None
            else "connectpt_direct_osm"
        ),
        "intermodal_nodes_source": str(intermodal_nodes_file) if intermodal_nodes_file is not None else None,
        "bridge_distance_threshold_m": IDUEDU_CONNECTPT_BRIDGE_DISTANCE_M if intermodal_nodes_file is not None else None,
        "boundary": str(boundary_path),
        "centre": str(centre_path),
        "modalities": [artifact.__dict__ for artifact in modality_artifacts],
    }

    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

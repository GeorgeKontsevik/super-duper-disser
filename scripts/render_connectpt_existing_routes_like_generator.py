from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONNECTPT_ROOT = ROOT / "connectpt"
if CONNECTPT_ROOT.exists() and str(CONNECTPT_ROOT) not in sys.path:
    sys.path.insert(0, str(CONNECTPT_ROOT))

from aggregated_spatial_pipeline.visualization import normalize_preview_gdf  # noqa: E402
from aggregated_spatial_pipeline.connectpt_data_pipeline.run_route_generator_external import _save_route_preview  # noqa: E402


def _read(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf if not gdf.empty else None


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _load_iduedu_route_edges(city_dir: Path, modality: str) -> gpd.GeoDataFrame | None:
    edges = _read(city_dir / "intermodal_graph_iduedu" / "graph_edges.parquet")
    if edges is None or edges.empty or "route" not in edges.columns or "type" not in edges.columns:
        return None
    edges = edges[(edges["type"].astype("string") == str(modality)) & edges["route"].notna()].copy()
    edges = edges[edges.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if edges.empty:
        return None
    edges["route_label"] = edges["route"].astype(str)
    return edges


def _route_values(value) -> set[str]:
    if value is None:
        return set()
    text = str(value)
    if not text or text == "<NA>" or text.lower() == "nan":
        return set()
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {text}
    if isinstance(parsed, (list, tuple, set)):
        return {str(item) for item in parsed if str(item)}
    return {str(parsed)}


def _load_iduedu_route_stops(city_dir: Path, modality: str, route_labels: set[str]) -> gpd.GeoDataFrame | None:
    nodes = _read(city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet")
    if nodes is None or nodes.empty or "route" not in nodes.columns:
        return None
    route_text = nodes["route"].astype("string")
    mask = route_text.notna() & route_text.apply(lambda value: bool(_route_values(value) & route_labels))
    if "type" in nodes.columns:
        type_text = nodes["type"].astype("string")
        mask &= type_text.isin([str(modality), "platform"])
    stops = nodes[mask].copy()
    stops = stops[stops.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    return stops if not stops.empty else None


def _ordered_component_sequence(component_graph: nx.Graph) -> list[int]:
    if component_graph.number_of_edges() == 0:
        return list(component_graph.nodes())
    endpoints = sorted([node for node, degree in component_graph.degree() if degree == 1])
    start = endpoints[0] if endpoints else sorted(component_graph.nodes())[0]
    sequence = [int(start)]
    visited_edges: set[tuple[int, int]] = set()
    current = start
    previous = None
    while len(visited_edges) < component_graph.number_of_edges():
        candidates = []
        for neighbor in sorted(component_graph.neighbors(current)):
            edge_key = tuple(sorted((int(current), int(neighbor))))
            if edge_key not in visited_edges and neighbor != previous:
                candidates.append(neighbor)
        if not candidates:
            for neighbor in sorted(component_graph.neighbors(current)):
                edge_key = tuple(sorted((int(current), int(neighbor))))
                if edge_key not in visited_edges:
                    candidates.append(neighbor)
                    break
        if not candidates:
            remaining = [
                tuple(sorted((int(u), int(v))))
                for u, v in component_graph.edges()
                if tuple(sorted((int(u), int(v)))) not in visited_edges
            ]
            if not remaining:
                break
            current = remaining[0][0]
            previous = None
            sequence.append(int(current))
            continue
        neighbor = candidates[0]
        visited_edges.add(tuple(sorted((int(current), int(neighbor)))))
        sequence.append(int(neighbor))
        previous, current = current, neighbor
    return sequence


def _build_existing_route_graph_and_tensor(
    *,
    city_dir: Path,
    modality: str,
    route_edges: gpd.GeoDataFrame,
) -> tuple[nx.Graph, torch.Tensor, list[dict], list[tuple[int, int]]]:
    nodes = _read(city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet")
    if nodes is None or nodes.empty:
        raise FileNotFoundError("Missing intermodal graph nodes for existing route preview.")
    nodes_by_id = nodes.set_index("index", drop=False)

    graph = nx.Graph()
    graph.graph["crs"] = route_edges.crs
    edge_nodes = set(route_edges["u"].astype(int)).union(set(route_edges["v"].astype(int)))
    for node_id in sorted(edge_nodes):
        if node_id not in nodes_by_id.index:
            continue
        row = nodes_by_id.loc[node_id]
        graph.add_node(int(node_id), x=float(row["x"]), y=float(row["y"]))

    for _, row in route_edges.iterrows():
        u = int(row["u"])
        v = int(row["v"])
        if u not in graph or v not in graph:
            continue
        graph.add_edge(
            u,
            v,
            geometry=row.geometry,
            weight=float(row.get("length_meter", 0.0) or 0.0),
            time_min=float(row.get("time_min", 0.0) or 0.0),
        )

    route_counts = route_edges["route_label"].astype(str).value_counts()
    sequences: list[list[int]] = []
    legend_rows: list[dict] = []
    endpoint_pairs: list[tuple[int, int]] = []
    for idx, route_label in enumerate(route_counts.index.tolist(), start=1):
        route_part = route_edges[route_edges["route_label"].astype(str) == route_label]
        route_graph = nx.Graph()
        for _, row in route_part.iterrows():
            u = int(row["u"])
            v = int(row["v"])
            if u in graph and v in graph:
                route_graph.add_edge(u, v)
        full_sequence: list[int] = []
        components = sorted(nx.connected_components(route_graph), key=len, reverse=True)
        for component in components:
            component_sequence = _ordered_component_sequence(route_graph.subgraph(component).copy())
            if len(component_sequence) < 2:
                continue
            endpoint_pairs.append((int(component_sequence[0]), int(component_sequence[-1])))
            full_sequence.extend(component_sequence)
        if len(full_sequence) >= 2:
            sequences.append(full_sequence)
            legend_rows.append(
                {
                    "display_route": f"route {len(sequences)}",
                    "existing_route": route_label,
                    "edge_count": int(route_counts[route_label]),
                }
            )

    if not sequences:
        raise ValueError(f"No route sequences could be built for modality={modality!r}")
    max_len = max(len(sequence) for sequence in sequences)
    padded = [sequence + [-1] * (max_len - len(sequence)) for sequence in sequences]
    return graph, torch.tensor(padded, dtype=torch.long), legend_rows, endpoint_pairs


def render_existing_routes(city_dir: Path, modality: str, output_path: Path) -> dict[str, int | str]:
    modality_dir = city_dir / "connectpt_osm" / modality
    if not modality_dir.exists():
        raise FileNotFoundError(f"Missing ConnectPT modality directory: {modality_dir}")

    boundary_path = _first_existing(
        [
            city_dir / "analysis_territory" / "buffer.parquet",
            city_dir / "connectpt_osm" / "boundary.parquet",
            city_dir / "blocksnet" / "boundary.parquet",
        ]
    )
    boundary = _read(boundary_path) if boundary_path else None
    boundary_plot = normalize_preview_gdf(boundary, target_crs="EPSG:3857")

    roads_path = _first_existing(
        [
            city_dir / "derived_layers" / "roads_drive_osmnx.parquet",
            city_dir / "street_pattern" / city_dir.name / "roads.geojson",
            city_dir / "blocksnet_raw_osm" / "roads.parquet",
            city_dir / "blocksnet" / "roads.parquet",
        ]
    )
    roads = normalize_preview_gdf(_read(roads_path) if roads_path else None, boundary_plot, target_crs="EPSG:3857")
    route_lines = _load_iduedu_route_edges(city_dir, modality)
    route_source = "iduedu_route_edges"
    if route_lines is None or route_lines.empty:
        route_lines = normalize_preview_gdf(_read(modality_dir / "projected_lines.parquet"), boundary_plot, target_crs="EPSG:3857")
        if route_lines is not None and not route_lines.empty:
            route_lines = route_lines.copy()
            route_lines["route_label"] = [str(idx + 1) for idx in range(len(route_lines))]
            route_source = "connectpt_projected_line_rows"
    route_labels = set(route_lines["route_label"].astype(str)) if route_lines is not None and not route_lines.empty else set()
    stops = _load_iduedu_route_stops(city_dir, modality, route_labels) if route_labels else None
    if stops is None or stops.empty:
        stops = _read(modality_dir / "aggregated_stops.parquet")
    stops = normalize_preview_gdf(stops, boundary_plot, target_crs="EPSG:3857")
    graph_edges = normalize_preview_gdf(_read(modality_dir / "graph_edges.parquet"), boundary_plot, target_crs="EPSG:3857")

    if route_lines is None or route_lines.empty:
        raise ValueError(f"No existing route lines found in {modality_dir}")

    route_graph, routes_tensor, legend_rows, endpoint_pairs = _build_existing_route_graph_and_tensor(
        city_dir=city_dir,
        modality=modality,
        route_edges=route_lines,
    )
    summary = {
        "modality": modality,
        "route_count": len(legend_rows),
        "cost": 0.0,
        "att": 0.0,
        "unserved_demand_pct": 0.0,
    }
    _save_route_preview(
        graph=route_graph,
        boundary=boundary,
        routes_tensor=routes_tensor,
        graph_node_labels=None,
        summary=summary,
        out_path=output_path,
        draw_network=True,
        roads=roads,
        title=f"ConnectPT existing routes ({modality})",
        footer_lines=[
            (
                f"routes={len(legend_rows)} | existing route edges={len(route_lines)} | "
                f"source={route_source}"
            )
        ],
        endpoint_node_pairs=endpoint_pairs,
    )
    legend_path = output_path.with_name(f"{output_path.stem}_route_legend.csv")
    pd.DataFrame(legend_rows).to_csv(legend_path, index=False)

    return {
        "output_path": str(output_path),
        "route_legend_path": str(legend_path),
        "route_source": route_source,
        "routes": int(len(legend_rows)),
        "endpoint_pairs": int(len(endpoint_pairs)),
        "route_segments": int(len(route_lines)),
        "stops": int(0 if stops is None else len(stops)),
        "graph_edges": int(0 if graph_edges is None else len(graph_edges)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render existing ConnectPT routes in the route-generator preview style.")
    parser.add_argument("--city-dir", required=True, type=Path)
    parser.add_argument("--modality", default="bus")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    city_dir = args.city_dir.resolve()
    output = args.output or city_dir / "preview_png" / "all_together" / f"pt_route_generator_{args.modality}_existing_only.png"
    summary = render_existing_routes(city_dir, args.modality, output.resolve())
    print(summary)


if __name__ == "__main__":
    main()

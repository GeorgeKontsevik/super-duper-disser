from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd
from huggingface_hub import hf_hub_download
import networkx as nx
import numpy as np
import osmnx as ox
import osmapi as osm
from rtree import index
from shapely.geometry import LineString, Point, Polygon
from tqdm.auto import tqdm


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
CLASSIFIER_DIR = REPO_ROOT / "street-pattern-classifier"

if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from block_dataset import BlockDataset
from classification import classify_blocks
from model import class_names

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the street-pattern classifier on a single city."
    )
    parser.add_argument(
        "--place",
        default="Montreal, Canada",
        help='Place string for OSM geocoding, for example "Montreal, Canada".',
    )
    parser.add_argument(
        "--network-type",
        default="drive",
        help='OSMnx network type, for example "drive" or "all".',
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=2000,
        help="Grid cell size in projected CRS units.",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=20_000,
        help="Buffer radius around the city-centre OSM node, in meters.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device. Use "cpu" by default.',
    )
    parser.add_argument(
        "--output",
        default=str(EXPERIMENTS_DIR / "outputs" / "montreal_predictions.json"),
        help="Where to save the prediction summary JSON.",
    )
    return parser.parse_args()


def _relation_get(api: osm.OsmApi, relation_id: int) -> dict:
    if hasattr(api, "RelationGet"):
        return api.RelationGet(relation_id)
    return api.relation_get(relation_id)


def _node_get(api: osm.OsmApi, node_id: int) -> dict:
    if hasattr(api, "NodeGet"):
        return api.NodeGet(node_id)
    return api.node_get(node_id)


def resolve_city_centre_node(place: str) -> tuple[dict, dict]:
    geocoded = ox.geocode_to_gdf(place)
    if geocoded.empty:
        raise ValueError(f"Could not geocode place: {place}")

    row = geocoded.iloc[0]
    relation_id = row.get("osm_id")
    osm_type = str(row.get("osm_type", "")).lower()

    if relation_id is None or osm_type != "relation":
        raise ValueError(
            f"Expected a relation for {place}, got osm_type={osm_type!r}, osm_id={relation_id!r}"
        )

    api = osm.OsmApi()
    relation = _relation_get(api, int(relation_id))
    members = relation.get("member") or relation.get("members") or []

    preferred_roles = ("admin_centre", "admin_center", "label", "capital")
    node_member = None
    for role in preferred_roles:
        node_member = next(
            (
                member
                for member in members
                if member.get("type") == "node" and member.get("role") == role
            ),
            None,
        )
        if node_member is not None:
            break

    if node_member is None:
        raise ValueError(
            f"Could not find a centre node for {place} in relation {relation_id}"
        )

    node = _node_get(api, int(node_member["ref"]))
    return relation, node


def build_buffer_polygon(node: dict, buffer_m: float):
    point = Point(float(node["lon"]), float(node["lat"]))
    point_gdf = gpd.GeoDataFrame({"geometry": [point]}, crs=4326)
    buffered = point_gdf.to_crs(3857).buffer(buffer_m)
    return gpd.GeoSeries(buffered, crs=3857).to_crs(4326).iloc[0]


def split_graph_by_grid_for_polygon(
    graph: nx.MultiDiGraph,
    polygon,
    grid_step: float = 1000,
):
    nodes, _ = ox.graph_to_gdfs(graph)

    minx, miny, maxx, maxy = polygon.bounds

    x_coords = np.arange(minx, maxx, grid_step)
    y_coords = np.arange(miny, maxy, grid_step)

    if len(x_coords) == 0 or x_coords[-1] < maxx:
        x_coords = np.append(x_coords, maxx)
    if len(y_coords) == 0 or y_coords[-1] < maxy:
        y_coords = np.append(y_coords, maxy)

    subgraphs = {}

    node_cell_idx = index.Index()
    node_cell_bounds = {}

    for i in range(len(x_coords) - 1):
        for j in range(len(y_coords) - 1):
            cell_polygon = Polygon(
                [
                    (x_coords[i], y_coords[j]),
                    (x_coords[i + 1], y_coords[j]),
                    (x_coords[i + 1], y_coords[j + 1]),
                    (x_coords[i], y_coords[j + 1]),
                ]
            )

            subgraph = nx.MultiDiGraph()
            subgraph.graph["crs"] = graph.graph["crs"]

            subgraphs[(i, j)] = {
                "graph": subgraph,
                "polygon": cell_polygon,
            }

    for i, (key, cell_data) in enumerate(subgraphs.items()):
        bounds = cell_data["polygon"].bounds
        node_cell_idx.insert(i, bounds)
        node_cell_bounds[i] = (key, cell_data)

    node_to_cell = {}
    for node_id, data in tqdm(graph.nodes(data=True), desc="Assigning nodes to cells"):
        point = Point(data["x"], data["y"])
        potential_cells = list(
            node_cell_idx.intersection((data["x"], data["y"], data["x"], data["y"]))
        )

        assigned_to_cell = False
        for cell_idx in potential_cells:
            key, cell_data = node_cell_bounds[cell_idx]
            if cell_data["polygon"].contains(point):
                cell_data["graph"].add_node(node_id, **data)
                node_to_cell[node_id] = key
                assigned_to_cell = True
                break

        if not assigned_to_cell:
            for cell_idx in potential_cells:
                key, cell_data = node_cell_bounds[cell_idx]
                if cell_data["polygon"].intersects(point):
                    cell_data["graph"].add_node(node_id, **data)
                    node_to_cell[node_id] = key
                    break

    edge_cell_idx = index.Index()
    edge_cell_data_by_idx = {}
    for i, (key, cell_data) in enumerate(subgraphs.items()):
        bounds = cell_data["polygon"].bounds
        edge_cell_idx.insert(i, bounds)
        edge_cell_data_by_idx[i] = (key, cell_data)

    edges_with_geometry = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        if "geometry" in data:
            line = data["geometry"]
        else:
            line = LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        edges_with_geometry.append((u, v, key, data, line))

    for u, v, key, data, line in tqdm(edges_with_geometry, desc="Assigning edges to cells"):
        u_cell = node_to_cell.get(u)
        v_cell = node_to_cell.get(v)
        line_bounds = line.bounds
        potential_cell_indices = list(edge_cell_idx.intersection(line_bounds))

        for cell_index in potential_cell_indices:
            cell_key, cell_data = edge_cell_data_by_idx[cell_index]
            cell_polygon = cell_data["polygon"]
            cell_bounds = cell_polygon.bounds

            if (
                line_bounds[0] > cell_bounds[2]
                or line_bounds[2] < cell_bounds[0]
                or line_bounds[1] > cell_bounds[3]
                or line_bounds[3] < cell_bounds[1]
            ):
                continue

            if not line.intersects(cell_polygon):
                continue

            subgraph = cell_data["graph"]
            intersection = line.intersection(cell_polygon)
            if intersection.is_empty:
                continue

            if intersection.geom_type == "LineString":
                segments = [intersection]
            elif intersection.geom_type == "MultiLineString":
                segments = list(intersection.geoms)
            else:
                continue

            for segment in segments:
                start_coord = segment.coords[0]
                end_coord = segment.coords[-1]

                start_node_exists = u in subgraph.nodes() if cell_key == u_cell else False
                end_node_exists = v in subgraph.nodes() if cell_key == v_cell else False

                if start_node_exists and end_node_exists:
                    start_node_id = u
                    end_node_id = v
                elif start_node_exists:
                    start_node_id = u
                    end_node_id = f"end_{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                    subgraph.add_node(end_node_id, x=end_coord[0], y=end_coord[1])
                elif end_node_exists:
                    start_node_id = f"start_{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                    end_node_id = v
                    subgraph.add_node(start_node_id, x=start_coord[0], y=start_coord[1])
                else:
                    start_node_id = f"start_{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                    end_node_id = f"end_{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                    subgraph.add_node(start_node_id, x=start_coord[0], y=start_coord[1])
                    subgraph.add_node(end_node_id, x=end_coord[0], y=end_coord[1])

                edge_data = data.copy()
                edge_data["geometry"] = segment
                edge_data["original_nodes"] = (u, v)
                edge_data["segment_id"] = f"{u}_{v}_{key}_{hash(segment.wkt) % 10000:04d}"
                edge_data["original_edge_key"] = key

                subgraph.add_edge(start_node_id, end_node_id, **edge_data)

    result = {}
    for key, cell_data in subgraphs.items():
        if len(cell_data["graph"].nodes) > 0:
            result[key] = {
                "graph": cell_data["graph"],
                "polygon": cell_data["polygon"],
            }

    return result


def main() -> None:
    args = parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models_dir = EXPERIMENTS_DIR / "models"

    stage_bar = tqdm(total=8, desc="Street-pattern pipeline", unit="stage")

    stage_bar.set_postfix_str("download model")
    model_path = hf_hub_download(
        repo_id="nochka/street-pattern-classifier",
        filename="best_model.pth",
        local_dir=str(models_dir),
    )
    stage_bar.update(1)

    stage_bar.set_postfix_str("resolve city relation and centre node")
    relation, centre_node = resolve_city_centre_node(args.place)
    stage_bar.update(1)

    stage_bar.set_postfix_str("build 20km buffer")
    polygon = build_buffer_polygon(centre_node, args.buffer_m)
    stage_bar.update(1)

    stage_bar.set_postfix_str("download street graph")
    graph = ox.graph_from_polygon(
        polygon,
        network_type=args.network_type,
        simplify=True,
    )
    graph = ox.project_graph(graph)
    stage_bar.update(1)

    stage_bar.set_postfix_str("project buffer polygon")
    polygon_projected = (
        gpd.GeoSeries([polygon], crs=4326).to_crs(graph.graph["crs"]).iloc[0]
    )
    stage_bar.update(1)

    stage_bar.set_postfix_str("split graph to subgraphs")
    subgraphs = split_graph_by_grid_for_polygon(
        graph,
        polygon_projected,
        grid_step=args.grid_step,
    )
    stage_bar.update(1)

    stage_bar.set_postfix_str("build dataset")
    dataset = BlockDataset(subgraphs)
    stage_bar.update(1)

    stage_bar.set_postfix_str("run inference")
    predictions, probabilities = classify_blocks(
        dataset,
        model_path=model_path,
        device=args.device,
    )
    stage_bar.update(1)
    stage_bar.close()

    counts = Counter(predictions.values())
    summary = {
        "place": args.place,
        "network_type": args.network_type,
        "grid_step": args.grid_step,
        "buffer_m": args.buffer_m,
        "device": args.device,
        "relation_id": relation.get("id"),
        "centre_node_id": centre_node.get("id"),
        "centre_node_lon": float(centre_node["lon"]),
        "centre_node_lat": float(centre_node["lat"]),
        "num_subgraphs": len(subgraphs),
        "num_predictions": len(predictions),
        "class_names": class_names,
        "class_counts": {
            class_names[class_id]: counts.get(class_id, 0)
            for class_id in range(len(class_names))
        },
        "predictions": {
            str(cell_id): {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "probabilities": probabilities[cell_id].tolist(),
            }
            for cell_id, class_id in predictions.items()
        },
    }

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("Classification complete.")
    print(f"Saved summary to {output_path}")
    print(
        f"Used relation {summary['relation_id']} and centre node {summary['centre_node_id']}"
    )
    print("Class counts:")
    for class_name, count in summary["class_counts"].items():
        print(f"  {class_name}: {count}")


if __name__ == "__main__":
    main()

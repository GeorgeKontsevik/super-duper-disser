from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
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
from shapely.ops import unary_union
from tqdm.auto import tqdm


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
CLASSIFIER_DIR = REPO_ROOT / "street-pattern-classifier"

if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from block_dataset import BlockDataset
from classification import classify_blocks
from model import class_names
from run_street_pattern_canada_centres import (
    prepare_city_roads,
    resolve_model_path,
    split_roads_by_grid_for_polygon,
)


CLASS_COLOR_PALETTE = [
    "#72d6c9",
    "#9fbce8",
    "#c9ef8f",
    "#f6ab8c",
    "#f7e0a6",
    "#a79aac",
]
COMPARISON_COLOR_PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
    "#1f78b4",
    "#b2df8a",
]


def _progress_log(message: str) -> None:
    tqdm.write(f"[street-pattern] {message}")
COUNTRY_ROADS_PATHS = {
    "canada": REPO_ROOT / "data_all_cities" / "hotosm_can_roads_lines_gpkg" / "hotosm_can_roads_lines_gpkg.gpkg",
    "usa": REPO_ROOT / "data_all_cities" / "hotosm_usa_roads_lines_gpkg" / "hotosm_usa_roads_lines_gpkg.gpkg",
    "us": REPO_ROOT / "data_all_cities" / "hotosm_usa_roads_lines_gpkg" / "hotosm_usa_roads_lines_gpkg.gpkg",
    "united states": REPO_ROOT / "data_all_cities" / "hotosm_usa_roads_lines_gpkg" / "hotosm_usa_roads_lines_gpkg.gpkg",
    "united states of america": REPO_ROOT / "data_all_cities" / "hotosm_usa_roads_lines_gpkg" / "hotosm_usa_roads_lines_gpkg.gpkg",
}


def _round_probabilities(values, digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def _infer_country_from_place(place: str) -> str | None:
    parts = [part.strip().lower() for part in place.split(",") if part.strip()]
    if not parts:
        return None
    return parts[-1]


def resolve_roads_source(place: str, road_source: str, roads: str | None) -> tuple[str, Path | None]:
    if roads:
        return "local", Path(roads).resolve()

    if road_source == "osm":
        return "osm", None

    country = _infer_country_from_place(place)
    if country is not None:
        candidate = COUNTRY_ROADS_PATHS.get(country)
        if candidate is not None and candidate.exists():
            return "local", candidate.resolve()

    if road_source == "local":
        raise FileNotFoundError(
            f"Could not resolve a local roads file for {place!r}. "
            "Pass --roads explicitly or use --road-source osm."
        )

    return "osm", None


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
        "--center-node-id",
        "--centre-node-id",
        dest="center_node_id",
        type=int,
        help=(
            "Optional OSM node id for city centre. "
            "If provided, this node is used directly and relation-centre lookup by place is skipped."
        ),
    )
    parser.add_argument(
        "--network-type",
        default="drive",
        help='OSMnx network type, for example "drive" or "all".',
    )
    parser.add_argument(
        "--road-source",
        choices=("auto", "osm", "local"),
        default="auto",
        help=(
            'Where to get roads from. "auto" uses a local country roads file when a known one exists, '
            'otherwise downloads from OSM.'
        ),
    )
    parser.add_argument(
        "--roads",
        help=(
            "Optional path to a local roads GeoPackage/GeoJSON. "
            'When provided, this path is used for --road-source local or auto.'
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        help=(
            "Historical OSM snapshot year. If omitted, uses the current live OSM road graph. "
            "For historical runs the graph is requested as of January 1 of that year."
        ),
    )
    parser.add_argument(
        "--compare-year",
        type=int,
        help=(
            "Optional second year for block-by-block comparison against --year. "
            "When provided, comparison outputs are saved inside the city output folder."
        ),
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
        "--model-path",
        default=str(EXPERIMENTS_DIR / "models" / "best_model.pth"),
        help="Local path to the classifier weights. Download is used only if this file is missing.",
    )
    parser.add_argument(
        "--output",
        default=str(EXPERIMENTS_DIR / "outputs" / "montreal_predictions.json"),
        help="Where to save the prediction summary JSON.",
    )
    parser.add_argument(
        "--map-coloring",
        choices=("top1", "multivariate", "vba"),
        default="multivariate",
        help=(
            'Map coloring mode: "top1" for argmax class color, '
            '"multivariate" for weighted color mix, '
            '"vba" for canonical PySAL value-by-alpha.'
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable OSMnx cache usage for this run (useful for a full recalculation).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(EXPERIMENTS_DIR / "outputs" / "cache" / "city_single"),
        help="Directory for per-run pickle caches of roads/subgraphs/dataset/predictions.",
    )
    return parser.parse_args()


def _relation_get(api: osm.OsmApi, relation_id: int) -> dict:
    if hasattr(api, "relation_get"):
        return api.relation_get(relation_id)
    return api.RelationGet(relation_id)


def _node_get(api: osm.OsmApi, node_id: int) -> dict:
    if hasattr(api, "node_get"):
        return api.node_get(node_id)
    return api.NodeGet(node_id)


def _cache_prefix(
    cache_dir: Path,
    *,
    place: str,
    year: int | None,
    network_type: str,
    grid_step: float,
    buffer_m: float,
    road_source_label: str,
    roads_path: Path | None,
    model_path: Path | str,
    device: str,
) -> Path:
    payload = {
        "place": place,
        "year": "current" if year is None else int(year),
        "network_type": network_type,
        "grid_step": float(grid_step),
        "buffer_m": float(buffer_m),
        "road_source": road_source_label,
        "roads_path": str(roads_path.resolve()) if roads_path is not None else None,
        "model_path": str(Path(model_path).resolve()),
        "device": str(device),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]
    return cache_dir / f"{_slugify(place)}__{payload['year']}__{digest}"


def _save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


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


def download_street_graph(polygon, network_type: str, year: int | None) -> nx.MultiDiGraph:
    old_overpass_url = getattr(ox.settings, "overpass_url", None)
    old_overpass_settings = ox.settings.overpass_settings

    try:
        if year is not None:
            date = f"{year}-01-01T00:00:00Z"
            if old_overpass_url is not None:
                ox.settings.overpass_url = "https://overpass-api.de/api"
            ox.settings.overpass_settings = (
                '[out:json][timeout:{timeout}][date:"' + date + '"]{maxsize}'
            )

        return ox.graph_from_polygon(
            polygon,
            network_type=network_type,
            simplify=True,
        )
    finally:
        if old_overpass_url is not None:
            ox.settings.overpass_url = old_overpass_url
        ox.settings.overpass_settings = old_overpass_settings


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


def build_city_prediction_gdf(
    place: str,
    relation: dict,
    centre_node: dict,
    subgraphs: dict,
    predictions: dict,
    probabilities: dict,
) -> gpd.GeoDataFrame:
    rows = []
    class_color_lookup = _class_color_lookup()
    for cell_id, class_id in predictions.items():
        cell_data = subgraphs.get(cell_id)
        if cell_data is None:
            continue

        raw_probability_values = [float(value) for value in probabilities[cell_id]]
        probability_values = _round_probabilities(raw_probability_values)
        top_indices = _top_k_indices(raw_probability_values, k=3)
        top1_idx = top_indices[0] if top_indices else int(class_id)
        top2_idx = top_indices[1] if len(top_indices) > 1 else top1_idx
        top3_idx = top_indices[2] if len(top_indices) > 2 else top2_idx
        row = {
            "place": place,
            "relation_id": relation.get("id"),
            "centre_node_id": centre_node.get("id"),
            "centre_lon": float(centre_node["lon"]),
            "centre_lat": float(centre_node["lat"]),
            "cell_id": str(cell_id),
            "cell_i": int(cell_id[0]) if isinstance(cell_id, tuple) else None,
            "cell_j": int(cell_id[1]) if isinstance(cell_id, tuple) else None,
            "class_id": int(class_id),
            "class_name": class_names[class_id],
            "top_probability": float(raw_probability_values[class_id]),
            "top1_class_id": int(top1_idx),
            "top1_class_name": class_names[top1_idx],
            "top1_probability": float(raw_probability_values[top1_idx]),
            "top2_class_id": int(top2_idx),
            "top2_class_name": class_names[top2_idx],
            "top2_probability": float(raw_probability_values[top2_idx]),
            "top3_class_id": int(top3_idx),
            "top3_class_name": class_names[top3_idx],
            "top3_probability": float(raw_probability_values[top3_idx]),
            "top3_signature": " > ".join(class_names[idx] for idx in top_indices),
            "multivariate_color": _multivariate_color_from_probabilities(
                raw_probability_values,
                class_color_lookup,
            ),
            "geometry": cell_data["polygon"],
        }
        for probability_index, probability_value in enumerate(probability_values):
            row[f"prob_{probability_index}"] = float(probability_value)
        rows.append(row)

    crs = None
    if subgraphs:
        first_cell = next(iter(subgraphs.values()))
        if "graph" in first_cell:
            crs = first_cell["graph"].graph.get("crs")
        else:
            crs = first_cell["roads"].crs
    if not rows:
        return gpd.GeoDataFrame(
            columns=["place", "cell_id", "class_id", "class_name", "top_probability", "geometry"],
            geometry="geometry",
            crs=crs,
        )

    prediction_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    if crs is not None:
        prediction_gdf = prediction_gdf.to_crs(4326)
    return prediction_gdf


def build_buffer_gdf(place: str, relation: dict, centre_node: dict, buffer_polygon_wgs84) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "place": place,
                "relation_id": relation.get("id"),
                "centre_node_id": centre_node.get("id"),
                "centre_lon": float(centre_node["lon"]),
                "centre_lat": float(centre_node["lat"]),
                "geometry": buffer_polygon_wgs84,
            }
        ],
        geometry="geometry",
        crs=4326,
    )


def build_centre_gdf(place: str, relation: dict, centre_node: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "place": place,
                "relation_id": relation.get("id"),
                "centre_node_id": centre_node.get("id"),
                "lon": float(centre_node["lon"]),
                "lat": float(centre_node["lat"]),
                "geometry": Point(float(centre_node["lon"]), float(centre_node["lat"])),
            }
        ],
        geometry="geometry",
        crs=4326,
    )


def _class_color_lookup() -> dict[str, str]:
    return {
        class_name: CLASS_COLOR_PALETTE[index % len(CLASS_COLOR_PALETTE)]
        for index, class_name in enumerate(class_names)
    }


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    clean = color.strip().lstrip("#")
    if len(clean) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    return tuple(int(clean[index : index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _top_k_indices(values: list[float], k: int) -> list[int]:
    if not values:
        return []
    ordered = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
    return [int(idx) for idx in ordered[:k]]


def _multivariate_color_from_probabilities(
    probability_values: list[float],
    color_lookup: dict[str, str],
) -> str:
    if not probability_values:
        return "#34495e"
    total = float(sum(max(0.0, float(value)) for value in probability_values))
    if total <= 0.0:
        return "#34495e"

    red = 0.0
    green = 0.0
    blue = 0.0
    for class_index, value in enumerate(probability_values):
        weight = max(0.0, float(value)) / total
        class_name = class_names[class_index]
        class_rgb = _hex_to_rgb(color_lookup[class_name])
        red += class_rgb[0] * weight
        green += class_rgb[1] * weight
        blue += class_rgb[2] * weight

    return _rgb_to_hex(
        (
            int(round(max(0.0, min(255.0, red)))),
            int(round(max(0.0, min(255.0, green)))),
            int(round(max(0.0, min(255.0, blue)))),
        )
    )


def _category_color_lookup(values: list[str]) -> dict[str, str]:
    unique_values = sorted({value for value in values if value is not None})
    return {
        value: COMPARISON_COLOR_PALETTE[index % len(COMPARISON_COLOR_PALETTE)]
        for index, value in enumerate(unique_values)
    }


def _graph_edge_length(edge_data: dict, graph: nx.MultiDiGraph, u, v) -> float:
    geometry = edge_data.get("geometry")
    if geometry is not None:
        return float(geometry.length)
    start = graph.nodes[u]
    end = graph.nodes[v]
    return float(LineString([(start["x"], start["y"]), (end["x"], end["y"])]).length)


def summarize_subgraph_metrics(subgraphs: dict) -> dict:
    metrics = {}
    for cell_id, cell_data in subgraphs.items():
        if "graph" in cell_data:
            graph = cell_data["graph"]
            total_length_m = 0.0
            for u, v, edge_data in graph.edges(data=True):
                total_length_m += _graph_edge_length(edge_data, graph, u, v)
            metrics[cell_id] = {
                "road_segment_count": int(graph.number_of_edges()),
                "road_length_m": round(total_length_m, 3),
            }
        else:
            roads = cell_data["roads"]
            metrics[cell_id] = {
                "road_segment_count": int(len(roads)),
                "road_length_m": round(float(roads.geometry.length.sum()), 3),
            }
    return metrics


def _year_label(year: int | None) -> str:
    return "current" if year is None else str(year)


def render_city_map(
    roads_wgs84: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    centre_gdf: gpd.GeoDataFrame,
    prediction_gdf: gpd.GeoDataFrame,
    title: str,
    output_path: Path,
    map_coloring: str = "multivariate",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if map_coloring == "vba":
        fig = plt.figure(figsize=(16.0, 10.0))
        grid = fig.add_gridspec(2, 1, height_ratios=(4.9, 1.3), hspace=0.2)
        ax = fig.add_subplot(grid[0, 0])
        legend_ax = fig.add_subplot(grid[1, 0])
    else:
        fig, ax = plt.subplots(figsize=(11.5, 10))
        legend_ax = None
    color_lookup = _class_color_lookup()

    if not roads_wgs84.empty:
        roads_wgs84.plot(ax=ax, color="#7f8c8d", linewidth=0.4, alpha=0.7)
    if not prediction_gdf.empty:
        prediction_plot_gdf = prediction_gdf.copy()
        if map_coloring == "vba":
            try:
                import mapclassify
                from splot.mapping import vba_choropleth, vba_legend
            except Exception:
                prediction_plot_gdf["plot_color"] = (
                    prediction_plot_gdf["class_name"].map(color_lookup).fillna("#34495e")
                )
                prediction_plot_gdf.plot(
                    ax=ax,
                    color=prediction_plot_gdf["plot_color"],
                    alpha=0.6,
                    edgecolor="#1f1f1f",
                    linewidth=0.35,
                )
                map_coloring = "top1"
            else:
                rgb_values = prediction_plot_gdf["class_id"].astype(float).to_numpy()
                rgb_bins = mapclassify.UserDefined(rgb_values, bins=list(range(len(class_names))))
                alpha_values = prediction_plot_gdf["top_probability"].astype(float).to_numpy()
                alpha_bins = mapclassify.Quantiles(alpha_values, k=min(5, max(2, len(prediction_plot_gdf))))
                base_colors = [
                    CLASS_COLOR_PALETTE[index % len(CLASS_COLOR_PALETTE)]
                    for index in range(len(class_names))
                ]
                cmap = ListedColormap(base_colors, name="street_pattern_classes")
                vba_choropleth(
                    x_var=prediction_plot_gdf["class_id"].astype(float),
                    y_var=prediction_plot_gdf["top_probability"].astype(float),
                    gdf=prediction_plot_gdf,
                    cmap=cmap,
                    rgb_mapclassify={"classifier": "user_defined", "bins": list(range(len(class_names)))},
                    alpha_mapclassify={
                        "classifier": "quantiles",
                        "k": min(5, max(2, len(prediction_plot_gdf))),
                    },
                    ax=ax,
                    legend=False,
                )
                if legend_ax is not None:
                    vba_legend(
                        rgb_bins=rgb_bins,
                        alpha_bins=alpha_bins,
                        cmap=cmap,
                        ax=legend_ax,
                    )
                    legend_ax.set_aspect("equal", adjustable="box")
                    legend_ax.set_title("VBA legend", fontsize=10)
                    legend_ax.set_xlabel("Top-1 probability bin", fontsize=9)
                    legend_ax.set_ylabel("Top-1 class", fontsize=9)
                    # legend_ax.xaxis.set_label_coords(0.5, -0.14)
                    # legend_ax.yaxis.set_label_coords(-0.08, 0.5)
                    alpha_label_count = len(getattr(alpha_bins, "bins", []))
                    if alpha_label_count > 0:
                        legend_ax.set_xticks(np.arange(alpha_label_count) + 0.5)
                        legend_ax.set_xticklabels(
                            [f"{float(value):.1f}" for value in alpha_bins.bins[:alpha_label_count]],
                            rotation=0,
                            fontsize=8,
                        )
                    rgb_label_count = min(len(class_names), len(getattr(rgb_bins, "bins", [])))
                    if rgb_label_count > 0:
                        legend_ax.set_yticks(np.arange(rgb_label_count) + 0.5)
                        legend_ax.set_yticklabels(class_names[:rgb_label_count], fontsize=8)
                    legend_ax.tick_params(axis="x", labelrotation=0, labelsize=8)
                    legend_ax.tick_params(axis="y", labelsize=8)

        if map_coloring == "top1":
            prediction_plot_gdf["plot_color"] = (
                prediction_plot_gdf["class_name"].map(color_lookup).fillna("#34495e")
            )
            prediction_plot_gdf.plot(
                ax=ax,
                color=prediction_plot_gdf["plot_color"],
                alpha=0.6,
                edgecolor="#1f1f1f",
                linewidth=0.35,
            )
        elif map_coloring == "multivariate":
            if "multivariate_color" in prediction_plot_gdf.columns:
                prediction_plot_gdf["plot_color"] = prediction_plot_gdf["multivariate_color"].fillna("#34495e")
            else:
                probability_columns = [
                    f"prob_{index}" for index in range(len(class_names))
                    if f"prob_{index}" in prediction_plot_gdf.columns
                ]
                if probability_columns:
                    prediction_plot_gdf["plot_color"] = prediction_plot_gdf[probability_columns].apply(
                        lambda row: _multivariate_color_from_probabilities(
                            [float(value) for value in row.values.tolist()],
                            color_lookup,
                        ),
                        axis=1,
                    )
                else:
                    prediction_plot_gdf["plot_color"] = (
                        prediction_plot_gdf["class_name"].map(color_lookup).fillna("#34495e")
                    )
            prediction_plot_gdf.plot(
                ax=ax,
                color=prediction_plot_gdf["plot_color"],
                alpha=0.6,
                edgecolor="#1f1f1f",
                linewidth=0.35,
            )
    buffer_gdf.boundary.plot(ax=ax, color="#111111", linewidth=1.0)
    centre_gdf.plot(ax=ax, color="#c0392b", markersize=20, zorder=5)

    if map_coloring != "vba":
        legend_handles = [
            Patch(facecolor=color_lookup[class_name], edgecolor="#1f1f1f", label=class_name, linewidth=0.35)
            for class_name in class_names
        ]
        legend_title = (
            "Street pattern class (top-1)"
            if map_coloring == "top1"
            else "Class palette (cell color = weighted class mix)"
        )
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8,
            title_fontsize=9,
        )

    if map_coloring == "top1":
        suffix = "top-1"
    elif map_coloring == "vba":
        suffix = "vba"
    else:
        suffix = "multivariate"
    ax.set_title(f"{title} ({suffix})")
    ax.set_axis_off()
    if map_coloring == "vba":
        fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.06, hspace=0.15)
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_comparison_map(
    roads_wgs84: gpd.GeoDataFrame,
    buffer_gdf: gpd.GeoDataFrame,
    centre_gdf: gpd.GeoDataFrame,
    comparison_gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))

    if not roads_wgs84.empty:
        roads_wgs84.plot(ax=ax, color="#7f8c8d", linewidth=0.35, alpha=0.6)

    if not comparison_gdf.empty:
        values = comparison_gdf[column].fillna("missing").astype(str).tolist()
        color_lookup = _category_color_lookup(values)
        plot_gdf = comparison_gdf.copy()
        plot_gdf["plot_color"] = plot_gdf[column].fillna("missing").astype(str).map(color_lookup)
        plot_gdf.plot(
            ax=ax,
            color=plot_gdf["plot_color"],
            alpha=0.55,
            edgecolor="#1f1f1f",
            linewidth=0.3,
        )
        legend_handles = [
            Patch(facecolor=color, edgecolor="#1f1f1f", label=label, linewidth=0.35)
            for label, color in color_lookup.items()
        ]
        ax.legend(
            handles=legend_handles,
            title=column,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=8,
            title_fontsize=9,
        )

    buffer_gdf.boundary.plot(ax=ax, color="#111111", linewidth=1.0)
    centre_gdf.plot(ax=ax, color="#c0392b", markersize=20, zorder=5)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_city_outputs(
    city_dir: Path,
    maps_dir: Path | None,
    place: str,
    relation: dict,
    centre_node: dict,
    summary: dict,
    roads_wgs84: gpd.GeoDataFrame,
    buffer_polygon_wgs84,
    prediction_gdf: gpd.GeoDataFrame,
    map_coloring: str = "multivariate",
) -> None:
    city_dir.mkdir(parents=True, exist_ok=True)
    map_filename = f"map_{map_coloring}.png"
    buffer_gdf = build_buffer_gdf(
        place=place,
        relation=relation,
        centre_node=centre_node,
        buffer_polygon_wgs84=buffer_polygon_wgs84,
    )
    centre_gdf = build_centre_gdf(place=place, relation=relation, centre_node=centre_node)

    (city_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    roads_wgs84.to_file(city_dir / "roads.geojson", driver="GeoJSON")
    buffer_gdf.to_file(city_dir / "buffer.geojson", driver="GeoJSON")
    centre_gdf.to_file(city_dir / "centre.geojson", driver="GeoJSON")
    if not prediction_gdf.empty:
        prediction_gdf.to_file(city_dir / "predicted_cells.geojson", driver="GeoJSON")
        prediction_gdf.drop(columns="geometry").to_csv(city_dir / "predicted_cells.csv", index=False)

    render_city_map(
        roads_wgs84=roads_wgs84,
        buffer_gdf=buffer_gdf,
        centre_gdf=centre_gdf,
        prediction_gdf=prediction_gdf,
        title=f"{place} street-pattern predictions",
        output_path=city_dir / map_filename,
        map_coloring=map_coloring,
    )

    if maps_dir is None:
        return

    render_city_map(
        roads_wgs84=roads_wgs84,
        buffer_gdf=buffer_gdf,
        centre_gdf=centre_gdf,
        prediction_gdf=prediction_gdf,
        title=f"{place} street-pattern predictions",
        output_path=maps_dir / f"{_slugify(place)}_{map_coloring}.png",
        map_coloring=map_coloring,
    )


def _snapshot_summary(
    place: str,
    year: int | None,
    network_type: str,
    grid_step: float,
    buffer_m: float,
    device: str,
    no_cache: bool,
    cache_dir: str | None,
    road_source: str,
    relation: dict,
    centre_node: dict,
    subgraphs: dict,
    predictions: dict,
    probabilities: dict,
) -> dict:
    counts = Counter(predictions.values())
    return {
        "place": place,
        "year": year,
        "network_type": network_type,
        "grid_step": grid_step,
        "buffer_m": buffer_m,
        "device": device,
        "no_cache": bool(no_cache),
        "cache_dir": cache_dir,
        "road_source": road_source,
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
                "probabilities": _round_probabilities(probabilities[cell_id]),
            }
            for cell_id, class_id in predictions.items()
        },
    }


def classify_snapshot(
    *,
    place: str,
    year: int | None,
    network_type: str,
    relation: dict,
    centre_node: dict,
    polygon_wgs84,
    polygon_projected,
    local_crs,
    model_path: Path | str,
    grid_step: float,
    buffer_m: float,
    device: str,
    no_cache: bool = False,
    cache_dir: Path | None = None,
    roads_path: Path | None = None,
    road_source_label: str = "osm",
):
    snapshot_label = _year_label(year)
    _progress_log(f"{snapshot_label}: start classification snapshot")
    graph_wgs84 = None
    roads_wgs84 = None
    subgraphs = None
    dataset = None
    predictions = None
    probabilities = None

    roads_cache_path = None
    subgraphs_cache_path = None
    dataset_cache_path = None
    predictions_cache_path = None

    if cache_dir is not None:
        prefix = _cache_prefix(
            cache_dir=cache_dir,
            place=place,
            year=year,
            network_type=network_type,
            grid_step=grid_step,
            buffer_m=buffer_m,
            road_source_label=road_source_label,
            roads_path=roads_path,
            model_path=model_path,
            device=device,
        )
        roads_cache_path = prefix.with_name(prefix.name + "__roads.pkl")
        subgraphs_cache_path = prefix.with_name(prefix.name + "__subgraphs.pkl")
        dataset_cache_path = prefix.with_name(prefix.name + "__dataset.pkl")
        predictions_cache_path = prefix.with_name(prefix.name + "__predictions.pkl")

    if not no_cache and subgraphs_cache_path is not None and subgraphs_cache_path.exists():
        subgraphs = _load_pickle(subgraphs_cache_path)
        _progress_log(f"{snapshot_label}: loaded cached subgraphs")

    if not no_cache and roads_cache_path is not None and roads_cache_path.exists():
        roads_cache = _load_pickle(roads_cache_path)
        if isinstance(roads_cache, dict) and "roads_wgs84" in roads_cache:
            roads_wgs84 = roads_cache["roads_wgs84"]
        elif isinstance(roads_cache, gpd.GeoDataFrame):
            roads_wgs84 = roads_cache
        if roads_wgs84 is not None:
            _progress_log(f"{snapshot_label}: loaded cached roads")

    if subgraphs is None or roads_wgs84 is None:
        _progress_log(f"{snapshot_label}: preparing roads and graph split (this can take time)")
        if roads_path is None:
            _progress_log(f"{snapshot_label}: downloading road graph from OSM")
            graph_wgs84 = download_street_graph(polygon_wgs84, network_type=network_type, year=year)
            _, roads_wgs84 = ox.graph_to_gdfs(graph_wgs84)
            roads_wgs84 = roads_wgs84.reset_index()
            graph_projected = ox.project_graph(graph_wgs84, to_crs=local_crs)
            _progress_log(f"{snapshot_label}: splitting projected graph by grid")
            subgraphs = split_graph_by_grid_for_polygon(
                graph_projected,
                polygon_projected,
                grid_step=grid_step,
            )
        else:
            _progress_log(f"{snapshot_label}: loading roads from local dataset")
            roads_projected, roads_wgs84, _, _ = prepare_city_roads(
                roads_path=roads_path,
                centre_node=centre_node,
                buffer_m=buffer_m,
            )
            _progress_log(f"{snapshot_label}: splitting roads by grid")
            subgraphs = split_roads_by_grid_for_polygon(
                roads_projected,
                polygon_projected,
                grid_step=grid_step,
                min_road_count=0,
                min_total_road_length=0.0,
            )

        if not no_cache and roads_cache_path is not None:
            _save_pickle(roads_cache_path, {"roads_wgs84": roads_wgs84})
        if not no_cache and subgraphs_cache_path is not None:
            _save_pickle(subgraphs_cache_path, subgraphs)

    if not no_cache and dataset_cache_path is not None and dataset_cache_path.exists():
        dataset = _load_pickle(dataset_cache_path)
        _progress_log(f"{snapshot_label}: loaded cached dataset")
    else:
        _progress_log(f"{snapshot_label}: building graph dataset/features")
        dataset = BlockDataset(subgraphs)
        if not no_cache and dataset_cache_path is not None:
            _save_pickle(dataset_cache_path, dataset)

    if not no_cache and predictions_cache_path is not None and predictions_cache_path.exists():
        cached_predictions = _load_pickle(predictions_cache_path)
        if isinstance(cached_predictions, dict):
            predictions = cached_predictions.get("predictions")
            probabilities = cached_predictions.get("probabilities")
            if predictions is not None and probabilities is not None:
                _progress_log(f"{snapshot_label}: loaded cached predictions")

    if predictions is None or probabilities is None:
        _progress_log(f"{snapshot_label}: running model inference")
        predictions, probabilities = classify_blocks(
            dataset,
            model_path=model_path,
            device=device,
        )
        _progress_log(f"{snapshot_label}: inference complete")
        if not no_cache and predictions_cache_path is not None:
            _save_pickle(
                predictions_cache_path,
                {"predictions": predictions, "probabilities": probabilities},
            )

    summary = _snapshot_summary(
        place=place,
        year=year,
        network_type=network_type,
        grid_step=grid_step,
        buffer_m=buffer_m,
        device=device,
        no_cache=no_cache,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        road_source=road_source_label,
        relation=relation,
        centre_node=centre_node,
        subgraphs=subgraphs,
        predictions=predictions,
        probabilities=probabilities,
    )
    prediction_gdf = build_city_prediction_gdf(
        place=place,
        relation=relation,
        centre_node=centre_node,
        subgraphs=subgraphs,
        predictions=predictions,
        probabilities=probabilities,
    )
    subgraph_metrics = summarize_subgraph_metrics(subgraphs)
    _progress_log(f"{snapshot_label}: snapshot done")
    return {
        "year": year,
        "road_source": road_source_label,
        "graph_wgs84": graph_wgs84,
        "roads_wgs84": roads_wgs84.to_crs(4326) if roads_wgs84.crs != "EPSG:4326" else roads_wgs84,
        "subgraphs": subgraphs,
        "predictions": predictions,
        "probabilities": probabilities,
        "prediction_gdf": prediction_gdf,
        "summary": summary,
        "subgraph_metrics": subgraph_metrics,
    }


def _primary_reason(
    class_changed: bool,
    length_delta_m: float,
    road_count_delta: int,
    top_probability_delta: float,
    available_a: bool,
    available_b: bool,
) -> str:
    if available_a and not available_b:
        return "missing_in_compare_year"
    if available_b and not available_a:
        return "new_in_compare_year"

    max_count_delta_major = 2
    max_length_delta_major = 250.0
    max_prob_delta_major = 0.15
    major_network_change = abs(length_delta_m) >= max_length_delta_major or abs(road_count_delta) >= max_count_delta_major

    if class_changed and major_network_change:
        return "class_changed_with_network_change"
    if class_changed:
        return "class_changed_without_large_network_change"
    if major_network_change:
        return "same_class_network_changed"
    if abs(top_probability_delta) >= max_prob_delta_major:
        return "same_class_confidence_shift"
    return "unchanged"


def build_comparison_gdf(
    *,
    place: str,
    year_a: int,
    year_b: int,
    snapshot_a: dict,
    snapshot_b: dict,
) -> gpd.GeoDataFrame:
    rows = []
    keys = sorted(set(snapshot_a["subgraphs"]) | set(snapshot_b["subgraphs"]), key=str)

    for cell_id in keys:
        cell_a = snapshot_a["subgraphs"].get(cell_id)
        cell_b = snapshot_b["subgraphs"].get(cell_id)
        pred_a = snapshot_a["predictions"].get(cell_id)
        pred_b = snapshot_b["predictions"].get(cell_id)
        probs_a = snapshot_a["probabilities"].get(cell_id)
        probs_b = snapshot_b["probabilities"].get(cell_id)
        metrics_a = snapshot_a["subgraph_metrics"].get(cell_id, {})
        metrics_b = snapshot_b["subgraph_metrics"].get(cell_id, {})

        geometry = None
        if cell_b is not None:
            geometry = cell_b["polygon"]
        elif cell_a is not None:
            geometry = cell_a["polygon"]
        if geometry is None:
            continue

        available_a = pred_a is not None
        available_b = pred_b is not None
        class_changed = available_a and available_b and pred_a != pred_b
        class_name_a = class_names[pred_a] if available_a else None
        class_name_b = class_names[pred_b] if available_b else None
        top_prob_a = float(max(_round_probabilities(probs_a))) if available_a else None
        top_prob_b = float(max(_round_probabilities(probs_b))) if available_b else None
        road_length_a = float(metrics_a.get("road_length_m", 0.0))
        road_length_b = float(metrics_b.get("road_length_m", 0.0))
        road_count_a = int(metrics_a.get("road_segment_count", 0))
        road_count_b = int(metrics_b.get("road_segment_count", 0))
        top_probability_delta = round((top_prob_b or 0.0) - (top_prob_a or 0.0), 3)
        road_length_delta = round(road_length_b - road_length_a, 3)
        road_count_delta = road_count_b - road_count_a
        change_reason = _primary_reason(
            class_changed=class_changed,
            length_delta_m=road_length_delta,
            road_count_delta=road_count_delta,
            top_probability_delta=top_probability_delta,
            available_a=available_a,
            available_b=available_b,
        )

        rows.append(
            {
                "place": place,
                "cell_id": str(cell_id),
                "class_changed": bool(class_changed),
                f"class_{year_a}": class_name_a,
                f"class_{year_b}": class_name_b,
                f"class_id_{year_a}": int(pred_a) if available_a else None,
                f"class_id_{year_b}": int(pred_b) if available_b else None,
                f"top_probability_{year_a}": top_prob_a,
                f"top_probability_{year_b}": top_prob_b,
                f"road_length_m_{year_a}": road_length_a,
                f"road_length_m_{year_b}": road_length_b,
                f"road_segment_count_{year_a}": road_count_a,
                f"road_segment_count_{year_b}": road_count_b,
                "top_probability_delta": top_probability_delta,
                "road_length_delta_m": road_length_delta,
                "road_segment_delta": road_count_delta,
                "change_reason": change_reason,
                "classes_before_after": f"{class_name_a} -> {class_name_b}",
                "why": (
                    f"{year_a}: class={class_name_a}, top_prob={top_prob_a}, roads={road_count_a}, length_m={road_length_a}; "
                    f"{year_b}: class={class_name_b}, top_prob={top_prob_b}, roads={road_count_b}, length_m={road_length_b}; "
                    f"delta_prob={top_probability_delta}, delta_length_m={road_length_delta}, delta_roads={road_count_delta}"
                ),
                "geometry": geometry,
            }
        )

    crs = next(iter(snapshot_b["subgraphs"].values()))["graph"].graph.get("crs") if snapshot_b["subgraphs"] else None
    comparison_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    if crs is not None:
        comparison_gdf = comparison_gdf.to_crs(4326)
    return comparison_gdf


def _difference_gdf(roads_a: gpd.GeoDataFrame, roads_b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if roads_a.empty or roads_b.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)

    union_a = unary_union(roads_a.geometry)
    difference = unary_union(roads_b.geometry).difference(union_a)
    if difference.is_empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)

    diff_gdf = gpd.GeoDataFrame({"geometry": [difference]}, geometry="geometry", crs=4326)
    diff_gdf = diff_gdf.explode(index_parts=False, ignore_index=True)
    diff_gdf = diff_gdf[diff_gdf.geometry.notna() & ~diff_gdf.geometry.is_empty].copy()
    return diff_gdf


def save_comparison_outputs(
    *,
    comparison_dir: Path,
    place: str,
    relation: dict,
    centre_node: dict,
    buffer_polygon_wgs84,
    year_a: int,
    year_b: int,
    comparison_summary: dict,
    comparison_gdf: gpd.GeoDataFrame,
    roads_a_wgs84: gpd.GeoDataFrame,
    roads_b_wgs84: gpd.GeoDataFrame,
) -> None:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    buffer_gdf = build_buffer_gdf(place=place, relation=relation, centre_node=centre_node, buffer_polygon_wgs84=buffer_polygon_wgs84)
    centre_gdf = build_centre_gdf(place=place, relation=relation, centre_node=centre_node)

    (comparison_dir / "summary.json").write_text(json.dumps(comparison_summary, ensure_ascii=False, indent=2))
    comparison_gdf.to_file(comparison_dir / "comparison_cells.geojson", driver="GeoJSON")
    comparison_gdf.drop(columns="geometry").to_csv(comparison_dir / "comparison_cells.csv", index=False)
    roads_a_wgs84.to_file(comparison_dir / f"roads_{year_a}.geojson", driver="GeoJSON")
    roads_b_wgs84.to_file(comparison_dir / f"roads_{year_b}.geojson", driver="GeoJSON")

    gpkg_path = comparison_dir / f"{_slugify(place)}_{year_a}_vs_{year_b}.gpkg"
    comparison_gdf[["geometry", "class_changed", "change_reason", "classes_before_after", "why"]].to_file(
        gpkg_path,
        layer="class_changed",
        driver="GPKG",
    )
    comparison_gdf[["geometry", f"class_{year_a}", f"class_id_{year_a}", f"top_probability_{year_a}"]].to_file(
        gpkg_path,
        layer=f"class_{year_a}",
        driver="GPKG",
    )
    comparison_gdf[["geometry", f"class_{year_b}", f"class_id_{year_b}", f"top_probability_{year_b}"]].to_file(
        gpkg_path,
        layer=f"class_{year_b}",
        driver="GPKG",
    )
    roads_a_wgs84.to_file(gpkg_path, layer=f"graph_{year_a}", driver="GPKG")
    roads_b_wgs84.to_file(gpkg_path, layer=f"graph_{year_b}", driver="GPKG")

    roads_added_gdf = _difference_gdf(roads_a_wgs84, roads_b_wgs84)
    roads_removed_gdf = _difference_gdf(roads_b_wgs84, roads_a_wgs84)
    if not roads_added_gdf.empty:
        roads_added_gdf.to_file(comparison_dir / f"roads_added_{year_b}_vs_{year_a}.geojson", driver="GeoJSON")
        roads_added_gdf.to_file(gpkg_path, layer=f"roads_added_{year_b}_vs_{year_a}", driver="GPKG")
    if not roads_removed_gdf.empty:
        roads_removed_gdf.to_file(comparison_dir / f"roads_removed_{year_b}_vs_{year_a}.geojson", driver="GeoJSON")
        roads_removed_gdf.to_file(gpkg_path, layer=f"roads_removed_{year_b}_vs_{year_a}", driver="GPKG")

    render_comparison_map(
        roads_wgs84=roads_b_wgs84,
        buffer_gdf=buffer_gdf,
        centre_gdf=centre_gdf,
        comparison_gdf=comparison_gdf,
        column="class_changed",
        title=f"{place}: changed cells {year_a} vs {year_b}",
        output_path=comparison_dir / "map_changed.png",
    )
    render_comparison_map(
        roads_wgs84=roads_b_wgs84,
        buffer_gdf=buffer_gdf,
        centre_gdf=centre_gdf,
        comparison_gdf=comparison_gdf,
        column="change_reason",
        title=f"{place}: why cells changed {year_a} vs {year_b}",
        output_path=comparison_dir / "map_change_reason.png",
    )


def main() -> None:
    args = parse_args()
    if args.no_cache and hasattr(ox.settings, "use_cache"):
        ox.settings.use_cache = False
    cache_dir = Path(args.cache_dir).resolve()
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_road_source, resolved_roads_path = resolve_roads_source(
        place=args.place,
        road_source=args.road_source,
        roads=args.roads,
    )
    if args.compare_year is not None and args.year is None:
        raise ValueError("Provide --year when using --compare-year.")
    if args.compare_year is not None and args.compare_year == args.year:
        raise ValueError("--compare-year must be different from --year.")
    if resolved_road_source == "local" and (args.year is not None or args.compare_year is not None):
        raise ValueError(
            "Historical --year/--compare-year runs currently require OSM roads. "
            "Use --road-source osm for those runs."
        )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models_dir = EXPERIMENTS_DIR / "models"
    model_path = resolve_model_path(model_path=Path(args.model_path).resolve(), models_dir=models_dir)
    city_slug = _slugify(args.place)
    city_root_dir = output_path.parent / city_slug
    city_root_dir.mkdir(parents=True, exist_ok=True)

    buffer_gdf = None

    total_stages = 4 if args.compare_year is None else 6
    stage_bar = tqdm(total=total_stages, desc="Street-pattern pipeline", unit="stage")

    stage_bar.set_postfix_str("resolve model")
    stage_bar.update(1)

    if args.center_node_id is not None:
        stage_bar.set_postfix_str("resolve centre node by OSM node id")
        api = osm.OsmApi()
        centre_node = _node_get(api, int(args.center_node_id))
        relation = {"id": None}
        try:
            geocoded = ox.geocode_to_gdf(args.place)
            if not geocoded.empty:
                row = geocoded.iloc[0]
                relation_id = row.get("osm_id")
                osm_type = str(row.get("osm_type", "")).lower()
                if relation_id is not None and osm_type == "relation":
                    relation = {"id": int(relation_id)}
        except Exception:
            pass
    else:
        stage_bar.set_postfix_str("resolve city relation and centre node")
        relation, centre_node = resolve_city_centre_node(args.place)
    stage_bar.update(1)

    if args.buffer_m >= 1000:
        buffer_label = f"{args.buffer_m / 1000:.1f}km"
    else:
        buffer_label = f"{int(args.buffer_m)}m"
    stage_bar.set_postfix_str(f"build {buffer_label} buffer")
    polygon = build_buffer_polygon(centre_node, args.buffer_m)
    buffer_gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=4326)
    local_crs = buffer_gdf.estimate_utm_crs() or "EPSG:3857"
    polygon_projected = buffer_gdf.to_crs(local_crs).iloc[0].geometry
    stage_bar.update(1)

    stage_bar.set_postfix_str(f"classify {_year_label(args.year)}")
    primary_snapshot = classify_snapshot(
        place=args.place,
        year=args.year,
        network_type=args.network_type,
        relation=relation,
        centre_node=centre_node,
        polygon_wgs84=polygon,
        polygon_projected=polygon_projected,
        local_crs=local_crs,
        model_path=model_path,
        grid_step=args.grid_step,
        buffer_m=args.buffer_m,
        device=args.device,
        no_cache=args.no_cache,
        cache_dir=cache_dir,
        roads_path=resolved_roads_path,
        road_source_label=resolved_road_source,
    )
    stage_bar.update(1)

    primary_dir = city_root_dir
    if args.year is not None or args.compare_year is not None:
        primary_dir = city_root_dir / f"year_{_year_label(args.year)}"
    save_city_outputs(
        city_dir=primary_dir,
        maps_dir=output_path.parent / "maps",
        place=args.place,
        relation=relation,
        centre_node=centre_node,
        summary=primary_snapshot["summary"],
        roads_wgs84=primary_snapshot["roads_wgs84"],
        buffer_polygon_wgs84=polygon,
        prediction_gdf=primary_snapshot["prediction_gdf"],
        map_coloring=args.map_coloring,
    )

    summary = primary_snapshot["summary"]
    if args.compare_year is not None:
        stage_bar.set_postfix_str(f"classify {args.compare_year}")
        comparison_snapshot = classify_snapshot(
            place=args.place,
            year=args.compare_year,
            network_type=args.network_type,
            relation=relation,
            centre_node=centre_node,
            polygon_wgs84=polygon,
            polygon_projected=polygon_projected,
            local_crs=local_crs,
            model_path=model_path,
            grid_step=args.grid_step,
            buffer_m=args.buffer_m,
            device=args.device,
            no_cache=args.no_cache,
            cache_dir=cache_dir,
            roads_path=resolved_roads_path,
            road_source_label=resolved_road_source,
        )
        stage_bar.update(1)

        comparison_year_dir = city_root_dir / f"year_{args.compare_year}"
        save_city_outputs(
            city_dir=comparison_year_dir,
            maps_dir=output_path.parent / "maps",
            place=args.place,
            relation=relation,
            centre_node=centre_node,
            summary=comparison_snapshot["summary"],
            roads_wgs84=comparison_snapshot["roads_wgs84"],
            buffer_polygon_wgs84=polygon,
            prediction_gdf=comparison_snapshot["prediction_gdf"],
            map_coloring=args.map_coloring,
        )

        stage_bar.set_postfix_str("build comparison outputs")
        comparison_gdf = build_comparison_gdf(
            place=args.place,
            year_a=args.year,
            year_b=args.compare_year,
            snapshot_a=primary_snapshot,
            snapshot_b=comparison_snapshot,
        )
        changed_count = int(comparison_gdf["class_changed"].sum()) if not comparison_gdf.empty else 0
        unchanged_count = int((~comparison_gdf["class_changed"]).sum()) if not comparison_gdf.empty else 0
        reason_counts = (
            comparison_gdf["change_reason"].value_counts(dropna=False).to_dict()
            if not comparison_gdf.empty
            else {}
        )
        transition_counts = (
            comparison_gdf.loc[comparison_gdf["class_changed"], "classes_before_after"].value_counts().to_dict()
            if not comparison_gdf.empty
            else {}
        )
        comparison_summary = {
            "place": args.place,
            "network_type": args.network_type,
            "grid_step": args.grid_step,
            "buffer_m": args.buffer_m,
            "device": args.device,
            "map_coloring": args.map_coloring,
            "no_cache": args.no_cache,
            "cache_dir": str(cache_dir),
            "road_source": resolved_road_source,
            "roads_path": str(resolved_roads_path) if resolved_roads_path is not None else None,
            "relation_id": relation.get("id"),
            "centre_node_id": centre_node.get("id"),
            "centre_node_lon": float(centre_node["lon"]),
            "centre_node_lat": float(centre_node["lat"]),
            "year_a": args.year,
            "year_b": args.compare_year,
            "cells_compared": int(len(comparison_gdf)),
            "cells_changed": changed_count,
            "cells_unchanged": unchanged_count,
            "change_reason_counts": reason_counts,
            "changed_transitions": transition_counts,
            "year_a_summary": primary_snapshot["summary"],
            "year_b_summary": comparison_snapshot["summary"],
        }
        comparison_dir = city_root_dir / f"comparison_{args.year}_vs_{args.compare_year}"
        save_comparison_outputs(
            comparison_dir=comparison_dir,
            place=args.place,
            relation=relation,
            centre_node=centre_node,
            buffer_polygon_wgs84=polygon,
            year_a=args.year,
            year_b=args.compare_year,
            comparison_summary=comparison_summary,
            comparison_gdf=comparison_gdf,
            roads_a_wgs84=primary_snapshot["roads_wgs84"],
            roads_b_wgs84=comparison_snapshot["roads_wgs84"],
        )
        summary = comparison_summary
        stage_bar.update(1)

    stage_bar.close()
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("Classification complete.")
    print(f"Saved summary to {output_path}")
    print(f"Saved city artifacts to {primary_dir}")
    print(f"Saved map to {primary_dir / f'map_{args.map_coloring}.png'}")
    print(f"Road source: {resolved_road_source}")
    if resolved_roads_path is not None:
        print(f"Roads file: {resolved_roads_path}")
    if args.compare_year is not None:
        print(f"Saved comparison artifacts to {city_root_dir / f'comparison_{args.year}_vs_{args.compare_year}'}")
    print(
        f"Used relation {relation.get('id')} and centre node {centre_node.get('id')}"
    )
    if args.compare_year is None:
        print("Class counts:")
        for class_name, count in summary["class_counts"].items():
            print(f"  {class_name}: {count}")
    else:
        print(
            f"Changed cells: {summary['cells_changed']} / {summary['cells_compared']}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from itertools import cycle
from pathlib import Path

import geopandas as gpd
from huggingface_hub import hf_hub_download
import numpy as np
import osmapi as osm
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from tqdm.auto import tqdm


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
CLASSIFIER_DIR = REPO_ROOT / "street-pattern-classifier"

if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from block_dataset import BlockDataset
from classification import classify_blocks
from model import class_names


TOP_10_CANADA_CITIES_2021 = [
    "Toronto",
    "Montreal",
    "Calgary",
    "Ottawa",
    "Edmonton",
    "Winnipeg",
    "Mississauga",
    "Vancouver",
    "Brampton",
    "Hamilton",
]

DISALLOWED_DRIVE_HIGHWAY_VALUES = {
    "abandoned",
    "bridleway",
    "bus_guideway",
    "construction",
    "corridor",
    "cycleway",
    "elevator",
    "escalator",
    "footway",
    "no",
    "path",
    "pedestrian",
    "planned",
    "platform",
    "proposed",
    "raceway",
    "razed",
    "service",
    "steps",
    "track",
}
DISALLOWED_DRIVE_ACCESS_VALUES = {"private"}
DISALLOWED_DRIVE_SERVICE_VALUES = {
    "alley",
    "driveway",
    "emergency_access",
    "parking",
    "parking_aisle",
    "private",
}
DISALLOWED_DRIVE_MOTOR_VALUES = {"no"}
CLASS_COLOR_PALETTE = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]


def _round_probabilities(values, digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the street-pattern classifier on local Canada road data, "
            "restricted to buffers around city centres resolved from OSM."
        )
    )
    parser.add_argument(
        "--roads",
        default=str(
            REPO_ROOT
            / "data_all_cities"
            / "hotosm_can_roads_lines_gpkg"
            / "hotosm_can_roads_lines_gpkg.gpkg"
        ),
        help="Path to the Canada roads GeoPackage.",
    )
    parser.add_argument(
        "--places",
        nargs="+",
        help='City names to process, for example "Montreal" "Toronto".',
    )
    parser.add_argument(
        "--top-10-canada",
        action="store_true",
        help=(
            "Process the 10 largest Canadian municipalities by 2021 Census population: "
            + ", ".join(TOP_10_CANADA_CITIES_2021)
            + "."
        ),
    )
    parser.add_argument(
        "--place-suffix",
        default="Canada",
        help='Suffix appended to every place when geocoding, for example "Canada".',
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=20_000,
        help="Buffer radius around each city centre, in meters.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=2_000,
        help="Grid cell size in projected CRS units.",
    )
    parser.add_argument(
        "--min-road-count",
        type=int,
        default=5,
        help="Skip grid cells with fewer clipped road segments than this.",
    )
    parser.add_argument(
        "--min-total-road-length",
        type=float,
        default=500.0,
        help="Skip grid cells whose total clipped road length is below this threshold.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for subgraph preparation.",
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
        default=str(EXPERIMENTS_DIR / "outputs" / "canada_city_centre_predictions.json"),
        help="Where to save the prediction summary JSON.",
    )
    parser.add_argument(
        "--geo-output",
        default=str(EXPERIMENTS_DIR / "outputs" / "canada_city_centre_predictions.geojson"),
        help="Where to save predicted grid cells as GeoJSON.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(EXPERIMENTS_DIR / "outputs" / "cache" / "canada_city_centre"),
        help="Directory for per-city pickle caches of expensive intermediate results.",
    )
    parser.add_argument(
        "--per-city-dir",
        default=str(EXPERIMENTS_DIR / "outputs" / "canada_city_centre" / "cities"),
        help="Directory for per-city outputs such as GeoJSON, CSV, and JSON summaries.",
    )
    parser.add_argument(
        "--maps-dir",
        default=str(EXPERIMENTS_DIR / "outputs" / "canada_city_centre" / "maps"),
        help="Directory for rendered per-city PNG maps.",
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
        help="Disable reading and writing intermediate pickle caches.",
    )
    args = parser.parse_args()
    if not args.places and not args.top_10_canada:
        parser.error("Provide --places or use --top-10-canada.")
    return args


def _relation_get(api: osm.OsmApi, relation_id: int) -> dict:
    if hasattr(api, "relation_get"):
        return api.relation_get(relation_id)
    return api.RelationGet(relation_id)


def _node_get(api: osm.OsmApi, node_id: int) -> dict:
    if hasattr(api, "node_get"):
        return api.node_get(node_id)
    return api.NodeGet(node_id)


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


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def _cache_prefix(
    cache_dir: Path,
    place_query: str,
    buffer_m: float,
    grid_step: float,
    min_road_count: int,
    min_total_road_length: float,
) -> Path:
    place_slug = _slugify(place_query)
    key = (
        f"{place_slug}"
        f"__buffer_{int(buffer_m)}"
        f"__grid_{int(grid_step)}"
        f"__mincnt_{int(min_road_count)}"
        f"__minlen_{int(min_total_road_length)}"
    )
    return cache_dir / key


def _save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _resolve_places(args: argparse.Namespace) -> list[str]:
    if args.top_10_canada:
        return TOP_10_CANADA_CITIES_2021
    return args.places


def _normalize_lines(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lines = roads[roads.geometry.notna()].copy()
    lines = lines.explode(index_parts=False, ignore_index=True)
    lines = lines[lines.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    normalized_geometries = []
    normalized_rows = []
    for _, row in lines.iterrows():
        geometry = row.geometry
        if isinstance(geometry, LineString):
            parts = [geometry]
        elif isinstance(geometry, MultiLineString):
            parts = list(geometry.geoms)
        else:
            continue

        for part in parts:
            if part.is_empty or len(part.coords) < 2:
                continue
            normalized_rows.append(row.drop(labels="geometry"))
            normalized_geometries.append(part)

    if not normalized_rows:
        return gpd.GeoDataFrame(columns=lines.columns, geometry=[], crs=lines.crs)

    normalized = gpd.GeoDataFrame(normalized_rows, geometry=normalized_geometries, crs=lines.crs)
    normalized["geometry"] = normalized.geometry.map(
        lambda geom: LineString([geom.coords[0], geom.coords[-1]])
        if len(geom.coords) > 2
        else geom
    )
    normalized = normalized[normalized.geometry.length > 0].copy()
    return normalized.reset_index(drop=True)


def _contains_value(value, blocked_values: set[str]) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_value(item, blocked_values) for item in value)

    parts = [part.strip().lower() for part in str(value).split(";") if part.strip()]
    return any(part in blocked_values for part in parts)


def _filter_roads_like_old_osm_pipeline(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    filtered = roads.copy()

    if "area" in filtered.columns:
        filtered = filtered[~filtered["area"].apply(lambda value: _contains_value(value, {"yes"}))].copy()
    if filtered.empty:
        return filtered

    if "highway" in filtered.columns:
        filtered = filtered[
            ~filtered["highway"].apply(
                lambda value: _contains_value(value, DISALLOWED_DRIVE_HIGHWAY_VALUES)
            )
        ].copy()
    if filtered.empty:
        return filtered

    if "access" in filtered.columns:
        filtered = filtered[
            ~filtered["access"].apply(
                lambda value: _contains_value(value, DISALLOWED_DRIVE_ACCESS_VALUES)
            )
        ].copy()
    if filtered.empty:
        return filtered

    if "motor_vehicle" in filtered.columns:
        filtered = filtered[
            ~filtered["motor_vehicle"].apply(
                lambda value: _contains_value(value, DISALLOWED_DRIVE_MOTOR_VALUES)
            )
        ].copy()
    if filtered.empty:
        return filtered

    if "motorcar" in filtered.columns:
        filtered = filtered[
            ~filtered["motorcar"].apply(
                lambda value: _contains_value(value, DISALLOWED_DRIVE_MOTOR_VALUES)
            )
        ].copy()
    if filtered.empty:
        return filtered

    if "service" in filtered.columns:
        filtered = filtered[
            ~filtered["service"].apply(
                lambda value: _contains_value(value, DISALLOWED_DRIVE_SERVICE_VALUES)
            )
        ].copy()

    return filtered


def split_roads_by_grid_for_polygon(
    roads: gpd.GeoDataFrame,
    polygon,
    grid_step: float = 1000,
    min_road_count: int = 0,
    min_total_road_length: float = 0.0,
):
    minx, miny, maxx, maxy = polygon.bounds

    x_coords = np.arange(minx, maxx, grid_step)
    y_coords = np.arange(miny, maxy, grid_step)

    if len(x_coords) == 0 or x_coords[-1] < maxx:
        x_coords = np.append(x_coords, maxx)
    if len(y_coords) == 0 or y_coords[-1] < maxy:
        y_coords = np.append(y_coords, maxy)

    subgraphs = {}
    road_sindex = roads.sindex

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
            candidate_idx = list(road_sindex.intersection(cell_polygon.bounds))
            if not candidate_idx:
                continue

            candidate_roads = roads.iloc[candidate_idx].copy()
            candidate_roads = candidate_roads[candidate_roads.intersects(cell_polygon)].copy()
            if candidate_roads.empty:
                continue

            clipped_roads = candidate_roads.copy()
            clipped_roads["geometry"] = clipped_roads.geometry.intersection(cell_polygon)
            clipped_roads = clipped_roads[clipped_roads.geometry.notna() & ~clipped_roads.geometry.is_empty]
            clipped_roads = _normalize_lines(clipped_roads)
            if clipped_roads.empty:
                continue

            if len(clipped_roads) < min_road_count:
                continue

            total_road_length = float(clipped_roads.geometry.length.sum())
            if total_road_length < min_total_road_length:
                continue

            subgraphs[(i, j)] = {"roads": clipped_roads, "polygon": cell_polygon}

    return subgraphs


def prepare_city_roads(
    roads_path: Path,
    centre_node: dict,
    buffer_m: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Polygon, object]:
    buffer_polygon_wgs84 = build_buffer_polygon(centre_node, buffer_m)

    buffer_gdf = gpd.GeoDataFrame({"geometry": [buffer_polygon_wgs84]}, crs=4326)
    local_crs = buffer_gdf.estimate_utm_crs() or "EPSG:3857"

    roads_subset = gpd.read_file(roads_path, mask=buffer_gdf)
    roads_subset = roads_subset[roads_subset.geometry.notna() & ~roads_subset.geometry.is_empty].copy()
    roads_subset = _filter_roads_like_old_osm_pipeline(roads_subset)
    if roads_subset.empty:
        raise ValueError("No road geometries intersect the city-centre buffer.")

    roads_wgs84 = roads_subset.to_crs(4326) if roads_subset.crs != "EPSG:4326" else roads_subset
    roads_projected = roads_subset.to_crs(local_crs)
    roads_projected = _normalize_lines(roads_projected)
    if roads_projected.empty:
        raise ValueError("Road clipping produced no valid line segments.")

    polygon_projected = buffer_gdf.to_crs(local_crs).iloc[0].geometry
    return roads_projected, roads_wgs84, buffer_polygon_wgs84, polygon_projected


def project_cached_city_roads(
    roads_wgs84: gpd.GeoDataFrame,
    buffer_polygon_wgs84,
) -> tuple[gpd.GeoDataFrame, object]:
    buffer_gdf = gpd.GeoDataFrame({"geometry": [buffer_polygon_wgs84]}, crs=4326)
    local_crs = buffer_gdf.estimate_utm_crs() or "EPSG:3857"

    roads_wgs84 = _filter_roads_like_old_osm_pipeline(roads_wgs84)
    roads_projected = roads_wgs84.to_crs(local_crs)
    roads_projected = _normalize_lines(roads_projected)
    if roads_projected.empty:
        raise ValueError("Road clipping produced no valid line segments.")

    polygon_projected = buffer_gdf.to_crs(local_crs).iloc[0].geometry
    return roads_projected, polygon_projected


def summarize_city(
    place_query: str,
    relation: dict,
    centre_node: dict,
    subgraphs: dict,
    predictions: dict,
    probabilities: dict,
    buffer_m: float,
    grid_step: float,
) -> dict:
    counts = Counter(predictions.values())
    return {
        "place": place_query,
        "buffer_m": buffer_m,
        "grid_step": grid_step,
        "relation_id": relation.get("id"),
        "centre_node_id": centre_node.get("id"),
        "centre_node_lon": float(centre_node["lon"]),
        "centre_node_lat": float(centre_node["lat"]),
        "num_subgraphs": len(subgraphs),
        "num_predictions": len(predictions),
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


def build_city_prediction_gdf(
    place_query: str,
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
            "place": place_query,
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

    if not rows:
        return gpd.GeoDataFrame(
            columns=["place", "cell_id", "class_id", "class_name", "top_probability", "geometry"],
            geometry="geometry",
            crs=None,
        )

    crs = next(iter(subgraphs.values()))["roads"].crs if subgraphs else None
    prediction_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    if crs is not None:
        prediction_gdf = prediction_gdf.to_crs(4326)
    return prediction_gdf


def build_buffer_gdf(
    place_query: str,
    relation: dict,
    centre_node: dict,
    buffer_polygon_wgs84,
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "place": place_query,
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


def build_centre_gdf(place_query: str, relation: dict, centre_node: dict) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "place": place_query,
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
    return {name: color for name, color in zip(class_names, cycle(CLASS_COLOR_PALETTE))}


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
        prediction_gdf = prediction_gdf.copy()
        if map_coloring == "vba":
            try:
                import mapclassify
                from splot.mapping import vba_choropleth, vba_legend
            except Exception:
                prediction_gdf["plot_color"] = prediction_gdf["class_name"].map(color_lookup).fillna("#34495e")
                prediction_gdf.plot(
                    ax=ax,
                    color=prediction_gdf["plot_color"],
                    alpha=0.6,
                    edgecolor="#1f1f1f",
                    linewidth=0.35,
                )
                map_coloring = "top1"
            else:
                rgb_values = prediction_gdf["class_id"].astype(float).to_numpy()
                rgb_bins = mapclassify.UserDefined(rgb_values, bins=list(range(len(class_names))))
                alpha_values = prediction_gdf["top_probability"].astype(float).to_numpy()
                alpha_bins = mapclassify.Quantiles(alpha_values, k=min(5, max(2, len(prediction_gdf))))
                base_colors = [
                    CLASS_COLOR_PALETTE[index % len(CLASS_COLOR_PALETTE)]
                    for index in range(len(class_names))
                ]
                cmap = ListedColormap(base_colors, name="street_pattern_classes")
                vba_choropleth(
                    x_var=prediction_gdf["class_id"].astype(float),
                    y_var=prediction_gdf["top_probability"].astype(float),
                    gdf=prediction_gdf,
                    cmap=cmap,
                    rgb_mapclassify={"classifier": "user_defined", "bins": list(range(len(class_names)))},
                    alpha_mapclassify={
                        "classifier": "quantiles",
                        "k": min(5, max(2, len(prediction_gdf))),
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
                    legend_ax.xaxis.set_label_coords(0.5, -0.14)
                    legend_ax.yaxis.set_label_coords(-0.08, 0.5)
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
            prediction_gdf["plot_color"] = prediction_gdf["class_name"].map(color_lookup).fillna("#34495e")
            prediction_gdf.plot(
                ax=ax,
                color=prediction_gdf["plot_color"],
                alpha=0.6,
                edgecolor="#1f1f1f",
                linewidth=0.35,
            )
        elif map_coloring == "multivariate":
            if "multivariate_color" in prediction_gdf.columns:
                prediction_gdf["plot_color"] = prediction_gdf["multivariate_color"].fillna("#34495e")
            else:
                probability_columns = [
                    f"prob_{index}" for index in range(len(class_names))
                    if f"prob_{index}" in prediction_gdf.columns
                ]
                if probability_columns:
                    prediction_gdf["plot_color"] = prediction_gdf[probability_columns].apply(
                        lambda row: _multivariate_color_from_probabilities(
                            [float(value) for value in row.values.tolist()],
                            color_lookup,
                        ),
                        axis=1,
                    )
                else:
                    prediction_gdf["plot_color"] = prediction_gdf["class_name"].map(color_lookup).fillna("#34495e")
            prediction_gdf.plot(
                ax=ax,
                color=prediction_gdf["plot_color"],
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


def save_city_outputs(
    city_dir: Path,
    maps_dir: Path,
    raw_place: str,
    place_query: str,
    relation: dict,
    centre_node: dict,
    city_summary: dict,
    roads_wgs84: gpd.GeoDataFrame,
    buffer_polygon_wgs84,
    prediction_gdf: gpd.GeoDataFrame,
    map_coloring: str = "multivariate",
) -> None:
    city_dir.mkdir(parents=True, exist_ok=True)
    buffer_gdf = build_buffer_gdf(
        place_query=place_query,
        relation=relation,
        centre_node=centre_node,
        buffer_polygon_wgs84=buffer_polygon_wgs84,
    )
    centre_gdf = build_centre_gdf(place_query=place_query, relation=relation, centre_node=centre_node)

    (city_dir / "summary.json").write_text(json.dumps(city_summary, ensure_ascii=False, indent=2))
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
        title=f"{raw_place} street-pattern predictions",
        output_path=maps_dir / f"{_slugify(raw_place)}_{map_coloring}.png",
        map_coloring=map_coloring,
    )


def resolve_model_path(model_path: Path, models_dir: Path) -> Path:
    if model_path.exists():
        return model_path

    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id="nochka/street-pattern-classifier",
        filename=model_path.name,
        local_dir=str(models_dir),
    )
    return Path(downloaded_path).resolve()


def main() -> None:
    args = parse_args()
    places = _resolve_places(args)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geo_output_path = Path(args.geo_output).resolve()
    geo_output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    per_city_dir = Path(args.per_city_dir).resolve()
    per_city_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = Path(args.maps_dir).resolve()
    maps_dir.mkdir(parents=True, exist_ok=True)
    models_dir = EXPERIMENTS_DIR / "models"
    model_path = Path(args.model_path).resolve()

    roads_path = Path(args.roads).resolve()
    if not roads_path.exists():
        raise FileNotFoundError(f"Road GeoPackage not found: {roads_path}")

    stage_bar = tqdm(total=2 + len(places), desc="Canada street-pattern pipeline", unit="stage")

    stage_bar.set_postfix_str("resolve model")
    model_path = resolve_model_path(model_path=model_path, models_dir=models_dir)
    stage_bar.update(1)

    stage_bar.set_postfix_str("ready")
    stage_bar.update(1)

    city_results = []
    failures = []
    city_prediction_gdfs = []

    for raw_place in places:
        place_query = raw_place if args.place_suffix.strip() == "" else f"{raw_place}, {args.place_suffix}"
        stage_bar.set_postfix_str(f"process {raw_place}")

        try:
            relation, centre_node = resolve_city_centre_node(place_query)
            cache_prefix = _cache_prefix(
                cache_dir=cache_dir,
                place_query=place_query,
                buffer_m=args.buffer_m,
                grid_step=args.grid_step,
                min_road_count=args.min_road_count,
                min_total_road_length=args.min_total_road_length,
            )
            roads_cache_path = cache_prefix.with_name(cache_prefix.name + "__roads.pkl")
            subgraphs_cache_path = cache_prefix.with_name(cache_prefix.name + "__subgraphs.pkl")
            dataset_cache_path = cache_prefix.with_name(cache_prefix.name + "__dataset.pkl")

            if not args.no_cache and roads_cache_path.exists():
                roads_cache = _load_pickle(roads_cache_path)
                required_cache_keys = {"roads_wgs84", "buffer_polygon_wgs84"}
                if required_cache_keys.issubset(roads_cache):
                    roads_wgs84 = roads_cache["roads_wgs84"]
                    buffer_polygon_wgs84 = roads_cache["buffer_polygon_wgs84"]
                    roads_projected, polygon_projected = project_cached_city_roads(
                        roads_wgs84=roads_wgs84,
                        buffer_polygon_wgs84=buffer_polygon_wgs84,
                    )
                else:
                    roads_projected, roads_wgs84, buffer_polygon_wgs84, polygon_projected = prepare_city_roads(
                        roads_path=roads_path,
                        centre_node=centre_node,
                        buffer_m=args.buffer_m,
                    )
                    if not args.no_cache:
                        _save_pickle(
                            roads_cache_path,
                            {
                                "roads_wgs84": roads_wgs84,
                                "buffer_polygon_wgs84": buffer_polygon_wgs84,
                            },
                        )
            else:
                roads_projected, roads_wgs84, buffer_polygon_wgs84, polygon_projected = prepare_city_roads(
                    roads_path=roads_path,
                    centre_node=centre_node,
                    buffer_m=args.buffer_m,
                )
                if not args.no_cache:
                    _save_pickle(
                        roads_cache_path,
                        {
                            "roads_wgs84": roads_wgs84,
                            "buffer_polygon_wgs84": buffer_polygon_wgs84,
                        },
                    )

            if not args.no_cache and subgraphs_cache_path.exists():
                subgraphs = _load_pickle(subgraphs_cache_path)
            else:
                subgraphs = split_roads_by_grid_for_polygon(
                    roads_projected,
                    polygon_projected,
                    grid_step=args.grid_step,
                    min_road_count=args.min_road_count,
                    min_total_road_length=args.min_total_road_length,
                )
                if not subgraphs:
                    raise ValueError("No non-empty subgraphs were produced.")
                if not args.no_cache:
                    _save_pickle(subgraphs_cache_path, subgraphs)

            if not subgraphs:
                raise ValueError("No non-empty subgraphs were produced.")

            if not args.no_cache and dataset_cache_path.exists():
                dataset = _load_pickle(dataset_cache_path)
            else:
                dataset = BlockDataset(subgraphs, workers=args.workers)
                if len(dataset) == 0:
                    raise ValueError("Dataset contains no valid subgraphs.")
                if not args.no_cache:
                    _save_pickle(dataset_cache_path, dataset)

            if len(dataset) == 0:
                raise ValueError("Dataset contains no valid subgraphs.")

            predictions, probabilities = classify_blocks(
                dataset,
                model_path=model_path,
                device=args.device,
            )

            city_summary = summarize_city(
                place_query=place_query,
                relation=relation,
                centre_node=centre_node,
                subgraphs=subgraphs,
                predictions=predictions,
                probabilities=probabilities,
                buffer_m=args.buffer_m,
                grid_step=args.grid_step,
            )
            city_results.append(city_summary)
            city_prediction_gdf = build_city_prediction_gdf(
                place_query=place_query,
                relation=relation,
                centre_node=centre_node,
                subgraphs=subgraphs,
                predictions=predictions,
                probabilities=probabilities,
            )
            if not city_prediction_gdf.empty:
                city_prediction_gdfs.append(city_prediction_gdf)
            save_city_outputs(
                city_dir=per_city_dir / _slugify(raw_place),
                maps_dir=maps_dir,
                raw_place=raw_place,
                place_query=place_query,
                relation=relation,
                centre_node=centre_node,
                city_summary=city_summary,
                roads_wgs84=roads_wgs84,
                buffer_polygon_wgs84=buffer_polygon_wgs84,
                prediction_gdf=city_prediction_gdf,
                map_coloring=args.map_coloring,
            )
        except Exception as exc:
            failures.append({"place": place_query, "error": str(exc)})
        finally:
            stage_bar.update(1)

    stage_bar.close()

    summary = {
        "roads": str(roads_path),
        "device": args.device,
        "map_coloring": args.map_coloring,
        "buffer_m": args.buffer_m,
        "grid_step": args.grid_step,
        "min_road_count": args.min_road_count,
        "min_total_road_length": args.min_total_road_length,
        "workers": args.workers,
        "cache_dir": str(cache_dir),
        "cache_enabled": not args.no_cache,
        "per_city_dir": str(per_city_dir),
        "maps_dir": str(maps_dir),
        "places_requested": places,
        "class_names": class_names,
        "cities_processed": len(city_results),
        "cities_failed": len(failures),
        "results": city_results,
        "failures": failures,
    }

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if city_prediction_gdfs:
        prediction_gdf = gpd.GeoDataFrame(
            pd.concat(city_prediction_gdfs, ignore_index=True),
            geometry="geometry",
            crs=city_prediction_gdfs[0].crs,
        )
        prediction_gdf.to_file(geo_output_path, driver="GeoJSON")
        print(f"Saved predicted cells to {geo_output_path}")
    print(f"Saved summary to {output_path}")
    print(f"Processed cities: {len(city_results)}")
    print(f"Failed cities: {len(failures)}")


if __name__ == "__main__":
    main()

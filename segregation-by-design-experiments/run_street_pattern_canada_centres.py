from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
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
        required=True,
        help='City names to process, for example "Montreal" "Toronto".',
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
        "--no-cache",
        action="store_true",
        help="Disable reading and writing intermediate pickle caches.",
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
) -> tuple[gpd.GeoDataFrame, Polygon, object]:
    buffer_polygon_wgs84 = build_buffer_polygon(centre_node, buffer_m)

    buffer_gdf = gpd.GeoDataFrame({"geometry": [buffer_polygon_wgs84]}, crs=4326)
    local_crs = buffer_gdf.estimate_utm_crs() or "EPSG:3857"

    roads_subset = gpd.read_file(roads_path, mask=buffer_gdf)
    roads_subset = roads_subset[roads_subset.geometry.notna() & ~roads_subset.geometry.is_empty].copy()
    if roads_subset.empty:
        raise ValueError("No road geometries intersect the city-centre buffer.")

    roads_wgs84 = roads_subset.to_crs(4326) if roads_subset.crs != "EPSG:4326" else roads_subset
    roads_projected = roads_subset.to_crs(local_crs)
    roads_projected = _normalize_lines(roads_projected)
    if roads_projected.empty:
        raise ValueError("Road clipping produced no valid line segments.")

    polygon_projected = buffer_gdf.to_crs(local_crs).iloc[0].geometry
    return roads_projected, buffer_polygon_wgs84, polygon_projected


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
    for cell_id, class_id in predictions.items():
        cell_data = subgraphs.get(cell_id)
        if cell_data is None:
            continue

        probability_values = _round_probabilities(probabilities[cell_id])
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
            "top_probability": float(max(probability_values)),
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


def main() -> None:
    args = parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geo_output_path = Path(args.geo_output).resolve()
    geo_output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    models_dir = EXPERIMENTS_DIR / "models"

    roads_path = Path(args.roads).resolve()
    if not roads_path.exists():
        raise FileNotFoundError(f"Road GeoPackage not found: {roads_path}")

    stage_bar = tqdm(total=2 + len(args.places), desc="Canada street-pattern pipeline", unit="stage")

    stage_bar.set_postfix_str("download model")
    model_path = hf_hub_download(
        repo_id="nochka/street-pattern-classifier",
        filename="best_model.pth",
        local_dir=str(models_dir),
    )
    stage_bar.update(1)

    stage_bar.set_postfix_str("ready")
    stage_bar.update(1)

    city_results = []
    failures = []
    city_prediction_gdfs = []

    for raw_place in args.places:
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
                roads_projected = roads_cache["roads_projected"]
                polygon_projected = roads_cache["polygon_projected"]
            else:
                roads_projected, _, polygon_projected = prepare_city_roads(
                    roads_path=roads_path,
                    centre_node=centre_node,
                    buffer_m=args.buffer_m,
                )
                if not args.no_cache:
                    _save_pickle(
                        roads_cache_path,
                        {
                            "roads_projected": roads_projected,
                            "polygon_projected": polygon_projected,
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

            city_results.append(
                summarize_city(
                    place_query=place_query,
                    relation=relation,
                    centre_node=centre_node,
                    subgraphs=subgraphs,
                    predictions=predictions,
                    probabilities=probabilities,
                    buffer_m=args.buffer_m,
                    grid_step=args.grid_step,
                )
            )
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
        except Exception as exc:
            failures.append({"place": place_query, "error": str(exc)})
        finally:
            stage_bar.update(1)

    stage_bar.close()

    summary = {
        "roads": str(roads_path),
        "device": args.device,
        "buffer_m": args.buffer_m,
        "grid_step": args.grid_step,
        "min_road_count": args.min_road_count,
        "min_total_road_length": args.min_total_road_length,
        "workers": args.workers,
        "cache_dir": str(cache_dir),
        "cache_enabled": not args.no_cache,
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

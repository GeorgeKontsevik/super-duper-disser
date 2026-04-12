from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata

DEFAULT_RESIDENTIAL_SHARE = 0.8
DEFAULT_AREA_PER_PERSON_SQM = 20.0


def slugify_place(place: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")
    return slug or "place"


def _load_blocksnet_symbols():
    from blocksnet.blocks.aggregation import aggregate_objects
    from blocksnet.blocks.assignment import assign_land_use
    from blocksnet.blocks.cutting import preprocess_urban_objects, cut_urban_blocks
    from blocksnet.blocks.postprocessing import postprocess_urban_blocks
    from blocksnet.enums import LandUse
    from blocksnet.preprocessing.imputing import impute_buildings

    return {
        "aggregate_objects": aggregate_objects,
        "assign_land_use": assign_land_use,
        "preprocess_urban_objects": preprocess_urban_objects,
        "cut_urban_blocks": cut_urban_blocks,
        "postprocess_urban_blocks": postprocess_urban_blocks,
        "LandUse": LandUse,
        "impute_buildings": impute_buildings,
    }


def _build_land_use_rules(land_use_enum):
    return {
        "residential": land_use_enum.RESIDENTIAL,
        "education": land_use_enum.RESIDENTIAL,
        "commercial": land_use_enum.BUSINESS,
        "retail": land_use_enum.BUSINESS,
        "religious": land_use_enum.BUSINESS,
        "forest": land_use_enum.RECREATION,
        "recreation_ground": land_use_enum.RECREATION,
        "grass": land_use_enum.RECREATION,
        "greenery": land_use_enum.RECREATION,
        "industrial": land_use_enum.INDUSTRIAL,
        "quarry": land_use_enum.INDUSTRIAL,
        "military": land_use_enum.SPECIAL,
        "cemetery": land_use_enum.SPECIAL,
        "landfill": land_use_enum.SPECIAL,
        "farmland": land_use_enum.AGRICULTURE,
        "animal_keeping": land_use_enum.AGRICULTURE,
        "greenhouse_horticulture": land_use_enum.AGRICULTURE,
        "plant_nursery": land_use_enum.AGRICULTURE,
        "vineyard": land_use_enum.AGRICULTURE,
        "allotments": land_use_enum.AGRICULTURE,
        "highway": land_use_enum.TRANSPORT,
        "railway": land_use_enum.TRANSPORT,
        "depot": land_use_enum.TRANSPORT,
    }


BC_TAGS = {
    "roads": {
        "highway": [
            "construction",
            "crossing",
            "living_street",
            "motorway",
            "motorway_link",
            "motorway_junction",
            "pedestrian",
            "primary",
            "primary_link",
            "raceway",
            "residential",
            "road",
            "secondary",
            "secondary_link",
            "services",
            "tertiary",
            "tertiary_link",
            "track",
            "trunk",
            "trunk_link",
            "turning_circle",
            "turning_loop",
            "unclassified",
        ],
        "service": ["living_street", "emergency_access"],
    },
    "railways": {
        "railway": "rail",
    },
    "water": {
        "riverbank": True,
        "reservoir": True,
        "basin": True,
        "dock": True,
        "canal": True,
        "pond": True,
        "natural": ["water", "bay"],
        "waterway": ["river", "canal", "ditch"],
        "landuse": "basin",
        "water": "lake",
    },
    "amenities_floor_context": {
        "amenity": ["parking", "parking_space"],
    },
    "services_pipeline2_raw": {
        "amenity": ["hospital", "clinic", "school", "kindergarten"],
        "healthcare": ["hospital", "clinic", "centre"],
    },
}

IS_LIVING_TAGS = ["residential", "house", "apartments", "detached", "terrace", "dormitory"]
BUILDINGS_OPTIONAL_COLUMNS = [
    "footprint_area",
    "build_floor_area",
    "living_area",
    "non_living_area",
    "population",
]


def _save_geodata(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        prepare_geodata_for_parquet(gdf).to_parquet(path)
    elif path.suffix.lower() == ".gpkg":
        gdf.to_file(path, driver="GPKG")
    else:
        gdf.to_file(path, driver="GeoJSON")


def _ensure_buildings_optional_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    for col in BUILDINGS_OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series(np.nan, index=out.index, dtype="float64")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def _get_boundaries_gdf(place: str) -> gpd.GeoDataFrame:
    return ox.geocode_to_gdf(place).reset_index(drop=True)


def _clip_to_boundary(gdf: gpd.GeoDataFrame, boundary_geom, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    boundary_gdf = gpd.GeoDataFrame({"geometry": [boundary_geom]}, crs=crs)
    if gdf.crs != boundary_gdf.crs:
        boundary_gdf = boundary_gdf.to_crs(gdf.crs)
    clipped = gdf.clip(boundary_gdf)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].reset_index(drop=True)
    return clipped


def _keep_geometry_types(
    gdf: gpd.GeoDataFrame,
    allowed_types: set[str] | tuple[str, ...] | list[str],
    *,
    layer_name: str,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    allowed_types = set(allowed_types)
    geom_types = gdf.geometry.geom_type
    mask = gdf.geometry.notna() & ~gdf.geometry.is_empty & geom_types.isin(allowed_types)
    dropped = int((~mask).sum())
    if dropped:
        logger.warning(
            "[blocksnet-raw] Dropping {} non-{} geometries from layer='{}'",
            dropped,
            ", ".join(sorted(allowed_types)),
            layer_name,
        )
    return gdf.loc[mask].reset_index(drop=True)


def _features_from_polygon_or_empty(boundary_geom, tags: dict, layer_name: str) -> gpd.GeoDataFrame:
    logger.info("[blocksnet-raw] OSM request started: layer='{}'", layer_name)
    started = time.time()
    try:
        gdf = ox.features_from_polygon(boundary_geom, tags)
        logger.info(
            "[blocksnet-raw] OSM request finished: layer='{}', features={}, elapsed={:.1f}s",
            layer_name,
            len(gdf),
            time.time() - started,
        )
        return gdf
    except Exception as exc:
        if exc.__class__.__name__ == "InsufficientResponseError":
            warnings.warn(
                f"[blocksnet_data_pipeline] OSM layer '{layer_name}' is empty for current territory; "
                "continuing with an empty layer.",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning(
                "[blocksnet-raw] OSM layer is empty: layer='{}', elapsed={:.1f}s (continuing)",
                layer_name,
                time.time() - started,
            )
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)
        logger.error(
            "[blocksnet-raw] OSM request failed: layer='{}', elapsed={:.1f}s, error={}",
            layer_name,
            time.time() - started,
            exc,
        )
        raise


def _roads_from_graph_or_empty(boundary_geom) -> gpd.GeoDataFrame:
    logger.info("[blocksnet-raw] OSM request started: layer='roads' (graph)")
    started = time.time()
    try:
        graph = ox.graph_from_polygon(
            boundary_geom,
            network_type="drive",
            retain_all=True,
            truncate_by_edge=True,
        )
        _, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)
        roads_gdf = edges[edges.geom_type.isin(["LineString", "MultiLineString"])].copy()
        roads_gdf = gpd.GeoDataFrame(roads_gdf[["geometry"]].copy(), geometry="geometry", crs=edges.crs)
        logger.info(
            "[blocksnet-raw] OSM request finished: layer='roads' (graph), features={}, elapsed={:.1f}s",
            len(roads_gdf),
            time.time() - started,
        )
        return roads_gdf
    except Exception as exc:
        if exc.__class__.__name__ == "InsufficientResponseError":
            warnings.warn(
                "[blocksnet_data_pipeline] OSM roads graph is empty for current territory; continuing with empty roads.",
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning(
                "[blocksnet-raw] OSM roads graph is empty, elapsed={:.1f}s (continuing)",
                time.time() - started,
            )
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)
        logger.error(
            "[blocksnet-raw] OSM request failed: layer='roads' (graph), elapsed={:.1f}s, error={}",
            time.time() - started,
            exc,
        )
        raise


def _get_urban_objects(boundary_geom):
    roads_gdf = _roads_from_graph_or_empty(boundary_geom)
    railways_gdf = _features_from_polygon_or_empty(boundary_geom, BC_TAGS["railways"], "railways")
    water_gdf = _features_from_polygon_or_empty(boundary_geom, BC_TAGS["water"], "water")

    water_gdf = water_gdf[water_gdf.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])]
    roads_gdf = roads_gdf[roads_gdf.geom_type.isin(["LineString", "MultiLineString"])]
    railways_gdf = railways_gdf[railways_gdf.geom_type.isin(["LineString", "MultiLineString"])]
    return water_gdf.reset_index(drop=True), roads_gdf.reset_index(drop=True), railways_gdf.reset_index(drop=True)


def _get_land_use(boundary_geom) -> gpd.GeoDataFrame:
    zones = _features_from_polygon_or_empty(boundary_geom, tags={"landuse": True}, layer_name="landuse")
    return zones[zones.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index()


def _get_buildings(boundary_geom, crs, impute_buildings):
    buildings_gdf = _features_from_polygon_or_empty(boundary_geom, tags={"building": True}, layer_name="buildings")
    buildings_gdf = buildings_gdf.reset_index(drop=True).to_crs(crs)
    if buildings_gdf.empty:
        return buildings_gdf
    buildings_gdf["is_living"] = buildings_gdf["building"].apply(lambda value: value in IS_LIVING_TAGS)
    buildings_gdf["number_of_floors"] = pd.to_numeric(buildings_gdf.get("building:levels"), errors="coerce")
    buildings_gdf = _ensure_buildings_optional_columns(buildings_gdf)
    return impute_buildings(buildings_gdf)


def _get_buildings_raw(boundary_geom) -> gpd.GeoDataFrame:
    buildings_gdf = _features_from_polygon_or_empty(boundary_geom, tags={"building": True}, layer_name="buildings")
    buildings_gdf = buildings_gdf[buildings_gdf.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    return buildings_gdf


def _get_floor_amenities_raw(boundary_geom) -> gpd.GeoDataFrame:
    amenities_gdf = _features_from_polygon_or_empty(
        boundary_geom,
        tags=BC_TAGS["amenities_floor_context"],
        layer_name="amenities_floor_context",
    )
    amenities_gdf = amenities_gdf[
        amenities_gdf.geom_type.isin(["Polygon", "MultiPolygon"])
    ].reset_index(drop=True)
    return amenities_gdf


def _normalize_raw_osm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    normalized = gdf.reset_index()
    for col in ("element", "id"):
        if col not in normalized.columns:
            normalized[col] = None
    if "element_type" in normalized.columns and "osmid" in normalized.columns:
        normalized["element"] = normalized["element"].fillna(normalized["element_type"])
        normalized["id"] = normalized["id"].fillna(normalized["osmid"])
    normalized["source_uid"] = normalized["element"].astype(str) + ":" + normalized["id"].astype(str)
    normalized = normalized.drop_duplicates(subset=["source_uid"]).reset_index(drop=True)
    return gpd.GeoDataFrame(normalized, geometry="geometry", crs=gdf.crs if gdf.crs is not None else 4326)


def _get_pipeline2_services_raw(boundary_geom) -> gpd.GeoDataFrame:
    services_gdf = _features_from_polygon_or_empty(
        boundary_geom,
        tags=BC_TAGS["services_pipeline2_raw"],
        layer_name="services_pipeline2_raw",
    )
    if services_gdf.empty:
        return services_gdf
    services_gdf = _normalize_raw_osm(services_gdf)
    return services_gdf.reset_index(drop=True)


def _add_population_proxy(
    blocks_gdf: gpd.GeoDataFrame,
    residential_share: float = DEFAULT_RESIDENTIAL_SHARE,
    area_per_person_sqm: float = DEFAULT_AREA_PER_PERSON_SQM,
) -> gpd.GeoDataFrame:
    result = blocks_gdf.copy()

    def _series_or_default(column: str, default: float = 0.0) -> pd.Series:
        if column not in result.columns:
            return pd.Series(default, index=result.index, dtype="float64")
        series = pd.to_numeric(result[column], errors="coerce")
        if not isinstance(series, pd.Series):
            return pd.Series(default, index=result.index, dtype="float64")
        return series.fillna(default)

    living_area = _series_or_default("living_area", 0.0)

    if "living_area" not in result.columns:
        build_floor_area = _series_or_default("build_floor_area", 0.0)
        living_area = build_floor_area * float(residential_share)

    result["living_area_proxy"] = living_area
    result["population_proxy"] = (living_area / float(area_per_person_sqm)).fillna(0.0)
    site_area = _series_or_default("site_area", 0.0)
    site_area = site_area.replace(0, np.nan)
    density_proxy = pd.to_numeric(result["population_proxy"] / site_area, errors="coerce")
    result["density_proxy"] = density_proxy.fillna(0.0)
    return result


def collect_blocksnet_raw_osm_bundle(
    place: str,
    output_dir: str | Path,
    boundary_path: str | Path | None = None,
    roads_override_path: str | Path | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if boundary_path is not None:
        boundaries_gdf = read_geodata(Path(boundary_path))
        if boundaries_gdf.empty:
            raise ValueError(f"Boundary override is empty: {boundary_path}")
    else:
        boundaries_gdf = _get_boundaries_gdf(place)

    boundary_geom = boundaries_gdf.union_all()

    saved_boundary_path = output_path / "boundary.parquet"
    _save_geodata(boundaries_gdf, saved_boundary_path)

    if roads_override_path is not None:
        roads_gdf = read_geodata(Path(roads_override_path))
        if roads_gdf.empty:
            raise ValueError(f"Shared roads override is empty: {roads_override_path}")
        logger.info(
            "[blocksnet-raw] Reusing shared roads layer: {} (features={})",
            Path(roads_override_path).name,
            len(roads_gdf),
        )
        water_gdf = _features_from_polygon_or_empty(boundary_geom, BC_TAGS["water"], "water")
        railways_gdf = _features_from_polygon_or_empty(boundary_geom, BC_TAGS["railways"], "railways")
        water_gdf = water_gdf[water_gdf.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])].reset_index(drop=True)
        roads_gdf = roads_gdf[roads_gdf.geom_type.isin(["LineString", "MultiLineString"])].reset_index(drop=True)
        railways_gdf = railways_gdf[railways_gdf.geom_type.isin(["LineString", "MultiLineString"])].reset_index(drop=True)
    else:
        water_gdf, roads_gdf, railways_gdf = _get_urban_objects(boundary_geom)
    land_use_gdf = _get_land_use(boundary_geom)
    buildings_gdf = _get_buildings_raw(boundary_geom)
    amenities_gdf = _get_floor_amenities_raw(boundary_geom)
    services_pipeline2_raw_gdf = _get_pipeline2_services_raw(boundary_geom)

    water_gdf = _clip_to_boundary(water_gdf, boundary_geom)
    roads_gdf = _clip_to_boundary(roads_gdf, boundary_geom)
    railways_gdf = _clip_to_boundary(railways_gdf, boundary_geom)
    land_use_gdf = _clip_to_boundary(land_use_gdf, boundary_geom)
    buildings_gdf = _clip_to_boundary(buildings_gdf, boundary_geom)
    amenities_gdf = _clip_to_boundary(amenities_gdf, boundary_geom)
    services_pipeline2_raw_gdf = _clip_to_boundary(services_pipeline2_raw_gdf, boundary_geom)

    water_gdf = _keep_geometry_types(
        water_gdf,
        {"LineString", "MultiLineString", "Polygon", "MultiPolygon"},
        layer_name="water",
    )
    roads_gdf = _keep_geometry_types(roads_gdf, {"LineString", "MultiLineString"}, layer_name="roads")
    railways_gdf = _keep_geometry_types(railways_gdf, {"LineString", "MultiLineString"}, layer_name="railways")
    land_use_gdf = _keep_geometry_types(land_use_gdf, {"Polygon", "MultiPolygon"}, layer_name="land_use")
    buildings_gdf = _keep_geometry_types(buildings_gdf, {"Polygon", "MultiPolygon"}, layer_name="buildings")
    amenities_gdf = _keep_geometry_types(amenities_gdf, {"Polygon", "MultiPolygon"}, layer_name="amenities_floor_context")
    services_pipeline2_raw_gdf = _keep_geometry_types(
        services_pipeline2_raw_gdf,
        {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"},
        layer_name="services_pipeline2_raw",
    )

    water_path = output_path / "water.parquet"
    roads_path = output_path / "roads.parquet"
    railways_path = output_path / "railways.parquet"
    land_use_path = output_path / "land_use.parquet"
    buildings_path = output_path / "buildings.parquet"
    amenities_path = output_path / "amenities.parquet"
    services_pipeline2_raw_path = output_path / "services_pipeline2_raw.parquet"

    _save_geodata(water_gdf, water_path)
    _save_geodata(roads_gdf, roads_path)
    _save_geodata(railways_gdf, railways_path)
    _save_geodata(land_use_gdf, land_use_path)
    _save_geodata(buildings_gdf, buildings_path)
    _save_geodata(amenities_gdf, amenities_path)
    _save_geodata(services_pipeline2_raw_gdf, services_pipeline2_raw_path)

    manifest = {
        "place": place,
        "slug": slugify_place(place),
        "boundary_source": str(Path(boundary_path).resolve()) if boundary_path is not None else "osmnx_geocode",
        "roads_source": str(Path(roads_override_path).resolve()) if roads_override_path is not None else "osmnx_drive_graph",
        "files": {
            "boundary": str(saved_boundary_path),
            "water": str(water_path),
            "roads": str(roads_path),
            "railways": str(railways_path),
            "land_use": str(land_use_path),
            "buildings": str(buildings_path),
            "amenities": str(amenities_path),
            "services_pipeline2_raw": str(services_pipeline2_raw_path),
        },
        "counts": {
            "water": len(water_gdf),
            "roads": len(roads_gdf),
            "railways": len(railways_gdf),
            "land_use_polygons": len(land_use_gdf),
            "buildings": len(buildings_gdf),
            "amenities_floor_context": len(amenities_gdf),
            "services_pipeline2_raw": len(services_pipeline2_raw_gdf),
        },
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def build_blocksnet_bundle(
    place: str,
    output_dir: str | Path,
    boundary_path: str | Path | None = None,
    prefetched_layers: dict[str, str] | None = None,
    buildings_override_path: str | Path | None = None,
) -> dict:
    symbols = _load_blocksnet_symbols()
    land_use_rules = _build_land_use_rules(symbols["LandUse"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if boundary_path is not None:
        boundaries_gdf = read_geodata(Path(boundary_path))
        if boundaries_gdf.empty:
            raise ValueError(f"Boundary override is empty: {boundary_path}")
    else:
        boundaries_gdf = _get_boundaries_gdf(place)
    boundary_geom = boundaries_gdf.union_all()
    local_crs = boundaries_gdf.estimate_utm_crs()
    boundaries_local = boundaries_gdf.to_crs(local_crs)

    boundary_path = output_path / "boundary.parquet"
    _save_geodata(boundaries_gdf, boundary_path)

    if prefetched_layers is not None:
        water_gdf = read_geodata(prefetched_layers["water"])
        roads_gdf = read_geodata(prefetched_layers["roads"])
        railways_gdf = read_geodata(prefetched_layers["railways"])
        land_use_gdf = read_geodata(prefetched_layers["land_use"])
        selected_buildings_path = str(buildings_override_path) if buildings_override_path is not None else prefetched_layers["buildings"]
        buildings_gdf = read_geodata(selected_buildings_path)
    else:
        water_gdf, roads_gdf, railways_gdf = _get_urban_objects(boundary_geom)
        land_use_gdf = _get_land_use(boundary_geom)
        buildings_gdf = _get_buildings_raw(boundary_geom)

    water_gdf = _clip_to_boundary(water_gdf, boundary_geom)
    roads_gdf = _clip_to_boundary(roads_gdf, boundary_geom)
    railways_gdf = _clip_to_boundary(railways_gdf, boundary_geom)
    land_use_gdf = _clip_to_boundary(land_use_gdf, boundary_geom)
    buildings_gdf = _clip_to_boundary(buildings_gdf, boundary_geom)

    water_gdf = _keep_geometry_types(
        water_gdf,
        {"LineString", "MultiLineString", "Polygon", "MultiPolygon"},
        layer_name="water",
    )
    roads_gdf = _keep_geometry_types(roads_gdf, {"LineString", "MultiLineString"}, layer_name="roads")
    railways_gdf = _keep_geometry_types(railways_gdf, {"LineString", "MultiLineString"}, layer_name="railways")
    land_use_gdf = _keep_geometry_types(land_use_gdf, {"Polygon", "MultiPolygon"}, layer_name="land_use")
    buildings_gdf = _keep_geometry_types(buildings_gdf, {"Polygon", "MultiPolygon"}, layer_name="buildings")

    water_gdf = water_gdf.to_crs(local_crs)
    roads_gdf = roads_gdf.to_crs(local_crs)
    railways_gdf = railways_gdf.to_crs(local_crs)
    land_use_gdf = land_use_gdf.to_crs(local_crs)
    buildings_gdf = buildings_gdf.to_crs(local_crs)

    _save_geodata(water_gdf, output_path / "water.parquet")
    _save_geodata(roads_gdf, output_path / "roads.parquet")
    _save_geodata(railways_gdf, output_path / "railways.parquet")

    lines_gdf, polygons_gdf = symbols["preprocess_urban_objects"](roads_gdf, railways_gdf, water_gdf)
    blocks_gdf = symbols["cut_urban_blocks"](boundaries_local, lines_gdf, polygons_gdf, buildings_gdf)
    blocks_gdf = symbols["postprocess_urban_blocks"](blocks_gdf)

    land_use_gdf = land_use_gdf.rename(columns={"landuse": "functional_zone"})
    blocks_with_land_use = symbols["assign_land_use"](blocks_gdf, land_use_gdf, land_use_rules)

    if "number_of_floors" not in buildings_gdf.columns:
        from_storey = pd.to_numeric(buildings_gdf.get("storey"), errors="coerce")
        if not isinstance(from_storey, pd.Series):
            from_storey = pd.Series(np.nan, index=buildings_gdf.index, dtype="float64")
        from_levels = pd.to_numeric(buildings_gdf.get("building:levels"), errors="coerce")
        if not isinstance(from_levels, pd.Series):
            from_levels = pd.Series(np.nan, index=buildings_gdf.index, dtype="float64")
        buildings_gdf["number_of_floors"] = from_storey.fillna(from_levels)
    if "is_living" in buildings_gdf.columns:
        buildings_gdf["is_living"] = pd.to_numeric(buildings_gdf["is_living"], errors="coerce").fillna(0).astype(bool)
    elif "building" in buildings_gdf.columns:
        buildings_gdf["is_living"] = buildings_gdf["building"].apply(lambda value: value in IS_LIVING_TAGS)
    buildings_gdf = _ensure_buildings_optional_columns(buildings_gdf)
    buildings_gdf = symbols["impute_buildings"](buildings_gdf)

    aggregated_buildings, _ = symbols["aggregate_objects"](blocks_with_land_use, buildings_gdf)
    building_columns = [
        column
        for column in ["build_floor_area", "footprint_area", "living_area", "non_living_area"]
        if column in aggregated_buildings.columns
    ]
    blocks_final = blocks_with_land_use.join(aggregated_buildings[building_columns])
    blocks_final = _add_population_proxy(blocks_final)
    # Population for downstream provision must be based on post-floor living context.
    # Keep population_total aligned with the newly derived population_proxy.
    population_proxy = pd.to_numeric(blocks_final.get("population_proxy"), errors="coerce")
    if isinstance(population_proxy, pd.Series):
        blocks_final["population_total"] = population_proxy.fillna(0.0).astype(float)
    else:
        blocks_final["population_total"] = 0.0

    blocks_path = output_path / "blocks.parquet"
    land_use_path = output_path / "land_use.parquet"
    buildings_path = output_path / "buildings.parquet"
    _save_geodata(blocks_final, blocks_path)
    _save_geodata(land_use_gdf, land_use_path)
    _save_geodata(buildings_gdf, buildings_path)

    manifest = {
        "place": place,
        "slug": slugify_place(place),
        "boundary_source": str(Path(boundary_path).resolve()) if boundary_path is not None else "osmnx_geocode",
        "crs": str(local_crs),
        "files": {
            "boundary": str(boundary_path),
            "water": str(output_path / "water.parquet"),
            "roads": str(output_path / "roads.parquet"),
            "railways": str(output_path / "railways.parquet"),
            "land_use": str(land_use_path),
            "buildings": str(buildings_path),
            "blocks": str(blocks_path),
        },
        "counts": {
            "blocks": len(blocks_final),
            "buildings": len(buildings_gdf),
            "land_use_polygons": len(land_use_gdf),
        },
        "population_proxy_formula": "living_area_proxy / 20.0; fallback living_area_proxy = build_floor_area * 0.8",
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest

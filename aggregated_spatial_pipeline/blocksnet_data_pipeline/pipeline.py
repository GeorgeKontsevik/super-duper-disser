from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

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
}

IS_LIVING_TAGS = ["residential", "house", "apartments", "detached", "terrace", "dormitory"]


def _save_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def _get_boundaries_gdf(place: str) -> gpd.GeoDataFrame:
    return ox.geocode_to_gdf(place).reset_index(drop=True)


def _get_urban_objects(boundary_geom):
    water_gdf = ox.features_from_polygon(boundary_geom, BC_TAGS["water"])
    roads_gdf = ox.features_from_polygon(boundary_geom, BC_TAGS["roads"])
    railways_gdf = ox.features_from_polygon(boundary_geom, BC_TAGS["railways"])

    water_gdf = water_gdf[water_gdf.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])]
    roads_gdf = roads_gdf[roads_gdf.geom_type.isin(["LineString", "MultiLineString"])]
    railways_gdf = railways_gdf[railways_gdf.geom_type.isin(["LineString", "MultiLineString"])]
    return water_gdf.reset_index(drop=True), roads_gdf.reset_index(drop=True), railways_gdf.reset_index(drop=True)


def _get_land_use(boundary_geom) -> gpd.GeoDataFrame:
    zones = ox.features_from_polygon(boundary_geom, tags={"landuse": True})
    return zones[zones.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index()


def _get_buildings(boundary_geom, crs, impute_buildings):
    buildings_gdf = ox.features_from_polygon(boundary_geom, tags={"building": True})
    buildings_gdf = buildings_gdf.reset_index(drop=True).to_crs(crs)
    buildings_gdf["is_living"] = buildings_gdf["building"].apply(lambda value: value in IS_LIVING_TAGS)
    buildings_gdf["number_of_floors"] = pd.to_numeric(buildings_gdf.get("building:levels"), errors="coerce")
    return impute_buildings(buildings_gdf)


def _add_population_proxy(
    blocks_gdf: gpd.GeoDataFrame,
    residential_share: float = DEFAULT_RESIDENTIAL_SHARE,
    area_per_person_sqm: float = DEFAULT_AREA_PER_PERSON_SQM,
) -> gpd.GeoDataFrame:
    result = blocks_gdf.copy()

    living_area = pd.to_numeric(result.get("living_area"), errors="coerce")
    if living_area is None:
        living_area = pd.Series(0.0, index=result.index)
    living_area = living_area.fillna(0.0)

    if "living_area" not in result.columns:
        build_floor_area = pd.to_numeric(result.get("build_floor_area"), errors="coerce")
        if build_floor_area is None:
            build_floor_area = pd.Series(0.0, index=result.index)
        build_floor_area = build_floor_area.fillna(0.0)
        living_area = build_floor_area * float(residential_share)

    result["living_area_proxy"] = living_area
    result["population_proxy"] = (living_area / float(area_per_person_sqm)).fillna(0.0)
    site_area = pd.to_numeric(result.get("site_area"), errors="coerce")
    if site_area is None:
        site_area = pd.Series(0.0, index=result.index)
    site_area = site_area.replace(0, pd.NA)
    result["density_proxy"] = (result["population_proxy"] / site_area).fillna(0.0)
    return result


def build_blocksnet_bundle(place: str, output_dir: str | Path) -> dict:
    symbols = _load_blocksnet_symbols()
    land_use_rules = _build_land_use_rules(symbols["LandUse"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    boundaries_gdf = _get_boundaries_gdf(place)
    boundary_geom = boundaries_gdf.union_all()
    local_crs = boundaries_gdf.estimate_utm_crs()
    boundaries_local = boundaries_gdf.to_crs(local_crs)

    boundary_path = output_path / "boundary.geojson"
    _save_geojson(boundaries_gdf, boundary_path)

    water_gdf, roads_gdf, railways_gdf = _get_urban_objects(boundary_geom)
    water_gdf = water_gdf.to_crs(local_crs)
    roads_gdf = roads_gdf.to_crs(local_crs)
    railways_gdf = railways_gdf.to_crs(local_crs)

    _save_geojson(water_gdf, output_path / "water.geojson")
    _save_geojson(roads_gdf, output_path / "roads.geojson")
    _save_geojson(railways_gdf, output_path / "railways.geojson")

    lines_gdf, polygons_gdf = symbols["preprocess_urban_objects"](roads_gdf, railways_gdf, water_gdf)
    blocks_gdf = symbols["cut_urban_blocks"](boundaries_local, lines_gdf, polygons_gdf)
    blocks_gdf = symbols["postprocess_urban_blocks"](blocks_gdf)

    land_use_gdf = _get_land_use(boundary_geom).to_crs(local_crs)
    land_use_gdf = land_use_gdf.rename(columns={"landuse": "functional_zone"})
    blocks_with_land_use = symbols["assign_land_use"](blocks_gdf, land_use_gdf, land_use_rules)

    buildings_gdf = _get_buildings(boundary_geom, local_crs, symbols["impute_buildings"])
    aggregated_buildings, _ = symbols["aggregate_objects"](blocks_with_land_use, buildings_gdf)
    building_columns = [
        column
        for column in ["build_floor_area", "footprint_area", "living_area", "non_living_area"]
        if column in aggregated_buildings.columns
    ]
    blocks_final = blocks_with_land_use.join(aggregated_buildings[building_columns])
    blocks_final = _add_population_proxy(blocks_final)

    blocks_path = output_path / "blocks.geojson"
    land_use_path = output_path / "land_use.geojson"
    buildings_path = output_path / "buildings.geojson"
    _save_geojson(blocks_final, blocks_path)
    _save_geojson(land_use_gdf, land_use_path)
    _save_geojson(buildings_gdf, buildings_path)

    manifest = {
        "place": place,
        "slug": slugify_place(place),
        "crs": str(local_crs),
        "files": {
            "boundary": str(boundary_path),
            "water": str(output_path / "water.geojson"),
            "roads": str(output_path / "roads.geojson"),
            "railways": str(output_path / "railways.geojson"),
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

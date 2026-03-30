from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def build_crosswalk(
    source_gdf: gpd.GeoDataFrame,
    target_gdf: gpd.GeoDataFrame,
    source_layer: str,
    target_layer: str,
) -> gpd.GeoDataFrame:
    source_id = f"{source_layer}_id"
    target_id = f"{target_layer}_id"

    source = source_gdf[[source_id, "geometry"]].copy()
    target = target_gdf[[target_id, "geometry"]].copy()

    local_crs = target.estimate_utm_crs() or source.estimate_utm_crs() or "EPSG:3857"
    source_local = source.to_crs(local_crs)
    target_local = target.to_crs(local_crs)

    source_local["source_area"] = source_local.geometry.area
    target_local["target_area"] = target_local.geometry.area

    overlay = gpd.overlay(source_local, target_local, how="intersection", keep_geom_type=False)
    overlay = overlay[overlay.geometry.notna() & ~overlay.geometry.is_empty].copy()
    if overlay.empty:
        return gpd.GeoDataFrame(
            columns=[
                source_id,
                target_id,
                "intersection_area",
                "source_share",
                "target_share",
                "geometry",
            ],
            geometry="geometry",
            crs=local_crs,
        )

    overlay["intersection_area"] = overlay.geometry.area
    overlay["source_share"] = overlay["intersection_area"] / overlay["source_area"]
    overlay["target_share"] = overlay["intersection_area"] / overlay["target_area"]
    overlay["source_layer"] = source_layer
    overlay["target_layer"] = target_layer
    return overlay.reset_index(drop=True)


def save_crosswalk(crosswalk_gdf: gpd.GeoDataFrame, output_path: Path, layer_name: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk_gdf.to_file(output_path, layer=layer_name, driver="GPKG")

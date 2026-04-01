from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet


def load_layer(path: Path, layer_id: str) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Layer {layer_id!r} loaded from {path} is empty.")
    if gdf.geometry.name not in gdf.columns:
        raise ValueError(f"Layer {layer_id!r} from {path} has no geometry column.")
    if gdf.crs is None:
        raise ValueError(f"Layer {layer_id!r} from {path} has no CRS.")
    return with_feature_id(gdf, layer_id)


def with_feature_id(gdf: gpd.GeoDataFrame, layer_id: str) -> gpd.GeoDataFrame:
    feature_id = f"{layer_id}_id"
    if feature_id in gdf.columns:
        return gdf

    result = gdf.copy()
    result[feature_id] = [f"{layer_id}_{idx}" for idx in range(len(result))]
    return result


def save_layer(gdf: gpd.GeoDataFrame, path: Path, layer_name: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        prepare_geodata_for_parquet(gdf).to_parquet(path)
        return
    if path.suffix.lower() == ".gpkg":
        gdf.to_file(path, layer=layer_name or "layer", driver="GPKG")
        return
    gdf.to_file(path, driver="GeoJSON")

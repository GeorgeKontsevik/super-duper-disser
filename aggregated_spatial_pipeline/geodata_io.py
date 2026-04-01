from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import geopandas as gpd
import pandas as pd


def read_geodata(path: str | Path) -> gpd.GeoDataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return gpd.read_parquet(source)
    return gpd.read_file(source)


def prepare_geodata_for_parquet(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    safe = gdf.copy()

    def _normalize(value):
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (list, tuple, set)):
            return json.dumps(list(value), ensure_ascii=False)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return value

    for col in [c for c in safe.columns if c != "geometry"]:
        if pd.api.types.is_object_dtype(safe[col]):
            safe[col] = safe[col].map(_normalize)
            non_null = safe[col].dropna()
            if not non_null.empty:
                python_types = {type(value) for value in non_null}
                if len(python_types) > 1:
                    safe[col] = safe[col].map(lambda value: None if pd.isna(value) else str(value))
    return safe


def ensure_parquet_from_geojson(path: str | Path) -> bool:
    target = Path(path)
    if target.suffix.lower() != ".parquet":
        return False
    if target.exists():
        return True
    geojson = target.with_suffix(".geojson")
    if not geojson.exists():
        return False
    gdf = gpd.read_file(geojson)
    target.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(target)
    return True

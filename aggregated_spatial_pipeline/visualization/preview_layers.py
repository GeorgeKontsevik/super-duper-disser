from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from aggregated_spatial_pipeline.geodata_io import read_geodata

from .map_canvas import normalize_preview_gdf
from .preview_theme import WATER_EDGE, WATER_FILL, WATER_LINE


def load_city_water_layer(city_dir: Path) -> gpd.GeoDataFrame | None:
    candidates = [
        city_dir / "blocksnet" / "water.parquet",
        city_dir / "blocksnet_raw_osm" / "water.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                water = read_geodata(candidate)
            except Exception:
                continue
            if water is not None and not water.empty:
                return water
    return None


def plot_water_layer(
    ax,
    water_layer: gpd.GeoDataFrame | None,
    *,
    boundary_layer: gpd.GeoDataFrame | None = None,
    target_crs: str = "EPSG:3857",
    polygon_zorder: int = 0,
    line_zorder: int = 1,
) -> gpd.GeoDataFrame | None:
    water_plot = normalize_preview_gdf(water_layer, boundary_layer, target_crs=target_crs)
    if water_plot is None or water_plot.empty:
        return water_plot

    polygon_mask = water_plot.geom_type.astype(str).isin(["Polygon", "MultiPolygon"])
    line_mask = water_plot.geom_type.astype(str).isin(["LineString", "MultiLineString"])

    polygons = water_plot[polygon_mask]
    lines = water_plot[line_mask]
    if not polygons.empty:
        polygons.plot(
            ax=ax,
            color=WATER_FILL,
            edgecolor=WATER_EDGE,
            linewidth=0.5,
            alpha=0.95,
            zorder=polygon_zorder,
        )
    if not lines.empty:
        lines.plot(
            ax=ax,
            color=WATER_LINE,
            linewidth=1.0,
            alpha=0.85,
            zorder=line_zorder,
        )
    return water_plot

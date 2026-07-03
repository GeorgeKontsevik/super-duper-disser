#!/usr/bin/env python3
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from render_service_access_diagnostics_service_sheets import (
    CITY_ORDER,
    CITY_RU,
    LABEL_COLORS,
    LABEL_ORDER,
    LABEL_RU,
    PLOT_ORDER,
    _load_city_layers,
    _plot_boundary,
)


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/_all_home_to_service_access_diagnostics.parquet"
JOINT_INPUTS = ROOT / "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"
OUT_DIR = ROOT / "aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/maps_by_service_ru"


def render(diagnostics, layers, rows, cols, figsize, output):
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=220)
    axes_flat = axes.ravel()

    for ax, city in zip(axes_flat, CITY_ORDER):
        points, boundary = layers[city]
        _plot_boundary(ax, boundary)
        sub = diagnostics[(diagnostics["city"] == city) & (diagnostics["service_name"] == "polyclinic")]
        gdf = gpd.GeoDataFrame(
            points.merge(sub[["building_idx", "access_diagnosis_label"]], on="building_idx", how="inner"),
            geometry="geometry",
            crs=points.crs,
        )
        for label in PLOT_ORDER:
            pts = gdf[gdf["access_diagnosis_label"] == label]
            if not pts.empty:
                ax.scatter(pts.geometry.x, pts.geometry.y, s=4.8, c=LABEL_COLORS[label], alpha=0.72,
                           linewidths=0, rasterized=True)
        ax.set_title(CITY_RU[city], fontsize=15, pad=6)

    for ax in axes_flat[len(CITY_ORDER):]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LABEL_COLORS[label],
               markeredgecolor="none", markersize=9, label=LABEL_RU[label])
        for label in LABEL_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=14,
               columnspacing=1.7, handletextpad=0.6, borderaxespad=0)
    fig.suptitle("Диагностика доступности: поликлиники", fontsize=25, y=0.988)
    fig.subplots_adjust(left=0.018, right=0.99, top=0.91, bottom=0.15, wspace=0.10, hspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


diagnostics = pd.read_parquet(
    DIAGNOSTICS,
    columns=["city", "service_name", "building_idx", "access_diagnosis_label"],
)
layers = _load_city_layers(JOINT_INPUTS, CITY_ORDER)
render(diagnostics, layers, 2, 10, (30, 8.8), OUT_DIR / "02_polikliniki_access_diagnostics_ru_2rows.png")
render(diagnostics, layers, 3, 7, (24, 11.5), OUT_DIR / "02_polikliniki_access_diagnostics_ru_3rows.png")

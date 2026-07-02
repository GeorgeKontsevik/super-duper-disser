from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "itmo-phd-thesis-template-en" / "thesis_repro" / "ch4_lbr_country_inputs" / "data" / "lbr_country_inputs.gpkg"
OUT = ROOT / "itmo-phd-thesis-template-en" / "images" / "ch4" / "lbr_inputs_panel_matched.png"
TMP = ROOT / "tmp" / "lbr_inputs_panel_matched.png"

CROPS = [
    ("авокадо", "avocado"),
    ("банан", "banana"),
    ("плантан", "plantain"),
    ("манго", "mango"),
    ("ананас", "pineapple"),
]

DESTINATIONS = [
    ("дороги", "roads", "#d2692c", "-"),
    ("города 5–100 тыс.", "cities_5_100k", "#6f4bc4", "o"),
    ("города 100 тыс.+", "cities_100k_plus", "#f39c34", "s"),
    ("порты", "ports", "#0c6b79", "P"),
    ("аэропорты", "airports", "#2ca25f", "^"),
]


def read_layer(layer: str) -> gpd.GeoDataFrame:
    return gpd.read_file(DATA, layer=layer).to_crs("EPSG:4326")


def padded_bounds(boundary: gpd.GeoDataFrame, pad_ratio: float = 0.05) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = max(maxx - minx, maxy - miny) * pad_ratio
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def draw_base(ax: plt.Axes, boundary: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]) -> None:
    boundary.boundary.plot(ax=ax, color="#777777", linewidth=1.0, zorder=1)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.9)


def main() -> None:
    boundary = read_layer("boundary")
    preview = read_layer("crop_preview_cells")
    clusters = read_layer("crop_cluster_nodes")
    roads = read_layer("road_edges")
    city_5_100 = read_layer("cities_5_100k")
    city_100 = read_layer("cities_100k_plus")
    ports = read_layer("ports")
    airports = read_layer("airports")
    bounds = padded_bounds(boundary)

    fig, axes = plt.subplots(2, 5, figsize=(18, 10.8))
    fig.suptitle(
        "Либерия: культуры, дорожная сеть и точки назначения",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )
    crop_handles = [
        Line2D([0], [0], marker=".", color="none", markerfacecolor="#7a7a7a", markeredgecolor="#7a7a7a", alpha=0.35, markersize=10, label="ячейки"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#d7191c", markersize=9, label="кластер"),
        Line2D([0], [0], marker="x", color="#2c7bb6", linestyle="none", markersize=8, label="узел"),
        Line2D([0], [0], color="#111111", linewidth=0.8, alpha=0.55, label="связь"),
    ]
    for ax, (title, crop) in zip(axes[0], CROPS, strict=True):
        draw_base(ax, boundary, bounds)
        raw_sub = preview[preview["crop_code"].eq(crop)]
        if not raw_sub.empty:
            sizes = np.clip(np.sqrt(raw_sub["harvested_area"].to_numpy(dtype=float)), 3, 20)
            ax.scatter(raw_sub.geometry.x, raw_sub.geometry.y, c="#7a7a7a", s=sizes, alpha=0.22, linewidths=0, zorder=2)

        cluster_sub = clusters[clusters["crop_code"].eq(crop)]
        if not cluster_sub.empty:
            for row in cluster_sub.itertuples(index=False):
                if np.isfinite(row.node_lon) and np.isfinite(row.node_lat):
                    ax.plot(
                        [row.representative_lon, row.node_lon],
                        [row.representative_lat, row.node_lat],
                        color="#111111",
                        linewidth=0.45,
                        alpha=0.5,
                        zorder=3,
                    )
            cluster_sizes = np.clip(np.sqrt(cluster_sub["harvested_area"].to_numpy(dtype=float)), 38, 160)
            ax.scatter(
                cluster_sub["representative_lon"],
                cluster_sub["representative_lat"],
                s=cluster_sizes,
                facecolors="none",
                edgecolors="#d7191c",
                linewidths=1.4,
                zorder=4,
            )
            node_sub = cluster_sub.dropna(subset=["node_lon", "node_lat"])
            if not node_sub.empty:
                ax.scatter(node_sub["node_lon"], node_sub["node_lat"], s=24, c="#2c7bb6", marker="x", linewidths=1.0, zorder=5)

        ax.set_title(title, fontsize=16, fontweight="bold", color="#555555", pad=8)
        if ax is axes[0, 2]:
            ax.legend(handles=crop_handles, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=13, handlelength=0.9, columnspacing=0.6)

    road_colors = {"paved": "#377dde", "unpaved": "#d2692c", "unknown": "#b8b8b8"}
    destination_frames = {
        "cities_5_100k": city_5_100,
        "cities_100k_plus": city_100,
        "ports": ports,
        "airports": airports,
    }
    for ax, (title, layer_name, color, marker) in zip(axes[1], DESTINATIONS, strict=True):
        draw_base(ax, boundary, bounds)
        if layer_name == "roads":
            surface = roads["surface_group"].fillna("unknown")
            for group, lw, alpha in [("unknown", 0.22, 0.55), ("unpaved", 0.32, 0.85), ("paved", 0.60, 0.90)]:
                sub = roads[surface.eq(group)]
                if not sub.empty:
                    sub.plot(ax=ax, color=road_colors[group], linewidth=lw, alpha=alpha, zorder=2)
            handles = [
                Line2D([0], [0], color=road_colors["paved"], linewidth=2.2, label="асфальт"),
                Line2D([0], [0], color=road_colors["unpaved"], linewidth=2.2, label="грунт"),
                Line2D([0], [0], color=road_colors["unknown"], linewidth=2.2, label="н/д"),
            ]
            ncol = 3
        else:
            layer = destination_frames[layer_name]
            if not layer.empty:
                if layer_name == "cities_5_100k":
                    sizes = np.clip(np.sqrt(layer["population"].fillna(5000).to_numpy(float)) / 3.0, 28, 110)
                elif layer_name == "cities_100k_plus":
                    sizes = np.clip(np.sqrt(layer["population"].fillna(100000).to_numpy(float)) / 9.0, 55, 125)
                else:
                    sizes = 58
                ax.scatter(layer.geometry.x, layer.geometry.y, s=sizes, c=color, marker=marker, edgecolors="#333333", linewidths=0.5, alpha=0.95, zorder=4)
            handles = [Line2D([], [], marker=marker, linestyle="None", markerfacecolor=color, markeredgecolor="#333333", markersize=9, label=title)]
            ncol = 1
        ax.set_title(title, fontsize=16, fontweight="bold", color="#555555", pad=8)
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=False, fontsize=13, handlelength=1.2, columnspacing=0.5)

    plt.subplots_adjust(left=0.02, right=0.995, top=0.92, bottom=0.075, wspace=0.02, hspace=0.28)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    TMP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    fig.savefig(TMP, dpi=220)
    plt.close(fig)
    for path in (OUT, TMP):
        assert path.exists() and path.stat().st_size > 50_000, path
        print(f"{path} | {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

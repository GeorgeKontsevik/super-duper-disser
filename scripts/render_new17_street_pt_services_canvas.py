#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregated_spatial_pipeline.visualization import (  # noqa: E402
    apply_preview_canvas,
    get_palette,
    load_city_water_layer,
    normalize_preview_gdf,
    plot_water_layer,
    save_preview_figure,
)
from aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters import CLASS_LABELS  # noqa: E402


NEW17_CITY_ORDER = [
    "adelaide_south_australia_australia",
    "amsterdam_netherlands",
    "arequipa_peru",
    "delft_netherlands",
    "hai_duong_h_i_d_ng_vietnam",
    "huainan_anhui_china",
    "jaynagar_bih_r_india",
    "kakogawacho_honmachi_hy_go_japan",
    "kananga_kasa_central_congo_kinshasa",
    "maracay_aragua_venezuela",
    "montes_claros_minas_gerais_brazil",
    "naihati_west_bengal_india",
    "narayanganj_dhaka_bangladesh",
    "nouakchott_nouakchott_ouest_mauritania",
    "spring_valley_nevada_united_states",
    "temuco_araucan_a_chile",
    "vologda_russia",
]

ACTIVE19_EUROPE_CITY_ORDER = [
    "bergen_norway",
    "bologna_italy",
    "bristol_united_kingdom",
    "brno_czechia",
    "coimbra_portugal",
    "debrecen_hungary",
    "dresden_germany",
    "freiburg_im_breisgau_germany",
    "gothenburg_sweden",
    "graz_austria",
    "innsbruck_austria",
    "krakow_poland",
    "linz_austria",
    "lyon_france",
    "marseille_france",
    "porto_portugal",
    "turin_italy",
    "turku_finland",
    "zaragoza_spain",
]

ALLOWED_PT_TYPES = {
    "bus",
    "tram",
    "subway",
    "train",
    "rail",
    "light_rail",
    "trolleybus",
    "ferry",
}


def _read_gdf(path: Path | None) -> gpd.GeoDataFrame | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    if gdf is None or gdf.empty:
        return None
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf if not gdf.empty else None


def _resolve_city_dir(city: str) -> Path:
    candidates = [
        ROOT / "aggregated_spatial_pipeline" / "outputs" / "experiments_new17_access_20260610" / "joint_inputs_merged" / city,
        ROOT / "aggregated_spatial_pipeline" / "outputs" / "active_19_good_cities_20260412" / "joint_inputs" / city,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing city bundle for {city}")


def _resolve_street_cells(city_dir: Path) -> Path:
    direct = city_dir / "street_pattern" / city_dir.name / "predicted_cells.geojson"
    if direct.exists():
        return direct
    candidates = sorted((city_dir / "street_pattern").glob("*/predicted_cells.geojson"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing predicted_cells.geojson for {city_dir.name}")


def _street_pattern_label(cells: gpd.GeoDataFrame) -> pd.Series:
    if "top1_class_name" in cells.columns:
        return cells["top1_class_name"].fillna("UNKNOWN").astype(str)
    if "class_name" in cells.columns:
        return cells["class_name"].fillna("UNKNOWN").astype(str)
    prob_cols = [c for c in CLASS_LABELS if c in cells.columns]
    if prob_cols:
        top = cells[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).idxmax(axis=1)
        return top.map(CLASS_LABELS).fillna("UNKNOWN").astype(str)
    return pd.Series(["UNKNOWN"] * len(cells), index=cells.index, dtype=object)


def _load_pt_edges(city_dir: Path) -> gpd.GeoDataFrame | None:
    edges = _read_gdf(city_dir / "intermodal_graph_iduedu" / "graph_edges.parquet")
    if edges is None or edges.empty:
        return None
    if "type" not in edges.columns:
        return None
    edge_type = edges["type"].astype("string").str.lower()
    mask = edge_type.isin(sorted(ALLOWED_PT_TYPES))
    if "route" in edges.columns:
        route_text = edges["route"].astype("string").fillna("").str.strip()
        mask &= route_text.ne("")
    edges = edges[mask].copy()
    edges = edges[edges.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    return edges if not edges.empty else None


def _load_services(city_dir: Path) -> gpd.GeoDataFrame | None:
    return _read_gdf(city_dir / "pipeline_2" / "services_raw" / "polyclinic.parquet")


def _pretty_city(city: str) -> str:
    parts = city.split("_")
    if len(parts) > 5:
        parts = parts[:5]
    return " ".join(parts).title()


def _nice_scale_length_m(span_m: float) -> int:
    candidates = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    target = max(span_m * 0.22, 100.0)
    for cand in candidates:
        if cand >= target:
            return cand
    return candidates[-1]


def _draw_scale_bar(ax) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    span_x = xmax - xmin
    span_y = ymax - ymin
    if span_x <= 0 or span_y <= 0:
        return
    length_m = _nice_scale_length_m(span_x)
    x0 = xmin + span_x * 0.06
    y0 = ymin + span_y * 0.08
    ax.plot([x0, x0 + length_m], [y0, y0], color="#111827", lw=2.8, zorder=10, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - span_y * 0.008, y0 + span_y * 0.008], color="#111827", lw=2.0, zorder=10)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - span_y * 0.008, y0 + span_y * 0.008], color="#111827", lw=2.0, zorder=10)
    label = f"{int(length_m / 1000)} km" if length_m >= 1000 else f"{int(length_m)} m"
    txt = ax.text(
        x0 + length_m / 2,
        y0 + span_y * 0.018,
        label,
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#111827",
        zorder=11,
    )
    txt.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])


def _draw_bottom_label(ax, city: str) -> None:
    txt = ax.text(
        0.5,
        -0.055,
        _pretty_city(city),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#1f2937",
        wrap=True,
    )
    txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])


def _render_city(ax, city_dir: Path, *, palette: dict[str, str]) -> dict[str, int]:
    city = city_dir.name
    boundary = normalize_preview_gdf(_read_gdf(city_dir / "analysis_territory" / "buffer.parquet"), target_crs="EPSG:3857")
    cells = _read_gdf(_resolve_street_cells(city_dir))
    if cells is None or cells.empty:
        raise ValueError(f"No street-pattern cells for {city}")
    cells = cells[cells.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    cells["pattern_label"] = _street_pattern_label(cells)
    cells["pattern_color"] = cells["pattern_label"].map(lambda v: palette.get(str(v), palette.get("unknown", "#d1d5db")))
    cells_plot = normalize_preview_gdf(cells, boundary, target_crs="EPSG:3857")

    pt_edges = normalize_preview_gdf(_load_pt_edges(city_dir), boundary, target_crs="EPSG:3857")
    services = normalize_preview_gdf(_load_services(city_dir), boundary, target_crs="EPSG:3857")
    water = load_city_water_layer(city_dir)

    apply_preview_canvas(fig=ax.figure, ax=ax, boundary_layer=boundary, title=None)
    if cells_plot is not None and not cells_plot.empty:
        cells_plot.plot(
            ax=ax,
            color=cells_plot["pattern_color"].astype(str),
            linewidth=0.03,
            edgecolor="#cbd5e1",
            alpha=0.86,
            zorder=1,
        )
    plot_water_layer(ax, water, boundary_layer=boundary, target_crs="EPSG:3857", polygon_zorder=2, line_zorder=2)
    if pt_edges is not None and not pt_edges.empty:
        pt_edges.plot(
            ax=ax,
            color="#0f172a",
            linewidth=0.6,
            alpha=0.72,
            zorder=3,
        )
    if services is not None and not services.empty:
        services = services[services.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
        if not services.empty:
            services.plot(
                ax=ax,
                color="#fde68a",
                markersize=22,
                alpha=0.95,
                edgecolor="white",
                linewidth=0.25,
                zorder=4,
            )
            services.plot(
                ax=ax,
                color="#b91c1c",
                markersize=11,
                alpha=1.0,
                edgecolor="#7f1d1d",
                linewidth=0.35,
                zorder=5,
            )
    _draw_scale_bar(ax)
    _draw_bottom_label(ax, city)
    ax.set_axis_off()
    return {
        "pt_edges": 0 if pt_edges is None else int(len(pt_edges)),
        "services": 0 if services is None else int(len(services)),
        "cells": int(len(cells_plot)) if cells_plot is not None else 0,
    }


def main() -> None:
    city_order = NEW17_CITY_ORDER + ACTIVE19_EUROPE_CITY_ORDER
    out_root = ROOT / "segregation-by-design-experiments" / "polyclinic_access_components" / "outputs_new17_20260610"
    out_png = out_root / "new17_plus_active19europe_street_pattern_pt_services_canvas.png"
    out_csv = out_root / "new17_plus_active19europe_street_pattern_pt_services_canvas_summary.csv"

    n = len(city_order)
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(23, 5.2 * nrows))
    axes = axes.flatten()
    palette = get_palette("street_patterns")
    rows: list[dict[str, int | str]] = []

    for idx, city in enumerate(city_order):
        city_dir = _resolve_city_dir(city)
        stats = _render_city(axes[idx], city_dir, palette=palette)
        rows.append({"city": city, **stats})

    for ax in axes[n:]:
        ax.set_axis_off()

    legend_items = [
        Patch(facecolor=palette.get("Loops & Lollipops", "#d1d5db"), edgecolor="none", label="Loops & Lollipops"),
        Patch(facecolor=palette.get("Irregular Grid", "#d1d5db"), edgecolor="none", label="Irregular Grid"),
        Patch(facecolor=palette.get("Regular Grid", "#d1d5db"), edgecolor="none", label="Regular Grid"),
        Patch(facecolor=palette.get("Warped Parallel", "#d1d5db"), edgecolor="none", label="Warped Parallel"),
        Patch(facecolor=palette.get("Broken Grid", "#d1d5db"), edgecolor="none", label="Broken Grid"),
        Patch(facecolor=palette.get("Sparse", "#d1d5db"), edgecolor="none", label="Sparse"),
        Line2D([0], [0], color="#0f172a", lw=2.0, alpha=0.75, label="PT network"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b91c1c", markeredgecolor="#7f1d1d", markersize=8, label="Polyclinic"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("New 17 Plus Active19 Europe: Street Pattern, PT Network, And Polyclinic Services", fontsize=21, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    save_preview_figure(fig, out_png, dpi=220)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(out_png)
    print(out_csv)


if __name__ == "__main__":
    main()

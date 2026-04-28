from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters import (
    CLASS_COLORS,
    CLASS_LABELS,
)
from aggregated_spatial_pipeline.visualization import (
    CANVAS_BACKGROUND,
    CANVAS_GRID,
    CANVAS_INK,
    CANVAS_MUTED,
    LEGEND_EDGE,
    LEGEND_FACE,
    apply_preview_canvas,
    footer_text,
    legend_bottom,
    load_city_water_layer,
    normalize_preview_gdf,
    order_street_pattern_classes,
    plot_water_layer,
    save_preview_figure,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    REPO_ROOT
    / "yana_experiments"
    / "street_pattern_route_comparison"
    / "bergen_norway"
    / "bus_existing_count_meanmax_033_n17_len9_25"
)
OUTPUT_DIR = RUN_DIR / "preview_png_segregated"


def _chart_style(ax) -> None:
    ax.set_facecolor(LEGEND_FACE)
    ax.tick_params(colors=CANVAS_INK, labelsize=9)
    ax.xaxis.label.set_color(CANVAS_INK)
    ax.yaxis.label.set_color(CANVAS_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(CANVAS_GRID)
    ax.spines["bottom"].set_color(CANVAS_GRID)
    ax.grid(axis="x", color=CANVAS_GRID, alpha=0.55, linewidth=0.7)
    ax.set_axisbelow(True)


def _title(ax, text: str) -> None:
    ax.set_title(text, fontsize=15, fontweight="bold", color=CANVAS_INK, pad=12)


def _display_route_key(route_key: str, max_chars: int = 34) -> str:
    scenario, _, label = str(route_key).partition(":")
    prefix = "cur" if scenario == "existing" else "gen" if scenario == "generated" else scenario
    label = label.replace("generated_", "")
    value = f"{prefix}: {label}"
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def _read_summary() -> dict:
    return json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))


def _read_inputs() -> tuple[dict, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    summary = _read_summary()
    city_dir = Path(summary["city_dir"])
    boundary = read_geodata(city_dir / "analysis_territory" / "buffer.parquet")
    cells = gpd.read_file(summary["street_pattern_cells"])
    class_col = summary.get("street_pattern_class_col") or "top1_class_name"
    if class_col not in cells.columns:
        for candidate in ("top1_class_name", "class_name", "predicted_class", "street_pattern_class"):
            if candidate in cells.columns:
                class_col = candidate
                break
    if class_col not in cells.columns:
        raise ValueError(f"Cannot find street-pattern class column in {summary['street_pattern_cells']}")
    cells = cells.rename(columns={class_col: "street_pattern_class"}).copy()
    cells["street_pattern_class"] = cells["street_pattern_class"].fillna("unknown").astype(str)
    roads_path = city_dir / "derived_layers" / "roads_drive_osmnx.parquet"
    roads = read_geodata(roads_path) if roads_path.exists() else None
    water = load_city_water_layer(city_dir)
    return summary, boundary, cells, roads, water


def _read_stats() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    stats_dir = RUN_DIR / "stats"
    route_stats = pd.read_csv(stats_dir / "route_street_pattern_stats.csv")
    route_class = pd.read_csv(stats_dir / "route_street_pattern_class_length.csv")
    coverage = pd.read_csv(stats_dir / "od_coverage_by_route_count.csv")
    existing_edges = gpd.read_parquet(stats_dir / "existing_route_edges.parquet")
    generated_edges = gpd.read_parquet(stats_dir / "generated_route_edges.parquet")
    for frame in (route_stats, route_class, coverage):
        frame["scenario"] = frame["scenario"].astype(str)
    route_stats["route_label"] = route_stats["route_label"].astype(str)
    route_class["route_label"] = route_class["route_label"].astype(str)
    existing_edges["route_label"] = existing_edges["route_label"].astype(str)
    generated_edges["route_label"] = generated_edges["route_label"].astype(str)
    return route_stats, route_class, coverage, existing_edges, generated_edges


def _class_handles(classes: list[str]) -> list[Patch]:
    return [
        Patch(facecolor=CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"]), edgecolor="none", label=class_name)
        for class_name in classes
    ]


def _plot_street_pattern_map(
    path: Path,
    *,
    boundary: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame | None,
    water: gpd.GeoDataFrame | None,
) -> None:
    boundary_plot = normalize_preview_gdf(boundary)
    cells_plot = normalize_preview_gdf(cells, boundary_plot)
    roads_plot = normalize_preview_gdf(roads, boundary_plot) if roads is not None else None

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    apply_preview_canvas(fig, ax, boundary_plot, title="Bergen Street Pattern")
    plot_water_layer(ax, water, boundary_layer=boundary_plot, polygon_zorder=1, line_zorder=2)

    present = order_street_pattern_classes(cells_plot["street_pattern_class"].dropna().astype(str).unique())
    for class_name in present:
        part = cells_plot[cells_plot["street_pattern_class"] == class_name]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            color=CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"]),
            edgecolor="#ffffff",
            linewidth=0.10,
            alpha=0.92,
            zorder=5,
        )
    if roads_plot is not None and not roads_plot.empty:
        roads_plot.plot(ax=ax, color="#334155", linewidth=0.22, alpha=0.18, zorder=8)

    legend_bottom(ax, _class_handles(present), max_cols=3, fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    save_preview_figure(fig, path)
    plt.close(fig)


def _plot_routes_map(
    path: Path,
    *,
    title: str,
    boundary: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame | None,
    water: gpd.GeoDataFrame | None,
    routes: gpd.GeoDataFrame,
    route_stats: pd.DataFrame,
) -> None:
    boundary_plot = normalize_preview_gdf(boundary)
    cells_plot = normalize_preview_gdf(cells, boundary_plot)
    roads_plot = normalize_preview_gdf(roads, boundary_plot) if roads is not None else None
    route_plot = normalize_preview_gdf(routes, boundary_plot)
    counts = route_stats.set_index("route_label")["street_pattern_class_count"].to_dict()
    route_plot["street_pattern_class_count"] = route_plot["route_label"].map(counts).fillna(0).astype(float)
    vmin = max(1.0, float(route_stats["street_pattern_class_count"].min()))
    vmax = max(vmin, float(route_stats["street_pattern_class_count"].max()))
    cmap = plt.get_cmap("magma_r")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    apply_preview_canvas(fig, ax, boundary_plot, title=title)
    plot_water_layer(ax, water, boundary_layer=boundary_plot, polygon_zorder=1, line_zorder=2)

    present = order_street_pattern_classes(cells_plot["street_pattern_class"].dropna().astype(str).unique())
    for class_name in present:
        part = cells_plot[cells_plot["street_pattern_class"] == class_name]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            color=CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"]),
            edgecolor="none",
            linewidth=0.0,
            alpha=0.28,
            zorder=5,
        )
    if roads_plot is not None and not roads_plot.empty:
        roads_plot.plot(ax=ax, color="#334155", linewidth=0.18, alpha=0.16, zorder=7)

    for route_label, part in route_plot.groupby("route_label", sort=False):
        count = float(counts.get(str(route_label), 0.0))
        part.plot(ax=ax, color=cmap(norm(count)), linewidth=2.35, alpha=0.96, zorder=12)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("street-pattern classes crossed", color=CANVAS_INK, fontsize=8)
    cbar.ax.tick_params(colors=CANVAS_INK, labelsize=8)
    footer_text(fig, ["Routes are colored by the number of distinct street-pattern classes they traverse."], y=0.014)
    ax.set_axis_off()
    fig.tight_layout()
    save_preview_figure(fig, path)
    plt.close(fig)


def _plot_route_mix(path: Path, route_class: pd.DataFrame, route_stats: pd.DataFrame) -> None:
    mix = route_class.copy()
    mix["route_key"] = mix["scenario"] + ":" + mix["route_label"]
    mix["route_km"] = mix["pt_length_m"] / 1000.0
    order_frame = route_stats.copy()
    order_frame["route_key"] = order_frame["scenario"] + ":" + order_frame["route_label"].astype(str)
    order_frame = order_frame.sort_values(["scenario", "street_pattern_class_count", "route_total_m"], ascending=[True, False, False])
    route_order = order_frame["route_key"].tolist()
    classes = order_street_pattern_classes(mix["street_pattern_class"].dropna().astype(str).unique())
    pivot = (
        mix.pivot_table(index="route_key", columns="street_pattern_class", values="route_km", aggfunc="sum", fill_value=0.0)
        .reindex(route_order)
        .fillna(0.0)
    )
    pivot = pivot[[c for c in classes if c in pivot.columns]]
    display_labels = [_display_route_key(route_key) for route_key in pivot.index]

    fig, ax = plt.subplots(figsize=(10.5, 9.0))
    fig.patch.set_facecolor(CANVAS_BACKGROUND)
    _chart_style(ax)
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for class_name in pivot.columns:
        values = pivot[class_name].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            color=CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"]),
            edgecolor=LEGEND_FACE,
            linewidth=0.35,
            label=class_name,
        )
        left += values
    ax.set_yticks(y, display_labels)
    ax.invert_yaxis()
    ax.set_xlabel("route length inside street-pattern class, km")
    ax.set_ylabel("")
    _title(ax, "Route Length Mix By Street Pattern")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(3, len(pivot.columns)),
        frameon=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        fontsize=8,
    )
    fig.tight_layout()
    save_preview_figure(fig, path)
    plt.close(fig)


def _plot_scenario_mix(path: Path, route_class: pd.DataFrame) -> None:
    mix = route_class.copy()
    mix["route_km"] = mix["pt_length_m"] / 1000.0
    classes = order_street_pattern_classes(mix["street_pattern_class"].dropna().astype(str).unique())
    pivot = (
        mix.pivot_table(index="scenario", columns="street_pattern_class", values="route_km", aggfunc="sum", fill_value=0.0)
        .reindex(["existing", "generated"])
        .fillna(0.0)
    )
    pivot = pivot[[c for c in classes if c in pivot.columns]]
    shares = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    fig.patch.set_facecolor(CANVAS_BACKGROUND)
    _chart_style(ax)
    left = np.zeros(len(shares))
    y = np.arange(len(shares))
    for class_name in shares.columns:
        values = shares[class_name].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            color=CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"]),
            edgecolor=LEGEND_FACE,
            linewidth=0.55,
            label=class_name,
        )
        left += values
    for idx, scenario in enumerate(shares.index):
        total = float(pivot.loc[scenario].sum())
        ax.text(1.01, idx, f"{total:.1f} km", va="center", ha="left", fontsize=9, color=CANVAS_MUTED)
    ax.set_yticks(y, shares.index.tolist())
    ax.set_xlim(0.0, 1.16)
    ax.set_xlabel("share of total route length")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    _title(ax, "Total Route Length Share By Street Pattern")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=min(3, len(shares.columns)),
        frameon=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        fontsize=8,
    )
    footer_text(fig, ["Length-weighted comparison of current and generated route sets."], y=0.012)
    fig.tight_layout()
    save_preview_figure(fig, path)
    plt.close(fig)


def _plot_class_count(path: Path, route_stats: pd.DataFrame) -> None:
    frame = route_stats.copy()
    frame["route_key"] = frame["scenario"] + ":" + frame["route_label"].astype(str)
    frame = frame.sort_values(["scenario", "street_pattern_class_count", "route_total_m"], ascending=[True, False, False])
    frame["display_route_key"] = frame["route_key"].map(_display_route_key)
    scenario_colors = {"existing": "#475569", "generated": "#a16207"}

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor(CANVAS_BACKGROUND)
    _chart_style(ax)
    y = np.arange(len(frame))
    ax.barh(
        y,
        frame["street_pattern_class_count"],
        color=frame["scenario"].map(scenario_colors).fillna("#64748b"),
        edgecolor=LEGEND_FACE,
        linewidth=0.45,
    )
    ax.set_yticks(y, frame["display_route_key"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("distinct street-pattern classes crossed")
    ax.set_ylabel("")
    ax.set_xlim(0, max(1, int(frame["street_pattern_class_count"].max())) + 0.7)
    _title(ax, "Street-Pattern Breadth Per Route")
    ax.legend(
        handles=[
            Line2D([0], [0], color=scenario_colors["existing"], linewidth=7, label="existing"),
            Line2D([0], [0], color=scenario_colors["generated"], linewidth=7, label="generated"),
        ],
        loc="lower right",
        frameon=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        fontsize=8,
    )
    fig.tight_layout()
    save_preview_figure(fig, path)
    plt.close(fig)


def _plot_od_coverage(path: Path, coverage: pd.DataFrame) -> None:
    colors = {"existing": "#475569", "generated": "#a16207"}
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    fig.patch.set_facecolor(CANVAS_BACKGROUND)
    _chart_style(ax)
    for scenario, part in coverage.groupby("scenario", sort=False):
        part = part.sort_values("route_count")
        ax.plot(
            part["route_count"],
            part["covered_share"],
            marker="o",
            markersize=4.5,
            linewidth=2.3,
            color=colors.get(scenario, "#64748b"),
            label=scenario,
        )
    ax.set_xlabel("routes included")
    ax.set_ylabel("direct OD share covered")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, max(0.05, min(1.0, float(coverage["covered_share"].max()) * 1.16)))
    _title(ax, "Route Count Versus Covered OD")
    ax.legend(frameon=True, facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE)
    footer_text(fig, ["Coverage uses direct same-route OD pairs on the ConnectPT OD matrix."], y=0.012)
    fig.tight_layout()
    save_preview_figure(fig, path)
    plt.close(fig)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.facecolor": CANVAS_BACKGROUND,
            "savefig.facecolor": CANVAS_BACKGROUND,
        }
    )
    summary, boundary, cells, roads, water = _read_inputs()
    route_stats, route_class, coverage, existing_edges, generated_edges = _read_stats()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _plot_street_pattern_map(
        OUTPUT_DIR / "01_street_pattern_dominant_class.png",
        boundary=boundary,
        cells=cells,
        roads=roads,
        water=water,
    )
    _plot_routes_map(
        OUTPUT_DIR / "02_existing_routes_on_street_pattern.png",
        title="Current Bus Routes On Street Pattern",
        boundary=boundary,
        cells=cells,
        roads=roads,
        water=water,
        routes=existing_edges,
        route_stats=route_stats[route_stats["scenario"] == "existing"],
    )
    _plot_routes_map(
        OUTPUT_DIR / "03_generated_routes_on_street_pattern.png",
        title="Generated Bare-OD Routes On Street Pattern",
        boundary=boundary,
        cells=cells,
        roads=roads,
        water=water,
        routes=generated_edges,
        route_stats=route_stats[route_stats["scenario"] == "generated"],
    )
    _plot_route_mix(OUTPUT_DIR / "04_route_length_mix_by_street_pattern.png", route_class, route_stats)
    _plot_scenario_mix(OUTPUT_DIR / "05_total_route_length_share_by_street_pattern.png", route_class)
    _plot_class_count(OUTPUT_DIR / "06_street_pattern_breadth_per_route.png", route_stats)
    _plot_od_coverage(OUTPUT_DIR / "07_route_count_vs_direct_od_coverage.png", coverage)

    manifest = {
        "source_run_dir": str(RUN_DIR),
        "city_dir": summary.get("city_dir"),
        "style_reference": "segregation-by-design-experiments preview canvas and street-pattern palette",
        "pngs": sorted(path.name for path in OUTPUT_DIR.glob("*.png")),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(manifest['pngs'])} PNGs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

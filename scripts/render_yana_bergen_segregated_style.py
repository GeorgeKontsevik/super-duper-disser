from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Point

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


BASELINE_RUN_DIR = (
    REPO_ROOT
    / "yana_experiments"
    / "street_pattern_route_comparison"
    / "bergen_norway"
    / "bus_existing_count_meanmax_033_n17_len9_25"
)
STREET_PATTERN_RUN_DIR = (
    REPO_ROOT
    / "yana_experiments"
    / "street_pattern_route_comparison"
    / "bergen_norway"
    / "bus_conn100_div100_n17_len9_25"
)
OUTPUT_DIR = STREET_PATTERN_RUN_DIR / "preview_png_segregated"
COMPARISON_LABEL = "connectivity+diversity generated"
FOCUS_CLASS = "Loops & Lollipops"
SCENARIO_ORDER = ["existing", "baseline generated", COMPARISON_LABEL]
SCENARIO_COLORS = {
    "existing": "#475569",
    "baseline generated": "#a16207",
    COMPARISON_LABEL: "#0f766e",
}
SCENARIO_PREFIXES = {
    "existing": "cur",
    "baseline generated": "base",
    COMPARISON_LABEL: "sp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3-way Bergen route comparison in segregated preview style.")
    parser.add_argument("--baseline-run-dir", type=Path, default=BASELINE_RUN_DIR)
    parser.add_argument("--baseline-label", default="Connectivity Baseline")
    parser.add_argument("--comparison-run-dir", type=Path, default=STREET_PATTERN_RUN_DIR)
    parser.add_argument("--comparison-label", default=COMPARISON_LABEL)
    parser.add_argument("--extra-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--extra-label", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


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
    prefix = SCENARIO_PREFIXES.get(scenario, scenario)
    label = label.replace("generated_", "")
    value = f"{prefix}: {label}"
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def _compact_scenario_title(scenario: str) -> str:
    if scenario == "baseline generated":
        return "Connectivity Baseline\nGenerated"
    if "connectivity+diversity" in scenario.lower():
        return "Conn.+Diversity\nGenerated"
    return scenario.title()


def _compact_grid_scenario_title(scenario: str) -> str:
    if scenario == "baseline generated":
        return "Connectivity Baseline"
    if "connectivity+diversity" in scenario.lower():
        return "Conn.+Diversity"
    return scenario.title()


def _read_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _load_graph_nodes(summary: dict, crs) -> gpd.GeoDataFrame:
    graph_path = Path(summary["generated_graph"])
    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    graph_crs = graph.graph.get("crs") or crs
    rows = []
    for order_index, node in enumerate(sorted(graph.nodes())):
        data = graph.nodes[node]
        rows.append(
            {
                "graph_node_id": int(node),
                "order_index": int(order_index),
                "geometry": Point(float(data["x"]), float(data["y"])),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=graph_crs)


def _stop_classes(graph_nodes: gpd.GeoDataFrame, cells: gpd.GeoDataFrame) -> pd.DataFrame:
    nodes = graph_nodes.copy()
    class_cells = cells[["street_pattern_class", "geometry"]].copy()
    if nodes.crs is None and class_cells.crs is not None:
        nodes = nodes.set_crs(class_cells.crs)
    if nodes.crs is not None and class_cells.crs is not None and nodes.crs != class_cells.crs:
        class_cells = class_cells.to_crs(nodes.crs)
    joined = nodes[["order_index", "geometry"]].sjoin(
        class_cells,
        how="left",
        predicate="within",
    )
    if joined.index.duplicated().any():
        joined = joined[~joined.index.duplicated(keep="first")]
    out = nodes[["order_index", "geometry"]].copy()
    out["street_pattern_class"] = joined["street_pattern_class"].fillna("unknown").astype(str).to_numpy()
    return pd.DataFrame(out.drop(columns="geometry"))


def _read_inputs(run_dir: Path) -> tuple[dict, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    summary = _read_summary(run_dir)
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


def _read_stats(
    run_dir: Path,
    *,
    generated_scenario: str,
    include_existing: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame | None, gpd.GeoDataFrame]:
    stats_dir = run_dir / "stats"
    route_stats = pd.read_csv(stats_dir / "route_street_pattern_stats.csv")
    route_class = pd.read_csv(stats_dir / "route_street_pattern_class_length.csv")
    coverage = pd.read_csv(stats_dir / "od_coverage_by_route_count.csv")
    existing_edges = gpd.read_parquet(stats_dir / "existing_route_edges.parquet") if include_existing else None
    generated_edges = gpd.read_parquet(stats_dir / "generated_route_edges.parquet")
    for frame in (route_stats, route_class, coverage):
        frame["scenario"] = frame["scenario"].astype(str)
        frame.loc[frame["scenario"] == "generated", "scenario"] = generated_scenario
        if not include_existing:
            frame.drop(frame[frame["scenario"] == "existing"].index, inplace=True)
    route_stats["route_label"] = route_stats["route_label"].astype(str)
    route_stats["route_key"] = route_stats["scenario"] + ":" + route_stats["route_label"].astype(str)
    route_class["route_label"] = route_class["route_label"].astype(str)
    if existing_edges is not None:
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


def _draw_routes_panel(
    fig,
    ax,
    *,
    title: str,
    boundary_plot: gpd.GeoDataFrame,
    cells_plot: gpd.GeoDataFrame,
    roads_plot: gpd.GeoDataFrame | None,
    water: gpd.GeoDataFrame | None,
    routes: gpd.GeoDataFrame,
    cmap,
    norm,
) -> None:
    route_plot = normalize_preview_gdf(routes, boundary_plot)
    apply_preview_canvas(fig, ax, boundary_plot, title=title, pad_ratio=0.055)
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
            alpha=0.25,
            zorder=5,
        )
    if roads_plot is not None and not roads_plot.empty:
        roads_plot.plot(ax=ax, color="#334155", linewidth=0.16, alpha=0.15, zorder=7)

    for count, part in route_plot.groupby("edge_route_count", sort=True):
        part.plot(ax=ax, color=cmap(norm(float(count))), linewidth=2.35, alpha=0.96, zorder=12)
    ax.set_axis_off()


def _with_edge_route_counts(routes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = routes.copy()
    if {"edge_a", "edge_b"}.issubset(out.columns):
        a = pd.to_numeric(out["edge_a"], errors="coerce").astype("Int64").astype(str)
        b = pd.to_numeric(out["edge_b"], errors="coerce").astype("Int64").astype(str)
        out["edge_key"] = a + ":" + b
    elif {"u", "v"}.issubset(out.columns):
        u = pd.to_numeric(out["u"], errors="coerce")
        v = pd.to_numeric(out["v"], errors="coerce")
        a = np.minimum(u, v).astype("Int64").astype(str)
        b = np.maximum(u, v).astype("Int64").astype(str)
        out["edge_key"] = a + ":" + b
    else:
        out["edge_key"] = out.geometry.apply(lambda geom: geom.wkb_hex if geom is not None else "")
    counts = (
        out.groupby("edge_key")["route_label"]
        .nunique()
        .rename("edge_route_count")
        .reset_index()
    )
    out = out.merge(counts, on="edge_key", how="left")
    out["edge_route_count"] = pd.to_numeric(out["edge_route_count"], errors="coerce").fillna(1).astype(float)
    return out.drop_duplicates("edge_key").reset_index(drop=True)


def _plot_routes_panels(
    path: Path,
    *,
    boundary: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame | None,
    water: gpd.GeoDataFrame | None,
    panels: list[tuple[str, gpd.GeoDataFrame, pd.DataFrame]],
    route_stats: pd.DataFrame,
) -> None:
    boundary_plot = normalize_preview_gdf(boundary)
    cells_plot = normalize_preview_gdf(cells, boundary_plot)
    roads_plot = normalize_preview_gdf(roads, boundary_plot) if roads is not None else None
    counted_panels = [(title, _with_edge_route_counts(routes), stats) for title, routes, stats in panels]
    vmin = 1.0
    vmax = max(
        vmin,
        max(float(routes["edge_route_count"].max()) for _, routes, _ in counted_panels if not routes.empty),
    )
    cmap = plt.get_cmap("magma_r")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, len(panels), figsize=(7.8 * len(panels), 8.6), gridspec_kw={"wspace": 0.0})
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(CANVAS_BACKGROUND)
    for ax, (title, routes, stats) in zip(axes, counted_panels):
        _draw_routes_panel(
            fig,
            ax,
            title=title,
            boundary_plot=boundary_plot,
            cells_plot=cells_plot,
            roads_plot=roads_plot,
            water=water,
            routes=routes,
            cmap=cmap,
            norm=norm,
        )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.018, pad=0.008)
    cbar.set_label("routes per edge", color=CANVAS_INK, fontsize=9)
    cbar.ax.tick_params(colors=CANVAS_INK, labelsize=8)
    footer_text(fig, ["Route edges are colored by how many routes use the same graph edge within each panel."], y=0.016)
    save_preview_figure(fig, path)
    plt.close(fig)


def _plot_route_mix(path: Path, route_class: pd.DataFrame, route_stats: pd.DataFrame) -> None:
    mix = route_class.copy()
    mix["route_key"] = mix["scenario"] + ":" + mix["route_label"]
    mix["route_km"] = mix["pt_length_m"] / 1000.0
    classes = order_street_pattern_classes(mix["street_pattern_class"].dropna().astype(str).unique())
    scenario_frames = []
    max_total = 0.0
    for scenario in SCENARIO_ORDER:
        stats_part = route_stats[route_stats["scenario"] == scenario].copy()
        if stats_part.empty:
            continue
        stats_part["route_key"] = stats_part["scenario"] + ":" + stats_part["route_label"].astype(str)
        stats_part = stats_part.sort_values("route_total_m", ascending=False)
        pivot = (
            mix[mix["scenario"] == scenario]
            .pivot_table(index="route_key", columns="street_pattern_class", values="route_km", aggfunc="sum", fill_value=0.0)
            .reindex(stats_part["route_key"].tolist())
            .fillna(0.0)
        )
        pivot = pivot[[c for c in classes if c in pivot.columns]]
        if not pivot.empty:
            max_total = max(max_total, float(pivot.sum(axis=1).max()))
            scenario_frames.append((scenario, pivot))
    if not scenario_frames:
        return

    fig, axes = plt.subplots(
        1,
        len(scenario_frames),
        figsize=(5.4 * len(scenario_frames), 7.6),
        sharex=True,
        gridspec_kw={"wspace": 0.08},
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(CANVAS_BACKGROUND)
    handles = []
    for ax, (scenario, pivot) in zip(axes, scenario_frames):
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
            if class_name not in {h.get_label() for h in handles}:
                handles.append(Patch(facecolor=CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"]), label=class_name))
            left += values
        totals = pivot.sum(axis=1).to_numpy()
        for yi, total in zip(y, totals):
            ax.text(total + max_total * 0.015, yi, f"{total:.1f}", va="center", ha="left", fontsize=7.5, color=CANVAS_MUTED)
        ax.set_yticks(y, [_display_route_key(route_key, max_chars=22) for route_key in pivot.index])
        ax.invert_yaxis()
        ax.set_xlim(0, max_total * 1.18 if max_total else 1.0)
        ax.set_xlabel("km")
        ax.set_ylabel("")
        _title(ax, scenario.title())
    fig.suptitle("Route Length And Street-Pattern Class Mix", fontsize=17, fontweight="bold", color=CANVAS_INK, y=0.98)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(5, len(handles)),
        frameon=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_preview_figure(fig, path)
    plt.close(fig)


def _generated_stop_mix(summary: dict, *, scenario: str, stop_classes: pd.DataFrame) -> pd.DataFrame:
    class_by_order = stop_classes.set_index("order_index")["street_pattern_class"].to_dict()
    rows = []
    for route_num, raw_route in enumerate(summary.get("generated_routes", []), start=1):
        route_label = f"generated_{route_num:02d}"
        for node_idx in raw_route:
            node_idx = int(node_idx)
            if node_idx < 0:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "route_label": route_label,
                    "street_pattern_class": class_by_order.get(node_idx, "unknown"),
                    "stop_count": 1,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["scenario", "route_label", "street_pattern_class", "stop_count"])
    return (
        pd.DataFrame(rows)
        .groupby(["scenario", "route_label", "street_pattern_class"], as_index=False)["stop_count"]
        .sum()
    )


def _existing_stop_mix(
    *,
    existing_edges: gpd.GeoDataFrame,
    route_stats: pd.DataFrame,
    graph_nodes: gpd.GeoDataFrame,
    stop_classes: pd.DataFrame,
    threshold_m: float,
) -> pd.DataFrame:
    edges = existing_edges.copy()
    nodes = graph_nodes.copy()
    if edges.crs is None and nodes.crs is not None:
        edges = edges.set_crs(nodes.crs)
    if edges.crs is not None and nodes.crs is not None and edges.crs != nodes.crs:
        edges = edges.to_crs(nodes.crs)
    class_by_order = stop_classes.set_index("order_index")["street_pattern_class"].to_dict()
    rows = []
    labels = route_stats.loc[route_stats["scenario"] == "existing", "route_label"].astype(str).tolist()
    for route_label in labels:
        part = edges[edges["route_label"].astype(str) == route_label]
        if part.empty:
            continue
        route_geom = part.geometry.union_all()
        matched = nodes.loc[nodes.geometry.distance(route_geom) <= threshold_m, "order_index"].astype(int)
        for node_idx in matched.tolist():
            rows.append(
                {
                    "scenario": "existing",
                    "route_label": route_label,
                    "street_pattern_class": class_by_order.get(node_idx, "unknown"),
                    "stop_count": 1,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["scenario", "route_label", "street_pattern_class", "stop_count"])
    return (
        pd.DataFrame(rows)
        .groupby(["scenario", "route_label", "street_pattern_class"], as_index=False)["stop_count"]
        .sum()
    )


def _build_stop_mix(
    *,
    baseline_summary: dict,
    generated_summaries: list[tuple[str, dict]],
    cells: gpd.GeoDataFrame,
    route_stats: pd.DataFrame,
    existing_edges: gpd.GeoDataFrame,
) -> pd.DataFrame:
    graph_nodes = _load_graph_nodes(baseline_summary, cells.crs)
    stop_classes = _stop_classes(graph_nodes, cells)
    threshold_m = float(baseline_summary.get("existing_stop_match_threshold_m", 100.0))
    pieces = [
        _existing_stop_mix(
            existing_edges=existing_edges,
            route_stats=route_stats,
            graph_nodes=graph_nodes,
            stop_classes=stop_classes,
            threshold_m=threshold_m,
        )
    ]
    pieces.extend(
        _generated_stop_mix(summary, scenario=scenario, stop_classes=stop_classes)
        for scenario, summary in generated_summaries
    )
    return pd.concat(pieces, ignore_index=True)


def _plot_route_length_and_stop_mix(
    path: Path,
    *,
    route_class: pd.DataFrame,
    stop_class: pd.DataFrame,
    route_stats: pd.DataFrame,
) -> None:
    length_mix = route_class.copy()
    length_mix["route_key"] = length_mix["scenario"] + ":" + length_mix["route_label"].astype(str)
    length_mix["route_km"] = length_mix["pt_length_m"] / 1000.0
    stop_mix = stop_class.copy()
    stop_mix["route_key"] = stop_mix["scenario"] + ":" + stop_mix["route_label"].astype(str)

    classes = order_street_pattern_classes(
        sorted(
            set(length_mix["street_pattern_class"].dropna().astype(str))
            | set(stop_mix["street_pattern_class"].dropna().astype(str))
        )
    )
    scenario_frames = []
    max_km = 0.0
    max_stops = 0.0
    for scenario in SCENARIO_ORDER:
        stats_part = route_stats[route_stats["scenario"] == scenario].copy()
        if stats_part.empty:
            continue
        stats_part["route_key"] = stats_part["scenario"] + ":" + stats_part["route_label"].astype(str)
        stats_part = stats_part.sort_values("route_total_m", ascending=False)
        route_order = stats_part["route_key"].tolist()
        length_pivot = (
            length_mix[length_mix["scenario"] == scenario]
            .pivot_table(index="route_key", columns="street_pattern_class", values="route_km", aggfunc="sum", fill_value=0.0)
            .reindex(route_order)
            .fillna(0.0)
        )
        stop_pivot = (
            stop_mix[stop_mix["scenario"] == scenario]
            .pivot_table(index="route_key", columns="street_pattern_class", values="stop_count", aggfunc="sum", fill_value=0.0)
            .reindex(route_order)
            .fillna(0.0)
        )
        length_pivot = length_pivot[[c for c in classes if c in length_pivot.columns]]
        stop_pivot = stop_pivot[[c for c in classes if c in stop_pivot.columns]]
        if length_pivot.empty and stop_pivot.empty:
            continue
        max_km = max(max_km, float(length_pivot.sum(axis=1).max()) if not length_pivot.empty else 0.0)
        max_stops = max(max_stops, float(stop_pivot.sum(axis=1).max()) if not stop_pivot.empty else 0.0)
        scenario_frames.append((scenario, length_pivot, stop_pivot))
    if not scenario_frames:
        return

    fig, axes = plt.subplots(
        len(scenario_frames),
        4,
        figsize=(24.0, 5.45 * len(scenario_frames)),
        gridspec_kw={"wspace": 0.12, "hspace": 0.40},
    )
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(CANVAS_BACKGROUND)
    handles = []

    for row_idx, (scenario, length_pivot, stop_pivot) in enumerate(scenario_frames):
        length_share = length_pivot.div(length_pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        stop_share = stop_pivot.div(stop_pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        for col_idx, (metric, pivot, max_total, xlabel, is_share) in enumerate(
            [
                ("length", length_pivot, max_km, "km", False),
                ("stops", stop_pivot, max_stops, "stops", False),
                ("length share", length_share, 1.0, "share of route length", True),
                ("stop share", stop_share, 1.0, "share of route stops", True),
            ]
        ):
            ax = axes[row_idx, col_idx]
            _chart_style(ax)
            y = np.arange(len(pivot))
            left = np.zeros(len(pivot))
            for class_name in pivot.columns:
                values = pivot[class_name].to_numpy(dtype=float)
                ax.barh(
                    y,
                    values,
                    left=left,
                    color=CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"]),
                    edgecolor=LEGEND_FACE,
                    linewidth=0.35,
                    label=class_name,
                )
                if class_name not in {h.get_label() for h in handles}:
                    handles.append(Patch(facecolor=CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"]), label=class_name))
                left += values
            totals = pivot.sum(axis=1).to_numpy(dtype=float)
            if not is_share:
                for yi, total in zip(y, totals):
                    label = f"{total:.1f}" if metric == "length" else f"{int(round(total))}"
                    ax.text(total + max_total * 0.015, yi, label, va="center", ha="left", fontsize=7.0, color=CANVAS_MUTED)
            ax.set_yticks(y, [_display_route_key(route_key, max_chars=24) for route_key in pivot.index])
            if col_idx > 0:
                ax.set_yticklabels([])
            ax.invert_yaxis()
            ax.set_xlim(0, max_total * (1.18 if not is_share else 1.0) if max_total else 1.0)
            if is_share:
                ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
            ax.set_xlabel(xlabel)
            ax.set_ylabel("")
            ax.set_title(
                f"{_compact_grid_scenario_title(scenario)}\n{metric.title()}",
                fontsize=12,
                fontweight="bold",
                color=CANVAS_INK,
                pad=10,
            )

    fig.suptitle(
        "Route Length And Stop Mix By Street-Pattern Class",
        fontsize=18,
        fontweight="bold",
        color=CANVAS_INK,
        y=0.995,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=min(5, len(handles)),
        frameon=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    save_preview_figure(fig, path)
    plt.close(fig)


def _plot_scenario_mix(path: Path, route_class: pd.DataFrame) -> None:
    mix = route_class.copy()
    mix["route_km"] = mix["pt_length_m"] / 1000.0
    classes = order_street_pattern_classes(mix["street_pattern_class"].dropna().astype(str).unique())
    pivot = (
        mix.pivot_table(index="scenario", columns="street_pattern_class", values="route_km", aggfunc="sum", fill_value=0.0)
        .reindex([scenario for scenario in SCENARIO_ORDER if scenario in set(mix["scenario"])])
        .fillna(0.0)
    )
    pivot = pivot[[c for c in classes if c in pivot.columns]]
    shares = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if FOCUS_CLASS in shares.columns:
        scenario_order = shares[FOCUS_CLASS].sort_values(ascending=False).index.tolist()
        shares = shares.reindex(scenario_order)
        pivot = pivot.reindex(scenario_order)

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
    ax.set_yticks(y, [_compact_scenario_title(str(scenario)) for scenario in shares.index])
    ax.invert_yaxis()
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
    footer_text(fig, [f"Rows are sorted by {FOCUS_CLASS} share of total route length."], y=0.012)
    fig.tight_layout()
    save_preview_figure(fig, path)
    plt.close(fig)


def _plot_class_count(path: Path, route_stats: pd.DataFrame) -> None:
    frame = route_stats.copy()
    frame["route_key"] = frame["scenario"] + ":" + frame["route_label"].astype(str)
    frame = frame.sort_values(["scenario", "street_pattern_class_count", "route_total_m"], ascending=[True, False, False])
    frame["display_route_key"] = frame["route_key"].map(_display_route_key)
    frame["scenario"] = pd.Categorical(frame["scenario"], categories=SCENARIO_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    fig.patch.set_facecolor(CANVAS_BACKGROUND)
    _chart_style(ax)
    y = np.arange(len(frame))
    ax.barh(
        y,
        frame["street_pattern_class_count"],
        color=frame["scenario"].astype(str).map(SCENARIO_COLORS).fillna("#64748b"),
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
            Line2D([0], [0], color=SCENARIO_COLORS[name], linewidth=7, label=name)
            for name in SCENARIO_ORDER
            if name in set(frame["scenario"].astype(str))
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
    colors = SCENARIO_COLORS
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
    fig.tight_layout()
    save_preview_figure(fig, path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    global BASELINE_RUN_DIR, STREET_PATTERN_RUN_DIR, OUTPUT_DIR, COMPARISON_LABEL, SCENARIO_ORDER, SCENARIO_COLORS, SCENARIO_PREFIXES
    BASELINE_RUN_DIR = args.baseline_run_dir.resolve()
    STREET_PATTERN_RUN_DIR = args.comparison_run_dir.resolve()
    OUTPUT_DIR = args.output_dir.resolve()
    COMPARISON_LABEL = str(args.comparison_label)
    extra_dirs = [path.resolve() for path in args.extra_run_dir]
    extra_labels = [str(label) for label in args.extra_label]
    if len(extra_dirs) != len(extra_labels):
        raise ValueError("--extra-run-dir and --extra-label must be provided the same number of times.")
    generated_runs = [
        (str(args.baseline_label), BASELINE_RUN_DIR),
        (COMPARISON_LABEL, STREET_PATTERN_RUN_DIR),
        *list(zip(extra_labels, extra_dirs)),
    ]
    SCENARIO_ORDER = ["existing", *[label for label, _ in generated_runs]]
    palette = ["#a16207", "#0f766e", "#7c3aed", "#dc2626", "#0284c7", "#16a34a"]
    SCENARIO_COLORS = {"existing": "#475569"}
    SCENARIO_COLORS.update({label: palette[idx % len(palette)] for idx, (label, _) in enumerate(generated_runs)})
    SCENARIO_PREFIXES = {"existing": "cur"}
    SCENARIO_PREFIXES.update({label: f"run{idx + 1}" for idx, (label, _) in enumerate(generated_runs)})
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.facecolor": CANVAS_BACKGROUND,
            "savefig.facecolor": CANVAS_BACKGROUND,
        }
    )
    summary, boundary, cells, roads, water = _read_inputs(BASELINE_RUN_DIR)
    baseline_stats, baseline_class, baseline_coverage, existing_edges, baseline_generated_edges = _read_stats(
        BASELINE_RUN_DIR,
        generated_scenario=generated_runs[0][0],
        include_existing=True,
    )
    stats_frames = [baseline_stats]
    class_frames = [baseline_class]
    coverage_frames = [baseline_coverage]
    generated_edges = [(generated_runs[0][0], baseline_generated_edges)]
    generated_summaries = [(generated_runs[0][0], summary)]
    for label, run_dir in generated_runs[1:]:
        stats, route_class_part, coverage_part, _, edges = _read_stats(
            run_dir,
            generated_scenario=label,
            include_existing=False,
        )
        stats_frames.append(stats)
        class_frames.append(route_class_part)
        coverage_frames.append(coverage_part)
        generated_edges.append((label, edges))
        generated_summaries.append((label, _read_summary(run_dir)))
    route_stats = pd.concat(stats_frames, ignore_index=True)
    route_class = pd.concat(class_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    if existing_edges is None:
        raise ValueError("Baseline run must include existing route edges for stop-mix rendering.")
    stop_class = _build_stop_mix(
        baseline_summary=summary,
        generated_summaries=generated_summaries,
        cells=cells,
        route_stats=route_stats,
        existing_edges=existing_edges,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_png in OUTPUT_DIR.glob("*.png"):
        stale_png.unlink()
    stop_class.to_csv(OUTPUT_DIR / "route_street_pattern_stop_count.csv", index=False)

    _plot_street_pattern_map(
        OUTPUT_DIR / "01_street_pattern_dominant_class.png",
        boundary=boundary,
        cells=cells,
        roads=roads,
        water=water,
    )
    _plot_routes_panels(
        OUTPUT_DIR / "05_existing_vs_generated_vs_street_pattern_routes.png",
        boundary=boundary,
        cells=cells,
        roads=roads,
        water=water,
        panels=[
            ("Current Bus Routes", existing_edges, route_stats[route_stats["scenario"] == "existing"]),
            *[
                (f"{label} Routes", edges, route_stats[route_stats["scenario"] == label])
                for label, edges in generated_edges
            ],
        ],
        route_stats=route_stats,
    )
    _plot_route_mix(OUTPUT_DIR / "06_route_length_mix_by_street_pattern.png", route_class, route_stats)
    _plot_scenario_mix(OUTPUT_DIR / "07_total_route_length_share_by_street_pattern.png", route_class)
    _plot_od_coverage(OUTPUT_DIR / "09_route_count_vs_direct_od_coverage.png", coverage)
    _plot_route_length_and_stop_mix(
        OUTPUT_DIR / "10_route_length_and_stop_mix_by_street_pattern.png",
        route_class=route_class,
        stop_class=stop_class,
        route_stats=route_stats,
    )

    manifest = {
        "baseline_run_dir": str(BASELINE_RUN_DIR),
        "comparison_run_dir": str(STREET_PATTERN_RUN_DIR),
        "comparison_label": COMPARISON_LABEL,
        "extra_run_dirs": [str(path) for path in extra_dirs],
        "extra_labels": extra_labels,
        "scenario_order": SCENARIO_ORDER,
        "city_dir": summary.get("city_dir"),
        "style_reference": "segregation-by-design-experiments preview canvas and street-pattern palette",
        "ad_hoc_tables": {
            "route_street_pattern_stop_count": str(OUTPUT_DIR / "route_street_pattern_stop_count.csv"),
        },
        "pngs": sorted(path.name for path in OUTPUT_DIR.glob("*.png")),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(manifest['pngs'])} PNGs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

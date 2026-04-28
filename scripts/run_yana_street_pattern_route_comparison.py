from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString, Point

from aggregated_spatial_pipeline.pipeline.run_pt_street_pattern_dependency import (
    _overlay_pt_with_street_pattern,
    _pick_class_column,
)
from aggregated_spatial_pipeline.visualization import (
    CANVAS_GRID,
    CANVAS_INK,
    LEGEND_EDGE,
    LEGEND_FACE,
    get_palette,
    order_street_pattern_classes,
)


DEFAULT_CITY_DIR = Path(
    "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/"
    "joint_inputs/freiburg_im_breisgau_germany"
)
DEFAULT_GENERATED_SUMMARY = Path(
    "yana_experiments/bare_od_route_generation/freiburg_im_breisgau_germany/"
    "bus_length_m_fixed_033_n6/summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "yana_experiments/street_pattern_route_comparison/freiburg_im_breisgau_germany/"
    "bus_length_m_fixed_033_n6"
)
CLASS_COLUMN_CANDIDATES = ("top1_class_name", "class_name", "predicted_class", "street_pattern_class")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current and generated bare-OD routes by street-pattern coverage and OD coverage."
    )
    parser.add_argument("--city-dir", type=Path, default=DEFAULT_CITY_DIR)
    parser.add_argument("--generated-summary", type=Path, default=DEFAULT_GENERATED_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modality", default="bus")
    parser.add_argument("--generated-graph-path", type=Path, default=None)
    parser.add_argument("--od-matrix-path", type=Path, default=None)
    parser.add_argument("--route-count", type=int, default=None)
    parser.add_argument(
        "--existing-route-policy",
        choices=("all", "closest_stop_count"),
        default="all",
        help=(
            "all: compare against all current routes of the modality and require enough generated routes. "
            "closest_stop_count: compare against --route-count current routes closest to generated stop count."
        ),
    )
    parser.add_argument("--existing-stop-match-threshold-m", type=float, default=100.0)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_od_matrix(path: Path) -> np.ndarray:
    od = pd.read_csv(path, index_col=0)
    if od.empty:
        raise ValueError(f"OD matrix is empty: {path}")
    values = od.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"OD matrix must be square after reading index column, got shape={values.shape}: {path}")
    return values


def _resolve_od_matrix_path(generated_summary_path: Path, modality: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return _require_path(explicit, "OD matrix")
    return _require_path(generated_summary_path.parent / f"{modality}_od_matrix.csv", "generated OD matrix")


def _resolve_generated_graph_path(city_dir: Path, modality: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return _require_path(explicit, "generated-route graph")
    return _require_path(city_dir / "connectpt_osm" / modality / "graph.pkl", "ConnectPT graph")


def _require_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _resolve_street_cells(city_dir: Path) -> Path:
    path = city_dir / "street_pattern" / city_dir.name / "predicted_cells.geojson"
    return _require_path(path, "street-pattern predicted cells")


def _load_routes_from_summary(summary: dict) -> list[list[int]]:
    raw = summary.get("routes_tensor")
    if not raw or not isinstance(raw, list) or not raw[0]:
        raise ValueError("Generated summary does not contain routes_tensor[0].")
    routes = [[int(node_idx) for node_idx in route if int(node_idx) >= 0] for route in raw[0]]
    if any(len(route) < 2 for route in routes):
        raise ValueError("Generated routes must contain at least two stops each.")
    return routes


def _load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as fh:
        graph = pickle.load(fh)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx graph in {path}, got {type(graph)!r}")
    return graph


def _graph_nodes_gdf(graph: nx.Graph, crs) -> gpd.GeoDataFrame:
    rows = []
    sorted_nodes = sorted(graph.nodes())
    for order_index, node in enumerate(sorted_nodes):
        data = graph.nodes[node]
        rows.append(
            {
                "node": int(node),
                "order_index": int(order_index),
                "geometry": Point(float(data["x"]), float(data["y"])),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def _edge_geometry(graph: nx.Graph, u: int, v: int) -> tuple[LineString | MultiLineString, float]:
    if not graph.has_edge(u, v):
        raise KeyError(f"Generated route uses non-adjacent graph nodes: {u}->{v}")
    data = graph.get_edge_data(u, v)
    if data is None:
        raise KeyError(f"Missing edge data for generated route edge: {u}->{v}")
    geometry = data.get("geometry")
    if geometry is None or geometry.is_empty:
        ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
        vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
        geometry = LineString([(ux, uy), (vx, vy)])
    length_m = data.get("length_m", data.get("weight"))
    if length_m is None:
        length_m = geometry.length
    return geometry, float(length_m)


def _generated_edges(
    routes_by_index: list[list[int]],
    graph: nx.Graph,
    crs,
) -> tuple[gpd.GeoDataFrame, list[set[int]], pd.DataFrame]:
    sorted_nodes = sorted(graph.nodes())
    records = []
    stop_sets: list[set[int]] = []
    node_records = []
    for route_num, route_indices in enumerate(routes_by_index, start=1):
        route_label = f"generated_{route_num:02d}"
        stop_sets.append(set(route_indices))
        mapped_nodes = [int(sorted_nodes[idx]) for idx in route_indices]
        node_records.append(
            {
                "type": "generated",
                "route_label": route_label,
                "stop_count": len(route_indices),
            }
        )
        for edge_num, (u, v) in enumerate(zip(mapped_nodes, mapped_nodes[1:])):
            geometry, length_m = _edge_geometry(graph, u, v)
            records.append(
                {
                    "edge_id": len(records),
                    "type": "generated",
                    "route_label": route_label,
                    "u": int(u),
                    "v": int(v),
                    "edge_order": edge_num,
                    "length_meter": length_m,
                    "geometry": geometry,
                }
            )
    edges = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    route_nodes = pd.DataFrame(node_records)
    return edges, stop_sets, route_nodes


def _select_existing_routes(
    route_stats: pd.DataFrame,
    *,
    modality: str,
    target_stop_count: int | None,
    route_count: int,
    policy: str,
) -> pd.DataFrame:
    work = route_stats[route_stats["type"].astype("string").str.lower() == modality.lower()].copy()
    if work.empty:
        raise ValueError(f"No current routes found for modality={modality!r}.")
    work["stop_count"] = pd.to_numeric(work["stop_count"], errors="coerce")
    work["route_total_m"] = pd.to_numeric(work["route_total_m"], errors="coerce")

    if policy == "all":
        selected = work.sort_values(["route_total_m", "route_label"], ascending=[False, True]).reset_index(drop=True)
        selected["stop_gap"] = 0
    elif policy == "closest_stop_count":
        if target_stop_count is None:
            raise ValueError("closest_stop_count policy requires target_stop_count.")
        selected = work.copy()
        selected["stop_gap"] = (selected["stop_count"] - target_stop_count).abs()
        selected = (
            selected.sort_values(
                ["stop_gap", "stop_count", "route_total_m", "route_label"],
                ascending=[True, True, True, True],
            )
            .head(route_count)
            .reset_index(drop=True)
        )
    else:
        raise ValueError(f"Unsupported existing-route-policy: {policy}")

    if len(selected) < route_count:
        raise ValueError(f"Only {len(selected)} current routes available, need {route_count}.")
    selected = selected.head(route_count).copy()
    selected["type"] = "existing"
    return selected


def _existing_edges(pt_edges: gpd.GeoDataFrame, selected: pd.DataFrame) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    labels = set(selected["route_label"].astype(str))
    work = pt_edges[pt_edges["route_label"].astype(str).isin(labels)].copy()
    if work.empty:
        raise ValueError("Selected current routes have no matching PT edges.")
    work["type"] = "existing"
    work = work.reset_index(drop=True)
    work["edge_id"] = np.arange(len(work), dtype=int)
    route_nodes = selected[["type", "route_label", "stop_count"]].copy()
    return work, route_nodes


def _route_pattern_tables(overlay: gpd.GeoDataFrame, route_nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_class = (
        overlay.groupby(["type", "route_label", "street_pattern_class"], as_index=False)
        .agg(pt_length_m=("intersect_length_m", "sum"))
    )
    route_totals = route_class.groupby(["type", "route_label"], as_index=False)["pt_length_m"].sum()
    route_totals = route_totals.rename(columns={"pt_length_m": "route_total_m"})
    route_class = route_class.merge(route_totals, on=["type", "route_label"], how="left")
    route_class["route_class_share"] = np.where(
        route_class["route_total_m"] > 0,
        route_class["pt_length_m"] / route_class["route_total_m"],
        0.0,
    )

    observed = route_class[route_class["pt_length_m"] > 0].copy()
    observed["p_log_p"] = np.where(
        observed["route_class_share"] > 0,
        observed["route_class_share"] * np.log(observed["route_class_share"]),
        0.0,
    )
    entropy = (
        observed.groupby(["type", "route_label"], as_index=False)
        .agg(
            street_pattern_class_count=("street_pattern_class", "nunique"),
            entropy_raw=("p_log_p", lambda s: -float(s.sum())),
        )
    )
    entropy["street_pattern_entropy"] = np.where(
        entropy["street_pattern_class_count"] > 1,
        entropy["entropy_raw"] / np.log(entropy["street_pattern_class_count"]),
        0.0,
    )
    entropy = entropy.drop(columns=["entropy_raw"])

    dominant = (
        route_class.sort_values(["type", "route_label", "pt_length_m"], ascending=[True, True, False])
        .groupby(["type", "route_label"], as_index=False)
        .first()
        .rename(
            columns={
                "street_pattern_class": "dominant_street_pattern_class",
                "pt_length_m": "dominant_class_length_m",
                "route_class_share": "dominant_class_share",
            }
        )
    )
    route_stats = (
        route_nodes.merge(route_totals, on=["type", "route_label"], how="left")
        .merge(entropy, on=["type", "route_label"], how="left")
        .merge(
            dominant[
                [
                    "type",
                    "route_label",
                    "dominant_street_pattern_class",
                    "dominant_class_length_m",
                    "dominant_class_share",
                ]
            ],
            on=["type", "route_label"],
            how="left",
        )
    )
    route_stats["route_total_m"] = route_stats["route_total_m"].fillna(0.0)
    route_stats["route_total_km"] = route_stats["route_total_m"] / 1000.0
    route_stats["street_pattern_class_count"] = route_stats["street_pattern_class_count"].fillna(0).astype(int)
    route_stats["street_pattern_entropy"] = route_stats["street_pattern_entropy"].fillna(0.0)
    route_stats = route_stats.sort_values(["type", "route_label"]).reset_index(drop=True)
    route_class = route_class.sort_values(["type", "route_label", "pt_length_m"], ascending=[True, True, False]).reset_index(drop=True)
    return route_class, route_stats


def _existing_route_stop_sets(
    existing_edges: gpd.GeoDataFrame,
    graph_nodes: gpd.GeoDataFrame,
    *,
    selected_labels: list[str],
    threshold_m: float,
) -> list[set[int]]:
    edges = existing_edges
    nodes = graph_nodes
    if edges.crs is None:
        edges = edges.set_crs(nodes.crs)
    if str(edges.crs) != str(nodes.crs):
        edges = edges.to_crs(nodes.crs)
    stop_sets: list[set[int]] = []
    for label in selected_labels:
        part = edges[edges["route_label"].astype(str) == str(label)]
        if part.empty:
            stop_sets.append(set())
            continue
        route_geom = part.geometry.union_all()
        distances = nodes.geometry.distance(route_geom)
        matched = nodes.loc[distances <= threshold_m, "order_index"].astype(int)
        stop_sets.append(set(matched.tolist()))
    return stop_sets


def _direct_od_coverage(
    od: np.ndarray,
    route_stop_sets: list[set[int]],
    *,
    scenario: str,
) -> pd.DataFrame:
    if od.ndim != 2 or od.shape[0] != od.shape[1]:
        raise ValueError(f"OD matrix must be square, got shape={od.shape}")
    n = od.shape[0]
    total_od = float(np.nansum(od))
    positive_pairs = (od > 0).copy()
    np.fill_diagonal(positive_pairs, False)
    total_pairs = int(positive_pairs.sum())
    covered = np.zeros((n, n), dtype=bool)
    rows = []
    for route_count, stop_set in enumerate(route_stop_sets, start=1):
        valid = sorted(idx for idx in stop_set if 0 <= idx < n)
        if valid:
            covered[np.ix_(valid, valid)] = True
        np.fill_diagonal(covered, False)
        covered_od = float(np.nansum(np.where(covered, od, 0.0)))
        covered_positive = covered & positive_pairs
        rows.append(
            {
                "scenario": scenario,
                "route_count": route_count,
                "covered_od": covered_od,
                "total_od": total_od,
                "covered_share": covered_od / total_od if total_od > 0 else 0.0,
                "covered_pairs": int(covered_positive.sum()),
                "total_pairs": total_pairs,
            }
        )
    return pd.DataFrame(rows)


def _class_colors(cells_local: gpd.GeoDataFrame) -> dict[str, str]:
    classes = sorted(cells_local["street_pattern_class"].astype("string").fillna("unknown").unique().tolist())
    ordered = order_street_pattern_classes(classes)
    palette = get_palette("street_patterns")
    return {cls: palette.get(cls, palette.get("unknown", "#d1d5db")) for cls in ordered}


def _plot_route_map(
    path: Path,
    *,
    title: str,
    cells_local: gpd.GeoDataFrame,
    routes: gpd.GeoDataFrame,
    route_stats: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_colors = _class_colors(cells_local)
    route_counts = route_stats.set_index("route_label")["street_pattern_class_count"].to_dict()
    labels = route_stats.set_index("route_label")["stop_count"].to_dict()
    max_count = max(route_counts.values()) if route_counts else 1
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(11, 11))
    fig.patch.set_facecolor("#f4f1ea")
    ax.set_facecolor("#f4f1ea")
    cell_colors = cells_local["street_pattern_class"].astype("string").fillna("unknown").map(class_colors).fillna("#d1d5db")
    cells_local.plot(ax=ax, color=cell_colors, alpha=0.32, linewidth=0.16, edgecolor=CANVAS_GRID, zorder=1)

    route_plot = routes.to_crs(cells_local.crs) if str(routes.crs) != str(cells_local.crs) else routes
    handles = []
    for idx, (route_label, part) in enumerate(route_plot.groupby("route_label", sort=False)):
        count = int(route_counts.get(route_label, 0))
        color = cmap(count / max_count if max_count else 0.0)
        part.plot(ax=ax, color=color, linewidth=2.5, alpha=0.95, zorder=3)
        union = part.geometry.union_all()
        point = union.representative_point()
        ax.annotate(
            f"{route_label} | {count} cls | {int(labels.get(route_label, 0))} stops",
            xy=(point.x, point.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=CANVAS_INK,
            bbox={"boxstyle": "round,pad=0.18", "fc": "#fffaf0", "ec": "#d6d3d1", "alpha": 0.88},
            zorder=5,
        )
        handles.append(Line2D([0], [0], color=color, linewidth=2.5, label=f"{route_label}: {count} classes"))
    if handles:
        ax.legend(handles=handles, loc="lower left", frameon=True, facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE, fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold", color=CANVAS_INK, pad=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_class_count_bars(path: Path, route_stats: pd.DataFrame) -> None:
    order = route_stats.sort_values(["scenario", "route_label"])["route_key"].tolist()
    colors = route_stats["scenario"].map({"existing": "#2563eb", "generated": "#d97706"}).fillna("#64748b")
    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.patch.set_facecolor("#f4f1ea")
    ax.set_facecolor("#fffaf0")
    ax.bar(route_stats["route_key"], route_stats["street_pattern_class_count"], color=colors, alpha=0.9)
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
    ax.set_ylabel("Street-Pattern Classes Crossed", color=CANVAS_INK)
    ax.set_title("Street-pattern breadth per route", fontsize=14, fontweight="bold", color=CANVAS_INK, pad=10)
    ax.grid(axis="y", alpha=0.22, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CANVAS_GRID)
    ax.spines["bottom"].set_color(CANVAS_GRID)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#2563eb", linewidth=8, label="existing"),
            Line2D([0], [0], color="#d97706", linewidth=8, label="generated"),
        ],
        frameon=True,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_od_coverage(path: Path, coverage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    fig.patch.set_facecolor("#f4f1ea")
    ax.set_facecolor("#fffaf0")
    colors = {"existing": "#2563eb", "generated": "#d97706"}
    for scenario, part in coverage.groupby("scenario", sort=False):
        part = part.sort_values("route_count")
        ax.plot(
            part["route_count"],
            part["covered_share"],
            marker="o",
            linewidth=2.2,
            color=colors.get(scenario, "#64748b"),
            label=scenario,
        )
    ax.set_xlabel("Routes Added", color=CANVAS_INK)
    ax.set_ylabel("Direct Covered OD Share", color=CANVAS_INK)
    ax.set_ylim(0, max(0.02, min(1.0, float(coverage["covered_share"].max()) * 1.18)))
    ax.set_title("Route count vs direct covered OD", fontsize=14, fontweight="bold", color=CANVAS_INK, pad=10)
    ax.grid(alpha=0.22, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(CANVAS_GRID)
    ax.spines["bottom"].set_color(CANVAS_GRID)
    ax.legend(frameon=True, facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE)
    fig.tight_layout()
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _assert_png_nonblank(path: Path) -> dict[str, int | float]:
    from PIL import Image, ImageStat

    image = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(image)
    extrema = image.getextrema()
    dynamic = sum(high - low for low, high in extrema)
    if dynamic <= 0:
        raise ValueError(f"PNG appears blank: {path}")
    return {"width": image.width, "height": image.height, "dynamic_range_sum": int(dynamic)}


def main() -> None:
    args = parse_args()
    city_dir = _require_path(args.city_dir, "city dir")
    generated_summary_path = _require_path(args.generated_summary, "generated route summary")
    output_dir = args.output_dir.resolve()
    stats_dir = output_dir / "stats"
    preview_dir = output_dir / "preview_png"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    graph_path = _resolve_generated_graph_path(city_dir, args.modality, args.generated_graph_path)
    graph_nodes_path = city_dir / "connectpt_osm" / args.modality / "graph_nodes.parquet"
    pt_edges_path = _require_path(city_dir / "pt_street_pattern_dependency" / "pt_edges_filtered.parquet", "current PT edges")
    current_route_stats_path = _require_path(city_dir / "pt_street_pattern_dependency" / "route_stats.csv", "current route stats")
    od_path = _resolve_od_matrix_path(generated_summary_path, args.modality, args.od_matrix_path)
    cells_path = _resolve_street_cells(city_dir)

    graph = _load_graph(graph_path)
    graph_crs = graph.graph.get("crs")
    if graph_crs is None and graph_nodes_path.exists():
        graph_crs = gpd.read_parquet(graph_nodes_path).crs
    graph_nodes = _graph_nodes_gdf(graph, graph_crs)
    cells = gpd.read_file(cells_path)
    class_col = _pick_class_column(cells, None)

    generated_summary = _read_json(generated_summary_path)
    generated_routes_all = _load_routes_from_summary(generated_summary)
    current_route_stats = pd.read_csv(current_route_stats_path)
    existing_modality_count = int(
        current_route_stats[current_route_stats["type"].astype("string").str.lower() == args.modality.lower()].shape[0]
    )
    route_count = int(args.route_count or (existing_modality_count if args.existing_route_policy == "all" else len(generated_routes_all)))
    if len(generated_routes_all) < route_count:
        raise ValueError(
            f"Generated route summary has {len(generated_routes_all)} routes, "
            f"but comparison needs {route_count} for policy={args.existing_route_policy!r}."
        )
    generated_routes = generated_routes_all[:route_count]
    target_stop_count = int(round(float(np.median([len(route) for route in generated_routes]))))
    selected_existing = _select_existing_routes(
        current_route_stats,
        modality=args.modality,
        target_stop_count=target_stop_count,
        route_count=route_count,
        policy=args.existing_route_policy,
    )
    pt_edges = gpd.read_parquet(pt_edges_path)
    existing_edges, existing_nodes = _existing_edges(pt_edges, selected_existing)
    generated_edges, generated_stop_sets, generated_nodes = _generated_edges(generated_routes, graph, graph_nodes.crs)

    existing_overlay, cells_local = _overlay_pt_with_street_pattern(existing_edges, cells, class_col=class_col)
    generated_overlay, _ = _overlay_pt_with_street_pattern(generated_edges, cells, class_col=class_col)
    existing_route_class, existing_stats = _route_pattern_tables(existing_overlay, existing_nodes)
    generated_route_class, generated_stats = _route_pattern_tables(generated_overlay, generated_nodes)

    existing_stats["scenario"] = "existing"
    generated_stats["scenario"] = "generated"
    existing_route_class["scenario"] = "existing"
    generated_route_class["scenario"] = "generated"
    route_stats = pd.concat([existing_stats, generated_stats], ignore_index=True)
    route_class = pd.concat([existing_route_class, generated_route_class], ignore_index=True)
    route_stats["route_key"] = route_stats["scenario"] + ":" + route_stats["route_label"].astype(str)

    selected_existing.to_csv(stats_dir / "selected_existing_routes.csv", index=False)
    route_stats.to_csv(stats_dir / "route_street_pattern_stats.csv", index=False)
    route_class.to_csv(stats_dir / "route_street_pattern_class_length.csv", index=False)
    existing_edges.to_parquet(stats_dir / "existing_route_edges.parquet", index=False)
    generated_edges.to_parquet(stats_dir / "generated_route_edges.parquet", index=False)

    selected_labels = selected_existing["route_label"].astype(str).tolist()
    existing_stop_sets = _existing_route_stop_sets(
        existing_edges,
        graph_nodes,
        selected_labels=selected_labels,
        threshold_m=float(args.existing_stop_match_threshold_m),
    )
    od = _read_od_matrix(od_path)
    coverage = pd.concat(
        [
            _direct_od_coverage(od, existing_stop_sets, scenario="existing"),
            _direct_od_coverage(od, generated_stop_sets, scenario="generated"),
        ],
        ignore_index=True,
    )
    coverage.to_csv(stats_dir / "od_coverage_by_route_count.csv", index=False)

    _plot_route_map(
        preview_dir / "01_existing_routes_street_pattern_count.png",
        title="Current routes: street-pattern classes crossed",
        cells_local=cells_local,
        routes=existing_edges,
        route_stats=existing_stats,
    )
    _plot_route_map(
        preview_dir / "02_generated_routes_street_pattern_count.png",
        title="Generated routes: street-pattern classes crossed",
        cells_local=cells_local,
        routes=generated_edges,
        route_stats=generated_stats,
    )
    _plot_class_count_bars(preview_dir / "03_route_street_pattern_class_count_comparison.png", route_stats)
    _plot_od_coverage(preview_dir / "04_route_count_vs_direct_od_coverage.png", coverage)

    preview_checks = {
        path.name: _assert_png_nonblank(path)
        for path in sorted(preview_dir.glob("*.png"))
    }
    summary = {
        "city_dir": str(city_dir),
        "generated_summary": str(generated_summary_path),
        "modality": args.modality,
        "route_count": route_count,
        "existing_route_policy": args.existing_route_policy,
        "existing_modality_route_count": existing_modality_count,
        "target_generated_stop_count": target_stop_count,
        "existing_stop_match_threshold_m": float(args.existing_stop_match_threshold_m),
        "street_pattern_cells": str(cells_path),
        "street_pattern_class_col": class_col,
        "generated_graph": str(graph_path),
        "od_matrix": str(od_path),
        "coverage_mode": "direct_same_route_on_connectpt_od",
        "counts": {
            "existing_routes": int(len(existing_stats)),
            "generated_routes": int(len(generated_stats)),
            "existing_edges": int(len(existing_edges)),
            "generated_edges": int(len(generated_edges)),
            "existing_overlay_segments": int(len(existing_overlay)),
            "generated_overlay_segments": int(len(generated_overlay)),
        },
        "selected_existing_routes": selected_existing[
            ["route_label", "stop_count", "stop_gap", "route_total_m"]
        ].to_dict(orient="records"),
        "generated_routes": generated_routes,
        "od_coverage_final": coverage.sort_values(["scenario", "route_count"])
        .groupby("scenario", as_index=False)
        .tail(1)
        .to_dict(orient="records"),
        "files": {
            "selected_existing_routes": str(stats_dir / "selected_existing_routes.csv"),
            "route_street_pattern_stats": str(stats_dir / "route_street_pattern_stats.csv"),
            "route_street_pattern_class_length": str(stats_dir / "route_street_pattern_class_length.csv"),
            "od_coverage_by_route_count": str(stats_dir / "od_coverage_by_route_count.csv"),
            "existing_route_edges": str(stats_dir / "existing_route_edges.parquet"),
            "generated_route_edges": str(stats_dir / "generated_route_edges.parquet"),
            "previews": {path.name: str(path) for path in sorted(preview_dir.glob("*.png"))},
        },
        "preview_checks": preview_checks,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()

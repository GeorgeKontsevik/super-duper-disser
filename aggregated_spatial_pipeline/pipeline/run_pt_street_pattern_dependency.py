from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from aggregated_spatial_pipeline.runtime_config import configure_logger


DEFAULT_PT_TYPES = ("bus", "tram", "trolleybus")
CLASS_COLUMN_CANDIDATES = ("top1_class_name", "class_name", "predicted_class", "street_pattern_class")


def _configure_logging() -> None:
    configure_logger("[pt_x_street_pattern]")


def _log(message: str) -> None:
    logger.bind(tag="[pt_x_street_pattern]").info(message)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay iDuedu PT graph routes (including subway network edges) with street-pattern cells "
            "and compute dependency summaries by class/modality/route."
        )
    )
    parser.add_argument("--joint-input-dir", default=None)
    parser.add_argument("--place", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pt-types", nargs="+", default=list(DEFAULT_PT_TYPES))
    parser.add_argument("--class-col", default=None)
    parser.add_argument("--top-routes", type=int, default=30)
    parser.add_argument("--min-segment-length-m", type=float, default=1.0)
    parser.add_argument(
        "--subway-stop-buffer-m",
        type=float,
        default=500.0,
        help="Buffer radius around subway stops for separate street-pattern analysis (meters). Set <=0 to disable.",
    )
    return parser.parse_args()


def _resolve_city_dir(args: argparse.Namespace) -> Path:
    if args.joint_input_dir:
        return Path(args.joint_input_dir).resolve()
    if args.place:
        slug = _slugify(str(args.place))
        return (
            Path(__file__).resolve().parents[2]
            / "aggregated_spatial_pipeline"
            / "outputs"
            / "joint_inputs"
            / slug
        ).resolve()
    raise ValueError("Provide either --joint-input-dir or --place.")


def _resolve_output_dir(city_dir: Path, args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    return (city_dir / "pt_street_pattern_dependency").resolve()


def _pick_class_column(cells: gpd.GeoDataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in cells.columns:
            raise KeyError(f"Requested class column is missing: {explicit}")
        return explicit
    for candidate in CLASS_COLUMN_CANDIDATES:
        if candidate in cells.columns:
            return candidate
    raise KeyError(f"Could not detect class column. Checked: {CLASS_COLUMN_CANDIDATES}")


def _prepare_pt_edges(
    edges: gpd.GeoDataFrame,
    *,
    pt_types: list[str],
    min_segment_length_m: float,
) -> gpd.GeoDataFrame:
    work = edges.copy()
    if work.empty:
        raise ValueError("Intermodal graph_edges layer is empty.")

    for col in ("u", "v", "type", "route"):
        if col not in work.columns:
            if col in ("route",):
                work[col] = None
            else:
                raise KeyError(f"Required column is missing in graph_edges: {col}")

    work["type"] = work["type"].astype("string").str.lower()
    allowed = {str(t).lower() for t in pt_types}
    work = work[work["type"].isin(allowed)].copy()
    if work.empty:
        raise ValueError(f"No PT edges for selected pt-types={sorted(allowed)}")

    work["length_meter"] = pd.to_numeric(work.get("length_meter"), errors="coerce")
    work = work[work["length_meter"].fillna(0.0) >= float(min_segment_length_m)].copy()
    work = work[work.geometry.notna() & ~work.geometry.is_empty].copy()
    if work.empty:
        raise ValueError("No PT edges with geometry and positive length after filtering.")

    route_raw = work["route"].astype("string").fillna("").str.strip()
    work["route_label"] = route_raw
    subway_mask = work["type"].eq("subway")
    empty_route = work["route_label"].eq("")
    work.loc[empty_route & subway_mask, "route_label"] = "subway_network"
    work.loc[empty_route & ~subway_mask, "route_label"] = work.loc[empty_route & ~subway_mask, "type"] + "_network"

    work["edge_a"] = np.minimum(pd.to_numeric(work["u"], errors="coerce"), pd.to_numeric(work["v"], errors="coerce"))
    work["edge_b"] = np.maximum(pd.to_numeric(work["u"], errors="coerce"), pd.to_numeric(work["v"], errors="coerce"))
    work["length_round_m"] = work["length_meter"].round(1)
    work = work.drop_duplicates(subset=["edge_a", "edge_b", "type", "route_label", "length_round_m"]).copy()
    work = work.reset_index(drop=True)
    work["edge_id"] = np.arange(len(work), dtype=int)
    return work


def _compute_route_node_counts(pt_edges: gpd.GeoDataFrame) -> pd.DataFrame:
    left = pt_edges[["type", "route_label", "u"]].rename(columns={"u": "node_id"})
    right = pt_edges[["type", "route_label", "v"]].rename(columns={"v": "node_id"})
    nodes = pd.concat([left, right], ignore_index=True)
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce")
    nodes = nodes.dropna(subset=["node_id"])
    stats = (
        nodes.groupby(["type", "route_label"], as_index=False)
        .agg(stop_count=("node_id", "nunique"))
        .sort_values(["type", "stop_count", "route_label"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return stats


def _overlay_pt_with_street_pattern(
    pt_edges: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    *,
    class_col: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cells_work = cells[[class_col, "geometry"]].copy().rename(columns={class_col: "street_pattern_class"})
    cells_work["street_pattern_class"] = cells_work["street_pattern_class"].astype("string").fillna("unknown")
    cells_work = cells_work[cells_work.geometry.notna() & ~cells_work.geometry.is_empty].copy()
    if cells_work.empty:
        raise ValueError("Street-pattern cells are empty after geometry cleanup.")

    if cells_work.crs is None:
        cells_work = cells_work.set_crs(4326)
    if pt_edges.crs is None:
        pt_edges = pt_edges.set_crs(4326)

    local_crs = cells_work.estimate_utm_crs() or "EPSG:3857"
    cells_local = cells_work.to_crs(local_crs)
    pt_local = pt_edges.to_crs(local_crs)

    overlay = gpd.overlay(
        pt_local[["edge_id", "type", "route_label", "length_meter", "geometry"]],
        cells_local[["street_pattern_class", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    overlay = overlay[overlay.geometry.notna() & ~overlay.geometry.is_empty].copy()
    if overlay.empty:
        raise ValueError("PT x street-pattern overlay is empty.")

    overlay["intersect_length_m"] = overlay.geometry.length
    overlay = overlay[overlay["intersect_length_m"] > 0].copy()
    if overlay.empty:
        raise ValueError("PT x street-pattern overlay contains no positive-length intersections.")

    return overlay, cells_local


def _compute_dependency_tables(
    overlay: gpd.GeoDataFrame,
    cells_local: gpd.GeoDataFrame,
    route_nodes: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    class_area = (
        cells_local.assign(class_area_m2=cells_local.geometry.area)
        .groupby("street_pattern_class", as_index=False)
        .agg(class_area_m2=("class_area_m2", "sum"))
    )
    class_area_total = float(class_area["class_area_m2"].sum())
    class_area["class_area_share"] = class_area["class_area_m2"] / class_area_total if class_area_total > 0 else 0.0

    class_len = (
        overlay.groupby("street_pattern_class", as_index=False)
        .agg(pt_length_m=("intersect_length_m", "sum"))
    )
    pt_total = float(class_len["pt_length_m"].sum())
    class_len["pt_length_share"] = class_len["pt_length_m"] / pt_total if pt_total > 0 else 0.0

    class_summary = class_area.merge(class_len, on="street_pattern_class", how="left")
    class_summary["pt_length_m"] = class_summary["pt_length_m"].fillna(0.0)
    class_summary["pt_length_share"] = class_summary["pt_length_share"].fillna(0.0)
    class_summary["pt_length_km"] = class_summary["pt_length_m"] / 1000.0
    class_summary["pt_length_per_km2"] = class_summary["pt_length_m"] / (class_summary["class_area_m2"] / 1_000_000.0)
    class_summary["pt_vs_area_location_quotient"] = np.where(
        class_summary["class_area_share"] > 0,
        class_summary["pt_length_share"] / class_summary["class_area_share"],
        np.nan,
    )
    class_summary = class_summary.sort_values("pt_length_m", ascending=False).reset_index(drop=True)

    class_modality = (
        overlay.groupby(["type", "street_pattern_class"], as_index=False)
        .agg(pt_length_m=("intersect_length_m", "sum"))
    )
    mod_totals = class_modality.groupby("type", as_index=False)["pt_length_m"].sum().rename(columns={"pt_length_m": "modality_total_m"})
    class_modality = class_modality.merge(mod_totals, on="type", how="left")
    class_modality["modality_class_share"] = np.where(
        class_modality["modality_total_m"] > 0,
        class_modality["pt_length_m"] / class_modality["modality_total_m"],
        0.0,
    )
    class_modality = class_modality.merge(
        class_area[["street_pattern_class", "class_area_share"]],
        on="street_pattern_class",
        how="left",
    )
    class_modality["modality_vs_area_lq"] = np.where(
        class_modality["class_area_share"] > 0,
        class_modality["modality_class_share"] / class_modality["class_area_share"],
        np.nan,
    )
    class_modality = class_modality.sort_values(["type", "pt_length_m"], ascending=[True, False]).reset_index(drop=True)

    route_class = (
        overlay.groupby(["type", "route_label", "street_pattern_class"], as_index=False)
        .agg(pt_length_m=("intersect_length_m", "sum"))
    )
    route_totals = route_class.groupby(["type", "route_label"], as_index=False)["pt_length_m"].sum().rename(
        columns={"pt_length_m": "route_total_m"}
    )
    route_class = route_class.merge(route_totals, on=["type", "route_label"], how="left")
    route_class["route_class_share"] = np.where(route_class["route_total_m"] > 0, route_class["pt_length_m"] / route_class["route_total_m"], 0.0)

    dominant = (
        route_class.sort_values(["type", "route_label", "pt_length_m"], ascending=[True, True, False])
        .groupby(["type", "route_label"], as_index=False)
        .first()
        .rename(columns={"street_pattern_class": "dominant_street_pattern_class", "pt_length_m": "dominant_class_length_m"})
    )
    route_stats = route_totals.merge(route_nodes, on=["type", "route_label"], how="left").merge(
        dominant[["type", "route_label", "dominant_street_pattern_class", "dominant_class_length_m", "route_class_share"]],
        on=["type", "route_label"],
        how="left",
    )
    route_stats["route_total_km"] = route_stats["route_total_m"] / 1000.0
    route_stats = route_stats.sort_values(["type", "route_total_m"], ascending=[True, False]).reset_index(drop=True)

    return {
        "class_summary": class_summary,
        "class_modality": class_modality,
        "route_class": route_class,
        "route_stats": route_stats,
    }


def _empty_geo_like(reference: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    crs = reference.crs if reference is not None else None
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def _prepare_subway_stops(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = nodes.copy()
    if work.empty or "geometry" not in work.columns:
        return _empty_geo_like(nodes)

    mask = pd.Series(False, index=work.index)
    if "type" in work.columns:
        node_type = work["type"].astype("string").str.lower()
        mask = mask | node_type.str.contains("subway", na=False)
    if "modality" in work.columns:
        modality = work["modality"].astype("string").str.lower()
        mask = mask | modality.eq("subway")

    work = work[mask].copy()
    if work.empty:
        return _empty_geo_like(nodes)

    work = work[work.geometry.notna() & ~work.geometry.is_empty].copy()
    if work.empty:
        return _empty_geo_like(nodes)

    geom_type = work.geometry.geom_type.astype("string")
    multipoint_mask = geom_type.eq("MultiPoint")
    if multipoint_mask.any():
        work.loc[multipoint_mask, "geometry"] = work.loc[multipoint_mask, "geometry"].centroid
    work = work[work.geometry.geom_type.eq("Point")].copy()
    if work.empty:
        return _empty_geo_like(nodes)

    work["_geom_wkb"] = work.geometry.to_wkb()
    work = work.drop_duplicates(subset=["_geom_wkb"]).copy()
    work = work.drop(columns=["_geom_wkb"])
    work = work.reset_index(drop=True)
    work["subway_stop_id"] = np.arange(len(work), dtype=int)
    return work


def _compute_subway_stop_buffer_tables(
    subway_stops: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
    *,
    class_col: str,
    buffer_m: float,
    local_crs: str | None = None,
) -> dict[str, object]:
    if subway_stops.empty:
        raise ValueError("No subway stops available for buffer analysis.")

    cells_work = cells[[class_col, "geometry"]].copy().rename(columns={class_col: "street_pattern_class"})
    cells_work["street_pattern_class"] = cells_work["street_pattern_class"].astype("string").fillna("unknown")
    cells_work = cells_work[cells_work.geometry.notna() & ~cells_work.geometry.is_empty].copy()
    if cells_work.empty:
        raise ValueError("Street-pattern cells are empty after geometry cleanup (subway buffer analysis).")

    if cells_work.crs is None:
        cells_work = cells_work.set_crs(4326)
    if subway_stops.crs is None:
        subway_stops = subway_stops.set_crs(4326)

    target_crs = local_crs or cells_work.estimate_utm_crs() or "EPSG:3857"
    cells_local = cells_work.to_crs(target_crs)
    stops_local = subway_stops.to_crs(target_crs)
    stops_local = stops_local[stops_local.geometry.notna() & ~stops_local.geometry.is_empty].copy()
    if stops_local.empty:
        raise ValueError("Subway stops are empty after projection cleanup.")

    buffers = stops_local[["subway_stop_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(float(buffer_m))
    buffers = buffers[buffers.geometry.notna() & ~buffers.geometry.is_empty].copy()
    if buffers.empty:
        raise ValueError("Subway stop buffers are empty.")

    if hasattr(buffers.geometry, "union_all"):
        union_geom = buffers.geometry.union_all()
    else:
        union_geom = buffers.geometry.unary_union
    if union_geom is None or union_geom.is_empty:
        raise ValueError("Failed to build subway buffer union geometry.")
    union_gdf = gpd.GeoDataFrame({"buffer_m": [float(buffer_m)]}, geometry=[union_geom], crs=target_crs)

    overlay = gpd.overlay(
        cells_local[["street_pattern_class", "geometry"]],
        union_gdf[["geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    overlay = overlay[overlay.geometry.notna() & ~overlay.geometry.is_empty].copy()
    if overlay.empty:
        raise ValueError("Subway buffer x street-pattern overlay is empty.")
    overlay["buffer_area_m2"] = overlay.geometry.area
    overlay = overlay[overlay["buffer_area_m2"] > 0].copy()
    if overlay.empty:
        raise ValueError("Subway buffer overlay has no positive-area intersections.")

    city_area = (
        cells_local.assign(city_area_m2=cells_local.geometry.area)
        .groupby("street_pattern_class", as_index=False)
        .agg(city_area_m2=("city_area_m2", "sum"))
    )
    city_total = float(city_area["city_area_m2"].sum())
    city_area["city_area_share"] = city_area["city_area_m2"] / city_total if city_total > 0 else 0.0

    buffer_area = (
        overlay.groupby("street_pattern_class", as_index=False)
        .agg(buffer_area_m2=("buffer_area_m2", "sum"))
    )
    buffer_total = float(buffer_area["buffer_area_m2"].sum())
    buffer_area["buffer_area_share"] = buffer_area["buffer_area_m2"] / buffer_total if buffer_total > 0 else 0.0

    summary = city_area.merge(buffer_area, on="street_pattern_class", how="left")
    summary["buffer_area_m2"] = summary["buffer_area_m2"].fillna(0.0)
    summary["buffer_area_share"] = summary["buffer_area_share"].fillna(0.0)
    summary["buffer_vs_city_lq"] = np.where(
        summary["city_area_share"] > 0,
        summary["buffer_area_share"] / summary["city_area_share"],
        np.nan,
    )
    summary = summary.sort_values("buffer_area_m2", ascending=False).reset_index(drop=True)

    stop_join = gpd.sjoin(
        stops_local[["subway_stop_id", "geometry"]],
        cells_local[["street_pattern_class", "geometry"]],
        how="left",
        predicate="within",
    )
    stop_join = stop_join.sort_values("subway_stop_id").drop_duplicates(subset=["subway_stop_id"], keep="first")
    stop_join["street_pattern_class"] = stop_join["street_pattern_class"].astype("string").fillna("unknown")
    stop_counts = (
        stop_join.groupby("street_pattern_class", as_index=False)
        .agg(stop_count=("subway_stop_id", "nunique"))
        .sort_values("stop_count", ascending=False)
        .reset_index(drop=True)
    )
    stop_total = float(stop_counts["stop_count"].sum())
    stop_counts["stop_share"] = stop_counts["stop_count"] / stop_total if stop_total > 0 else 0.0

    return {
        "summary": summary,
        "stop_counts": stop_counts,
        "stops_local": stops_local,
        "buffers": buffers,
        "buffer_union": union_gdf,
    }


def _save_previews(
    *,
    cells_local: gpd.GeoDataFrame,
    overlay: gpd.GeoDataFrame,
    class_modality: pd.DataFrame,
    output_dir: Path,
    city_dir: Path,
    subway_stops_local: gpd.GeoDataFrame | None = None,
    subway_buffer_union: gpd.GeoDataFrame | None = None,
    subway_buffer_m: float | None = None,
) -> dict[str, str]:
    preview_dir = output_dir / "preview_png"
    preview_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = city_dir / "preview_png" / "all_together"
    shared_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    classes = sorted(cells_local["street_pattern_class"].astype("string").fillna("unknown").unique().tolist())
    class_cmap = plt.get_cmap("tab20", max(len(classes), 1))
    class_colors: dict[str, tuple[float, float, float, float]] = {
        cls: class_cmap(i % class_cmap.N) for i, cls in enumerate(classes)
    }
    street_handles = [
        Patch(facecolor=class_colors[cls], edgecolor="#6b7280", alpha=0.35, label=cls)
        for cls in classes
    ]

    # Preview 1: map overlay
    fig, ax = plt.subplots(figsize=(11, 11))
    cell_colors = cells_local["street_pattern_class"].astype("string").fillna("unknown").map(class_colors)
    cells_local.plot(ax=ax, color=cell_colors, alpha=0.35, linewidth=0.2, edgecolor="#6b7280")

    color_map = {"bus": "#1d4ed8", "tram": "#059669", "trolleybus": "#d97706", "subway": "#7c3aed"}
    modality_handles: list[Line2D] = []
    for modality, color in color_map.items():
        part = overlay[overlay["type"] == modality]
        if part.empty:
            continue
        part.plot(ax=ax, color=color, linewidth=1.2, alpha=0.9, label=modality)
        modality_handles.append(Line2D([0], [0], color=color, linewidth=2, label=modality))

    context_handles: list[Line2D] = []
    if subway_buffer_union is not None and not subway_buffer_union.empty:
        buffer_plot = subway_buffer_union
        if (
            buffer_plot.crs is not None
            and cells_local.crs is not None
            and str(buffer_plot.crs) != str(cells_local.crs)
        ):
            buffer_plot = buffer_plot.to_crs(cells_local.crs)
        buffer_plot.boundary.plot(ax=ax, color="#7f1d1d", linewidth=1.8, alpha=0.95)
        if subway_buffer_m is not None and subway_buffer_m > 0:
            label = f"subway {int(round(subway_buffer_m))}m buffer"
        else:
            label = "subway buffer"
        context_handles.append(Line2D([0], [0], color="#7f1d1d", linewidth=2, label=label))

    if subway_stops_local is not None and not subway_stops_local.empty:
        stops_plot = subway_stops_local
        if (
            stops_plot.crs is not None
            and cells_local.crs is not None
            and str(stops_plot.crs) != str(cells_local.crs)
        ):
            stops_plot = stops_plot.to_crs(cells_local.crs)
        stops_plot.plot(ax=ax, color="#be185d", markersize=14, alpha=0.9)
        context_handles.append(
            Line2D([0], [0], marker="o", linestyle="None", color="#be185d", markersize=6, label="subway stops")
        )

    if street_handles:
        street_legend = ax.legend(
            handles=street_handles,
            loc="upper left",
            title="Street pattern",
            frameon=True,
            fontsize=8,
        )
        ax.add_artist(street_legend)
    combined_handles = [*modality_handles, *context_handles]
    if combined_handles:
        ax.legend(handles=combined_handles, loc="lower left", title="PT context", frameon=True)
    ax.set_title("PT network overlay on street-pattern classes")
    ax.set_axis_off()
    map_path = preview_dir / "pt_street_pattern_overlay_map.png"
    fig.tight_layout()
    fig.savefig(map_path, dpi=180)
    plt.close(fig)
    outputs["overlay_map"] = str(map_path)

    # Preview 2: modality shares by street-pattern class
    pivot = (
        class_modality.pivot_table(
            index="street_pattern_class",
            columns="type",
            values="pt_length_m",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )
    if not pivot.empty:
        shares = pivot.div(pivot.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        fig, ax = plt.subplots(figsize=(12, 6))
        shares.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[color_map.get(col, "#6b7280") for col in shares.columns],
        )
        ax.set_ylabel("Share within class")
        ax.set_xlabel("Street pattern class")
        ax.set_ylim(0, 1.0)
        ax.set_title("PT modality composition by street-pattern class")
        ax.legend(title="Modality", loc="upper right")
        fig.tight_layout()
        bar_path = preview_dir / "pt_street_pattern_modality_shares.png"
        fig.savefig(bar_path, dpi=180)
        plt.close(fig)
        outputs["modality_shares"] = str(bar_path)

    for key, src in list(outputs.items()):
        src_path = Path(src)
        dst_path = shared_dir / src_path.name
        try:
            shutil.copy2(src_path, dst_path)
            outputs[f"{key}_shared"] = str(dst_path)
        except Exception:
            pass

    return outputs


def main() -> None:
    _configure_logging()
    args = parse_args()
    city_dir = _resolve_city_dir(args)
    output_dir = _resolve_output_dir(city_dir, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    edges_path = city_dir / "intermodal_graph_iduedu" / "graph_edges.parquet"
    nodes_path = city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet"
    cells_path = city_dir / "street_pattern" / city_dir.name / "predicted_cells.geojson"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing intermodal edges: {edges_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing intermodal nodes: {nodes_path}")
    if not cells_path.exists():
        raise FileNotFoundError(f"Missing street-pattern cells: {cells_path}")

    _log(f"Loading intermodal PT edges: {edges_path.name}")
    edges = gpd.read_parquet(edges_path)
    _log(f"Loading intermodal PT nodes: {nodes_path.name}")
    nodes = gpd.read_parquet(nodes_path)
    _log(f"Loading street-pattern cells: {cells_path.name}")
    cells = gpd.read_file(cells_path)
    class_col = _pick_class_column(cells, args.class_col)

    pt_types = [str(t).lower() for t in args.pt_types]
    pt_edges = _prepare_pt_edges(edges, pt_types=pt_types, min_segment_length_m=float(args.min_segment_length_m))
    route_nodes = _compute_route_node_counts(pt_edges)
    overlay, cells_local = _overlay_pt_with_street_pattern(pt_edges, cells, class_col=class_col)
    tables = _compute_dependency_tables(overlay, cells_local, route_nodes)

    subway_result: dict[str, object] | None = None
    subway_files: dict[str, str] = {}
    subway_manifest: dict[str, object] = {"enabled": False}
    buffer_m = float(args.subway_stop_buffer_m)
    if buffer_m > 0:
        subway_stops = _prepare_subway_stops(nodes)
        if subway_stops.empty:
            _log("Subway stop buffer analysis: no subway stops found in graph_nodes; skipping.")
            subway_manifest = {"enabled": False, "reason": "no_subway_stops_found", "buffer_m": buffer_m}
        else:
            subway_result = _compute_subway_stop_buffer_tables(
                subway_stops=subway_stops,
                cells=cells,
                class_col=class_col,
                buffer_m=buffer_m,
                local_crs=str(cells_local.crs) if cells_local.crs is not None else None,
            )
            subway_stops_out = output_dir / "subway_stops_filtered.parquet"
            subway_buffers_out = output_dir / "subway_stop_buffers.parquet"
            subway_summary_out = output_dir / "subway_stop_buffer_class_summary.csv"
            subway_stop_counts_out = output_dir / "subway_stop_class_counts.csv"

            subway_stops.to_parquet(subway_stops_out)
            buffers = subway_result["buffers"]
            assert isinstance(buffers, gpd.GeoDataFrame)
            buffers.to_parquet(subway_buffers_out)
            summary = subway_result["summary"]
            assert isinstance(summary, pd.DataFrame)
            summary.to_csv(subway_summary_out, index=False)
            stop_counts = subway_result["stop_counts"]
            assert isinstance(stop_counts, pd.DataFrame)
            stop_counts.to_csv(subway_stop_counts_out, index=False)

            subway_files = {
                "subway_stops_parquet": str(subway_stops_out),
                "subway_stop_buffers_parquet": str(subway_buffers_out),
                "subway_buffer_class_summary_csv": str(subway_summary_out),
                "subway_stop_class_counts_csv": str(subway_stop_counts_out),
            }
            subway_manifest = {
                "enabled": True,
                "buffer_m": buffer_m,
                "counts": {
                    "subway_stops": int(len(subway_stops)),
                    "buffer_classes": int(len(summary)),
                },
                "files": subway_files,
            }
    else:
        subway_manifest = {"enabled": False, "reason": "disabled_by_flag", "buffer_m": buffer_m}

    pt_edges_out = output_dir / "pt_edges_filtered.parquet"
    overlay_out = output_dir / "pt_street_pattern_overlay.parquet"
    class_summary_out = output_dir / "class_dependency_summary.csv"
    class_modality_out = output_dir / "class_modality_length.csv"
    route_stats_out = output_dir / "route_stats.csv"
    route_class_out = output_dir / "route_class_length.csv"

    pt_edges.to_parquet(pt_edges_out)
    overlay.to_parquet(overlay_out)
    tables["class_summary"].to_csv(class_summary_out, index=False)
    tables["class_modality"].to_csv(class_modality_out, index=False)
    tables["route_stats"].to_csv(route_stats_out, index=False)
    tables["route_class"].to_csv(route_class_out, index=False)

    previews = _save_previews(
        cells_local=cells_local,
        overlay=overlay,
        class_modality=tables["class_modality"],
        output_dir=output_dir,
        city_dir=city_dir,
        subway_stops_local=subway_result["stops_local"] if subway_result is not None else None,
        subway_buffer_union=subway_result["buffer_union"] if subway_result is not None else None,
        subway_buffer_m=buffer_m if buffer_m > 0 else None,
    )

    route_stats = tables["route_stats"].copy()
    if int(args.top_routes) > 0:
        route_stats = route_stats.head(int(args.top_routes))

    manifest = {
        "city_dir": str(city_dir),
        "experiment": "pt_x_street_pattern_dependency",
        "pt_types": pt_types,
        "street_pattern_class_col": class_col,
        "subway_stop_buffer": subway_manifest,
        "counts": {
            "street_pattern_cells": int(len(cells)),
            "pt_edges_filtered": int(len(pt_edges)),
            "overlay_segments": int(len(overlay)),
            "routes_total": int(tables["route_stats"][["type", "route_label"]].drop_duplicates().shape[0]),
            "subway_routes_without_explicit_route_label": int(
                (
                    (pt_edges["type"] == "subway")
                    & (pt_edges["route_label"] == "subway_network")
                ).sum()
            ),
        },
        "files": {
            "pt_edges_filtered": str(pt_edges_out),
            "overlay": str(overlay_out),
            "class_summary_csv": str(class_summary_out),
            "class_modality_csv": str(class_modality_out),
            "route_stats_csv": str(route_stats_out),
            "route_class_csv": str(route_class_out),
            **subway_files,
            **previews,
        },
        "top_routes_preview": route_stats.to_dict(orient="records"),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.pipeline.run_pt_street_pattern_dependency import (
    _compute_dependency_tables,
    _compute_route_node_counts,
    _overlay_pt_with_street_pattern,
    _pick_class_column,
    _prepare_pt_edges,
)
from aggregated_spatial_pipeline.runtime_config import configure_logger, ensure_repo_mplconfigdir
from aggregated_spatial_pipeline.visualization import (
    footer_text,
    get_palette,
    order_street_pattern_classes,
    save_preview_figure,
)


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT
SUBPROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SUBPROJECT_ROOT / "outputs"
DEFAULT_JOINT_INPUT_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs"
DEFAULT_SERVICE_ACCESSIBILITY_ROOT = (
    REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "experiments_active19_20260412" / "service_accessibility_street_pattern"
)
DEFAULT_CITIES = ("warsaw_poland", "berlin_germany")
DEFAULT_MODALITIES = ("bus", "tram", "trolleybus")
DEFAULT_CLASS_ID_NAME = {
    0: "Loops & Lollipops",
    1: "Irregular Grid",
    2: "Regular Grid",
    3: "Warped Parallel",
    4: "Sparse",
    5: "Broken Grid",
}
STREET_PATTERN_PALETTE = get_palette("street_patterns")


def _reindex_street_pattern_columns(pivot: pd.DataFrame) -> pd.DataFrame:
    ordered = order_street_pattern_classes([str(c) for c in pivot.columns])
    return pivot.reindex(columns=ordered, fill_value=0.0)

ensure_repo_mplconfigdir("mpl-route-pattern-street-pattern", root=REPO_ROOT)


@dataclass(frozen=True)
class CityPaths:
    slug: str
    city_dir: Path
    output_dir: Path
    edges_path: Path
    nodes_path: Path
    street_cells_path: Path
    service_accessibility_blocks_path: Path | None


def _configure_logging() -> None:
    configure_logger("[route-pattern-street-pattern]")


def _log(message: str) -> None:
    logger.bind(tag="[route-pattern-street-pattern]").info(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse which street-pattern classes PT routes pass through, using simple route-length overlays."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=list(DEFAULT_CITIES),
        help="City slugs inside aggregated_spatial_pipeline/outputs/joint_inputs, or 'all'.",
    )
    parser.add_argument(
        "--joint-input-root",
        default=str(DEFAULT_JOINT_INPUT_ROOT),
        help="Root with prepared city bundles.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for this experiment subproject.",
    )
    parser.add_argument(
        "--service-accessibility-root",
        default=str(DEFAULT_SERVICE_ACCESSIBILITY_ROOT),
        help="Root with prepared service_accessibility blocks_experiment.parquet for population-vs-route comparison.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        help="PT modalities to analyse.",
    )
    parser.add_argument(
        "--class-col",
        default=None,
        help="Optional explicit street-pattern class column in predicted_cells.geojson.",
    )
    parser.add_argument(
        "--top-routes",
        type=int,
        default=20,
        help="Number of longest routes to show in per-city preview.",
    )
    parser.add_argument(
        "--min-segment-length-m",
        type=float,
        default=1.0,
        help="Minimum PT segment length to keep.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild per-city prepared outputs.",
    )
    parser.add_argument(
        "--cross-city-only",
        action="store_true",
        help="Skip per-city outputs and compute only pooled cross-city outputs from prepared city tables.",
    )
    parser.add_argument(
        "--require-ready-data",
        action="store_true",
        help="Use only exact prepared city inputs; do not auto-discover fallback cell paths.",
    )
    return parser.parse_args()


def _resolve_city_paths(
    slug: str,
    joint_input_root: Path,
    output_root: Path,
    service_accessibility_root: Path | None,
) -> CityPaths:
    city_dir = (joint_input_root / slug).resolve()
    if not city_dir.exists():
        raise FileNotFoundError(f"City bundle not found: {city_dir}")
    paths = CityPaths(
        slug=slug,
        city_dir=city_dir,
        output_dir=(output_root / slug).resolve(),
        edges_path=city_dir / "intermodal_graph_iduedu" / "graph_edges.parquet",
        nodes_path=city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet",
        street_cells_path=city_dir / "street_pattern" / slug / "predicted_cells.geojson",
        service_accessibility_blocks_path=(
            (service_accessibility_root / slug / "prepared" / "blocks_experiment.parquet").resolve()
            if service_accessibility_root is not None
            else None
        ),
    )
    for check in (paths.edges_path, paths.nodes_path, paths.street_cells_path):
        if not check.exists():
            raise FileNotFoundError(f"Required input not found for city {slug}: {check}")
    return paths


def _discover_available_city_slugs(
    joint_input_root: Path,
    output_root: Path,
    service_accessibility_root: Path | None,
) -> list[str]:
    slugs: list[str] = []
    for city_dir in sorted(p for p in joint_input_root.iterdir() if p.is_dir()):
        try:
            _resolve_city_paths(city_dir.name, joint_input_root, output_root, service_accessibility_root)
        except Exception:
            continue
        slugs.append(city_dir.name)
    return slugs


def _plot_city_top_routes(route_stats: pd.DataFrame, output_path: Path, *, top_routes: int) -> None:
    if route_stats.empty:
        return
    plot_df = route_stats.sort_values("route_total_m", ascending=False).head(max(1, top_routes)).copy()
    plot_df["route_key"] = plot_df["type"].astype(str) + " | " + plot_df["route_label"].astype(str)
    plot_df["route_total_km"] = pd.to_numeric(plot_df["route_total_km"], errors="coerce").fillna(
        pd.to_numeric(plot_df["route_total_m"], errors="coerce").fillna(0.0) / 1000.0
    )
    present_classes = plot_df["dominant_street_pattern_class"].dropna().astype(str).unique().tolist()
    hue_order = order_street_pattern_classes(present_classes)
    hue_palette = {c: STREET_PATTERN_PALETTE.get(c, STREET_PATTERN_PALETTE.get("unknown", "#d1d5db")) for c in hue_order}
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.5 * len(plot_df) + 1)))
    sns.barplot(
        data=plot_df,
        x="route_total_km",
        y="route_key",
        hue="dominant_street_pattern_class",
        hue_order=hue_order if hue_order else None,
        palette=hue_palette if hue_palette else None,
        dodge=False,
        ax=ax,
    )
    ax.set_title("Longest Routes By Dominant Street Pattern", fontsize=16, fontweight="bold")
    ax.set_xlabel("route length (km)")
    ax.set_ylabel("")
    footer_text(fig, ["Bar color shows which street-pattern class dominates the route by length."], y=0.012)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_city_modality_heatmap(route_class: pd.DataFrame, output_path: Path) -> None:
    if route_class.empty:
        return
    plot_df = route_class.copy()
    plot_df["pt_length_m"] = pd.to_numeric(plot_df["pt_length_m"], errors="coerce").fillna(0.0)
    grouped = (
        plot_df.groupby(["type", "street_pattern_class"], as_index=False)["pt_length_m"].sum()
    )
    totals = grouped.groupby("type")["pt_length_m"].transform("sum")
    grouped["length_share"] = grouped["pt_length_m"] / totals.where(totals > 0, np.nan)
    pivot = grouped.pivot(index="type", columns="street_pattern_class", values="length_share").fillna(0.0)
    pivot = _reindex_street_pattern_columns(pivot)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.8 * len(pivot.index) + 1)))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.4, ax=ax, vmin=0.0, vmax=1.0)
    ax.set_title("Route Length Share By Modality And Street Pattern", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_cross_city_heatmap(city_class: pd.DataFrame, output_path: Path) -> None:
    if city_class.empty:
        return
    pivot = city_class.pivot(index="city", columns="street_pattern_class", values="route_length_share").fillna(0.0)
    pivot = _reindex_street_pattern_columns(pivot)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.45 * len(pivot.index) + 1.5)))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.3, ax=ax, vmin=0.0, vmax=1.0)
    ax.set_title("Cross-City Route Length Share By Street Pattern", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_cross_city_modality_canvas(
    city_modality_class: pd.DataFrame,
    output_path: Path,
    city_population_class: pd.DataFrame | None = None,
) -> None:
    if city_modality_class.empty:
        return
    present_modalities = sorted(city_modality_class["type"].dropna().astype(str).unique().tolist())
    preferred_modalities = ["bus", "tram", "trolleybus"]
    modalities = [m for m in preferred_modalities if (m in present_modalities or m == "trolleybus")]
    modalities.extend([m for m in present_modalities if m not in modalities])
    include_population = city_population_class is not None and not city_population_class.empty
    if not modalities and not include_population:
        return
    ncols = len(modalities) + (1 if include_population else 0)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=(5.4 * ncols, max(5.0, 0.45 * city_modality_class["city"].nunique() + 1.5)),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    city_order = sorted(city_modality_class["city"].dropna().astype(str).unique().tolist())
    class_order = order_street_pattern_classes(city_modality_class["street_pattern_class"].dropna().astype(str).unique().tolist())

    for ax, modality in zip(axes_flat[: len(modalities)], modalities):
        modality_df = city_modality_class[city_modality_class["type"].astype(str) == modality].copy()
        if modality_df.empty:
            pivot = pd.DataFrame(0.0, index=city_order, columns=class_order)
        else:
            pivot = modality_df.pivot(index="city", columns="street_pattern_class", values="route_length_share").fillna(0.0)
            pivot = pivot.reindex(index=city_order, columns=class_order, fill_value=0.0)
        pivot = _reindex_street_pattern_columns(pivot)
        if pivot.empty:
            ax.axis("off")
            continue
        sns.heatmap(pivot, cmap="PuBuGn", annot=True, fmt=".2f", linewidths=0.2, ax=ax, vmin=0.0, vmax=1.0, cbar=False)
        ax.set_title(modality, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("" if ax is not axes_flat[0] else "city")
    if include_population:
        pop_ax = axes_flat[len(modalities)]
        pop_pivot = city_population_class.pivot(index="city", columns="street_pattern_class", values="population_share").fillna(0.0)
        pop_pivot = _reindex_street_pattern_columns(pop_pivot)
        if pop_pivot.empty:
            pop_ax.axis("off")
        else:
            sns.heatmap(
                pop_pivot,
                cmap="Blues",
                annot=True,
                fmt=".2f",
                linewidths=0.2,
                ax=pop_ax,
                vmin=0.0,
                vmax=1.0,
                cbar=False,
            )
            pop_ax.set_title("population", fontsize=13, fontweight="bold")
            pop_ax.set_xlabel("")
            pop_ax.set_ylabel("")
    fig.suptitle("Cross-City Route Length Share By Street Pattern And Modality (+ Population)", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_cross_city_population_heatmap(city_population_class: pd.DataFrame, output_path: Path) -> None:
    if city_population_class.empty:
        return
    pivot = city_population_class.pivot(index="city", columns="street_pattern_class", values="population_share").fillna(0.0)
    pivot = _reindex_street_pattern_columns(pivot)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.45 * len(pivot.index) + 1.5)))
    sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".2f", linewidths=0.3, ax=ax, vmin=0.0, vmax=1.0)
    ax.set_title("Cross-City Population Share By Street Pattern", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_cross_city_population_modality_canvas(city_modality_population: pd.DataFrame, output_path: Path) -> None:
    if city_modality_population.empty:
        return
    modalities = sorted(city_modality_population["type"].dropna().astype(str).unique().tolist())
    if not modalities:
        return
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(modalities),
        figsize=(5.4 * len(modalities), max(5.0, 0.45 * city_modality_population["city"].nunique() + 1.5)),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, modality in zip(axes_flat, modalities):
        modality_df = city_modality_population[city_modality_population["type"].astype(str) == modality].copy()
        pivot = modality_df.pivot(index="city", columns="street_pattern_class", values="population_share").fillna(0.0)
        pivot = _reindex_street_pattern_columns(pivot)
        if pivot.empty:
            ax.axis("off")
            continue
        sns.heatmap(pivot, cmap="YlGn", annot=True, fmt=".2f", linewidths=0.2, ax=ax, vmin=0.0, vmax=1.0, cbar=False)
        ax.set_title(modality, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("" if ax is not axes_flat[0] else "city")
    fig.suptitle("Cross-City Population Share By Street Pattern And Modality Canvas", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_cross_city_route_population_gap_heatmap(city_gap: pd.DataFrame, output_path: Path) -> None:
    if city_gap.empty:
        return
    pivot = city_gap.pivot(index="city", columns="street_pattern_class", values="route_minus_population_share").fillna(0.0)
    pivot = _reindex_street_pattern_columns(pivot)
    if pivot.empty:
        return
    vmax = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float)))) if pivot.size else 0.0
    vmax = max(vmax, 0.05)
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.45 * len(pivot.index) + 1.5)))
    sns.heatmap(pivot, cmap="coolwarm", center=0.0, annot=True, fmt=".2f", linewidths=0.3, ax=ax, vmin=-vmax, vmax=vmax)
    ax.set_title("Route Share Minus Population Share By Street Pattern", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_cross_city_route_population_gap_modality_canvas(city_modality_gap: pd.DataFrame, output_path: Path) -> None:
    if city_modality_gap.empty:
        return
    modalities = sorted(city_modality_gap["type"].dropna().astype(str).unique().tolist())
    if not modalities:
        return
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(modalities),
        figsize=(5.4 * len(modalities), max(5.0, 0.45 * city_modality_gap["city"].nunique() + 1.5)),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    global_vmax = 0.05
    for modality in modalities:
        modality_df = city_modality_gap[city_modality_gap["type"].astype(str) == modality].copy()
        pivot = modality_df.pivot(index="city", columns="street_pattern_class", values="route_minus_population_share").fillna(0.0)
        pivot = _reindex_street_pattern_columns(pivot)
        if pivot.size:
            global_vmax = max(global_vmax, float(np.nanmax(np.abs(pivot.to_numpy(dtype=float)))))
    for ax, modality in zip(axes_flat, modalities):
        modality_df = city_modality_gap[city_modality_gap["type"].astype(str) == modality].copy()
        pivot = modality_df.pivot(index="city", columns="street_pattern_class", values="route_minus_population_share").fillna(0.0)
        pivot = _reindex_street_pattern_columns(pivot)
        if pivot.empty:
            ax.axis("off")
            continue
        sns.heatmap(
            pivot,
            cmap="coolwarm",
            center=0.0,
            annot=True,
            fmt=".2f",
            linewidths=0.2,
            ax=ax,
            vmin=-global_vmax,
            vmax=global_vmax,
            cbar=False,
        )
        ax.set_title(modality, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("" if ax is not axes_flat[0] else "city")
    fig.suptitle("Route Share Minus Population Share By Street Pattern And Modality", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _load_city_population_pattern(paths: CityPaths) -> pd.DataFrame:
    if paths.service_accessibility_blocks_path is None or not paths.service_accessibility_blocks_path.exists():
        return pd.DataFrame()
    blocks = read_geodata(paths.service_accessibility_blocks_path)
    if blocks.empty:
        return pd.DataFrame()
    dominant = blocks.get("street_pattern_dominant_class")
    population = pd.to_numeric(blocks.get("population"), errors="coerce").fillna(0.0)
    if dominant is None or float(population.sum()) <= 0.0:
        return pd.DataFrame()
    grouped = (
        pd.DataFrame(
            {
                "city": paths.slug,
                "street_pattern_class": dominant.fillna("unknown").astype(str),
                "population": population,
            }
        )
        .groupby(["city", "street_pattern_class"], as_index=False)["population"]
        .sum()
    )
    grouped = grouped[grouped["street_pattern_class"].astype(str).str.strip().str.lower() != "unknown"].copy()
    if grouped.empty:
        return pd.DataFrame()
    total = float(grouped["population"].sum())
    grouped["population_share"] = grouped["population"] / total if total > 0 else np.nan
    return grouped


def _build_class_id_name_map(cells: gpd.GeoDataFrame) -> dict[int, str]:
    mapping: dict[int, str] = dict(DEFAULT_CLASS_ID_NAME)
    if {"class_id", "class_name"}.issubset(cells.columns):
        pairs = cells[["class_id", "class_name"]].dropna().copy()
        if not pairs.empty:
            pairs["class_id"] = pd.to_numeric(pairs["class_id"], errors="coerce")
            pairs = pairs.dropna(subset=["class_id"])
            for _, row in pairs.iterrows():
                class_id = int(row["class_id"])
                class_name = str(row["class_name"]).strip()
                if class_name:
                    mapping[class_id] = class_name
    if {"top1_class_id", "top1_class_name"}.issubset(cells.columns):
        pairs = cells[["top1_class_id", "top1_class_name"]].dropna().copy()
        if not pairs.empty:
            pairs["top1_class_id"] = pd.to_numeric(pairs["top1_class_id"], errors="coerce")
            pairs = pairs.dropna(subset=["top1_class_id"])
            for _, row in pairs.iterrows():
                class_id = int(row["top1_class_id"])
                class_name = str(row["top1_class_name"]).strip()
                if class_name:
                    mapping[class_id] = class_name
    return mapping


def _compute_multivariate_route_class(
    paths: CityPaths,
    *,
    modalities: list[str],
    min_segment_length_m: float,
) -> pd.DataFrame:
    edges = gpd.read_parquet(paths.edges_path)
    cells = gpd.read_file(paths.street_cells_path)
    pt_edges = _prepare_pt_edges(edges, pt_types=modalities, min_segment_length_m=float(min_segment_length_m))

    prob_cols: list[str] = []
    for col in cells.columns:
        col_s = str(col)
        if re.fullmatch(r"prob_\d+", col_s):
            prob_cols.append(col_s)
    prob_cols = sorted(prob_cols, key=lambda c: int(c.split("_", 1)[1]))
    if not prob_cols:
        return pd.DataFrame(columns=["type", "route_label", "street_pattern_class", "pt_length_m", "route_total_m", "route_class_share", "city"])

    class_map = _build_class_id_name_map(cells)
    keep_cols = ["geometry"] + [c for c in prob_cols if c in cells.columns]
    cells_prob = cells[keep_cols].copy()
    cells_prob = cells_prob[cells_prob.geometry.notna() & ~cells_prob.geometry.is_empty].copy()
    if cells_prob.empty:
        return pd.DataFrame(columns=["type", "route_label", "street_pattern_class", "pt_length_m", "route_total_m", "route_class_share", "city"])
    if cells_prob.crs is None:
        cells_prob = cells_prob.set_crs(4326)
    local_crs = cells_prob.estimate_utm_crs() or "EPSG:3857"
    cells_local = cells_prob.to_crs(local_crs)
    pt_local = pt_edges.to_crs(cells_local.crs)

    overlay = gpd.overlay(
        pt_local[["edge_id", "type", "route_label", "length_meter", "geometry"]],
        cells_local[prob_cols + ["geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    overlay = overlay[overlay.geometry.notna() & ~overlay.geometry.is_empty].copy()
    if overlay.empty:
        return pd.DataFrame(columns=["type", "route_label", "street_pattern_class", "pt_length_m", "route_total_m", "route_class_share", "city"])
    overlay["intersect_length_m"] = overlay.geometry.length
    overlay = overlay[overlay["intersect_length_m"] > 0].copy()
    if overlay.empty:
        return pd.DataFrame(columns=["type", "route_label", "street_pattern_class", "pt_length_m", "route_total_m", "route_class_share", "city"])

    probs = overlay[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    prob_sum = probs.sum(axis=1)
    probs_norm = probs.div(prob_sum.where(prob_sum > 0, np.nan), axis=0).fillna(0.0)

    long_parts: list[pd.DataFrame] = []
    base = overlay[["type", "route_label", "intersect_length_m"]].copy()
    for col in prob_cols:
        class_id = int(col.split("_", 1)[1])
        class_name = class_map.get(class_id, f"class_{class_id}")
        contrib = base["intersect_length_m"] * probs_norm[col]
        mask = contrib > 0
        if not mask.any():
            continue
        part = base.loc[mask, ["type", "route_label"]].copy()
        part["street_pattern_class"] = class_name
        part["pt_length_m"] = contrib.loc[mask].to_numpy()
        long_parts.append(part)

    if not long_parts:
        return pd.DataFrame(columns=["type", "route_label", "street_pattern_class", "pt_length_m", "route_total_m", "route_class_share", "city"])
    route_class = pd.concat(long_parts, ignore_index=True)
    route_class = (
        route_class.groupby(["type", "route_label", "street_pattern_class"], as_index=False)["pt_length_m"]
        .sum()
    )
    route_class = route_class[
        route_class["street_pattern_class"].astype(str).str.strip().str.lower() != "unknown"
    ].copy()
    route_totals = route_class.groupby(["type", "route_label"], as_index=False)["pt_length_m"].sum().rename(
        columns={"pt_length_m": "route_total_m"}
    )
    route_class = route_class.merge(route_totals, on=["type", "route_label"], how="left")
    route_class["route_class_share"] = np.where(
        route_class["route_total_m"] > 0,
        route_class["pt_length_m"] / route_class["route_total_m"],
        0.0,
    )
    route_class["city"] = paths.slug
    return route_class


def _load_city_population_pattern_multivariate(paths: CityPaths) -> pd.DataFrame:
    if paths.service_accessibility_blocks_path is None or not paths.service_accessibility_blocks_path.exists():
        return pd.DataFrame()
    blocks = read_geodata(paths.service_accessibility_blocks_path)
    if blocks.empty:
        return pd.DataFrame()
    cells = gpd.read_file(paths.street_cells_path)
    class_map = _build_class_id_name_map(cells)

    pop_series = pd.to_numeric(blocks.get("population"), errors="coerce").fillna(0.0)
    if float(pop_series.sum()) <= 0.0:
        return pd.DataFrame()

    # Prefer block-level prob_* columns if available.
    prob_cols = sorted(
        [str(c) for c in blocks.columns if re.fullmatch(r"prob_\d+", str(c))],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    probs = pd.DataFrame(index=blocks.index)
    col_to_class: dict[str, str] = {}
    if prob_cols:
        probs = blocks[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
        for col in prob_cols:
            class_id = int(col.split("_", 1)[1])
            col_to_class[col] = class_map.get(class_id, f"class_{class_id}")
    else:
        # Fallback: use named probability columns if numeric ids are unavailable.
        named_cols = [str(c) for c in blocks.columns if str(c).startswith("street_pattern_prob_")]
        if not named_cols:
            return pd.DataFrame()
        probs = blocks[named_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
        for col in named_cols:
            suffix = col.removeprefix("street_pattern_prob_")
            pretty = suffix.replace("_", " ").title().replace("Lollipops", "& Lollipops")
            if pretty.lower() == "loops & lollipops":
                pretty = "Loops & Lollipops"
            col_to_class[col] = pretty

    if probs.empty:
        return pd.DataFrame()
    prob_sum = probs.sum(axis=1)
    probs_norm = probs.div(prob_sum.where(prob_sum > 0, np.nan), axis=0).fillna(0.0)

    parts: list[pd.DataFrame] = []
    for col in probs_norm.columns:
        class_name = col_to_class.get(str(col), str(col))
        contrib = pop_series * probs_norm[col]
        mask = contrib > 0
        if not mask.any():
            continue
        part = pd.DataFrame(
            {
                "city": paths.slug,
                "street_pattern_class": class_name,
                "population": contrib.loc[mask].to_numpy(),
            }
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    grouped = pd.concat(parts, ignore_index=True).groupby(
        ["city", "street_pattern_class"], as_index=False
    )["population"].sum()
    grouped = grouped[grouped["street_pattern_class"].astype(str).str.strip().str.lower() != "unknown"].copy()
    if grouped.empty:
        return pd.DataFrame()
    total = float(grouped["population"].sum())
    grouped["population_share"] = grouped["population"] / total if total > 0 else np.nan
    return grouped


def _city_outputs_exist(paths: CityPaths) -> bool:
    return (
        (paths.output_dir / "summary.json").exists()
        and (paths.output_dir / "stats" / "route_class_length.csv").exists()
        and (paths.output_dir / "stats" / "route_stats.csv").exists()
    )


def _write_city_outputs(
    paths: CityPaths,
    *,
    route_class: pd.DataFrame,
    route_stats: pd.DataFrame,
    class_summary: pd.DataFrame,
    class_modality: pd.DataFrame,
    summary: dict[str, object],
    top_routes: int,
) -> None:
    stats_dir = paths.output_dir / "stats"
    preview_dir = paths.output_dir / "preview_png"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    route_class.to_csv(stats_dir / "route_class_length.csv", index=False)
    route_stats.to_csv(stats_dir / "route_stats.csv", index=False)
    class_summary.to_csv(stats_dir / "class_dependency_summary.csv", index=False)
    class_modality.to_csv(stats_dir / "class_modality_length.csv", index=False)
    (paths.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_city_top_routes(route_stats, preview_dir / "01_top_routes_by_dominant_pattern.png", top_routes=top_routes)
    _plot_city_modality_heatmap(route_class, preview_dir / "02_modality_class_length_share_heatmap.png")


def _prepare_city_outputs(
    paths: CityPaths,
    *,
    modalities: list[str],
    class_col: str | None,
    top_routes: int,
    min_segment_length_m: float,
    no_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if _city_outputs_exist(paths) and not no_cache:
        route_class = pd.read_csv(paths.output_dir / "stats" / "route_class_length.csv")
        route_stats = pd.read_csv(paths.output_dir / "stats" / "route_stats.csv")
        return route_class, route_stats

    edges = gpd.read_parquet(paths.edges_path)
    nodes = gpd.read_parquet(paths.nodes_path)
    cells = gpd.read_file(paths.street_cells_path)
    class_column = _pick_class_column(cells, class_col)

    pt_edges = _prepare_pt_edges(edges, pt_types=modalities, min_segment_length_m=float(min_segment_length_m))
    route_nodes = _compute_route_node_counts(pt_edges)
    overlay, cells_local = _overlay_pt_with_street_pattern(pt_edges, cells, class_col=class_column)
    tables = _compute_dependency_tables(overlay, cells_local, route_nodes)
    route_class = tables["route_class"].copy()
    route_stats = tables["route_stats"].copy()
    route_class["city"] = paths.slug
    route_stats["city"] = paths.slug

    summary = {
        "city_slug": paths.slug,
        "route_count": int(len(route_stats)),
        "modalities": sorted(route_stats["type"].dropna().astype(str).unique().tolist()),
        "street_pattern_classes": sorted(route_class["street_pattern_class"].dropna().astype(str).unique().tolist()),
        "files": {
            "route_class_length": str((paths.output_dir / "stats" / "route_class_length.csv").resolve()),
            "route_stats": str((paths.output_dir / "stats" / "route_stats.csv").resolve()),
            "class_dependency_summary": str((paths.output_dir / "stats" / "class_dependency_summary.csv").resolve()),
            "class_modality_length": str((paths.output_dir / "stats" / "class_modality_length.csv").resolve()),
        },
    }
    _write_city_outputs(
        paths,
        route_class=route_class,
        route_stats=route_stats,
        class_summary=tables["class_summary"],
        class_modality=tables["class_modality"],
        summary=summary,
        top_routes=top_routes,
    )
    return route_class, route_stats


def _write_cross_city_outputs(
    output_root: Path,
    *,
    route_class_frames: list[pd.DataFrame],
    route_stats_frames: list[pd.DataFrame],
    city_population_frames: list[pd.DataFrame],
    route_class_multivariate_frames: list[pd.DataFrame] | None = None,
    city_population_multivariate_frames: list[pd.DataFrame] | None = None,
) -> None:
    cross_dir = output_root / "_cross_city"
    stats_dir = cross_dir / "stats"
    preview_dir = cross_dir / "preview_png"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    route_class_all = pd.concat(route_class_frames, ignore_index=True)
    route_class_all = route_class_all[
        route_class_all["street_pattern_class"].astype(str).str.strip().str.lower() != "unknown"
    ].copy()
    route_stats_all = pd.concat(route_stats_frames, ignore_index=True)

    route_class_all["pt_length_m"] = pd.to_numeric(route_class_all["pt_length_m"], errors="coerce").fillna(0.0)
    city_totals = (
        route_class_all.groupby("city", as_index=False)["pt_length_m"]
        .sum()
        .rename(columns={"pt_length_m": "city_total_route_length_m"})
    )
    city_class = (
        route_class_all.groupby(["city", "street_pattern_class"], as_index=False)["pt_length_m"]
        .sum()
        .merge(city_totals, on="city", how="left")
    )
    city_class["route_length_share"] = city_class["pt_length_m"] / city_class["city_total_route_length_m"].where(
        city_class["city_total_route_length_m"] > 0, np.nan
    )

    city_modality_totals = (
        route_class_all.groupby(["city", "type"], as_index=False)["pt_length_m"]
        .sum()
        .rename(columns={"pt_length_m": "city_modality_total_route_length_m"})
    )
    city_modality_class = (
        route_class_all.groupby(["city", "type", "street_pattern_class"], as_index=False)["pt_length_m"]
        .sum()
        .merge(city_modality_totals, on=["city", "type"], how="left")
    )
    city_modality_class["route_length_share"] = city_modality_class["pt_length_m"] / city_modality_class[
        "city_modality_total_route_length_m"
    ].where(city_modality_class["city_modality_total_route_length_m"] > 0, np.nan)

    pooled_modality_class = (
        route_class_all.groupby(["type", "street_pattern_class"], as_index=False)["pt_length_m"]
        .sum()
    )
    pooled_modality_totals = pooled_modality_class.groupby("type")["pt_length_m"].transform("sum")
    pooled_modality_class["route_length_share"] = pooled_modality_class["pt_length_m"] / pooled_modality_totals.where(
        pooled_modality_totals > 0, np.nan
    )

    city_population_class = pd.concat(city_population_frames, ignore_index=True) if city_population_frames else pd.DataFrame()
    city_modality_population = pd.DataFrame()
    city_route_population_gap = pd.DataFrame()
    city_modality_population_gap = pd.DataFrame()
    if not city_population_class.empty:
        city_modality_population = city_modality_class[["city", "type"]].drop_duplicates().merge(
            city_population_class[["city", "street_pattern_class", "population_share"]],
            on="city",
            how="left",
        )
        city_modality_population["population_share"] = pd.to_numeric(
            city_modality_population["population_share"], errors="coerce"
        ).fillna(0.0)
        city_route_population_gap = city_class.merge(
            city_population_class[["city", "street_pattern_class", "population_share"]],
            on=["city", "street_pattern_class"],
            how="outer",
        )
        city_route_population_gap["route_length_share"] = pd.to_numeric(
            city_route_population_gap["route_length_share"], errors="coerce"
        ).fillna(0.0)
        city_route_population_gap["population_share"] = pd.to_numeric(
            city_route_population_gap["population_share"], errors="coerce"
        ).fillna(0.0)
        city_route_population_gap["route_minus_population_share"] = (
            city_route_population_gap["route_length_share"] - city_route_population_gap["population_share"]
        )
        city_modality_population_gap = city_modality_class.merge(
            city_population_class[["city", "street_pattern_class", "population_share"]],
            on=["city", "street_pattern_class"],
            how="outer",
        )
        city_modality_population_gap["route_length_share"] = pd.to_numeric(
            city_modality_population_gap["route_length_share"], errors="coerce"
        ).fillna(0.0)
        city_modality_population_gap["population_share"] = pd.to_numeric(
            city_modality_population_gap["population_share"], errors="coerce"
        ).fillna(0.0)
        city_modality_population_gap["route_minus_population_share"] = (
            city_modality_population_gap["route_length_share"] - city_modality_population_gap["population_share"]
        )

    route_class_all.to_csv(stats_dir / "route_class_length_all_cities.csv", index=False)
    route_stats_all.to_csv(stats_dir / "route_stats_all_cities.csv", index=False)
    city_class.to_csv(stats_dir / "city_route_length_share_by_class.csv", index=False)
    city_modality_class.to_csv(stats_dir / "city_modality_route_length_share_by_class.csv", index=False)
    pooled_modality_class.to_csv(stats_dir / "pooled_modality_route_length_share_by_class.csv", index=False)
    if not city_population_class.empty:
        city_population_class.to_csv(stats_dir / "city_population_share_by_class.csv", index=False)
    if not city_route_population_gap.empty:
        city_route_population_gap.to_csv(stats_dir / "city_route_minus_population_share_by_class.csv", index=False)
    if not city_modality_population_gap.empty:
        city_modality_population_gap.to_csv(stats_dir / "city_modality_route_minus_population_share_by_class.csv", index=False)

    _plot_cross_city_heatmap(city_class, preview_dir / "01_city_class_route_length_share_heatmap.png")
    _plot_cross_city_modality_canvas(
        city_modality_class,
        preview_dir / "02_city_class_route_length_share_by_modality_canvas.png",
        city_population_class=city_population_class,
    )
    multivariate_canvas_path = preview_dir / "06_city_class_route_length_share_by_modality_canvas_multivariate.png"
    if route_class_multivariate_frames:
        route_class_multi_all = pd.concat(route_class_multivariate_frames, ignore_index=True)
        route_class_multi_all["pt_length_m"] = pd.to_numeric(route_class_multi_all["pt_length_m"], errors="coerce").fillna(0.0)
        route_class_multi_all = route_class_multi_all[
            route_class_multi_all["street_pattern_class"].astype(str).str.strip().str.lower() != "unknown"
        ].copy()
        route_class_multi_all.to_csv(stats_dir / "route_class_length_all_cities_multivariate.csv", index=False)
        city_modality_totals_multi = (
            route_class_multi_all.groupby(["city", "type"], as_index=False)["pt_length_m"]
            .sum()
            .rename(columns={"pt_length_m": "city_modality_total_route_length_m"})
        )
        city_modality_class_multi = (
            route_class_multi_all.groupby(["city", "type", "street_pattern_class"], as_index=False)["pt_length_m"]
            .sum()
            .merge(city_modality_totals_multi, on=["city", "type"], how="left")
        )
        city_modality_class_multi["route_length_share"] = city_modality_class_multi["pt_length_m"] / city_modality_class_multi[
            "city_modality_total_route_length_m"
        ].where(city_modality_class_multi["city_modality_total_route_length_m"] > 0, np.nan)
        city_modality_class_multi.to_csv(stats_dir / "city_modality_route_length_share_by_class_multivariate.csv", index=False)
        city_population_multi = (
            pd.concat(city_population_multivariate_frames, ignore_index=True)
            if city_population_multivariate_frames
            else pd.DataFrame()
        )
        if not city_population_multi.empty:
            city_population_multi.to_csv(stats_dir / "city_population_share_by_class_multivariate.csv", index=False)
        _plot_cross_city_modality_canvas(
            city_modality_class_multi,
            multivariate_canvas_path,
            city_population_class=(city_population_multi if not city_population_multi.empty else city_population_class),
        )
    if not city_population_class.empty:
        _plot_cross_city_population_heatmap(
            city_population_class,
            preview_dir / "03_city_class_population_share_heatmap.png",
        )
    if not city_route_population_gap.empty:
        _plot_cross_city_route_population_gap_heatmap(
            city_route_population_gap,
            preview_dir / "04_city_class_route_minus_population_share_heatmap.png",
        )
    if not city_modality_population_gap.empty:
        _plot_cross_city_route_population_gap_modality_canvas(
            city_modality_population_gap,
            preview_dir / "05_city_class_route_minus_population_share_by_modality_canvas.png",
        )

    summary = {
        "cities": sorted(route_stats_all["city"].dropna().astype(str).unique().tolist()),
        "city_count": int(route_stats_all["city"].nunique()),
        "route_count": int(len(route_stats_all)),
        "files": {
            "route_class_length_all_cities": str((stats_dir / "route_class_length_all_cities.csv").resolve()),
            "route_stats_all_cities": str((stats_dir / "route_stats_all_cities.csv").resolve()),
            "city_route_length_share_by_class": str((stats_dir / "city_route_length_share_by_class.csv").resolve()),
            "city_modality_route_length_share_by_class": str((stats_dir / "city_modality_route_length_share_by_class.csv").resolve()),
            "pooled_modality_route_length_share_by_class": str((stats_dir / "pooled_modality_route_length_share_by_class.csv").resolve()),
            "city_class_route_length_share_by_modality_canvas": str(
                (preview_dir / "02_city_class_route_length_share_by_modality_canvas.png").resolve()
            ),
        },
    }
    if not city_population_class.empty:
        summary["files"]["city_population_share_by_class"] = str((stats_dir / "city_population_share_by_class.csv").resolve())
        summary["files"]["city_class_population_share_heatmap"] = str(
            (preview_dir / "03_city_class_population_share_heatmap.png").resolve()
        )
    if route_class_multivariate_frames:
        summary["files"]["route_class_length_all_cities_multivariate"] = str(
            (stats_dir / "route_class_length_all_cities_multivariate.csv").resolve()
        )
        summary["files"]["city_modality_route_length_share_by_class_multivariate"] = str(
            (stats_dir / "city_modality_route_length_share_by_class_multivariate.csv").resolve()
        )
        if city_population_multivariate_frames:
            summary["files"]["city_population_share_by_class_multivariate"] = str(
                (stats_dir / "city_population_share_by_class_multivariate.csv").resolve()
            )
        summary["files"]["city_class_route_length_share_by_modality_canvas_multivariate"] = str(
            multivariate_canvas_path.resolve()
        )
    if not city_route_population_gap.empty:
        summary["files"]["city_route_minus_population_share_by_class"] = str(
            (stats_dir / "city_route_minus_population_share_by_class.csv").resolve()
        )
        summary["files"]["city_class_route_minus_population_share_heatmap"] = str(
            (preview_dir / "04_city_class_route_minus_population_share_heatmap.png").resolve()
        )
    if not city_modality_population_gap.empty:
        summary["files"]["city_modality_route_minus_population_share_by_class"] = str(
            (stats_dir / "city_modality_route_minus_population_share_by_class.csv").resolve()
        )
        summary["files"]["city_class_route_minus_population_share_by_modality_canvas"] = str(
            (preview_dir / "05_city_class_route_minus_population_share_by_modality_canvas.png").resolve()
        )
    (cross_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    _configure_logging()
    args = parse_args()
    joint_input_root = Path(args.joint_input_root).resolve()
    output_root = Path(args.output_root).resolve()
    service_accessibility_root = Path(args.service_accessibility_root).resolve() if args.service_accessibility_root else None
    output_root.mkdir(parents=True, exist_ok=True)

    if list(args.cities) == ["all"]:
        city_list = _discover_available_city_slugs(joint_input_root, output_root, service_accessibility_root)
    else:
        city_list = [str(city) for city in args.cities]

    _log("Running route-pattern vs street-pattern experiments for " + ", ".join(city_list))

    route_class_frames: list[pd.DataFrame] = []
    route_stats_frames: list[pd.DataFrame] = []
    city_population_frames: list[pd.DataFrame] = []
    route_class_multivariate_frames: list[pd.DataFrame] = []
    city_population_multivariate_frames: list[pd.DataFrame] = []
    if not args.cross_city_only:
        for slug in city_list:
            paths = _resolve_city_paths(slug, joint_input_root, output_root, service_accessibility_root)
            _log(f"[{slug}] extracting route/class length shares from PT overlay.")
            route_class, route_stats = _prepare_city_outputs(
                paths,
                modalities=list(args.modalities),
                class_col=args.class_col,
                top_routes=int(args.top_routes),
                min_segment_length_m=float(args.min_segment_length_m),
                no_cache=bool(args.no_cache),
            )
            route_class_frames.append(route_class)
            route_stats_frames.append(route_stats)
            route_class_multi = _compute_multivariate_route_class(
                paths,
                modalities=list(args.modalities),
                min_segment_length_m=float(args.min_segment_length_m),
            )
            if not route_class_multi.empty:
                route_class_multivariate_frames.append(route_class_multi)
            population_frame = _load_city_population_pattern(paths)
            if not population_frame.empty:
                city_population_frames.append(population_frame)
            population_multi = _load_city_population_pattern_multivariate(paths)
            if not population_multi.empty:
                city_population_multivariate_frames.append(population_multi)
            _log(f"[{slug}] done: {paths.output_dir}")
    else:
        for slug in city_list:
            paths = _resolve_city_paths(slug, joint_input_root, output_root, service_accessibility_root)
            route_class_path = paths.output_dir / "stats" / "route_class_length.csv"
            route_stats_path = paths.output_dir / "stats" / "route_stats.csv"
            if not route_class_path.exists() or not route_stats_path.exists():
                raise FileNotFoundError(
                    f"Cross-city route-pattern summary requires prepared per-city outputs: {paths.output_dir}"
                )
            route_class_frames.append(pd.read_csv(route_class_path))
            route_stats_frames.append(pd.read_csv(route_stats_path))
            route_class_multi = _compute_multivariate_route_class(
                paths,
                modalities=list(args.modalities),
                min_segment_length_m=float(args.min_segment_length_m),
            )
            if not route_class_multi.empty:
                route_class_multivariate_frames.append(route_class_multi)
            population_frame = _load_city_population_pattern(paths)
            if not population_frame.empty:
                city_population_frames.append(population_frame)
            population_multi = _load_city_population_pattern_multivariate(paths)
            if not population_multi.empty:
                city_population_multivariate_frames.append(population_multi)

    if len(route_stats_frames) >= 2:
        _log(f"Writing cross-city route-pattern outputs for {len(route_stats_frames)} cities.")
        _write_cross_city_outputs(
            output_root,
            route_class_frames=route_class_frames,
            route_stats_frames=route_stats_frames,
            city_population_frames=city_population_frames,
            route_class_multivariate_frames=route_class_multivariate_frames,
            city_population_multivariate_frames=city_population_multivariate_frames,
        )


if __name__ == "__main__":
    main()

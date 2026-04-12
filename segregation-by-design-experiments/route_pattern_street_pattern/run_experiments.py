from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
from loguru import logger

from aggregated_spatial_pipeline.pipeline.run_pt_street_pattern_dependency import (
    _compute_dependency_tables,
    _compute_route_node_counts,
    _overlay_pt_with_street_pattern,
    _pick_class_column,
    _prepare_pt_edges,
)
from aggregated_spatial_pipeline.runtime_config import configure_logger, ensure_repo_mplconfigdir
from aggregated_spatial_pipeline.visualization import footer_text, save_preview_figure


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT
SUBPROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SUBPROJECT_ROOT / "outputs"
DEFAULT_JOINT_INPUT_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs"
DEFAULT_CITIES = ("warsaw_poland", "berlin_germany")
DEFAULT_MODALITIES = ("bus", "tram", "trolleybus")

ensure_repo_mplconfigdir("mpl-route-pattern-street-pattern", root=REPO_ROOT)


@dataclass(frozen=True)
class CityPaths:
    slug: str
    city_dir: Path
    output_dir: Path
    edges_path: Path
    nodes_path: Path
    street_cells_path: Path


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


def _resolve_city_paths(slug: str, joint_input_root: Path, output_root: Path) -> CityPaths:
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
    )
    for check in (paths.edges_path, paths.nodes_path, paths.street_cells_path):
        if not check.exists():
            raise FileNotFoundError(f"Required input not found for city {slug}: {check}")
    return paths


def _discover_available_city_slugs(joint_input_root: Path, output_root: Path) -> list[str]:
    slugs: list[str] = []
    for city_dir in sorted(p for p in joint_input_root.iterdir() if p.is_dir()):
        try:
            _resolve_city_paths(city_dir.name, joint_input_root, output_root)
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
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.5 * len(plot_df) + 1)))
    sns.barplot(
        data=plot_df,
        x="route_total_km",
        y="route_key",
        hue="dominant_street_pattern_class",
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


def _plot_cross_city_modality_heatmap(city_modality_class: pd.DataFrame, output_path: Path) -> None:
    if city_modality_class.empty:
        return
    plot_df = city_modality_class.copy()
    plot_df["city_modality"] = plot_df["city"].astype(str) + " | " + plot_df["type"].astype(str)
    pivot = plot_df.pivot(index="city_modality", columns="street_pattern_class", values="route_length_share").fillna(0.0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.28 * len(pivot.index) + 1.5)))
    sns.heatmap(pivot, cmap="PuBuGn", annot=False, linewidths=0.2, ax=ax, vmin=0.0, vmax=1.0)
    ax.set_title("Cross-City Route Length Share By City-Modality", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


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
) -> None:
    cross_dir = output_root / "_cross_city"
    stats_dir = cross_dir / "stats"
    preview_dir = cross_dir / "preview_png"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    route_class_all = pd.concat(route_class_frames, ignore_index=True)
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

    route_class_all.to_csv(stats_dir / "route_class_length_all_cities.csv", index=False)
    route_stats_all.to_csv(stats_dir / "route_stats_all_cities.csv", index=False)
    city_class.to_csv(stats_dir / "city_route_length_share_by_class.csv", index=False)
    city_modality_class.to_csv(stats_dir / "city_modality_route_length_share_by_class.csv", index=False)
    pooled_modality_class.to_csv(stats_dir / "pooled_modality_route_length_share_by_class.csv", index=False)

    _plot_cross_city_heatmap(city_class, preview_dir / "01_city_class_route_length_share_heatmap.png")
    _plot_cross_city_modality_heatmap(
        city_modality_class,
        preview_dir / "02_city_modality_class_route_length_share_heatmap.png",
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
        },
    }
    (cross_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    _configure_logging()
    args = parse_args()
    joint_input_root = Path(args.joint_input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if list(args.cities) == ["all"]:
        city_list = _discover_available_city_slugs(joint_input_root, output_root)
    else:
        city_list = [str(city) for city in args.cities]

    _log("Running route-pattern vs street-pattern experiments for " + ", ".join(city_list))

    route_class_frames: list[pd.DataFrame] = []
    route_stats_frames: list[pd.DataFrame] = []
    if not args.cross_city_only:
        for slug in city_list:
            paths = _resolve_city_paths(slug, joint_input_root, output_root)
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
            _log(f"[{slug}] done: {paths.output_dir}")
    else:
        for slug in city_list:
            paths = _resolve_city_paths(slug, joint_input_root, output_root)
            route_class_path = paths.output_dir / "stats" / "route_class_length.csv"
            route_stats_path = paths.output_dir / "stats" / "route_stats.csv"
            if not route_class_path.exists() or not route_stats_path.exists():
                raise FileNotFoundError(
                    f"Cross-city route-pattern summary requires prepared per-city outputs: {paths.output_dir}"
                )
            route_class_frames.append(pd.read_csv(route_class_path))
            route_stats_frames.append(pd.read_csv(route_stats_path))

    if len(route_stats_frames) >= 2:
        _log(f"Writing cross-city route-pattern outputs for {len(route_stats_frames)} cities.")
        _write_cross_city_outputs(
            output_root,
            route_class_frames=route_class_frames,
            route_stats_frames=route_stats_frames,
        )


if __name__ == "__main__":
    main()

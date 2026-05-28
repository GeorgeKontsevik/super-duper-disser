from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.runtime_config import configure_logger, ensure_repo_mplconfigdir
from aggregated_spatial_pipeline.runtime_paths import repo_root, street_pattern_python
from aggregated_spatial_pipeline.visualization import apply_preview_canvas, get_palette, normalize_preview_gdf


REPO_ROOT = repo_root()
DEFAULT_JOINT_INPUT_ROOT = (
    REPO_ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "active_19_good_cities_20260412"
    / "joint_inputs"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "experiments_active19_20260412"
    / "street_pattern_stop_highway_filter"
)
CLASS_COLORS = get_palette("street_patterns")
PROB_COLUMNS = [f"prob_{idx}" for idx in range(6)]


ensure_repo_mplconfigdir("mpl-street-pattern-stop-highway-filter", root=REPO_ROOT)


def _log(message: str) -> None:
    logger.bind(tag="[street-pattern-stop-highway]").info(message)


def _warn(message: str) -> None:
    logger.bind(tag="[street-pattern-stop-highway]").warning(message)


def _configure_logging() -> None:
    configure_logger("[street-pattern-stop-highway]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run street-pattern classification after keeping only road highway classes "
            "that appear near PT stops, then compare against the full-road baseline."
        )
    )
    parser.add_argument("--joint-input-root", default=str(DEFAULT_JOINT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--cities", nargs="+", help="City slugs to process. Defaults to every child dir.")
    parser.add_argument(
        "--pt-types",
        nargs="+",
        default=["bus", "tram", "trolleybus", "subway"],
        help="PT edge/modal types used to select stop nodes.",
    )
    parser.add_argument("--stop-buffer-m", type=float, default=30.0)
    parser.add_argument("--min-road-count", type=int, default=None)
    parser.add_argument("--min-total-road-length", type=float, default=None)
    parser.add_argument("--grid-step", type=float, default=None)
    parser.add_argument("--buffer-m", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional first-N city limit for smoke runs.",
    )
    return parser.parse_args()


def normalize_highway_values(values: pd.Series | Iterable[object]) -> list[set[str]]:
    normalized: list[set[str]] = []
    for value in values:
        items: list[object]
        if value is None:
            items = []
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            items = list(value)
        else:
            text = str(value).strip()
            if not text:
                items = []
            elif text.startswith("[") and text.endswith("]"):
                try:
                    parsed = ast.literal_eval(text)
                    items = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
                except Exception:
                    items = [text]
            elif ";" in text:
                items = text.split(";")
            else:
                items = [text]
        clean = {str(item).strip().lower() for item in items if str(item).strip()}
        normalized.append(clean)
    return normalized


def _explode_highways(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    work = frame.copy()
    if "highway" not in work.columns:
        work["highway"] = "unknown"
    work["highway_norm"] = normalize_highway_values(work["highway"])
    work = work.explode("highway_norm", ignore_index=False)
    work = work[work["highway_norm"].notna() & work["highway_norm"].astype(str).ne("")].copy()
    work["highway_norm"] = work["highway_norm"].astype(str)
    return work


def pick_stop_highway_types(
    roads: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    *,
    stop_buffer_m: float,
) -> tuple[set[str], gpd.GeoDataFrame]:
    if roads.empty:
        raise ValueError("Road layer is empty.")
    if stops.empty:
        raise ValueError("Stop layer is empty.")
    if roads.crs is None:
        roads = roads.set_crs(4326)
    if stops.crs is None:
        stops = stops.set_crs(4326)

    local_crs = roads.estimate_utm_crs() or stops.estimate_utm_crs() or "EPSG:3857"
    roads_local = _explode_highways(roads.to_crs(local_crs))
    stops_local = stops.to_crs(local_crs)
    stop_buffers = gpd.GeoDataFrame(geometry=stops_local.geometry.buffer(float(stop_buffer_m)), crs=local_crs)
    stop_buffers = stop_buffers[stop_buffers.geometry.notna() & ~stop_buffers.geometry.is_empty].copy()
    if stop_buffers.empty:
        raise ValueError("Stop buffers are empty.")

    joined = gpd.sjoin(
        roads_local,
        stop_buffers,
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return set(), roads_local.iloc[0:0].copy()
    matched = roads_local.loc[joined.index.unique()].copy()
    selected = set(joined["highway_norm"].dropna().astype(str))
    return selected, matched


def filter_roads_by_highway_types(roads: gpd.GeoDataFrame, highway_types: set[str]) -> gpd.GeoDataFrame:
    exploded = _explode_highways(roads)
    keep_index = exploded[exploded["highway_norm"].isin(highway_types)].index.unique()
    filtered = roads.loc[keep_index].copy()
    return filtered.reset_index(drop=True)


def _city_slug_from_dir(city_dir: Path) -> str:
    return city_dir.name


def _baseline_summary_path(city_dir: Path) -> Path:
    slug = _city_slug_from_dir(city_dir)
    return city_dir / "street_pattern" / f"{slug}_summary.json"


def _baseline_cells_path(city_dir: Path) -> Path:
    slug = _city_slug_from_dir(city_dir)
    return city_dir / "street_pattern" / slug / "predicted_cells.geojson"


def _baseline_roads_path(city_dir: Path) -> Path:
    candidate = city_dir / "derived_layers" / "roads_drive_osmnx.parquet"
    if candidate.exists():
        return candidate
    slug = _city_slug_from_dir(city_dir)
    return city_dir / "street_pattern" / slug / "roads.geojson"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_cities(root: Path, requested: list[str] | None, limit: int | None) -> list[Path]:
    if requested:
        cities = [root / city for city in requested]
    else:
        cities = sorted(path for path in root.iterdir() if path.is_dir())
    if limit is not None:
        cities = cities[: int(limit)]
    return cities


def _load_intermodal_stop_points(city_dir: Path, pt_types: list[str]) -> gpd.GeoDataFrame:
    nodes_path = city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet"
    edges_path = city_dir / "intermodal_graph_iduedu" / "graph_edges.parquet"
    allowed = {value.lower() for value in pt_types}
    if nodes_path.exists() and edges_path.exists():
        nodes = gpd.read_parquet(nodes_path)
        edges = gpd.read_parquet(edges_path)
        if "type" in edges.columns and {"u", "v"}.issubset(edges.columns):
            edge_types = edges["type"].astype("string").str.lower()
            pt_edges = edges[edge_types.isin(allowed)].copy()
            node_ids = set(pd.to_numeric(pt_edges["u"], errors="coerce").dropna().astype(int))
            node_ids.update(pd.to_numeric(pt_edges["v"], errors="coerce").dropna().astype(int))
            if node_ids:
                node_id_col = "index" if "index" in nodes.columns else None
                if node_id_col is not None:
                    stops = nodes[pd.to_numeric(nodes[node_id_col], errors="coerce").isin(node_ids)].copy()
                else:
                    stops = nodes[nodes.index.isin(node_ids)].copy()
                stops = stops[stops.geometry.notna() & ~stops.geometry.is_empty].copy()
                if not stops.empty:
                    return stops

    frames: list[gpd.GeoDataFrame] = []
    for pt_type in allowed:
        path = city_dir / "connectpt_osm" / pt_type / "aggregated_stops.parquet"
        if path.exists():
            stops = gpd.read_parquet(path)
            stops["pt_type"] = pt_type
            frames.append(stops)
    if not frames:
        raise FileNotFoundError(f"No intermodal or connectpt stop layers found for {city_dir.name}.")
    stops = pd.concat(frames, ignore_index=True)
    stops = gpd.GeoDataFrame(stops, geometry="geometry", crs=frames[0].crs)
    return stops[stops.geometry.notna() & ~stops.geometry.is_empty].copy()


def _run_classifier(
    *,
    city_dir: Path,
    filtered_roads_geojson: Path,
    output_summary: Path,
    baseline_summary: dict,
    args: argparse.Namespace,
) -> Path:
    slug = _city_slug_from_dir(city_dir)
    script_path = REPO_ROOT / "segregation-by-design-experiments" / "run_street_pattern_city.py"
    python = street_pattern_python(REPO_ROOT)
    if not python.exists():
        raise FileNotFoundError(f"Street-pattern python is missing: {python}")

    place = str(baseline_summary.get("place") or slug.replace("_", " ").title())
    buffer_m = float(args.buffer_m if args.buffer_m is not None else baseline_summary.get("buffer_m", 7000.0))
    grid_step = float(args.grid_step if args.grid_step is not None else baseline_summary.get("grid_step", 500.0))
    min_road_count = int(
        args.min_road_count if args.min_road_count is not None else baseline_summary.get("min_road_count", 5)
    )
    min_total_road_length = float(
        args.min_total_road_length
        if args.min_total_road_length is not None
        else baseline_summary.get("min_total_road_length", 500.0)
    )

    command = [
        str(python),
        str(script_path),
        "--place",
        place,
        "--device",
        str(args.device),
        "--output",
        str(output_summary),
        "--buffer-m",
        str(buffer_m),
        "--grid-step",
        str(grid_step),
        "--min-road-count",
        str(min_road_count),
        "--min-total-road-length",
        str(min_total_road_length),
        "--road-source",
        "local",
        "--roads",
        str(filtered_roads_geojson),
    ]
    if baseline_summary.get("centre_node_id"):
        command.extend(["--center-node-id", str(int(baseline_summary["centre_node_id"]))])
    if baseline_summary.get("relation_id"):
        command.extend(["--relation-id", str(int(baseline_summary["relation_id"]))])
    boundary_path = city_dir / "connectpt_osm" / "boundary.parquet"
    if boundary_path.exists():
        command.extend(["--boundary-path", str(boundary_path)])
    if args.no_cache:
        command.append("--no-cache")

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    _log(f"[{slug}] Running filtered street-pattern classification.")
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))
    filtered_cells = output_summary.parent / slug / "predicted_cells.geojson"
    if not filtered_cells.exists():
        raise FileNotFoundError(f"Filtered classifier did not write predicted cells: {filtered_cells}")
    return filtered_cells


def build_class_comparison(
    full: gpd.GeoDataFrame,
    filtered: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, int]]:
    if "cell_id" not in full.columns or "cell_id" not in filtered.columns:
        raise KeyError("Both predicted cell layers must contain cell_id.")
    class_col = "top1_class_name" if "top1_class_name" in full.columns else "class_name"
    filtered_class_col = "top1_class_name" if "top1_class_name" in filtered.columns else "class_name"
    full_cols = ["cell_id", class_col, "geometry"] + [col for col in PROB_COLUMNS if col in full.columns]
    filtered_cols = ["cell_id", filtered_class_col] + [col for col in PROB_COLUMNS if col in filtered.columns]
    left = full[full_cols].rename(columns={class_col: "full_class"}).copy()
    right = filtered[filtered_cols].rename(columns={filtered_class_col: "filtered_class"}).copy()
    comparison = left.merge(right, on="cell_id", how="left", suffixes=("_full", "_filtered"))
    comparison["filtered_class"] = comparison["filtered_class"].fillna("dropped")
    comparison["class_changed"] = (comparison["full_class"] != comparison["filtered_class"]).map(bool)
    for col in PROB_COLUMNS:
        full_col = f"{col}_full"
        filtered_col = f"{col}_filtered"
        if full_col in comparison.columns and filtered_col in comparison.columns:
            comparison[f"delta_{col}"] = pd.to_numeric(comparison[filtered_col], errors="coerce").fillna(0.0) - pd.to_numeric(
                comparison[full_col], errors="coerce"
            ).fillna(0.0)
    comparison = gpd.GeoDataFrame(comparison, geometry="geometry", crs=full.crs)
    confusion = (
        comparison.groupby(["full_class", "filtered_class"], dropna=False)
        .size()
        .reset_index(name="cell_count")
        .sort_values(["full_class", "filtered_class"])
        .reset_index(drop=True)
    )
    matched = comparison["filtered_class"].ne("dropped")
    summary = {
        "baseline_cells": int(len(full)),
        "filtered_cells": int(len(filtered)),
        "matched_cells": int(matched.sum()),
        "dropped_cells": int((~matched).sum()),
        "changed_matched_cells": int((comparison.loc[matched, "class_changed"]).sum()),
    }
    return comparison, confusion, summary


def _class_share(frame: gpd.GeoDataFrame, class_col: str, label: str) -> pd.DataFrame:
    counts = frame[class_col].fillna("unknown").astype(str).value_counts().rename_axis("class_name").reset_index(name="cell_count")
    counts["variant"] = label
    total = counts["cell_count"].sum()
    counts["cell_share"] = counts["cell_count"] / total if total else 0.0
    return counts[["variant", "class_name", "cell_count", "cell_share"]]


def _write_before_after_preview(
    *,
    full: gpd.GeoDataFrame,
    filtered: gpd.GeoDataFrame,
    comparison: gpd.GeoDataFrame,
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_plot = normalize_preview_gdf(full, target_crs="EPSG:3857")
    filtered_plot = normalize_preview_gdf(filtered, full_plot, target_crs="EPSG:3857")
    comparison_plot = normalize_preview_gdf(comparison, full_plot, target_crs="EPSG:3857")
    boundary = full_plot[["geometry"]].dissolve()

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    variants = [
        (axes[0], full_plot, "Full road graph", "full_class"),
        (axes[1], filtered_plot.rename(columns={"top1_class_name": "filtered_class"}), "Stop-highway road classes", "filtered_class"),
    ]
    for ax, frame, subtitle, col in variants:
        apply_preview_canvas(fig, ax, boundary, title=subtitle)
        for class_name, color in CLASS_COLORS.items():
            part = frame[frame[col].fillna("unknown").astype(str).eq(class_name)] if col in frame.columns else frame.iloc[0:0]
            if not part.empty:
                part.plot(ax=ax, color=color, edgecolor="white", linewidth=0.15, alpha=0.95)

    apply_preview_canvas(fig, axes[2], boundary, title="Changed / dropped cells")
    unchanged = comparison_plot[~comparison_plot["class_changed"]]
    changed = comparison_plot[
        comparison_plot["class_changed"] & comparison_plot["filtered_class"].ne("dropped")
    ]
    dropped = comparison_plot[comparison_plot["filtered_class"].eq("dropped")]
    if not unchanged.empty:
        unchanged.plot(ax=axes[2], color="#e5e7eb", edgecolor="white", linewidth=0.12)
    if not changed.empty:
        changed.plot(ax=axes[2], color="#d95f02", edgecolor="white", linewidth=0.2)
    if not dropped.empty:
        dropped.plot(ax=axes[2], color="#525252", edgecolor="white", linewidth=0.2)

    handles = [
        Patch(facecolor=color, label=class_name)
        for class_name, color in CLASS_COLORS.items()
        if class_name != "unknown"
    ]
    handles.extend(
        [
            Patch(facecolor="#d95f02", label="changed"),
            Patch(facecolor="#525252", label="dropped"),
        ]
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_city(city_dir: Path, output_root: Path, args: argparse.Namespace) -> dict:
    slug = _city_slug_from_dir(city_dir)
    baseline_summary_path = _baseline_summary_path(city_dir)
    baseline_cells_path = _baseline_cells_path(city_dir)
    baseline_roads_path = _baseline_roads_path(city_dir)
    if not baseline_summary_path.exists() or not baseline_cells_path.exists():
        raise FileNotFoundError(f"[{slug}] Missing baseline street-pattern outputs.")
    if not baseline_roads_path.exists():
        raise FileNotFoundError(f"[{slug}] Missing baseline road layer.")

    out_dir = output_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary = _load_json(baseline_summary_path)
    roads = read_geodata(baseline_roads_path)
    stops = _load_intermodal_stop_points(city_dir, [str(t).lower() for t in args.pt_types])
    highway_types, matched_roads = pick_stop_highway_types(roads, stops, stop_buffer_m=float(args.stop_buffer_m))
    if not highway_types:
        raise ValueError(f"[{slug}] No road highway types intersect stop buffers.")

    filtered_roads = filter_roads_by_highway_types(roads, highway_types)
    filtered_roads_parquet = out_dir / "roads_stop_highway_filtered.parquet"
    filtered_roads_geojson = out_dir / "roads_stop_highway_filtered.geojson"
    filtered_roads.to_parquet(filtered_roads_parquet)
    filtered_roads.to_file(filtered_roads_geojson, driver="GeoJSON")
    matched_roads.to_parquet(out_dir / "roads_near_stops_matched.parquet")
    pd.DataFrame({"highway": sorted(highway_types)}).to_csv(out_dir / "stop_highway_types.csv", index=False)

    filtered_summary_path = out_dir / f"{slug}_summary.json"
    filtered_cells_path = _run_classifier(
        city_dir=city_dir,
        filtered_roads_geojson=filtered_roads_geojson,
        output_summary=filtered_summary_path,
        baseline_summary=baseline_summary,
        args=args,
    )
    full = gpd.read_file(baseline_cells_path)
    filtered = gpd.read_file(filtered_cells_path)
    comparison, confusion, comparison_summary = build_class_comparison(full, filtered)

    stats_dir = out_dir / "stats"
    preview_dir = out_dir / "preview_png"
    stats_dir.mkdir(exist_ok=True)
    comparison.to_file(out_dir / "comparison_cells.geojson", driver="GeoJSON")
    confusion.to_csv(stats_dir / "confusion_matrix.csv", index=False)
    full_class_col = "top1_class_name" if "top1_class_name" in full.columns else "class_name"
    filtered_class_col = "top1_class_name" if "top1_class_name" in filtered.columns else "class_name"
    class_share = pd.concat(
        [
            _class_share(full, full_class_col, "full"),
            _class_share(filtered, filtered_class_col, "stop_highway"),
        ],
        ignore_index=True,
    )
    class_share.to_csv(stats_dir / "class_share_comparison.csv", index=False)
    before_after_path = preview_dir / "01_full_vs_stop_highway_top1.png"
    _write_before_after_preview(
        full=full.rename(columns={full_class_col: "full_class"}),
        filtered=filtered,
        comparison=comparison,
        output_path=before_after_path,
        title=f"{slug}: full road graph vs stop-highway road classes",
    )

    summary = {
        "city": slug,
        "pt_types": [str(t).lower() for t in args.pt_types],
        "stop_buffer_m": float(args.stop_buffer_m),
        "selected_highway_types": sorted(highway_types),
        "baseline_roads": int(len(roads)),
        "matched_stop_buffer_roads": int(len(matched_roads)),
        "filtered_roads": int(len(filtered_roads)),
        **comparison_summary,
        "files": {
            "baseline_cells": str(baseline_cells_path.resolve()),
            "filtered_cells": str(filtered_cells_path.resolve()),
            "filtered_roads_parquet": str(filtered_roads_parquet.resolve()),
            "comparison_cells": str((out_dir / "comparison_cells.geojson").resolve()),
            "class_share_comparison": str((stats_dir / "class_share_comparison.csv").resolve()),
            "confusion_matrix": str((stats_dir / "confusion_matrix.csv").resolve()),
            "before_after_png": str(before_after_path.resolve()),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(
        f"[{slug}] Done: highways={len(highway_types)}, filtered_roads={len(filtered_roads)}/"
        f"{len(roads)}, changed_matched_cells={comparison_summary['changed_matched_cells']}."
    )
    return summary


def main() -> None:
    _configure_logging()
    args = parse_args()
    joint_input_root = Path(args.joint_input_root).resolve()
    output_root = Path(args.output_root).resolve()
    city_dirs = _resolve_cities(joint_input_root, args.cities, args.limit)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    failures = []
    for city_dir in city_dirs:
        try:
            summaries.append(run_city(city_dir, output_root, args))
        except Exception as exc:  # noqa: BLE001
            failures.append({"city": city_dir.name, "error": str(exc)})
            _warn(f"[{city_dir.name}] Failed: {exc}")
    run_summary = {
        "city_count": len(summaries),
        "failed_count": len(failures),
        "cities": [item["city"] for item in summaries],
        "failures": failures,
    }
    (output_root / "summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

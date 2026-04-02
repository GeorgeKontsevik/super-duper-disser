from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from shapely.geometry import box

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata

from .crosswalks import build_crosswalk
from .transfers import apply_transfer_rule


os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-asp-pipeline3")

CLASS_LABELS = {
    "prob_0": "Loops & Lollipops",
    "prob_1": "Irregular Grid",
    "prob_2": "Regular Grid",
    "prob_3": "Warped Parallel",
    "prob_4": "Sparse",
    "prob_5": "Broken Grid",
}

CLASS_COLORS = {
    "Loops & Lollipops": "#0f766e",
    "Irregular Grid": "#0ea5e9",
    "Regular Grid": "#16a34a",
    "Warped Parallel": "#f97316",
    "Sparse": "#64748b",
    "Broken Grid": "#dc2626",
    "unknown": "#d1d5db",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline_3 preparation step: transfer street-pattern class probabilities "
            "from street-grid cells to quarters using area-overlap shares."
        )
    )
    parser.add_argument(
        "--joint-input-dir",
        default=None,
        help=(
            "Path to pipeline_1 city bundle, e.g. "
            "/.../aggregated_spatial_pipeline/outputs/joint_inputs/barcelona_spain"
        ),
    )
    parser.add_argument(
        "--place",
        default=None,
        help=(
            "Optional place name used to auto-resolve "
            "aggregated_spatial_pipeline/outputs/joint_inputs/<place_slug>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: <joint-input-dir>/pipeline_3",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild the street-pattern-to-quarters transfer even if cached outputs exist.",
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


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


def _log(message: str) -> None:
    logger.info(f"[pipeline_3_street_pattern] {message}")


def _warn(message: str) -> None:
    logger.warning(f"[pipeline_3_street_pattern] {message}")


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:DD MMM HH:mm}</green> | <level>{level}</level> | <level>{message}</level>",
        colorize=True,
    )


def _save_previews(
    transferred,
    prob_columns: list[str],
    preview_dir: Path,
    *,
    boundary_gdf=None,
    use_cache: bool,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    preview_dir.mkdir(parents=True, exist_ok=True)
    dominant_path = preview_dir / "20_quarters_street_pattern_dominant_class.png"
    multivariate_path = preview_dir / "22_quarters_street_pattern_multivariate.png"
    outputs: dict[str, str] = {}

    if use_cache and dominant_path.exists() and multivariate_path.exists():
        _log(
            "Preview step: using cached quarter-level street-pattern previews: "
            f"{dominant_path.name}, {multivariate_path.name}"
        )
        outputs["dominant_class_png"] = str(dominant_path)
        outputs["multivariate_png"] = str(multivariate_path)
        return outputs

    plot_gdf = transferred.copy()
    plot_gdf = plot_gdf[plot_gdf.geometry.notna() & ~plot_gdf.geometry.is_empty].copy()
    if plot_gdf.empty:
        return outputs
    if plot_gdf.crs is not None:
        try:
            plot_gdf = plot_gdf.to_crs("EPSG:3857")
        except Exception:
            pass

    boundary_plot = None
    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_plot = boundary_gdf.copy()
        boundary_plot = boundary_plot[boundary_plot.geometry.notna() & ~boundary_plot.geometry.is_empty].copy()
        if not boundary_plot.empty and boundary_plot.crs is not None:
            try:
                boundary_plot = boundary_plot.to_crs(plot_gdf.crs)
            except Exception:
                pass
    if boundary_plot is None or boundary_plot.empty:
        boundary_plot = plot_gdf[["geometry"]].copy()
        try:
            boundary_union = boundary_plot.union_all()
            boundary_plot = gpd.GeoDataFrame({"geometry": [boundary_union]}, crs=plot_gdf.crs)
        except Exception:
            boundary_plot = None
    outer_bg = None
    outer_bounds = None
    if boundary_plot is not None and not boundary_plot.empty:
        try:
            minx, miny, maxx, maxy = boundary_plot.total_bounds
            pad_x = max((maxx - minx) * 0.08, 250.0)
            pad_y = max((maxy - miny) * 0.08, 250.0)
            outer_bounds = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
            outer_bg = gpd.GeoDataFrame(
                {"geometry": [box(*outer_bounds)]},
                crs=boundary_plot.crs,
            )
        except Exception:
            outer_bg = None
            outer_bounds = None

    prob_frame = plot_gdf[prob_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    covered_mass = prob_frame.sum(axis=1)
    dominant_prob_col = prob_frame.idxmax(axis=1)
    dominant_label = dominant_prob_col.map(CLASS_LABELS).fillna("unknown")
    dominant_label = dominant_label.where(covered_mass > 0.0, "unknown")
    plot_gdf["covered_mass"] = covered_mass
    plot_gdf["dominant_class"] = dominant_label

    rgb = np.zeros((len(plot_gdf), 3), dtype=float)
    for prob_col in prob_columns:
        label = CLASS_LABELS.get(prob_col, "unknown")
        color = np.array(to_rgb(CLASS_COLORS.get(label, "#d1d5db")), dtype=float)
        weight = pd.to_numeric(plot_gdf[prob_col], errors="coerce").fillna(0.0).to_numpy()[:, None]
        rgb += weight * color
    remainder = np.clip(1.0 - covered_mass.to_numpy(dtype=float), 0.0, 1.0)[:, None]
    rgb += remainder * np.array(to_rgb(CLASS_COLORS["unknown"]), dtype=float)
    rgb = np.clip(rgb, 0.0, 1.0)
    plot_gdf["multivariate_color"] = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in rgb
    ]

    _log("Preview step: rendering quarter-level street-pattern dominant-class map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor("#6b6b6b")
    ax.set_facecolor("#6b6b6b")
    if outer_bg is not None and not outer_bg.empty:
        outer_bg.plot(ax=ax, facecolor="#6b6b6b", edgecolor="none", alpha=1.0, zorder=-20)
    if boundary_plot is not None and not boundary_plot.empty:
        boundary_plot.plot(ax=ax, facecolor="#f7f0dd", edgecolor="none", linewidth=0.0, alpha=1.0, zorder=-10)
    legend_handles = []
    class_order = [label for label in CLASS_LABELS.values() if label in set(plot_gdf["dominant_class"])]
    if "unknown" in set(plot_gdf["dominant_class"]):
        class_order.append("unknown")
    for class_name in class_order:
        part = plot_gdf[plot_gdf["dominant_class"] == class_name]
        if part.empty:
            continue
        color = CLASS_COLORS.get(class_name, "#d1d5db")
        part.plot(ax=ax, color=color, linewidth=0.05, edgecolor="#f8fafc", alpha=0.95, zorder=2)
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=class_name))
    if boundary_plot is not None and not boundary_plot.empty:
        boundary_plot.boundary.plot(ax=ax, color="#ffffff", linewidth=1.4, zorder=3)
    if outer_bounds is not None:
        ax.set_xlim(outer_bounds[0], outer_bounds[2])
        ax.set_ylim(outer_bounds[1], outer_bounds[3])
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        fontsize=9,
    )
    ax.set_title("Street Pattern Dominant Class On Quarters", fontsize=19, fontweight="bold", color="#ffffff", pad=18)
    ax.set_axis_off()
    fig.savefig(dominant_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    outputs["dominant_class_png"] = str(dominant_path)
    _log(f"Preview step: saved quarter-level street-pattern dominant-class map: {dominant_path.name}")

    _log("Preview step: rendering quarter-level street-pattern multivariate map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor("#6b6b6b")
    ax.set_facecolor("#6b6b6b")
    if outer_bg is not None and not outer_bg.empty:
        outer_bg.plot(ax=ax, facecolor="#6b6b6b", edgecolor="none", alpha=1.0, zorder=-20)
    if boundary_plot is not None and not boundary_plot.empty:
        boundary_plot.plot(ax=ax, facecolor="#f7f0dd", edgecolor="none", linewidth=0.0, alpha=1.0, zorder=-10)
    plot_gdf.plot(
        ax=ax,
        color=plot_gdf["multivariate_color"].astype(str),
        linewidth=0.05,
        edgecolor="#f8fafc",
        alpha=0.95,
        zorder=2,
    )
    if boundary_plot is not None and not boundary_plot.empty:
        boundary_plot.boundary.plot(ax=ax, color="#ffffff", linewidth=1.4, zorder=3)
    if outer_bounds is not None:
        ax.set_xlim(outer_bounds[0], outer_bounds[2])
        ax.set_ylim(outer_bounds[1], outer_bounds[3])
    legend_handles = [
        Patch(facecolor=CLASS_COLORS[label], edgecolor="none", label=label)
        for label in CLASS_LABELS.values()
    ]
    legend_handles.append(Patch(facecolor=CLASS_COLORS["unknown"], edgecolor="none", label="uncovered"))
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        fontsize=9,
    )
    ax.set_title("Street Pattern Multivariate Mix On Quarters", fontsize=19, fontweight="bold", color="#ffffff", pad=18)
    ax.set_axis_off()
    fig.savefig(multivariate_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    outputs["multivariate_png"] = str(multivariate_path)
    _log(f"Preview step: saved quarter-level street-pattern multivariate map: {multivariate_path.name}")
    return outputs


def _build_quarter_enriched_layer(city_dir: Path, transferred):
    units_path = city_dir / "pipeline_2" / "prepared" / "units_union.parquet"
    matrix_path = city_dir / "pipeline_2" / "prepared" / "adj_matrix_time_min_union.parquet"
    solver_root = city_dir / "pipeline_2" / "solver_inputs"

    enriched = transferred.copy()
    if units_path.exists():
        units = read_geodata(units_path)
        unit_columns = [
            column
            for column in (
                "population",
                "residential",
                "living_area",
                "living_area_proxy",
                "has_living_buildings",
                "capacity_health",
                "capacity_post",
                "capacity_culture",
                "capacity_port",
                "capacity_airport",
                "capacity_marina",
                "demand_base",
            )
            if column in units.columns
        ]
        if unit_columns:
            enriched = enriched.join(units[unit_columns], how="left", rsuffix="_units")

    if matrix_path.exists():
        matrix_union = pd.read_parquet(matrix_path)
        matrix_numeric = matrix_union.apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
        matrix_numeric = matrix_numeric.where(np.isfinite(matrix_numeric), np.nan)
        accessibility_mean = matrix_numeric.mean(axis=1, skipna=True)
        enriched["accessibility_time_mean_pt"] = accessibility_mean.reindex(enriched.index).astype(float)

    service_outputs: dict[str, list[str]] = {}
    if solver_root.exists():
        for blocks_path in sorted(solver_root.glob("*/blocks_solver.parquet")):
            service = blocks_path.parent.name
            solver_blocks = pd.read_parquet(blocks_path)
            service_columns = [
                column
                for column in (
                    "capacity",
                    "demand",
                    "demand_within",
                    "demand_without",
                    "capacity_left",
                    "provision",
                    "provision_strong",
                    "provision_weak",
                )
                if column in solver_blocks.columns
            ]
            if not service_columns:
                continue
            renamed = solver_blocks[service_columns].rename(
                columns={column: f"{service}_{column}" for column in service_columns}
            )
            enriched = enriched.join(renamed, how="left")
            service_outputs[service] = list(renamed.columns)
    if "has_living_buildings" in enriched.columns:
        enriched["has_living_buildings"] = (
            enriched["has_living_buildings"].astype("boolean").fillna(False).astype(bool)
        )
    zero_fill_prefixes = (
        "capacity_",
        "airport_",
        "culture_",
        "health_",
        "marina_",
        "port_",
        "post_",
    )
    for column in enriched.columns:
        if column == "population" or column.startswith(zero_fill_prefixes):
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    return enriched, service_outputs


def main() -> None:
    _configure_logging()
    args = parse_args()
    city_dir = _resolve_city_dir(args)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else city_dir / "pipeline_3"
    prepared_dir = output_dir / "prepared"
    preview_dir = city_dir / "preview_png" / "all_together"
    manifest_path = output_dir / "manifest_street_pattern_to_quarters.json"
    transferred_path = prepared_dir / "quarters_with_street_pattern_probs.parquet"
    enriched_path = prepared_dir / "quarters_enriched_pipeline3.parquet"
    crosswalk_path = prepared_dir / "crosswalk_street_grid_to_quarters.parquet"
    dominant_path = preview_dir / "20_quarters_street_pattern_dominant_class.png"
    multivariate_path = preview_dir / "22_quarters_street_pattern_multivariate.png"
    legacy_mass_path = preview_dir / "21_quarters_street_pattern_covered_mass.png"

    _log(f"Using city bundle: {city_dir}")
    if args.no_cache:
        _warn("Cache mode: disabled (--no-cache). Street-pattern transfer will be rebuilt.")
    else:
        _log("Cache mode: enabled")

    if legacy_mass_path.exists():
        try:
            legacy_mass_path.unlink()
        except Exception:
            pass

    if (
        transferred_path.exists()
        and enriched_path.exists()
        and manifest_path.exists()
        and dominant_path.exists()
        and multivariate_path.exists()
        and (not args.no_cache)
    ):
        _log(f"Using cached street-pattern-to-quarters transfer: {transferred_path}")
        _log(f"Done. Manifest: {manifest_path}")
        return

    street_grid_path = city_dir / "derived_layers" / "street_grid_buffered.parquet"
    quarters_path = city_dir / "derived_layers" / "quarters_clipped.parquet"
    boundary_path = city_dir / "analysis_territory" / "buffer.parquet"
    if not street_grid_path.exists():
        raise FileNotFoundError(f"Missing street grid layer: {street_grid_path}")
    if not quarters_path.exists():
        raise FileNotFoundError(f"Missing quarters layer: {quarters_path}")

    _log("Loading street grid and quarters layers...")
    street = read_geodata(street_grid_path).reset_index(drop=True)
    quarters = read_geodata(quarters_path).reset_index(drop=True)
    boundary = read_geodata(boundary_path) if boundary_path.exists() else None
    if street.empty:
        raise ValueError(f"Street grid is empty: {street_grid_path}")
    if quarters.empty:
        raise ValueError(f"Quarters are empty: {quarters_path}")

    street["street_grid_id"] = street.index.astype(str)
    quarters["quarters_id"] = quarters.index.astype(str)

    _log(
        f"Building crosswalk street_grid -> quarters "
        f"(street cells={len(street)}, quarters={len(quarters)})..."
    )
    crosswalk = build_crosswalk(street, quarters, "street_grid", "quarters")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepare_geodata_for_parquet(crosswalk).to_parquet(crosswalk_path)
    _log(f"Crosswalk ready: {crosswalk_path} ({len(crosswalk)} intersections)")

    _log(
        "Transferring street-pattern probability distribution to quarters "
        "using target_share as overlap weight..."
    )
    transferred = apply_transfer_rule(
        source_gdf=street,
        target_gdf=quarters,
        crosswalk_gdf=crosswalk,
        source_layer="street_grid",
        target_layer="quarters",
        attribute="street_pattern_prob_area_shares",
        aggregation_method="weighted_sum",
        weight_field="target_share",
    )
    prob_columns = [column for column in transferred.columns if column.startswith("prob_")]
    if not prob_columns:
        raise RuntimeError("Street-pattern transfer produced no probability columns on quarters.")

    prepare_geodata_for_parquet(transferred).to_parquet(transferred_path)
    row_sum = transferred[prob_columns].sum(axis=1)
    _log("Joining pipeline_2 quarter-level indicators into one enriched quarter layer...")
    enriched, service_output_columns = _build_quarter_enriched_layer(city_dir, transferred)
    prepare_geodata_for_parquet(enriched).to_parquet(enriched_path)
    _log("Generating preview PNGs for quarter-level street-pattern mix...")
    preview_outputs = _save_previews(
        transferred,
        prob_columns,
        preview_dir,
        boundary_gdf=boundary,
        use_cache=(not args.no_cache),
    )
    manifest = {
        "pipeline": "pipeline_3_street_pattern_to_quarters",
        "city_bundle": str(city_dir),
        "street_grid": str(street_grid_path),
        "quarters": str(quarters_path),
        "crosswalk": str(crosswalk_path),
        "output": str(transferred_path),
        "quarters_enriched": str(enriched_path),
        "probability_columns": prob_columns,
        "pipeline_2_service_columns": service_output_columns,
        "previews": preview_outputs,
        "crosswalk_intersections": int(len(crosswalk)),
        "street_cells": int(len(street)),
        "quarters_count": int(len(quarters)),
        "row_probability_sum_stats": {
            "min": float(row_sum.min()),
            "mean": float(row_sum.mean()),
            "median": float(row_sum.median()),
            "max": float(row_sum.max()),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        f"Transfer ready: {transferred_path} "
        f"(prob cols={len(prob_columns)}, mean covered probability mass={row_sum.mean():.4f})"
    )
    _log(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

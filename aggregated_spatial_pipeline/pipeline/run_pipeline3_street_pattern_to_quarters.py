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

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata
from aggregated_spatial_pipeline.runtime_config import configure_logger, ensure_repo_mplconfigdir
from aggregated_spatial_pipeline.visualization import (
    apply_preview_canvas,
    legend_bottom,
    normalize_preview_gdf,
    save_preview_figure,
)

from .crosswalks import build_crosswalk
from .transfers import apply_transfer_rule


ensure_repo_mplconfigdir("mpl-asp-pipeline3")

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
def _log_name(path: Path | str | None) -> str:
    if path is None:
        return "none"
    try:
        return Path(path).name
    except Exception:
        return str(path)


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
    parser.add_argument(
        "--min-covered-mass",
        type=float,
        default=0.25,
        help=(
            "Minimum total transferred street-pattern probability mass required to keep "
            "quarter-level morphology. Lower-covered quarters are marked as uncovered."
        ),
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
    logger.bind(tag="[pipeline_3_street_pattern]").info(message)


def _warn(message: str) -> None:
    logger.bind(tag="[pipeline_3_street_pattern]").warning(message)


def _configure_logging() -> None:
    configure_logger("[pipeline_3_street_pattern]")


def _apply_low_coverage_mask(
    transferred,
    prob_columns: list[str],
    *,
    min_covered_mass: float,
):
    covered_mass = transferred[prob_columns].sum(axis=1)
    result = transferred.copy()
    result["street_pattern_covered_mass"] = covered_mass.astype(float)
    result["street_pattern_coverage_ok"] = covered_mass >= float(min_covered_mass)
    low_coverage_mask = ~result["street_pattern_coverage_ok"]
    if low_coverage_mask.any():
        result.loc[low_coverage_mask, prob_columns] = 0.0
    return result, covered_mass, low_coverage_mask


def _draw_multivariate_scale_legend(
    fig,
    *,
    class_colors: dict[str, str],
    uncovered_color: str,
    title: str,
    text_color: str,
):
    import matplotlib.colors as mcolors
    import numpy as np

    legend_ax = fig.add_axes([0.79, 0.18, 0.18, 0.46])
    legend_ax.set_facecolor(fig.get_facecolor())

    class_labels = list(class_colors.keys())
    coverage_levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    uncovered_rgb = np.array(mcolors.to_rgb(uncovered_color), dtype=float)
    swatches = np.zeros((len(class_labels), len(coverage_levels), 3), dtype=float)
    for row_idx, class_name in enumerate(class_labels):
        class_rgb = np.array(mcolors.to_rgb(class_colors[class_name]), dtype=float)
        for col_idx, coverage in enumerate(coverage_levels):
            swatches[row_idx, col_idx, :] = uncovered_rgb * (1.0 - coverage) + class_rgb * coverage

    legend_ax.imshow(swatches, aspect="auto", interpolation="nearest")
    legend_ax.set_xticks(range(len(coverage_levels)))
    legend_ax.set_xticklabels([f"{int(level * 100)}%" for level in coverage_levels], fontsize=8, color=text_color)
    legend_ax.set_yticks(range(len(class_labels)))
    legend_ax.set_yticklabels(class_labels, fontsize=8, color=text_color)
    legend_ax.xaxis.tick_top()
    legend_ax.tick_params(length=0)
    legend_ax.set_title(title, fontsize=10, color=text_color, pad=10, loc="left")
    for spine in legend_ax.spines.values():
        spine.set_edgecolor("#d1d5db")
        spine.set_linewidth(0.8)
    return legend_ax


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
    boundary_plot = normalize_preview_gdf(boundary_gdf, target_crs="EPSG:3857")
    plot_gdf = normalize_preview_gdf(plot_gdf, boundary_plot, target_crs="EPSG:3857")
    if boundary_plot is None or boundary_plot.empty:
        boundary_plot = plot_gdf[["geometry"]].copy()
        try:
            boundary_union = boundary_plot.union_all()
            boundary_plot = gpd.GeoDataFrame({"geometry": [boundary_union]}, crs=plot_gdf.crs)
        except Exception:
            boundary_plot = None

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
    apply_preview_canvas(fig, ax, boundary_plot, title="Street Pattern Dominant Class On Quarters")
    legend_handles = []
    class_order = [label for label in CLASS_LABELS.values() if label in set(plot_gdf["dominant_class"])]
    if "unknown" in set(plot_gdf["dominant_class"]):
        class_order.append("unknown")
    for class_name in class_order:
        part = plot_gdf[plot_gdf["dominant_class"] == class_name]
        if part.empty:
            continue
        color = CLASS_COLORS.get(class_name, "#d1d5db")
        part.plot(ax=ax, color=color, linewidth=0.05, edgecolor="#d1d5db", alpha=0.92, zorder=2)
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=class_name))
    legend_bottom(ax, legend_handles, max_cols=3, fontsize=9)
    ax.set_axis_off()
    save_preview_figure(fig, dominant_path)
    plt.close(fig)
    outputs["dominant_class_png"] = str(dominant_path)
    _log(f"Preview step: saved quarter-level street-pattern dominant-class map: {dominant_path.name}")

    _log("Preview step: rendering quarter-level street-pattern multivariate map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    apply_preview_canvas(fig, ax, boundary_plot, title="Street Pattern Multivariate Mix On Quarters")
    plot_gdf.plot(
        ax=ax,
        color=plot_gdf["multivariate_color"].astype(str),
        linewidth=0.05,
        edgecolor="#d1d5db",
        alpha=0.92,
        zorder=2,
    )
    fig.subplots_adjust(right=0.76)
    _draw_multivariate_scale_legend(
        fig,
        class_colors={label: CLASS_COLORS[label] for label in CLASS_LABELS.values()},
        uncovered_color=CLASS_COLORS["unknown"],
        title="Coverage x Anchor Color",
        text_color="#f8fafc",
    )
    ax.set_axis_off()
    save_preview_figure(fig, multivariate_path)
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
            if "provision" not in solver_blocks.columns and "provision_strong" in solver_blocks.columns:
                # Backward compatibility with old pipeline_2 outputs.
                solver_blocks = solver_blocks.copy()
                solver_blocks["provision"] = solver_blocks["provision_strong"]
            service_columns = [
                column
                for column in (
                    "capacity",
                    "demand",
                    "demand_within",
                    "demand_without",
                    "capacity_left",
                    "provision",
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

    _log(f"Using city bundle: {city_dir.name}")
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
        _log(f"Using cached street-pattern-to-quarters transfer: {_log_name(transferred_path)}")
        _log(f"Done. Manifest: {_log_name(manifest_path)}")
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
    _log(f"Crosswalk ready: {_log_name(crosswalk_path)} ({len(crosswalk)} intersections)")

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

    transferred, covered_mass_before_filter, low_coverage_mask = _apply_low_coverage_mask(
        transferred,
        prob_columns,
        min_covered_mass=float(args.min_covered_mass),
    )
    if low_coverage_mask.any():
        _warn(
            "Quarter coverage filter: masked street-pattern morphology for "
            f"{int(low_coverage_mask.sum())} quarters with covered_mass < {float(args.min_covered_mass):.2f}."
        )

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
        "min_covered_mass": float(args.min_covered_mass),
        "low_coverage_quarters_masked": int(low_coverage_mask.sum()),
        "row_probability_sum_stats": {
            "min": float(row_sum.min()),
            "mean": float(row_sum.mean()),
            "median": float(row_sum.median()),
            "max": float(row_sum.max()),
        },
        "covered_mass_before_filter_stats": {
            "min": float(covered_mass_before_filter.min()),
            "mean": float(covered_mass_before_filter.mean()),
            "median": float(covered_mass_before_filter.median()),
            "max": float(covered_mass_before_filter.max()),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _log(
        f"Transfer ready: {_log_name(transferred_path)} "
        f"(prob cols={len(prob_columns)}, mean covered probability mass={row_sum.mean():.4f})"
    )
    _log(f"Done. Manifest: {_log_name(manifest_path)}")


if __name__ == "__main__":
    main()

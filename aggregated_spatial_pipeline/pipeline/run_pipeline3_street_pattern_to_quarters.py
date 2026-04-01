from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

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
        required=True,
        help=(
            "Path to pipeline_1 city bundle, e.g. "
            "/.../aggregated_spatial_pipeline/outputs/joint_inputs/barcelona_spain"
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


def _log(message: str) -> None:
    logger.info(f"[pipeline_3_street_pattern] {message}")


def _warn(message: str) -> None:
    logger.warning(f"[pipeline_3_street_pattern] {message}")


def _save_previews(
    transferred,
    prob_columns: list[str],
    preview_dir: Path,
    *,
    use_cache: bool,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    preview_dir.mkdir(parents=True, exist_ok=True)
    dominant_path = preview_dir / "20_quarters_street_pattern_dominant_class.png"
    mass_path = preview_dir / "21_quarters_street_pattern_covered_mass.png"
    multivariate_path = preview_dir / "22_quarters_street_pattern_multivariate.png"
    outputs: dict[str, str] = {}

    if use_cache and dominant_path.exists() and mass_path.exists() and multivariate_path.exists():
        outputs["dominant_class_png"] = str(dominant_path)
        outputs["covered_mass_png"] = str(mass_path)
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

    fig, ax = plt.subplots(figsize=(12, 12))
    legend_handles = []
    class_order = [label for label in CLASS_LABELS.values() if label in set(plot_gdf["dominant_class"])]
    if "unknown" in set(plot_gdf["dominant_class"]):
        class_order.append("unknown")
    for class_name in class_order:
        part = plot_gdf[plot_gdf["dominant_class"] == class_name]
        if part.empty:
            continue
        color = CLASS_COLORS.get(class_name, "#d1d5db")
        part.plot(ax=ax, color=color, linewidth=0.05, edgecolor="#f8fafc", alpha=0.95)
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=class_name))
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        fontsize=9,
    )
    ax.set_title("Street Pattern Dominant Class On Quarters", fontsize=12)
    ax.set_axis_off()
    fig.savefig(dominant_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs["dominant_class_png"] = str(dominant_path)

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_gdf.plot(
        ax=ax,
        column="covered_mass",
        cmap="YlOrRd",
        linewidth=0.05,
        edgecolor="#f8fafc",
        legend=True,
        legend_kwds={"label": "covered probability mass", "location": "bottom"},
        vmin=0.0,
        vmax=max(1.0, float(np.nanmax(plot_gdf["covered_mass"]))),
    )
    ax.set_title("Street Pattern Coverage Mass On Quarters", fontsize=12)
    ax.set_axis_off()
    fig.savefig(mass_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs["covered_mass_png"] = str(mass_path)

    fig, ax = plt.subplots(figsize=(12, 12))
    plot_gdf.plot(
        ax=ax,
        color=plot_gdf["multivariate_color"].astype(str),
        linewidth=0.05,
        edgecolor="#f8fafc",
        alpha=0.95,
    )
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
    ax.set_title("Street Pattern Multivariate Mix On Quarters", fontsize=12)
    ax.set_axis_off()
    fig.savefig(multivariate_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs["multivariate_png"] = str(multivariate_path)
    return outputs


def main() -> None:
    args = parse_args()
    city_dir = Path(args.joint_input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else city_dir / "pipeline_3"
    prepared_dir = output_dir / "prepared"
    preview_dir = city_dir / "preview_png" / "all_together"
    manifest_path = output_dir / "manifest_street_pattern_to_quarters.json"
    transferred_path = prepared_dir / "quarters_with_street_pattern_probs.parquet"
    crosswalk_path = prepared_dir / "crosswalk_street_grid_to_quarters.parquet"
    dominant_path = preview_dir / "20_quarters_street_pattern_dominant_class.png"
    mass_path = preview_dir / "21_quarters_street_pattern_covered_mass.png"
    multivariate_path = preview_dir / "22_quarters_street_pattern_multivariate.png"

    _log(f"Using city bundle: {city_dir}")
    if args.no_cache:
        _warn("Cache mode: disabled (--no-cache). Street-pattern transfer will be rebuilt.")
    else:
        _log("Cache mode: enabled")

    if (
        transferred_path.exists()
        and manifest_path.exists()
        and dominant_path.exists()
        and mass_path.exists()
        and multivariate_path.exists()
        and (not args.no_cache)
    ):
        _log(f"Using cached street-pattern-to-quarters transfer: {transferred_path}")
        _log(f"Done. Manifest: {manifest_path}")
        return

    street_grid_path = city_dir / "derived_layers" / "street_grid_buffered.parquet"
    quarters_path = city_dir / "derived_layers" / "quarters_clipped.parquet"
    if not street_grid_path.exists():
        raise FileNotFoundError(f"Missing street grid layer: {street_grid_path}")
    if not quarters_path.exists():
        raise FileNotFoundError(f"Missing quarters layer: {quarters_path}")

    _log("Loading street grid and quarters layers...")
    street = read_geodata(street_grid_path).reset_index(drop=True)
    quarters = read_geodata(quarters_path).reset_index(drop=True)
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
    _log("Generating preview PNGs for quarter-level street-pattern mix...")
    preview_outputs = _save_previews(transferred, prob_columns, preview_dir, use_cache=(not args.no_cache))
    manifest = {
        "pipeline": "pipeline_3_street_pattern_to_quarters",
        "city_bundle": str(city_dir),
        "street_grid": str(street_grid_path),
        "quarters": str(quarters_path),
        "crosswalk": str(crosswalk_path),
        "output": str(transferred_path),
        "probability_columns": prob_columns,
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

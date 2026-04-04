from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata
from aggregated_spatial_pipeline.runtime_config import configure_logger
from aggregated_spatial_pipeline.visualization import (
    apply_preview_canvas,
    clip_to_preview_boundary,
    footer_text,
    legend_bottom,
    normalize_preview_gdf,
    save_preview_figure,
)


ADDITIONAL_COLUMNS = [
    "residential",
    "business",
    "recreation",
    "industrial",
    "transport",
    "special",
    "agriculture",
]
POPULATION_COLUMNS = ["population", "population_total", "population_proxy", "pop_total", "residents", "res_population"]


def _positive_quantile(series: pd.Series, q: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean[clean.notna() & np.isfinite(clean) & clean.gt(0)]
    if clean.empty:
        return None
    return float(clean.quantile(q))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run space-matrix imputation in dedicated sm_imputation runtime.")
    parser.add_argument("--place")
    parser.add_argument("--joint-input-dir")
    parser.add_argument("--blocks-path", dest="blocks_path")
    parser.add_argument("--quarters-path", dest="blocks_path_legacy")
    parser.add_argument("--output-path")
    parser.add_argument("--summary-path")
    parser.add_argument("--boundary-path")
    parser.add_argument("--preview-dir")
    parser.add_argument("--n-clusters", type=int, default=11)
    args = parser.parse_args()
    args.blocks_path = args.blocks_path or args.blocks_path_legacy
    if not any([args.place, args.joint_input_dir, args.blocks_path]):
        parser.error("Provide --place or --joint-input-dir or explicit --blocks-path/--output-path/--summary-path.")
    return args


def _configure_logging() -> None:
    configure_logger("[sm-imputer]")


def _compute_site_area_m2(blocks: gpd.GeoDataFrame) -> pd.Series:
    if blocks.empty:
        return pd.Series(dtype="float64", index=blocks.index)
    work = blocks[["geometry"]].copy()
    if work.crs is None:
        work = work.set_crs(4326)
    local_crs = work.estimate_utm_crs() or "EPSG:3857"
    local = work.to_crs(local_crs)
    return pd.to_numeric(local.geometry.area, errors="coerce").rename("site_area_m2")


def _prepare_sm_input(blocks: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, pd.Series, pd.Series, pd.Series]:
    result = blocks.copy()
    site_area = _compute_site_area_m2(result).replace(0, np.nan)
    build_floor_area = pd.to_numeric(result.get("build_floor_area"), errors="coerce")
    footprint_area = pd.to_numeric(result.get("footprint_area"), errors="coerce")
    fsi_base = pd.to_numeric(build_floor_area / site_area, errors="coerce").replace([np.inf, -np.inf], np.nan)
    gsi_base = pd.to_numeric(footprint_area / site_area, errors="coerce").replace([np.inf, -np.inf], np.nan)
    result["site_area"] = site_area
    result["fsi"] = fsi_base
    result["gsi"] = gsi_base
    for column in ADDITIONAL_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result, site_area, fsi_base, gsi_base


def _filter_accessibility_blocks(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if blocks is None or blocks.empty:
        return blocks
    for column in POPULATION_COLUMNS:
        if column in blocks.columns:
            population = pd.to_numeric(blocks[column], errors="coerce").fillna(0.0)
            selected = blocks[population > 0].copy()
            return selected if not selected.empty else blocks
    return blocks


def _build_target_mask(input_gdf: gpd.GeoDataFrame) -> pd.Series:
    build_floor_area = pd.to_numeric(input_gdf.get("build_floor_area"), errors="coerce")
    footprint_area = pd.to_numeric(input_gdf.get("footprint_area"), errors="coerce")
    fsi_base = pd.to_numeric(input_gdf.get("fsi"), errors="coerce")
    gsi_base = pd.to_numeric(input_gdf.get("gsi"), errors="coerce")
    target = (
        build_floor_area.isna()
        | footprint_area.isna()
        | (build_floor_area <= 0)
        | (footprint_area <= 0)
        | fsi_base.isna()
        | gsi_base.isna()
        | (fsi_base <= 0)
        | (gsi_base <= 0)
    )
    return target.fillna(False)


def _build_reference_mask(input_gdf: gpd.GeoDataFrame, target_mask: pd.Series) -> tuple[pd.Series, dict[str, float | int | None]]:
    known_mask = (~target_mask).copy()
    fsi_base = pd.to_numeric(input_gdf.get("fsi"), errors="coerce")
    gsi_base = pd.to_numeric(input_gdf.get("gsi"), errors="coerce")
    positive_known = known_mask & fsi_base.gt(0) & gsi_base.gt(0)
    fsi_cap = _positive_quantile(fsi_base[positive_known], 0.95)
    gsi_cap = _positive_quantile(gsi_base[positive_known], 0.95)
    reference_mask = positive_known.copy()
    if fsi_cap is not None:
        reference_mask &= fsi_base.le(fsi_cap)
    if gsi_cap is not None:
        reference_mask &= gsi_base.le(gsi_cap)
    if int(reference_mask.sum()) < 25:
        reference_mask = positive_known
    caps = {
        "fsi_cap_q95": fsi_cap,
        "gsi_cap_q95": gsi_cap,
        "reference_rows_before_cap": int(positive_known.sum()),
        "reference_rows_after_cap": int(reference_mask.sum()),
    }
    return reference_mask, caps


def _build_example_case(
    *,
    prepared: gpd.GeoDataFrame,
    known_mask: pd.Series,
    clusters_requested: int,
    caps: dict[str, float | int | None],
):
    from catboost import CatBoostClassifier
    from sm_imputation.examples.imputers.sm import CLUSTER_COLUMN, SITE_AREA_COLUMN, SmImputer, Spacematrix

    working_blocks = _filter_accessibility_blocks(prepared)
    known_mask = known_mask.reindex(prepared.index).fillna(False)
    reference_pool = working_blocks.loc[known_mask.reindex(working_blocks.index).fillna(False)].copy()
    if reference_pool.empty or len(reference_pool) < 3:
        return None
    try:
        example_idx = reference_pool.sample(n=1, random_state=42).index[0]
    except Exception:
        example_idx = reference_pool.index[0]

    demo_input = prepared.loc[known_mask].copy()
    if example_idx not in demo_input.index:
        return None

    original_row = demo_input.loc[example_idx].copy()
    sm = Spacematrix(max(2, min(max(1, int(clusters_requested)), len(reference_pool))), 42)
    sm_df, clusters_df = sm.run(reference_pool)
    original_cluster = None
    if example_idx in sm_df.index:
        cluster_value = pd.to_numeric(sm_df.loc[example_idx, CLUSTER_COLUMN], errors="coerce")
        if not pd.isna(cluster_value) and int(cluster_value) >= 0:
            original_cluster = int(cluster_value)
    demo_input.loc[example_idx, ["build_floor_area", "footprint_area", "fsi", "gsi"]] = np.nan
    demo_target_ids = [example_idx]
    demo_known_count = max(0, int(len(demo_input) - 1))
    demo_clusters = min(max(1, int(clusters_requested)), demo_known_count)
    if demo_clusters < 2:
        return None

    classifier_cols = [SITE_AREA_COLUMN, *ADDITIONAL_COLUMNS]
    classifier = CatBoostClassifier(verbose=False, allow_writing_files=False)
    classifier.fit(reference_pool[classifier_cols], sm_df[CLUSTER_COLUMN])
    predicted_probs = classifier.predict_proba(demo_input.loc[[example_idx], classifier_cols])
    predicted_labels = classifier.classes_
    predicted_top_idx = int(np.argmax(predicted_probs[0])) if len(predicted_probs) else 0
    predicted_cluster = int(predicted_labels[predicted_top_idx]) if len(predicted_labels) else None
    top_cluster_candidates = []
    if len(predicted_labels):
        ranked = sorted(
            zip(predicted_labels.tolist(), predicted_probs[0].tolist()),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        top_cluster_candidates = [
            {"cluster": int(label), "probability": float(prob)}
            for label, prob in ranked
        ]

    imputer = SmImputer(
        demo_input,
        features_cols=["fsi", "gsi"],
        additional_cols=ADDITIONAL_COLUMNS,
        n_clusters=demo_clusters,
    )
    imputed = imputer.impute(demo_target_ids)
    imputed_fsi = pd.to_numeric(imputed["fsi"], errors="coerce").reindex(demo_target_ids)
    imputed_gsi = pd.to_numeric(imputed["gsi"], errors="coerce").reindex(demo_target_ids)
    if caps.get("fsi_cap_q95") is not None:
        imputed_fsi = imputed_fsi.clip(upper=float(caps["fsi_cap_q95"]))
    if caps.get("gsi_cap_q95") is not None:
        imputed_gsi = imputed_gsi.clip(upper=float(caps["gsi_cap_q95"]))

    site_area = pd.to_numeric(original_row.get("site_area"), errors="coerce")
    original_fsi = pd.to_numeric(original_row.get("fsi"), errors="coerce")
    original_gsi = pd.to_numeric(original_row.get("gsi"), errors="coerce")
    new_fsi = pd.to_numeric(imputed_fsi.iloc[0], errors="coerce") if not imputed_fsi.empty else np.nan
    new_gsi = pd.to_numeric(imputed_gsi.iloc[0], errors="coerce") if not imputed_gsi.empty else np.nan
    example_geometry = prepared.loc[[example_idx], ["geometry"]].copy()

    additional_profile = []
    for col in ADDITIONAL_COLUMNS:
        value = pd.to_numeric(original_row.get(col), errors="coerce")
        if pd.isna(value):
            continue
        additional_profile.append((col, float(value)))
    additional_profile.sort(key=lambda item: item[1], reverse=True)
    top_profile = [{"name": name, "value": value} for name, value in additional_profile[:4] if value > 0]
    original_cluster_fsi = None
    original_cluster_gsi = None
    predicted_cluster_fsi = None
    predicted_cluster_gsi = None
    alternative_cluster = None
    alternative_cluster_probability = None
    alternative_cluster_fsi = None
    alternative_cluster_gsi = None
    if original_cluster is not None and original_cluster in clusters_df.index:
        original_cluster_fsi = float(pd.to_numeric(clusters_df.loc[original_cluster, "fsi"], errors="coerce"))
        original_cluster_gsi = float(pd.to_numeric(clusters_df.loc[original_cluster, "gsi"], errors="coerce"))
    if predicted_cluster is not None and predicted_cluster in clusters_df.index:
        predicted_cluster_fsi = float(pd.to_numeric(clusters_df.loc[predicted_cluster, "fsi"], errors="coerce"))
        predicted_cluster_gsi = float(pd.to_numeric(clusters_df.loc[predicted_cluster, "gsi"], errors="coerce"))
    for candidate in top_cluster_candidates:
        candidate_cluster = candidate.get("cluster")
        if candidate_cluster is None:
            continue
        if predicted_cluster is not None and int(candidate_cluster) == int(predicted_cluster):
            continue
        alternative_cluster = int(candidate_cluster)
        alternative_cluster_probability = float(candidate.get("probability", 0.0))
        if alternative_cluster in clusters_df.index:
            alternative_cluster_fsi = float(pd.to_numeric(clusters_df.loc[alternative_cluster, "fsi"], errors="coerce"))
            alternative_cluster_gsi = float(pd.to_numeric(clusters_df.loc[alternative_cluster, "gsi"], errors="coerce"))
        break

    return {
        "block_id": str(example_idx),
        "geometry": example_geometry,
        "hidden_target_fields": ["build_floor_area", "footprint_area", "fsi", "gsi"],
        "predictor_fields_kept": ["site_area", *ADDITIONAL_COLUMNS],
        "clusters_used": int(demo_clusters),
        "original_class": original_cluster,
        "imputed_class": predicted_cluster,
        "top_class_candidates": top_cluster_candidates,
        "original_class_fsi": original_cluster_fsi,
        "original_class_gsi": original_cluster_gsi,
        "imputed_class_fsi": predicted_cluster_fsi,
        "imputed_class_gsi": predicted_cluster_gsi,
        "alternative_class": alternative_cluster,
        "alternative_class_probability": alternative_cluster_probability,
        "alternative_class_fsi": alternative_cluster_fsi,
        "alternative_class_gsi": alternative_cluster_gsi,
        "site_area": None if pd.isna(site_area) else float(site_area),
        "original_fsi": None if pd.isna(original_fsi) else float(original_fsi),
        "original_gsi": None if pd.isna(original_gsi) else float(original_gsi),
        "imputed_fsi": None if pd.isna(new_fsi) else float(new_fsi),
        "imputed_gsi": None if pd.isna(new_gsi) else float(new_gsi),
        "original_build_floor_area": None if pd.isna(original_row.get("build_floor_area")) else float(pd.to_numeric(original_row.get("build_floor_area"), errors="coerce")),
        "original_footprint_area": None if pd.isna(original_row.get("footprint_area")) else float(pd.to_numeric(original_row.get("footprint_area"), errors="coerce")),
        "imputed_build_floor_area": None if pd.isna(new_fsi) or pd.isna(site_area) else float(new_fsi * site_area),
        "imputed_footprint_area": None if pd.isna(new_gsi) or pd.isna(site_area) else float(new_gsi * site_area),
        "top_landuse_profile": top_profile,
    }


def _read_optional_geodata(path: Path | None) -> gpd.GeoDataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        gdf = read_geodata(path)
    except Exception:
        return None
    if gdf is None or gdf.empty:
        return None
    return gdf


def _resolve_city_dir(place: str | None, joint_input_dir: str | None) -> Path | None:
    if joint_input_dir:
        return Path(joint_input_dir).resolve()
    if place:
        return (Path(__file__).resolve().parents[2] / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs" / _slugify(place)).resolve()
    return None


def _resolve_run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None, str | None]:
    city_bundle = _resolve_city_dir(args.place, args.joint_input_dir)
    if args.blocks_path:
        blocks_path = Path(args.blocks_path).resolve()
    elif city_bundle is not None:
        blocks_path = (city_bundle / "derived_layers" / "quarters_clipped.parquet").resolve()
    else:
        raise ValueError("Could not resolve blocks path.")

    if args.output_path:
        output_path = Path(args.output_path).resolve()
    elif city_bundle is not None:
        output_path = (city_bundle / "derived_layers" / "quarters_sm_imputed.parquet").resolve()
    else:
        raise ValueError("Could not resolve output path.")

    if args.summary_path:
        summary_path = Path(args.summary_path).resolve()
    elif city_bundle is not None:
        summary_path = (city_bundle / "derived_layers" / "sm_imputation_summary.json").resolve()
    else:
        raise ValueError("Could not resolve summary path.")

    preview_dir = args.preview_dir
    boundary_path = args.boundary_path
    if boundary_path is None and city_bundle is not None:
        candidate = city_bundle / "analysis_territory" / "buffer.parquet"
        if candidate.exists():
            boundary_path = str(candidate)
    return blocks_path, output_path, summary_path, city_bundle, preview_dir or None


def _derive_city_bundle(output_path: Path) -> Path | None:
    if output_path.parent.name != "derived_layers":
        return None
    return output_path.parent.parent


def _resolve_preview_dir(output_path: Path, explicit_preview_dir: str | None) -> tuple[Path, Path]:
    if explicit_preview_dir:
        preview_dir = Path(explicit_preview_dir).resolve()
    else:
        city_bundle = _derive_city_bundle(output_path)
        if city_bundle is not None:
            preview_dir = city_bundle / "preview_png" / "all_together"
        else:
            preview_dir = output_path.parent
    stage_dir = preview_dir.parent / "stages" / "sm_imputation_ready" if preview_dir.name == "all_together" else preview_dir / "stages" / "sm_imputation_ready"
    return preview_dir, stage_dir


def _resolve_boundary_path(output_path: Path, explicit_boundary_path: str | None) -> Path | None:
    if explicit_boundary_path:
        return Path(explicit_boundary_path).resolve()
    city_bundle = _derive_city_bundle(output_path)
    if city_bundle is None:
        return None
    candidate = city_bundle / "analysis_territory" / "buffer.parquet"
    return candidate if candidate.exists() else None


def _clip_to_preview_boundary(
    gdf: gpd.GeoDataFrame | None,
    boundary_layer: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame | None:
    return clip_to_preview_boundary(gdf, boundary_layer)


def _save_sm_previews(
    *,
    blocks: gpd.GeoDataFrame,
    summary: dict,
    output_path: Path,
    preview_dir: Path,
    stage_dir: Path,
    boundary_path: Path | None,
    example_case: dict | None,
) -> dict[str, str]:
    import shutil
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    boundary_gdf = _read_optional_geodata(boundary_path)
    if boundary_gdf is None or boundary_gdf.empty:
        boundary_gdf = blocks[["geometry"]].copy()

    def _apply_preview_theme(fig, ax, boundary_layer: gpd.GeoDataFrame | None, *, title: str | None = None) -> None:
        apply_preview_canvas(fig, ax, boundary_layer, title=title, min_pad=100.0)

    def _legend_bottom(ax, handles: list) -> None:
        legend_bottom(ax, handles)

    def _footer_text(fig, lines: list[str] | None) -> None:
        footer_text(fig, lines)

    def _normalize(gdf: gpd.GeoDataFrame | None, *, clip_to: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame | None:
        return normalize_preview_gdf(gdf, clip_to, target_crs="EPSG:3857")

    preview_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    boundary_norm = _normalize(boundary_gdf)
    blocks_norm = _normalize(_filter_accessibility_blocks(blocks), clip_to=boundary_norm)
    outputs: dict[str, str] = {}

    def _save(fig, stem: str) -> None:
        output_png = preview_dir / f"{stem}.png"
        save_preview_figure(fig, output_png)
        plt.close(fig)
        try:
            shutil.copy2(output_png, stage_dir / output_png.name)
        except Exception:
            pass
        outputs[stem] = str(output_png)
        logger.info("Saved preview: {}", output_png.name)

    def _fmt_number(value: float | None, digits: int = 3) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"

    def _fmt_profile(profile: list[dict] | None) -> str:
        if not profile:
            return "none"
        chunks = []
        for item in profile[:4]:
            name = str(item.get("name", "unknown"))
            value = item.get("value")
            if value is None or pd.isna(value):
                continue
            chunks.append(f"{name}={float(value):.2f}")
        return ", ".join(chunks) if chunks else "none"

    def _fmt_class(value) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"class {int(value)}"

    def _fmt_probability(value) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{100.0 * float(value):.1f}%"

    if blocks_norm is None or blocks_norm.empty:
        return outputs

    fig, ax = plt.subplots(figsize=(12, 12))
    legend_handles = []
    groups = [
        (
            blocks_norm[pd.to_numeric(blocks_norm.get("sm_imputation_used"), errors="coerce").fillna(0) >= 0.5],
            "#f59e0b",
            "imputed",
        ),
        (
            blocks_norm[
                (pd.to_numeric(blocks_norm.get("sm_imputation_target"), errors="coerce").fillna(0) >= 0.5)
                & (pd.to_numeric(blocks_norm.get("sm_imputation_used"), errors="coerce").fillna(0) < 0.5)
            ],
            "#dc2626",
            "targeted but unresolved",
        ),
        (
            blocks_norm[pd.to_numeric(blocks_norm.get("sm_imputation_target"), errors="coerce").fillna(0) < 0.5],
            "#93c5fd",
            "existing built form",
        ),
    ]
    for gdf, color, label in groups:
        if gdf is None or gdf.empty:
            continue
        gdf.plot(ax=ax, color=color, alpha=0.92, linewidth=0.05, edgecolor="#d1d5db")
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=label))
    _apply_preview_theme(fig, ax, boundary_norm, title="SM Imputation Status")
    _legend_bottom(ax, legend_handles)
    _footer_text(
        fig,
        [
            f"targeted={int(summary.get('target_blocks', 0))}, imputed={int(summary.get('imputed_blocks', 0))}, reference={int(summary.get('reference_blocks', 0))}",
            f"clusters={int(summary.get('clusters_used', 0))}" if not summary.get("skipped") else f"skipped: {summary.get('skip_reason') or 'unknown'}",
        ],
    )
    ax.set_axis_off()
    _save(fig, "sm_imputation_status")

    for column_name, stem, title, cmap_name in [
        ("sm_fsi_final", "sm_imputation_fsi_final", "SM Imputation Final FSI", "YlOrRd"),
        ("sm_gsi_final", "sm_imputation_gsi_final", "SM Imputation Final GSI", "PuBuGn"),
    ]:
        plot_gdf = blocks_norm.copy()
        plot_gdf[column_name] = pd.to_numeric(plot_gdf.get(column_name), errors="coerce")
        plot_gdf = plot_gdf[plot_gdf[column_name].notna() & plot_gdf[column_name].gt(0)].copy()
        if plot_gdf.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 12))
        plot_gdf.plot(
            ax=ax,
            column=column_name,
            cmap=cmap_name,
            linewidth=0.05,
            edgecolor="#d1d5db",
            alpha=0.92,
            legend=True,
            legend_kwds={"shrink": 0.72, "label": column_name},
        )
        if boundary_norm is not None and not boundary_norm.empty:
            boundary_norm.boundary.plot(ax=ax, color="#111111", linewidth=1.0)
        _apply_preview_theme(fig, ax, boundary_norm, title=title)
        ax.set_axis_off()
        _save(fig, stem)

    if example_case is not None and example_case.get("geometry") is not None:
        example_geom = _normalize(example_case["geometry"])
        if example_geom is not None and not example_geom.empty:
            fig, ax = plt.subplots(figsize=(12, 12))
            base_blocks = blocks_norm.copy()
            _apply_preview_theme(fig, ax, boundary_norm, title="SM Imputation Example Block")
            base_blocks.plot(ax=ax, color="#e5e7eb", alpha=0.98, linewidth=0.12, edgecolor="#cbd5e1")
            if boundary_norm is not None and not boundary_norm.empty:
                boundary_norm.boundary.plot(ax=ax, color="#111111", linewidth=1.0)
            example_geom.plot(ax=ax, facecolor="#f59e0b", alpha=0.6, edgecolor="#7c2d12", linewidth=3.0)
            representative = example_geom.copy()
            representative["geometry"] = representative.geometry.representative_point()
            representative.plot(ax=ax, color="#7c2d12", markersize=35, marker="o")

            zoom_bounds = None
            try:
                minx, miny, maxx, maxy = example_geom.total_bounds
                span = max(maxx - minx, maxy - miny)
                pad = max(span * 2.2, 60.0)
                cx = (minx + maxx) / 2.0
                cy = (miny + maxy) / 2.0
                zoom_bounds = (cx - pad, cx + pad, cy - pad, cy + pad)
            except Exception:
                pass

            if zoom_bounds is not None:
                minx, maxx, miny, maxy = zoom_bounds
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)
                ax.set_aspect("equal", adjustable="box")

            fig.subplots_adjust(bottom=0.23, top=0.92)
            footer_lines = [
                f"block={example_case.get('block_id')} | site_area={_fmt_number(example_case.get('site_area'), 1)} | land use: {_fmt_profile(example_case.get('top_landuse_profile'))}",
                f"weighted output: fsi {_fmt_number(example_case.get('original_fsi'))} -> {_fmt_number(example_case.get('imputed_fsi'))}, gsi {_fmt_number(example_case.get('original_gsi'))} -> {_fmt_number(example_case.get('imputed_gsi'))}",
                f"alternative non-top-1 class: {_fmt_class(example_case.get('alternative_class'))} at {_fmt_probability(example_case.get('alternative_class_probability'))} | fsi={_fmt_number(example_case.get('alternative_class_fsi'))}, gsi={_fmt_number(example_case.get('alternative_class_gsi'))}",
                f"areas: build_floor_area {_fmt_number(example_case.get('original_build_floor_area'), 1)} -> {_fmt_number(example_case.get('imputed_build_floor_area'), 1)} | footprint_area {_fmt_number(example_case.get('original_footprint_area'), 1)} -> {_fmt_number(example_case.get('imputed_footprint_area'), 1)}",
            ]
            footer_text(
                fig,
                footer_lines,
                y=0.035,
                fontsize=8,
                color="#1f2937",
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": "#f7f0dd",
                    "edgecolor": "#9ca3af",
                    "alpha": 0.98,
                },
            )
            legend_handles = [
                Patch(facecolor="#e5e7eb", edgecolor="#cbd5e1", label="other blocks"),
                Patch(facecolor="#f59e0b", edgecolor="#7c2d12", label="changed example block"),
                Line2D([0], [0], color="#111111", linewidth=2, label="analysis boundary"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper right",
                frameon=True,
                facecolor="#f7f0dd",
                edgecolor="#9ca3af",
                fontsize=9,
            )
            ax.set_axis_off()
            _save(fig, "sm_imputation_example_block")

    return outputs


def main() -> None:
    _configure_logging()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    args = parse_args()
    blocks_path, output_path, summary_path, city_bundle, resolved_preview_dir = _resolve_run_paths(args)
    logger.info(
        "Starting dedicated sm-imputation: blocks={}, output={}",
        blocks_path.name,
        output_path.name,
    )

    from sm_imputation.examples.imputers.sm import SmImputer

    preview_dir, stage_dir = _resolve_preview_dir(output_path, resolved_preview_dir)
    boundary_path = _resolve_boundary_path(output_path, args.boundary_path)

    blocks = read_geodata(blocks_path)
    prepared, site_area, fsi_base, gsi_base = _prepare_sm_input(blocks)
    target_mask = _build_target_mask(prepared)
    known_mask, caps = _build_reference_mask(prepared, target_mask)

    clusters_requested = max(1, int(args.n_clusters))
    clusters_used = min(clusters_requested, int(known_mask.sum()))
    skipped = False
    skip_reason = None
    imputed_count = 0
    final_fsi = fsi_base.copy()
    final_gsi = gsi_base.copy()
    imputed_fsi = pd.Series(np.nan, index=prepared.index, dtype="float64")
    imputed_gsi = pd.Series(np.nan, index=prepared.index, dtype="float64")

    if int(target_mask.sum()) == 0:
        skipped = True
        skip_reason = "no_target_blocks"
        logger.info("sm-imputation skipped: no target blocks detected.")
    elif int(known_mask.sum()) < 2:
        skipped = True
        skip_reason = "insufficient_known_blocks"
        logger.warning("sm-imputation skipped: only {} reference blocks are available.", int(known_mask.sum()))
    elif clusters_used < 2:
        skipped = True
        skip_reason = "insufficient_clusters"
        logger.warning("sm-imputation skipped: less than 2 clusters can be formed from known blocks.")
    else:
        logger.info(
            "sm-imputation: reference_blocks={}, target_blocks={}, clusters={}, fsi_cap_q95={}, gsi_cap_q95={}",
            int(known_mask.sum()),
            int(target_mask.sum()),
            clusters_used,
            None if caps["fsi_cap_q95"] is None else round(float(caps["fsi_cap_q95"]), 3),
            None if caps["gsi_cap_q95"] is None else round(float(caps["gsi_cap_q95"]), 3),
        )
        model_input = prepared.loc[known_mask | target_mask].copy()
        imputer = SmImputer(
            model_input,
            features_cols=["fsi", "gsi"],
            additional_cols=ADDITIONAL_COLUMNS,
            n_clusters=clusters_used,
        )
        target_ids = model_input.index[target_mask.reindex(model_input.index).fillna(False)].tolist()
        imputed = imputer.impute(target_ids)
        imputed_fsi.loc[target_ids] = pd.to_numeric(imputed["fsi"], errors="coerce").reindex(target_ids)
        imputed_gsi.loc[target_ids] = pd.to_numeric(imputed["gsi"], errors="coerce").reindex(target_ids)
        if caps["fsi_cap_q95"] is not None:
            imputed_fsi.loc[target_ids] = imputed_fsi.loc[target_ids].clip(upper=float(caps["fsi_cap_q95"]))
        if caps["gsi_cap_q95"] is not None:
            imputed_gsi.loc[target_ids] = imputed_gsi.loc[target_ids].clip(upper=float(caps["gsi_cap_q95"]))
        final_fsi.loc[target_ids] = imputed_fsi.loc[target_ids]
        final_gsi.loc[target_ids] = imputed_gsi.loc[target_ids]
        imputed_count = int((imputed_fsi.loc[target_ids].notna() & imputed_gsi.loc[target_ids].notna()).sum())
        logger.info("sm-imputation finished: imputed {} target blocks.", imputed_count)

    output = blocks.copy()
    output["sm_site_area_m2"] = site_area
    output["sm_fsi_base"] = fsi_base
    output["sm_gsi_base"] = gsi_base
    output["sm_imputation_target"] = target_mask.astype(int)
    output["sm_imputation_used"] = (target_mask & imputed_fsi.notna() & imputed_gsi.notna()).astype(int)
    output["sm_fsi_imputed"] = imputed_fsi
    output["sm_gsi_imputed"] = imputed_gsi
    output["sm_fsi_final"] = final_fsi
    output["sm_gsi_final"] = final_gsi
    output["sm_build_floor_area_final"] = pd.to_numeric(final_fsi * site_area, errors="coerce")
    output["sm_footprint_area_final"] = pd.to_numeric(final_gsi * site_area, errors="coerce")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_geodata_for_parquet(output).to_parquet(output_path)
    example_case = None
    if not skipped:
        try:
            example_case = _build_example_case(
                prepared=prepared,
                known_mask=known_mask,
                clusters_requested=clusters_requested,
                caps=caps,
            )
        except Exception as exc:
            logger.warning("sm-imputation example preview skipped: {}", exc)

    summary = {
        "blocks_path": str(blocks_path),
        "output_path": str(output_path),
        "blocks_total": int(len(output)),
        "target_blocks": int(target_mask.sum()),
        "reference_blocks": int(known_mask.sum()),
        "imputed_blocks": int(output["sm_imputation_used"].sum()),
        "clusters_requested": clusters_requested,
        "clusters_used": int(clusters_used if not skipped else 0),
        "skipped": bool(skipped),
        "skip_reason": skip_reason,
        "fsi_positive_before": int(fsi_base.gt(0).sum()),
        "gsi_positive_before": int(gsi_base.gt(0).sum()),
        "fsi_positive_after": int(final_fsi.gt(0).sum()),
        "gsi_positive_after": int(final_gsi.gt(0).sum()),
        **caps,
        "rows_total": int(len(output)),
        "target_rows": int(target_mask.sum()),
        "known_rows": int(known_mask.sum()),
        "imputed_rows": int(output["sm_imputation_used"].sum()),
    }
    preview_outputs = _save_sm_previews(
        blocks=output,
        summary=summary,
        output_path=output_path,
        preview_dir=preview_dir,
        stage_dir=stage_dir,
        boundary_path=boundary_path,
        example_case=example_case,
    )
    if example_case is not None:
        summary["example_block"] = {
            k: v for k, v in example_case.items() if k != "geometry"
        }
    summary["preview_outputs"] = preview_outputs
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

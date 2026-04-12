from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata
from aggregated_spatial_pipeline.runtime_config import configure_logger
from aggregated_spatial_pipeline.visualization import apply_preview_canvas, footer_text, legend_bottom, normalize_preview_gdf, save_preview_figure
from aggregated_spatial_pipeline.pipeline.run_sm_imputation_external import (
    ADDITIONAL_COLUMNS,
    _prepare_sm_input,
    _build_target_mask,
    _build_reference_mask,
)


def _configure_logging() -> None:
    configure_logger("[sm-imputation-scenario]")


def _log(message: str) -> None:
    logger.bind(tag="[sm-imputation-scenario]").info(message)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def _resolve_city_dir(place: str | None, joint_input_dir: str | None) -> Path:
    if joint_input_dir:
        return Path(joint_input_dir).resolve()
    if place:
        return (
            Path(__file__).resolve().parents[2]
            / "aggregated_spatial_pipeline"
            / "outputs"
            / "joint_inputs"
            / _slugify(place)
        ).resolve()
    raise ValueError("Provide either --joint-input-dir or --place.")


def _base_quarters_path(city_dir: Path) -> Path:
    return city_dir / "derived_layers" / "quarters_clipped.parquet"


def _model_bundle_dir(city_dir: Path) -> Path:
    return city_dir / "sm_imputation" / "model_bundle"


def _scenario_dir(city_dir: Path, scenario_name: str) -> Path:
    return city_dir / "scenario_sm_imputation" / _slugify(scenario_name)


def _load_sm_classes():
    from catboost import CatBoostClassifier
    from sm_imputation.examples.imputers.sm import CLUSTER_COLUMN, SITE_AREA_COLUMN, Spacematrix

    return CatBoostClassifier, CLUSTER_COLUMN, SITE_AREA_COLUMN, Spacematrix


def _select_probable_other_land_use(
    *,
    classifier,
    classifier_cols: list[str],
    site_area_col: str,
    site_area_value: float,
    source_land_use: str | None,
    base_row: pd.Series,
) -> tuple[str, list[dict[str, float]]]:
    candidates: list[str] = [lu for lu in ADDITIONAL_COLUMNS if lu != source_land_use]
    if not candidates:
        candidates = list(ADDITIONAL_COLUMNS)

    ranked: list[dict[str, float]] = []
    for lu in candidates:
        feature_row: dict[str, float] = {}
        for col in classifier_cols:
            if col == site_area_col:
                feature_row[col] = float(site_area_value)
            elif col in ADDITIONAL_COLUMNS:
                feature_row[col] = float(1.0 if col == lu else 0.0)
            else:
                feature_row[col] = float(pd.to_numeric(base_row.get(col), errors="coerce"))
        probs = classifier.predict_proba(pd.DataFrame([feature_row], columns=classifier_cols))[0]
        top_prob = float(np.nanmax(probs)) if len(probs) else float("nan")
        ranked.append({"land_use": lu, "score_top_cluster_probability": top_prob})

    ranked = sorted(ranked, key=lambda item: item["score_top_cluster_probability"], reverse=True)
    return str(ranked[0]["land_use"]), ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare per-city SM imputation bundles and apply quarter-level SM scenarios.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Train and save SM imputation bundle for one city.")
    prepare.add_argument("--joint-input-dir", default=None)
    prepare.add_argument("--place", default=None)
    prepare.add_argument("--n-clusters", type=int, default=11)

    apply = subparsers.add_parser("apply", help="Apply an SM-imputation-driven quarter land-use scenario.")
    apply.add_argument("--joint-input-dir", default=None)
    apply.add_argument("--place", default=None)
    apply.add_argument("--scenario-name", required=True)
    apply.add_argument("--quarter-index", required=True, help="Quarter row index to modify.")
    apply.add_argument("--target-land-use", required=True, choices=[*list(ADDITIONAL_COLUMNS), "probable_other"])
    apply.add_argument("--source-quarters-path", default=None, help="Optional source scenario quarters parquet. Defaults to base quarters.")

    return parser.parse_args()


def _prepare_status_preview(
    *,
    blocks: gpd.GeoDataFrame,
    target_mask: pd.Series,
    reference_mask: pd.Series,
    out_path: Path,
    title: str,
    footer_lines: list[str] | None = None,
) -> str | None:
    from matplotlib.patches import Patch

    plot = blocks.copy()
    plot = plot[plot.geometry.notna() & ~plot.geometry.is_empty].copy()
    if plot.empty:
        return None
    plot["status"] = "other"
    plot.loc[target_mask.reindex(plot.index).fillna(False), "status"] = "target"
    plot.loc[reference_mask.reindex(plot.index).fillna(False), "status"] = "reference"
    boundary_plot = normalize_preview_gdf(plot[["geometry"]], target_crs="EPSG:3857")
    plot = normalize_preview_gdf(plot, boundary_plot, target_crs="EPSG:3857")

    fig, ax = plt.subplots(figsize=(12, 10))
    apply_preview_canvas(fig, ax, boundary_plot, title=title)
    color_map = {"reference": "#2563eb", "target": "#f59e0b", "other": "#d1d5db"}
    labels = {"reference": "reference quarters", "target": "target quarters", "other": "other quarters"}
    handles = []
    for status in ("other", "reference", "target"):
        part = plot[plot["status"] == status]
        if part.empty:
            continue
        part.plot(ax=ax, color=color_map[status], linewidth=0.05, edgecolor="#d1d5db", alpha=0.92)
        handles.append(Patch(facecolor=color_map[status], edgecolor="none", label=labels[status]))
    legend_bottom(ax, handles, max_cols=3, fontsize=10)
    footer_text(fig, footer_lines)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview_figure(fig, out_path)
    plt.close(fig)
    return str(out_path)


def _prepare_delta_preview(
    *,
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    value_col_before: str,
    value_col_after: str,
    out_path: Path,
    title: str,
) -> str | None:
    if before.empty or after.empty:
        return None
    left = before.copy()
    right = after.copy()
    left.index = left.index.astype(str)
    right.index = right.index.astype(str)
    common_idx = [idx for idx in left.index if idx in right.index]
    if not common_idx:
        return None
    plot = right.loc[common_idx, ["geometry"]].copy()
    plot["delta"] = (
        pd.to_numeric(right.loc[common_idx, value_col_after], errors="coerce").fillna(0.0)
        - pd.to_numeric(left.loc[common_idx, value_col_before], errors="coerce").fillna(0.0)
    )
    plot = plot[plot.geometry.notna() & ~plot.geometry.is_empty].copy()
    if plot.empty:
        return None
    boundary_plot = normalize_preview_gdf(plot[["geometry"]], target_crs="EPSG:3857")
    plot = normalize_preview_gdf(plot, boundary_plot, target_crs="EPSG:3857")
    vmax = float(np.nanmax(np.abs(pd.to_numeric(plot["delta"], errors="coerce").to_numpy()))) if len(plot) else 0.0
    vmax = max(vmax, 0.05)
    fig, ax = plt.subplots(figsize=(12, 10))
    apply_preview_canvas(fig, ax, boundary_plot, title=title)
    plot.plot(
        ax=ax,
        column="delta",
        cmap="RdYlGn",
        linewidth=0.05,
        edgecolor="#d1d5db",
        legend=True,
        vmin=-vmax,
        vmax=vmax,
        legend_kwds={"label": "delta, positive = higher"},
    )
    ax.set_axis_off()
    footer_text(fig, [f"min={float(plot['delta'].min()):.3f}, max={float(plot['delta'].max()):.3f}"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview_figure(fig, out_path)
    plt.close(fig)
    return str(out_path)


def _save_prepare_previews(
    *,
    prepared: gpd.GeoDataFrame,
    target_mask: pd.Series,
    reference_mask: pd.Series,
    output_dir: Path,
) -> dict[str, str]:
    previews: dict[str, str] = {}
    status = _prepare_status_preview(
        blocks=prepared,
        target_mask=target_mask,
        reference_mask=reference_mask,
        out_path=output_dir / "sm_prepare_reference_target_status.png",
        title="SM Imputation Training Status",
        footer_lines=[
            f"reference={int(reference_mask.sum())}, target={int(target_mask.sum())}, total={int(len(prepared))}",
        ],
    )
    if status:
        previews["status"] = status
    for col, stem, title in [
        ("fsi", "sm_prepare_fsi_base.png", "SM Imputation Base FSI"),
        ("gsi", "sm_prepare_gsi_base.png", "SM Imputation Base GSI"),
    ]:
        plot = prepared.copy()
        plot[col] = pd.to_numeric(plot.get(col), errors="coerce")
        plot = plot[plot.geometry.notna() & ~plot.geometry.is_empty & plot[col].notna() & plot[col].gt(0)].copy()
        if plot.empty:
            continue
        boundary_plot = normalize_preview_gdf(plot[["geometry"]], target_crs="EPSG:3857")
        plot = normalize_preview_gdf(plot, boundary_plot, target_crs="EPSG:3857")
        fig, ax = plt.subplots(figsize=(12, 10))
        apply_preview_canvas(fig, ax, boundary_plot, title=title)
        plot.plot(ax=ax, column=col, cmap="YlOrRd", linewidth=0.05, edgecolor="#d1d5db", legend=True)
        ax.set_axis_off()
        out_path = output_dir / stem
        save_preview_figure(fig, out_path)
        plt.close(fig)
        previews[col] = str(out_path)
    return previews


def _compute_basic_classifier_metrics(
    *,
    known: gpd.GeoDataFrame,
    labels: pd.Series,
    classifier_cols: list[str],
    catboost_cls,
) -> dict:
    metrics: dict = {
        "enabled": False,
        "reason": None,
        "n_rows_total": int(len(known)),
        "n_classes": 0,
        "train_rows": 0,
        "holdout_rows": 0,
    }

    try:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
        from sklearn.model_selection import train_test_split
    except Exception:
        metrics["reason"] = "sklearn_not_available"
        return metrics

    y = pd.to_numeric(labels, errors="coerce")
    x = known[classifier_cols].copy()
    valid_mask = y.notna()
    x = x.loc[valid_mask]
    y = y.loc[valid_mask].astype(int)
    if len(x) < 10:
        metrics["reason"] = "too_few_rows_for_holdout"
        return metrics

    class_counts = y.value_counts()
    metrics["n_classes"] = int(len(class_counts))
    if len(class_counts) < 2:
        metrics["reason"] = "single_class"
        return metrics
    if int(class_counts.min()) < 2:
        metrics["reason"] = "insufficient_rows_per_class"
        return metrics

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    metrics["train_rows"] = int(len(x_train))
    metrics["holdout_rows"] = int(len(x_test))
    if len(x_test) == 0:
        metrics["reason"] = "empty_holdout"
        return metrics

    eval_model = catboost_cls(verbose=False, allow_writing_files=False)
    eval_model.fit(x_train, y_train)

    pred = eval_model.predict(x_test)
    pred = pd.to_numeric(np.asarray(pred).reshape(-1), errors="coerce").astype(int)
    proba = np.asarray(eval_model.predict_proba(x_test), dtype=float)
    model_classes = np.asarray(eval_model.classes_, dtype=int)
    class_to_pos = {int(cls): int(i) for i, cls in enumerate(model_classes)}
    y_test_idx = np.asarray([class_to_pos[int(v)] for v in y_test.tolist()], dtype=int)
    top_order = np.argsort(-proba, axis=1)
    top2_acc = float(np.mean([y_test_idx[i] in top_order[i, :2] for i in range(len(y_test_idx))])) if len(y_test_idx) else None
    top3_acc = float(np.mean([y_test_idx[i] in top_order[i, :3] for i in range(len(y_test_idx))])) if len(y_test_idx) else None

    metrics.update(
        {
            "enabled": True,
            "reason": None,
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro")),
            "logloss": float(log_loss(y_test, proba, labels=model_classes.tolist())),
            "top2_accuracy": top2_acc,
            "top3_accuracy": top3_acc,
            "class_counts_total": {str(int(k)): int(v) for k, v in class_counts.sort_index().to_dict().items()},
        }
    )
    return metrics


def _assess_classifier_quality(metrics: dict) -> dict:
    if not metrics.get("enabled"):
        return {
            "quality_verdict": "unknown",
            "is_bad_metric": None,
            "reason": str(metrics.get("reason") or "metrics_not_available"),
        }

    macro_f1 = float(metrics.get("macro_f1", 0.0))
    bacc = float(metrics.get("balanced_accuracy", 0.0))

    if macro_f1 < 0.25 or bacc < 0.30:
        verdict = "poor"
        is_bad = 1
    elif macro_f1 < 0.40 or bacc < 0.45:
        verdict = "weak"
        is_bad = 0
    else:
        verdict = "decent"
        is_bad = 0
    return {
        "quality_verdict": verdict,
        "is_bad_metric": int(is_bad),
        "reason": None,
    }


def _prepare_command(args: argparse.Namespace) -> None:
    city_dir = _resolve_city_dir(args.place, args.joint_input_dir)
    quarters_path = _base_quarters_path(city_dir)
    bundle_dir = _model_bundle_dir(city_dir)
    previews_dir = bundle_dir / "preview_png"
    if not quarters_path.exists():
        raise FileNotFoundError(f"Missing quarters layer: {quarters_path}")

    CatBoostClassifier, CLUSTER_COLUMN, SITE_AREA_COLUMN, Spacematrix = _load_sm_classes()
    blocks = read_geodata(quarters_path)
    prepared, _site_area, _fsi_base, _gsi_base = _prepare_sm_input(blocks)
    target_mask = _build_target_mask(prepared)
    reference_mask, caps = _build_reference_mask(prepared, target_mask)
    clusters_requested = max(1, int(args.n_clusters))
    clusters_used = min(clusters_requested, int(reference_mask.sum()))

    bundle_dir.mkdir(parents=True, exist_ok=True)
    previews = _save_prepare_previews(prepared=prepared, target_mask=target_mask, reference_mask=reference_mask, output_dir=previews_dir)

    summary = {
        "city_dir": str(city_dir),
        "quarters_path": str(quarters_path),
        "bundle_dir": str(bundle_dir),
        "rows_total": int(len(prepared)),
        "target_rows": int(target_mask.sum()),
        "reference_rows": int(reference_mask.sum()),
        "clusters_requested": int(clusters_requested),
        "clusters_used": int(clusters_used if clusters_used >= 2 else 0),
        "skipped": False,
        "skip_reason": None,
        **caps,
        "preview_outputs": previews,
    }

    if int(target_mask.sum()) == 0:
        summary["skipped"] = True
        summary["skip_reason"] = "no_target_rows"
        _log("SM prepare skipped: no target rows found.")
    elif int(reference_mask.sum()) < 2:
        summary["skipped"] = True
        summary["skip_reason"] = "insufficient_reference_rows"
        _log(f"SM prepare skipped: only {int(reference_mask.sum())} reference rows available.")
    elif clusters_used < 2:
        summary["skipped"] = True
        summary["skip_reason"] = "insufficient_clusters"
        _log(f"SM prepare skipped: clusters_used={clusters_used} < 2.")
    else:
        known = prepared.loc[reference_mask].copy()
        sm = Spacematrix(clusters_used, 42)
        sm_df, clusters_df = sm.run(known)
        classifier_cols = [SITE_AREA_COLUMN, *ADDITIONAL_COLUMNS]
        basic_metrics = _compute_basic_classifier_metrics(
            known=known,
            labels=sm_df[CLUSTER_COLUMN],
            classifier_cols=classifier_cols,
            catboost_cls=CatBoostClassifier,
        )
        quality_assessment = _assess_classifier_quality(basic_metrics)
        classifier = CatBoostClassifier(verbose=False, allow_writing_files=False)
        classifier.fit(known[classifier_cols], sm_df[CLUSTER_COLUMN])

        classifier_path = bundle_dir / "cluster_classifier.cbm"
        clusters_path = bundle_dir / "cluster_medians.parquet"
        reference_path = bundle_dir / "reference_quarters.parquet"
        labeled_reference_path = bundle_dir / "reference_quarters_with_clusters.parquet"
        classifier.save_model(str(classifier_path))
        clusters_df.to_parquet(clusters_path)
        prepare_geodata_for_parquet(known).to_parquet(reference_path)
        labeled_reference = known.copy()
        labeled_reference[CLUSTER_COLUMN] = sm_df.loc[known.index, CLUSTER_COLUMN]
        prepare_geodata_for_parquet(labeled_reference).to_parquet(labeled_reference_path)

        cluster_counts = (
            pd.to_numeric(sm_df[CLUSTER_COLUMN], errors="coerce").fillna(-1).astype(int).value_counts().sort_index().to_dict()
        )
        feature_importance = {}
        try:
            fi = classifier.get_feature_importance()
            feature_importance = {
                name: float(value)
                for name, value in zip(classifier_cols, fi.tolist(), strict=False)
            }
        except Exception:
            feature_importance = {}

        summary.update(
            {
                "classifier_path": str(classifier_path),
                "clusters_path": str(clusters_path),
                "reference_path": str(reference_path),
                "labeled_reference_path": str(labeled_reference_path),
                "classifier_features": classifier_cols,
                "cluster_counts": {str(k): int(v) for k, v in cluster_counts.items()},
                "feature_importance": feature_importance,
                "classifier_basic_metrics": basic_metrics,
                "classifier_quality_assessment": quality_assessment,
            }
        )
        _log(
            f"SM prepare complete: city={city_dir.name}, "
            f"rows={len(prepared)}, target={int(target_mask.sum())}, reference={int(reference_mask.sum())}, "
            f"clusters={clusters_used}, fsi_cap_q95={caps.get('fsi_cap_q95')}, gsi_cap_q95={caps.get('gsi_cap_q95')}"
        )
        if basic_metrics.get("enabled"):
            _log(
                "SM prepare classifier holdout metrics: "
                f"acc={basic_metrics.get('accuracy'):.3f}, "
                f"bacc={basic_metrics.get('balanced_accuracy'):.3f}, "
                f"macro_f1={basic_metrics.get('macro_f1'):.3f}, "
                f"logloss={basic_metrics.get('logloss'):.3f}, "
                f"top2={basic_metrics.get('top2_accuracy'):.3f}, "
                f"top3={basic_metrics.get('top3_accuracy'):.3f}"
            )
            _log(
                "SM prepare classifier holdout metrics (kv): "
                f"enabled=1 "
                f"n_rows_total={int(basic_metrics.get('n_rows_total', 0))} "
                f"train_rows={int(basic_metrics.get('train_rows', 0))} "
                f"holdout_rows={int(basic_metrics.get('holdout_rows', 0))} "
                f"n_classes={int(basic_metrics.get('n_classes', 0))} "
                f"accuracy={float(basic_metrics.get('accuracy')):.6f} "
                f"balanced_accuracy={float(basic_metrics.get('balanced_accuracy')):.6f} "
                f"macro_f1={float(basic_metrics.get('macro_f1')):.6f} "
                f"logloss={float(basic_metrics.get('logloss')):.6f} "
                f"top2_accuracy={float(basic_metrics.get('top2_accuracy')):.6f} "
                f"top3_accuracy={float(basic_metrics.get('top3_accuracy')):.6f}"
            )
            _log(
                "SM prepare classifier quality: "
                f"quality_verdict={quality_assessment.get('quality_verdict')} "
                f"is_bad_metric={quality_assessment.get('is_bad_metric')}"
            )
        else:
            _log(f"SM prepare classifier holdout metrics skipped: {basic_metrics.get('reason')}")
            _log(
                "SM prepare classifier quality: "
                f"quality_verdict={quality_assessment.get('quality_verdict')} "
                f"is_bad_metric={quality_assessment.get('is_bad_metric')} "
                f"reason={quality_assessment.get('reason')}"
            )
        if feature_importance:
            ranked = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
            _log("SM prepare feature importance: " + ", ".join(f"{name}={value:.3f}" for name, value in ranked[:6]))

    summary_path = bundle_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(f"Saved SM prepare summary: {summary_path}")


def _ensure_scenario_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    defaults = {
        "modified_by_sm_imputer": False,
        "sm_imputer_scenario_id": None,
        "sm_imputer_source_land_use": None,
        "sm_imputer_target_land_use": None,
        "sm_imputer_model_bundle_dir": None,
        "sm_imputer_cluster_top1": np.nan,
        "sm_imputer_cluster_top1_prob": np.nan,
        "sm_imputer_fsi_before": np.nan,
        "sm_imputer_fsi_after": np.nan,
        "sm_imputer_gsi_before": np.nan,
        "sm_imputer_gsi_after": np.nan,
        "sm_imputer_build_floor_area_before": np.nan,
        "sm_imputer_build_floor_area_after": np.nan,
        "sm_imputer_footprint_area_before": np.nan,
        "sm_imputer_footprint_area_after": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


def _save_apply_previews(
    *,
    before: gpd.GeoDataFrame,
    after: gpd.GeoDataFrame,
    changed_idx: str,
    output_dir: Path,
) -> dict[str, str]:
    previews: dict[str, str] = {}
    before_i = before.copy()
    after_i = after.copy()
    before_i.index = before_i.index.astype(str)
    after_i.index = after_i.index.astype(str)
    changed_mask = after_i.index == changed_idx

    status = _prepare_status_preview(
        blocks=after_i,
        target_mask=pd.Series(changed_mask, index=after_i.index),
        reference_mask=pd.Series(False, index=after_i.index),
        out_path=output_dir / "sm_scenario_changed_quarter_status.png",
        title="SM Scenario Changed Quarter",
        footer_lines=[f"changed_quarter={changed_idx}"],
    )
    if status:
        previews["status"] = status

    for before_col, after_col, stem, title in [
        ("build_floor_area", "build_floor_area", "sm_scenario_build_floor_area_delta.png", "SM Scenario Build Floor Area Delta"),
        ("footprint_area", "footprint_area", "sm_scenario_footprint_area_delta.png", "SM Scenario Footprint Area Delta"),
    ]:
        p = _prepare_delta_preview(
            before=before_i,
            after=after_i,
            value_col_before=before_col,
            value_col_after=after_col,
            out_path=output_dir / stem,
            title=title,
        )
        if p:
            previews[stem] = p
    return previews


def _apply_command(args: argparse.Namespace) -> None:
    city_dir = _resolve_city_dir(args.place, args.joint_input_dir)
    bundle_dir = _model_bundle_dir(city_dir)
    bundle_summary_path = bundle_dir / "summary.json"
    if not bundle_summary_path.exists():
        raise FileNotFoundError(f"Missing SM model bundle summary. Run prepare first: {bundle_summary_path}")
    bundle_summary = json.loads(bundle_summary_path.read_text(encoding="utf-8"))
    if bundle_summary.get("skipped"):
        raise RuntimeError(f"SM model bundle is marked as skipped: {bundle_summary.get('skip_reason')}")

    source_path = Path(args.source_quarters_path).resolve() if args.source_quarters_path else _base_quarters_path(city_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source quarters path: {source_path}")
    before = read_geodata(source_path)
    after = _ensure_scenario_columns(before)
    before.index = before.index.astype(str)
    after.index = after.index.astype(str)

    quarter_idx = str(args.quarter_index)
    if quarter_idx not in after.index:
        raise KeyError(f"Quarter index not found: {quarter_idx}")

    CatBoostClassifier, _CLUSTER_COLUMN, SITE_AREA_COLUMN, _Spacematrix = _load_sm_classes()
    classifier = CatBoostClassifier(verbose=False, allow_writing_files=False)
    classifier.load_model(str(bundle_dir / "cluster_classifier.cbm"))
    clusters_df = pd.read_parquet(bundle_dir / "cluster_medians.parquet")
    classifier_cols = bundle_summary.get("classifier_features") or [SITE_AREA_COLUMN, *ADDITIONAL_COLUMNS]
    caps = {
        "fsi_cap_q95": bundle_summary.get("fsi_cap_q95"),
        "gsi_cap_q95": bundle_summary.get("gsi_cap_q95"),
    }

    prepared_before, site_area_before, _fsi_base_before, _gsi_base_before = _prepare_sm_input(after)
    base_row = prepared_before.loc[quarter_idx].copy()
    fsi_before = float(pd.to_numeric(prepared_before.loc[quarter_idx].get("fsi"), errors="coerce"))
    gsi_before = float(pd.to_numeric(prepared_before.loc[quarter_idx].get("gsi"), errors="coerce"))
    site_area_value = float(pd.to_numeric(site_area_before.loc[quarter_idx], errors="coerce"))
    source_land_use = str(after.loc[quarter_idx].get("land_use")) if pd.notna(after.loc[quarter_idx].get("land_use")) else None
    resolved_target_land_use = str(args.target_land_use)
    probable_other_ranked: list[dict[str, float]] = []
    if args.target_land_use == "probable_other":
        resolved_target_land_use, probable_other_ranked = _select_probable_other_land_use(
            classifier=classifier,
            classifier_cols=classifier_cols,
            site_area_col=SITE_AREA_COLUMN,
            site_area_value=site_area_value,
            source_land_use=source_land_use,
            base_row=base_row,
        )

    for col in ADDITIONAL_COLUMNS:
        after.loc[quarter_idx, col] = float(1.0 if col == resolved_target_land_use else 0.0)
    after.loc[quarter_idx, "land_use"] = str(resolved_target_land_use)
    after.loc[quarter_idx, "share"] = float(1.0)

    prepared_after, _site_area_after, _fsi_base, _gsi_base = _prepare_sm_input(after)
    row = prepared_after.loc[[quarter_idx]].copy()
    probs = classifier.predict_proba(row[classifier_cols])
    labels = classifier.classes_
    aligned_clusters = clusters_df.loc[labels]
    fsi_values = pd.to_numeric(aligned_clusters["fsi"], errors="coerce").to_numpy(dtype=float)
    gsi_values = pd.to_numeric(aligned_clusters["gsi"], errors="coerce").to_numpy(dtype=float)
    weighted_fsi = float((probs[0] * fsi_values).sum())
    weighted_gsi = float((probs[0] * gsi_values).sum())
    if caps["fsi_cap_q95"] is not None:
        weighted_fsi = float(min(weighted_fsi, float(caps["fsi_cap_q95"])))
    if caps["gsi_cap_q95"] is not None:
        weighted_gsi = float(min(weighted_gsi, float(caps["gsi_cap_q95"])))
    build_floor_after = float(weighted_fsi * site_area_value) if np.isfinite(site_area_value) else np.nan
    footprint_after = float(weighted_gsi * site_area_value) if np.isfinite(site_area_value) else np.nan
    top_idx = int(np.argmax(probs[0])) if len(probs) else 0
    top_cluster = int(labels[top_idx]) if len(labels) else None
    top_prob = float(probs[0][top_idx]) if len(labels) else None

    after.loc[quarter_idx, "modified_by_sm_imputer"] = True
    after.loc[quarter_idx, "sm_imputer_scenario_id"] = _slugify(args.scenario_name)
    after.loc[quarter_idx, "sm_imputer_source_land_use"] = source_land_use
    after.loc[quarter_idx, "sm_imputer_target_land_use"] = str(resolved_target_land_use)
    after.loc[quarter_idx, "sm_imputer_model_bundle_dir"] = str(bundle_dir)
    after.loc[quarter_idx, "sm_imputer_cluster_top1"] = top_cluster
    after.loc[quarter_idx, "sm_imputer_cluster_top1_prob"] = top_prob
    build_floor_before = float(pd.to_numeric(before.loc[quarter_idx].get("build_floor_area"), errors="coerce"))
    footprint_before = float(pd.to_numeric(before.loc[quarter_idx].get("footprint_area"), errors="coerce"))
    after.loc[quarter_idx, "sm_imputer_fsi_before"] = fsi_before
    after.loc[quarter_idx, "sm_imputer_fsi_after"] = weighted_fsi
    after.loc[quarter_idx, "sm_imputer_gsi_before"] = gsi_before
    after.loc[quarter_idx, "sm_imputer_gsi_after"] = weighted_gsi
    after.loc[quarter_idx, "sm_imputer_build_floor_area_before"] = build_floor_before
    after.loc[quarter_idx, "sm_imputer_build_floor_area_after"] = build_floor_after
    after.loc[quarter_idx, "sm_imputer_footprint_area_before"] = footprint_before
    after.loc[quarter_idx, "sm_imputer_footprint_area_after"] = footprint_after
    after.loc[quarter_idx, "sm_imputation_target"] = 1
    after.loc[quarter_idx, "sm_imputation_used"] = 1
    after.loc[quarter_idx, "sm_fsi_final"] = weighted_fsi
    after.loc[quarter_idx, "sm_gsi_final"] = weighted_gsi
    after.loc[quarter_idx, "sm_build_floor_area_final"] = build_floor_after
    after.loc[quarter_idx, "sm_footprint_area_final"] = footprint_after
    after.loc[quarter_idx, "build_floor_area"] = build_floor_after
    after.loc[quarter_idx, "footprint_area"] = footprint_after

    out_dir = _scenario_dir(city_dir, args.scenario_name)
    previews_dir = out_dir / "preview_png"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quarters.parquet"
    prepare_geodata_for_parquet(after).to_parquet(out_path)
    previews = _save_apply_previews(before=before, after=after, changed_idx=quarter_idx, output_dir=previews_dir)

    ranked_candidates = sorted(
        [{"cluster": int(label), "probability": float(prob)} for label, prob in zip(labels.tolist(), probs[0].tolist(), strict=False)],
        key=lambda item: item["probability"],
        reverse=True,
    )[:5]
    summary = {
        "city_dir": str(city_dir),
        "source_quarters_path": str(source_path),
        "output_quarters_path": str(out_path),
        "scenario_name": _slugify(args.scenario_name),
        "quarter_index": quarter_idx,
        "source_land_use": source_land_use,
        "target_land_use_requested": str(args.target_land_use),
        "target_land_use": str(resolved_target_land_use),
        "probable_other_ranked_candidates": probable_other_ranked,
        "top_cluster": top_cluster,
        "top_cluster_probability": top_prob,
        "top_cluster_candidates": ranked_candidates,
        "site_area_m2": site_area_value,
        "fsi_before": fsi_before,
        "fsi_after": weighted_fsi,
        "gsi_before": gsi_before,
        "gsi_after": weighted_gsi,
        "build_floor_area_before": build_floor_before,
        "build_floor_area_after": build_floor_after,
        "footprint_area_before": footprint_before,
        "footprint_area_after": footprint_after,
        "preview_outputs": previews,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    top_prob_text = f"{top_prob:.3f}" if top_prob is not None else "nan"
    _log(
        f"SM scenario applied: city={city_dir.name}, scenario={summary['scenario_name']}, "
        f"quarter={quarter_idx}, land_use={source_land_use} -> {resolved_target_land_use}, "
        f"cluster_top1={top_cluster}, prob_top1={top_prob_text}, "
        f"fsi={weighted_fsi:.3f}, gsi={weighted_gsi:.3f}, "
        f"build_floor_area={build_floor_after:.1f}, footprint_area={footprint_after:.1f}"
    )
    _log(f"Saved SM scenario summary: {summary_path}")


def main() -> None:
    _configure_logging()
    args = parse_args()
    if args.command == "prepare":
        _prepare_command(args)
    elif args.command == "apply":
        _apply_command(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

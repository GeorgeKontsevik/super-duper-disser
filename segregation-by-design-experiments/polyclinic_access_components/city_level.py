from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/gk/Code/super-duper-disser")
PROJECT_DIR = ROOT / "segregation-by-design-experiments/polyclinic_access_components"
OUTPUT_DIR = PROJECT_DIR / "outputs/city_level"
PIPELINE2_PREPARE_SOLVER_INPUTS_MODULE_PATH = ROOT / "aggregated_spatial_pipeline/pipeline/run_pipeline2_prepare_solver_inputs.py"
ACTIVE19_ROOT = ROOT / "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"
NEW17_ROOT = ROOT / "aggregated_spatial_pipeline/outputs/experiments_new17_access_20260610/joint_inputs_merged"
OLD23_ROOT = ROOT / "aggregated_spatial_pipeline/outputs/experiments_old23_access_20260609/joint_inputs_merged"

SOURCE_PRIORITY = ["active19", "new17", "old23", "new5"]

ACTIVE19_CITIES = [
    "bergen_norway",
    "bologna_italy",
    "bristol_united_kingdom",
    "brno_czechia",
    "coimbra_portugal",
    "debrecen_hungary",
    "dresden_germany",
    "freiburg_im_breisgau_germany",
    "gothenburg_sweden",
    "graz_austria",
    "innsbruck_austria",
    "krakow_poland",
    "linz_austria",
    "lyon_france",
    "marseille_france",
    "porto_portugal",
    "turin_italy",
    "turku_finland",
    "zaragoza_spain",
]

NEW17_ELIGIBLE_CITIES = [
    "adelaide_south_australia_australia",
    "amsterdam_netherlands",
    "arequipa_peru",
    "delft_netherlands",
    "hai_duong_h_i_d_ng_vietnam",
    "huainan_anhui_china",
    "jaynagar_bih_r_india",
    "kakogawacho_honmachi_hy_go_japan",
    "kananga_kasa_central_congo_kinshasa",
    "maracay_aragua_venezuela",
    "montes_claros_minas_gerais_brazil",
    "naihati_west_bengal_india",
    "narayanganj_dhaka_bangladesh",
    "spring_valley_nevada_united_states",
    "temuco_araucan_a_chile",
    "vologda_russia",
]

OLD23_ELIGIBLE_UNIQUE_CITIES = [
    "vienna_austria",
]

ACCESS_DIAGNOSTICS_BY_SOURCE = {
    "active19": ROOT
    / "aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/_all_home_to_service_access_diagnostics.parquet",
    "new17": ROOT
    / "aggregated_spatial_pipeline/outputs/experiments_new17_access_20260610/_all_home_to_service_access_diagnostics.parquet",
    "old23": ROOT
    / "aggregated_spatial_pipeline/outputs/experiments_old23_access_20260609/service_access_diagnostics/_all_home_to_service_access_diagnostics.parquet",
}

ROOT_BY_SOURCE = {
    "active19": ACTIVE19_ROOT,
    "new17": NEW17_ROOT,
    "old23": OLD23_ROOT,
}
OK_LABELS = {"ok_walk", "ok_pt_only"}
POLYCLINIC = "polyclinic"
TARGET_PROVISION_090 = 0.9
PLACEMENT_TARGET090_ROOT = "placement_exact_target90"
STREET_PATTERN_CLASS_TO_COLUMN = {
    "Irregular Grid": "share_irregular_grid",
    "Loops & Lollipops": "share_loops_lollipops",
    "Regular Grid": "share_regular_grid",
    "Warped Parallel": "share_warped_parallel",
    "Broken Grid": "share_broken_grid",
    "Sparse": "share_sparse",
}
STREET_PATTERN_CLASS_ORDER = list(STREET_PATTERN_CLASS_TO_COLUMN.keys())
RQ_OUTCOMES = ["coverage", "accessibility_gap_share", "additional_polyclinics_needed_to_0_9"]
RQ_PREDICTORS = [
    "street_pattern_cells",
    "share_irregular_grid",
    "share_loops_lollipops",
    "share_regular_grid",
    "share_warped_parallel",
    "share_broken_grid",
    "share_sparse",
    "pt_modality_count",
    "pt_route_count",
    "pt_stop_count",
]


def _load_pipeline2_prepare_solver_inputs_module():
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec = importlib.util.spec_from_file_location("pipeline2_prepare_solver_inputs_local", PIPELINE2_PREPARE_SOLVER_INPUTS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {PIPELINE2_PREPARE_SOLVER_INPUTS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _city_dir(source: str, city: str) -> Path:
    return ROOT_BY_SOURCE[source] / city


def build_default_city_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for city in ACTIVE19_CITIES:
        if city in seen:
            continue
        seen.add(city)
        rows.append(
            {
                "city": city,
                "source": "active19",
                "source_priority": SOURCE_PRIORITY.index("active19"),
                "city_dir": str(_city_dir("active19", city)),
                "access_diagnostics_path": str(ACCESS_DIAGNOSTICS_BY_SOURCE["active19"]),
            }
        )

    for city in NEW17_ELIGIBLE_CITIES:
        if city in seen:
            continue
        seen.add(city)
        rows.append(
            {
                "city": city,
                "source": "new17",
                "source_priority": SOURCE_PRIORITY.index("new17"),
                "city_dir": str(_city_dir("new17", city)),
                "access_diagnostics_path": str(ACCESS_DIAGNOSTICS_BY_SOURCE["new17"]),
            }
        )

    for city in OLD23_ELIGIBLE_UNIQUE_CITIES:
        if city in seen:
            continue
        seen.add(city)
        rows.append(
            {
                "city": city,
                "source": "old23",
                "source_priority": SOURCE_PRIORITY.index("old23"),
                "city_dir": str(_city_dir("old23", city)),
                "access_diagnostics_path": str(ACCESS_DIAGNOSTICS_BY_SOURCE["old23"]),
            }
        )

    return pd.DataFrame(rows).sort_values(["source_priority", "city"]).reset_index(drop=True)


def verify_city_registry_bundle(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in registry.to_dict(orient="records"):
        city_dir = Path(str(row["city_dir"]))
        city = str(row["city"])

        street_pattern_summary = city_dir / "street_pattern" / f"{city}_summary.json"
        graph_path = city_dir / "intermodal_graph_iduedu/graph.pkl"
        solver_blocks_path = city_dir / "pipeline_2/solver_inputs/polyclinic/blocks_solver.parquet"
        solver_summary_path = city_dir / "pipeline_2/solver_inputs/polyclinic/summary.json"
        services_raw_path = city_dir / "pipeline_2/services_raw/polyclinic.parquet"

        rows.append(
            {
                **row,
                "street_pattern_summary_path": str(street_pattern_summary),
                "graph_path": str(graph_path),
                "solver_blocks_path": str(solver_blocks_path),
                "solver_summary_path": str(solver_summary_path),
                "services_raw_path": str(services_raw_path),
                "has_street_pattern": street_pattern_summary.exists(),
                "has_graph": graph_path.exists(),
                "has_solver_blocks": solver_blocks_path.exists(),
                "has_solver_summary": solver_summary_path.exists(),
                "has_services_raw": services_raw_path.exists(),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_priority", "city"]).reset_index(drop=True)


def build_city_level_baseline_coverage(registry: pd.DataFrame) -> pd.DataFrame:
    diagnostics_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for row in registry.to_dict(orient="records"):
        diagnostics_path = str(row["access_diagnostics_path"])
        if diagnostics_path not in diagnostics_cache:
            diagnostics_cache[diagnostics_path] = pd.read_parquet(
                diagnostics_path,
                columns=["city", "service_name", "access_diagnosis_label"],
            )
        df = diagnostics_cache[diagnostics_path]
        city_df = df[(df["city"] == row["city"]) & (df["service_name"] == POLYCLINIC)].copy()
        ok_mask = city_df["access_diagnosis_label"].isin(OK_LABELS)
        n_homes = int(len(city_df))
        rows.append(
            {
                **row,
                "n_homes": n_homes,
                "ok_count": int(ok_mask.sum()),
                "not_ok_count": int((~ok_mask).sum()),
                "coverage": float(ok_mask.mean()) if n_homes else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    sort_cols = [col for col in ["source_priority", "city"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def load_solver_summary_fields(summary_path: Path) -> dict[str, float]:
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    return {
        "blocks_count": float(payload.get("blocks_count", 0.0)),
        "demand_total": float(payload.get("demand_total", 0.0)),
        "accessibility_gap_total": float(payload.get("demand_without_total", 0.0)),
        "capacity_total": float(payload.get("capacity_total", 0.0)),
        "provision_total": float(payload.get("provision_total", 0.0)),
    }


def load_street_pattern_mix_fields(summary_path: Path) -> dict[str, float]:
    payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    total = float(payload.get("num_predictions", 0.0) or 0.0)
    class_counts = payload.get("class_counts", {}) or {}
    row: dict[str, float] = {"street_pattern_cells": total}
    for class_name, column in STREET_PATTERN_CLASS_TO_COLUMN.items():
        count = float(class_counts.get(class_name, 0.0) or 0.0)
        row[column] = count / total if total else 0.0
    return row


def load_pt_descriptor_fields(city_dir: Path) -> dict[str, float]:
    connectpt_osm_dir = Path(city_dir) / "connectpt_osm"
    modality_count = 0.0
    route_count = 0.0
    stop_count = 0.0

    if connectpt_osm_dir.exists():
        for modality_dir in sorted(path for path in connectpt_osm_dir.iterdir() if path.is_dir()):
            lines_path = modality_dir / "lines.parquet"
            stops_path = modality_dir / "aggregated_stops.parquet"
            if not lines_path.exists() or not stops_path.exists():
                continue
            modality_count += 1.0
            route_count += float(len(pd.read_parquet(lines_path)))
            stop_count += float(len(pd.read_parquet(stops_path)))

    return {
        "pt_modality_count": modality_count,
        "pt_route_count": route_count,
        "pt_stop_count": stop_count,
    }


def _pick_street_pattern_class_column(cells: pd.DataFrame) -> str:
    for candidate in ("top1_class_name", "class_name", "street_pattern_class", "predicted_class"):
        if candidate in cells.columns:
            return candidate
    raise KeyError("No street-pattern class column found in predicted street-pattern cells.")


def _ordered_street_pattern_classes(labels: pd.Series | list[str] | tuple[str, ...] | set[str]) -> list[str]:
    present = {str(value) for value in labels if pd.notna(value)}
    ordered = [label for label in STREET_PATTERN_CLASS_ORDER if label in present]
    if "unknown" in present:
        ordered.append("unknown")
    extras = sorted(present - set(ordered))
    return ordered + extras


def transfer_street_pattern_cells_to_blocks(
    blocks,
    cells,
):
    import geopandas as gpd

    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from aggregated_spatial_pipeline.pipeline.crosswalks import build_crosswalk
    from aggregated_spatial_pipeline.pipeline.transfers import apply_transfer_rule

    class_col = _pick_street_pattern_class_column(cells)
    prob_cols = [column for column in cells.columns if str(column).startswith("prob_")]

    source_columns = ["geometry", class_col, *prob_cols]
    source = cells[source_columns].copy()
    source["grid_id"] = source.index.astype(str)
    if "class_name" not in source.columns:
        source["class_name"] = source[class_col].astype("string").fillna("unknown")

    target = blocks[["geometry"]].copy()
    target["block_id"] = blocks.index.astype(str)
    target["block_name"] = blocks.index.astype(str)

    polygon_types = {"Polygon", "MultiPolygon"}
    source_poly = source[source.geometry.geom_type.isin(polygon_types)].copy()
    target_poly = target[target.geometry.geom_type.isin(polygon_types)].copy()
    if source_poly.empty or target_poly.empty:
        empty = target.copy()
        empty["street_pattern_dominant_class"] = "unknown"
        empty["street_pattern_covered_mass"] = 0.0
        return gpd.GeoDataFrame(empty, geometry="geometry", crs=blocks.crs)

    crosswalk = build_crosswalk(source_poly, target_poly, "grid", "block")
    transferred_poly = target_poly.copy()
    if prob_cols:
        transferred_poly = apply_transfer_rule(
            source_gdf=source_poly,
            target_gdf=transferred_poly,
            crosswalk_gdf=crosswalk,
            source_layer="grid",
            target_layer="block",
            attribute="street_pattern_probs",
            aggregation_method="weighted_mean",
            weight_field="intersection_area",
        )
    transferred_poly = apply_transfer_rule(
        source_gdf=source_poly,
        target_gdf=transferred_poly,
        crosswalk_gdf=crosswalk,
        source_layer="grid",
        target_layer="block",
        attribute="street_pattern_class",
        aggregation_method="majority_vote",
        weight_field="intersection_area",
    )

    transferred = target.copy()
    for column in transferred_poly.columns:
        if column == "geometry":
            continue
        transferred[column] = pd.Series(index=transferred.index, dtype=transferred_poly[column].dtype)
        transferred.loc[transferred_poly.index, column] = transferred_poly[column]

    for prob_col in prob_cols:
        if prob_col not in transferred.columns:
            transferred[prob_col] = 0.0
        transferred[prob_col] = pd.to_numeric(transferred[prob_col], errors="coerce").fillna(0.0)

    if prob_cols:
        transferred["street_pattern_covered_mass"] = transferred[prob_cols].sum(axis=1)
        fallback = pd.Series("unknown", index=transferred.index, dtype=object)
        covered_mask = transferred["street_pattern_covered_mass"] > 0.0
        if covered_mask.any():
            fallback.loc[covered_mask] = (
                transferred.loc[covered_mask, prob_cols]
                .idxmax(axis=1)
                .map(
                    {
                        "prob_0": "Loops & Lollipops",
                        "prob_1": "Irregular Grid",
                        "prob_2": "Regular Grid",
                        "prob_3": "Warped Parallel",
                        "prob_4": "Sparse",
                        "prob_5": "Broken Grid",
                    }
                )
                .fillna("unknown")
            )
    else:
        transferred["street_pattern_covered_mass"] = 0.0
        fallback = pd.Series("unknown", index=transferred.index, dtype=object)

    transferred["street_pattern_dominant_class"] = (
        transferred.get("street_pattern_class", fallback)
        .fillna(fallback)
        .astype(str)
    )
    no_class_mask = transferred["street_pattern_dominant_class"].str.strip().eq("")
    transferred.loc[no_class_mask, "street_pattern_dominant_class"] = fallback.loc[no_class_mask]
    return gpd.GeoDataFrame(transferred, geometry="geometry", crs=blocks.crs)


def _share_frame_from_values(
    values: pd.Series,
    *,
    count_col: str,
    share_col: str,
    label_col: str = "street_pattern_dominant_class",
) -> pd.DataFrame:
    prepared = values.fillna("unknown").astype(str)
    counts = prepared.value_counts(dropna=False).rename_axis(label_col).reset_index(name=count_col)
    total = float(counts[count_col].sum())
    counts[share_col] = counts[count_col] / total if total > 0.0 else 0.0
    return counts


def build_city_target90_pattern_lift_rows(
    *,
    city: str,
    block_patterns: pd.DataFrame,
    candidate_block_names: list[str] | pd.Series,
    selected_block_names: list[str] | pd.Series,
) -> pd.DataFrame:
    base = block_patterns[["block_name", "street_pattern_dominant_class"]].copy()
    base["block_name"] = base["block_name"].astype(str)
    base["street_pattern_dominant_class"] = base["street_pattern_dominant_class"].fillna("unknown").astype(str)

    candidate_names = {str(value) for value in candidate_block_names}
    selected_names = {str(value) for value in selected_block_names}

    city_counts = _share_frame_from_values(
        base["street_pattern_dominant_class"],
        count_col="city_count",
        share_col="city_share",
    )
    candidate_counts = _share_frame_from_values(
        base.loc[base["block_name"].isin(candidate_names), "street_pattern_dominant_class"],
        count_col="candidate_count",
        share_col="candidate_share",
    )
    selected_counts = _share_frame_from_values(
        base.loc[base["block_name"].isin(selected_names), "street_pattern_dominant_class"],
        count_col="selected_count",
        share_col="selected_share",
    )

    out = city_counts.merge(candidate_counts, on="street_pattern_dominant_class", how="outer")
    out = out.merge(selected_counts, on="street_pattern_dominant_class", how="outer")
    for column in ("city_count", "city_share", "candidate_count", "candidate_share", "selected_count", "selected_share"):
        out[column] = pd.to_numeric(out.get(column), errors="coerce").fillna(0.0)

    out["placement_lift_vs_city"] = out["selected_share"] - out["city_share"]
    out["placement_lift_vs_candidates"] = out["selected_share"] - out["candidate_share"]
    out["placement_ratio_vs_city"] = np.where(out["city_share"] > 0.0, out["selected_share"] / out["city_share"], np.nan)
    out["placement_ratio_vs_candidates"] = np.where(
        out["candidate_share"] > 0.0,
        out["selected_share"] / out["candidate_share"],
        np.nan,
    )
    out["city"] = city
    order_map = {label: idx for idx, label in enumerate(_ordered_street_pattern_classes(out["street_pattern_dominant_class"]))}
    out["sort_key"] = out["street_pattern_dominant_class"].map(order_map)
    return out.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)


def build_overall_target90_pattern_lift_rows(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "street_pattern_dominant_class",
                "city_count",
                "city_share",
                "candidate_count",
                "candidate_share",
                "selected_count",
                "selected_share",
                "placement_lift_vs_city",
                "placement_lift_vs_candidates",
                "placement_ratio_vs_city",
                "placement_ratio_vs_candidates",
            ]
        )

    grouped = (
        detail.groupby("street_pattern_dominant_class", as_index=False)[["city_count", "candidate_count", "selected_count"]]
        .sum()
    )
    city_total = float(grouped["city_count"].sum())
    candidate_total = float(grouped["candidate_count"].sum())
    selected_total = float(grouped["selected_count"].sum())
    grouped["city_share"] = grouped["city_count"] / city_total if city_total > 0.0 else 0.0
    grouped["candidate_share"] = grouped["candidate_count"] / candidate_total if candidate_total > 0.0 else 0.0
    grouped["selected_share"] = grouped["selected_count"] / selected_total if selected_total > 0.0 else 0.0
    grouped["placement_lift_vs_city"] = grouped["selected_share"] - grouped["city_share"]
    grouped["placement_lift_vs_candidates"] = grouped["selected_share"] - grouped["candidate_share"]
    grouped["placement_ratio_vs_city"] = np.where(
        grouped["city_share"] > 0.0,
        grouped["selected_share"] / grouped["city_share"],
        np.nan,
    )
    grouped["placement_ratio_vs_candidates"] = np.where(
        grouped["candidate_share"] > 0.0,
        grouped["selected_share"] / grouped["candidate_share"],
        np.nan,
    )
    order_map = {label: idx for idx, label in enumerate(_ordered_street_pattern_classes(grouped["street_pattern_dominant_class"]))}
    grouped["sort_key"] = grouped["street_pattern_dominant_class"].map(order_map)
    return grouped.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)


def _safe_share(numerator: pd.Series | float, denominator: float) -> pd.Series | float:
    if float(denominator) <= 0.0:
        if isinstance(numerator, pd.Series):
            return pd.Series(0.0, index=numerator.index)
        return 0.0
    return numerator / float(denominator)


def build_pattern_demand_supply_rows(city: str, solver_blocks: pd.DataFrame) -> pd.DataFrame:
    class_col = "street_pattern_dominant_class" if "street_pattern_dominant_class" in solver_blocks.columns else "street_pattern_top1_class"
    if class_col not in solver_blocks.columns:
        class_col = "street_pattern_dominant_class"
        solver_blocks = solver_blocks.copy()
        solver_blocks[class_col] = "unknown"

    prepared = pd.DataFrame(
        {
            "street_pattern_dominant_class": solver_blocks[class_col].fillna("unknown").astype(str),
            "demand": pd.to_numeric(solver_blocks.get("demand", 0.0), errors="coerce").fillna(0.0),
            "capacity": pd.to_numeric(solver_blocks.get("capacity", 0.0), errors="coerce").fillna(0.0),
            "provision": pd.to_numeric(solver_blocks.get("provision", 0.0), errors="coerce").fillna(0.0),
            "demand_without": pd.to_numeric(solver_blocks.get("demand_without", 0.0), errors="coerce").fillna(0.0),
            "demand_left": pd.to_numeric(solver_blocks.get("demand_left", 0.0), errors="coerce").fillna(0.0),
        }
    )
    prepared["unmet_demand"] = prepared["demand_without"] + prepared["demand_left"]

    grouped = (
        prepared.groupby("street_pattern_dominant_class", as_index=False)
        .agg(
            block_count=("street_pattern_dominant_class", "size"),
            demand=("demand", "sum"),
            capacity=("capacity", "sum"),
            provision=("provision", "sum"),
            unmet_demand=("unmet_demand", "sum"),
        )
    )
    demand_total = float(grouped["demand"].sum())
    capacity_total = float(grouped["capacity"].sum())
    provision_total = float(grouped["provision"].sum())
    unmet_total = float(grouped["unmet_demand"].sum())

    grouped["demand_share"] = _safe_share(grouped["demand"], demand_total)
    grouped["capacity_share"] = _safe_share(grouped["capacity"], capacity_total)
    grouped["provision_share"] = _safe_share(grouped["provision"], provision_total)
    grouped["unmet_share"] = _safe_share(grouped["unmet_demand"], unmet_total)
    grouped["coverage_proxy"] = np.where(
        grouped["demand"] > 0.0,
        (grouped["demand"] - grouped["unmet_demand"]) / grouped["demand"],
        0.0,
    )
    grouped["capacity_per_1000_demand"] = np.where(
        grouped["demand"] > 0.0,
        grouped["capacity"] / grouped["demand"] * 1000.0,
        np.nan,
    )
    grouped["supply_demand_share_gap"] = grouped["capacity_share"] - grouped["demand_share"]
    grouped["unmet_demand_share_gap"] = grouped["unmet_share"] - grouped["demand_share"]
    grouped["city"] = city
    order_map = {label: idx for idx, label in enumerate(_ordered_street_pattern_classes(grouped["street_pattern_dominant_class"]))}
    grouped["sort_key"] = grouped["street_pattern_dominant_class"].map(order_map)
    return grouped.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)


def build_overall_pattern_demand_supply_rows(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "street_pattern_dominant_class",
                "block_count",
                "demand",
                "capacity",
                "provision",
                "unmet_demand",
                "demand_share",
                "capacity_share",
                "provision_share",
                "unmet_share",
                "coverage_proxy",
                "capacity_per_1000_demand",
                "supply_demand_share_gap",
                "unmet_demand_share_gap",
            ]
        )
    grouped = (
        detail.groupby("street_pattern_dominant_class", as_index=False)[
            ["block_count", "demand", "capacity", "provision", "unmet_demand"]
        ]
        .sum()
    )
    demand_total = float(grouped["demand"].sum())
    capacity_total = float(grouped["capacity"].sum())
    provision_total = float(grouped["provision"].sum())
    unmet_total = float(grouped["unmet_demand"].sum())
    grouped["demand_share"] = _safe_share(grouped["demand"], demand_total)
    grouped["capacity_share"] = _safe_share(grouped["capacity"], capacity_total)
    grouped["provision_share"] = _safe_share(grouped["provision"], provision_total)
    grouped["unmet_share"] = _safe_share(grouped["unmet_demand"], unmet_total)
    grouped["coverage_proxy"] = np.where(
        grouped["demand"] > 0.0,
        (grouped["demand"] - grouped["unmet_demand"]) / grouped["demand"],
        0.0,
    )
    grouped["capacity_per_1000_demand"] = np.where(
        grouped["demand"] > 0.0,
        grouped["capacity"] / grouped["demand"] * 1000.0,
        np.nan,
    )
    grouped["supply_demand_share_gap"] = grouped["capacity_share"] - grouped["demand_share"]
    grouped["unmet_demand_share_gap"] = grouped["unmet_share"] - grouped["demand_share"]
    order_map = {label: idx for idx, label in enumerate(_ordered_street_pattern_classes(grouped["street_pattern_dominant_class"]))}
    grouped["sort_key"] = grouped["street_pattern_dominant_class"].map(order_map)
    return grouped.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)


def build_pattern_access_failure_rows(city: str, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame(columns=["city", "street_pattern_dominant_class", "home_count", "ok_count", "not_ok_count", "coverage"])
    class_col = "home_street_pattern_class"
    prepared = diagnostics.copy()
    prepared["street_pattern_dominant_class"] = prepared.get(class_col, "unknown")
    prepared["street_pattern_dominant_class"] = prepared["street_pattern_dominant_class"].fillna("unknown").astype(str)
    prepared["is_ok"] = prepared["access_diagnosis_label"].isin(OK_LABELS)
    grouped = (
        prepared.groupby("street_pattern_dominant_class", as_index=False)
        .agg(
            home_count=("access_diagnosis_label", "size"),
            ok_count=("is_ok", "sum"),
        )
    )
    grouped["not_ok_count"] = grouped["home_count"] - grouped["ok_count"]
    grouped["coverage"] = np.where(grouped["home_count"] > 0, grouped["ok_count"] / grouped["home_count"], 0.0)
    for label in sorted(prepared["access_diagnosis_label"].dropna().astype(str).unique()):
        col = f"share_{label}"
        counts = prepared[prepared["access_diagnosis_label"].astype(str) == label].groupby("street_pattern_dominant_class").size()
        grouped[col] = grouped["street_pattern_dominant_class"].map(counts).fillna(0.0) / grouped["home_count"]
    grouped["city"] = city
    order_map = {label: idx for idx, label in enumerate(_ordered_street_pattern_classes(grouped["street_pattern_dominant_class"]))}
    grouped["sort_key"] = grouped["street_pattern_dominant_class"].map(order_map)
    return grouped.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)


def build_overall_pattern_access_failure_rows(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["street_pattern_dominant_class", "home_count", "ok_count", "not_ok_count", "coverage"])
    share_cols = [col for col in detail.columns if col.startswith("share_")]
    work = detail.copy()
    for col in share_cols:
        work[f"count_{col[6:]}"] = pd.to_numeric(work[col], errors="coerce").fillna(0.0) * pd.to_numeric(work["home_count"], errors="coerce").fillna(0.0)
    count_cols = [f"count_{col[6:]}" for col in share_cols]
    grouped = (
        work.groupby("street_pattern_dominant_class", as_index=False)[["home_count", "ok_count", "not_ok_count", *count_cols]]
        .sum()
    )
    grouped["coverage"] = np.where(grouped["home_count"] > 0, grouped["ok_count"] / grouped["home_count"], 0.0)
    for share_col in share_cols:
        count_col = f"count_{share_col[6:]}"
        grouped[share_col] = np.where(grouped["home_count"] > 0, grouped[count_col] / grouped["home_count"], 0.0)
    grouped = grouped.drop(columns=count_cols)
    order_map = {label: idx for idx, label in enumerate(_ordered_street_pattern_classes(grouped["street_pattern_dominant_class"]))}
    grouped["sort_key"] = grouped["street_pattern_dominant_class"].map(order_map)
    return grouped.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)


def build_pattern_pt_route_rows(city: str, route_class_length: pd.DataFrame) -> pd.DataFrame:
    if route_class_length.empty:
        return pd.DataFrame(columns=["city", "street_pattern_dominant_class", "route_pattern_records", "pt_length_m", "pt_length_share"])
    prepared = route_class_length.copy()
    prepared["street_pattern_dominant_class"] = prepared["street_pattern_class"].fillna("unknown").astype(str)
    prepared["pt_length_m"] = pd.to_numeric(prepared.get("pt_length_m", 0.0), errors="coerce").fillna(0.0)
    grouped = (
        prepared.groupby("street_pattern_dominant_class", as_index=False)
        .agg(route_pattern_records=("street_pattern_dominant_class", "size"), pt_length_m=("pt_length_m", "sum"))
    )
    total = float(grouped["pt_length_m"].sum())
    grouped["pt_length_share"] = _safe_share(grouped["pt_length_m"], total)
    grouped["city"] = city
    order_map = {label: idx for idx, label in enumerate(_ordered_street_pattern_classes(grouped["street_pattern_dominant_class"]))}
    grouped["sort_key"] = grouped["street_pattern_dominant_class"].map(order_map)
    return grouped.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)


def build_overall_pattern_pt_route_rows(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["street_pattern_dominant_class", "route_pattern_records", "pt_length_m", "pt_length_share"])
    grouped = detail.groupby("street_pattern_dominant_class", as_index=False)[["route_pattern_records", "pt_length_m"]].sum()
    total = float(grouped["pt_length_m"].sum())
    grouped["pt_length_share"] = _safe_share(grouped["pt_length_m"], total)
    order_map = {label: idx for idx, label in enumerate(_ordered_street_pattern_classes(grouped["street_pattern_dominant_class"]))}
    grouped["sort_key"] = grouped["street_pattern_dominant_class"].map(order_map)
    return grouped.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)


def scale_unmet_demand_to_target_provision(
    solver_blocks: pd.DataFrame,
    *,
    target_provision: float,
) -> dict[str, object]:
    demand = pd.to_numeric(solver_blocks.get("demand", 0.0), errors="coerce").fillna(0.0)
    demand_without = pd.to_numeric(solver_blocks.get("demand_without", 0.0), errors="coerce").fillna(0.0)
    demand_left = pd.to_numeric(solver_blocks.get("demand_left", 0.0), errors="coerce").fillna(0.0)

    demand_total = float(demand.sum())
    full_gap_total = float((demand_without + demand_left).sum())
    baseline_provision = float((demand_total - full_gap_total) / demand_total) if demand_total > 0.0 else 0.0
    target_provision = float(target_provision)
    target_additional_total = max(0.0, (target_provision * demand_total) - (demand_total - full_gap_total))
    target_additional_total = min(target_additional_total, full_gap_total)
    target_fraction = (target_additional_total / full_gap_total) if full_gap_total > 0.0 else 0.0

    return {
        "demand_total": demand_total,
        "full_gap_total": full_gap_total,
        "baseline_provision": baseline_provision,
        "target_provision": target_provision,
        "target_unmet_total": target_additional_total,
        "target_fraction_of_full_gap": target_fraction,
        "scaled_demand_without": demand_without * target_fraction,
        "scaled_demand_left": demand_left * target_fraction,
    }


def build_placement_result_row(
    *,
    city: str,
    summary_after_path: Path,
    demand_total: float,
    target_provision: float,
    baseline_provision: float,
) -> dict[str, object]:
    payload = json.loads(Path(summary_after_path).read_text(encoding="utf-8"))
    files = payload.get("files", {}) or {}
    after_unmet_total = float(payload.get("demand_without_after_total", 0.0)) + float(payload.get("demand_left_after_total", 0.0))
    achieved_provision_after = float((float(demand_total) - after_unmet_total) / float(demand_total)) if float(demand_total) > 0.0 else 0.0
    return {
        "city": city,
        "target_provision": float(target_provision),
        "baseline_provision": float(baseline_provision),
        "achieved_provision_after": achieved_provision_after,
        "additional_polyclinics_needed_to_0_9": float(payload.get("new_count", 0.0)),
        "selected_count_after": float(payload.get("selected_count", 0.0)),
        "expanded_count_after": float(payload.get("expanded_count", 0.0)),
        "capacity_added_total": float(payload.get("capacity_added_total", 0.0)),
        "demand_without_after_total": float(payload.get("demand_without_after_total", 0.0)),
        "demand_left_after_total": float(payload.get("demand_left_after_total", 0.0)),
        "summary_after_path": str(summary_after_path),
        "status_preview_png": files.get("status_preview_png"),
        "after_preview_png": files.get("after_preview_png"),
    }


def build_failed_placement_result_row(
    *,
    city: str,
    error: str,
    demand_total: float,
    full_gap_total: float,
    baseline_provision: float,
    target_provision: float,
    target_unmet_total: float,
    target_fraction_of_full_gap: float,
) -> dict[str, object]:
    return {
        "city": city,
        "placement_status": "failed",
        "placement_error": error,
        "target_provision": float(target_provision),
        "baseline_provision": float(baseline_provision),
        "achieved_provision_after": np.nan,
        "additional_polyclinics_needed_to_0_9": np.nan,
        "selected_count_after": np.nan,
        "expanded_count_after": np.nan,
        "capacity_added_total": np.nan,
        "demand_without_after_total": np.nan,
        "demand_left_after_total": np.nan,
        "summary_after_path": None,
        "status_preview_png": None,
        "after_preview_png": None,
        "demand_total": float(demand_total),
        "full_gap_total": float(full_gap_total),
        "target_unmet_total": float(target_unmet_total),
        "target_fraction_of_full_gap": float(target_fraction_of_full_gap),
    }


def build_solver_summary_rows(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in registry.to_dict(orient="records"):
        city_dir = Path(str(row["city_dir"]))
        summary_path = city_dir / "pipeline_2/solver_inputs/polyclinic/summary.json"
        rows.append({"city": row["city"], **load_solver_summary_fields(summary_path)})
    return pd.DataFrame(rows)


def sort_registry_for_targeted_placement(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in registry.to_dict(orient="records"):
        city_dir = Path(str(row["city_dir"]))
        summary_path = city_dir / "pipeline_2/solver_inputs/polyclinic/summary.json"
        summary = load_solver_summary_fields(summary_path)
        rows.append(
            {
                **row,
                "placement_blocks_count": float(summary.get("blocks_count", 0.0)),
                "placement_demand_total": float(summary.get("demand_total", 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["placement_blocks_count", "placement_demand_total", "city"]).reset_index(drop=True)


def select_registry_subset_for_tiered_run(
    registry: pd.DataFrame,
    *,
    max_cities: int | None = None,
    cities: list[str] | None = None,
) -> pd.DataFrame:
    if {"placement_blocks_count", "placement_demand_total"}.issubset(registry.columns):
        ordered = registry.copy()
        ordered["placement_blocks_count"] = pd.to_numeric(ordered["placement_blocks_count"], errors="coerce").fillna(0.0)
        ordered["placement_demand_total"] = pd.to_numeric(ordered["placement_demand_total"], errors="coerce").fillna(0.0)
        ordered = ordered.sort_values(["placement_blocks_count", "placement_demand_total", "city"]).reset_index(drop=True)
    else:
        ordered = sort_registry_for_targeted_placement(registry)

    if cities:
        requested = {str(city) for city in cities}
        ordered = ordered[ordered["city"].astype(str).isin(requested)].copy()

    if max_cities is not None:
        ordered = ordered.head(int(max_cities)).copy()

    return ordered.reset_index(drop=True)


def build_street_pattern_rows(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in registry.to_dict(orient="records"):
        city_dir = Path(str(row["city_dir"]))
        city = str(row["city"])
        summary_path = city_dir / "street_pattern" / f"{city}_summary.json"
        rows.append({"city": city, **load_street_pattern_mix_fields(summary_path)})
    return pd.DataFrame(rows)


def build_pt_descriptor_rows(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in registry.to_dict(orient="records"):
        city_dir = Path(str(row["city_dir"]))
        rows.append({"city": row["city"], **load_pt_descriptor_fields(city_dir)})
    return pd.DataFrame(rows)


def load_solver_block_patterns(city_dir: Path, city: str) -> pd.DataFrame:
    import geopandas as gpd

    blocks_path = Path(city_dir) / "derived_layers/blocks_clipped.parquet"
    cells_path = Path(city_dir) / "street_pattern" / city / "predicted_cells.geojson"
    blocks = gpd.read_parquet(blocks_path)
    cells = gpd.read_file(cells_path)
    transferred = transfer_street_pattern_cells_to_blocks(blocks, cells)
    return transferred[["block_name", "street_pattern_dominant_class"]].copy()


def build_pattern_system_experiment_tables(registry: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if registry is None:
        registry = build_default_city_registry()

    diagnostics_cache: dict[str, pd.DataFrame] = {}
    demand_supply_rows: list[pd.DataFrame] = []
    access_failure_rows: list[pd.DataFrame] = []
    pt_route_rows: list[pd.DataFrame] = []

    for row in registry.to_dict(orient="records"):
        city = str(row["city"])
        city_dir = Path(str(row["city_dir"]))

        solver_blocks_path = city_dir / "pipeline_2/solver_inputs/polyclinic/blocks_solver.parquet"
        if solver_blocks_path.exists():
            solver_blocks = pd.read_parquet(solver_blocks_path)
            try:
                block_patterns = load_solver_block_patterns(city_dir, city)
                solver_blocks = solver_blocks.copy()
                solver_blocks["block_name"] = solver_blocks.get("name", solver_blocks.index.astype(str)).astype(str)
                block_patterns["block_name"] = block_patterns["block_name"].astype(str)
                solver_blocks = solver_blocks.merge(block_patterns, on="block_name", how="left")
            except Exception as exc:  # noqa: BLE001
                print(f"pattern transfer failed for {city}: {exc}")
            demand_supply_rows.append(build_pattern_demand_supply_rows(city, solver_blocks))

        diagnostics_path = str(row["access_diagnostics_path"])
        if diagnostics_path not in diagnostics_cache:
            diagnostics_cache[diagnostics_path] = pd.read_parquet(
                diagnostics_path,
                columns=["city", "service_name", "access_diagnosis_label", "home_street_pattern_class"],
            )
        diagnostics = diagnostics_cache[diagnostics_path]
        city_diagnostics = diagnostics[(diagnostics["city"] == city) & (diagnostics["service_name"] == POLYCLINIC)].copy()
        access_failure_rows.append(build_pattern_access_failure_rows(city, city_diagnostics))

        route_class_path = city_dir / "pt_street_pattern_dependency/route_class_length.csv"
        if route_class_path.exists():
            route_class_length = pd.read_csv(route_class_path)
            pt_route_rows.append(build_pattern_pt_route_rows(city, route_class_length))

    demand_supply_detail = pd.concat(demand_supply_rows, ignore_index=True) if demand_supply_rows else pd.DataFrame()
    access_failure_detail = pd.concat(access_failure_rows, ignore_index=True) if access_failure_rows else pd.DataFrame()
    pt_route_detail = pd.concat(pt_route_rows, ignore_index=True) if pt_route_rows else pd.DataFrame()

    return {
        "pattern_demand_supply_by_city": demand_supply_detail,
        "pattern_demand_supply_overall": build_overall_pattern_demand_supply_rows(demand_supply_detail),
        "pattern_access_failures_by_city": access_failure_detail,
        "pattern_access_failures_overall": build_overall_pattern_access_failure_rows(access_failure_detail),
        "pattern_pt_routes_by_city": pt_route_detail,
        "pattern_pt_routes_overall": build_overall_pattern_pt_route_rows(pt_route_detail),
    }


def run_targeted_placement_for_city(
    city_dir: Path,
    *,
    city: str,
    target_provision: float = TARGET_PROVISION_090,
    placement_root_name: str = PLACEMENT_TARGET090_ROOT,
    use_genetic: bool = False,
) -> dict[str, object]:
    import geopandas as gpd

    city_dir = Path(city_dir)
    manifest_path = city_dir / "pipeline_2/manifest_prepare_solver_inputs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    solver_dir = city_dir / "pipeline_2/solver_inputs" / POLYCLINIC
    solver_blocks = gpd.read_parquet(solver_dir / "blocks_solver.parquet")
    sub_mx = pd.read_parquet(solver_dir / "adj_matrix_time_min.parquet")
    blocks_ref = gpd.read_parquet(city_dir / "derived_layers/blocks_clipped.parquet")
    boundary = gpd.read_parquet(city_dir / "analysis_territory/buffer.parquet")
    preview_dir = city_dir / "preview_png/all_together"
    output_dir = city_dir / "pipeline_2" / placement_root_name / POLYCLINIC

    scaled = scale_unmet_demand_to_target_provision(solver_blocks, target_provision=target_provision)
    if float(scaled["target_unmet_total"]) <= 0.0:
        return {
            "city": city,
            "target_provision": float(target_provision),
            "baseline_provision": float(scaled["baseline_provision"]),
            "achieved_provision_after": float(scaled["baseline_provision"]),
            "additional_polyclinics_needed_to_0_9": 0.0,
            "selected_count_after": 0.0,
            "expanded_count_after": 0.0,
            "capacity_added_total": 0.0,
            "demand_without_after_total": float(pd.to_numeric(solver_blocks.get("demand_without", 0.0), errors="coerce").fillna(0.0).sum()),
            "demand_left_after_total": float(pd.to_numeric(solver_blocks.get("demand_left", 0.0), errors="coerce").fillna(0.0).sum()),
            "summary_after_path": None,
            "status_preview_png": None,
            "after_preview_png": None,
            "demand_total": float(scaled["demand_total"]),
            "full_gap_total": float(scaled["full_gap_total"]),
            "target_unmet_total": float(scaled["target_unmet_total"]),
            "target_fraction_of_full_gap": float(scaled["target_fraction_of_full_gap"]),
        }

    scaled_blocks = solver_blocks.copy()
    scaled_blocks["demand_without"] = scaled["scaled_demand_without"]
    scaled_blocks["demand_left"] = scaled["scaled_demand_left"]

    pipeline2_module = _load_pipeline2_prepare_solver_inputs_module()
    try:
        outputs = pipeline2_module._run_exact_placement_for_service(
            scaled_blocks,
            sub_mx,
            POLYCLINIC,
            output_dir,
            preview_dir=preview_dir,
            blocks_ref=blocks_ref,
            boundary=boundary,
            use_genetic=bool(use_genetic),
            progress=False,
            prefer_existing=False,
            allow_existing_expansion=False,
            capacity_mode="fixed_mean",
            use_cache=True,
        )
    except Exception as exc:  # noqa: BLE001
        return build_failed_placement_result_row(
            city=city,
            error=str(exc),
            demand_total=float(scaled["demand_total"]),
            full_gap_total=float(scaled["full_gap_total"]),
            baseline_provision=float(scaled["baseline_provision"]),
            target_provision=float(target_provision),
            target_unmet_total=float(scaled["target_unmet_total"]),
            target_fraction_of_full_gap=float(scaled["target_fraction_of_full_gap"]),
        )

    result = build_placement_result_row(
        city=city,
        summary_after_path=Path(outputs["summary_after"]),
        demand_total=float(scaled["demand_total"]),
        target_provision=float(target_provision),
        baseline_provision=float(scaled["baseline_provision"]),
    )
    result["demand_total"] = float(scaled["demand_total"])
    result["full_gap_total"] = float(scaled["full_gap_total"])
    result["target_unmet_total"] = float(scaled["target_unmet_total"])
    result["target_fraction_of_full_gap"] = float(scaled["target_fraction_of_full_gap"])
    result["placement_status"] = "ok"
    result["placement_error"] = None
    return result


def build_targeted_placement_rows(
    registry: pd.DataFrame,
    *,
    target_provision: float = TARGET_PROVISION_090,
    placement_root_name: str = PLACEMENT_TARGET090_ROOT,
    use_genetic: bool = False,
) -> pd.DataFrame:
    ordered_registry = sort_registry_for_targeted_placement(registry)
    rows: list[dict[str, object]] = []
    for row in ordered_registry.to_dict(orient="records"):
        placement_row = run_targeted_placement_for_city(
            Path(str(row["city_dir"])),
            city=str(row["city"]),
            target_provision=target_provision,
            placement_root_name=placement_root_name,
            use_genetic=use_genetic,
        )
        rows.append(
            {
                **row,
                **placement_row,
            }
        )
    out = pd.DataFrame(rows)
    sort_cols = [col for col in ["source_priority", "city"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _merge_city_level_layers(
    registry: pd.DataFrame,
    baseline: pd.DataFrame,
    solver_rows: pd.DataFrame,
    street_rows: pd.DataFrame,
    pt_rows: pd.DataFrame,
) -> pd.DataFrame:
    out = registry.merge(baseline, on=["city", "source", "source_priority", "city_dir", "access_diagnostics_path"], how="left")
    out = out.merge(solver_rows, on="city", how="left")
    out = out.merge(street_rows, on="city", how="left")
    out = out.merge(pt_rows, on="city", how="left")

    demand = pd.to_numeric(out.get("demand_total"), errors="coerce").fillna(0.0)
    gap = pd.to_numeric(out.get("accessibility_gap_total"), errors="coerce").fillna(0.0)
    out["accessibility_gap_share"] = gap.where(demand > 0.0, 0.0) / demand.where(demand > 0.0, 1.0)
    sort_cols = [col for col in ["source_priority", "city"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def build_city_level_research_dataset(registry: pd.DataFrame | None = None) -> pd.DataFrame:
    if registry is None:
        registry = build_default_city_registry()
    baseline = build_city_level_baseline_coverage(registry)
    solver_rows = build_solver_summary_rows(registry)
    street_rows = build_street_pattern_rows(registry)
    pt_rows = build_pt_descriptor_rows(registry)
    return _merge_city_level_layers(
        registry=registry,
        baseline=baseline,
        solver_rows=solver_rows,
        street_rows=street_rows,
        pt_rows=pt_rows,
    )


def build_city_level_target90_dataset(
    registry: pd.DataFrame | None = None,
    *,
    target_provision: float = TARGET_PROVISION_090,
    placement_root_name: str = PLACEMENT_TARGET090_ROOT,
    use_genetic: bool = False,
) -> pd.DataFrame:
    if registry is None:
        registry = build_default_city_registry()
    base = build_city_level_research_dataset(registry)
    placement = build_targeted_placement_rows(
        registry,
        target_provision=target_provision,
        placement_root_name=placement_root_name,
        use_genetic=use_genetic,
    )
    keep_cols = [
        "city",
        "target_provision",
        "baseline_provision",
        "achieved_provision_after",
        "additional_polyclinics_needed_to_0_9",
        "selected_count_after",
        "expanded_count_after",
        "capacity_added_total",
        "demand_without_after_total",
        "demand_left_after_total",
        "summary_after_path",
        "status_preview_png",
        "after_preview_png",
        "demand_total",
        "full_gap_total",
        "target_unmet_total",
        "target_fraction_of_full_gap",
        "placement_status",
        "placement_error",
    ]
    placement = placement[keep_cols].copy()
    out = base.merge(placement, on="city", how="left")
    sort_cols = [col for col in ["source_priority", "city"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def build_research_question_association_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for outcome in RQ_OUTCOMES:
        if outcome not in df.columns:
            continue
        for predictor in RQ_PREDICTORS:
            if predictor not in df.columns:
                continue
            subset = df[[outcome, predictor]].apply(pd.to_numeric, errors="coerce").dropna()
            n = int(len(subset))
            rho = float(subset[outcome].corr(subset[predictor], method="spearman")) if n >= 2 else float("nan")
            rows.append(
                {
                    "outcome": outcome,
                    "predictor": predictor,
                    "n": n,
                    "spearman_rho": rho,
                }
            )
    return pd.DataFrame(rows)


def load_city_target90_pattern_lift_rows(
    *,
    city: str,
    city_dir: Path,
) -> pd.DataFrame:
    import geopandas as gpd

    city_dir = Path(city_dir)
    blocks_path = city_dir / "derived_layers/blocks_clipped.parquet"
    cells_path = city_dir / "street_pattern" / city / "predicted_cells.geojson"
    solver_path = city_dir / "pipeline_2/solver_inputs/polyclinic/blocks_solver.parquet"
    after_path = city_dir / "pipeline_2" / PLACEMENT_TARGET090_ROOT / POLYCLINIC / "blocks_solver_after.parquet"

    if not blocks_path.exists() or not cells_path.exists() or not solver_path.exists() or not after_path.exists():
        return pd.DataFrame()

    blocks = gpd.read_parquet(blocks_path)
    cells = gpd.read_file(cells_path)
    solver = pd.read_parquet(solver_path)
    after = pd.read_parquet(after_path)

    transferred = transfer_street_pattern_cells_to_blocks(blocks, cells)
    block_patterns = transferred[["block_name", "street_pattern_dominant_class"]].copy()
    candidate_series = solver["name"] if "name" in solver.columns else pd.Series(dtype=object)
    candidate_block_names = candidate_series.astype(str).tolist()
    if {"placement_status", "name"}.issubset(after.columns):
        selected_block_names = after.loc[
            after["placement_status"].astype(str) == "new",
            "name",
        ].astype(str).tolist()
    else:
        selected_block_names = []
    return build_city_target90_pattern_lift_rows(
        city=city,
        block_patterns=block_patterns,
        candidate_block_names=candidate_block_names,
        selected_block_names=selected_block_names,
    )


def build_target90_pattern_lift_detail(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for row in registry.to_dict(orient="records"):
        city_rows = load_city_target90_pattern_lift_rows(
            city=str(row["city"]),
            city_dir=Path(str(row["city_dir"])),
        )
        if not city_rows.empty:
            rows.append(city_rows)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    sort_cols = [col for col in ["city", "street_pattern_dominant_class"] if col in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def render_target90_pattern_lift_png(
    detail: pd.DataFrame,
    overall: pd.DataFrame,
    out_path: Path,
) -> Path:
    if overall.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No completed target90 pattern-lift runs yet.", ha="center", va="center")
        ax.axis("off")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    class_order = _ordered_street_pattern_classes(overall["street_pattern_dominant_class"])
    overall_plot = overall.set_index("street_pattern_dominant_class").reindex(class_order).fillna(0.0).reset_index()
    cities = sorted(detail["city"].dropna().astype(str).unique().tolist()) if not detail.empty else []
    heatmap = pd.DataFrame(index=cities, columns=class_order, dtype=float)
    if not detail.empty:
        pivot = detail.pivot(index="city", columns="street_pattern_dominant_class", values="placement_lift_vs_candidates")
        heatmap.loc[pivot.index, pivot.columns] = pivot
    heatmap = heatmap.fillna(0.0)
    vmax = float(np.abs(heatmap.to_numpy(dtype=float)).max()) if not heatmap.empty else 0.0
    vmax = max(vmax, 1e-9)

    fig, (ax_top, ax_mid, ax_bottom) = plt.subplots(
        3,
        1,
        figsize=(16, 18),
        gridspec_kw={"height_ratios": [1.1, 1.1, 2.6]},
    )

    y = np.arange(len(overall_plot))
    ax_top.barh(y - 0.18, overall_plot["city_share"], height=0.34, color="#94a3b8", label="city")
    ax_top.barh(y + 0.18, overall_plot["candidate_share"], height=0.34, color="#64748b", label="candidates")
    ax_top.barh(y, overall_plot["selected_share"], height=0.16, color="#1d4ed8", label="selected new")
    ax_top.set_yticks(y)
    ax_top.set_yticklabels(overall_plot["street_pattern_dominant_class"])
    ax_top.set_xlim(0.0, 1.0)
    ax_top.set_xlabel("share")
    ax_top.set_title("Target 0.9 selected polyclinic blocks: aggregated street-pattern shares")
    ax_top.legend(frameon=False, loc="lower right")

    colors = np.where(overall_plot["placement_lift_vs_candidates"] >= 0.0, "#2563eb", "#dc2626")
    ax_mid.barh(overall_plot["street_pattern_dominant_class"], overall_plot["placement_lift_vs_candidates"], color=colors)
    ax_mid.axvline(0.0, color="#111827", linewidth=1.0)
    ax_mid.set_xlim(-1.0, 1.0)
    ax_mid.set_xlabel("selected share - candidate share")
    ax_mid.set_title("Target 0.9 placement lift by street pattern (vs candidate baseline)")

    image = ax_bottom.imshow(heatmap.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax_bottom.set_xticks(np.arange(len(class_order)))
    ax_bottom.set_xticklabels(class_order, rotation=45, ha="right")
    ax_bottom.set_yticks(np.arange(len(cities)))
    ax_bottom.set_yticklabels(cities)
    ax_bottom.set_title("City-level placement lift by street pattern (vs candidate baseline)")
    ax_bottom.set_xlabel("street pattern")
    ax_bottom.set_ylabel("city")
    fig.colorbar(image, ax=ax_bottom, fraction=0.025, pad=0.02, label="lift")

    fig.suptitle("Polyclinic target 0.9 placement lift", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_target90_overview_png(
    dataset: pd.DataFrame,
    associations: pd.DataFrame,
    out_path: Path,
) -> Path:
    outcome = "additional_polyclinics_needed_to_0_9"
    assoc = associations[associations["outcome"] == outcome].copy()
    assoc = assoc.sort_values("spearman_rho")
    cities = dataset.sort_values(outcome, ascending=False).copy()

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(16, 18),
        gridspec_kw={"height_ratios": [1.0, 2.2]},
    )

    ax_top.barh(assoc["predictor"], assoc["spearman_rho"], color=np.where(assoc["spearman_rho"] >= 0.0, "#2563eb", "#dc2626"))
    ax_top.axvline(0.0, color="#111827", linewidth=1.0)
    ax_top.set_xlim(-1.0, 1.0)
    ax_top.set_xlabel("Spearman rho")
    ax_top.set_title("Additional polyclinics to 0.9: cross-city associations")

    ax_bottom.bar(range(len(cities)), cities[outcome], color="#1d4ed8", alpha=0.9)
    ax_bottom.set_xticks(range(len(cities)))
    ax_bottom.set_xticklabels(cities["city"], rotation=90, fontsize=8)
    ax_bottom.set_ylabel("new polyclinics needed")
    ax_bottom.set_title("Additional polyclinics needed to reach 0.9 target provision by city")

    fig.suptitle("Polyclinic placement target 0.9", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_city_level_outputs(output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = build_default_city_registry()
    verified = verify_city_registry_bundle(registry)
    baseline = build_city_level_baseline_coverage(registry)
    dataset = build_city_level_research_dataset(registry)
    associations = build_research_question_association_summary(dataset)
    target90_dataset = build_city_level_target90_dataset(registry)
    target90_associations = build_research_question_association_summary(target90_dataset)
    target90_pattern_lift_detail = build_target90_pattern_lift_detail(registry)
    target90_pattern_lift_overall = build_overall_target90_pattern_lift_rows(target90_pattern_lift_detail)

    outputs = {
        "city_registry": output_dir / "city_registry.csv",
        "city_registry_verified": output_dir / "city_registry_verified.csv",
        "city_baseline_coverage": output_dir / "city_baseline_coverage.csv",
        "city_research_question_dataset": output_dir / "city_research_question_dataset.csv",
        "city_research_question_association_summary": output_dir / "city_research_question_association_summary.csv",
        "city_target90_dataset": output_dir / "city_target90_dataset.csv",
        "city_target90_association_summary": output_dir / "city_target90_association_summary.csv",
        "city_target90_overview_png": output_dir / "city_target90_overview.png",
        "city_target90_pattern_lift_detail": output_dir / "city_target90_pattern_lift_detail.csv",
        "city_target90_pattern_lift_overall": output_dir / "city_target90_pattern_lift_overall.csv",
        "city_target90_pattern_lift_png": output_dir / "city_target90_pattern_lift.png",
    }
    registry.to_csv(outputs["city_registry"], index=False)
    verified.to_csv(outputs["city_registry_verified"], index=False)
    baseline.to_csv(outputs["city_baseline_coverage"], index=False)
    dataset.to_csv(outputs["city_research_question_dataset"], index=False)
    associations.to_csv(outputs["city_research_question_association_summary"], index=False)
    target90_dataset.to_csv(outputs["city_target90_dataset"], index=False)
    target90_associations.to_csv(outputs["city_target90_association_summary"], index=False)
    render_target90_overview_png(target90_dataset, target90_associations, outputs["city_target90_overview_png"])
    target90_pattern_lift_detail.to_csv(outputs["city_target90_pattern_lift_detail"], index=False)
    target90_pattern_lift_overall.to_csv(outputs["city_target90_pattern_lift_overall"], index=False)
    render_target90_pattern_lift_png(
        target90_pattern_lift_detail,
        target90_pattern_lift_overall,
        outputs["city_target90_pattern_lift_png"],
    )
    return outputs


def write_tiered_target90_outputs(
    *,
    output_dir: Path | None = None,
    max_cities: int | None = None,
    cities: list[str] | None = None,
    placement_root_name: str = PLACEMENT_TARGET090_ROOT,
    target_provision: float = TARGET_PROVISION_090,
    use_genetic: bool = False,
) -> dict[str, Path]:
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = select_registry_subset_for_tiered_run(
        build_default_city_registry(),
        max_cities=max_cities,
        cities=cities,
    )
    verified = verify_city_registry_bundle(registry)
    target90_dataset = build_city_level_target90_dataset(
        registry,
        target_provision=target_provision,
        placement_root_name=placement_root_name,
        use_genetic=use_genetic,
    )
    target90_associations = build_research_question_association_summary(target90_dataset)
    target90_pattern_lift_detail = build_target90_pattern_lift_detail(registry)
    target90_pattern_lift_overall = build_overall_target90_pattern_lift_rows(target90_pattern_lift_detail)

    outputs = {
        "tiered_registry": output_dir / "city_target90_tiered_registry.csv",
        "tiered_registry_verified": output_dir / "city_target90_tiered_registry_verified.csv",
        "tiered_dataset": output_dir / "city_target90_tiered_dataset.csv",
        "tiered_association_summary": output_dir / "city_target90_tiered_association_summary.csv",
        "tiered_overview_png": output_dir / "city_target90_tiered_overview.png",
        "tiered_pattern_lift_detail": output_dir / "city_target90_tiered_pattern_lift_detail.csv",
        "tiered_pattern_lift_overall": output_dir / "city_target90_tiered_pattern_lift_overall.csv",
        "tiered_pattern_lift_png": output_dir / "city_target90_tiered_pattern_lift.png",
    }
    registry.to_csv(outputs["tiered_registry"], index=False)
    verified.to_csv(outputs["tiered_registry_verified"], index=False)
    target90_dataset.to_csv(outputs["tiered_dataset"], index=False)
    target90_associations.to_csv(outputs["tiered_association_summary"], index=False)
    render_target90_overview_png(target90_dataset, target90_associations, outputs["tiered_overview_png"])
    target90_pattern_lift_detail.to_csv(outputs["tiered_pattern_lift_detail"], index=False)
    target90_pattern_lift_overall.to_csv(outputs["tiered_pattern_lift_overall"], index=False)
    render_target90_pattern_lift_png(
        target90_pattern_lift_detail,
        target90_pattern_lift_overall,
        outputs["tiered_pattern_lift_png"],
    )
    return outputs


def write_integrated_pattern_system_outputs(output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = build_pattern_system_experiment_tables()
    outputs = {
        "pattern_demand_supply_by_city": output_dir / "pattern_system_demand_supply_by_city.csv",
        "pattern_demand_supply_overall": output_dir / "pattern_system_demand_supply_overall.csv",
        "pattern_access_failures_by_city": output_dir / "pattern_system_access_failures_by_city.csv",
        "pattern_access_failures_overall": output_dir / "pattern_system_access_failures_overall.csv",
        "pattern_pt_routes_by_city": output_dir / "pattern_system_pt_routes_by_city.csv",
        "pattern_pt_routes_overall": output_dir / "pattern_system_pt_routes_overall.csv",
    }
    for key, path in outputs.items():
        tables[key].to_csv(path, index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for city-level output CSV/PNG files.",
    )
    parser.add_argument(
        "--pattern-system-only",
        action="store_true",
        help="Write integrated road/PT/demand/supply/coverage pattern-system tables without running target90 placement.",
    )
    parser.add_argument(
        "--tiered-target90",
        action="store_true",
        help="Run target90 placement only for an explicitly bounded city subset ordered from small to large.",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=None,
        help="Maximum number of smallest cities to include in --tiered-target90.",
    )
    parser.add_argument(
        "--cities",
        type=str,
        default=None,
        help="Comma-separated city ids to include in --tiered-target90 after size ordering.",
    )
    parser.add_argument(
        "--placement-root-name",
        type=str,
        default=PLACEMENT_TARGET090_ROOT,
        help="Per-city pipeline_2 placement output directory name.",
    )
    args = parser.parse_args()

    cities = [city.strip() for city in str(args.cities).split(",") if city.strip()] if args.cities else None

    if args.pattern_system_only:
        outputs = write_integrated_pattern_system_outputs(output_dir=args.output_dir)
    elif args.tiered_target90:
        outputs = write_tiered_target90_outputs(
            output_dir=args.output_dir,
            max_cities=args.max_cities,
            cities=cities,
            placement_root_name=args.placement_root_name,
        )
    else:
        outputs = write_city_level_outputs(output_dir=args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

"""Attach service-gap and access-component diagnostics to route strategy outputs.

This is a postprocess layer for the Brno route-generation experiment. It does
not rerun accessibility or placement; it reads the already produced solver block
outputs and the old home-to-service component diagnostics, then writes compact
tables and figures for comparing where unmet demand is access-solvable by PT
routes versus first/last-mile blocked.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString


DEFAULT_CITY = "brno_czechia"
DEFAULT_SERVICE = "polyclinic"
DEFAULT_JOINT_ROOT = Path("aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs")
DEFAULT_DIAGNOSTICS_ROOT = Path(
    "aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics"
)
DIAGNOSTICS_ROOTS = [
    DEFAULT_DIAGNOSTICS_ROOT,
    Path("aggregated_spatial_pipeline/outputs/experiments_new5_access_20260609/service_access_diagnostics"),
    Path("aggregated_spatial_pipeline/outputs/experiments_old23_access_20260609/service_access_diagnostics"),
    Path("aggregated_spatial_pipeline/outputs/experiments_old23_access_20260609_pilot/service_access_diagnostics"),
]
DEFAULT_PATTERN_ROOT = Path(
    "aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_accessibility_street_pattern"
)
DEFAULT_EXPERIMENT_ROOT = Path(
    "segregation-by-design-experiments/polyclinic_access_components/outputs/"
    "route_strategy_service_reduction_20260612"
)

OK_LABELS = {"ok_walk", "ok_pt_only"}
ROUTE_SOLVABLE_LABELS = {
    "failed_in_vehicle_gt_threshold",
    "failed_transfer_gt_threshold",
    "failed_no_pt_path",
    "failed_access_egress_sum_gt_threshold",
    "failed_total_gt_threshold_no_single_component_gt_threshold",
}
FIRST_LAST_BLOCKED_LABELS = {
    "failed_access_gt_threshold",
    "failed_egress_gt_threshold",
    "failed_multi_component_gt_threshold",
}


@dataclass(frozen=True)
class StrategyInput:
    name: str
    blocks_relpath: str


STRATEGIES = [
    StrategyInput("baseline_no_routes", "pipeline_2/placement_exact_target90_cap800/{service}/blocks_solver_after.parquet"),
    StrategyInput("placement_assignment", "pipeline_2/placement_exact_target90_cap800_after_routes_A/{service}/blocks_solver_after.parquet"),
    StrategyInput(
        "general_connectivity",
        "pipeline_2/placement_exact_target90_cap800_after_routes_general_connectivity/{service}/blocks_solver_after.parquet",
    ),
    StrategyInput(
        "existing_service",
        "pipeline_2/placement_exact_target90_cap800_after_routes_existing_service/{service}/blocks_solver_after.parquet",
    ),
    StrategyInput(
        "candidate_service",
        "pipeline_2/placement_exact_target90_cap800_after_routes_candidate_service/{service}/blocks_solver_after.parquet",
    ),
    StrategyInput(
        "candidate_or_existing_service",
        "pipeline_2/placement_exact_target90_cap800_after_routes_candidate_or_existing_service/{service}/blocks_solver_after.parquet",
    ),
]


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _classify_gap(access: pd.Series, capacity: pd.Series, eps: float = 1e-9) -> pd.Series:
    labels = np.full(len(access), "ok", dtype=object)
    access_gap = access.to_numpy() > eps
    capacity_gap = capacity.to_numpy() > eps
    labels[access_gap & ~capacity_gap] = "access_only"
    labels[~access_gap & capacity_gap] = "capacity_only"
    labels[access_gap & capacity_gap] = "mixed_access_capacity"
    return pd.Series(labels, index=access.index)


def _read_strategy_blocks(city_root: Path, service: str, strategy: StrategyInput) -> gpd.GeoDataFrame:
    path = city_root / strategy.blocks_relpath.format(service=service)
    if not path.exists():
        raise FileNotFoundError(path)
    blocks = gpd.read_parquet(path).copy()
    blocks["block_id"] = blocks.index.astype(str)
    blocks["strategy"] = strategy.name
    blocks["service_name"] = service
    return blocks


def _strategy_blocks_path(
    *,
    city_root: Path,
    experiment_city_dir: Path,
    service: str,
    strategy: StrategyInput,
) -> Path:
    summary_path = experiment_city_dir / "route_count_selection_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        sub = summary[summary["strategy"].astype(str).eq(strategy.name)].copy()
        if sub.empty:
            raise FileNotFoundError(f"Missing strategy={strategy.name} in {summary_path}")
        sub["requested_routes"] = pd.to_numeric(sub.get("requested_routes", 0), errors="coerce").fillna(0)
        sub["actual_routes"] = pd.to_numeric(sub.get("actual_routes", 0), errors="coerce").fillna(0)
        row = sub.sort_values(["requested_routes", "actual_routes"], ascending=[False, False]).iloc[0]
        candidate_dir = Path(str(row["candidate_dir"]))
        path = candidate_dir / "placement" / "blocks_solver_after.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    return city_root / strategy.blocks_relpath.format(service=service)


def _read_strategy_blocks_from_experiment(
    *,
    city_root: Path,
    experiment_city_dir: Path,
    service: str,
    strategy: StrategyInput,
) -> gpd.GeoDataFrame:
    path = _strategy_blocks_path(
        city_root=city_root,
        experiment_city_dir=experiment_city_dir,
        service=service,
        strategy=strategy,
    )
    if not path.exists():
        raise FileNotFoundError(path)
    blocks = gpd.read_parquet(path).copy()
    blocks["block_id"] = blocks.index.astype(str)
    blocks["strategy"] = strategy.name
    blocks["service_name"] = service
    return blocks


def _resolve_diagnostics_root(requested_root: Path, city: str) -> Path:
    requested_path = requested_root / city / "home_to_service_access_diagnostics.parquet"
    if requested_path.exists():
        return requested_root
    for root in DIAGNOSTICS_ROOTS:
        if (root / city / "home_to_service_access_diagnostics.parquet").exists():
            return root
    raise FileNotFoundError(requested_path)


def _resolve_city_root(requested_city_root: Path, experiment_city_dir: Path) -> Path:
    if requested_city_root.exists():
        return requested_city_root
    manifest_path = experiment_city_dir / "route_count_selection_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        best = payload.get("best") or payload.get("best_result") or {}
        summary_path = best.get("snapshot_connectpt_summary")
        if summary_path and not pd.isna(summary_path):
            summary_file = Path(str(summary_path))
            if summary_file.exists():
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                city_dir = summary.get("city_dir")
                if city_dir and Path(city_dir).exists():
                    return Path(city_dir)
    for summary_file in sorted(experiment_city_dir.glob("*/routes_*/connectpt_bus_summary.json")):
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        city_dir = summary.get("city_dir")
        if city_dir and Path(city_dir).exists():
            return Path(city_dir)
    return requested_city_root


def _prepare_gap_layer(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = blocks.copy()
    out["route_access_gap"] = _num(out, "demand_without")
    out["route_capacity_gap"] = _num(out, "demand_left")
    out["route_total_gap"] = out["route_access_gap"] + out["route_capacity_gap"]
    out["route_gap_type"] = _classify_gap(out["route_access_gap"], out["route_capacity_gap"])

    out["after_access_gap"] = _num(out, "demand_without_after", default=np.nan)
    out["after_capacity_gap"] = _num(out, "demand_left_after", default=np.nan)
    out["after_access_gap"] = out["after_access_gap"].fillna(out["route_access_gap"])
    out["after_capacity_gap"] = out["after_capacity_gap"].fillna(out["route_capacity_gap"])
    out["after_total_gap"] = out["after_access_gap"] + out["after_capacity_gap"]
    out["after_gap_type"] = _classify_gap(out["after_access_gap"], out["after_capacity_gap"])

    out["target_unmet_demand"] = _num(out, "target_unmet_demand")
    out["closed_by_placement"] = (out["target_unmet_demand"] - out["after_total_gap"]).clip(lower=0.0)
    out["new_service_block"] = _num(out, "optimized_capacity_added") > 1e-9
    out["existing_service_block"] = _num(out, "capacity") > 1e-9
    keep = [
        "strategy",
        "service_name",
        "block_id",
        "population",
        "demand",
        "capacity",
        "optimized_capacity_added",
        "placement_status",
        "target_unmet_demand",
        "provision",
        "provision_after",
        "route_access_gap",
        "route_capacity_gap",
        "route_total_gap",
        "route_gap_type",
        "after_access_gap",
        "after_capacity_gap",
        "after_total_gap",
        "after_gap_type",
        "closed_by_placement",
        "new_service_block",
        "existing_service_block",
        "geometry",
    ]
    keep.extend([c for c in out.columns if str(c).startswith("street_pattern_")])
    return out[[c for c in keep if c in out.columns]]


def _attach_street_patterns(gaps: gpd.GeoDataFrame, pattern_root: Path, city: str) -> gpd.GeoDataFrame:
    pattern_path = pattern_root / city / "prepared/blocks_with_street_pattern.parquet"
    if not pattern_path.exists():
        return gaps
    patterns = gpd.read_parquet(pattern_path)
    pattern_cols = [c for c in patterns.columns if str(c).startswith("street_pattern_")]
    if not pattern_cols:
        return gaps
    if patterns.crs != gaps.crs:
        patterns = patterns.to_crs(gaps.crs)
    left = gaps[["geometry"]].copy()
    left["join_id"] = np.arange(len(left))
    left = left.set_geometry(left.geometry.representative_point())
    joined = gpd.sjoin(left, patterns[pattern_cols + ["geometry"]], how="left", predicate="within")
    joined = joined.sort_values("join_id").drop_duplicates("join_id")
    enriched = gaps.copy()
    for col in pattern_cols:
        enriched[col] = joined.set_index("join_id")[col].reindex(np.arange(len(gaps))).to_numpy()
    return enriched


def _aggregate_component_diagnostics(
    *,
    diag_path: Path,
    city_root: Path,
    service: str,
    blocks: gpd.GeoDataFrame,
) -> pd.DataFrame:
    buildings_path = city_root / "blocksnet/buildings.parquet"
    if not diag_path.exists():
        raise FileNotFoundError(diag_path)
    if not buildings_path.exists():
        raise FileNotFoundError(buildings_path)

    diag = pd.read_parquet(diag_path)
    diag = diag[diag["service_name"].eq(service)].copy()
    buildings = gpd.read_parquet(buildings_path)
    geom = buildings.geometry.iloc[diag["building_idx"].astype(int).to_numpy()].reset_index(drop=True)
    points = gpd.GeoDataFrame(diag.reset_index(drop=True), geometry=geom.representative_point(), crs=buildings.crs)
    if blocks.crs != points.crs:
        points = points.to_crs(blocks.crs)

    block_index = blocks[["block_id", "geometry"]].copy()
    joined = gpd.sjoin(points, block_index, how="left", predicate="within")
    labels = joined["access_diagnosis_label"].astype(str)
    joined["failed_access"] = ~labels.isin(OK_LABELS)
    joined["route_solvable_failed"] = labels.isin(ROUTE_SOLVABLE_LABELS)
    joined["first_last_mile_blocked_failed"] = labels.isin(FIRST_LAST_BLOCKED_LABELS)
    joined["home_to_stop_blocked_failed"] = labels.eq("failed_access_gt_threshold")
    joined["stop_to_service_blocked_failed"] = labels.eq("failed_egress_gt_threshold")
    joined["pt_in_vehicle_failed"] = labels.isin(
        ["failed_in_vehicle_gt_threshold", "failed_transfer_gt_threshold", "failed_no_pt_path"]
    )

    agg = joined.groupby("block_id", dropna=False).agg(
        building_obs_count=("building_idx", "size"),
        failed_access_count=("failed_access", "sum"),
        route_solvable_failed_count=("route_solvable_failed", "sum"),
        first_last_mile_blocked_failed_count=("first_last_mile_blocked_failed", "sum"),
        home_to_stop_blocked_failed_count=("home_to_stop_blocked_failed", "sum"),
        stop_to_service_blocked_failed_count=("stop_to_service_blocked_failed", "sum"),
        pt_in_vehicle_failed_count=("pt_in_vehicle_failed", "sum"),
    )
    agg = agg.reset_index()
    denom = agg["failed_access_count"].replace(0, np.nan)
    agg["failed_access_share"] = agg["failed_access_count"] / agg["building_obs_count"].replace(0, np.nan)
    agg["route_solvable_share_of_failed"] = agg["route_solvable_failed_count"] / denom
    agg["first_last_mile_blocked_share_of_failed"] = agg["first_last_mile_blocked_failed_count"] / denom
    return agg.fillna(0.0)


def _read_component_diagnostics(city_root: Path, diagnostics_root: Path, city: str, service: str, blocks: gpd.GeoDataFrame) -> pd.DataFrame:
    return _aggregate_component_diagnostics(
        diag_path=diagnostics_root / city / "home_to_service_access_diagnostics.parquet",
        city_root=city_root,
        service=service,
        blocks=blocks,
    )


def _read_strategy_component_diagnostics(
    *,
    city_root: Path,
    diagnostics_root: Path,
    experiment_city_dir: Path,
    city: str,
    service: str,
    gaps: gpd.GeoDataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    recomputed_root = experiment_city_dir / "gap_diagnostics" / "recomputed_access_components"

    for strategy in [s.name for s in STRATEGIES]:
        strategy_blocks = gaps[gaps["strategy"].eq(strategy)][["block_id", "geometry"]].copy()
        if strategy == "baseline_no_routes":
            comp = _read_component_diagnostics(city_root, diagnostics_root, city, service, strategy_blocks)
            comp["component_source"] = "baseline_component_diagnostics"
        else:
            strategy_diag_path = recomputed_root / strategy / city / "home_to_service_access_diagnostics.parquet"
            if not strategy_diag_path.exists():
                continue
            comp = _aggregate_component_diagnostics(
                diag_path=strategy_diag_path,
                city_root=city_root,
                service=service,
                blocks=strategy_blocks,
            )
            comp["component_source"] = str(strategy_diag_path)
        comp["strategy"] = strategy
        rows.append(comp)
    return pd.concat(rows, ignore_index=True)


def _attach_residential_building_stats(gaps: gpd.GeoDataFrame, city_root: Path) -> gpd.GeoDataFrame:
    buildings_path = city_root / "blocksnet/buildings.parquet"
    if not buildings_path.exists():
        return gaps
    buildings = gpd.read_parquet(buildings_path)
    if "is_living" not in buildings.columns:
        return gaps
    living = buildings[buildings["is_living"].fillna(False).astype(bool)].copy()
    if living.empty:
        out = gaps.copy()
        out["residential_building_count"] = 0
        out["residential_population_sum"] = 0.0
        return out
    living["residential_population"] = pd.to_numeric(living.get("population", 0.0), errors="coerce").fillna(0.0)
    living = living.set_geometry(living.geometry.representative_point())
    if living.crs != gaps.crs:
        living = living.to_crs(gaps.crs)
    baseline_blocks = gaps[gaps["strategy"].eq("baseline_no_routes")][["block_id", "geometry"]].copy()
    joined = gpd.sjoin(
        living[["residential_population", "geometry"]],
        baseline_blocks,
        how="left",
        predicate="within",
    )
    stats = (
        joined.dropna(subset=["block_id"])
        .groupby("block_id", as_index=False)
        .agg(
            residential_building_count=("residential_population", "size"),
            residential_population_sum=("residential_population", "sum"),
        )
    )
    out = gaps.merge(stats, on="block_id", how="left")
    out["residential_building_count"] = pd.to_numeric(
        out["residential_building_count"], errors="coerce"
    ).fillna(0).astype(int)
    out["residential_population_sum"] = pd.to_numeric(
        out["residential_population_sum"], errors="coerce"
    ).fillna(0.0)
    return out


def _summarize(gaps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, sub in gaps.groupby("strategy", sort=False):
        row = {
            "strategy": strategy,
            "blocks": int(len(sub)),
            "new_service_blocks": int(sub["new_service_block"].sum()),
            "existing_service_blocks": int(sub["existing_service_block"].sum()),
            "route_access_gap": float(sub["route_access_gap"].sum()),
            "route_capacity_gap": float(sub["route_capacity_gap"].sum()),
            "route_total_gap": float(sub["route_total_gap"].sum()),
            "after_access_gap": float(sub["after_access_gap"].sum()),
            "after_capacity_gap": float(sub["after_capacity_gap"].sum()),
            "after_total_gap": float(sub["after_total_gap"].sum()),
            "closed_by_placement": float(sub["closed_by_placement"].sum()),
        }
        for stage_col in ["route_gap_type", "after_gap_type"]:
            counts = sub[stage_col].value_counts()
            for label in ["ok", "access_only", "capacity_only", "mixed_access_capacity"]:
                row[f"{stage_col}_{label}_blocks"] = int(counts.get(label, 0))
        rows.append(row)
    return pd.DataFrame(rows)


def _read_route_stage_summaries(experiment_city_dir: Path) -> pd.DataFrame:
    fallback_summary = pd.DataFrame()
    fallback_path = experiment_city_dir / "strategy_summary.csv"
    if fallback_path.exists():
        fallback_summary = pd.read_csv(fallback_path)
    rows: list[dict[str, object]] = [
        {
            "strategy": "baseline_no_routes",
            "route_stage_access_before": np.nan,
            "route_stage_capacity_before": np.nan,
            "route_stage_access_after": np.nan,
            "route_stage_capacity_after": np.nan,
        }
    ]
    for strategy in [s.name for s in STRATEGIES if s.name != "baseline_no_routes"]:
        path = experiment_city_dir / strategy / "polyclinic_summary_after_routes.json"
        if not path.exists():
            if "strategy" not in fallback_summary.columns:
                fallback = pd.DataFrame()
            else:
                fallback = fallback_summary[fallback_summary["strategy"].eq(strategy)]
            if not fallback.empty:
                row = fallback.iloc[0]
                before = float(row.get("access_before", np.nan))
                after = float(row.get("access_after", np.nan))
                rows.append(
                    {
                        "strategy": strategy,
                        "route_stage_access_before": before,
                        "route_stage_capacity_before": 0.0 if np.isfinite(before) else np.nan,
                        "route_stage_access_after": after,
                        "route_stage_capacity_after": 0.0 if np.isfinite(after) else np.nan,
                    }
                )
                continue
            rows.append(
                {
                    "strategy": strategy,
                    "route_stage_access_before": np.nan,
                    "route_stage_capacity_before": np.nan,
                    "route_stage_access_after": np.nan,
                    "route_stage_capacity_after": np.nan,
                }
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "strategy": strategy,
                "route_stage_access_before": float(payload.get("accessibility_gap_before", np.nan)),
                "route_stage_capacity_before": float(payload.get("capacity_gap_before", np.nan)),
                "route_stage_access_after": float(payload.get("accessibility_gap_after", np.nan)),
                "route_stage_capacity_after": float(payload.get("capacity_gap_after", np.nan)),
            }
        )
    out = pd.DataFrame(rows)
    out["route_stage_total_before"] = out["route_stage_access_before"] + out["route_stage_capacity_before"]
    out["route_stage_total_after"] = out["route_stage_access_after"] + out["route_stage_capacity_after"]
    out["route_stage_total_delta"] = out["route_stage_total_after"] - out["route_stage_total_before"]
    return out


def _render_summary_bars(summary: pd.DataFrame, out_path: Path) -> None:
    strategies = summary["strategy"].tolist()
    x = np.arange(len(strategies))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.5))
    route_after = summary.get("route_stage_total_after", pd.Series(np.nan, index=summary.index))
    ax.bar(x - width, route_after, width, label="After generated routes: total gap", color="#2f6f9f")
    ax.bar(x, summary["route_access_gap"], width, label="Placement input: access gap", color="#8ecae6")
    ax.bar(
        x,
        summary["route_capacity_gap"],
        width,
        bottom=summary["route_access_gap"],
        label="Placement input: capacity gap",
        color="#e9a178",
    )
    ax.bar(x + width, summary["after_access_gap"], width, label="After placement: access gap", color="#93c47d")
    ax.bar(
        x + width,
        summary["after_capacity_gap"],
        width,
        bottom=summary["after_access_gap"],
        label="After placement: capacity gap",
        color="#b6d7a8",
    )
    ax.set_title("Brno polyclinic unmet demand: generated route effect vs placement remainder")
    ax.set_ylabel("Demand units")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=25, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _render_route_solvable_scatter(gaps: pd.DataFrame, out_path: Path) -> None:
    base = gaps[gaps["strategy"].eq("baseline_no_routes")].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = np.clip(base["target_unmet_demand"].to_numpy(), 3, None) * 8
    colors = base["route_solvable_share_of_failed"].to_numpy()
    sc = ax.scatter(
        base["route_solvable_share_of_failed"],
        base["route_total_gap"],
        s=sizes,
        c=colors,
        cmap="viridis",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.45,
    )
    ax.set_title("Where route-solvable failures coincide with unmet demand")
    ax.set_xlabel("Route-solvable share among failed building observations")
    ax.set_ylabel("Baseline unmet demand in block")
    ax.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Route-solvable share")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _render_gap_maps(gaps: gpd.GeoDataFrame, out_path: Path) -> None:
    selected = [
        "baseline_no_routes",
        "general_connectivity",
        "existing_service",
        "candidate_service",
        "placement_assignment",
        "candidate_or_existing_service",
    ]
    plot_gdf = gaps[gaps["strategy"].isin(selected)].copy()
    vmax = float(plot_gdf["target_unmet_demand"].quantile(0.98)) or 1.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, strategy in zip(axes.ravel(), selected):
        sub = plot_gdf[plot_gdf["strategy"].eq(strategy)]
        sub.plot(ax=ax, color="#eeeeee", linewidth=0, edgecolor="none")
        sub[sub["target_unmet_demand"] > 0].plot(
            ax=ax,
            column="target_unmet_demand",
            cmap="magma",
            vmin=0,
            vmax=vmax,
            linewidth=0.05,
            edgecolor="white",
            legend=False,
        )
        new_sites = sub[sub["new_service_block"]]
        if len(new_sites):
            new_sites.geometry.representative_point().plot(ax=ax, color="#1b9e77", markersize=11, alpha=0.9)
        ax.set_title(strategy)
        ax.set_axis_off()
    fig.suptitle("Placement target unmet demand by block; green points are new selected services", y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


OLDSTYLE_GAP_COLORS = {
    "ok": "#2faf5f",
    "access_only": "#2f6de1",
    "capacity_only": "#d66a5e",
    "mixed_access_capacity": "#8b5fbf",
    "missing_data": "#b8bec7",
}
OLDSTYLE_GAP_LABELS = {
    "ok": "no unmet demand",
    "access_only": "accessibility gap (demand_without)",
    "capacity_only": "capacity gap (demand_left)",
    "mixed_access_capacity": "both gaps",
    "missing_data": "missing provision data",
}
OLDSTYLE_ORDER = ("ok", "capacity_only", "access_only", "mixed_access_capacity", "missing_data")


def _oldstyle_plot_strategy_map(
    ax: plt.Axes,
    sub: gpd.GeoDataFrame,
    gap_col: str,
    title: str,
    show_new_sites: bool = True,
) -> None:
    ax.set_facecolor("#6f6f6c")
    plot_sub = _filter_populated(sub)
    for status in OLDSTYLE_ORDER:
        part = plot_sub[plot_sub[gap_col].eq(status)]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            color=OLDSTYLE_GAP_COLORS[status],
            linewidth=0.08,
            edgecolor="#f8f5eb",
            alpha=0.9,
        )
    if show_new_sites:
        new_sites = plot_sub[plot_sub["new_service_block"]]
        if len(new_sites):
            new_sites.geometry.representative_point().plot(
                ax=ax,
                color="#ffd166",
                edgecolor="#1f2937",
                linewidth=0.45,
                markersize=13,
                zorder=5,
            )
    ax.set_title(title, fontsize=11, fontweight="bold", color="#f8fafc", pad=8)
    ax.set_axis_off()


def _render_oldstyle_contact_sheet(gaps: gpd.GeoDataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    selected = [
        "baseline_no_routes",
        "placement_assignment",
        "general_connectivity",
        "existing_service",
        "candidate_service",
        "candidate_or_existing_service",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#6f6f6c")
    summary_by_strategy = summary.set_index("strategy")
    for ax, strategy in zip(axes.ravel(), selected):
        sub = _filter_populated(gaps[gaps["strategy"].eq(strategy)])
        if strategy in summary_by_strategy.index:
            row = summary_by_strategy.loc[strategy]
            route_delta = row.get("route_stage_total_delta", np.nan)
            delta_text = "route delta=n/a" if pd.isna(route_delta) else f"route delta={route_delta:.0f}"
            title = (
                f"{strategy}\nnew={int(row['new_service_blocks'])}, "
                f"{delta_text}, after gap={row['after_total_gap']:.0f}"
            )
        else:
            title = strategy
        _oldstyle_plot_strategy_map(ax, sub, "route_gap_type", title)
    handles = [
        Patch(facecolor=OLDSTYLE_GAP_COLORS[key], edgecolor="none", label=OLDSTYLE_GAP_LABELS[key])
        for key in OLDSTYLE_ORDER
        if (gaps["route_gap_type"].eq(key).any() or key == "ok")
    ]
    handles.append(Patch(facecolor="#ffd166", edgecolor="#1f2937", label="new selected service"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="#f8f5eb",
        edgecolor="#d6d3d1",
        fontsize=10,
    )
    fig.suptitle(
        "Polyclinic: unmet-demand type before final placement",
        fontsize=18,
        fontweight="bold",
        color="white",
        y=0.985,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.95))
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_oldstyle_single_lp(
    gaps: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
    stage: str,
) -> list[Path]:
    if stage == "after":
        gap_col = "after_gap_type"
        provision_col = "provision_after"
        title_suffix = "after placement"
    else:
        gap_col = "route_gap_type"
        provision_col = "provision"
        title_suffix = "before final placement"

    out_paths: list[Path] = []
    summary_by_strategy = summary.set_index("strategy")
    for strategy, raw_sub in gaps.groupby("strategy", sort=False):
        sub = _filter_populated(raw_sub)
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 12),
            gridspec_kw={"height_ratios": [5.2, 1.35], "hspace": 0.14},
            facecolor="#6f6f6c",
        )
        ax, hist_ax = axes
        row = summary_by_strategy.loc[strategy] if strategy in summary_by_strategy.index else None
        counts = sub[gap_col].value_counts()
        title = (
            f"{strategy}: polyclinic unmet-demand type | "
            f"no_gap={int(counts.get('ok', 0))}, "
            f"cap_gap={int(counts.get('capacity_only', 0))}, "
            f"access_gap={int(counts.get('access_only', 0))}, "
            f"both={int(counts.get('mixed_access_capacity', 0))}"
        )
        if row is not None:
            title += f" | new={int(row['new_service_blocks'])}"
        _oldstyle_plot_strategy_map(ax, sub, gap_col, title, show_new_sites=True)
        handles = [
            Patch(facecolor=OLDSTYLE_GAP_COLORS[key], edgecolor="none", label=OLDSTYLE_GAP_LABELS[key])
            for key in OLDSTYLE_ORDER
            if sub[gap_col].eq(key).any()
        ]
        handles.append(Patch(facecolor="#ffd166", edgecolor="#1f2937", label="new selected service"))
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(handles)),
            frameon=True,
            facecolor="#f8f5eb",
            edgecolor="#d6d3d1",
            fontsize=9,
        )

        hist_ax.set_facecolor("#fff8e6")
        if provision_col in sub.columns:
            values = pd.to_numeric(sub[provision_col], errors="coerce").dropna().clip(0.0, 1.0)
        else:
            values = pd.Series(dtype=float)
        if len(values):
            bins = np.linspace(0.0, 1.0, 21)
            hist_ax.hist(values.to_numpy(dtype=float), bins=bins, color="#a8b8cf", edgecolor="#64748b", alpha=0.95)
            hist_ax.axvline(1.0, color="#16a34a", linestyle="--", linewidth=1.4, label="target=1.0")
            hist_ax.set_xlim(0.0, 1.0)
            hist_ax.set_xlabel("provision")
            hist_ax.set_ylabel("blocks")
            hist_ax.set_title(f"Provision histogram ({title_suffix})", fontsize=10)
            hist_ax.grid(alpha=0.2, axis="y")
            hist_ax.legend(loc="upper left", fontsize=8, frameon=True, facecolor="#fffdf8", edgecolor="#d6d3d1")
        else:
            hist_ax.text(0.5, 0.5, "no provision values", ha="center", va="center")
            hist_ax.set_axis_off()
        total_access = float(sub["after_access_gap" if stage == "after" else "route_access_gap"].sum())
        total_capacity = float(sub["after_capacity_gap" if stage == "after" else "route_capacity_gap"].sum())
        fig.text(
            0.5,
            0.02,
            f"{title_suffix}: accessibility={total_access:.0f}, capacity={total_capacity:.0f}",
            ha="center",
            va="center",
            color="#1f2937",
            fontsize=9,
        )
        fig.subplots_adjust(left=0.06, right=0.97, bottom=0.06, top=0.94, hspace=0.20)
        out_path = out_dir / f"oldstyle_lp_{strategy}_{stage}.png"
        fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def _filter_populated(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "residential_population_sum" in gdf.columns:
        return gdf[pd.to_numeric(gdf["residential_population_sum"], errors="coerce").fillna(0.0) > 0.0].copy()
    if "residential_building_count" in gdf.columns:
        return gdf[pd.to_numeric(gdf["residential_building_count"], errors="coerce").fillna(0.0) > 0.0].copy()
    if "population" in gdf.columns:
        return gdf[pd.to_numeric(gdf["population"], errors="coerce").fillna(0.0) > 0.0].copy()
    return gdf[pd.to_numeric(gdf.get("demand", 0.0), errors="coerce").fillna(0.0) > 0.0].copy()


def _build_residential_points_for_plot(city_root: Path, gaps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    buildings_path = city_root / "blocksnet/buildings.parquet"
    if not buildings_path.exists():
        return gpd.GeoDataFrame(columns=["block_id", "residential_population", "geometry"], geometry="geometry", crs=gaps.crs)
    buildings = gpd.read_parquet(buildings_path)
    if "is_living" not in buildings.columns:
        return gpd.GeoDataFrame(columns=["block_id", "residential_population", "geometry"], geometry="geometry", crs=gaps.crs)
    living = buildings[buildings["is_living"].fillna(False).astype(bool)].copy()
    living["residential_population"] = pd.to_numeric(living.get("population", 1.0), errors="coerce").fillna(1.0)
    living = living.set_geometry(living.geometry.representative_point())
    if living.crs != gaps.crs:
        living = living.to_crs(gaps.crs)
    baseline_blocks = gaps[gaps["strategy"].eq("baseline_no_routes")][["block_id", "geometry"]].copy()
    joined = gpd.sjoin(
        living[["residential_population", "geometry"]],
        baseline_blocks,
        how="inner",
        predicate="within",
    )
    return gpd.GeoDataFrame(
        joined[["block_id", "residential_population", "geometry"]].copy(),
        geometry="geometry",
        crs=gaps.crs,
    )


def _oldstyle_plot_residential_points(
    ax: plt.Axes,
    sub: gpd.GeoDataFrame,
    residential_points: gpd.GeoDataFrame,
    gap_col: str,
    title: str,
) -> None:
    ax.set_facecolor("#6f6f6c")
    bounds = sub.total_bounds
    if np.isfinite(bounds).all():
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    block_status = sub.set_index("block_id")[gap_col].to_dict()
    pts = residential_points.copy()
    pts["gap_type"] = pts["block_id"].map(block_status).fillna("missing_data")
    pts["size"] = np.clip(np.sqrt(pd.to_numeric(pts["residential_population"], errors="coerce").fillna(1.0)), 2.0, 18.0)
    for status in OLDSTYLE_ORDER:
        part = pts[pts["gap_type"].eq(status)]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            color=OLDSTYLE_GAP_COLORS[status],
            markersize=part["size"],
            alpha=0.75,
            linewidth=0,
            zorder=3,
        )
    new_sites = sub[sub["new_service_block"]]
    if len(new_sites):
        new_sites.geometry.representative_point().plot(
            ax=ax,
            color="#ffd166",
            edgecolor="#1f2937",
            linewidth=0.45,
            markersize=13,
            zorder=5,
        )
    ax.set_title(title, fontsize=10, fontweight="bold", color="#f8fafc", pad=6)
    ax.set_axis_off()


ACCESS_COMPONENT_COLORS = {
    "no_access_gap": "#cbd5e1",
    "route_solvable": "#00a6a6",
    "first_last_mile_blocked": "#f97316",
    "pt_in_vehicle_or_transfer": "#4338ca",
    "mixed_or_unknown": "#c084fc",
    "not_recomputed": "#111827",
}
ACCESS_COMPONENT_LABELS = {
    "no_access_gap": "no access gap",
    "route_solvable": "route-solvable access failure",
    "first_last_mile_blocked": "first/last-mile blocked",
    "pt_in_vehicle_or_transfer": "PT in-vehicle/transfer failure",
    "mixed_or_unknown": "overall access failure",
    "not_recomputed": "not recomputed",
}
ACCESS_COMPONENT_ORDER = (
    "no_access_gap",
    "route_solvable",
    "first_last_mile_blocked",
    "pt_in_vehicle_or_transfer",
    "mixed_or_unknown",
    "not_recomputed",
)

ACTION_TRIAGE_COLORS = {
    "ok": "#94a3b8",
    "route_may_help": "#0891b2",
    "walk_stop_access_first": "#facc15",
    "capacity_or_service_placement": "#ef4444",
    "mixed_intervention": "#a855f7",
    "not_recomputed": "#111827",
}
ACTION_TRIAGE_LABELS = {
    "ok": "already ok",
    "route_may_help": "route may help",
    "walk_stop_access_first": "walk/stop access first",
    "capacity_or_service_placement": "capacity/service placement",
    "mixed_intervention": "mixed intervention",
    "not_recomputed": "not recomputed",
}
ACTION_TRIAGE_ORDER = (
    "ok",
    "route_may_help",
    "walk_stop_access_first",
    "capacity_or_service_placement",
    "mixed_intervention",
    "not_recomputed",
)


def _derive_combined_access_capacity_triage(gaps: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gaps.copy()
    route_solvable = pd.to_numeric(out.get("route_solvable_failed_count", 0.0), errors="coerce").fillna(0.0)
    first_last = pd.to_numeric(out.get("first_last_mile_blocked_failed_count", 0.0), errors="coerce").fillna(0.0)
    pt_inside = pd.to_numeric(out.get("pt_in_vehicle_failed_count", 0.0), errors="coerce").fillna(0.0)
    has_access_gap = pd.to_numeric(out["route_access_gap"], errors="coerce").fillna(0.0) > 1e-9
    has_capacity_gap = pd.to_numeric(out["route_capacity_gap"], errors="coerce").fillna(0.0) > 1e-9

    access_component = np.full(len(out), "no_access_gap", dtype=object)
    access_component[has_access_gap.to_numpy()] = "mixed_or_unknown"
    route_mask = has_access_gap & (route_solvable >= first_last) & (route_solvable > 0)
    first_last_mask = has_access_gap & (first_last > route_solvable) & (first_last > 0)
    pt_mask = has_access_gap & (pt_inside > 0) & (pt_inside >= first_last) & (pt_inside >= route_solvable)
    access_component[route_mask.to_numpy()] = "route_solvable"
    access_component[first_last_mask.to_numpy()] = "first_last_mile_blocked"
    access_component[pt_mask.to_numpy()] = "pt_in_vehicle_or_transfer"
    not_recomputed = out.get("component_source", pd.Series("", index=out.index)).eq("not_recomputed")
    access_component[not_recomputed.to_numpy()] = "not_recomputed"
    out["access_component_label"] = access_component

    action = np.full(len(out), "ok", dtype=object)
    access_only = has_access_gap & ~has_capacity_gap
    capacity_only = has_capacity_gap & ~has_access_gap
    mixed = has_access_gap & has_capacity_gap
    action[capacity_only.to_numpy()] = "capacity_or_service_placement"
    action[mixed.to_numpy()] = "mixed_intervention"
    action[(access_only & out["access_component_label"].eq("route_solvable")).to_numpy()] = "route_may_help"
    action[(access_only & out["access_component_label"].eq("pt_in_vehicle_or_transfer")).to_numpy()] = "route_may_help"
    action[(access_only & out["access_component_label"].eq("first_last_mile_blocked")).to_numpy()] = "walk_stop_access_first"
    action[(access_only & out["access_component_label"].eq("mixed_or_unknown")).to_numpy()] = "mixed_intervention"
    action[not_recomputed.to_numpy()] = "not_recomputed"
    out["action_triage_label"] = action
    return out


def _oldstyle_plot_points_by_label(
    ax: plt.Axes,
    sub: gpd.GeoDataFrame,
    residential_points: gpd.GeoDataFrame,
    label_col: str,
    colors: dict[str, str],
    order: tuple[str, ...],
    title: str,
    show_new_sites: bool = True,
) -> None:
    ax.set_facecolor("#6f6f6c")
    bounds = sub.total_bounds
    if np.isfinite(bounds).all():
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
    block_status = sub.set_index("block_id")[label_col].to_dict()
    pts = residential_points.copy()
    pts["plot_label"] = pts["block_id"].map(block_status).fillna(order[-1])
    pts["size"] = np.clip(np.sqrt(pd.to_numeric(pts["residential_population"], errors="coerce").fillna(1.0)), 2.0, 18.0)
    for label in order:
        part = pts[pts["plot_label"].eq(label)]
        if part.empty:
            continue
        alpha = 0.38 if label in {"no_access_gap", "ok"} else 0.82
        part.plot(
            ax=ax,
            color=colors[label],
            markersize=part["size"],
            alpha=alpha,
            linewidth=0,
            zorder=3,
        )
    if show_new_sites:
        new_sites = sub[sub["new_service_block"]]
        if len(new_sites):
            new_sites.geometry.representative_point().plot(
                ax=ax,
                color="#ffd166",
                edgecolor="#1f2937",
                linewidth=0.45,
                markersize=13,
                zorder=5,
            )
    ax.set_title(title, fontsize=9, fontweight="bold", color="#f8fafc", pad=5)
    ax.set_axis_off()


def _component_source_note(sub: gpd.GeoDataFrame) -> str:
    if "component_source" not in sub.columns:
        return "components: baseline"
    sources = {str(v) for v in sub["component_source"].dropna().unique()}
    if "not_recomputed" in sources:
        return "components: not recomputed"
    real_sources = {s for s in sources if s not in {"baseline_component_diagnostics", "no_component_observations"}}
    if real_sources:
        return "components: recomputed"
    return "components: baseline"


def _render_combined_service_access_triage(
    gaps: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    city_root: Path,
    out_path: Path,
) -> None:
    residential_points = _build_residential_points_for_plot(city_root, gaps)
    selected = [
        "baseline_no_routes",
        "placement_assignment",
        "general_connectivity",
        "existing_service",
        "candidate_service",
        "candidate_or_existing_service",
    ]
    fig, axes = plt.subplots(len(selected), 3, figsize=(13.5, 20), facecolor="#6f6f6c")
    summary_by_strategy = summary.set_index("strategy")
    for row_idx, strategy in enumerate(selected):
        sub = _filter_populated(gaps[gaps["strategy"].eq(strategy)])
        row = summary_by_strategy.loc[strategy] if strategy in summary_by_strategy.index else None
        new_count = int(row["new_service_blocks"]) if row is not None else int(sub["new_service_block"].sum())
        route_delta = row.get("route_stage_total_delta", np.nan) if row is not None else np.nan
        delta_text = "route Δ=n/a" if pd.isna(route_delta) else f"route Δ={route_delta:.0f}"
        prefix = f"{strategy}\nnew={new_count}, {delta_text}"
        component_note = _component_source_note(sub)
        _oldstyle_plot_points_by_label(
            axes[row_idx, 0],
            sub,
            residential_points,
            "route_gap_type",
            OLDSTYLE_GAP_COLORS,
            OLDSTYLE_ORDER,
            f"{prefix}\n1 service gap type",
        )
        _oldstyle_plot_points_by_label(
            axes[row_idx, 1],
            sub,
            residential_points,
            "access_component_label",
            ACCESS_COMPONENT_COLORS,
            ACCESS_COMPONENT_ORDER,
            f"2 access failure component\n{component_note}",
            show_new_sites=False,
        )
        _oldstyle_plot_points_by_label(
            axes[row_idx, 2],
            sub,
            residential_points,
            "action_triage_label",
            ACTION_TRIAGE_COLORS,
            ACTION_TRIAGE_ORDER,
            f"3 action triage\n{component_note}",
        )

    gap_handles = [
        Patch(facecolor=OLDSTYLE_GAP_COLORS[key], edgecolor="none", label=OLDSTYLE_GAP_LABELS[key])
        for key in ("ok", "access_only", "capacity_only", "mixed_access_capacity")
        if gaps["route_gap_type"].eq(key).any() or key == "ok"
    ]
    component_handles = [
        Patch(facecolor=ACCESS_COMPONENT_COLORS[key], edgecolor="none", label=ACCESS_COMPONENT_LABELS[key])
        for key in ACCESS_COMPONENT_ORDER
        if gaps["access_component_label"].eq(key).any() or key == "no_access_gap"
    ]
    action_handles = [
        Patch(facecolor=ACTION_TRIAGE_COLORS[key], edgecolor="none", label=ACTION_TRIAGE_LABELS[key])
        for key in ACTION_TRIAGE_ORDER
        if gaps["action_triage_label"].eq(key).any() or key == "ok"
    ]
    legend_style = {
        "frameon": True,
        "facecolor": "#f8f5eb",
        "edgecolor": "#d6d3d1",
        "fontsize": 7.5,
        "title_fontsize": 8.5,
    }
    leg1 = fig.legend(
        handles=gap_handles + [Patch(facecolor="#ffd166", edgecolor="#1f2937", label="new selected service")],
        loc="lower center",
        bbox_to_anchor=(0.205, 0.028),
        ncol=1,
        title="1 service gap",
        **legend_style,
    )
    fig.add_artist(leg1)
    leg2 = fig.legend(
        handles=component_handles,
        loc="lower center",
        bbox_to_anchor=(0.515, 0.028),
        ncol=1,
        title="2 access component",
        **legend_style,
    )
    fig.add_artist(leg2)
    fig.legend(
        handles=action_handles + [Patch(facecolor="#ffd166", edgecolor="#1f2937", label="new selected service")],
        loc="lower center",
        bbox_to_anchor=(0.82, 0.028),
        ncol=1,
        title="3 action triage",
        **legend_style,
    )
    fig.suptitle(
        "Polyclinic combined diagnostics: service gap, access component, action triage",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.988,
    )
    fig.tight_layout(rect=(0.02, 0.13, 0.98, 0.965))
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _full_solution_after_layer(
    gaps: gpd.GeoDataFrame,
    city_root: Path,
    experiment_city_dir: Path,
    service: str,
    strategy: str = "candidate_service",
) -> gpd.GeoDataFrame:
    strategy_input = next((item for item in STRATEGIES if item.name == strategy), None)
    if strategy_input is None:
        raise ValueError(f"Unknown strategy: {strategy}")
    path = _strategy_blocks_path(
        city_root=city_root,
        experiment_city_dir=experiment_city_dir,
        service=service,
        strategy=strategy_input,
    )
    if not path.exists():
        raise FileNotFoundError(path)
    blocks = gpd.read_parquet(path).copy()
    blocks["block_id"] = blocks.index.astype(str)
    blocks["strategy"] = f"{strategy}_full_solution_after"
    blocks["service_name"] = service
    after = _prepare_gap_layer(blocks)
    after["route_access_gap"] = after["after_access_gap"]
    after["route_capacity_gap"] = after["after_capacity_gap"]
    after["route_total_gap"] = after["after_total_gap"]
    after["route_gap_type"] = after["after_gap_type"]

    baseline_cols = [
        "block_id",
        "residential_building_count",
        "residential_population_sum",
    ]
    baseline_stats = gaps[gaps["strategy"].eq("baseline_no_routes")][
        [c for c in baseline_cols if c in gaps.columns]
    ].drop_duplicates("block_id")
    if not baseline_stats.empty:
        after = after.merge(baseline_stats, on="block_id", how="left")

    component_cols = [
        "block_id",
        "building_obs_count",
        "failed_access_count",
        "route_solvable_failed_count",
        "first_last_mile_blocked_failed_count",
        "home_to_stop_blocked_failed_count",
        "stop_to_service_blocked_failed_count",
        "pt_in_vehicle_failed_count",
        "failed_access_share",
        "route_solvable_share_of_failed",
        "first_last_mile_blocked_share_of_failed",
        "component_source",
    ]
    components = gaps[gaps["strategy"].eq(strategy)][[c for c in component_cols if c in gaps.columns]].drop_duplicates(
        "block_id"
    )
    if not components.empty:
        after = after.merge(components, on="block_id", how="left")
    for col in [c for c in component_cols if c not in {"block_id", "component_source"}]:
        if col in after.columns:
            after[col] = pd.to_numeric(after[col], errors="coerce").fillna(0.0)
    if "component_source" in after.columns:
        after["component_source"] = after["component_source"].fillna("no_component_observations")
    after = _derive_combined_access_capacity_triage(after)
    return gpd.GeoDataFrame(after, geometry="geometry", crs=blocks.crs)


def _render_before_after_access_components(
    gaps: gpd.GeoDataFrame,
    city_root: Path,
    experiment_city_dir: Path,
    service: str,
    out_path: Path,
) -> None:
    residential_points = _build_residential_points_for_plot(city_root, gaps)
    before = _filter_populated(gaps[gaps["strategy"].eq("baseline_no_routes")])
    after = _filter_populated(
        _full_solution_after_layer(
            gaps,
            city_root,
            experiment_city_dir,
            service,
            strategy="placement_assignment",
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.2), facecolor="#6f6f6c")
    _oldstyle_plot_points_by_label(
        axes[0, 0],
        before,
        residential_points,
        "route_gap_type",
        OLDSTYLE_GAP_COLORS,
        OLDSTYLE_ORDER,
        "1 service gap type\nBEFORE",
        show_new_sites=False,
    )
    _oldstyle_plot_points_by_label(
        axes[0, 1],
        before,
        residential_points,
        "access_component_label",
        ACCESS_COMPONENT_COLORS,
        ACCESS_COMPONENT_ORDER,
        "2 access failure component\nBEFORE",
        show_new_sites=False,
    )
    _oldstyle_plot_points_by_label(
        axes[1, 0],
        after,
        residential_points,
        "route_gap_type",
        OLDSTYLE_GAP_COLORS,
        OLDSTYLE_ORDER,
        "1 service gap type\nAFTER",
        show_new_sites=False,
    )
    _oldstyle_plot_points_by_label(
        axes[1, 1],
        after,
        residential_points,
        "access_component_label",
        ACCESS_COMPONENT_COLORS,
        ACCESS_COMPONENT_ORDER,
        "2 access failure component\nAFTER",
        show_new_sites=False,
    )

    legend_style = {
        "frameon": True,
        "facecolor": "#f8f5eb",
        "edgecolor": "#d6d3d1",
        "fontsize": 7.5,
        "title_fontsize": 8.5,
    }
    gap_handles = [
        Patch(facecolor=OLDSTYLE_GAP_COLORS[key], edgecolor="none", label=OLDSTYLE_GAP_LABELS[key])
        for key in ("ok", "access_only", "capacity_only", "mixed_access_capacity")
        if before["route_gap_type"].eq(key).any() or after["route_gap_type"].eq(key).any() or key == "ok"
    ]
    component_handles = [
        Patch(facecolor=ACCESS_COMPONENT_COLORS[key], edgecolor="none", label=ACCESS_COMPONENT_LABELS[key])
        for key in ACCESS_COMPONENT_ORDER
        if before["access_component_label"].eq(key).any() or after["access_component_label"].eq(key).any() or key == "no_access_gap"
    ]
    leg1 = fig.legend(
        handles=gap_handles,
        loc="lower center",
        bbox_to_anchor=(0.28, 0.02),
        ncol=1,
        title="1 service gap",
        **legend_style,
    )
    fig.add_artist(leg1)
    leg2 = fig.legend(
        handles=component_handles,
        loc="lower center",
        bbox_to_anchor=(0.72, 0.02),
        ncol=1,
        title="2 access component",
        **legend_style,
    )
    fig.add_artist(leg2)
    fig.suptitle(
        "Access diagnostics: before and after",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=0.975,
    )
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.94))
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _read_selection_summary(experiment_city_dir: Path) -> pd.DataFrame:
    path = experiment_city_dir / "route_count_selection_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    summary = pd.read_csv(path)
    if "candidate_dir" in summary.columns:
        summary["candidate_dir"] = summary["candidate_dir"].astype(str)
    for col in ["requested_routes", "actual_routes", "new_count", "selected_count"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return summary


def _read_best_selection(experiment_city_dir: Path, selection_summary: pd.DataFrame) -> dict[str, object]:
    manifest_path = experiment_city_dir / "route_count_selection_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        best = payload.get("best") or payload.get("best_result")
        if isinstance(best, dict):
            return best
    if selection_summary.empty:
        return {"strategy": "placement_assignment", "candidate_dir": str(experiment_city_dir)}
    work = selection_summary.copy()
    for col in ["new_count", "actual_routes", "requested_routes"]:
        if col not in work.columns:
            work[col] = np.nan
    work["_new_sort"] = work["new_count"].fillna(np.inf)
    work["_actual_sort"] = work["actual_routes"].fillna(np.inf)
    work["_requested_sort"] = work["requested_routes"].fillna(np.inf)
    return work.sort_values(["_new_sort", "_actual_sort", "_requested_sort", "strategy"]).iloc[0].to_dict()


def _candidate_dir_for_strategy(experiment_city_dir: Path, selection_summary: pd.DataFrame, strategy: str) -> Path | None:
    if selection_summary.empty or "strategy" not in selection_summary.columns:
        return None
    sub = selection_summary[selection_summary["strategy"].astype(str).eq(strategy)].copy()
    if sub.empty:
        return None
    for col in ["requested_routes", "actual_routes"]:
        if col not in sub.columns:
            sub[col] = 0
    row = sub.sort_values(["requested_routes", "actual_routes"], ascending=[False, False]).iloc[0]
    candidate_dir = row.get("candidate_dir")
    if pd.isna(candidate_dir):
        return None
    return Path(str(candidate_dir))


def _read_blocks_from_candidate(candidate_dir: Path | None) -> gpd.GeoDataFrame | None:
    if candidate_dir is None:
        return None
    path = candidate_dir / "placement" / "blocks_solver_after.parquet"
    if not path.exists():
        return None
    blocks = gpd.read_parquet(path).copy()
    blocks["block_id"] = blocks.index.astype(str)
    return blocks


def _candidate_route_snapshot_dir(candidate_dir: Path | None) -> Path | None:
    if candidate_dir is None:
        return None
    path = candidate_dir / "snapshots" / "intermodal_replaced"
    if path.exists():
        return path
    path = candidate_dir / "gap_diagnostics" / "recomputed_access_components" / "snapshots" / "intermodal_replaced"
    if path.exists():
        return path
    return None


def _read_generated_route_lines(candidate_dir: Path | None, crs) -> gpd.GeoDataFrame:
    snapshot = _candidate_route_snapshot_dir(candidate_dir)
    if snapshot is None:
        return gpd.GeoDataFrame(columns=["route_name", "geometry"], geometry="geometry", crs=crs)
    gen_path = snapshot / "bus_generated_route_edges.parquet"
    nodes_path = snapshot / "graph_nodes.parquet"
    if not gen_path.exists() or not nodes_path.exists():
        return gpd.GeoDataFrame(columns=["route_name", "geometry"], geometry="geometry", crs=crs)
    gen = pd.read_parquet(gen_path)
    nodes = gpd.read_parquet(nodes_path)
    if gen.empty or nodes.empty:
        return gpd.GeoDataFrame(columns=["route_name", "geometry"], geometry="geometry", crs=nodes.crs if hasattr(nodes, "crs") else crs)
    node_key = "index" if "index" in nodes.columns else nodes.index.name
    if node_key is None:
        nodes = nodes.reset_index().rename(columns={"index": "_node_index"})
        node_key = "_node_index"
    node_geom = nodes.set_index(node_key).geometry.to_dict()
    rows: list[dict[str, object]] = []
    for row in gen.itertuples(index=False):
        u = getattr(row, "intermodal_u")
        v = getattr(row, "intermodal_v")
        if u not in node_geom or v not in node_geom:
            continue
        a = node_geom[u]
        b = node_geom[v]
        if a is None or b is None or a.is_empty or b.is_empty:
            continue
        rows.append(
            {
                "route_name": str(getattr(row, "route_name", "generated_bus")),
                "geometry": LineString([a, b]),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=nodes.crs or crs)


def _read_existing_route_lines(candidate_dir: Path | None, crs) -> gpd.GeoDataFrame:
    snapshot = _candidate_route_snapshot_dir(candidate_dir)
    if snapshot is None:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    path = snapshot / "graph_edges_source.parquet"
    if not path.exists():
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=crs)
    edges = gpd.read_parquet(path)
    if "type" not in edges.columns:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=edges.crs or crs)
    route_types = {"bus", "tram", "subway"}
    out = edges[edges["type"].astype(str).isin(route_types)].copy()
    if "route" in out.columns:
        out = out[out["route"].notna()].copy()
    return gpd.GeoDataFrame(out, geometry="geometry", crs=edges.crs or crs)


def _target_unmet_column(blocks: gpd.GeoDataFrame) -> pd.Series:
    if "target_unmet_demand" in blocks.columns:
        return pd.to_numeric(blocks["target_unmet_demand"], errors="coerce").fillna(0.0)
    return pd.to_numeric(blocks.get("demand_without", 0.0), errors="coerce").fillna(0.0)


def _plot_route_strategy_map(
    ax: plt.Axes,
    blocks: gpd.GeoDataFrame,
    candidate_dir: Path | None,
    title: str,
    *,
    highlight: bool = False,
    show_existing_routes: bool = False,
    show_services: bool = True,
    show_generated_routes: bool = True,
    detail_roads: bool = False,
    vmax: float | None = None,
) -> None:
    ax.set_facecolor("#f8f5eb")
    plot_blocks = _filter_populated(blocks)
    if plot_blocks.empty:
        plot_blocks = blocks.copy()
    target = _target_unmet_column(plot_blocks)
    plot_blocks = plot_blocks.assign(_target_unmet=target)
    base = plot_blocks[plot_blocks["_target_unmet"].le(0)]
    if len(base):
        base.plot(ax=ax, color="#f5f1e8", edgecolor="#d7d0c2", linewidth=0.15, zorder=1)
    unmet = plot_blocks[plot_blocks["_target_unmet"].gt(0)]
    if len(unmet):
        unmet.plot(
            ax=ax,
            column="_target_unmet",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax or float(unmet["_target_unmet"].quantile(0.98)) or 1.0,
            linewidth=0.12,
            edgecolor="#ffffff",
            alpha=0.86,
            zorder=2,
        )
    if detail_roads:
        existing = _read_existing_route_lines(candidate_dir, blocks.crs)
        if not existing.empty:
            existing.plot(ax=ax, color="#9aa7b8", linewidth=0.55, alpha=0.45, zorder=3)
    elif show_existing_routes:
        existing = _read_existing_route_lines(candidate_dir, blocks.crs)
        if not existing.empty:
            existing.plot(ax=ax, color="#7d99bf", linewidth=0.7, alpha=0.38, zorder=3)

    if show_generated_routes:
        gen = _read_generated_route_lines(candidate_dir, blocks.crs)
        if not gen.empty:
            colors = ["#2f80ed", "#f97316", "#16a34a", "#7c3aed", "#db2777"]
            for idx, (route_name, route) in enumerate(gen.groupby("route_name", sort=True)):
                color = colors[idx % len(colors)]
                route.plot(ax=ax, color="white", linewidth=4.2, alpha=0.9, zorder=7)
                route.plot(ax=ax, color=color, linewidth=2.4, alpha=0.72, zorder=8)

    if show_services:
        existing_services = plot_blocks[pd.to_numeric(plot_blocks.get("capacity", 0.0), errors="coerce").fillna(0.0) > 0]
        new_services = plot_blocks[pd.to_numeric(plot_blocks.get("optimized_capacity_added", 0.0), errors="coerce").fillna(0.0) > 0]
        if len(existing_services):
            existing_services.geometry.representative_point().plot(
                ax=ax,
                color="#111827",
                marker="s",
                markersize=12,
                alpha=0.82,
                zorder=9,
            )
        if len(new_services):
            new_services.geometry.representative_point().plot(
                ax=ax,
                color="#10b981",
                marker="*",
                edgecolor="white",
                linewidth=0.35,
                markersize=32,
                alpha=0.9,
                zorder=10,
            )
    if highlight:
        circle = plt.Circle(
            (0.5, 0.5),
            0.485,
            transform=ax.transAxes,
            fill=False,
            color="#19b36a",
            linewidth=2.3,
            zorder=20,
            clip_on=False,
        )
        ax.add_patch(circle)
    bounds = plot_blocks.total_bounds
    if np.isfinite(bounds).all():
        pad_x = max((bounds[2] - bounds[0]) * 0.04, 1.0)
        pad_y = max((bounds[3] - bounds[1]) * 0.04, 1.0)
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_title(title, fontsize=8.5, fontweight="bold", color="#263238", pad=5)
    ax.set_axis_off()


def _render_strategy_overview_maps(
    gaps: gpd.GeoDataFrame,
    experiment_city_dir: Path,
    selection_summary: pd.DataFrame,
    out_path: Path,
    city_label: str,
    service: str,
) -> None:
    selected = [
        "placement_assignment",
        "general_connectivity",
        "existing_service",
        "candidate_service",
        "candidate_or_existing_service",
    ]
    rows: list[dict[str, object]] = []
    for strategy in selected:
        candidate_dir = _candidate_dir_for_strategy(experiment_city_dir, selection_summary, strategy)
        blocks = _read_blocks_from_candidate(candidate_dir)
        if blocks is None:
            sub = gaps[gaps["strategy"].eq(strategy)].copy()
            blocks = sub if not sub.empty else gaps[gaps["strategy"].eq("baseline_no_routes")].copy()
        summary_row = selection_summary[selection_summary["strategy"].astype(str).eq(strategy)]
        new_count = int(pd.to_numeric(blocks.get("optimized_capacity_added", 0.0), errors="coerce").fillna(0.0).gt(0).sum())
        actual_routes = np.nan
        if not summary_row.empty:
            row = summary_row.iloc[0]
            new_count = int(row["new_count"]) if pd.notna(row.get("new_count")) else new_count
            actual_routes = row.get("actual_routes", np.nan)
        rows.append({"strategy": strategy, "candidate_dir": candidate_dir, "blocks": blocks, "new_count": new_count, "actual_routes": actual_routes})
    min_new = min(row["new_count"] for row in rows) if rows else None
    vmax_parts = []
    for row in rows:
        vals = _target_unmet_column(row["blocks"])
        if len(vals):
            vmax_parts.append(float(vals.quantile(0.98)))
    vmax = max(vmax_parts) if vmax_parts else 1.0
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), facecolor="#f7f3ea")
    axes_flat = axes.ravel()
    for ax, row in zip(axes_flat, rows):
        routes_text = "routes=n/a" if pd.isna(row["actual_routes"]) else f"routes={int(row['actual_routes'])}"
        title = f"{row['strategy']}\nnew sites: {row['new_count']} | {routes_text}"
        _plot_route_strategy_map(
            ax,
            row["blocks"],
            row["candidate_dir"],
            title,
            highlight=(min_new is not None and row["new_count"] == min_new),
            show_existing_routes=False,
            show_services=True,
            show_generated_routes=True,
            vmax=vmax,
        )
    axes_flat[-1].set_axis_off()
    handles = [
        Line2D([0], [0], color="#2f80ed", lw=2.4, label="generated route 1"),
        Line2D([0], [0], color="#f97316", lw=2.4, label="generated route 2"),
        Line2D([0], [0], color="#16a34a", lw=2.4, label="generated route 3"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#111827", markersize=6, label="existing service block"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#10b981", markeredgecolor="white", markersize=10, label="proposed new service block"),
        Line2D([0], [0], color="#19b36a", lw=2.4, label="minimum new-sites option"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(
        f"{city_label}: {service} route strategies and minimum-service options",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.94))
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _render_best_option_routes_services(
    gaps: gpd.GeoDataFrame,
    experiment_city_dir: Path,
    selection_summary: pd.DataFrame,
    out_path: Path,
    city_label: str,
    service: str,
) -> None:
    best = _read_best_selection(experiment_city_dir, selection_summary)
    strategy = str(best.get("strategy", "baseline_no_routes"))
    candidate_dir_value = best.get("candidate_dir")
    candidate_dir = Path(str(candidate_dir_value)) if candidate_dir_value is not None and not pd.isna(candidate_dir_value) else None
    blocks = _read_blocks_from_candidate(candidate_dir)
    if blocks is None:
        blocks = gaps[gaps["strategy"].eq(strategy)].copy()
    if blocks.empty:
        blocks = gaps[gaps["strategy"].eq("baseline_no_routes")].copy()
    new_count = int(best.get("new_count", pd.to_numeric(blocks.get("optimized_capacity_added", 0.0), errors="coerce").fillna(0.0).gt(0).sum()))
    actual_routes = best.get("actual_routes", np.nan)
    routes_text = "routes=n/a" if pd.isna(actual_routes) else f"{int(actual_routes)} generated routes"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), facecolor="#f7f3ea")
    _plot_route_strategy_map(
        axes[0],
        blocks,
        candidate_dir,
        f"Best option: routes\n{strategy} | {routes_text} | new sites: {new_count}",
        highlight=True,
        show_existing_routes=True,
        show_services=False,
        show_generated_routes=True,
        detail_roads=False,
    )
    _plot_route_strategy_map(
        axes[1],
        blocks,
        candidate_dir,
        f"Best option: services\nexisting services + proposed new sites: {new_count}",
        highlight=True,
        show_existing_routes=False,
        show_services=True,
        show_generated_routes=False,
    )
    handles = [
        Line2D([0], [0], color="#7d99bf", lw=1.2, alpha=0.7, label="existing PT route edges"),
        Line2D([0], [0], color="#2f80ed", lw=2.4, label="generated route 1"),
        Line2D([0], [0], color="#f97316", lw=2.4, label="generated route 2"),
        Line2D([0], [0], color="#16a34a", lw=2.4, label="generated route 3"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#111827", markersize=6, label="existing service block"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#10b981", markeredgecolor="white", markersize=10, label="proposed new service block"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{city_label}: best route-improvement option with routes and services", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0.02, 0.09, 0.98, 0.91))
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _merge_vertical_pngs(title: str, inputs: list[Path], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    images = [Image.open(path).convert("RGB") for path in inputs if path.exists()]
    if not images:
        raise FileNotFoundError("No rendered panels to merge")
    width = max(image.width for image in images)
    title_h = 92
    gap = 18
    margin = 22
    height = title_h + margin + sum(image.height for image in images) + gap * (len(images) - 1) + margin
    canvas = Image.new("RGB", (width + 2 * margin, height), "#f7f3ea")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial Bold.ttf", 34)
        sub_font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    draw.text((margin, 22), title, fill="#1f2937", font=font)
    draw.text((margin, 62), "Diagnostics + route-strategy overview + best option routes/services", fill="#4b5563", font=sub_font)
    y = title_h + margin
    for image in images:
        x = margin + (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _render_full_final_canvas(
    gaps: gpd.GeoDataFrame,
    city_root: Path,
    experiment_city_dir: Path,
    service: str,
    city: str,
    out_path: Path,
) -> None:
    selection_summary = _read_selection_summary(experiment_city_dir)
    tmp_dir = out_path.parent / f".{out_path.stem}_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = tmp_dir / "01_access_diagnostics.png"
    overview = tmp_dir / "02_strategy_overview.png"
    best = tmp_dir / "03_best_routes_services.png"
    _render_before_after_access_components(gaps, city_root, experiment_city_dir, service, diagnostics)
    _render_strategy_overview_maps(gaps, experiment_city_dir, selection_summary, overview, city, service)
    _render_best_option_routes_services(gaps, experiment_city_dir, selection_summary, best, city, service)
    _merge_vertical_pngs(f"{city}: {service} full route-strategy canvas", [diagnostics, overview, best], out_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _render_oldstyle_before_after_contact_sheet(
    gaps: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    selected = [
        "baseline_no_routes",
        "placement_assignment",
        "general_connectivity",
        "existing_service",
        "candidate_service",
        "candidate_or_existing_service",
    ]
    fig, axes = plt.subplots(len(selected), 2, figsize=(10, 20), facecolor="#6f6f6c")
    summary_by_strategy = summary.set_index("strategy")
    for row_idx, strategy in enumerate(selected):
        sub = _filter_populated(gaps[gaps["strategy"].eq(strategy)])
        row = summary_by_strategy.loc[strategy] if strategy in summary_by_strategy.index else None
        new_count = int(row["new_service_blocks"]) if row is not None else int(sub["new_service_block"].sum())
        route_delta = row.get("route_stage_total_delta", np.nan) if row is not None else np.nan
        delta_text = "route Δ=n/a" if pd.isna(route_delta) else f"route Δ={route_delta:.0f}"
        before_counts = sub["route_gap_type"].value_counts()
        after_counts = sub["after_gap_type"].value_counts()
        _oldstyle_plot_strategy_map(
            axes[row_idx, 0],
            sub,
            "route_gap_type",
            (
                f"{strategy}\nBEFORE placement | new={new_count}, {delta_text}\n"
                f"no={int(before_counts.get('ok', 0))}, access={int(before_counts.get('access_only', 0))}"
            ),
        )
        _oldstyle_plot_strategy_map(
            axes[row_idx, 1],
            sub,
            "after_gap_type",
            (
                f"{strategy}\nAFTER placement | new={new_count}\n"
                f"no={int(after_counts.get('ok', 0))}, access={int(after_counts.get('access_only', 0))}"
            ),
        )
    handles = [
        Patch(facecolor=OLDSTYLE_GAP_COLORS[key], edgecolor="none", label=OLDSTYLE_GAP_LABELS[key])
        for key in OLDSTYLE_ORDER
        if (gaps["route_gap_type"].eq(key).any() or gaps["after_gap_type"].eq(key).any() or key == "ok")
    ]
    handles.append(Patch(facecolor="#ffd166", edgecolor="#1f2937", label="new selected service"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="#f8f5eb",
        edgecolor="#d6d3d1",
        fontsize=9,
    )
    fig.suptitle(
        "Polyclinic: unmet-demand type before / after placement (populated quarters only)",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.988,
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.965))
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _render_oldstyle_residential_points_before_after(
    gaps: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    city_root: Path,
    out_path: Path,
) -> None:
    residential_points = _build_residential_points_for_plot(city_root, gaps)
    selected = [
        "baseline_no_routes",
        "placement_assignment",
        "general_connectivity",
        "existing_service",
        "candidate_service",
        "candidate_or_existing_service",
    ]
    fig, axes = plt.subplots(len(selected), 2, figsize=(10, 20), facecolor="#6f6f6c")
    summary_by_strategy = summary.set_index("strategy")
    for row_idx, strategy in enumerate(selected):
        sub = _filter_populated(gaps[gaps["strategy"].eq(strategy)])
        row = summary_by_strategy.loc[strategy] if strategy in summary_by_strategy.index else None
        new_count = int(row["new_service_blocks"]) if row is not None else int(sub["new_service_block"].sum())
        route_delta = row.get("route_stage_total_delta", np.nan) if row is not None else np.nan
        delta_text = "route Δ=n/a" if pd.isna(route_delta) else f"route Δ={route_delta:.0f}"
        _oldstyle_plot_residential_points(
            axes[row_idx, 0],
            sub,
            residential_points,
            "route_gap_type",
            f"{strategy}\nBEFORE placement | new={new_count}, {delta_text}",
        )
        _oldstyle_plot_residential_points(
            axes[row_idx, 1],
            sub,
            residential_points,
            "after_gap_type",
            f"{strategy}\nAFTER placement | new={new_count}",
        )
    handles = [
        Patch(facecolor=OLDSTYLE_GAP_COLORS[key], edgecolor="none", label=OLDSTYLE_GAP_LABELS[key])
        for key in OLDSTYLE_ORDER
        if (gaps["route_gap_type"].eq(key).any() or gaps["after_gap_type"].eq(key).any() or key == "ok")
    ]
    handles.append(Patch(facecolor="#ffd166", edgecolor="#1f2937", label="new selected service"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="#f8f5eb",
        edgecolor="#d6d3d1",
        fontsize=9,
    )
    fig.suptitle(
        "Polyclinic: unmet-demand type on residential buildings only",
        fontsize=16,
        fontweight="bold",
        color="white",
        y=0.988,
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.965))
    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _render_oldstyle_pattern_share_heatmap(pattern_summary: pd.DataFrame, out_path: Path) -> None:
    if pattern_summary.empty:
        return
    work = pattern_summary.copy()
    denom = work.groupby("strategy")["route_total_gap"].transform("sum").replace(0, np.nan)
    work["unmet_share"] = (work["route_total_gap"] / denom).fillna(0.0)
    order = [
        "Loops & Lollipops",
        "Irregular Grid",
        "Regular Grid",
        "Warped Parallel",
        "Sparse",
        "Broken Grid",
    ]
    strategies = [s.name for s in STRATEGIES]
    pivot = (
        work.pivot_table(
            index="strategy",
            columns="dominant_block_pattern",
            values="unmet_share",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=strategies)
        .fillna(0.0)
    )
    cols = [c for c in order if c in pivot.columns] + [c for c in pivot.columns if c not in order]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", vmin=0, vmax=max(0.01, float(pivot.to_numpy().max())))
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="#334155")
    ax.set_title("Unmet Demand By Dominant Street Pattern (weighted share)", fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.subplots_adjust(left=0.32, right=0.92, bottom=0.28, top=0.86)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_gap_solvability_matrix(summary_by_pattern: pd.DataFrame, out_path: Path) -> None:
    if summary_by_pattern.empty:
        return
    base = summary_by_pattern[summary_by_pattern["strategy"].eq("baseline_no_routes")].copy()
    if base.empty:
        base = summary_by_pattern.drop_duplicates("dominant_block_pattern").copy()
    base = base.sort_values("route_solvable_gap_weight", ascending=False)
    x = np.arange(len(base))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        x,
        base["route_solvable_gap_weight"],
        label="Route-solvable unmet-demand weight",
        color="#2f7f5f",
        alpha=0.85,
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        base["first_last_mile_blocked_failed_count"],
        label="First/last-mile blocked failed observations",
        color="#b85c38",
        marker="o",
        linewidth=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(base["dominant_block_pattern"], rotation=25, ha="right")
    ax.set_title("Baseline route-solvable unmet-demand signal by street pattern")
    ax.set_ylabel("route_total_gap * route_solvable_share")
    ax2.set_ylabel("failed building observations")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _pattern_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    pattern_col = None
    for candidate in ["street_pattern_dominant_class", "dominant_street_pattern", "pattern_class"]:
        if candidate in gaps.columns:
            pattern_col = candidate
            break
    if pattern_col is None:
        return pd.DataFrame()
    work = gaps.copy()
    work["dominant_block_pattern"] = work[pattern_col].fillna("unknown").astype(str)
    work["route_solvable_gap_weight"] = work["route_total_gap"] * work["route_solvable_share_of_failed"]
    return (
        work.groupby(["strategy", "dominant_block_pattern"], dropna=False)
        .agg(
            blocks=("block_id", "size"),
            route_total_gap=("route_total_gap", "sum"),
            after_total_gap=("after_total_gap", "sum"),
            route_solvable_failed_count=("route_solvable_failed_count", "sum"),
            first_last_mile_blocked_failed_count=("first_last_mile_blocked_failed_count", "sum"),
            route_solvable_gap_weight=("route_solvable_gap_weight", "sum"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default=DEFAULT_CITY)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--joint-root", type=Path, default=DEFAULT_JOINT_ROOT)
    parser.add_argument("--diagnostics-root", type=Path, default=DEFAULT_DIAGNOSTICS_ROOT)
    parser.add_argument("--pattern-root", type=Path, default=DEFAULT_PATTERN_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--only-final-canvas", action="store_true")
    parser.add_argument("--final-canvas-out", type=Path, default=None)
    args = parser.parse_args()

    experiment_city_dir = args.experiment_root / args.city
    city_root = _resolve_city_root(args.joint_root / args.city, experiment_city_dir)
    diagnostics_root = _resolve_diagnostics_root(args.diagnostics_root, args.city)
    out_dir = experiment_city_dir / "gap_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    layers: list[gpd.GeoDataFrame] = []
    for strategy in STRATEGIES:
        blocks = _read_strategy_blocks_from_experiment(
            city_root=city_root,
            experiment_city_dir=experiment_city_dir,
            service=args.service,
            strategy=strategy,
        )
        layers.append(_prepare_gap_layer(blocks))
    gaps = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True), crs=layers[0].crs)
    gaps = _attach_street_patterns(gaps, args.pattern_root, args.city)
    gaps = _attach_residential_building_stats(gaps, city_root)

    components = _read_strategy_component_diagnostics(
        city_root=city_root,
        diagnostics_root=diagnostics_root,
        experiment_city_dir=experiment_city_dir,
        city=args.city,
        service=args.service,
        gaps=gaps,
    )
    component_strategies = set(components["strategy"].astype(str).unique()) if not components.empty else set()
    gaps = gaps.merge(components, on=["strategy", "block_id"], how="left")
    component_cols = [c for c in components.columns if c not in {"strategy", "block_id", "component_source"}]
    for col in component_cols:
        gaps[col] = pd.to_numeric(gaps[col], errors="coerce").fillna(0.0)
    if "component_source" in gaps.columns:
        missing_source = gaps["component_source"].isna()
        missing_strategy = ~gaps["strategy"].astype(str).isin(component_strategies)
        gaps.loc[missing_source & missing_strategy, "component_source"] = "not_recomputed"
        gaps["component_source"] = gaps["component_source"].fillna("no_component_observations")
    gaps = _derive_combined_access_capacity_triage(gaps)

    summary = _summarize(gaps)
    route_stage_summary = _read_route_stage_summaries(experiment_city_dir)
    summary = summary.merge(route_stage_summary, on="strategy", how="left")
    pattern_summary = _pattern_summary(gaps)

    gaps.to_parquet(out_dir / "strategy_block_gap_diagnostics.parquet", index=False)
    combined_cols = [
        "strategy",
        "service_name",
        "block_id",
        "population",
        "residential_building_count",
        "residential_population_sum",
        "route_gap_type",
        "route_access_gap",
        "route_capacity_gap",
        "after_gap_type",
        "after_access_gap",
        "after_capacity_gap",
        "failed_access_count",
        "route_solvable_failed_count",
        "first_last_mile_blocked_failed_count",
        "home_to_stop_blocked_failed_count",
        "stop_to_service_blocked_failed_count",
        "pt_in_vehicle_failed_count",
        "component_source",
        "access_component_label",
        "action_triage_label",
        "new_service_block",
        "existing_service_block",
    ]
    gaps[[c for c in combined_cols if c in gaps.columns]].to_csv(
        out_dir / "combined_block_access_capacity_triage.csv",
        index=False,
    )
    summary.round(6).to_csv(out_dir / "strategy_gap_summary.csv", index=False)
    route_stage_summary.round(6).to_csv(out_dir / "strategy_route_stage_summary.csv", index=False)
    components.round(6).to_csv(out_dir / "block_route_solvability_diagnostics.csv", index=False)
    if not pattern_summary.empty:
        pattern_summary.round(6).to_csv(out_dir / "strategy_gap_by_street_pattern.csv", index=False)

    if args.only_final_canvas:
        final_canvas_out = args.final_canvas_out or out_dir / "oldstyle_before_only_service_gap_access_components.png"
        final_canvas_out.parent.mkdir(parents=True, exist_ok=True)
        _render_full_final_canvas(
            gaps,
            city_root,
            experiment_city_dir,
            args.service,
            args.city,
            final_canvas_out,
        )
        manifest = {
            "city": args.city,
            "service": args.service,
            "output_dir": str(out_dir),
            "final_canvas": str(final_canvas_out),
            "strategies": [s.name for s in STRATEGIES],
            "block_rows": int(len(gaps)),
            "component_blocks": int(len(components)),
            "png_count_in_city_output": 0,
            "notes": [
                "Only the final full route-strategy canvas was rendered.",
                "The final canvas combines access diagnostics, route-strategy overview maps, and best-option route/service maps.",
                "Strategy-specific route-stage before/after totals are read from route-count experiment outputs when available.",
            ],
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"output_dir": str(out_dir), "final_canvas": str(final_canvas_out), "block_rows": int(len(gaps))},
                ensure_ascii=False,
            )
        )
        return

    _render_summary_bars(summary, out_dir / "strategy_gap_summary_bars.png")
    _render_route_solvable_scatter(gaps, out_dir / "baseline_route_solvable_vs_unmet_scatter.png")
    _render_gap_maps(gaps, out_dir / "strategy_route_stage_unmet_maps.png")
    _render_oldstyle_contact_sheet(gaps, summary, out_dir / "oldstyle_strategy_gap_type_contact_sheet.png")
    _render_oldstyle_before_after_contact_sheet(
        gaps,
        summary,
        out_dir / "oldstyle_before_after_gap_type_contact_sheet.png",
    )
    _render_oldstyle_residential_points_before_after(
        gaps,
        summary,
        city_root,
        out_dir / "oldstyle_before_after_residential_buildings.png",
    )
    _render_combined_service_access_triage(
        gaps,
        summary,
        city_root,
        out_dir / "oldstyle_combined_service_gap_access_components.png",
    )
    _render_before_after_access_components(
        gaps,
        city_root,
        experiment_city_dir,
        args.service,
        out_dir / "oldstyle_before_only_service_gap_access_components.png",
    )
    oldstyle_before = _render_oldstyle_single_lp(gaps, summary, out_dir, stage="before")
    oldstyle_after = _render_oldstyle_single_lp(gaps, summary, out_dir, stage="after")
    if not pattern_summary.empty:
        _render_gap_solvability_matrix(pattern_summary, out_dir / "strategy_route_solvable_gap_by_pattern_heatmap.png")
        _render_oldstyle_pattern_share_heatmap(pattern_summary, out_dir / "oldstyle_unmet_share_by_pattern_strategy.png")

    manifest = {
        "city": args.city,
        "service": args.service,
        "output_dir": str(out_dir),
        "strategies": [s.name for s in STRATEGIES],
        "block_rows": int(len(gaps)),
        "component_blocks": int(len(components)),
        "oldstyle_png_count": int(4 + len(oldstyle_before) + len(oldstyle_after) + (0 if pattern_summary.empty else 1)),
        "notes": [
            "Strategy-specific route-stage before/after totals are read from polyclinic_summary_after_routes.json.",
            "Block-level parquets do not preserve strategy-specific route-stage gap changes; route_access_gap/route_capacity_gap are placement-input block gaps.",
            "after_access_gap/after_capacity_gap are the remaining gaps after exact placement.",
            "Route-solvable diagnostics are building-observation weighted, joined by building representative points to solver blocks.",
            "When recomputed per-strategy component diagnostics exist under gap_diagnostics/recomputed_access_components, they are used per strategy; otherwise baseline diagnostics are used as fallback.",
            "Oldstyle maps filter by residential_population_sum from blocksnet/buildings when available.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "block_rows": int(len(gaps))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

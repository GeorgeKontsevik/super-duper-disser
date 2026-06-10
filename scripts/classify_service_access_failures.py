#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SERVICES = ["hospital", "polyclinic", "school", "kindergarten"]
DEFAULT_THRESHOLD_MIN = 15.0
LABEL_ORDER = [
    "ok_walk",
    "ok_pt_only",
    "failed_no_pt_path",
    "failed_access_gt_threshold",
    "failed_egress_gt_threshold",
    "failed_access_egress_sum_gt_threshold",
    "failed_in_vehicle_gt_threshold",
    "failed_transfer_gt_threshold",
    "failed_multi_component_gt_threshold",
    "failed_total_gt_threshold_no_single_component_gt_threshold",
]
LABEL_COLORS = {
    "ok_walk": "#16a34a",
    "ok_pt_only": "#2563eb",
    "failed_no_pt_path": "#475569",
    "failed_access_gt_threshold": "#f59e0b",
    "failed_egress_gt_threshold": "#fb7185",
    "failed_access_egress_sum_gt_threshold": "#f97316",
    "failed_in_vehicle_gt_threshold": "#dc2626",
    "failed_transfer_gt_threshold": "#7c3aed",
    "failed_multi_component_gt_threshold": "#8b5cf6",
    "failed_total_gt_threshold_no_single_component_gt_threshold": "#6b7280",
}
LABEL_DISPLAY_SHORT = {
    "ok_walk": "walk",
    "ok_pt_only": "pt_only",
    "failed_no_pt_path": "no_pt",
    "failed_access_gt_threshold": "home_stop>T",
    "failed_egress_gt_threshold": "stop_service>T",
    "failed_access_egress_sum_gt_threshold": "both_walks>T",
    "failed_in_vehicle_gt_threshold": "in_vehicle>T",
    "failed_transfer_gt_threshold": "transfer>T",
    "failed_multi_component_gt_threshold": "multi>T",
    "failed_total_gt_threshold_no_single_component_gt_threshold": "sum>T_none>T",
}


def _city_dirs(base: Path) -> list[Path]:
    return sorted([p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")])


def _maps_subdir_name(home_street_patterns: list[str] | None) -> str:
    if not home_street_patterns:
        return "maps"
    slug = "_".join(pattern.lower().replace("&", "and").replace(" ", "_") for pattern in home_street_patterns)
    return f"maps_home_patterns_{slug}"


def _effective_pt_total_min(pt_time_min: float, pt_total_decomposed_time_min: float) -> float:
    if not np.isfinite(pt_time_min):
        return float("inf")
    return float(pt_total_decomposed_time_min)


def _classify_access_failure(
    walk_only_min: float,
    pt_total_min: float,
    access_walk_time_min: float,
    egress_walk_time_min: float,
    in_vehicle_time_min: float,
    transfer_time_min: float,
    threshold_min: float,
) -> str:
    if np.isfinite(walk_only_min) and walk_only_min <= threshold_min:
        return "ok_walk"

    if np.isfinite(pt_total_min) and pt_total_min <= threshold_min:
        return "ok_pt_only"

    if not np.isfinite(pt_total_min):
        return "failed_no_pt_path"

    exceeded: list[str] = []
    if np.isfinite(access_walk_time_min) and access_walk_time_min > threshold_min:
        exceeded.append("failed_access_gt_threshold")
    if np.isfinite(egress_walk_time_min) and egress_walk_time_min > threshold_min:
        exceeded.append("failed_egress_gt_threshold")
    if np.isfinite(in_vehicle_time_min) and in_vehicle_time_min > threshold_min:
        exceeded.append("failed_in_vehicle_gt_threshold")
    if np.isfinite(transfer_time_min) and transfer_time_min > threshold_min:
        exceeded.append("failed_transfer_gt_threshold")

    if len(exceeded) == 1:
        return exceeded[0]
    if len(exceeded) > 1:
        return "failed_multi_component_gt_threshold"
    if (
        np.isfinite(access_walk_time_min)
        and np.isfinite(egress_walk_time_min)
        and access_walk_time_min + egress_walk_time_min > threshold_min
    ):
        return "failed_access_egress_sum_gt_threshold"
    return "failed_total_gt_threshold_no_single_component_gt_threshold"


def _living_building_points(buildings_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(buildings_path)
    living_flag = pd.to_numeric(gdf["is_living"], errors="coerce").fillna(0).astype(float) > 0.0
    living = gdf[living_flag].copy()
    if living.empty:
        return living
    living = living.reset_index(drop=False).rename(columns={"index": "building_idx"})
    living["geometry"] = living.geometry.representative_point()
    return living[["building_idx", "geometry"]]


def _buildings_access_path(derived: Path) -> Path:
    is_living_path = derived / "buildings_is_living_enriched.parquet"
    if is_living_path.exists():
        return is_living_path
    return derived / "buildings_floor_enriched.parquet"


def _panel_summary_text(counts: pd.Series, max_line_chars: int = 44) -> str:
    parts = [
        f"{LABEL_DISPLAY_SHORT.get(label, label)}={int(counts.get(label, 0))}"
        for label in LABEL_ORDER
        if int(counts.get(label, 0)) > 0
    ]
    if not parts:
        return ""
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}, {part}"
        if len(candidate) <= max_line_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = part
    if current:
        lines.append(current)
    return "\n".join(lines)


def _load_city_diagnostics(
    city: str,
    walk_root: Path,
    pt_lt_root: Path,
    pt_ge_root: Path,
    threshold_min: float,
) -> pd.DataFrame:
    walk_path = walk_root / city / "residential_to_services_top1.parquet"
    if not walk_path.exists():
        return pd.DataFrame()
    walk_df = pd.read_parquet(
        walk_path,
        columns=[
            "building_idx",
            "service_name",
            "walk_time_min",
            "home_street_pattern_class",
            "service_street_pattern_class",
            "nearest_service_name",
        ],
    )

    pt_frames = []
    for root in (pt_lt_root, pt_ge_root):
        pt_path = root / city / "residential_to_services_pt_top1.parquet"
        if pt_path.exists():
            pt_frames.append(
                pd.read_parquet(
                    pt_path,
                    columns=[
                        "building_idx",
                        "service_name",
                        "pt_time_min",
                        "access_walk_time_min",
                        "egress_walk_time_min",
                        "transport_time_min",
                        "transfer_time_min",
                        "pt_total_decomposed_time_min",
                    ],
                )
            )
    if not pt_frames:
        return pd.DataFrame()

    pt_df = pd.concat(pt_frames, ignore_index=True)
    pt_df = pt_df.drop_duplicates(subset=["building_idx", "service_name"], keep="first")
    merged = walk_df.merge(pt_df, on=["building_idx", "service_name"], how="left")
    merged["threshold_min"] = float(threshold_min)
    merged["effective_pt_total_min"] = merged.apply(
        lambda row: _effective_pt_total_min(
            pt_time_min=float(row["pt_time_min"]) if pd.notna(row["pt_time_min"]) else float("inf"),
            pt_total_decomposed_time_min=(
                float(row["pt_total_decomposed_time_min"])
                if pd.notna(row["pt_total_decomposed_time_min"])
                else float("inf")
            ),
        ),
        axis=1,
    )
    merged["access_diagnosis_label"] = merged.apply(
        lambda row: _classify_access_failure(
            walk_only_min=float(row["walk_time_min"]) if pd.notna(row["walk_time_min"]) else float("inf"),
            pt_total_min=float(row["effective_pt_total_min"]),
            access_walk_time_min=float(row["access_walk_time_min"]) if pd.notna(row["access_walk_time_min"]) else 0.0,
            egress_walk_time_min=float(row["egress_walk_time_min"]) if pd.notna(row["egress_walk_time_min"]) else 0.0,
            in_vehicle_time_min=float(row["transport_time_min"]) if pd.notna(row["transport_time_min"]) else 0.0,
            transfer_time_min=float(row["transfer_time_min"]) if pd.notna(row["transfer_time_min"]) else 0.0,
            threshold_min=threshold_min,
        ),
        axis=1,
    )
    return merged


def _render_city_map(
    city: str,
    classified: pd.DataFrame,
    buildings_root: Path,
    boundary_path: Path,
    out_path: Path,
    home_street_patterns: list[str] | None = None,
) -> None:
    building_points = _living_building_points(_buildings_access_path(buildings_root / city / "derived_layers"))
    if building_points.empty:
        return
    gdf = building_points.merge(classified, on="building_idx", how="inner")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=building_points.crs)
    if home_street_patterns:
        gdf = gdf[gdf["home_street_pattern_class"].isin(home_street_patterns)].copy()
    if gdf.empty:
        return
    boundary = gpd.read_parquet(boundary_path / city / "blocksnet" / "boundary.parquet")

    services = [s for s in DEFAULT_SERVICES if s in gdf["service_name"].unique()]
    if not services:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=220)
    axes = axes.flatten()
    plot_order = [
        "failed_no_pt_path",
        "failed_transfer_gt_threshold",
        "failed_access_gt_threshold",
        "failed_egress_gt_threshold",
        "failed_access_egress_sum_gt_threshold",
        "failed_in_vehicle_gt_threshold",
        "failed_multi_component_gt_threshold",
        "failed_total_gt_threshold_no_single_component_gt_threshold",
        "ok_pt_only",
        "ok_walk",
    ]
    for ax, service in zip(axes, DEFAULT_SERVICES, strict=False):
        ax.set_axis_off()
        boundary.plot(ax=ax, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.7)
        sub = gdf[gdf["service_name"] == service].copy()
        if sub.empty:
            ax.set_title(f"{service} (no data)", fontsize=11)
            continue
        counts = sub["access_diagnosis_label"].value_counts()
        for label in plot_order:
            points = sub[sub["access_diagnosis_label"] == label]
            if points.empty:
                continue
            points.plot(
                ax=ax,
                color=LABEL_COLORS[label],
                markersize=2.2,
                alpha=0.55,
            )
        ax.set_title(f"{service} (n={len(sub)})", fontsize=11)
        ax.text(
            0.01,
            0.01,
            _panel_summary_text(counts),
            transform=ax.transAxes,
            fontsize=6.5,
            ha="left",
            va="bottom",
            color="#334155",
        )
    for ax in axes[len(DEFAULT_SERVICES) :]:
        ax.set_axis_off()

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[label], markersize=7, label=label)
        for label in LABEL_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    title = f"{city}: home -> service accessibility diagnosis"
    if home_street_patterns:
        title += f" ({', '.join(home_street_patterns)})"
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.12, wspace=0.20, hspace=0.22)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--walk-root",
        type=Path,
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_top1"),
    )
    parser.add_argument(
        "--pt-walk-lt-root",
        type=Path,
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_pt_top1_walk_lt15"),
    )
    parser.add_argument(
        "--pt-walk-ge-root",
        type=Path,
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_pt_top1_walk15plus"),
    )
    parser.add_argument(
        "--joint-inputs-root",
        type=Path,
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics"),
    )
    parser.add_argument("--threshold-min", type=float, default=DEFAULT_THRESHOLD_MIN)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--home-street-patterns", nargs="*", default=None)
    args = parser.parse_args()

    city_dirs = _city_dirs(args.walk_root)
    if args.cities:
        wanted = set(args.cities)
        city_dirs = [c for c in city_dirs if c.name in wanted]

    out_root = args.out_root
    maps_root = out_root / _maps_subdir_name(args.home_street_patterns)
    maps_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary_rows = []

    for city_dir in city_dirs:
        city = city_dir.name
        city_df = _load_city_diagnostics(
            city=city,
            walk_root=args.walk_root,
            pt_lt_root=args.pt_walk_lt_root,
            pt_ge_root=args.pt_walk_ge_root,
            threshold_min=args.threshold_min,
        )
        if city_df.empty:
            continue
        city_df["city"] = city
        all_rows.append(city_df)
        city_out_dir = out_root / city
        city_out_dir.mkdir(parents=True, exist_ok=True)
        city_df.to_parquet(city_out_dir / "home_to_service_access_diagnostics.parquet", index=False)

        summary = (
            city_df.groupby(["city", "service_name", "access_diagnosis_label"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        total = summary.groupby(["city", "service_name"])["count"].transform("sum")
        summary["share"] = summary["count"] / total
        summary_rows.append(summary)
        summary.to_csv(city_out_dir / "home_to_service_access_diagnostics_summary.csv", index=False)

        _render_city_map(
            city=city,
            classified=city_df[
                ["building_idx", "service_name", "access_diagnosis_label", "home_street_pattern_class"]
            ],
            buildings_root=args.joint_inputs_root,
            boundary_path=args.joint_inputs_root,
            out_path=maps_root / f"{city}_home_to_service_access_diagnostics.png",
            home_street_patterns=args.home_street_patterns,
        )

    if not all_rows:
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_parquet(out_root / "_all_home_to_service_access_diagnostics.parquet", index=False)

    combined_summary = pd.concat(summary_rows, ignore_index=True)
    combined_summary.to_csv(out_root / "_all_home_to_service_access_diagnostics_summary.csv", index=False)


if __name__ == "__main__":
    main()

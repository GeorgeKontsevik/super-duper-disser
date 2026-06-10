#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(
    "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412"
)
JOINT_INPUTS_ROOT = Path(
    "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"
)
HOME_ROOT = ROOT / "residential_to_pt_top3"
SERVICE_ROOT = ROOT / "services_to_pt_top3"
HOME_SERVICE_ROOT = ROOT / "residential_to_services_top1"
HOME_SERVICE_PT15_ROOT = ROOT / "residential_to_services_pt_top1_walk15plus"
HOME_SERVICE_PTLT15_ROOT = ROOT / "residential_to_services_pt_top1_walk_lt15"
OUT_ROOT = ROOT / "pt_access_distributions"
OUT_HOMES_TO_PT = OUT_ROOT / "homes_to_pt"
OUT_SERVICES_TO_PT = OUT_ROOT / "services_to_pt"
OUT_HOME_TO_SERVICE_WALK = OUT_ROOT / "home_to_service_walk"
OUT_HOME_TO_SERVICE_PT15 = OUT_ROOT / "home_to_service_pt_walk15plus"
OUT_HOME_TO_SERVICE_PTLT15 = OUT_ROOT / "home_to_service_pt_walklt15"
SERVICES = ["hospital", "polyclinic", "school", "kindergarten"]
WALK_MIN_PER_M = 0.012
STREET_PATTERN_ORDER = [
    "Regular Grid",
    "Irregular Grid",
    "Warped Parallel",
    "Broken Grid",
    "Sparse",
    "Loops & Lollipops",
]

sns.set_theme(style="whitegrid", context="notebook")


def _city_density(city: str) -> float:
    blocksnet_root = JOINT_INPUTS_ROOT / city / "blocksnet"
    buildings = pd.read_parquet(blocksnet_root / "buildings.parquet", columns=["population"])
    boundary = gpd.read_parquet(blocksnet_root / "boundary.parquet")
    if "population" not in buildings.columns:
        raise KeyError(f"{city}: missing buildings.population")
    if "geometry" not in boundary.columns:
        raise KeyError(f"{city}: missing boundary geometry")

    utm_crs = boundary.estimate_utm_crs()
    area_km2 = float(boundary.to_crs(utm_crs).geometry.area.sum()) / 1_000_000
    if area_km2 <= 0:
        raise ValueError(f"{city}: non-positive boundary area")

    population = float(buildings["population"].fillna(0).sum())
    return population / area_km2


def _city_order_by_density(cities: list[str]) -> list[str]:
    return sorted(cities, key=_city_density, reverse=True)


def _street_pattern_order(patterns: list[str]) -> list[str]:
    known = [pattern for pattern in STREET_PATTERN_ORDER if pattern in patterns]
    extra = sorted(pattern for pattern in patterns if pattern not in STREET_PATTERN_ORDER)
    return known + extra


def _split_strong_outliers(values: list[float]) -> tuple[list[float], int, float]:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return [], 0, float("nan")

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return series.tolist(), 0, float("inf")

    cutoff = q3 + 3.0 * iqr
    kept = series[series <= cutoff]
    removed_count = int((series > cutoff).sum())
    return kept.tolist(), removed_count, cutoff


def _load_home_rank1() -> pd.DataFrame:
    rows = []
    for city_dir in sorted([p for p in HOME_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        p = city_dir / "residential_to_pt_top3.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["rank", "walk_distance_m", "street_pattern_class"])
        df = df[df["rank"] == 1].copy()
        if df.empty:
            continue
        df["city"] = city_dir.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_service_rank1() -> pd.DataFrame:
    rows = []
    for city_dir in sorted([p for p in SERVICE_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        for service in SERVICES:
            p = city_dir / f"{service}_to_pt_top3.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p, columns=["rank", "walk_distance_m", "service_name", "street_pattern_class"])
            df = df[df["rank"] == 1].copy()
            if df.empty:
                continue
            df["city"] = city_dir.name
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_home_to_service_top1() -> pd.DataFrame:
    rows = []
    for city_dir in sorted([p for p in HOME_SERVICE_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        p = city_dir / "residential_to_services_top1.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(
            p,
            columns=["service_name", "walk_time_min", "walk_distance_m", "home_street_pattern_class"],
        )
        if df.empty:
            continue
        df["city"] = city_dir.name
        df["street_pattern_class"] = df["home_street_pattern_class"]
        rows.append(df.drop(columns=["home_street_pattern_class"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_home_to_service_pt_top1(root: Path) -> pd.DataFrame:
    rows = []
    for city_dir in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        p = city_dir / "residential_to_services_pt_top1.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(
            p,
            columns=["service_name", "pt_time_min", "home_street_pattern_class"],
        )
        if df.empty:
            continue
        df["city"] = city_dir.name
        df["walk_time_min"] = pd.to_numeric(df["pt_time_min"], errors="coerce")
        df["street_pattern_class"] = df["home_street_pattern_class"]
        rows.append(df.drop(columns=["pt_time_min", "home_street_pattern_class"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_home_to_service_pt_components(root: Path) -> pd.DataFrame:
    rows = []
    for city_dir in sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]):
        p = city_dir / "residential_to_services_pt_top1.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(
            p,
            columns=[
                "service_name",
                "access_egress_walk_time_min",
                "transport_time_min",
                "other_edge_time_min",
            ],
        )
        if df.empty:
            continue
        df["city"] = city_dir.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _finite(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["walk_distance_m"] = pd.to_numeric(out["walk_distance_m"], errors="coerce")
    out["walk_time_min"] = out["walk_distance_m"] * WALK_MIN_PER_M
    return out[out["walk_distance_m"].notna() & (out["walk_distance_m"] != float("inf"))].copy()


def _known_street_pattern(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["street_pattern_class"] = out["street_pattern_class"].fillna("UNKNOWN").astype(str)
    return out[out["street_pattern_class"].str.upper() != "UNKNOWN"].copy()


def _add_15_min_reference(ax: plt.Axes) -> None:
    ax.axvline(15, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.9)


def _add_15_min_axis_label(ax: plt.Axes) -> None:
    ax.text(
        15,
        -0.22,
        "15 min",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color="#dc2626",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )


def _save_city_boxplot(df: pd.DataFrame, title: str, out_path: Path, x_label: str) -> None:
    city_order = _city_order_by_density(df["city"].drop_duplicates().tolist())
    n = len(city_order)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3 * n, 4)), dpi=220, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, city in zip(axes, city_order, strict=False):
        group = df[df["city"] == city]
        kept, removed_count, cutoff = _split_strong_outliers(group["walk_time_min"].tolist())
        plot_df = pd.DataFrame({"walk_time_min": kept})
        sns.histplot(
            data=plot_df,
            x="walk_time_min",
            bins=40,
            color="#60a5fa",
            edgecolor="white",
            ax=ax,
        )
        ax.set_title(f"{city} (n={len(group)})", loc="left", fontsize=10)
        ax.set_ylabel("count")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.6, alpha=0.8)
        _add_15_min_reference(ax)
        if removed_count > 0:
            ax.text(
                0.99,
                -0.22,
                f"excluded outliers: {removed_count} > {cutoff:.1f} min",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#6b7280",
            )
    axes[-1].set_xlabel(x_label)
    _add_15_min_axis_label(axes[-1])
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_histogram(df: pd.DataFrame, group_col: str, title: str, out_path: Path, x_label: str) -> None:
    groups = [g for _, g in df.groupby(group_col)]
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3 * n, 4)), dpi=220, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (name, group) in zip(axes, df.groupby(group_col), strict=False):
        kept, removed_count, cutoff = _split_strong_outliers(group["walk_time_min"].tolist())
        plot_df = pd.DataFrame({"walk_time_min": kept})
        sns.histplot(data=plot_df, x="walk_time_min", bins=40, color="#60a5fa", edgecolor="white", ax=ax)
        ax.set_title(f"{name} (n={len(group)})", loc="left", fontsize=10)
        ax.set_ylabel("count")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.6, alpha=0.8)
        _add_15_min_reference(ax)
        if removed_count > 0:
            ax.text(
                0.99,
                -0.22,
                f"excluded outliers: {removed_count} > {cutoff:.1f} min",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#6b7280",
            )
    axes[-1].set_xlabel(x_label)
    _add_15_min_axis_label(axes[-1])
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_group_boxplot(df: pd.DataFrame, group_col: str, title: str, out_path: Path, x_label: str) -> None:
    if group_col == "street_pattern_class":
        group_order = _street_pattern_order(df[group_col].drop_duplicates().tolist())
    else:
        group_order = (
            df.groupby(group_col)["walk_time_min"]
            .median()
            .sort_values()
            .index
            .tolist()
        )
    n = len(group_order)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3 * n, 4)), dpi=220, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, group_name in zip(axes, group_order, strict=False):
        group = df[df[group_col] == group_name]
        kept, removed_count, cutoff = _split_strong_outliers(group["walk_time_min"].tolist())
        plot_df = pd.DataFrame({"walk_time_min": kept})
        sns.histplot(
            data=plot_df,
            x="walk_time_min",
            bins=40,
            color="#60a5fa",
            edgecolor="white",
            ax=ax,
        )
        ax.set_title(f"{group_name} (n={len(group)})", loc="left", fontsize=10)
        ax.set_ylabel("count")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.6, alpha=0.8)
        _add_15_min_reference(ax)
        if removed_count > 0:
            ax.text(
                0.99,
                -0.22,
                f"excluded outliers: {removed_count} > {cutoff:.1f} min",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#6b7280",
            )
    axes[-1].set_xlabel(x_label)
    _add_15_min_axis_label(axes[-1])
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _save_city_component_distributions(df: pd.DataFrame, title: str, out_path: Path) -> None:
    city_order = _city_order_by_density(df["city"].drop_duplicates().tolist())
    n = len(city_order)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3 * n, 4)), dpi=220, sharex=True)
    if n == 1:
        axes = [axes]
    for ax, city in zip(axes, city_order, strict=False):
        group = df[df["city"] == city].copy()
        walk_kept, walk_removed_count, walk_cutoff = _split_strong_outliers(
            pd.to_numeric(group["access_egress_walk_time_min"], errors="coerce").tolist()
        )
        transport_kept, transport_removed_count, transport_cutoff = _split_strong_outliers(
            pd.to_numeric(group["transport_time_min"], errors="coerce").tolist()
        )
        walk_df = pd.DataFrame({"minutes": walk_kept})
        transport_df = pd.DataFrame({"minutes": transport_kept})
        sns.histplot(
            data=walk_df,
            x="minutes",
            bins=35,
            color="#93c5fd",
            edgecolor=None,
            alpha=0.45,
            stat="count",
            ax=ax,
            label="Walk segments in PT path",
        )
        sns.histplot(
            data=transport_df,
            x="minutes",
            bins=35,
            color="#2563eb",
            edgecolor=None,
            alpha=0.45,
            stat="count",
            ax=ax,
            label="In-vehicle PT",
        )
        ax.set_title(f"{city} (n={len(group)})", loc="left", fontsize=10)
        ax.set_ylabel("count")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.6, alpha=0.8)
        removed_notes = []
        if walk_removed_count > 0:
            removed_notes.append(f"walk: {walk_removed_count} > {walk_cutoff:.1f}")
        if transport_removed_count > 0:
            removed_notes.append(f"pt: {transport_removed_count} > {transport_cutoff:.1f}")
        if removed_notes:
            ax.text(
                0.99,
                -0.22,
                "excluded outliers: " + "; ".join(removed_notes) + " min",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#6b7280",
            )
    axes[-1].set_xlabel("Component time, min")
    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_HOMES_TO_PT.mkdir(parents=True, exist_ok=True)
    OUT_SERVICES_TO_PT.mkdir(parents=True, exist_ok=True)
    OUT_HOME_TO_SERVICE_WALK.mkdir(parents=True, exist_ok=True)
    OUT_HOME_TO_SERVICE_PT15.mkdir(parents=True, exist_ok=True)
    OUT_HOME_TO_SERVICE_PTLT15.mkdir(parents=True, exist_ok=True)

    homes = _finite(_load_home_rank1())
    services = _finite(_load_service_rank1())
    home_services = _finite(_load_home_to_service_top1())
    home_services_pt15 = _load_home_to_service_pt_top1(HOME_SERVICE_PT15_ROOT)
    home_services_ptlt15 = _load_home_to_service_pt_top1(HOME_SERVICE_PTLT15_ROOT)
    home_services_pt15_components = _load_home_to_service_pt_components(HOME_SERVICE_PT15_ROOT)
    home_services_ptlt15_components = _load_home_to_service_pt_components(HOME_SERVICE_PTLT15_ROOT)
    homes_sp = _known_street_pattern(homes) if not homes.empty else homes
    services_sp = _known_street_pattern(services) if not services.empty else services
    home_services_sp = _known_street_pattern(home_services) if not home_services.empty else home_services
    home_services_pt15_sp = _known_street_pattern(home_services_pt15) if not home_services_pt15.empty else home_services_pt15
    home_services_ptlt15_sp = _known_street_pattern(home_services_ptlt15) if not home_services_ptlt15.empty else home_services_ptlt15

    if not homes.empty:
        _save_city_boxplot(
            homes,
            "Homes: reachable walk-time distribution to nearest PT stop by city (rank=1)",
            OUT_HOMES_TO_PT / "homes_rank1_city_boxplot_minutes.png",
            "Walk time to nearest PT stop, min",
        )
    if not homes_sp.empty:
        _save_group_boxplot(
            homes_sp,
            "street_pattern_class",
            "Homes: reachable walk-time distribution to nearest PT stop by home street pattern (rank=1)",
            OUT_HOMES_TO_PT / "homes_rank1_street_pattern_boxplot_minutes.png",
            "Walk time to nearest PT stop, min",
        )

    if not services.empty:
        _save_histogram(
            services,
            "service_name",
            "Services: reachable walk-time distribution to nearest PT stop by service (rank=1, all cities)",
            OUT_SERVICES_TO_PT / "services_rank1_hist_by_service_minutes.png",
            "Walk time to nearest PT stop, min",
        )
        _save_city_boxplot(
            services,
            "Services: reachable walk-time distribution to nearest PT stop by city (rank=1, all services)",
            OUT_SERVICES_TO_PT / "services_rank1_city_boxplot_minutes.png",
            "Walk time to nearest PT stop, min",
        )
    if not services_sp.empty:
        _save_group_boxplot(
            services_sp,
            "street_pattern_class",
            "Services: reachable walk-time distribution to nearest PT stop by street pattern (rank=1, all services)",
            OUT_SERVICES_TO_PT / "services_rank1_street_pattern_boxplot_minutes.png",
            "Walk time to nearest PT stop, min",
        )

        for service in SERVICES:
            subset = services[services["service_name"] == service].copy()
            if subset.empty:
                continue
            _save_city_boxplot(
                subset,
                f"{service}: reachable walk-time distribution to nearest PT stop by city (rank=1)",
                OUT_SERVICES_TO_PT / f"{service}_rank1_city_boxplot_minutes.png",
                "Walk time to nearest PT stop, min",
            )
            subset_sp = services_sp[services_sp["service_name"] == service].copy()
            if subset_sp.empty:
                continue
            _save_group_boxplot(
                subset_sp,
                "street_pattern_class",
                f"{service}: reachable walk-time distribution to nearest PT stop by street pattern (rank=1)",
                OUT_SERVICES_TO_PT / f"{service}_rank1_street_pattern_boxplot_minutes.png",
                "Walk time to nearest PT stop, min",
            )

    if not home_services.empty:
        for service in SERVICES:
            subset = home_services[home_services["service_name"] == service].copy()
            if subset.empty:
                continue
            _save_city_boxplot(
                subset,
                f"Homes to nearest {service}: reachable walk-time distribution by city",
                OUT_HOME_TO_SERVICE_WALK / f"home_to_{service}_city_boxplot_minutes.png",
                f"Walk time to nearest {service}, min",
            )

    if not home_services_sp.empty:
        for service in SERVICES:
            subset_sp = home_services_sp[home_services_sp["service_name"] == service].copy()
            if subset_sp.empty:
                continue
            _save_group_boxplot(
                subset_sp,
                "street_pattern_class",
                f"Homes to nearest {service}: reachable walk-time distribution by home street pattern",
                OUT_HOME_TO_SERVICE_WALK / f"home_to_{service}_street_pattern_boxplot_minutes.png",
                f"Walk time to nearest {service}, min",
            )

    if not home_services_pt15.empty:
        for service in SERVICES:
            subset = home_services_pt15[home_services_pt15["service_name"] == service].copy()
            if subset.empty:
                continue
            _save_city_boxplot(
                subset,
                f"Homes to nearest {service} by PT: city distribution (walk >= 15 min)",
                OUT_HOME_TO_SERVICE_PT15 / f"home_to_{service}_pt_city_hist_minutes_walk15plus.png",
                f"PT time to nearest {service}, min",
            )
            component_subset = home_services_pt15_components[
                home_services_pt15_components["service_name"] == service
            ].copy()
            if not component_subset.empty:
                _save_city_component_distributions(
                    component_subset,
                    f"Homes to nearest {service} by PT: component distributions by city (walk >= 15 min)",
                    OUT_HOME_TO_SERVICE_PT15 / f"home_to_{service}_pt_city_component_split_walk15plus.png",
                )

    if not home_services_pt15_sp.empty:
        for service in SERVICES:
            subset_sp = home_services_pt15_sp[home_services_pt15_sp["service_name"] == service].copy()
            if subset_sp.empty:
                continue
            _save_group_boxplot(
                subset_sp,
                "street_pattern_class",
                f"Homes to nearest {service} by PT: home-street-pattern distribution (walk >= 15 min)",
                OUT_HOME_TO_SERVICE_PT15 / f"home_to_{service}_pt_street_pattern_hist_minutes_walk15plus.png",
                f"PT time to nearest {service}, min",
            )

    if not home_services_ptlt15.empty:
        for service in SERVICES:
            subset = home_services_ptlt15[home_services_ptlt15["service_name"] == service].copy()
            if subset.empty:
                continue
            _save_city_boxplot(
                subset,
                f"Homes to nearest {service} by PT: city distribution (walk < 15 min)",
                OUT_HOME_TO_SERVICE_PTLT15 / f"home_to_{service}_pt_city_hist_minutes_walklt15.png",
                f"PT time to nearest {service}, min",
            )
            component_subset = home_services_ptlt15_components[
                home_services_ptlt15_components["service_name"] == service
            ].copy()
            if not component_subset.empty:
                _save_city_component_distributions(
                    component_subset,
                    f"Homes to nearest {service} by PT: component distributions by city (walk < 15 min)",
                    OUT_HOME_TO_SERVICE_PTLT15 / f"home_to_{service}_pt_city_component_split_walklt15.png",
                )

    if not home_services_ptlt15_sp.empty:
        for service in SERVICES:
            subset_sp = home_services_ptlt15_sp[home_services_ptlt15_sp["service_name"] == service].copy()
            if subset_sp.empty:
                continue
            _save_group_boxplot(
                subset_sp,
                "street_pattern_class",
                f"Homes to nearest {service} by PT: home-street-pattern distribution (walk < 15 min)",
                OUT_HOME_TO_SERVICE_PTLT15 / f"home_to_{service}_pt_street_pattern_hist_minutes_walklt15.png",
                f"PT time to nearest {service}, min",
            )


if __name__ == "__main__":
    main()

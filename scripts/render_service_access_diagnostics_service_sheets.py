#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


SERVICE_ORDER = ["hospital", "polyclinic", "school", "kindergarten"]
SERVICE_RU = {
    "hospital": ("01_bolnitsy_access_diagnostics_ru.png", "больницы"),
    "polyclinic": ("02_polikliniki_access_diagnostics_ru.png", "поликлиники"),
    "school": ("03_shkoly_access_diagnostics_ru.png", "школы"),
    "kindergarten": ("04_detskie_sady_access_diagnostics_ru.png", "детские сады"),
}

CITY_RU = {
    "bergen_norway": "Берген, Норвегия",
    "bologna_italy": "Болонья, Италия",
    "bristol_united_kingdom": "Бристоль, Великобритания",
    "brno_czechia": "Брно, Чехия",
    "coimbra_portugal": "Коимбра, Португалия",
    "debrecen_hungary": "Дебрецен, Венгрия",
    "dresden_germany": "Дрезден, Германия",
    "freiburg_im_breisgau_germany": "Фрайбург, Германия",
    "gothenburg_sweden": "Гетеборг, Швеция",
    "graz_austria": "Грац, Австрия",
    "hrodna_belarus": "Гродно, Беларусь",
    "innsbruck_austria": "Инсбрук, Австрия",
    "kaliningrad_russia": "Калининград, Россия",
    "linz_austria": "Линц, Австрия",
    "lyon_france": "Лион, Франция",
    "novi_sad_serbia": "Нови-Сад, Сербия",
    "porto_portugal": "Порту, Португалия",
    "turin_italy": "Турин, Италия",
    "turku_finland": "Турку, Финляндия",
    "zaragoza_spain": "Сарагоса, Испания",
}

CITY_ORDER = list(CITY_RU)

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

PLOT_ORDER = [
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

LABEL_RU = {
    "ok_walk": "доступно пешком",
    "ok_pt_only": "доступно на ОТ",
    "failed_no_pt_path": "нет пути по ОТ",
    "failed_access_gt_threshold": "дом - остановка > 15 мин",
    "failed_egress_gt_threshold": "остановка - сервис > 15 мин",
    "failed_access_egress_sum_gt_threshold": "оба пеших участка > 15 мин",
    "failed_in_vehicle_gt_threshold": "поездка в ОТ > 15 мин",
    "failed_transfer_gt_threshold": "пересадки > 15 мин",
    "failed_multi_component_gt_threshold": "несколько компонент > 15 мин",
    "failed_total_gt_threshold_no_single_component_gt_threshold": "сумма > 15 мин, без доминирующей компоненты",
}

LABEL_RU_SHORT = {
    "ok_walk": "пешком",
    "ok_pt_only": "ОТ",
    "failed_no_pt_path": "нет ОТ",
    "failed_access_gt_threshold": "дом-ост",
    "failed_egress_gt_threshold": "ост-сервис",
    "failed_access_egress_sum_gt_threshold": "пешие",
    "failed_in_vehicle_gt_threshold": "в ОТ",
    "failed_transfer_gt_threshold": "пересадки",
    "failed_multi_component_gt_threshold": "много",
    "failed_total_gt_threshold_no_single_component_gt_threshold": "сумма",
}


def _buildings_path(city_root: Path) -> Path:
    derived = city_root / "derived_layers"
    is_living_path = derived / "buildings_is_living_enriched.parquet"
    if is_living_path.exists():
        return is_living_path
    return derived / "buildings_floor_enriched.parquet"


def _living_points(city_root: Path) -> gpd.GeoDataFrame:
    buildings = gpd.read_parquet(_buildings_path(city_root))
    if "is_living" in buildings.columns:
        living_mask = pd.to_numeric(buildings["is_living"], errors="coerce").fillna(0).astype(float) > 0
        buildings = buildings[living_mask].copy()
    buildings = buildings.reset_index(drop=False).rename(columns={"index": "building_idx"})
    buildings["geometry"] = buildings.geometry.representative_point()
    return buildings[["building_idx", "geometry"]]


def _panel_stats(counts: pd.Series) -> str:
    parts = []
    for label in LABEL_ORDER:
        value = int(counts.get(label, 0))
        if value:
            parts.append(f"{LABEL_RU_SHORT[label]}={value}")
    lines = []
    current = ""
    for part in parts:
        candidate = part if not current else f"{current}, {part}"
        if len(candidate) <= 36:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    return "\n".join(lines)


def _load_city_layers(joint_inputs_root: Path, cities: list[str]) -> dict[str, tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:
    layers = {}
    for city in cities:
        city_root = joint_inputs_root / city
        points = _living_points(city_root)
        boundary = gpd.read_parquet(city_root / "blocksnet" / "boundary.parquet")
        metric_crs = boundary.estimate_utm_crs()
        if metric_crs is not None:
            boundary = boundary.to_crs(metric_crs)
            points = points.to_crs(metric_crs)
        layers[city] = (points, boundary)
    return layers


def _plot_boundary(ax, boundary: gpd.GeoDataFrame) -> None:
    boundary.plot(ax=ax, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.95)
    minx, miny, maxx, maxy = boundary.total_bounds
    pad_x = (maxx - minx) * 0.025
    pad_y = (maxy - miny) * 0.025
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def render_service_sheet(
    diagnostics: pd.DataFrame,
    layers: dict[str, tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]],
    service: str,
    cities: list[str],
    out_path: Path,
) -> None:
    _, service_ru = SERVICE_RU[service]
    fig, axes = plt.subplots(5, 4, figsize=(16, 22), dpi=240)
    axes_flat = axes.ravel()

    for ax, city in zip(axes_flat, cities, strict=True):
        points, boundary = layers[city]
        _plot_boundary(ax, boundary)

        sub = diagnostics[(diagnostics["city"] == city) & (diagnostics["service_name"] == service)].copy()
        gdf = points.merge(sub[["building_idx", "access_diagnosis_label"]], on="building_idx", how="inner")
        if not gdf.empty:
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=points.crs)
            for label in PLOT_ORDER:
                pts = gdf[gdf["access_diagnosis_label"] == label]
                if pts.empty:
                    continue
                ax.scatter(
                    pts.geometry.x,
                    pts.geometry.y,
                    s=5.2,
                    c=LABEL_COLORS[label],
                    alpha=0.72,
                    linewidths=0,
                    rasterized=True,
                )

        ax.set_title(CITY_RU.get(city, city), fontsize=15.6, pad=7)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=LABEL_COLORS[label],
            markeredgecolor="none",
            markersize=10,
            label=LABEL_RU[label],
        )
        for label in LABEL_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=15,
        columnspacing=2.0,
        handletextpad=0.7,
        borderaxespad=0.0,
    )
    fig.suptitle(f"Диагностика доступности: {service_ru}", fontsize=25, y=0.992)
    fig.subplots_adjust(left=0.032, right=0.99, top=0.942, bottom=0.098, wspace=0.10, hspace=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/_all_home_to_service_access_diagnostics.parquet"),
    )
    parser.add_argument(
        "--joint-inputs-root",
        type=Path,
        default=Path("aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/maps_by_service_ru"),
    )
    parser.add_argument("--services", nargs="*", default=SERVICE_ORDER)
    parser.add_argument("--cities", nargs="*", default=CITY_ORDER)
    args = parser.parse_args()

    diagnostics = pd.read_parquet(
        args.diagnostics,
        columns=["city", "service_name", "building_idx", "access_diagnosis_label"],
    )
    cities = [city for city in args.cities if city in set(diagnostics["city"])]
    if len(cities) != 20:
        raise ValueError(f"Expected 20 cities for 4x5 sheet, got {len(cities)}: {cities}")

    layers = _load_city_layers(args.joint_inputs_root, cities)
    for service in args.services:
        if service not in SERVICE_RU:
            raise ValueError(f"Unknown service: {service}")
        filename, _ = SERVICE_RU[service]
        render_service_sheet(
            diagnostics=diagnostics,
            layers=layers,
            service=service,
            cities=cities,
            out_path=args.out_dir / filename,
        )
        print(args.out_dir / filename)


if __name__ == "__main__":
    main()

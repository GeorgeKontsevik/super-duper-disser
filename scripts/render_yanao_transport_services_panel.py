from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "arctic_access" / "data" / "processed" / "yanao_kras"
BOUNDARY = DATA / "yanao_kras_admin_boundary.geojson"
OUT = ROOT / "itmo-phd-thesis-template-en" / "images" / "ch4" / "arctic" / "yanao_transport_services_panel.png"
TMP = ROOT / "tmp" / "yanao_transport_services_panel.png"

TRANSPORT_LAYERS = [
    ("круглогодичные дороги", "year_round_roads", "#EBCB8B"),
    ("авиация", ["plane_warm", "plane_cold"], "#BF616A"),
    ("водный транспорт", ["water_ship", "water_boat"], "#B48EAD"),
    ("зимники", ["winter_tr"], "#3469A2"),
    ("сезонные дороги", "seasonal_roads", "#008b8b"),
]

SERVICE_LAYERS = [
    ("культура", "culture", "#9b59b6", "o"),
    ("здравоохранение", "health", "#3498db", "o"),
    ("порт", "port", "#ffd700", "s"),
    ("аэропорт", "airport", "#00bcd4", "^"),
    ("малый порт", "marina", "#ff69b4", "o"),
]


def nonzero_lines(transport: gpd.GeoDataFrame, columns: list[str]) -> gpd.GeoDataFrame:
    mask = False
    for col in columns:
        values = transport[col].astype(str).str.replace(",", ".", regex=False)
        mask = mask | (values.astype(float) > 0)
    return transport.loc[mask].copy()


def positive(transport: gpd.GeoDataFrame, column: str):
    values = transport[column].astype(str).str.replace(",", ".", regex=False)
    return values.astype(float) > 0


def transport_layer(transport: gpd.GeoDataFrame, spec: list[str] | str) -> gpd.GeoDataFrame:
    if spec == "year_round_roads":
        return transport.loc[positive(transport, "car_warm") & positive(transport, "car_cold")].copy()
    if spec == "seasonal_roads":
        return transport.loc[
            positive(transport, "car_warm")
            & ~positive(transport, "car_cold")
            & ~positive(transport, "winter_tr")
        ].copy()
    return nonzero_lines(transport, spec)


def padded_bounds(*layers: gpd.GeoDataFrame, pad_ratio: float = 0.08) -> tuple[float, float, float, float]:
    minx = min(layer.total_bounds[0] for layer in layers)
    miny = min(layer.total_bounds[1] for layer in layers)
    maxx = max(layer.total_bounds[2] for layer in layers)
    maxy = max(layer.total_bounds[3] for layer in layers)
    pad = max(maxx - minx, maxy - miny) * pad_ratio
    return minx - pad, miny - pad, maxx + pad, maxy + pad


def draw_base(ax, settlements: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, bounds: tuple[float, float, float, float]):
    boundary.boundary.plot(ax=ax, color="#777777", linewidth=1.0, zorder=1)
    settlements.plot(ax=ax, color="#d8d8d8", markersize=8, alpha=0.75, zorder=2)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(0.9)


def main() -> None:
    settlements = gpd.read_file(DATA / "df_settlements_yanao_kras.geojson")
    transport = gpd.read_file(DATA / "df_time_yanao_kras.geojson")
    boundary = gpd.read_file(BOUNDARY).to_crs(settlements.crs)
    bounds = padded_bounds(settlements, transport)

    fig, axes = plt.subplots(2, 5, figsize=(18, 10.8))
    fig.suptitle(
        "ЯНАО — Красноярский край: транспортные связи, поселения и сервисы",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )
    for ax, (title, columns, color) in zip(axes[0], TRANSPORT_LAYERS):
        draw_base(ax, settlements, boundary, bounds)
        layer = transport_layer(transport, columns)
        layer.plot(ax=ax, color=color, linewidth=1.35, alpha=0.88, zorder=3)
        settlements.plot(ax=ax, color="#333333", markersize=9, alpha=0.9, zorder=4)
        handle = Line2D([], [], color=color, linewidth=2.2, label=title)
        ax.set_title(title, fontsize=16, fontweight="bold", color="#555555", pad=8)
        ax.legend(handles=[handle], loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=13, handlelength=1.2)

    for ax, (title, service, color, marker) in zip(axes[1], SERVICE_LAYERS):
        draw_base(ax, settlements, boundary, bounds)
        services = gpd.read_file(DATA / f"df_{service}_yanao_kras.geojson")
        capacity = services["capacity"].fillna(0).astype(float)
        providers = services.loc[capacity > 0].copy()
        nonproviders = services.loc[capacity <= 0].copy()
        nonproviders.plot(
            ax=ax,
            color="#111111",
            markersize=18,
            alpha=0.95,
            zorder=4,
        )
        providers.plot(
            ax=ax,
            color=color,
            marker=marker,
            markersize=48,
            edgecolor="#333333",
            linewidth=0.5,
            zorder=5,
        )
        ax.set_title(title, fontsize=16, fontweight="bold", color="#555555", pad=8)
        handle = Line2D(
            [],
            [],
            marker=marker,
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="#333333",
            markersize=9,
            label=title,
        )
        ax.legend(handles=[handle], loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=13, handlelength=1.0)

    plt.subplots_adjust(left=0.02, right=0.995, top=0.92, bottom=0.075, wspace=0.02, hspace=0.28)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    TMP.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    fig.savefig(TMP, dpi=220)
    plt.close(fig)
    for path in (OUT, TMP):
        assert path.exists() and path.stat().st_size > 50_000, path
        print(f"{path} | {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

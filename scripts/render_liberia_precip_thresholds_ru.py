from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "equatorial" / "outputs" / "road_weekly_scenarios" / "LBR" / "2024_full_year_db_cell_overlay_lbr" / "factor_boxplots_cell" / "weekly_factor_value_diagnostics.csv"
OUT = ROOT / "itmo-phd-thesis-template-en" / "images" / "ch4" / "lbr_precip_weekly_thresholds_ru.png"
TMP = ROOT / "tmp" / "lbr_precip_weekly_thresholds_ru.png"

MONTHS_RU = {
    1: "янв",
    2: "фев",
    3: "мар",
    4: "апр",
    5: "май",
    6: "июн",
    7: "июл",
    8: "авг",
    9: "сен",
    10: "окт",
    11: "ноя",
    12: "дек",
}


def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["week_start"])
    df = df[
        df["scenario"].eq("unknown_as_unpaved")
        & df["surface_scope"].eq("all")
        & df["factor"].eq("era5_tp_sum_weekly_mm")
    ].sort_values("week_start")

    stats = [
        {
            "label": row.week_start,
            "whislo": row.min_value,
            "q1": row.q25,
            "med": row.median,
            "q3": row.q75,
            "whishi": row.max_value,
            "fliers": [],
        }
        for row in df.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(22, 5.8))
    ax.bxp(
        stats,
        positions=range(len(stats)),
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#d9edf7", edgecolor="#6baed6", linewidth=1.0),
        whiskerprops=dict(color="#333333", linewidth=1.0),
        capprops=dict(color="#333333", linewidth=1.0),
        medianprops=dict(color="#f28e2b", linewidth=1.4),
    )

    paved = [50, 100, 200, 300]
    unpaved_unknown = [50, 100, 150, 250]
    for value in paved:
        ax.axhline(value, color="#4ea3ff", linestyle="--", linewidth=1.3, alpha=0.85)
    for value in unpaved_unknown:
        ax.axhline(value, color="#ff6b7a", linestyle=":", linewidth=1.4, alpha=0.9)

    tick_idx = list(range(0, len(df), 5))
    labels = []
    for i in tick_idx:
        day = df.iloc[i]["week_start"]
        labels.append(f"{day.day:02d} {MONTHS_RU[day.month]}")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_xlim(-1, len(df))
    ax.set_ylim(0, 500)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("мм/неделю", fontsize=17)
    ax.set_xlabel("начало недели", fontsize=17)
    ax.set_title("Драйвер осадков: недельная сумма ERA5, Либерия, все дороги, неизвестные как грунтовые", fontsize=18)
    ax.legend(
        handles=[
            Line2D([], [], color="#4ea3ff", linestyle="--", linewidth=1.7, label="с покрытием: 50, 100, 200, 300 мм/нед"),
            Line2D([], [], color="#ff6b7a", linestyle=":", linewidth=1.9, label="без покрытия + неизвестные: 50, 100, 150, 250 мм/нед"),
        ],
        title="использованные пороги",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fontsize=13,
        title_fontsize=13,
        frameon=True,
        ncol=2,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.87, bottom=0.30)

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

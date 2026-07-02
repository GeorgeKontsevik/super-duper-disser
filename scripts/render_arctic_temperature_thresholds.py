from __future__ import annotations

import ast
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCTIC = ROOT / "arctic_access"
sys.path.insert(0, str(ARCTIC))

from scripts.calculator.calculator_transport_prob import get_transport_probability
from scripts.plotter.plotter_transport_mode_prob import (
    calculate_probability_curves,
    find_all_threshold_intersections,
    generate_temperature_range,
    mode_label,
)
from scripts.preprocesser.constants import threshold, transport_modes, transport_modes_color


DATA = ARCTIC / "data" / "processed"
OUT_DIR = ROOT / "itmo-phd-thesis-template-en" / "images" / "ch4" / "arctic"
TMP_DIR = ROOT / "tmp"

MONTHS_RU = {
    "Jan": "янв",
    "Feb": "фев",
    "Mar": "мар",
    "Apr": "апр",
    "May": "май",
    "Jun": "июн",
    "Jul": "июл",
    "Aug": "авг",
    "Sep": "сен",
    "Oct": "окт",
    "Nov": "ноя",
    "Dec": "дек",
}


def load_weekly_temperatures(regions: list[str]) -> list[list[float]]:
    by_week: dict[int, list[float]] = defaultdict(list)
    for region in regions:
        climate_path = DATA / f"df_climate_{region}.csv"
        df = pd.read_csv(climate_path, usecols=["temperature"])
        for text in df["temperature"]:
            daily = ast.literal_eval(text)
            for date_key, value in daily.items():
                day = datetime.strptime(date_key, "%Y%m%d")
                week = min((day.timetuple().tm_yday - 1) // 7, 52)
                by_week[week].append(float(value))
    return [by_week[i] for i in range(53)]


def threshold_temperatures() -> dict[str, list[float]]:
    temps = generate_temperature_range(-70, 60, 2000)
    curves = calculate_probability_curves(temps, transport_modes, get_transport_probability)
    return {
        mode: [float(v) for v in values]
        for mode, values in find_all_threshold_intersections(temps, curves, threshold).items()
    }


def render(regions: list[str], title_suffix: str, output_name: str) -> Path:
    weekly = load_weekly_temperatures(regions)
    thresholds = threshold_temperatures()

    fig, ax = plt.subplots(figsize=(22, 5.8))
    positions = np.arange(53)
    ax.boxplot(
        weekly,
        positions=positions,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="#d9edf7", edgecolor="#6baed6", linewidth=1.0),
        whiskerprops=dict(color="#333333", linewidth=1.0),
        capprops=dict(color="#333333", linewidth=1.0),
        medianprops=dict(color="#f28e2b", linewidth=1.4),
    )

    used_labels: set[str] = set()
    for mode, values in thresholds.items():
        color = transport_modes_color.get(mode, "#666666")
        for temp in values:
            label = f"{mode_label(mode)}: {temp:.1f}°C"
            ax.axhline(
                temp,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label=label if label not in used_labels else None,
            )
            used_labels.add(label)

    week_starts = [datetime(2024, 1, 1) + timedelta(days=7 * i) for i in range(53)]
    tick_idx = list(range(0, 53, 5))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [f"{week_starts[i].day:02d} {MONTHS_RU[week_starts[i].strftime('%b')]}" for i in tick_idx],
        rotation=35,
        ha="right",
        fontsize=13,
    )
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(-1, 53)
    ax.set_ylim(-70, 52)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel("°C", fontsize=15)
    ax.set_xlabel("начало недели", fontsize=15)
    ax.set_title(
        f"Температурный драйвер: годовой ход температуры и пороги транспорта — {title_suffix}",
        fontsize=18,
    )
    ax.legend(
        title=f"пороги при p={threshold}",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        fontsize=13,
        title_fontsize=13,
        frameon=True,
        ncol=4,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.87, bottom=0.30)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / output_name
    fig.savefig(out, dpi=220)
    fig.savefig(TMP_DIR / output_name, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    outputs = [
        render(["yanao_kras"], "ЯНАО — Красноярский край", "arctic_temperature_thresholds_yanao_kras.png"),
    ]
    for path in outputs:
        assert path.exists() and path.stat().st_size > 50_000, path
        print(f"{path} | {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT / "aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"
HEAT_ROOT = ROOT / "thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/heat_joint_inputs"
OUT_DIR = ROOT / "thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/placement_tables_ru"
OUT_PATH = OUT_DIR / "polyclinic_heat_vs_baseline_compact_ru.png"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(34, True)
F_SUB = font(23, True)
F_HEAD = font(18, True)
F_TEXT = font(18)
F_NOTE = font(15)

BLACK = (28, 28, 28)
GRAY = (100, 100, 100)
GRID = (220, 224, 228)
HEAD = (242, 244, 247)
ACCENT = (236, 242, 248)
WHITE = (255, 255, 255)


CITY_RU = {
    "gothenburg_sweden": "Гётеборг",
    "graz_austria": "Грац",
    "hrodna_belarus": "Гродно",
    "innsbruck_austria": "Инсбрук",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows() -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    cities = ["gothenburg_sweden", "graz_austria", "hrodna_belarus", "innsbruck_austria"]
    rows1: list[list[str]] = []
    rows2: list[list[str]] = []
    rows3: list[list[str]] = []
    for city in cities:
        bs = load_json(BASE_ROOT / city / "pipeline_2/solver_inputs/polyclinic/summary.json")
        hs = load_json(HEAT_ROOT / city / "pipeline_2/solver_inputs/polyclinic/summary.json")
        bp = load_json(BASE_ROOT / city / "pipeline_2/placement_exact/polyclinic/summary_after.json")
        hp = load_json(HEAT_ROOT / city / "pipeline_2/placement_exact/polyclinic/summary_after.json")

        rows1.append(
            [
                CITY_RU[city],
                f"{int(bs['demand_without_total'])}",
                f"{int(hs['demand_without_total'])}",
                f"+{int(hs['demand_without_total'] - bs['demand_without_total'])}",
                f"{int(bp['selected_count'])}",
                f"{int(hp['selected_count'])}",
                f"+{int(hp['selected_count'] - bp['selected_count'])}",
                f"{int(bp['new_count'])}",
                f"{int(hp['new_count'])}",
                f"+{int(hp['new_count'] - bp['new_count'])}",
            ]
        )
        rows2.append(
            [
                CITY_RU[city],
                f"{bs['provision_total']:.3f}",
                f"{hs['provision_total']:.3f}",
                f"{hs['provision_total'] - bs['provision_total']:+.3f}",
                f"{int(bs['demand_within_total'])}",
                f"{int(hs['demand_within_total'])}",
                f"{int(hs['demand_within_total'] - bs['demand_within_total']):+d}",
            ]
        )
        rows3.append(
            [
                CITY_RU[city],
                f"{int(bp['capacity_added_total'])}",
                f"{int(hp['capacity_added_total'])}",
                f"{int(hp['capacity_added_total'] - bp['capacity_added_total']):+d}",
                f"{bp['provision_total_after']:.3f}",
                f"{hp['provision_total_after']:.3f}",
            ]
        )
    return rows1, rows2, rows3


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt, fill=BLACK):
    draw.text(xy, text, font=fnt, fill=fill)


def draw_table(
    draw: ImageDraw.ImageDraw,
    top: int,
    title: str,
    columns: list[tuple[str, int]],
    rows: list[list[str]],
) -> int:
    left = 36
    row_h = 44
    head_h = 46
    title_gap = 14
    draw_text(draw, (left, top), title, F_SUB)
    y = top + title_gap + 28

    xs = [left]
    for _, w in columns:
        xs.append(xs[-1] + w)

    draw.rounded_rectangle((left, y, xs[-1], y + head_h), radius=12, fill=HEAD, outline=GRID, width=1)
    for i, (label, _) in enumerate(columns):
        draw_text(draw, (xs[i] + 10, y + 12), label, F_HEAD)
    for x in xs[1:-1]:
        draw.line((x, y, x, y + head_h + row_h * len(rows)), fill=GRID, width=1)

    for r, row in enumerate(rows):
        y1 = y + head_h + r * row_h
        y2 = y1 + row_h
        fill = ACCENT if r % 2 == 0 else WHITE
        draw.rectangle((left, y1, xs[-1], y2), fill=fill, outline=GRID, width=1)
        for c, value in enumerate(row):
            draw_text(draw, (xs[c] + 10, y1 + 11), value, F_TEXT)

    return y + head_h + row_h * len(rows)


def main() -> None:
    rows1, rows2, rows3 = build_rows()
    width = 1660
    height = 980
    img = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(img)

    draw_text(draw, (36, 28), "Поликлиники: влияние heat на оптимальное размещение", F_TITLE)
    draw_text(draw, (36, 74), "Baseline vs heat, 4 города", F_NOTE, fill=GRAY)

    y = 122
    y = draw_table(
        draw,
        y,
        "1. Необеспеченный спрос и число выбранных точек",
        [
            ("Город", 180),
            ("Без heat", 120),
            ("С heat", 120),
            ("Δ спрос", 110),
            ("Выбр. без", 120),
            ("Выбр. с", 110),
            ("Δ", 70),
            ("Новых без", 120),
            ("Новых с", 110),
            ("Δ", 70),
        ],
        rows1,
    )
    y = draw_table(
        draw,
        y + 28,
        "2. Обеспеченность до размещения",
        [
            ("Город", 180),
            ("Prov. без", 130),
            ("Prov. с", 120),
            ("Δ", 90),
            ("Внутри 10 мин без", 170),
            ("Внутри 10 мин с", 160),
            ("Δ", 90),
        ],
        rows2,
    )
    y = draw_table(
        draw,
        y + 28,
        "3. Добавляемая мощность и итоговая обеспеченность после размещения",
        [
            ("Город", 180),
            ("Мощн. без", 150),
            ("Мощн. с", 140),
            ("Δ", 100),
            ("Prov. после без", 170),
            ("Prov. после с", 160),
        ],
        rows3,
    )

    draw_text(
        draw,
        (36, height - 34),
        "Heat применён только к пешеходным рёбрам; числа взяты из summary.json и summary_after.json.",
        F_NOTE,
        fill=GRAY,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()

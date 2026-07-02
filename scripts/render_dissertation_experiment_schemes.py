from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp"
OUT1 = OUT_DIR / "dissertation_position1_scheme.png"
OUT2 = OUT_DIR / "dissertation_position2_scheme.png"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(34, True)
F_LABEL = font(28, True)
F_SEG = font(24, True)
F_TEXT = font(20)
F_SMALL = font(18)
F_TINY = font(16)

BLACK = (24, 24, 24)
GRAY = (120, 120, 120)
LINE = (26, 107, 146)
PALE = (233, 244, 250)
PALE2 = (232, 246, 242)
GRID = (210, 219, 227)
RED = (196, 74, 74)
GREEN = (38, 143, 96)


def wrap(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fnt, fill=BLACK, line_gap: int = 4):
    words = text.split()
    line = ""
    start_y = y
    for word in words:
        cand = f"{line} {word}".strip()
        if line and draw.textlength(cand, font=fnt) > width:
            draw.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + line_gap
            line = word
        else:
            line = cand
    if line:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y - start_y


def round_box(draw: ImageDraw.ImageDraw, box, outline=GRID, fill="white", width=2, radius=18):
    draw.rounded_rectangle(box, radius=radius, outline=outline, fill=fill, width=width)


def down_arrow(draw: ImageDraw.ImageDraw, x: int, y_top: int, y_bottom: int, color=LINE, width: int = 7):
    draw.line((x, y_top, x, y_bottom - 26), fill=color, width=width)
    draw.polygon(
        [(x, y_bottom), (x - 24, y_bottom - 34), (x - 10, y_bottom - 34), (x - 10, y_bottom - 92),
         (x + 10, y_bottom - 92), (x + 10, y_bottom - 34), (x + 24, y_bottom - 34)],
        outline=color,
        fill="white",
    )


def h_arrow(draw: ImageDraw.ImageDraw, x1: int, y: int, x2: int, color=BLACK, width: int = 4, head: int = 14):
    draw.line((x1, y, x2, y), fill=color, width=width)
    draw.polygon([(x2, y), (x2 - head, y - 8), (x2 - head, y + 8)], fill=color)


def segment(draw: ImageDraw.ImageDraw, x1: int, x2: int, y: int, title: str, refs: str, fill):
    box = (x1, y - 58, x2, y + 18)
    round_box(draw, box, outline=LINE, fill=fill, width=3, radius=18)
    wrap(draw, title, x1 + 16, y - 44, x2 - x1 - 32, F_SEG)
    ref_w = draw.textlength(refs, font=F_TINY)
    draw.text(((x1 + x2 - ref_w) / 2, y + 28), refs, font=F_TINY, fill=GRAY)


def base_scheme(title: str, top_label: str, left_label: str, right_label: str, bottom_label: str, segments: list, arrows: list, accent_fill):
    w, h = 2300, 980
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    d.text((70, 36), title, font=F_TITLE, fill=BLACK)

    d.text((980, 108), top_label, font=F_LABEL, fill=BLACK)
    d.text((78, 500), left_label, font=font(40, True), fill=BLACK)
    right_x = w - 80 - d.textlength(right_label, font=font(40, True))
    d.text((right_x, 500), right_label, font=font(40, True), fill=BLACK)
    d.text((1080, 798), bottom_label, font=font(34, False), fill=BLACK)

    y_base = 610
    d.line((130, y_base, 2140, y_base), fill=LINE, width=8)

    for x, label in arrows:
        down_arrow(d, x, 132, 330)
        if label:
            tw = d.textlength(label, font=F_TEXT)
            d.text((x - tw / 2, 346), label, font=F_TEXT, fill=GRAY)

    # upper transport logic line
    upper_y = 556
    prev_end = 160
    for item in segments:
        x1, x2 = item["x1"], item["x2"]
        d.line((prev_end, upper_y, x1 - 16, upper_y), fill=LINE, width=6)
        segment(d, x1, x2, upper_y, item["title"], item["refs"], accent_fill)
        prev_end = x2 + 16
    d.line((prev_end, upper_y, 2140, upper_y), fill=LINE, width=6)

    for a in range(len(segments) - 1):
        h_arrow(d, segments[a]["x2"] + 10, upper_y - 20, segments[a + 1]["x1"] - 10)

    # bottom result row
    for item in segments:
        cx = (item["x1"] + item["x2"]) // 2
        round_box(d, (cx - 220, 696, cx + 220, 794), outline=GRID, fill="white", width=2, radius=16)
        wrap(d, item["effect"], cx - 194, 718, 388, F_SMALL, BLACK, 4)
        d.line((cx, upper_y + 22, cx, 692), fill=GRID, width=2)

    return img


position1_segments = [
    {
        "x1": 250,
        "x2": 700,
        "title": "Арктическая сезонная сеть",
        "refs": "[EFN / Arctic]",
        "effect": "Смена поставщика, временная изоляция, сезонные зоны обслуживания.",
    },
    {
        "x1": 860,
        "x2": 1340,
        "title": "Экваториальный климатический стресс",
        "refs": "[Equatorial case]",
        "effect": "Выявляются чувствительные связи и хрупкие маршруты.",
    },
    {
        "x1": 1520,
        "x2": 2010,
        "title": "Сезонная значимость опорных НП",
        "refs": "[Support settlements]",
        "effect": "Меняется ранжирование и сезонная значимость опорных точек.",
    },
]

position2_segments = [
    {
        "x1": 230,
        "x2": 700,
        "title": "Оптимизация матрицы доступности",
        "refs": "[GA + CLSCP-SO]",
        "effect": "Потребность в новых поликлиниках снижается с 13 до 9.",
    },
    {
        "x1": 850,
        "x2": 1335,
        "title": "Транспортная интервенция против дорожной",
        "refs": "[Telmana]",
        "effect": "Лучшие сценарии снижают потребность с 14 до 13 и 12.",
    },
    {
        "x1": 1490,
        "x2": 2020,
        "title": "Сервис-ориентированная генерация маршрутов",
        "refs": "[7 cities]",
        "effect": "В 6 из 7 городов нужна меньшая добавка новых объектов.",
    },
]


def main():
    img1 = base_scheme(
        title="Положение 1. Эксперименты по оценке сезонно-устойчивой транспортной доступности",
        top_label="Внешняя среда",
        left_label="А",
        right_label="Б",
        bottom_label="Сеть и доступность",
        segments=position1_segments,
        arrows=[(480, "температура"), (1100, "осадки / стресс"), (1760, "сезонная смена состояния сети")],
        accent_fill=PALE,
    )
    img2 = base_scheme(
        title="Положение 2. Эксперименты по оптимальному размещению с учетом изменения связности",
        top_label="Сетевые интервенции",
        left_label="Спрос",
        right_label="Сервис",
        bottom_label="Сеть, матрица доступности и размещение",
        segments=position2_segments,
        arrows=[(430, "улучшение связности"), (1080, "дорога / маршрут ОТ"), (1760, "целевая функция связности")],
        accent_fill=PALE2,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img1.save(OUT1)
    img2.save(OUT2)
    print(OUT1)
    print(OUT2)


if __name__ == "__main__":
    main()

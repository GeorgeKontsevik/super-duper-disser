from pathlib import Path
import csv

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "accessibility_literature_groups.csv"
OUT_DIR = ROOT / "itmo-phd-thesis-template-en" / "Dissertation"
OUT_CLASS = OUT_DIR / "accessibility_literature_classification.png"
OUT_MATRIX = OUT_DIR / "accessibility_intervention_matrix.png"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(34, True)
F_SUB = font(22, True)
F_TEXT = font(18)
F_SMALL = font(16)
F_TINY = font(14)

BLACK = (28, 28, 28)
GRAY = (120, 120, 120)
GRID = (224, 228, 232)
TEAL = (14, 139, 138)
TEAL_FILL = (227, 246, 245)
BLUE = (66, 117, 181)
BLUE_FILL = (228, 237, 251)
ORANGE = (228, 121, 52)
ORANGE_FILL = (255, 234, 221)
GREEN = (82, 152, 89)
GREEN_FILL = (225, 243, 226)
RED = (210, 90, 90)
RED_FILL = (253, 232, 232)
PURPLE = (126, 103, 183)
PURPLE_FILL = (236, 232, 250)
YELLOW_FILL = (255, 246, 210)


GROUP_META = {
    "A": {
        "title": "Базовая location-covering классика",
        "summary": "Размещение объектов при заданном пороге времени/дистанции покрытия.",
        "sources": "LSCP, MCLP, MEXCLP, MALP",
        "family": "Размещение объектов",
        "color": BLUE_FILL,
    },
    "B": {
        "title": "Размещение + проектирование сети",
        "summary": "Совместный выбор объектов, мощностей и сетевых связей.",
        "sources": "Melkote & Daskin; Pourrezaie-Khaligh; Starita",
        "family": "Совместное решение",
        "color": TEAL_FILL,
    },
    "C": {
        "title": "Улучшение ребер и сети",
        "summary": "Улучшение ребер сети как способ поднять покрытие и доступность.",
        "sources": "Murawski & Church; Baldomero-Naranjo; Akhlaghi",
        "family": "Совместное решение",
        "color": TEAL_FILL,
    },
    "D": {
        "title": "15-minute city и временная доступность",
        "summary": "Временная изменчивость доступности и 15-minute city как рамка.",
        "sources": "Willberg; Graells-Garrido; Wang et al.",
        "family": "Время и режим работы",
        "color": ORANGE_FILL,
    },
    "E": {
        "title": "Тепловая и теневая доступность",
        "summary": "Тепловой стресс, тень и маршрутная доступность в жаре.",
        "sources": "Wang & He; Wolf et al.",
        "family": "Среда и риски",
        "color": RED_FILL,
    },
    "F": {
        "title": "PT, расписания и надежность",
        "summary": "Расписания, headway и надежность времени поездки.",
        "sources": "Conway; Zang; ReVelle & Hogan",
        "family": "Время и режим работы",
        "color": ORANGE_FILL,
    },
    "G": {
        "title": "Велосипед + PT / multimodal",
        "summary": "Мультимодальная доступность на связке велосипед + PT.",
        "sources": "Geurs et al.; Rybels",
        "family": "Мультимодальность",
        "color": GREEN_FILL,
    },
    "H": {
        "title": "Паводки, сбои и hazard access",
        "summary": "Сбои, затопление и деградация проходимости сети.",
        "sources": "Pregnolato; Shahdani; Akhlaghi",
        "family": "Среда и риски",
        "color": RED_FILL,
    },
    "I": {
        "title": "Робастное и стохастическое покрытие",
        "summary": "Покрытие с вероятностью занятости, неопределенностью и regret.",
        "sources": "Daskin; ReVelle & Hogan; Lutter; Coco",
        "family": "Размещение объектов",
        "color": PURPLE_FILL,
    },
}


MATRIX_COLUMNS = [
    ("recurring", "Recurring\nvariability"),
    ("hazards", "Hazards /\nshocks"),
    ("degradation", "Structural\ndegradation"),
    ("routing", "Что меняется\nв графе / маршрутизации"),
    ("network", "Сетевой\nобъект анализа"),
    ("multi", "Мультимодальная\nинтеграция"),
]


MATRIX_MARKS = {
    "A": {"network"},
    "B": {"network"},
    "C": {"degradation", "network"},
    "D": {"recurring", "routing"},
    "E": {"recurring", "hazards", "degradation", "routing"},
    "F": {"recurring"},
    "G": {"multi", "network"},
    "H": {"hazards", "degradation", "routing"},
    "I": {"recurring"},
}


def wrap(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fnt, fill=BLACK, line_gap: int = 4):
    words = text.split()
    line = ""
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
    return y


def load_groups():
    groups = {}
    with DATA.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row["Группа"][0]
            groups.setdefault(key, []).append(row)
    return groups


def draw_classification(groups):
    w, h = 2200, 1440
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    d.text((55, 34), "Литература по сетевой доступности: классификация по исследовательским веткам", font=F_TITLE, fill=BLACK)
    d.text((58, 86), "Основано на группах A-I из собранной CSV; каждая ветка показывает, что именно меняется в постановке.", font=F_TEXT, fill=GRAY)

    families = [
        ("Размещение объектов", ["A", "I"], BLUE),
        ("Совместное решение", ["B", "C"], TEAL),
        ("Время и режим работы", ["D", "F"], ORANGE),
        ("Среда и риски", ["E", "H"], RED),
        ("Мультимодальность", ["G"], GREEN),
    ]

    top_y = 250
    family_w = 390
    gap = 28
    x_positions = [55 + i * (family_w + gap) for i in range(len(families))]

    trunk_x = w // 2
    d.rounded_rectangle((trunk_x - 250, 132, trunk_x + 250, 208), radius=18, fill=YELLOW_FILL, outline=BLACK, width=3)
    d.text((trunk_x - 222, 151), "Сетевая доступность в литературе", font=F_SUB, fill=BLACK)
    d.text((trunk_x - 222, 180), "от location-covering до heat, PT, flood и multimodal", font=F_SMALL, fill=GRAY)

    for fx in x_positions:
        d.line((trunk_x, 208, fx + family_w / 2, top_y), fill=GRID, width=4)

    for (family, keys, color), x in zip(families, x_positions):
        d.rounded_rectangle((x, top_y, x + family_w, top_y + 72), radius=16, fill="white", outline=color, width=4)
        d.text((x + 20, top_y + 18), family, font=F_SUB, fill=BLACK)
        y = top_y + 108
        for key in keys:
            meta = GROUP_META[key]
            d.rounded_rectangle((x, y, x + family_w, y + 210), radius=16, fill=meta["color"], outline=(170, 178, 186), width=2)
            d.text((x + 18, y + 14), f"{key}. {meta['title']}", font=F_SUB, fill=BLACK)
            yy = wrap(d, meta["summary"], x + 18, y + 54, family_w - 36, F_TEXT, BLACK, 4)
            d.text((x + 18, yy + 6), "Примеры:", font=F_SMALL, fill=GRAY)
            wrap(d, meta["sources"], x + 88, yy + 6, family_w - 106, F_SMALL, BLACK, 3)
            d.text((x + 18, y + 178), f"{len(groups.get(key, []))} источн.", font=F_TINY, fill=GRAY)
            y += 235

    OUT_CLASS.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_CLASS)


def draw_matrix(groups):
    w, h = 2460, 1060
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    d.text((55, 36), "Те же группы литературы: какие изменения сети и графа они рассматривают", font=F_TITLE, fill=BLACK)
    d.text((58, 86), "Без классификации по intervention и 15-minute city: только про тип изменчивости, деградации и сетевую логику.", font=F_TEXT, fill=GRAY)

    left = 55
    top = 165
    label_w = 560
    col_w = 255
    row_h = 76

    d.rounded_rectangle((left, top, left + label_w, top + 74), radius=14, fill=(246, 247, 249), outline=GRID, width=2)
    d.text((left + 18, top + 22), "Группа и основной смысл", font=F_SUB, fill=BLACK)

    for i, (_, title) in enumerate(MATRIX_COLUMNS):
        x = left + label_w + i * col_w
        d.rounded_rectangle((x, top, x + col_w - 12, top + 74), radius=14, fill=(246, 247, 249), outline=GRID, width=2)
        wrap(d, title, x + 16, top + 16, col_w - 44, F_SMALL, BLACK, 2)

    row_y = top + 96
    order = list("ABCDEFGHI")
    for idx, key in enumerate(order):
        meta = GROUP_META[key]
        bg = (252, 252, 252) if idx % 2 == 0 else (247, 249, 251)
        d.rounded_rectangle((left, row_y, left + label_w, row_y + row_h - 8), radius=12, fill=bg, outline=GRID, width=2)
        d.text((left + 16, row_y + 14), f"{key}. {meta['title']}", font=F_SMALL, fill=BLACK)
        wrap(d, meta["summary"], left + 16, row_y + 38, label_w - 32, F_TINY, GRAY, 2)

        for i, (col_key, _) in enumerate(MATRIX_COLUMNS):
            x = left + label_w + i * col_w
            d.rounded_rectangle((x, row_y, x + col_w - 12, row_y + row_h - 8), radius=12, fill=bg, outline=GRID, width=2)
            if col_key in MATRIX_MARKS[key]:
                cx = x + (col_w - 12) / 2
                cy = row_y + (row_h - 8) / 2
                fill = TEAL_FILL
                outline = TEAL
                d.rounded_rectangle((cx - 58, cy - 20, cx + 58, cy + 20), radius=12, fill=fill, outline=outline, width=3)
                d.text((cx - 22, cy - 12), "есть", font=F_TEXT, fill=BLACK)

        d.text((left + label_w - 90, row_y + 14), f"{len(groups.get(key, []))}", font=F_SMALL, fill=GRAY)
        row_y += row_h

    legend_y = row_y + 10
    d.rounded_rectangle((55, legend_y, 2405, legend_y + 54), radius=14, fill="white", outline=GRID, width=2)
    d.rounded_rectangle((78, legend_y + 16, 190, legend_y + 52), radius=10, fill=TEAL_FILL, outline=TEAL, width=3)
    d.text((115, legend_y + 23), "есть", font=F_SMALL, fill=BLACK)
    d.text((208, legend_y + 21), "эта группа явно работает с данным типом сетевого изменения или графовой логики", font=F_SMALL, fill=BLACK)

    OUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_MATRIX)


if __name__ == "__main__":
    groups = load_groups()
    draw_classification(groups)
    draw_matrix(groups)
    print(OUT_CLASS)
    print(OUT_MATRIX)

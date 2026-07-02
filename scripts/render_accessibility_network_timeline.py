from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "accessibility_network_timeline.png"

W, H = 3600, 900
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_YEAR = font(34)
F_LANE = font(34, True)
F_BOX = font(24, True)
F_SMALL = font(20)

black = (24, 24, 24)
gray = (145, 145, 145)
grid = (226, 226, 226)

left = 430
right = 3420
top = 78
lane_y = {
    "placement": 145,
    "variability": 310,
    "graph": 470,
}
lane_label_x = 80
lane_h = 100
box_h = 104

years = [
    ("1970-е", 520),
    ("1980-е", 900),
    ("1990-е", 1280),
    ("2001", 1660),
    ("2010-е", 2090),
    ("2020+", 2510),
    ("2024", 3000),
    ("2025-2026", 3330),
]


def wrap(text: str, x: int, y: int, width: int, fnt, fill=black, line_gap: int = 4):
    words = text.split()
    line = ""
    for word in words:
        cand = f"{line} {word}".strip()
        if line and d.textlength(cand, font=fnt) > width:
            d.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + line_gap
            line = word
        else:
            line = cand
    if line:
        d.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def lane_label(y: int, title: str):
    d.multiline_text((lane_label_x, y + 6), title.replace(" ", "\n", 1), font=F_LANE, fill=black, spacing=6)
    d.line((left - 40, y + lane_h // 2, right - 80, y + lane_h // 2), fill=grid, width=3)


def arrow(x1, y1, x2, y2, color=black, width=4):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    d.polygon([(x2, y2), (x2 - 18, y2 - 9), (x2 - 18, y2 + 9)], fill=color)


def box(x: int, y: int, w: int, title: str, body: str, refs: str = ""):
    d.rectangle((x, y, x + w, y + box_h), fill="white", outline=(70, 70, 70), width=2)
    yy = wrap(title, x + 18, y + 14, w - 36, F_BOX, black, 4)
    yy = wrap(body, x + 18, yy, w - 36, F_SMALL, gray, 3)
    if refs:
        d.text((x + 18, y + 78), refs, font=F_SMALL, fill=gray)


for label, x in years:
    d.text((x - 52, top), label, font=F_YEAR, fill=black)
    d.line((x, top + 46, x, 630), fill=(236, 236, 236), width=2)

lane_label(lane_y["placement"], "Размещение и покрытие")
lane_label(lane_y["variability"], "Вариативность доступности")
lane_label(lane_y["graph"], "Сеть и граф")

# Lane 1: placement / covering
box(470, 180, 420, "A. Классика covering", "порог времени/дистанции задает достижимость", "1971-2004")
box(980, 180, 420, "I. Стохастич. и робаст. покрытие", "вариативность входит как uncertainty", "1983-2020+")
box(1510, 180, 420, "B. Размещение + network design", "placement увязывают с links и сетью", "2001-2024")
box(2480, 180, 420, "C. Улучшение ребер сети", "improved links меняют покрытие", "2009-2023")

# Lane 2: variability
box(1040, 345, 420, "F. PT, расписания и надежность", "headway, delays и режим работы", "1989-2022")
box(2100, 345, 430, "D. Временная N-минутная доступность", "доступность меняется по времени и группам", "2021-2026")
box(2600, 345, 470, "E. Heat / shade access", "heat, shade и route costs меняют доступность", "2022-2025")

# Lane 3: network / graph
box(2100, 505, 410, "G. Bike + PT / multimodal", "граф достижения меняется через связку режимов", "2016-2026")
box(2600, 505, 470, "H. Hazard access / disruptions", "links и route choice меняются при shocks", "2022-2023")

# Arrows
arrow(890, lane_y["placement"] + 50, 980, lane_y["placement"] + 50)
arrow(1400, lane_y["placement"] + 50, 1510, lane_y["placement"] + 50)
arrow(1930, lane_y["placement"] + 50, 2480, lane_y["placement"] + 50)

arrow(1460, lane_y["variability"] + 50, 2100, lane_y["variability"] + 50)
arrow(2530, lane_y["variability"] + 50, 2600, lane_y["variability"] + 50)

arrow(2510, lane_y["graph"] + 50, 2600, lane_y["graph"] + 50)

# Bottom synthesis
final_x, final_y, final_w, final_h = 2280, 655, 980, 82
d.rounded_rectangle((final_x, final_y, final_x + final_w, final_y + final_h), radius=8, fill=(231, 247, 246), outline=(12, 142, 140), width=3)
d.text((final_x + 24, final_y + 12), "Общий вывод", font=F_BOX, fill=black)
d.text((final_x + 24, final_y + 48), "сначала влияния изучали по отдельности, затем сеть и граф стали частью самой постановки", font=F_SMALL, fill=gray)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "accessibility_bridge_scheme.png"

W, H = 2200, 1040
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(38, True)
F_H = font(28, True)
F_TEXT = font(22)
F_SMALL = font(18)
F_TINY = font(16)

BLACK = (28, 28, 28)
GRAY = (135, 135, 135)
GRID = (225, 228, 232)
TEAL = (13, 144, 143)
TEAL_FILL = (231, 247, 246)
ORANGE = (229, 121, 54)
ORANGE_FILL = (255, 236, 224)
RED = (222, 88, 88)
RED_FILL = (253, 234, 234)
LIGHT = (249, 250, 252)


def wrap(text: str, x: int, y: int, width: int, fnt, fill=BLACK, line_gap: int = 5):
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


def round_box(x1, y1, x2, y2, outline=GRID, fill="white", width=3, radius=18):
    d.rounded_rectangle((x1, y1, x2, y2), radius=radius, outline=outline, fill=fill, width=width)


def arrow(x1, y1, x2, y2, color=BLACK, width=4, head=18):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    if y2 >= y1:
        pts = [(x2, y2), (x2 - head, y2 - head), (x2 + head, y2 - head)]
    else:
        pts = [(x2, y2), (x2 - head, y2 + head), (x2 + head, y2 + head)]
    d.polygon(pts, fill=color)


def chip(x: int, y: int, text: str, outline, fill):
    w = int(d.textlength(text, font=F_TINY)) + 34
    round_box(x, y, x + w, y + 32, outline=outline, fill=fill, width=2, radius=10)
    d.text((x + 14, y + 7), text, font=F_TINY, fill=BLACK)
    return w


d.text((58, 38), "Почему нужен переход от разрозненных влияний к интегрированной задаче размещения", font=F_TITLE, fill=BLACK)
d.text((60, 92), "Промежуточный мост между управленческой идеей вариативной N-минутной доступности и модельной интегрированной постановкой.", font=F_TEXT, fill=GRAY)

# Outer frame.
frame = (34, 140, 2160, 990)
for x in range(frame[0], frame[2], 30):
    d.line((x, frame[1], min(x + 16, frame[2]), frame[1]), fill=(165, 165, 165), width=4)
    d.line((x, frame[3], min(x + 16, frame[2]), frame[3]), fill=(165, 165, 165), width=4)
for y in range(frame[1], frame[3], 30):
    d.line((frame[0], y, frame[0], min(y + 16, frame[3])), fill=(165, 165, 165), width=4)
    d.line((frame[2], y, frame[2], min(y + 16, frame[3])), fill=(165, 165, 165), width=4)

# Title band.
round_box(560, 160, 1635, 238, outline=TEAL, fill="white", width=4, radius=18)
d.text((600, 184), "Нестатичность доступности известна, но в литературе чаще разобрана по частям", font=F_H, fill=BLACK)

col_y = 290
col_h = 470
col_w = 620
gap = 70
x1 = 80
x2 = x1 + col_w + gap
x3 = x2 + col_w + gap

cols = [
    (x1, "Что меняет доступность", TEAL, TEAL_FILL),
    (x2, "Как это обычно изучают", ORANGE, ORANGE_FILL),
    (x3, "Почему этого мало для размещения", RED, RED_FILL),
]

for x, title, color, fill in cols:
    round_box(x, col_y, x + col_w, col_y + col_h, outline=color, fill="white", width=4, radius=18)
    round_box(x + 16, col_y + 16, x + col_w - 16, col_y + 74, outline=color, fill=fill, width=2, radius=14)
    d.text((x + 24, col_y + 32), title, font=F_H, fill=BLACK)

# Column 1 content.
y = col_y + 105
items1 = [
    ("Hazards / disruptions", "паводки, closures, shocks"),
    ("Деградация сети", "плохая дорога, снижение скорости, chronic unreliability"),
    ("Heat / shade / seasonality", "жара, тень, сезонная проходимость"),
    ("PT reliability / schedule", "headway, задержки, режим работы"),
    ("Multimodal use", "как реально комбинируют walking + PT"),
]
for title, body in items1:
    round_box(x1 + 18, y, x1 + col_w - 18, y + 64, outline=GRID, fill=LIGHT, width=2, radius=12)
    d.text((x1 + 34, y + 12), title, font=F_TEXT, fill=BLACK)
    d.text((x1 + 34, y + 38), body, font=F_TINY, fill=GRAY)
    y += 78

# Column 2 content.
y = col_y + 105
items2 = [
    ("Отдельные кейсы", "каждый фактор рассматривают как самостоятельный источник вариативности"),
    ("Сценарный фон", "изменение сети учитывают как внешний сценарий, а не часть placement-задачи"),
    ("Поправка к маршруту", "меняется travel time, path choice или comfort, но не сама логика размещения"),
    ("Частная метрика", "оценивают один режим: heat, reliability, flood, walking"),
]
for title, body in items2:
    round_box(x2 + 18, y, x2 + col_w - 18, y + 84, outline=GRID, fill=LIGHT, width=2, radius=12)
    d.text((x2 + 34, y + 12), title, font=F_TEXT, fill=BLACK)
    wrap(body, x2 + 34, y + 38, col_w - 86, F_TINY, GRAY, 3)
    y += 98

yy = y + 6
d.text((x2 + 24, yy), "Общий паттерн:", font=F_SMALL, fill=BLACK)
wrap("доступность признается вариативной, но влияния чаще остаются фрагментированными.", x2 + 170, yy, col_w - 195, F_SMALL, GRAY, 3)

# Column 3 content.
y = col_y + 105
items3 = [
    ("Размещать все равно нужно", "управленческое решение требует выбрать что и где открывать / усиливать"),
    ("Но размещение реализуется через сеть", "достижимость объекта задается маршрутом, PT и состоянием links"),
    ("Одна и та же точка может быть по-разному доступна", "в зависимости от жары, PT, деградации, shocks и route choice"),
    ("Значит, нельзя решать по отдельности", "нужно совместно учитывать placement и изменяющуюся сеть"),
]
for i, (title, body) in enumerate(items3):
    box_fill = RED_FILL if i == len(items3) - 1 else LIGHT
    box_outline = RED if i == len(items3) - 1 else GRID
    round_box(x3 + 18, y, x3 + col_w - 18, y + 92, outline=box_outline, fill=box_fill, width=2, radius=12)
    d.text((x3 + 34, y + 12), title, font=F_TEXT, fill=BLACK)
    wrap(body, x3 + 34, y + 38, col_w - 86, F_TINY, GRAY if i < len(items3) - 1 else BLACK, 3)
    y += 106

# Horizontal arrows between columns.
mid_y = col_y + 215
arrow(x1 + col_w + 10, mid_y, x2 - 18, mid_y, color=BLACK, width=4, head=14)
arrow(x2 + col_w + 10, mid_y, x3 - 18, mid_y, color=BLACK, width=4, head=14)

# Bottom synthesis.
arrow(W // 2, 780, W // 2, 840, color=TEAL, width=5, head=18)
round_box(410, 848, 1790, 955, outline=TEAL, fill=TEAL_FILL, width=4, radius=18)
d.text((450, 874), "Вывод: если цель — не просто измерить, а обеспечить N-минутную доступность,", font=F_H, fill=BLACK)
d.text((450, 910), "то нужна интегрированная постановка: размещение объектов + изменяющаяся сеть + сценарии ее функционирования.", font=F_TEXT, fill=BLACK)

# Accent chips.
chip(1530, 46, "мост к нижнему блоку", TEAL, TEAL_FILL)
chip(1720, 46, "не про отдельный фактор", ORANGE, ORANGE_FILL)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)

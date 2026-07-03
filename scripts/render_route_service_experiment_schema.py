from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "route_service_experiment_schema.png"
W, H = 2580, 735

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(48, True)
F_H = font(31, True)
F = font(24)
F_SMALL = font(21)
F_LEGEND = font(28)

BLACK = (25, 25, 25)
GRAY = (150, 150, 150)
PANEL = (205, 205, 205)
RED = (220, 95, 100)
RED_FILL = (255, 212, 212)
GREEN = (78, 165, 92)
GREEN_FILL = (210, 238, 210)
PURPLE = (165, 105, 190)
TEAL = (0, 140, 137)
OCHRE = (230, 184, 92)

BASE_NODES = {
    "a": (0.08, 0.34),
    "b": (0.25, 0.12),
    "c": (0.29, 0.49),
    "d": (0.48, 0.30),
    "e": (0.51, 0.66),
    "f": (0.70, 0.12),
    "g": (0.72, 0.47),
    "h": (0.88, 0.29),
    "i": (0.91, 0.70),
}
EDGES = [("a", "c"), ("b", "c"), ("b", "d"), ("c", "d"), ("c", "e"),
         ("d", "f"), ("d", "g"), ("e", "g"), ("e", "i"), ("f", "h"),
         ("g", "h"), ("g", "i")]
KINDS = {"a": "demand", "c": "demand", "e": "service", "g": "service"}


def text_center(x, y, text, fnt, fill=BLACK):
    box = d.textbbox((0, 0), text, font=fnt)
    d.text((x - (box[2] - box[0]) / 2, y), text, font=fnt, fill=fill)


def panel(x, y, w, h, title):
    d.rounded_rectangle((x, y, x + w, y + h), radius=16, fill="white", outline=PANEL, width=3)
    d.multiline_text((x + 24, y + 22), title, font=F_H, fill=BLACK, spacing=2)


def dashed_line(a, b, fill, width=4, dash=12, gap=8):
    ax, ay = a
    bx, by = b
    length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    if not length:
        return
    ux, uy = (bx - ax) / length, (by - ay) / length
    pos = 0
    while pos < length:
        end = min(pos + dash, length)
        d.line((ax + ux * pos, ay + uy * pos, ax + ux * end, ay + uy * end), fill=fill, width=width)
        pos += dash + gap


def node(p, kind="plain", r=14):
    fill, outline = {
        "plain": ("white", BLACK),
        "demand": (RED_FILL, RED),
        "service": (GREEN_FILL, GREEN),
        "candidate": ("white", GREEN),
    }[kind]
    box = (p[0] - r, p[1] - r, p[0] + r, p[1] + r)
    if kind == "candidate":
        d.ellipse(box, fill=fill)
        for start in range(0, 360, 35):
            d.arc(box, start, start + 20, fill=outline, width=3)
    else:
        d.ellipse(box, fill=fill, outline=outline, width=2)


def points(box):
    x, y, w, h = box
    return {name: (x + px * w, y + py * h) for name, (px, py) in BASE_NODES.items()}


def draw_graph(box, synthetic=(), new_road=(), route=(), candidates=()):
    p = points(box)
    for u, v in EDGES:
        d.line((*p[u], *p[v]), fill=GRAY, width=3)
    for u, v in synthetic:
        d.line((*p[u], *p[v]), fill=TEAL, width=7)
    for u, v in new_road:
        dashed_line(p[u], p[v], OCHRE, width=7)
    for u, v in route:
        d.line((*p[u], *p[v]), fill=PURPLE, width=8)
    for name, pos in p.items():
        node(pos, "candidate" if name in candidates else KINDS.get(name, "plain"))
    return p


def additional_services(x, y, count):
    for i in range(count):
        node((x + i * 48, y), "candidate", r=17)


def arrow(x1, x2, y):
    d.line((x1, y, x2 - 18, y), fill=GRAY, width=9)
    d.polygon([(x2, y), (x2 - 24, y - 16), (x2 - 24, y + 16)], fill=GRAY)


def legend_node(x, y, kind, label):
    node((x, y), kind, r=17)
    d.text((x + 28, y - 13), label, font=F_LEGEND, fill=BLACK)


def legend_line(x, y, color, label, dashed=False):
    (dashed_line if dashed else d.line)((x, y, x + 66, y) if not dashed else (x, y),
                                         fill=color, width=7) if not dashed else dashed_line((x, y), (x + 66, y), color, 7)
    d.text((x + 82, y - 13), label, font=F_LEGEND, fill=BLACK)


d.text((45, 35), "Может ли улучшение маршрутов сократить число новых объектов?", font=F_TITLE, fill=BLACK)

Y, PH = 120, 480
panels = [(45, 430), (500, 535), (1060, 650), (1735, 800)]
titles = [
    "1. Базовый результат",
    "2. Синтетическое улучшение\n    связности",
    "3. Дорога или маршрут?",
    "4. Куда направлять новый маршрут?",
]
for (x, w), title in zip(panels, titles):
    panel(x, Y, w, PH, title)

# 1. Baseline.
draw_graph((85, 235, 350, 350), candidates=("b", "f", "h", "i"))
additional_services(165, 565, 4)

# 2. Synthetic links.
text_center(630, 225, "исходная сеть", F)
draw_graph((535, 270, 190, 230), candidates=("b", "f", "h", "i"))
additional_services(550, 565, 4)
arrow(735, 790, 390)
text_center(895, 225, "связность кварталов ↑", F, TEAL)
draw_graph((800, 270, 190, 230), synthetic=(("a", "d"), ("c", "g"), ("e", "h")), candidates=("h", "i"))
additional_services(790, 565, 2)

# 3. Road versus route.
d.line((1385, 190, 1385, 580), fill=PANEL, width=2)
text_center(1225, 188, "новая дорога", F)
draw_graph((1090, 245, 270, 270), new_road=(("c", "g"),), candidates=("f", "h", "i"))
additional_services(1145, 565, 3)
text_center(1545, 188, "новый маршрут", F)
draw_graph((1410, 245, 270, 270), route=(("a", "c"), ("c", "e"), ("e", "g"), ("g", "h")), candidates=("h", "i"))
additional_services(1475, 565, 2)

# 4. Route objective.
d.line((2135, 190, 2135, 580), fill=PANEL, width=2)
text_center(1935, 184, "общая связность", F)
draw_graph((1765, 245, 330, 300), route=(("a", "c"), ("c", "d"), ("d", "f"), ("d", "g")), candidates=("h", "i"))
additional_services(1875, 565, 2)
text_center(2345, 174, "к сервису /\nпотенциальному сервису", F)
draw_graph((2170, 245, 330, 300), route=(("a", "c"), ("c", "e"), ("e", "g"), ("g", "h"), ("g", "i")), candidates=("h",))
additional_services(2305, 565, 1)

# Legend.
legend_node(55, 665, "demand", "квартал со спросом")
legend_node(390, 665, "service", "существующий сервис")
legend_node(750, 665, "candidate", "кандидат на новый сервис")
legend_line(1210, 665, GRAY, "дорога")
legend_line(1450, 665, OCHRE, "новая дорога", dashed=True)
legend_line(1775, 665, PURPLE, "новый маршрут")
legend_line(2115, 665, TEAL, "синтетическая связь")

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)

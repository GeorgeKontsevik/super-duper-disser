from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "part1_multilayer_task_comparison.png"

W, H = 1800, 1180
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(42, True)
F_H = font(30, True)
F = font(18)
FB = font(18, True)
FL = font(20, True)

black = (25, 25, 25)
road_plane = (246, 251, 255)
env_plane = (255, 248, 222)
red = (232, 94, 98)
red_fill = (255, 205, 205)
green = (82, 168, 91)
green_fill = (205, 236, 205)
blue = (72, 134, 205)
blue_fill = (205, 226, 250)
orange = (242, 137, 48)
teal = (0, 140, 137)
purple = (164, 98, 205)
fail = (220, 95, 95)
dash = (185, 185, 185)

edges = [
    ("o", "b"),
    ("o", "a"),
    ("a", "d"),
    ("b", "c"),
    ("c", "f"),
    ("c", "e"),
    ("d", "e"),
    ("e", "g"),
]

pts_rel = {
    "o": (0, 0),
    "a": (20, 30),
    "b": (70, -20),
    "c": (125, -8),
    "d": (92, 22),
    "e": (158, 18),
    "f": (195, -25),
    "g": (208, 35),
}


def pts_at(cx, cy):
    return {k: (cx + x, cy + y) for k, (x, y) in pts_rel.items()}


def label(x, y, text):
    bb = d.textbbox((0, 0), text, font=FB)
    pad = 7
    d.rounded_rectangle(
        (x, y, x + bb[2] + pad * 2, y + bb[3] + pad * 2),
        6,
        fill="white",
        outline=(185, 195, 205),
        width=2,
    )
    d.text((x + pad, y + pad - 1), text, font=FB, fill=black)


def plane(x, y, w=540, h=58, depth=55, fill=road_plane):
    d.polygon(
        [(x, y + h), (x + depth, y), (x + w, y), (x + w - depth, y + h)],
        fill=fill,
        outline=(190, 210, 230),
    )


def env_layers(x, y, w=540):
    for i in reversed(range(4)):
        plane(x + 18 * i, y + 22 * i, w, 58, 55, env_plane)
    label(x + w + 20, y + 90, "Слои среды")


def line(a, b, color=black, width=2, dashed=False):
    if not dashed:
        d.line((*a, *b), fill=color, width=width)
        return

    ax, ay = a
    bx, by = b
    n = max(1, int(math.hypot(bx - ax, by - ay) // 18))
    for i in range(n):
        if i % 2 == 0:
            t1 = i / n
            t2 = (i + 1) / n
            d.line(
                (
                    ax + (bx - ax) * t1,
                    ay + (by - ay) * t1,
                    ax + (bx - ax) * t2,
                    ay + (by - ay) * t2,
                ),
                fill=color,
                width=width,
            )


def xmark(p):
    x, y = p
    d.line((x - 6, y - 6, x + 6, y + 6), fill=fail, width=3)
    d.line((x - 6, y + 6, x + 6, y - 6), fill=fail, width=3)


def node(p, kind="white", r=10):
    fill = {
        "red": red_fill,
        "green": green_fill,
        "blue": blue_fill,
        "white": "white",
    }[kind]
    outline = {"red": red, "green": green, "blue": blue, "white": black}[kind]
    d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=fill, outline=outline, width=2)


def redraw(points, demand=("o",), supply=("a", "c", "d"), mixed=("e",)):
    for k in demand:
        node(points[k], "red")
    for k in supply:
        node(points[k], "green")
    for k in mixed:
        node(points[k], "blue")
    for k in points:
        if k not in demand and k not in supply and k not in mixed:
            node(points[k], "white")


def road_graph(cx, cy, failures=(), demand=("o",), supply=("a", "c", "d"), mixed=("e",)):
    points = pts_at(cx, cy)
    failed = {tuple(sorted(x)) for x in failures}
    for u, v in edges:
        bad = tuple(sorted((u, v))) in failed
        line(points[u], points[v], dash if bad else black, 3 if bad else 2, bad)
        if bad:
            xmark(((points[u][0] + points[v][0]) // 2, (points[u][1] + points[v][1]) // 2))
    redraw(points, demand, supply, mixed)
    return points


def arrow(points, color, width=4):
    for a, b in zip(points, points[1:]):
        line(a, b, color, width)
    a, b = points[-2], points[-1]
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    size = 12
    d.polygon(
        [
            b,
            (b[0] - size * math.cos(ang - 0.45), b[1] - size * math.sin(ang - 0.45)),
            (b[0] - size * math.cos(ang + 0.45), b[1] - size * math.sin(ang + 0.45)),
        ],
        fill=color,
    )


def transport_path(points, path, color, broken_segment=None):
    broken = tuple(sorted(broken_segment)) if broken_segment else None
    for u, v in zip(path, path[1:]):
        bad = tuple(sorted((u, v))) == broken
        if bad:
            line(points[u], points[v], dash, 5, True)
            xmark(((points[u][0] + points[v][0]) // 2, (points[u][1] + points[v][1]) // 2))
        else:
            arrow([(points[u][0], points[u][1] - 1), (points[v][0], points[v][1] - 1)], color, 5)


def road_layer(x, y, text, demand=("o",), supply=("a", "c", "d"), mixed=("e",)):
    plane(x, y)
    road_graph(x + 110, y + 4, (), demand, supply, mixed)
    label(x + 455, y + 16, text)


def transport_on_road(x, y, text, paths):
    plane(x, y)
    points = road_graph(x + 110, y + 4, ())
    for path, color in paths:
        transport_path(points, path, color)
    redraw(points)
    label(x + 455, y + 16, text)


def logistic_layer(y, text, color, path, demand=("o",)):
    plane(85, y)
    points = road_graph(
        195,
        y + 4,
        (("c", "e"), ("e", "g")),
        demand=demand,
        supply=("a", "c", "d", "e"),
        mixed=(),
    )
    arrow([(points[k][0], points[k][1] - 1) for k in path], color, 4)
    redraw(points, demand=demand, supply=("a", "c", "d", "e"), mixed=())
    label(600, y + 16, text)


def service_layer(y, text, color, path, broken_segment):
    plane(940, y)
    points = road_graph(1050, y + 4, ())
    transport_path(points, path, color, broken_segment)
    redraw(points)
    label(1405, y + 16, text)


d.text((45, 36), "Мультиуровневость в двух постановках", font=F_TITLE, fill=black)
d.text((80, 130), "Логистическая задача", font=F_H, fill=black)
d.text((905, 130), "Сервисно-транспортная задача", font=F_H, fill=black)

road_layer(85, 220, "исходный дорожный слой", demand=("o", "b"), supply=("a", "c", "d", "e"), mixed=())

road_layer(940, 205, "исходный дорожный слой")
transport_on_road(
    940,
    310,
    "исходный транспортный слой",
    [(["o", "b", "c"], orange), (["o", "a", "d"], teal), (["o", "a", "d", "e"], purple)],
)

env_layers(85, 390, 520)
env_layers(940, 430, 520)

logistic_layer(660, "origin class 1", red, ["o", "a"], ("o",))
logistic_layer(795, "origin class 2", green, ["b", "c"], ("b",))
logistic_layer(930, "origin class 3", blue, ["o", "a", "d"], ("o",))

service_layer(700, "destination class 1", orange, ["o", "b", "c"], ("b", "c"))
service_layer(835, "destination class 2", teal, ["o", "a", "d"], ("a", "d"))
service_layer(970, "destination class 3", purple, ["o", "a", "d", "e"], ("o", "a"))

d.text((95, 1045), "Легенда", font=FL, fill=black)


def legend_node(x, y, kind, text):
    node((x, y), kind, 10)
    d.text((x + 22, y - 10), text, font=F, fill=black)


legend_node(95, 1085, "red", "спрос / origin")
legend_node(325, 1085, "green", "предложение / destination")
legend_node(610, 1085, "blue", "mixed / transit")

arrow([(95, 1130), (145, 1130)], red, 4)
d.text((162, 1120), "поток спроса к предложению / слой класса", font=F, fill=black)
d.line((560, 1130, 620, 1130), fill=orange, width=5)
d.text((638, 1120), "транспортная OD-связь / слой", font=F, fill=black)
line((900, 1130), (960, 1130), dash, 4, True)
d.text((978, 1120), "недоступная связь", font=F, fill=black)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)

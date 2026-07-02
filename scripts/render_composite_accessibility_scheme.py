from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "composite_accessibility_external_environment.png"

W, H = 2400, 1420
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(43, True)
F_H = font(27, True)
F_TEXT = font(22)
F_SMALL = font(18)
F_TINY = font(16)

BLACK = (28, 28, 28)
GRAY = (120, 124, 130)
GRID = (212, 216, 221)
LIGHT = (249, 250, 251)
RED = (225, 94, 101)
RED_FILL = (255, 235, 236)
GREEN = (70, 167, 117)
GREEN_FILL = (231, 247, 237)
BLUE = (83, 142, 202)
BLUE_FILL = (234, 243, 252)
TEAL = (13, 145, 143)
TEAL_FILL = (230, 247, 246)
ORANGE = (225, 161, 61)
ORANGE_FILL = (255, 245, 224)
PURPLE = (141, 102, 190)


def rbox(box, outline=GRID, fill="white", width=3, radius=18):
    d.rounded_rectangle(box, radius=radius, outline=outline, fill=fill, width=width)


def centered(text: str, box, fnt, fill=BLACK, spacing=5):
    left, top, right, bottom = box
    lines = text.split("\n")
    heights = [d.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total = sum(heights) + spacing * (len(lines) - 1)
    y = top + (bottom - top - total) / 2
    for line, height in zip(lines, heights):
        width = d.textlength(line, font=fnt)
        d.text((left + (right - left - width) / 2, y), line, font=fnt, fill=fill)
        y += height + spacing


def wrap(text: str, x: int, y: int, width: int, fnt, fill=BLACK, gap=4):
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and d.textlength(candidate, font=fnt) > width:
            d.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + gap
            line = word
        else:
            line = candidate
    if line:
        d.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + gap
    return y


def arrow(x1, y1, x2, y2, color=BLACK, width=5, head=18):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 >= x1 else -1
        d.polygon([(x2, y2), (x2 - sign * head, y2 - head // 2), (x2 - sign * head, y2 + head // 2)], fill=color)
    else:
        sign = 1 if y2 >= y1 else -1
        d.polygon([(x2, y2), (x2 - head // 2, y2 - sign * head), (x2 + head // 2, y2 - sign * head)], fill=color)


def node(cx: int, cy: int, radius: int, outline, fill, title: str, subtitle: str):
    d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=outline, fill=fill, width=5)
    centered(title, (cx - radius, cy - radius + 12, cx + radius, cy + 7), F_H)
    centered(subtitle, (cx - radius + 6, cy + 8, cx + radius - 6, cy + radius - 8), F_TINY, GRAY, 2)


def segment(box, title: str, subtitle: str, outline, fill):
    rbox(box, outline=outline, fill=fill, width=4, radius=16)
    centered(title, (box[0], box[1] + 10, box[2], box[1] + 53), F_H)
    centered(subtitle, (box[0] + 8, box[1] + 53, box[2] - 8, box[3] - 8), F_TINY, GRAY, 2)


d.text((55, 35), "Составная доступность в изменяющейся внешней среде", font=F_TITLE, fill=BLACK)
d.text((57, 91), "Что формирует путь от спроса к сервису, что меняет этот путь и что проверяют эксперименты", font=F_TEXT, fill=GRAY)

# External environment.
rbox((55, 145, 2345, 310), outline=RED, fill=RED_FILL, width=4, radius=22)
d.text((88, 169), "ВНЕШНЯЯ СРЕДА", font=F_H, fill=RED)
chips = [("осадки", 620), ("температура", 925), ("жара", 1270)]
for label, x in chips:
    rbox((x, 166, x + 260, 222), outline=RED, fill="white", width=2, radius=14)
    centered(label, (x, 166, x + 260, 222), F_TEXT)
centered(
    "меняет проходимость дорог, стоимость движения и конфигурацию транспортных маршрутов",
    (260, 238, 2140, 291),
    F_TEXT,
)
for x in (420, 910, 1455, 1980):
    arrow(x, 311, x, 365, color=RED, width=4, head=16)

# Composite route.
path_y = 495
node(135, path_y, 72, RED, RED_FILL, "A", "спрос\nпостоянный")
segment((265, 425, 535, 565), "пешком", "по дорожной сети", ORANGE, ORANGE_FILL)
node(650, path_y, 68, GREEN, GREEN_FILL, "О", "остановка")
segment((770, 415, 1190, 575), "общественный транспорт", "маршрут проходит\nпо дорожной сети", TEAL, TEAL_FILL)
node(1310, path_y, 68, GREEN, GREEN_FILL, "О", "остановка")
segment((1425, 425, 1695, 565), "пешком", "по дорожной сети", ORANGE, ORANGE_FILL)
node(1830, path_y, 72, BLUE, BLUE_FILL, "B", "существующий\nсервис")

for x1, x2 in ((207, 260), (540, 578), (718, 765), (1195, 1238), (1378, 1420), (1700, 1758)):
    arrow(x1, path_y, x2, path_y, width=4, head=15)

rbox((1935, 405, 2345, 585), outline=PURPLE, fill="white", width=4, radius=18)
centered("ДОСТУПНОСТЬ", (1960, 425, 2320, 474), F_H, PURPLE)
centered("подход к остановке\n+ поездка\n+ путь до сервиса", (1970, 480, 2310, 565), F_SMALL, BLACK, 4)
arrow(1905, path_y, 1930, path_y, color=PURPLE, width=4, head=14)

centered(
    "Стохастическая доступность зависит от положения A и B в сети и от состояния соединяющего их пути",
    (175, 610, 2225, 670),
    F_H,
)

# Fixed and controlled parts.
controls = [
    (55, 720, 735, 870, RED, RED_FILL, "Фиксируем", "величину спроса"),
    (860, 720, 1540, 870, TEAL, TEAL_FILL, "Меняем связность", "новая дорога · новый маршрут"),
    (1665, 720, 2345, 870, BLUE, BLUE_FILL, "Приближаем предложение", "размещение нового сервиса ближе к спросу"),
]
for x1, y1, x2, y2, color, fill, title, body in controls:
    rbox((x1, y1, x2, y2), outline=color, fill=fill, width=4, radius=18)
    centered(title, (x1 + 15, y1 + 16, x2 - 15, y1 + 62), F_H, color)
    centered(body, (x1 + 24, y1 + 70, x2 - 24, y2 - 18), F_TEXT)

# Experiments.
d.text((55, 924), "КАК ЭТО ПРОВЕРЯЕТСЯ В ЭКСПЕРИМЕНТАХ", font=F_H, fill=BLACK)
cards = [
    (55, 980, 585, 1325, ORANGE, ORANGE_FILL, "ЭКВАТОР", "Внешний фактор", "осадки", "Что меняется", "дороги деградируют", "Результат", "меняется доступность"),
    (640, 980, 1170, 1325, BLUE, BLUE_FILL, "АРКТИКА", "Внешний фактор", "температура", "Что меняется", "маршруты реконфигурируются", "Результат", "меняются потоки к сервисам"),
    (1225, 980, 1755, 1325, TEAL, TEAL_FILL, "ТЕЛЬМАНА · РЕАЛЬНЫЙ СОВМЕСТНЫЙ КЕЙС", "Изменение территории", "проект новой дороги", "Совместно", "дорога · маршрут · сервисы", "Проверка", "способы улучшить доступность"),
    (1810, 980, 2345, 1325, RED, RED_FILL, "ЖАРА · РЕАЛЬНЫЙ СОВМЕСТНЫЙ КЕЙС", "Внешний фактор", "тепловая нагрузка", "Совместно", "пешеходный путь · сервисы", "Проверка", "доступность существующих сервисов"),
]
for x1, y1, x2, y2, color, fill, title, k1, v1, k2, v2, k3, v3 in cards:
    rbox((x1, y1, x2, y2), outline=color, fill="white", width=4, radius=18)
    rbox((x1 + 12, y1 + 12, x2 - 12, y1 + 79), outline=color, fill=fill, width=2, radius=13)
    centered(title, (x1 + 20, y1 + 17, x2 - 20, y1 + 74), F_SMALL, BLACK, 2)
    y = y1 + 103
    for key, value in ((k1, v1), (k2, v2), (k3, v3)):
        d.text((x1 + 28, y), key, font=F_TINY, fill=color)
        y = wrap(value, x1 + 28, y + 24, x2 - x1 - 56, F_TEXT, BLACK, 3) + 17

d.text((58, 1365), "Дорога является общей физической основой пешеходных и транспортных участков пути.", font=F_SMALL, fill=GRAY)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
assert OUT.exists() and OUT.stat().st_size > 50_000, "PNG не создан или оказался пустым"
print(f"{OUT} | {W}x{H} | {OUT.stat().st_size:,} bytes")

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "facility_location_tnd_timeline.png"

W, H = 3600, 1120
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(42, True)
F_SUB = font(27)
F_YEAR = font(34)
F_LANE = font(34, True)
F_BOX = font(28, True)
F_SMALL = font(20)
F_NOTE = font(22)

black = (24, 24, 24)
gray = (150, 150, 150)
grid = (222, 222, 222)
teal = (52, 145, 143)
light_teal = (235, 249, 248)
light_gray = (248, 248, 248)
dash = (0, 139, 139)

left = 430
right = 3420
top = 78
lane_y = {
    "flp": 145,
    "road": 300,
    "walk": 470,
    "transit": 640,
    "multi": 810,
}
lane_label_x = 80
lane_h = 100
box_h = 122

years = [
    ("1960-е", 520),
    ("1970-е", 840),
    ("1980-е", 1160),
    ("1990-е", 1530),
    ("2001", 1910),
    ("2010-е", 2290),
    ("2020+", 2690),
    ("2024", 2985),
    ("2025-2026", 3270),
]


def draw_wrapped(text: str, x: int, y: int, width: int, fnt, fill=black, line_gap=5):
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


def arrow(x1, y1, x2, y2, color=black, width=4):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    if x2 >= x1:
        pts = [(x2, y2), (x2 - 18, y2 - 9), (x2 - 18, y2 + 9)]
    else:
        pts = [(x2, y2), (x2 + 18, y2 - 9), (x2 + 18, y2 + 9)]
    d.polygon(pts, fill=color)


def box(x: int, y: int, w: int, title: str, body: str, refs: str = "", accent=False, h: int = box_h):
    fill = "white"
    d.rectangle((x, y, x + w, y + h), fill=fill, outline=(70, 70, 70), width=2)
    yy = draw_wrapped(title, x + 18, y + 15, w - 36, F_BOX, black, 5)
    if refs:
        refs_y = min(y + h - F_SMALL.size - 8, yy + 2)
        d.text((x + 18, refs_y), refs, font=F_SMALL, fill=(70, 70, 70))

def lane_label(y: int, title: str, subtitle: str):
    left_title = title.replace(" ", "\n", 1)
    x = lane_label_x
    if title in {"Дорожная сеть", "Пешеходная сеть", "Транспортная сеть"}:
        x += 55
    bbox = d.multiline_textbbox((0, 0), left_title, font=F_LANE, spacing=6)
    text_h = bbox[3] - bbox[1]
    y_text = y + (box_h - text_h) / 2 + 28
    d.multiline_text((x, y_text), left_title, font=F_LANE, fill=black, spacing=6)


def dashed_rect(x1, y1, x2, y2):
    step = 34
    for x in range(x1, x2, step):
        d.line((x, y1, min(x + 18, x2), y1), fill=dash, width=5)
        d.line((x, y2, min(x + 18, x2), y2), fill=dash, width=5)
    for y in range(y1, y2, step):
        d.line((x1, y, x1, min(y + 18, y2)), fill=dash, width=5)
        d.line((x2, y, x2, min(y + 18, y2)), fill=dash, width=5)


for label, x in years:
    d.text((x - 45, top), label, font=F_YEAR, fill=black)

lane_label(lane_y["flp"], "Размещение сервисов", "что и где открыть")
lane_label(lane_y["road"], "Дорожная сеть", "какие links и связи меняются")
lane_label(lane_y["walk"], "Пешеходная сеть", "как меняется пешая достижимость")
lane_label(lane_y["transit"], "Транспортная сеть", "как меняется PT и service level")
lane_label(lane_y["multi"], "Слои сети", "как связаны дороги, транспорт и среда")

box(400, 180, 400, "Классическое размещение объектов", "где открыть объект, если сеть уже задана", "[20][21]")
box(850, 180, 340, "Модели покрытия спроса", "объект нужен там, где спрос попадает в допустимое время", "[22][23]")
box(1460, 180, 340, "Влияние структуры сети", "топология сети меняет результат размещения", "[31]-[34]")
box(1900, 180, 350, "Совместное размещение и сеть", "выбираем не только объекты, но и связи сети", "[41]", True)
box(2580, 180, 470, "Прикладные задачи FLNDP", "медицина, справедливость, доступность, неоднородный спрос, топология дорог", "[42][43][45]", True)

box(1680, 335, 460, "Изменение дорожной сети", "связи сети как альтернатива новым объектам", "[41]")
box(2340, 335, 420, "Сбои дорожной сети", "деградация, перекрытия, подтопление, внешние шоки", "[42][55]")
box(2810, 335, 500, "Добавление и улучшение дорог", "новые связи и изменение дорожных ребер", "[24]-[26]")

box(2060, 505, 460, "Маршруты и время в пути", "выбор маршрута и время достижения объекта", "[27]-[30]")
box(1460, 505, 420, "Пешая доступность при жаре", "пешая достижимость и стоимость маршрута при жаре", "[53][54]")
box(2810, 505, 500, "Доступные пешеходные сегменты", "пешая достижимость через изменяемые сегменты", "[62]")

box(1460, 675, 460, "Надежность общественного транспорта", "интервалы движения, задержки, надежность", "[51][52][53]")
box(2250, 675, 420, "Связка велосипеда и PT", "пересадки и связанные поездки", "[57]")
box(2810, 675, 420, "Тарифы и уровень сервиса", "тарифы и качество сервиса влияют на доступность", "[44]")

box(2275, 845, 330, "Мультимодальная мобильность", "доступность складывается из разных видов транспорта", "[49]")
box(2660, 845, 330, "Транспорт как слои сети", "виды транспорта описываются как связанные слои", "[48]", True)
box(3100, 845, 330, "Взаимосвязь слоев сети", "динамика и взаимное влияние связанных слоев", "[50]", True)

# Horizontal arrows by lane.
for y, pairs in [
    (lane_y["flp"] + 92, [(800, 850), (1190, 1460), (1800, 1900), (2250, 2580)]),
    (lane_y["road"] + 92, [(2140, 2340), (2760, 2810)]),
    (lane_y["walk"] + 92, [(1880, 2060), (2520, 2810)]),
    (lane_y["transit"] + 92, [(1920, 2250), (2670, 2810)]),
    (lane_y["multi"] + 92, [(2605, 2660), (2990, 3100)]),
]:
    for x1, x2 in pairs:
        arrow(x1, y, x2, y)

# Convergence to the final framing.
final_x, final_y, final_w, final_h = 2680, 995, 760, 82
d.rounded_rectangle((final_x, final_y, final_x + final_w, final_y + final_h), radius=8, fill=light_gray, outline=black, width=3)
d.text((final_x + 24, final_y + 12), "Интегрированная модель доступности", font=F_BOX, fill=black)
d.text((final_x + 24, final_y + 48), "[42][43][44][48][50]", font=F_SMALL, fill=(70, 70, 70))


OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)

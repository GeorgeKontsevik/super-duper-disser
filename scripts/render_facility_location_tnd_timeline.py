from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "facility_location_tnd_timeline.png"

W, H = 3600, 1350
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(42, True)
F_SUB = font(27)
F_YEAR = font(30)
F_LANE = font(30, True)
F_BOX = font(32, True)
F_SMALL = font(24)
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
top = 190
lane_y = {
    "flp": 330,
    "network": 655,
    "multi": 980,
}
lane_label_x = 80
lane_h = 188

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


def box(x: int, y: int, w: int, title: str, body: str, refs: str = "", accent=False):
    fill = "white"
    d.rectangle((x, y, x + w, y + 120), fill=fill, outline=(70, 70, 70), width=2)
    draw_wrapped(title, x + 18, y + 15, w - 36, F_BOX, black, 5)
    if refs:
        d.text((x + 18, y + 88), refs, font=F_SMALL, fill=(70, 70, 70))

def lane_label(y: int, title: str, subtitle: str):
    d.text((lane_label_x, y + 42), title, font=F_LANE, fill=black)
    d.line((left - 40, y + lane_h // 2, right - 80, y + lane_h // 2), fill=grid, width=3)
    d.ellipse((left - 56, y + lane_h // 2 - 10, left - 36, y + lane_h // 2 + 10), fill=black)


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
    d.line((x, top + 55, x, 1190), fill=(235, 235, 235), width=2)

dashed_rect(2520, 240, 3485, 1160)

lane_label(lane_y["flp"], "Размещение сервисов", "что и где открыть")
lane_label(lane_y["network"], "Изменение сети", "какие связи изменить")
lane_label(lane_y["multi"], "Слои сети", "как связаны дороги, транспорт и среда")

box(470, 364, 300, "Классические FLP", "где открыть склад / сервис, если сеть уже задана", "[20][21]")
box(830, 364, 310, "Покрытие спроса", "объект нужен там, где спрос попадает в допустимое время", "[22][23]")
box(1460, 364, 340, "Сеть влияет на FLP", "топология сети меняет результат location-allocation", "[31]-[34]")
box(1900, 364, 350, "FL + network design", "выбираем не только объекты, но и связи сети", "[41]", True)
box(2580, 364, 470, "Прикладные FLNDP", "healthcare, equity, accessibility, heterogeneous demand, road topology", "[42][43][45]", True)

box(1080, 689, 330, "Transport NDP", "как построить / изменить links в транспортной сети", "[24]-[26]")
box(1460, 689, 330, "Routing + time", "размещение связывается с маршрутами и временем пути", "[27]-[30]")
box(1900, 689, 350, "Network modification", "изменение links становится альтернативой новым объектам", "[41]", True)
box(2310, 689, 340, "Disruption", "сеть может деградировать; параметры links неопределенны", "[42]")
box(2730, 689, 320, "Transit + fares", "сеть ОТ и тарифы входят в задачу сервисов", "[44]", True)

box(2275, 1014, 330, "Multimodal mobility", "городская мобильность складывается из разных видов транспорта", "[49]")
box(2660, 1014, 330, "Multilayer mobility", "виды транспорта описываются как связанные слои", "[48]", True)
box(3100, 1014, 330, "Multilayer science", "важна созависимость и динамика слоев сети", "[50]", True)

# Horizontal arrows by lane.
for y, pairs in [
    (lane_y["flp"] + 92, [(770, 830), (1140, 1460), (1800, 1900), (2250, 2580)]),
    (lane_y["network"] + 92, [(1410, 1460), (1790, 1900), (2250, 2310), (2650, 2730)]),
    (lane_y["multi"] + 92, [(2605, 2660), (2990, 3100)]),
]:
    for x1, x2 in pairs:
        arrow(x1, y, x2, y)

# Convergence to the final framing.
final_x, final_y, final_w, final_h = 2700, 1210, 720, 105
d.rounded_rectangle((final_x, final_y, final_x + final_w, final_y + final_h), radius=8, fill=light_gray, outline=black, width=3)
d.text((final_x + 24, final_y + 18), "Интегрированная постановка", font=F_BOX, fill=black)
d.text((final_x + 24, final_y + 66), "[42][43][44][48][50]", font=F_SMALL, fill=(70, 70, 70))


OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)

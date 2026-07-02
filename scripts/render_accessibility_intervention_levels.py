from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "accessibility_intervention_levels.png"

W, H = 2200, 1340
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(36, True)
F_H = font(28, True)
F_TEXT = font(20)
F_SMALL = font(17)
F_TINY = font(15)

BLACK = (28, 28, 28)
GRAY = (130, 130, 130)
GRID = (224, 228, 232)
TEAL = (10, 142, 140)
TEAL_FILL = (231, 247, 246)
ORANGE = (229, 121, 54)
ORANGE_FILL = (255, 238, 228)
RED = (221, 88, 88)
RED_FILL = (252, 234, 234)
LIGHT = (248, 249, 251)


def wrap(text: str, x: int, y: int, width: int, fnt, fill=BLACK, gap: int = 4):
    words = text.split()
    line = ""
    for word in words:
        cand = f"{line} {word}".strip()
        if line and d.textlength(cand, font=fnt) > width:
            d.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + gap
            line = word
        else:
            line = cand
    if line:
        d.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + gap
    return y


def rbox(x1, y1, x2, y2, outline, fill="white", width=3, radius=18):
    d.rounded_rectangle((x1, y1, x2, y2), radius=radius, outline=outline, fill=fill, width=width)


def arrow(x1, y1, x2, y2, color=BLACK, width=4):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    d.polygon([(x2, y2), (x2 - 16, y2 - 9), (x2 - 16, y2 + 9)], fill=color)


cols = [
    (70, "Операционные", TEAL, TEAL_FILL, [
        ("пробки и локальные задержки", "[51][52]"),
        ("светофоры и приоритет", "[51]"),
        ("headway / расписания", "[51][52]"),
        ("временные closures и сбои", "[52][53]"),
    ]),
    (760, "Сетевые / инфраструктурные", ORANGE, ORANGE_FILL, [
        ("качество дороги и links", "[54][56]"),
        ("пешеходная / вело-связность", "[57][62]"),
        ("надежность PT как свойство сети", "[52][57]"),
        ("новые связи и реконструкция узлов", "[54][55][56]"),
    ]),
    (1450, "Структурные / пространственные", RED, RED_FILL, [
        ("наличие или отсутствие сервиса", "[58][59][62]"),
        ("морфология и urban form", "[60][61]"),
        ("плотность и mixed-use", "[58][60]"),
        ("пространственное распределение спроса", "[58][59]"),
    ]),
]

top = 36
col_w = 610
col_h = 330

for x, title, color, fill, bullets in cols:
    rbox(x, top, x + col_w, top + col_h, outline=color, fill="white", width=4)
    rbox(x + 18, top + 18, x + col_w - 18, top + 78, outline=color, fill=fill, width=2, radius=14)
    d.text((x + 28, top + 35), title, font=F_H, fill=BLACK)

    y = top + 96
    for bullet, refs in bullets:
        rbox(x + 22, y, x + col_w - 22, y + 50, outline=GRID, fill=LIGHT, width=2, radius=12)
        d.text((x + 42, y + 18), "•", font=F_H, fill=color)
        d.text((x + col_w - 150, y + 15), refs, font=F_TINY, fill=GRAY)
        wrap(bullet, x + 68, y + 13, col_w - 250, F_TEXT, BLACK, 3)
        y += 56

arrow(680, 238, 760, 238)
arrow(1370, 238, 1450, 238)

rbox(70, 700, 2130, 1270, outline=GRID, fill="white", width=3)
d.text((94, 728), "Ссылки", font=F_H, fill=BLACK)

refs = [
    "[51] Conway, Byrd & van der Linden (2017) - schedule- and headway-based accessibility networks.",
    "[52] Zang et al. (2022) - travel-time reliability review at link, route, network levels.",
    "[53] ReVelle & Hogan (1989) - coverage with stated reliability.",
    "[54] Murawski & Church (2009) - road improvements to increase access to rural health services.",
    "[55] Baldomero-Naranjo et al. (2022) - edge upgrading in MCLP under budget.",
    "[56] Akhlaghi, Campbell & Demir (2023) - flood mitigation road-network design.",
    "[57] Geurs, La Paix & van Weperen (2016) - bicycle-train integration and PT accessibility.",
    "[58] Melkote & Daskin (2001) - facility location and transportation network design.",
    "[59] Pourrezaie-Khaligh et al. (2022) - healthcare FLND with equity and accessibility.",
    "[60] Willberg, Fink & Toivonen (2023) - temporal variation in walking accessibility.",
    "[61] Wang & He (2022) - walkability under heat stress in 15-minute cities.",
    "[62] Starita et al. (2024) - facilities plus accessible road segments for non-motorized access.",
]

y = 770
for ref in refs:
    y = wrap(ref, 94, y, 1980, F_SMALL, BLACK, 4) + 8

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)

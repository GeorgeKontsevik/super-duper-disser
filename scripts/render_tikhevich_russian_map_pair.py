from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "itmo-phd-thesis-template-en" / "images" / "ch4" / "optimal_local"
LEFT_SOURCE = SOURCE_DIR / "tikhevich_new_services.jpeg"
RIGHT_SOURCE = SOURCE_DIR / "tikhevich_link_improvements.jpeg"
OUT = SOURCE_DIR / "tikhevich_russian_map_pair.png"

W, H = 2400, 1120
MARGIN = 45
GAP = 45
PANEL_W = (W - MARGIN * 2 - GAP) // 2
MAP_H = 800

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
F_LEGEND = ImageFont.truetype(FONT, 30)
F_CAPTION = ImageFont.truetype(BOLD, 36)

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def fit(source, crop, width, height):
    source = source.crop(crop)
    source.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(source, ((width - source.width) // 2, (height - source.height) // 2))
    return canvas


def plus_marker(x, y, color):
    d.ellipse((x - 13, y - 13, x + 13, y + 13), fill="white", outline=color, width=3)
    d.line((x - 7, y, x + 7, y), fill=color, width=4)
    d.line((x, y - 7, x, y + 7), fill=color, width=4)


def centered_multiline(x, y, width, text):
    box = d.multiline_textbbox((0, 0), text, font=F_CAPTION, align="center", spacing=8)
    d.multiline_text((x + (width - (box[2] - box[0])) / 2, y), text, font=F_CAPTION,
                     fill=(20, 20, 20), align="center", spacing=8)


left = fit(Image.open(LEFT_SOURCE).convert("RGB"), (100, 80, 3400, 2250), PANEL_W, MAP_H)
right = fit(Image.open(RIGHT_SOURCE).convert("RGB"), (100, 0, 3400, 1900), PANEL_W, MAP_H)

left_x = MARGIN
right_x = MARGIN + PANEL_W + GAP
img.paste(left, (left_x, 10))
img.paste(right, (right_x, 10))

# Replace the English legend embedded in the left source.
d.rectangle((left_x + 5, 90, left_x + 430, 235), fill="white")
plus_marker(left_x + 220, 858, (220, 45, 50))
d.text((left_x + 245, 838), "Существующие сервисы", font=F_LEGEND, fill=(20, 20, 20))
plus_marker(left_x + 675, 858, (50, 210, 45))
d.text((left_x + 700, 838), "Новые сервисы", font=F_LEGEND, fill=(20, 20, 20))

# Russian legend for accessibility improvement intervals.
legend_y = 825
d.multiline_text((right_x + 15, legend_y - 12), "Сокращение времени\nдоступности между\nкварталами", font=F_LEGEND,
                 fill=(20, 20, 20), spacing=3)
colors = [(20, 238, 120), (80, 222, 202), (55, 177, 190), (43, 128, 184), (68, 76, 143)]
labels = ["2–3,8", "3,8–5,4", "5,4–8", "8–10,5", "10,5–13,2"]
start_x = right_x + 300
for i, (color, label) in enumerate(zip(colors, labels)):
    x = start_x + i * 165
    d.line((x, legend_y + 33, x + 58, legend_y + 33), fill=color, width=13)
    d.text((x + 67, legend_y + 13), label, font=F_LEGEND, fill=(20, 20, 20))

centered_multiline(left_x, 970, PANEL_W, "а) Размещение существующих и\nдополнительных сервисов")
centered_multiline(right_x, 970, PANEL_W, "б) Распределение улучшения транспортной\nсвязности между кварталами")

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, dpi=(300, 300))
print(OUT)

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "itmo-phd-thesis-template-en/images/ch4/optimal_local/polyclinic_route_strategy_4x3_ru.png"
OUT = ROOT / "itmo-phd-thesis-template-en/images/ch4/optimal_local/polyclinic_route_strategy_3x4_ru.png"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
F_HEADER = ImageFont.truetype(BOLD, 34)
F_SUB = ImageFont.truetype(FONT, 25)
F_ROW = ImageFont.truetype(BOLD, 29)

source = Image.open(SOURCE).convert("RGBA")
canvas = Image.new("RGBA", (2500, 1840), "white")
d = ImageDraw.Draw(canvas)

cities = [("Берген", "Норвегия"), ("Брно", "Чехия"), ("Какогава", "Япония"), ("Краков", "Польша")]
strategies = ["общая\nсвязность", "к существующим\nсервисам", "к потенциальным\nновым сервисам"]
x_crops = [(490, 1030), (1045, 1585), (1600, 2140)]
y_crops = [(120, 660), (625, 1165), (1130, 1670), (1635, 2175)]

left = 300
top = 185
cell_w = 535
cell_h = 465
map_size = 400

for col, (city, country) in enumerate(cities):
    x = left + col * cell_w + cell_w // 2
    city_box = d.textbbox((0, 0), city, font=F_HEADER)
    country_box = d.textbbox((0, 0), country, font=F_SUB)
    d.text((x - (city_box[2] - city_box[0]) / 2, 28), city, font=F_HEADER, fill=(31, 41, 46))
    d.text((x - (country_box[2] - country_box[0]) / 2, 72), country, font=F_SUB, fill=(91, 106, 117))

for row, strategy in enumerate(strategies):
    y = top + row * cell_h
    box = d.multiline_textbbox((0, 0), strategy, font=F_ROW, align="center", spacing=2)
    d.multiline_text((150 - (box[2] - box[0]) / 2, y + 185), strategy, font=F_ROW,
                     fill=(31, 41, 46), align="center", spacing=2)
    for col in range(4):
        x1, x2 = x_crops[row]
        y1, y2 = y_crops[col]
        crop = source.crop((x1, y1, x2, y2)).resize((map_size, map_size), Image.Resampling.LANCZOS)
        canvas.alpha_composite(crop, (left + col * cell_w + 42, y + 20))

legend = source.crop((120, 2180, 2120, 2429))
legend.thumbnail((1950, 220), Image.Resampling.LANCZOS)
canvas.alpha_composite(legend, ((canvas.width - legend.width) // 2, 1600))

OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT)
print(OUT)

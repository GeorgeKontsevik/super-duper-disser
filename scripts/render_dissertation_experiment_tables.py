from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp"
OUT1 = OUT_DIR / "dissertation_position1_experiments.png"
OUT2 = OUT_DIR / "dissertation_position2_experiments.png"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


F_TITLE = font(34, True)
F_HEAD = font(24, True)
F_TEXT = font(21)
F_SMALL = font(18)

BLACK = (28, 28, 28)
GRAY = (110, 110, 110)
GRID = (222, 226, 230)
BLUE = (230, 240, 252)
TEAL = (228, 246, 244)
WHITE = (255, 255, 255)


def wrap(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, fnt, fill=BLACK, line_gap: int = 5):
    words = text.split()
    line = ""
    start_y = y
    for word in words:
        cand = f"{line} {word}".strip()
        if line and draw.textlength(cand, font=fnt) > width:
            draw.text((x, y), line, font=fnt, fill=fill)
            y += fnt.size + line_gap
            line = word
        else:
            line = cand
    if line:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y - start_y


def cell(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill=BLACK, pad_x: int = 16, pad_y: int = 14, line_gap: int = 5):
    x1, y1, x2, y2 = box
    wrap(draw, text, x1 + pad_x, y1 + pad_y, x2 - x1 - 2 * pad_x, fnt, fill, line_gap)


def draw_table(out_path: Path, title: str, position_text: str, rows: list[dict], accent):
    w = 2440
    header_h = 78
    row_h = 194
    top_pad = 118
    left = 24
    bottom_pad = 26
    h = top_pad + header_h + row_h * len(rows) + bottom_pad

    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    d.text((left, 28), title, font=F_TITLE, fill=BLACK)

    cols = [
        ("Положение", 500),
        ("Гипотеза", 620),
        ("Эксперимент", 660),
        ("Проверяемый эффект", 620),
    ]

    xs = [left]
    for _, width in cols:
        xs.append(xs[-1] + width)

    y0 = top_pad

    d.rounded_rectangle((left, y0, xs[-1], y0 + header_h), radius=18, fill=WHITE, outline=GRID, width=2)
    d.rectangle((left, y0, xs[1], y0 + header_h), fill=accent)

    for i in range(1, len(xs) - 1):
        d.line((xs[i], y0, xs[i], y0 + header_h + row_h * len(rows)), fill=GRID, width=2)

    for i, (label, _) in enumerate(cols):
        d.text((xs[i] + 16, y0 + 24), label, font=F_HEAD, fill=BLACK)

    pos_box = (left, y0 + header_h, xs[1], y0 + header_h + row_h * len(rows))
    d.rounded_rectangle(pos_box, radius=18, fill=accent, outline=GRID, width=2)
    cell(d, pos_box, position_text, F_TEXT, pad_x=18, pad_y=18, line_gap=7)

    for idx, row in enumerate(rows):
        y1 = y0 + header_h + idx * row_h
        y2 = y1 + row_h
        d.rounded_rectangle((xs[1], y1, xs[-1], y2), radius=0, fill=WHITE, outline=GRID, width=2)
        d.line((left, y2, xs[-1], y2), fill=GRID, width=2)

        hypothesis_pad = 84 if "tag" in row else 16
        cell(d, (xs[1], y1, xs[2], y2), row["hypothesis"], F_TEXT, pad_x=hypothesis_pad)
        cell(d, (xs[2], y1, xs[3], y2), row["experiment"], F_TEXT)
        cell(d, (xs[3], y1, xs[4], y2), row["effect"], F_TEXT)

        if "tag" in row:
            d.rounded_rectangle((xs[1] + 14, y1 + 12, xs[1] + 74, y1 + 44), radius=14, fill=accent, outline=None)
            d.text((xs[1] + 28, y1 + 18), row["tag"], font=F_SMALL, fill=BLACK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


position1 = {
    "title": "Положение 1. Экспериментальная проверка метода оценки транспортной доступности",
    "position": (
        "Метод оценки транспортной доступности на основе критерия сезонной устойчивости, "
        "отличающийся применением методов сетевого анализа и учетом стохастической динамики "
        "многоуровневой сети."
    ),
    "rows": [
        {
            "tag": "H1",
            "hypothesis": (
                "Сезонная реконфигурация транспортной сети существенно меняет сервисную доступность "
                "и отношения «потребитель–поставщик»."
            ),
            "experiment": (
                "Арктический EFN-эксперимент: сравнить состояния сети по месяцам и сезонам "
                "в 4 системах расселения при температурно-зависимых режимах транспорта."
            ),
            "effect": (
                "Меняются сервисные зоны и поставщики услуг: до 40% поселений переключают "
                "поставщика, 15-20% временно изолируются на 2-4 месяца."
            ),
        },
        {
            "tag": "H2",
            "hypothesis": (
                "Климатическое воздействие на разные типы ребер меняет доступность неравномерно; "
                "уязвимость сети зависит от типа дороги и наличия альтернатив."
            ),
            "experiment": (
                "Экваториальный эксперимент: сравнить доступность и критические связи при климатическом "
                "стрессе для дорожной сети с покрытием и без покрытия и цепочек поставок."
            ),
            "effect": (
                "Выявляются чувствительные неасфальтированные связи, меняются достижимые маршруты "
                "и пространственный профиль уязвимости."
            ),
        },
        {
            "tag": "H3",
            "hypothesis": (
                "Статическая оценка транспортной доступности недоучитывает сезонную значимость "
                "населенных пунктов по сравнению с сезонно-устойчивой оценкой."
            ),
            "experiment": (
                "Сравнить базовую методику выделения опорных населенных пунктов и оценку, "
                "в которой учтена сезонная реконфигурация сети для группы сервисов «Здравоохранение»."
            ),
            "effect": (
                "Меняется ранжирование населенных пунктов, выделяются локальные опорные пункты "
                "и их сезонная значимость."
            ),
        },
    ],
}


position2 = {
    "title": "Положение 2. Экспериментальная проверка метода оптимального размещения",
    "position": (
        "Метод интеллектуальной поддержки принятия решений оптимального размещения объектов обслуживания, "
        "отличающийся применением оптимизации сети и обеспечивающий уменьшение необходимого "
        "количества объектов обслуживания."
    ),
    "rows": [
        {
            "tag": "H4",
            "hypothesis": (
                "Положительное изменение транспортной связности может уменьшить минимальное число "
                "новых объектов обслуживания."
            ),
            "experiment": (
                "Оптимизировать отдельные значения матрицы доступности генетическим алгоритмом, "
                "затем решать CLSCP-SO при нормативе 15 минут."
            ),
            "effect": (
                "Требуемое число новых поликлиник снижается с 13 до 9; меняется их оптимальное "
                "размещение."
            ),
        },
        {
            "tag": "H5",
            "hypothesis": (
                "Маршрутная интервенция в сети ОТ может давать сопоставимый или лучший эффект, "
                "чем локальное дорожное улучшение."
            ),
            "experiment": (
                "Сценарий квартала Тельмана: сравнить базовый сценарий, добавление дороги, "
                "добавление маршрута ОТ и совместное добавление дороги и маршрута."
            ),
            "effect": (
                "Меняется матрица транспортной доступности; лучшие сценарии снижают потребность "
                "в новых поликлиниках с 14 до 13 и 12."
            ),
        },
        {
            "tag": "H6",
            "hypothesis": (
                "В задаче размещения сервис-ориентированная целевая функция связности эффективнее, "
                "чем усредненная связность по всему городу."
            ),
            "experiment": (
                "Сравнить 3 стратегии генерации маршрутов на 7 городах: средняя связность по всему городу, "
                "только кварталы с существующим сервисом, только кварталы-кандидаты под сервис."
            ),
            "effect": (
                "В 6 из 7 городов сервис-ориентированная связность дает меньшее число новых "
                "объектов обслуживания."
            ),
        },
    ],
}


def main():
    draw_table(OUT1, position1["title"], position1["position"], position1["rows"], BLUE)
    draw_table(OUT2, position2["title"], position2["position"], position2["rows"], TEAL)
    print(OUT1)
    print(OUT2)


if __name__ == "__main__":
    main()

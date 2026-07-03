#!/usr/bin/env python3
"""Render Russian May–October Sankey diagrams for Arctic services and regions."""

import json
import pickle
import sys
from pathlib import Path

from PIL import Image
from transliterate import translit


ROOT = Path(__file__).resolve().parents[1]
ARCTIC = ROOT / "arctic_access"
sys.path[:0] = [str(ROOT), str(ARCTIC)]

from scripts.plotter.plotter_flow_sankey import create_clean_sankey  # noqa: E402


REGIONS = {
    "yanao_kras": "ЯНАО — Красноярский край",
    "mezen": "Мезень",
    "nao": "НАО",
    "yakut_chuk": "Якутия — Чукотка",
}
SELECTED_SQUARE = (
    ("nao", "health"),
    ("yakut_chuk", "port"),
    ("mezen", "port"),
    ("yanao_kras", "airport"),
)


def russian_names(region):
    path = ARCTIC / f"data/processed/{region}/df_post_{region}.geojson"
    names = {feature["properties"]["name"] for feature in json.loads(path.read_text())["features"]}
    return {translit(name, "ru", reversed=True): name for name in names}


def main():
    output_dir = ROOT / "itmo-phd-thesis-template-en/images/ch4/arctic/sankey_ru"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for region, service in SELECTED_SQUARE:
        with (ARCTIC / f"notebooks/all_results_{region}.pkl").open("rb") as file:
            results = pickle.load(file)[region]
        labels = russian_names(region)
        figure = create_clean_sankey(
            results[service]["stats"].graphs[4:10],
            month_start=4,
            service_name=service,
            node_labels=labels,
            agglomeration_name=REGIONS[region],
            show=False,
        )
        path = output_dir / f"{region}_{service}_may_oct_ru_4x3.png"
        figure.write_image(path, width=1200, height=900, scale=2)
        rendered.append(path)
        print(path)

    images = [Image.open(path).convert("RGB") for path in rendered]
    sheet = Image.new("RGB", (4800, 3600), "white")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 2) * 2400, (index // 2) * 1800))
    sheet_path = output_dir / "selected_4x3_may_oct_ru.png"
    sheet.save(sheet_path)
    print(sheet_path)


if __name__ == "__main__":
    main()

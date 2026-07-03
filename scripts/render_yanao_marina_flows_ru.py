#!/usr/bin/env python3
"""Render May and August YANAO small-port flow diagrams in Russian."""

import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCTIC = ROOT / "arctic_access"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ARCTIC))

from scripts.plotter.plotter_circular_network_sankey_style import (  # noqa: E402
    plot_circular_network_sankey_style,
)


def main():
    with (ARCTIC / "notebooks/all_results_yanao_kras.pkl").open("rb") as file:
        results = pickle.load(file)

    graphs = results["yanao_kras"]["marina"]["stats"].graphs
    output_dir = ROOT / "itmo-phd-thesis-template-en/images/ch4/arctic"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, month, filename in (
        (4, "May", "yanao_kras_small_port_flow_may_ru.png"),
        (7, "Aug", "yanao_kras_small_port_flow_aug_ru.png"),
    ):
        figure = plot_circular_network_sankey_style(
            graphs[index], service_name="marina", month_name=month, language="ru"
        )
        figure.write_image(output_dir / filename, width=1400, height=1400, scale=2)
        print(output_dir / filename)


if __name__ == "__main__":
    main()

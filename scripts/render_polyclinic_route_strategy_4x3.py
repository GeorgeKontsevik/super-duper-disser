from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "segregation-by-design-experiments/polyclinic_access_components/outputs"
    / "overnight_route_strategy_batch_20260613_routes3_finalcanvas"
    / "_scenario_grid_7x3_regenerated_20260620"
    / "polyclinic_7x3_round_maps_ru_gray_biglegend_white.png"
)
OUT = ROOT / "itmo-phd-thesis-template-en/images/ch4/optimal_local/polyclinic_route_strategy_4x3_ru.png"

# Cuts use the blank scanline gaps between complete circular outlines.
# Retained rows: Bergen, Brno, Kakogawa, Krakow, followed by the legend.
SLICES = [(0, 666), (1169, 1673), (2680, 3184), (3184, 3688), (3688, 3939)]

source = Image.open(SOURCE).convert("RGBA")
parts = [source.crop((0, top, source.width, bottom)) for top, bottom in SLICES]
result = Image.new("RGBA", (source.width, sum(part.height for part in parts)), "white")

y = 0
for part in parts:
    result.alpha_composite(part, (0, y))
    y += part.height

OUT.parent.mkdir(parents=True, exist_ok=True)
result.save(OUT)
print(OUT)

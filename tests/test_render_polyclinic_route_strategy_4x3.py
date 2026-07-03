import subprocess
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_polyclinic_route_strategy_4x3.py"
OUTPUT = ROOT / "itmo-phd-thesis-template-en" / "images" / "ch4" / "optimal_local" / "polyclinic_route_strategy_4x3_ru.png"


class PolyclinicRouteStrategy4x3Test(unittest.TestCase):
    def test_renderer_writes_four_city_grid(self):
        OUTPUT.unlink(missing_ok=True)
        subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)
        with Image.open(OUTPUT) as image:
            self.assertEqual(image.width, 2240)
            self.assertLess(image.height, 2800)
            self.assertEqual(image.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()

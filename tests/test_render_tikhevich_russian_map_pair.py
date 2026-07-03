import subprocess
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_tikhevich_russian_map_pair.py"
OUTPUT = ROOT / "itmo-phd-thesis-template-en" / "images" / "ch4" / "optimal_local" / "tikhevich_russian_map_pair.png"


class TikhevichRussianMapPairTest(unittest.TestCase):
    def test_renderer_writes_expected_png(self):
        OUTPUT.unlink(missing_ok=True)
        subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)
        with Image.open(OUTPUT) as image:
            self.assertEqual(image.size, (2400, 1120))
            self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()

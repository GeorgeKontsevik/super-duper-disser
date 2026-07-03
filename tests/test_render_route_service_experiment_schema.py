import subprocess
import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_route_service_experiment_schema.py"
OUTPUT = ROOT / "itmo-phd-thesis-template-en" / "Dissertation" / "route_service_experiment_schema.png"


class RouteServiceExperimentSchemaTest(unittest.TestCase):
    def test_scenario_uses_expected_additional_service_counts(self):
        source = SCRIPT.read_text()
        self.assertIn('additional_services(165, 565, 4)', source)
        self.assertIn('additional_services(550, 565, 4)', source)
        self.assertIn('additional_services(790, 565, 2)', source)
        self.assertIn('additional_services(1145, 565, 3)', source)
        self.assertIn('additional_services(1475, 565, 2)', source)
        self.assertIn('additional_services(1875, 565, 2)', source)
        self.assertIn('additional_services(2305, 565, 1)', source)

    def test_renderer_writes_expected_png(self):
        OUTPUT.unlink(missing_ok=True)
        subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True)

        with Image.open(OUTPUT) as image:
            self.assertEqual(image.size, (2580, 735))
            self.assertEqual(image.mode, "RGB")
            colors = {rgb for _, rgb in image.getcolors(maxcolors=image.width * image.height)}

        for expected in {
            (255, 212, 212),
            (210, 238, 210),
            (165, 105, 190),
            (0, 140, 137),
        }:
            self.assertIn(expected, colors)


if __name__ == "__main__":
    unittest.main()

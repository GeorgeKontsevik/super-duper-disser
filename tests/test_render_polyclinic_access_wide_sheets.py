import subprocess
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_polyclinic_access_wide_sheets.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
OUT_DIR = ROOT / "aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/maps_by_service_ru"


class PolyclinicAccessWideSheetsTest(unittest.TestCase):
    def test_renderer_writes_two_layouts(self):
        subprocess.run([str(PYTHON), str(SCRIPT)], cwd=ROOT, check=True)
        for name in (
            "02_polikliniki_access_diagnostics_ru_2rows.png",
            "02_polikliniki_access_diagnostics_ru_3rows.png",
        ):
            with Image.open(OUT_DIR / name) as image:
                self.assertGreater(image.width, image.height)
                self.assertEqual(image.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()

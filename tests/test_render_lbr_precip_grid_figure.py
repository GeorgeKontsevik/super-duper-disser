from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd
from shapely.geometry import box


SCRIPT_PATH = Path("/Users/gk/Code/super-duper-disser/equatorial/scripts/render_lbr_precip_grid_figure.py")
SPEC = importlib.util.spec_from_file_location("render_lbr_precip_grid_figure", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None


class RenderLbrPrecipGridFigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SPEC.loader.exec_module(MODULE)

    def test_selects_wettest_week_inside_requested_months(self) -> None:
        weekly = pd.DataFrame(
            [
                {"week_start": "2024-03-18", "median_mm": 150.0},
                {"week_start": "2024-07-15", "median_mm": 45.0},
                {"week_start": "2024-08-12", "median_mm": 70.0},
                {"week_start": "2024-08-19", "median_mm": 84.0},
            ]
        )

        selected = MODULE.select_wettest_week(weekly, months=[7, 8])

        self.assertEqual(selected.isoformat(), "2024-08-19")

    def test_masks_era5_points_outside_country_boundary(self) -> None:
        grid = pd.DataFrame(
            [
                {"cell_lon": 0.5, "cell_lat": 0.5, "tp_sum_weekly_mm": 10.0},
                {"cell_lon": 1.5, "cell_lat": 0.5, "tp_sum_weekly_mm": 20.0},
                {"cell_lon": 0.5, "cell_lat": -0.5, "tp_sum_weekly_mm": 30.0},
            ]
        )

        masked = MODULE.mask_points_inside_country(grid, box(0.0, 0.0, 1.0, 1.0))

        self.assertEqual(len(masked), 1)
        self.assertEqual(float(masked.iloc[0]["tp_sum_weekly_mm"]), 10.0)


if __name__ == "__main__":
    unittest.main()

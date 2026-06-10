from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path("/Users/gk/Code/super-duper-disser/equatorial/scripts/render_paper_rainfall_mechanism_experiment.py")
SPEC = importlib.util.spec_from_file_location("render_paper_rainfall_mechanism_experiment", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None


class RenderPaperRainfallMechanismExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SPEC.loader.exec_module(MODULE)

    def test_build_country_rainfall_summary_aggregates_weekly_severity(self) -> None:
        weekly_rain = pd.DataFrame(
            [
                {"country_code": "AAA", "week_start": "2024-01-01", "median": 100.0, "q75": 140.0},
                {"country_code": "AAA", "week_start": "2024-01-08", "median": 70.0, "q75": 90.0},
                {"country_code": "BBB", "week_start": "2024-01-01", "median": 40.0, "q75": 50.0},
                {"country_code": "BBB", "week_start": "2024-01-08", "median": 30.0, "q75": 45.0},
            ]
        )

        summary = MODULE.build_country_rainfall_summary(weekly_rain)
        aaa = summary.loc[summary["country_code"].eq("AAA")].iloc[0]
        bbb = summary.loc[summary["country_code"].eq("BBB")].iloc[0]

        self.assertEqual(int(aaa["rainy_weeks_ge_75mm"]), 1)
        self.assertEqual(int(aaa["extreme_weeks_ge_125mm_q75"]), 1)
        self.assertAlmostEqual(float(aaa["total_weekly_median_mm"]), 170.0)
        self.assertAlmostEqual(float(bbb["total_weekly_median_mm"]), 70.0)

    def test_build_country_burden_summary_from_weekly_keeps_zero_burden_countries(self) -> None:
        cells = pd.DataFrame(
            [
                {"country_code": "AAA", "week_start": "2024-01-01", "median_delta_minutes": 240.0},
                {"country_code": "AAA", "week_start": "2024-01-08", "median_delta_minutes": 180.0},
                {"country_code": "BBB", "week_start": "2024-01-01", "median_delta_minutes": 0.0},
            ]
        )
        dest_crop_points = pd.DataFrame(
            [
                {"country_code": "AAA", "annual_severe_burden_h": 10.0, "affected_cluster_weight": 100.0, "mean_affected_delay_h": 5.0, "affected_weeks": 3},
            ]
        )
        countries = ["AAA", "BBB"]

        summary = MODULE.build_country_burden_summary_from_weekly(cells, dest_crop_points, countries)
        aaa = summary.loc[summary["country_code"].eq("AAA")].iloc[0]
        bbb = summary.loc[summary["country_code"].eq("BBB")].iloc[0]

        self.assertAlmostEqual(float(aaa["total_burden_h"]), 1.0)
        self.assertAlmostEqual(float(aaa["total_affected_exposure"]), 100.0)
        self.assertAlmostEqual(float(aaa["max_delay_h"]), 4.0)
        self.assertAlmostEqual(float(aaa["median_affected_weeks"]), 1.0)
        self.assertAlmostEqual(float(bbb["total_burden_h"]), 0.0)
        self.assertAlmostEqual(float(bbb["total_affected_exposure"]), 0.0)

    def test_combine_country_mechanism_summary_detects_rank_mismatch(self) -> None:
        rainfall = pd.DataFrame(
            [
                {"country_code": "AAA", "total_weekly_median_mm": 300.0, "median_weekly_median_mm": 100.0, "max_weekly_q75_mm": 160.0, "rainy_weeks_ge_75mm": 3, "extreme_weeks_ge_125mm_q75": 2},
                {"country_code": "BBB", "total_weekly_median_mm": 200.0, "median_weekly_median_mm": 70.0, "max_weekly_q75_mm": 120.0, "rainy_weeks_ge_75mm": 1, "extreme_weeks_ge_125mm_q75": 0},
                {"country_code": "CCC", "total_weekly_median_mm": 100.0, "median_weekly_median_mm": 40.0, "max_weekly_q75_mm": 80.0, "rainy_weeks_ge_75mm": 0, "extreme_weeks_ge_125mm_q75": 0},
            ]
        )
        burden = pd.DataFrame(
            [
                {"country_code": "AAA", "total_burden_h": 20.0, "total_affected_exposure": 100.0, "max_delay_h": 9.0, "median_affected_weeks": 5.0},
                {"country_code": "BBB", "total_burden_h": 50.0, "total_affected_exposure": 300.0, "max_delay_h": 12.0, "median_affected_weeks": 8.0},
                {"country_code": "CCC", "total_burden_h": 10.0, "total_affected_exposure": 80.0, "max_delay_h": 4.0, "median_affected_weeks": 2.0},
            ]
        )

        combined = MODULE.combine_country_mechanism_summary(rainfall, burden)
        bbb = combined.loc[combined["country_code"].eq("BBB")].iloc[0]
        aaa = combined.loc[combined["country_code"].eq("AAA")].iloc[0]

        self.assertEqual(int(bbb["rain_rank"]), 2)
        self.assertEqual(int(bbb["burden_rank"]), 1)
        self.assertEqual(int(bbb["rank_gap"]), 1)
        self.assertEqual(int(aaa["rank_gap"]), -1)

    def test_select_high_rain_contrast_weeks_omits_critical_marker_when_high_rain_has_no_burden(self) -> None:
        weekly = pd.DataFrame(
            [
                {"week_start": "2024-05-06", "median": 95.0, "weekly_burden_h": 0.0},
                {"week_start": "2024-05-13", "median": 101.5, "weekly_burden_h": 0.0},
                {"week_start": "2024-05-20", "median": 88.0, "weekly_burden_h": 0.0},
                {"week_start": "2024-05-27", "median": 40.0, "weekly_burden_h": 0.0},
            ]
        )

        contrast = MODULE.select_high_rain_contrast_weeks(weekly)

        self.assertIn("high_rain_low_burden", contrast)
        self.assertNotIn("high_rain_high_burden", contrast)


if __name__ == "__main__":
    unittest.main()

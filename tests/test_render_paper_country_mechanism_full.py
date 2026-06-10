from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path("/Users/gk/Code/super-duper-disser/equatorial/scripts/render_paper_country_mechanism_full.py")
SPEC = importlib.util.spec_from_file_location("render_paper_country_mechanism_full", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None


class RenderPaperCountryMechanismFullTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        SPEC.loader.exec_module(MODULE)

    def test_aggregate_route_mix_builds_country_level_surface_and_remoteness_metrics(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "country_code": "AAA",
                    "crop_code": "banana",
                    "candidate_rank": 1,
                    "dest_type": "port",
                    "dest_rank": 1,
                    "dest_id": 10,
                    "cluster_cell_count": 100,
                    "total_travel_time_h": 10.0,
                    "surface_group": "unpaved",
                    "surface_travel_time_pct": 0.40,
                },
                {
                    "country_code": "AAA",
                    "crop_code": "banana",
                    "candidate_rank": 1,
                    "dest_type": "port",
                    "dest_rank": 1,
                    "dest_id": 10,
                    "cluster_cell_count": 100,
                    "total_travel_time_h": 10.0,
                    "surface_group": "unknown",
                    "surface_travel_time_pct": 0.10,
                },
                {
                    "country_code": "AAA",
                    "crop_code": "banana",
                    "candidate_rank": 1,
                    "dest_type": "port",
                    "dest_rank": 2,
                    "dest_id": 11,
                    "cluster_cell_count": 50,
                    "total_travel_time_h": 20.0,
                    "surface_group": "unpaved_synthetic_line",
                    "surface_travel_time_pct": 0.50,
                },
                {
                    "country_code": "BBB",
                    "crop_code": "mango",
                    "candidate_rank": 2,
                    "dest_type": "airport",
                    "dest_rank": 1,
                    "dest_id": 12,
                    "cluster_cell_count": 10,
                    "total_travel_time_h": 5.0,
                    "surface_group": "paved",
                    "surface_travel_time_pct": 0.90,
                },
            ]
        )

        summary = MODULE.aggregate_route_mix(frame)
        aaa = summary.loc[summary["country_code"].eq("AAA")].iloc[0]

        self.assertAlmostEqual(float(aaa["weighted_baseline_travel_time_h"]), 13.3333333333, places=6)
        self.assertAlmostEqual(float(aaa["unpaved_time_share"]), 0.2666666666, places=6)
        self.assertAlmostEqual(float(aaa["unknown_time_share"]), 0.0666666666, places=6)
        self.assertAlmostEqual(float(aaa["synthetic_time_share"]), 0.1666666666, places=6)
        self.assertEqual(int(aaa["route_count"]), 2)

    def test_standardized_regression_returns_positive_fit_and_coefficients(self) -> None:
        frame = pd.DataFrame(
            {
                "rain": [1.0, 2.0, 3.0, 4.0, 5.0],
                "remote": [2.0, 1.0, 4.0, 3.0, 5.0],
                "unpaved": [1.0, 1.5, 2.0, 2.5, 3.0],
                "burden": [10.0, 12.0, 17.0, 19.0, 25.0],
            }
        )

        result = MODULE.run_standardized_regression(frame, "burden", ["rain", "remote", "unpaved"])

        self.assertIn("r_squared", result)
        self.assertGreater(float(result["r_squared"]), 0.0)
        self.assertEqual(len(result["coefficients"]), 3)
        self.assertTrue(all("predictor" in row and "beta" in row for row in result["coefficients"]))

    def test_bootstrap_spearman_summary_returns_ci_and_direction(self) -> None:
        frame = pd.DataFrame(
            {
                "x": np.arange(1, 16, dtype=float),
                "y": np.arange(1, 16, dtype=float) * 2.0,
            }
        )

        summary = MODULE.bootstrap_spearman_summary(frame, "x", "y", n_boot=200, seed=7)

        self.assertGreater(summary["rho"], 0.9)
        self.assertGreater(summary["ci_low"], 0.7)
        self.assertGreater(summary["ci_high"], 0.9)

    def test_coefficient_summary_marks_ci_excluding_zero_as_supported(self) -> None:
        model = {
            "coefficients": [
                {"predictor": "rain", "beta": 0.4},
                {"predictor": "remote", "beta": -0.1},
            ]
        }
        boot = pd.DataFrame(
            {
                "rain": [0.2, 0.3, 0.4, 0.5, 0.6],
                "remote": [-0.2, -0.1, 0.0, 0.1, 0.2],
            }
        )

        summary = MODULE.summarize_bootstrap_coefficients(model, boot)
        rain = next(row for row in summary if row["predictor"] == "rain")
        remote = next(row for row in summary if row["predictor"] == "remote")

        self.assertTrue(rain["supported"])
        self.assertFalse(remote["supported"])

    def test_new_mechanism_plots_write_png_outputs(self) -> None:
        full = pd.DataFrame(
            {
                "country_code": ["AAA", "BBB", "CCC", "DDD"],
                "total_burden_h": [1000.0, 300.0, 20.0, 0.0],
                "threshold_impact_ratio_actual": [1.5, 1.2, 0.2, 0.05],
                "weighted_baseline_travel_time_h": [20.0, 8.0, 15.0, 3.0],
                "actual_unpaved_time_share": [0.25, 0.10, 0.60, 0.02],
            }
        )
        fitted = full.assign(
            log_burden_h=np.log1p(full["total_burden_h"]),
            log_threshold_impact=np.log1p(full["threshold_impact_ratio_actual"]),
            log_remoteness_h=np.log1p(full["weighted_baseline_travel_time_h"]),
            fitted_z=[1.0, 0.4, -0.2, -1.2],
            residual_z=[0.8, -0.1, 0.2, -0.9],
        )
        model = {"r_squared": 0.7, "adjusted_r_squared": 0.55}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outputs = [
                MODULE.plot_rain_delay_quadrants(full, tmp_path / "quadrants.png"),
                MODULE.plot_mechanism_ladder(full, tmp_path / "ladder.png"),
                MODULE.plot_residual_case_bars(fitted, model, tmp_path / "residuals.png"),
            ]

            for output in outputs:
                path = Path(output["path"])
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

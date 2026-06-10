from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path("/Users/gk/Code/super-duper-disser/scripts/classify_service_access_failures.py")
SPEC = importlib.util.spec_from_file_location("classify_service_access_failures", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ClassifyServiceAccessFailuresTest(unittest.TestCase):
    def test_maps_subdir_name_for_home_patterns(self) -> None:
        name = MODULE._maps_subdir_name(["Broken Grid", "Sparse"])
        self.assertEqual(name, "maps_home_patterns_broken_grid_sparse")

    def test_panel_summary_text_wraps_and_uses_short_labels(self) -> None:
        counts = pd.Series(
            {
                "ok_walk": 10,
                "failed_access_gt_threshold": 20,
                "failed_egress_gt_threshold": 5,
                "failed_access_egress_sum_gt_threshold": 3,
                "failed_in_vehicle_gt_threshold": 2,
            }
        )
        text = MODULE._panel_summary_text(counts, max_line_chars=30)
        self.assertIn("walk=10", text)
        self.assertIn("home_stop>T=20", text)
        self.assertIn("both_walks>T=3", text)
        self.assertIn("\n", text)

    def test_ok_walk(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=12.0,
            pt_total_min=18.0,
            access_walk_time_min=8.0,
            egress_walk_time_min=2.0,
            in_vehicle_time_min=5.0,
            transfer_time_min=3.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "ok_walk")

    def test_ok_pt_only(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=24.0,
            pt_total_min=14.0,
            access_walk_time_min=7.0,
            egress_walk_time_min=2.0,
            in_vehicle_time_min=4.0,
            transfer_time_min=1.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "ok_pt_only")

    def test_failed_access_gt_threshold(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=30.0,
            pt_total_min=26.0,
            access_walk_time_min=16.0,
            egress_walk_time_min=4.0,
            in_vehicle_time_min=5.0,
            transfer_time_min=1.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "failed_access_gt_threshold")

    def test_failed_egress_gt_threshold(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=30.0,
            pt_total_min=26.0,
            access_walk_time_min=5.0,
            egress_walk_time_min=16.0,
            in_vehicle_time_min=4.0,
            transfer_time_min=1.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "failed_egress_gt_threshold")

    def test_failed_in_vehicle_gt_threshold(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=35.0,
            pt_total_min=26.0,
            access_walk_time_min=5.0,
            egress_walk_time_min=3.0,
            in_vehicle_time_min=17.0,
            transfer_time_min=1.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "failed_in_vehicle_gt_threshold")

    def test_failed_transfer_gt_threshold(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=35.0,
            pt_total_min=26.0,
            access_walk_time_min=5.0,
            egress_walk_time_min=3.0,
            in_vehicle_time_min=4.0,
            transfer_time_min=16.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "failed_transfer_gt_threshold")

    def test_failed_access_egress_sum_gt_threshold(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=35.0,
            pt_total_min=20.0,
            access_walk_time_min=9.0,
            egress_walk_time_min=8.0,
            in_vehicle_time_min=2.0,
            transfer_time_min=1.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "failed_access_egress_sum_gt_threshold")

    def test_failed_multi_component_gt_threshold(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=35.0,
            pt_total_min=40.0,
            access_walk_time_min=16.0,
            egress_walk_time_min=17.0,
            in_vehicle_time_min=4.0,
            transfer_time_min=3.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "failed_multi_component_gt_threshold")

    def test_failed_total_gt_threshold_without_single_component_breach(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=35.0,
            pt_total_min=20.0,
            access_walk_time_min=7.0,
            egress_walk_time_min=4.0,
            in_vehicle_time_min=6.0,
            transfer_time_min=3.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "failed_total_gt_threshold_no_single_component_gt_threshold")

    def test_pt_unavailable_is_failed_no_pt_path(self) -> None:
        label = MODULE._classify_access_failure(
            walk_only_min=35.0,
            pt_total_min=float("inf"),
            access_walk_time_min=0.0,
            egress_walk_time_min=0.0,
            in_vehicle_time_min=0.0,
            transfer_time_min=0.0,
            threshold_min=15.0,
        )
        self.assertEqual(label, "failed_no_pt_path")

    def test_effective_pt_total_uses_inf_when_path_missing(self) -> None:
        total = MODULE._effective_pt_total_min(
            pt_time_min=float("inf"),
            pt_total_decomposed_time_min=4.0,
        )
        self.assertEqual(total, float("inf"))


if __name__ == "__main__":
    unittest.main()

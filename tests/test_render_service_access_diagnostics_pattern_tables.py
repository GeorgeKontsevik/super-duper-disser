from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path("/Users/gk/Code/super-duper-disser/scripts/render_service_access_diagnostics_pattern_tables.py")
SPEC = importlib.util.spec_from_file_location("render_service_access_diagnostics_pattern_tables", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ServiceAccessPatternTablesTests(unittest.TestCase):
    def test_service_matrix_row_normalizes_and_orders_patterns(self) -> None:
        df = pd.DataFrame(
            {
                "service_name": ["hospital"] * 6,
                "access_diagnosis_label": [
                    "ok_walk",
                    "ok_pt_only",
                    "failed_multi_component_gt_threshold",
                    "ok_walk",
                    "failed_no_pt_path",
                    "ok_walk",
                ],
                "home_street_pattern_class": [
                    "Broken Grid",
                    "Broken Grid",
                    "Regular Grid",
                    "Regular Grid",
                    "Warped Parallel",
                    None,
                ],
            }
        )
        counts, shares, excluded_unknown = MODULE._service_matrix(df, "hospital")
        self.assertEqual(excluded_unknown, 1)
        self.assertEqual(list(counts.index), ["Regular Grid", "Warped Parallel", "Broken Grid"])
        self.assertEqual(list(counts.columns), MODULE.LABEL_ORDER)
        self.assertAlmostEqual(float(shares.loc["Broken Grid"].sum()), 1.0)
        self.assertAlmostEqual(float(shares.loc["Regular Grid"].sum()), 1.0)
        self.assertEqual(int(counts.loc["Warped Parallel", "failed_no_pt_path"]), 1)

    def test_render_service_heatmap_writes_png(self) -> None:
        counts = pd.DataFrame(
            [[10, 5, 1, 2, 0, 1, 0, 1, 0, 0], [2, 1, 0, 3, 4, 0, 1, 0, 0, 0]],
            index=["Regular Grid", "Irregular Grid"],
            columns=MODULE.LABEL_ORDER,
        )
        shares = counts.div(counts.sum(axis=1), axis=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "heatmap.png"
            MODULE._render_service_heatmap(
                "hospital",
                counts,
                shares,
                excluded_unknown=7,
                out_path=out_path,
                title_suffix="label share by home street pattern",
                colorbar_label="share within home street pattern",
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_render_combined_service_heatmaps_writes_png(self) -> None:
        counts = pd.DataFrame(
            [[10, 5, 1, 2, 0, 1, 0, 1, 0, 0], [2, 1, 0, 3, 4, 0, 1, 0, 0, 0]],
            index=["Regular Grid", "Irregular Grid"],
            columns=MODULE.LABEL_ORDER,
        )
        shares = counts.div(counts.sum(axis=1), axis=0)
        matrices = [("hospital", counts, shares, 2), ("school", counts, shares, 0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "combined.png"
            MODULE._render_combined_service_heatmaps(
                matrices,
                out_path=out_path,
                title_suffix="Combined",
                colorbar_label="share",
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_service_context_matrix_row_normalizes_and_orders_patterns(self) -> None:
        df = pd.DataFrame(
            {
                "service_name": ["hospital"] * 6,
                "access_diagnosis_label": [
                    "ok_walk",
                    "ok_pt_only",
                    "failed_multi_component_gt_threshold",
                    "ok_walk",
                    "failed_no_pt_path",
                    "ok_walk",
                ],
                "service_street_pattern_class": [
                    "Broken Grid",
                    "Broken Grid",
                    "Regular Grid",
                    "Regular Grid",
                    "Warped Parallel",
                    None,
                ],
            }
        )
        counts, shares, excluded_unknown = MODULE._service_matrix(
            df,
            "hospital",
            pattern_column="service_street_pattern_class",
        )
        self.assertEqual(excluded_unknown, 1)
        self.assertEqual(list(counts.index), ["Regular Grid", "Warped Parallel", "Broken Grid"])
        self.assertEqual(list(counts.columns), MODULE.LABEL_ORDER)
        self.assertAlmostEqual(float(shares.loc["Broken Grid"].sum()), 1.0)
        self.assertEqual(int(counts.loc["Warped Parallel", "failed_no_pt_path"]), 1)

    def test_pair_matrix_builds_expected_row_labels_and_normalizes(self) -> None:
        df = pd.DataFrame(
            {
                "service_name": ["school"] * 5,
                "access_diagnosis_label": [
                    "ok_walk",
                    "failed_access_gt_threshold",
                    "failed_egress_gt_threshold",
                    "failed_access_gt_threshold",
                    "ok_pt_only",
                ],
                "home_street_pattern_class": [
                    "Regular Grid",
                    "Regular Grid",
                    "Broken Grid",
                    "Broken Grid",
                    None,
                ],
                "service_street_pattern_class": [
                    "Warped Parallel",
                    "Warped Parallel",
                    "Regular Grid",
                    "Regular Grid",
                    "Sparse",
                ],
            }
        )
        counts, shares, excluded_unknown = MODULE._pair_matrix(df, "school")
        self.assertEqual(excluded_unknown, 1)
        self.assertEqual(
            list(counts.index),
            ["Regular Grid -> Warped Parallel", "Broken Grid -> Regular Grid"],
        )
        self.assertAlmostEqual(float(shares.loc["Regular Grid -> Warped Parallel"].sum()), 1.0)
        self.assertEqual(int(counts.loc["Broken Grid -> Regular Grid", "failed_egress_gt_threshold"]), 1)


if __name__ == "__main__":
    unittest.main()

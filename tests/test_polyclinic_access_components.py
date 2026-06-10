from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd
import tempfile


MODULE_PATH = Path(
    "/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/run_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("polyclinic_access_components", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PolyclinicAccessComponentsTests(unittest.TestCase):
    def test_add_component_flags_marks_threshold_pass_fail(self) -> None:
        df = pd.DataFrame(
            {
                "walk_time_min": [10.0, 20.0],
                "effective_pt_total_min": [12.0, 22.0],
                "access_walk_time_min": [5.0, 16.0],
                "egress_walk_time_min": [7.0, 13.0],
                "transport_time_min": [3.0, 17.0],
                "transfer_time_min": [0.0, 2.0],
            }
        )

        flagged = MODULE._add_component_flags(df, threshold_min=15.0)

        self.assertEqual(flagged["walk_direct_ok"].tolist(), [True, False])
        self.assertEqual(flagged["pt_total_ok"].tolist(), [True, False])
        self.assertEqual(flagged["access_ok"].tolist(), [True, False])
        self.assertEqual(flagged["egress_ok"].tolist(), [True, True])
        self.assertEqual(flagged["in_vehicle_ok"].tolist(), [True, False])
        self.assertEqual(flagged["transfer_ok"].tolist(), [True, True])
        self.assertEqual(flagged["access_egress_sum_ok"].tolist(), [True, False])

    def test_build_component_summary_returns_share_ok_by_city(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["a", "a", "b"],
                "walk_direct_ok": [True, False, False],
                "pt_total_ok": [True, True, False],
                "access_ok": [True, False, True],
                "egress_ok": [True, True, False],
                "in_vehicle_ok": [True, True, False],
                "transfer_ok": [True, True, True],
                "access_egress_sum_ok": [True, False, False],
            }
        )

        summary = MODULE._build_component_summary(df)

        city_a_walk = summary[(summary["city"] == "a") & (summary["component"] == "walk_direct_ok")].iloc[0]
        city_b_pt = summary[(summary["city"] == "b") & (summary["component"] == "pt_total_ok")].iloc[0]
        self.assertEqual(int(city_a_walk["n"]), 2)
        self.assertAlmostEqual(float(city_a_walk["share_ok"]), 0.5)
        self.assertEqual(int(city_b_pt["ok_count"]), 0)
        self.assertAlmostEqual(float(city_b_pt["share_ok"]), 0.0)

    def test_build_overall_summary_uses_ok_labels(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "access_diagnosis_label": [
                    "ok_walk",
                    "failed_access_gt_threshold",
                    "ok_pt_only",
                    "failed_in_vehicle_gt_threshold",
                ],
            }
        )

        summary = MODULE._build_overall_summary(df)

        city_a = summary[summary["city"] == "a"].iloc[0]
        city_b = summary[summary["city"] == "b"].iloc[0]
        self.assertEqual(int(city_a["ok_count"]), 1)
        self.assertEqual(int(city_a["not_ok_count"]), 1)
        self.assertAlmostEqual(float(city_a["share_ok"]), 0.5)
        self.assertEqual(int(city_b["ok_count"]), 1)
        self.assertEqual(int(city_b["not_ok_count"]), 1)

    def test_build_requested_summary_matches_requested_buckets(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["a", "a", "a", "a", "a"],
                "access_diagnosis_label": [
                    "ok_walk",
                    "ok_pt_only",
                    "failed_access_gt_threshold",
                    "failed_in_vehicle_gt_threshold",
                    "failed_total_gt_threshold_no_single_component_gt_threshold",
                ],
                "walk_direct_ok": [True, False, False, False, False],
                "pt_total_ok": [True, True, False, False, False],
                "access_ok": [True, True, False, True, True],
                "in_vehicle_ok": [True, True, True, False, True],
                "egress_ok": [True, True, True, True, True],
            }
        )

        summary = MODULE._build_requested_summary(df)
        row = summary.iloc[0]

        self.assertAlmostEqual(float(row["pct_ok_overall"]), 0.4)
        self.assertAlmostEqual(float(row["pct_not_ok_overall"]), 0.6)
        self.assertAlmostEqual(float(row["pct_ok_walk_only"]), 0.2)
        self.assertAlmostEqual(float(row["pct_ok_walk_plus_pt"]), 0.2)
        self.assertAlmostEqual(float(row["pct_not_ok_home_to_stop_overall"]), 0.2)
        self.assertAlmostEqual(float(row["pct_not_ok_pt_only_overall"]), 0.2)
        self.assertAlmostEqual(float(row["pct_not_ok_stop_to_service_overall"]), 0.0)
        self.assertAlmostEqual(float(row["pct_not_ok_both_walks_overall"]), 0.0)
        self.assertAlmostEqual(float(row["pct_not_ok_multi_component_overall"]), 0.0)
        self.assertAlmostEqual(float(row["pct_not_ok_sum_no_single_overall"]), 0.2)
        self.assertAlmostEqual(
            float(row["pct_not_ok_home_to_stop_overall"])
            + float(row["pct_not_ok_pt_only_overall"])
            + float(row["pct_not_ok_stop_to_service_overall"])
            + float(row["pct_not_ok_both_walks_overall"])
            + float(row["pct_not_ok_multi_component_overall"])
            + float(row["pct_not_ok_sum_no_single_overall"]),
            float(row["pct_not_ok_overall"]),
        )

    def test_round_for_export_rounds_numeric_columns_to_3_digits(self) -> None:
        df = pd.DataFrame({"a": [0.12349], "b": [2], "c": ["x"]})
        rounded = MODULE._round_for_export(df, digits=3)
        self.assertEqual(float(rounded.loc[0, "a"]), 0.123)
        self.assertEqual(float(rounded.loc[0, "b"]), 2.0)
        self.assertEqual(rounded.loc[0, "c"], "x")

    def test_requested_overall_sections_split_into_three_groups(self) -> None:
        row = pd.Series(
            {
                "pct_ok_overall": 0.6,
                "pct_not_ok_overall": 0.4,
                "pct_ok_walk_only": 0.5,
                "pct_ok_walk_plus_pt": 0.1,
                "pct_not_ok_home_to_stop_overall": 0.2,
                "pct_not_ok_pt_only_overall": 0.09,
                "pct_not_ok_stop_to_service_overall": 0.01,
                "pct_not_ok_both_walks_overall": 0.03,
                "pct_not_ok_multi_component_overall": 0.04,
                "pct_not_ok_sum_no_single_overall": 0.03,
            }
        )
        sections = MODULE._requested_overall_sections(row)
        self.assertEqual(len(sections), 4)
        self.assertEqual(sections[0][0], "Overall")
        self.assertEqual(len(sections[0][1]), 2)
        self.assertEqual(sections[1][0], "OK breakdown")
        self.assertEqual(len(sections[1][1]), 2)
        self.assertEqual(sections[2][0], "Single-component not OK")
        self.assertEqual(len(sections[2][1]), 3)
        self.assertEqual(sections[3][0], "Combined / multi-component not OK")
        self.assertEqual(len(sections[3][1]), 3)

    def test_build_single_component_pattern_summaries_uses_expected_pattern_contexts(self) -> None:
        df = pd.DataFrame(
            {
                "access_diagnosis_label": [
                    "failed_access_gt_threshold",
                    "failed_access_gt_threshold",
                    "failed_in_vehicle_gt_threshold",
                    "failed_egress_gt_threshold",
                ],
                "home_street_pattern_class": ["Regular Grid", "Regular Grid", "Broken Grid", "Irregular Grid"],
                "service_street_pattern_class": ["Sparse", "Sparse", "Warped Parallel", "Loops & Lollipops"],
            }
        )
        summaries = MODULE._build_single_component_pattern_summaries(df)

        access = summaries["home_to_stop_not_ok"]
        self.assertEqual(access.iloc[0]["pattern_context"], "home")
        self.assertEqual(access.iloc[0]["pattern_value"], "Regular Grid")
        self.assertAlmostEqual(float(access.iloc[0]["share"]), 1.0)

        pt_segment = summaries["pt_segment_not_ok"]
        self.assertTrue(pt_segment.empty)
        self.assertEqual(list(pt_segment.columns), ["pattern_context", "pattern_value", "n", "share"])

        egress = summaries["stop_to_service_not_ok"]
        self.assertEqual(egress.iloc[0]["pattern_context"], "service")
        self.assertEqual(egress.iloc[0]["pattern_value"], "Loops & Lollipops")

    def test_build_single_component_pattern_summaries_by_city_returns_city_shares(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "access_diagnosis_label": [
                    "failed_access_gt_threshold",
                    "failed_access_gt_threshold",
                    "failed_access_gt_threshold",
                    "failed_egress_gt_threshold",
                ],
                "home_street_pattern_class": ["Regular Grid", "Broken Grid", "Regular Grid", "Sparse"],
                "service_street_pattern_class": ["Sparse", "Sparse", "Warped Parallel", "Loops & Lollipops"],
            }
        )
        summaries = MODULE._build_single_component_pattern_summaries_by_city(df)
        access = summaries["home_to_stop_not_ok"]
        city_a = access[access["city"] == "a"].sort_values("pattern_value").reset_index(drop=True)
        self.assertEqual(city_a["pattern_value"].tolist(), ["Broken Grid", "Regular Grid"])
        self.assertEqual(city_a["share"].tolist(), [0.5, 0.5])
        egress = summaries["stop_to_service_not_ok"]
        self.assertEqual(egress.iloc[0]["city"], "b")
        self.assertAlmostEqual(float(egress.iloc[0]["share"]), 1.0)

    def test_build_single_component_pattern_raw_returns_unaggregated_rows(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["a", "a"],
                "access_diagnosis_label": ["failed_access_gt_threshold", "failed_access_gt_threshold"],
                "home_street_pattern_class": ["Regular Grid", "Broken Grid"],
                "service_street_pattern_class": ["Sparse", "Sparse"],
            }
        )
        raw = MODULE._build_single_component_pattern_raw(df, "home_to_stop_not_ok")
        self.assertEqual(list(raw.columns), MODULE.PATTERN_RAW_COLUMNS)
        self.assertEqual(raw["pattern_value"].tolist(), ["Regular Grid", "Broken Grid"])
        self.assertEqual(raw["weight"].tolist(), [1.0, 1.0])

    def test_aggregate_pattern_raw_supports_overall_and_by_city(self) -> None:
        raw = pd.DataFrame(
            {
                "city": ["a", "a", "b"],
                "pattern_context": ["home", "home", "home"],
                "pattern_value": ["X", "Y", "X"],
                "weight": [1.0, 3.0, 2.0],
            }
        )
        overall = MODULE._aggregate_pattern_raw(raw)
        self.assertEqual(overall.iloc[0]["pattern_value"], "X")
        self.assertAlmostEqual(float(overall.iloc[0]["n"]), 3.0)
        by_city = MODULE._aggregate_pattern_raw(raw, by_city=True)
        city_a_x = by_city[(by_city["city"] == "a") & (by_city["pattern_value"] == "X")].iloc[0]
        self.assertAlmostEqual(float(city_a_x["share"]), 0.25)

    def test_complete_city_pattern_summary_adds_missing_cities_with_zeros(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["b", "b"],
                "pattern_context": ["pt_path", "pt_path"],
                "pattern_value": ["X", "Y"],
                "n": [2.0, 1.0],
                "share": [2 / 3, 1 / 3],
            }
        )
        completed = MODULE._complete_city_pattern_summary(df, ["a", "b", "c"])
        self.assertEqual(completed["city"].drop_duplicates().tolist(), ["a", "b", "c"])
        city_a = completed[completed["city"] == "a"].sort_values("pattern_value").reset_index(drop=True)
        self.assertEqual(city_a["n"].tolist(), [0.0, 0.0])
        self.assertEqual(city_a["share"].tolist(), [0.0, 0.0])

    def test_aggregate_city_share_mean_averages_city_percentages(self) -> None:
        city_df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "pattern_context": ["city_street_pattern"] * 4,
                "pattern_value": ["X", "Y", "X", "Y"],
                "n": [1.0, 3.0, 9.0, 1.0],
                "share": [0.25, 0.75, 0.9, 0.1],
            }
        )
        overall = MODULE._aggregate_city_share_mean(city_df)
        x = overall[overall["pattern_value"] == "X"].iloc[0]
        y = overall[overall["pattern_value"] == "Y"].iloc[0]
        self.assertAlmostEqual(float(x["share"]), 0.575)
        self.assertAlmostEqual(float(y["share"]), 0.425)

    def test_build_pt_route_pattern_summary_aggregates_length_share(self) -> None:
        df = pd.DataFrame(
            {
                "street_pattern_class": ["A", "B", "A"],
                "pt_length_m": [60.0, 30.0, 10.0],
            }
        )
        summary = MODULE._build_pt_route_pattern_summary(df)
        self.assertEqual(summary.iloc[0]["pattern_value"], "A")
        self.assertAlmostEqual(float(summary.iloc[0]["share"]), 0.7)
        self.assertAlmostEqual(float(summary.iloc[1]["share"]), 0.3)

    def test_expand_path_route_patterns_allocates_route_time_by_class_share(self) -> None:
        path_routes = pd.DataFrame(
            {
                "city": ["x", "x"],
                "route_label": ["R1", "R2"],
                "route_time_min": [10.0, 5.0],
            }
        )
        route_class = pd.DataFrame(
            {
                "city": ["x", "x", "x"],
                "route_label": ["R1", "R1", "R2"],
                "street_pattern_class": ["A", "B", "B"],
                "route_class_share": [0.6, 0.4, 1.0],
            }
        )
        expanded = MODULE._expand_path_route_patterns(path_routes, route_class)
        grouped = expanded.groupby("street_pattern_class", as_index=False)["allocated_route_time_min"].sum()
        a = grouped[grouped["street_pattern_class"] == "A"].iloc[0]
        b = grouped[grouped["street_pattern_class"] == "B"].iloc[0]
        self.assertAlmostEqual(float(a["allocated_route_time_min"]), 6.0)
        self.assertAlmostEqual(float(b["allocated_route_time_min"]), 9.0)

    def test_filter_polyclinic_diagnostics_by_labels(self) -> None:
        df = pd.DataFrame(
            {
                "service_name": ["polyclinic", "polyclinic", "hospital"],
                "access_diagnosis_label": [
                    "failed_total_gt_threshold_no_single_component_gt_threshold",
                    "failed_in_vehicle_gt_threshold",
                    "failed_total_gt_threshold_no_single_component_gt_threshold",
                ],
                "city": ["a", "a", "a"],
                "building_idx": [1, 2, 3],
            }
        )
        out = MODULE._filter_polyclinic_diagnostics_by_labels(
            df,
            {"failed_total_gt_threshold_no_single_component_gt_threshold"},
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.iloc[0]["building_idx"]), 1)

    def test_filter_polyclinic_diagnostics_by_labels_none_returns_all_polyclinic(self) -> None:
        df = pd.DataFrame(
            {
                "service_name": ["polyclinic", "polyclinic", "hospital"],
                "access_diagnosis_label": ["ok_walk", "failed_in_vehicle_gt_threshold", "ok_walk"],
                "city": ["a", "a", "a"],
                "building_idx": [1, 2, 3],
            }
        )
        out = MODULE._filter_polyclinic_diagnostics_by_labels(df, None)
        self.assertEqual(out["building_idx"].tolist(), [1, 2])

    def test_render_combined_single_component_patterns_png_writes_file(self) -> None:
        summaries = {
            "home_to_stop_not_ok": pd.DataFrame(
                {"pattern_value": ["A", "B"], "share": [0.6, 0.4]}
            ),
            "pt_segment_not_ok": pd.DataFrame(
                {"pattern_value": ["A -> X", "B -> Y"], "share": [0.7, 0.3]}
            ),
            "stop_to_service_not_ok": pd.DataFrame(
                {"pattern_value": ["X", "Y"], "share": [0.8, 0.2]}
            ),
            "sum_no_single_component_not_ok": pd.DataFrame(
                {"pattern_value": ["A", "B"], "share": [0.55, 0.45]}
            ),
            "all_polyclinic_pt_paths": pd.DataFrame(
                {"pattern_value": ["A", "B"], "share": [0.65, 0.35]}
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "combined.png"
            MODULE._render_combined_single_component_patterns_png(summaries, out_path=out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_render_city_pattern_heatmap_png_writes_file(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "pattern_value": ["X", "Y", "X", "Y"],
                "share": [0.7, 0.3, 0.4, 0.6],
                "n": [7, 3, 4, 6],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "city_heatmap.png"
            MODULE._render_city_pattern_heatmap_png(df, title="City heatmap", out_path=out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_render_stacked_city_pattern_heatmaps_png_writes_file(self) -> None:
        top_df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "pattern_value": ["X", "Y", "X", "Y"],
                "share": [0.7, 0.3, 0.4, 0.6],
                "n": [7, 3, 4, 6],
            }
        )
        bottom_df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "pattern_value": ["X", "Y", "X", "Y"],
                "share": [0.5, 0.5, 0.2, 0.8],
                "n": [5, 5, 2, 8],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "stacked_city_heatmap.png"
            MODULE._render_stacked_city_pattern_heatmaps_png(
                top_df,
                bottom_df,
                top_title="Top",
                bottom_title="Bottom",
                out_path=out_path,
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_render_overall_and_city_pattern_png_writes_file(self) -> None:
        overall_df = pd.DataFrame(
            {
                "pattern_value": ["X", "Y"],
                "share": [0.7, 0.3],
                "n": [7, 3],
            }
        )
        city_df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "pattern_value": ["X", "Y", "X", "Y"],
                "share": [0.7, 0.3, 0.4, 0.6],
                "n": [7, 3, 4, 6],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "overall_and_city.png"
            MODULE._render_overall_and_city_pattern_png(
                overall_df,
                city_df,
                title="Test",
                out_path=out_path,
                city_order=["a", "b"],
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_render_pt_overall_and_city_pattern_png_writes_file(self) -> None:
        overall_df = pd.DataFrame(
            {
                "pattern_value": ["X", "Y"],
                "share": [0.6, 0.4],
                "n": [6, 4],
            }
        )
        case_city_df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "pattern_value": ["X", "Y", "X", "Y"],
                "share": [0.8, 0.2, 0.3, 0.7],
                "n": [8, 2, 3, 7],
            }
        )
        baseline_city_df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "pattern_value": ["X", "Y", "X", "Y"],
                "share": [0.5, 0.5, 0.4, 0.6],
                "n": [5, 5, 4, 6],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "pt_overall_and_city.png"
            MODULE._render_pt_overall_and_city_pattern_png(
                overall_df,
                case_city_df,
                baseline_city_df,
                title="PT case",
                baseline_title="PT baseline",
                out_path=out_path,
                city_order=["a", "b"],
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_build_city_street_pattern_raw_from_cells(self) -> None:
        df = pd.DataFrame(
            {
                "top1_class_name": ["A", "A", "B", None],
            }
        )
        raw = MODULE._build_city_street_pattern_raw_from_cells(df, city="x")
        self.assertEqual(raw["city"].tolist(), ["x", "x", "x", "x"])
        self.assertEqual(raw["pattern_value"].tolist(), ["A", "A", "B", "UNKNOWN"])
        self.assertEqual(raw["weight"].tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_render_four_panel_pattern_png_writes_file(self) -> None:
        top_overall_df = pd.DataFrame(
            {
                "pattern_value": ["X", "Y"],
                "share": [0.6, 0.4],
                "n": [6, 4],
            }
        )
        upper_city_df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "pattern_value": ["X", "Y", "X", "Y"],
                "share": [0.8, 0.2, 0.3, 0.7],
                "n": [8, 2, 3, 7],
            }
        )
        lower_overall_df = pd.DataFrame(
            {
                "pattern_value": ["X", "Y"],
                "share": [0.55, 0.45],
                "n": [5.5, 4.5],
            }
        )
        lower_city_df = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "pattern_value": ["X", "Y", "X", "Y"],
                "share": [0.5, 0.5, 0.4, 0.6],
                "n": [5, 5, 4, 6],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "four_panel.png"
            MODULE._render_four_panel_pattern_png(
                top_overall_df,
                upper_city_df,
                lower_overall_df,
                lower_city_df,
                top_title="Top",
                upper_city_title="Upper city",
                lower_overall_title="Lower overall",
                lower_city_title="Lower city",
                out_path=out_path,
                city_order=["a", "b"],
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)

    def test_combined_pattern_layout_groups_have_shared_colors_and_gaps(self) -> None:
        specs = MODULE._combined_pattern_layout_specs()
        self.assertEqual([spec["key"] for spec in specs], [
            "home_to_stop_not_ok",
            "stop_to_service_not_ok",
            "pt_segment_not_ok",
            "sum_no_single_component_not_ok",
            "all_polyclinic_pt_paths",
        ])
        self.assertEqual(specs[0]["color"], specs[1]["color"])
        self.assertEqual(specs[2]["color"], specs[3]["color"])
        self.assertNotEqual(specs[1]["color"], specs[2]["color"])
        self.assertNotEqual(specs[3]["color"], specs[4]["color"])
        self.assertEqual([spec["grid_row"] for spec in specs], [0, 1, 3, 4, 6])


if __name__ == "__main__":
    unittest.main()

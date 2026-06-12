from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
import tempfile
import json
from shapely.geometry import Polygon


REPO_ROOT = Path("/Users/gk/Code/super-duper-disser")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = Path(
    "/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/run_experiments.py"
)
SPEC = importlib.util.spec_from_file_location("polyclinic_access_components", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CITY_LEVEL_MODULE_PATH = Path(
    "/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/city_level.py"
)
CITY_LEVEL_SPEC = importlib.util.spec_from_file_location("polyclinic_access_city_level", CITY_LEVEL_MODULE_PATH)
assert CITY_LEVEL_SPEC is not None and CITY_LEVEL_SPEC.loader is not None
CITY_LEVEL_MODULE = importlib.util.module_from_spec(CITY_LEVEL_SPEC)
CITY_LEVEL_SPEC.loader.exec_module(CITY_LEVEL_MODULE)

ACCESS_FIRST_MODULE_PATH = Path(
    "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/pipeline/run_pipeline2_accessibility_first.py"
)
ACCESS_FIRST_SPEC = importlib.util.spec_from_file_location("pipeline2_accessibility_first", ACCESS_FIRST_MODULE_PATH)
assert ACCESS_FIRST_SPEC is not None and ACCESS_FIRST_SPEC.loader is not None
ACCESS_FIRST_MODULE = importlib.util.module_from_spec(ACCESS_FIRST_SPEC)
ACCESS_FIRST_SPEC.loader.exec_module(ACCESS_FIRST_MODULE)


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


class PolyclinicCityLevelRegistryTests(unittest.TestCase):
    def test_build_default_city_registry_returns_deduped_large_sample(self) -> None:
        registry = CITY_LEVEL_MODULE.build_default_city_registry()

        self.assertEqual(len(registry), 36)
        self.assertEqual(registry["city"].nunique(), 36)
        self.assertIn("vienna_austria", registry["city"].tolist())
        self.assertIn("amsterdam_netherlands", registry["city"].tolist())
        self.assertNotIn("nouakchott_nouakchott_ouest_mauritania", registry["city"].tolist())
        self.assertNotIn("adelaide_south_australia_australia", registry[registry["source"] == "new5"]["city"].tolist())

    def test_build_default_city_registry_uses_expected_source_priority(self) -> None:
        registry = CITY_LEVEL_MODULE.build_default_city_registry()
        rows = registry.set_index("city")

        self.assertEqual(rows.loc["bergen_norway", "source"], "active19")
        self.assertEqual(rows.loc["amsterdam_netherlands", "source"], "new17")
        self.assertEqual(rows.loc["vienna_austria", "source"], "old23")

    def test_select_registry_subset_orders_by_placement_size_and_limit(self) -> None:
        registry = pd.DataFrame(
            {
                "city": ["large_city", "small_city", "medium_city"],
                "source": ["active19", "active19", "active19"],
                "source_priority": [0, 0, 0],
                "city_dir": ["/tmp/large", "/tmp/small", "/tmp/medium"],
                "placement_blocks_count": [300.0, 10.0, 100.0],
                "placement_demand_total": [3000.0, 100.0, 1000.0],
            }
        )

        subset = CITY_LEVEL_MODULE.select_registry_subset_for_tiered_run(registry, max_cities=2)

        self.assertEqual(subset["city"].tolist(), ["small_city", "medium_city"])

    def test_select_registry_subset_filters_explicit_city_list_after_sorting(self) -> None:
        registry = pd.DataFrame(
            {
                "city": ["large_city", "small_city", "medium_city"],
                "source": ["active19", "active19", "active19"],
                "source_priority": [0, 0, 0],
                "city_dir": ["/tmp/large", "/tmp/small", "/tmp/medium"],
                "placement_blocks_count": [300.0, 10.0, 100.0],
                "placement_demand_total": [3000.0, 100.0, 1000.0],
            }
        )

        subset = CITY_LEVEL_MODULE.select_registry_subset_for_tiered_run(registry, cities=["medium_city", "large_city"])

        self.assertEqual(subset["city"].tolist(), ["medium_city", "large_city"])

    def test_verify_city_registry_bundle_flags_core_artifacts(self) -> None:
        registry = CITY_LEVEL_MODULE.build_default_city_registry()
        verified = CITY_LEVEL_MODULE.verify_city_registry_bundle(registry)

        amsterdam = verified[verified["city"] == "amsterdam_netherlands"].iloc[0]
        bergen = verified[verified["city"] == "bergen_norway"].iloc[0]

        self.assertTrue(bool(bergen["has_street_pattern"]))
        self.assertTrue(bool(bergen["has_graph"]))
        self.assertTrue(bool(bergen["has_solver_blocks"]))
        self.assertTrue(bool(amsterdam["has_street_pattern"]))
        self.assertTrue(bool(amsterdam["has_graph"]))
        self.assertTrue(bool(amsterdam["has_solver_blocks"]))

    def test_build_city_level_baseline_coverage_aggregates_by_registry_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            active_path = tmp / "active.parquet"
            new_path = tmp / "new.parquet"

            pd.DataFrame(
                {
                    "city": ["bergen_norway", "bergen_norway", "ignored_city"],
                    "service_name": ["polyclinic", "polyclinic", "hospital"],
                    "access_diagnosis_label": ["ok_walk", "failed_access_gt_threshold", "ok_walk"],
                }
            ).to_parquet(active_path, index=False)
            pd.DataFrame(
                {
                    "city": ["amsterdam_netherlands", "amsterdam_netherlands", "vienna_austria"],
                    "service_name": ["polyclinic", "polyclinic", "polyclinic"],
                    "access_diagnosis_label": ["ok_pt_only", "failed_in_vehicle_gt_threshold", "ok_walk"],
                }
            ).to_parquet(new_path, index=False)

            registry = pd.DataFrame(
                {
                    "city": ["bergen_norway", "amsterdam_netherlands", "vienna_austria"],
                    "source": ["active19", "new17", "old23"],
                    "access_diagnostics_path": [str(active_path), str(new_path), str(new_path)],
                }
            )

            baseline = CITY_LEVEL_MODULE.build_city_level_baseline_coverage(registry)
            rows = baseline.set_index("city")

            self.assertAlmostEqual(float(rows.loc["bergen_norway", "coverage"]), 0.5)
            self.assertEqual(int(rows.loc["bergen_norway", "n_homes"]), 2)
            self.assertAlmostEqual(float(rows.loc["amsterdam_netherlands", "coverage"]), 0.5)
            self.assertAlmostEqual(float(rows.loc["vienna_austria", "coverage"]), 1.0)

    def test_load_solver_summary_fields_reads_gap_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_path = tmp / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "demand_total": 100.0,
                        "demand_without_total": 40.0,
                        "capacity_total": 200.0,
                        "provision_total": 0.6,
                    }
                ),
                encoding="utf-8",
            )
            row = CITY_LEVEL_MODULE.load_solver_summary_fields(summary_path)
            self.assertEqual(row["demand_total"], 100.0)
            self.assertEqual(row["accessibility_gap_total"], 40.0)
            self.assertEqual(row["capacity_total"], 200.0)
            self.assertEqual(row["provision_total"], 0.6)

    def test_load_street_pattern_mix_fields_converts_counts_to_shares(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_path = tmp / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "num_predictions": 10,
                        "class_counts": {
                            "Irregular Grid": 5,
                            "Loops & Lollipops": 3,
                            "Regular Grid": 2,
                            "Warped Parallel": 0,
                            "Broken Grid": 0,
                            "Sparse": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            row = CITY_LEVEL_MODULE.load_street_pattern_mix_fields(summary_path)
            self.assertEqual(row["street_pattern_cells"], 10.0)
            self.assertAlmostEqual(row["share_irregular_grid"], 0.5)
            self.assertAlmostEqual(row["share_loops_lollipops"], 0.3)
            self.assertAlmostEqual(row["share_regular_grid"], 0.2)

    def test_load_pt_descriptor_fields_counts_modalities_routes_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            connectpt_osm = tmp / "connectpt_osm"
            bus = connectpt_osm / "bus"
            tram = connectpt_osm / "tram"
            bus.mkdir(parents=True)
            tram.mkdir(parents=True)

            pd.DataFrame({"route_id": [1, 2]}).to_parquet(bus / "lines.parquet", index=False)
            pd.DataFrame({"stop_id": [1, 2, 3]}).to_parquet(bus / "aggregated_stops.parquet", index=False)
            pd.DataFrame({"route_id": [1]}).to_parquet(tram / "lines.parquet", index=False)
            pd.DataFrame({"stop_id": [1, 2]}).to_parquet(tram / "aggregated_stops.parquet", index=False)

            row = CITY_LEVEL_MODULE.load_pt_descriptor_fields(tmp)
            self.assertEqual(row["pt_modality_count"], 2.0)
            self.assertEqual(row["pt_route_count"], 3.0)
            self.assertEqual(row["pt_stop_count"], 5.0)

    def test_build_city_level_research_dataset_merges_all_layers(self) -> None:
        registry = pd.DataFrame(
            {
                "city": ["a_city"],
                "source": ["active19"],
                "source_priority": [0],
                "city_dir": ["/tmp/city"],
                "access_diagnostics_path": ["/tmp/access.parquet"],
            }
        )
        baseline = pd.DataFrame(
            {
                "city": ["a_city"],
                "source": ["active19"],
                "source_priority": [0],
                "city_dir": ["/tmp/city"],
                "access_diagnostics_path": ["/tmp/access.parquet"],
                "n_homes": [100],
                "ok_count": [70],
                "not_ok_count": [30],
                "coverage": [0.7],
            }
        )
        enriched = CITY_LEVEL_MODULE._merge_city_level_layers(
            registry=registry,
            baseline=baseline,
            solver_rows=pd.DataFrame([{"city": "a_city", "demand_total": 100.0, "accessibility_gap_total": 30.0}]),
            street_rows=pd.DataFrame([{"city": "a_city", "street_pattern_cells": 10.0, "share_irregular_grid": 0.5}]),
            pt_rows=pd.DataFrame([{"city": "a_city", "pt_modality_count": 2.0, "pt_route_count": 20.0}]),
        )
        row = enriched.iloc[0]
        self.assertAlmostEqual(float(row["coverage"]), 0.7)
        self.assertAlmostEqual(float(row["accessibility_gap_total"]), 30.0)
        self.assertAlmostEqual(float(row["share_irregular_grid"]), 0.5)
        self.assertAlmostEqual(float(row["pt_route_count"]), 20.0)
        self.assertAlmostEqual(float(row["accessibility_gap_share"]), 0.3)

    def test_build_research_question_association_summary_reports_spearman(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["a", "b", "c", "d"],
                "coverage": [0.1, 0.2, 0.3, 0.4],
                "accessibility_gap_share": [0.9, 0.8, 0.7, 0.6],
                "share_irregular_grid": [0.4, 0.3, 0.2, 0.1],
                "pt_route_count": [10, 20, 30, 40],
            }
        )
        summary = CITY_LEVEL_MODULE.build_research_question_association_summary(df)
        coverage_irregular = summary[
            (summary["outcome"] == "coverage") & (summary["predictor"] == "share_irregular_grid")
        ].iloc[0]
        coverage_pt = summary[
            (summary["outcome"] == "coverage") & (summary["predictor"] == "pt_route_count")
        ].iloc[0]
        self.assertAlmostEqual(float(coverage_irregular["spearman_rho"]), -1.0)
        self.assertAlmostEqual(float(coverage_pt["spearman_rho"]), 1.0)

    def test_build_city_level_target90_dataset_preserves_placement_failure_status(self) -> None:
        registry = pd.DataFrame(
            {
                "city": ["a_city"],
                "source": ["active19"],
                "source_priority": [0],
                "city_dir": ["/tmp/city"],
                "access_diagnostics_path": ["/tmp/access.parquet"],
            }
        )
        base = pd.DataFrame({"city": ["a_city"], "coverage": [0.2]})
        placement = pd.DataFrame(
            {
                "city": ["a_city"],
                "target_provision": [0.9],
                "baseline_provision": [0.1],
                "achieved_provision_after": [float("nan")],
                "additional_polyclinics_needed_to_0_9": [float("nan")],
                "selected_count_after": [float("nan")],
                "expanded_count_after": [float("nan")],
                "capacity_added_total": [float("nan")],
                "demand_without_after_total": [float("nan")],
                "demand_left_after_total": [float("nan")],
                "summary_after_path": [None],
                "status_preview_png": [None],
                "after_preview_png": [None],
                "demand_total": [100.0],
                "full_gap_total": [90.0],
                "target_unmet_total": [80.0],
                "target_fraction_of_full_gap": [0.888],
                "placement_status": ["failed"],
                "placement_error": ["solver failed"],
            }
        )
        original_base = CITY_LEVEL_MODULE.build_city_level_research_dataset
        original_placement = CITY_LEVEL_MODULE.build_targeted_placement_rows
        try:
            CITY_LEVEL_MODULE.build_city_level_research_dataset = lambda _registry: base
            CITY_LEVEL_MODULE.build_targeted_placement_rows = lambda *args, **kwargs: placement

            dataset = CITY_LEVEL_MODULE.build_city_level_target90_dataset(registry)
        finally:
            CITY_LEVEL_MODULE.build_city_level_research_dataset = original_base
            CITY_LEVEL_MODULE.build_targeted_placement_rows = original_placement

        row = dataset.iloc[0]
        self.assertEqual(row["placement_status"], "failed")
        self.assertEqual(row["placement_error"], "solver failed")

    def test_scale_unmet_demand_to_target_provision_hits_absolute_target(self) -> None:
        df = pd.DataFrame(
            {
                "demand": [50.0, 50.0],
                "demand_without": [20.0, 0.0],
                "demand_left": [10.0, 0.0],
            }
        )
        scaled = CITY_LEVEL_MODULE.scale_unmet_demand_to_target_provision(df, target_provision=0.9)
        self.assertAlmostEqual(float(scaled["target_unmet_total"]), 20.0)
        self.assertAlmostEqual(float(scaled["baseline_provision"]), 0.7)
        self.assertAlmostEqual(float(scaled["target_fraction_of_full_gap"]), 2 / 3)
        self.assertAlmostEqual(float(scaled["scaled_demand_without"].sum()), 13.333333333333334)
        self.assertAlmostEqual(float(scaled["scaled_demand_left"].sum()), 6.666666666666667)

    def test_scale_unmet_demand_to_target_provision_returns_zero_if_baseline_already_enough(self) -> None:
        df = pd.DataFrame(
            {
                "demand": [50.0, 50.0],
                "demand_without": [5.0, 0.0],
                "demand_left": [0.0, 0.0],
            }
        )
        scaled = CITY_LEVEL_MODULE.scale_unmet_demand_to_target_provision(df, target_provision=0.9)
        self.assertAlmostEqual(float(scaled["target_unmet_total"]), 0.0)
        self.assertAlmostEqual(float(scaled["target_fraction_of_full_gap"]), 0.0)
        self.assertAlmostEqual(float(scaled["scaled_demand_without"].sum()), 0.0)

    def test_build_placement_result_row_combines_summary_with_target_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            summary_path = tmp / "summary_after.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "new_count": 4,
                        "expanded_count": 0,
                        "selected_count": 11,
                        "demand_without_after_total": 7.0,
                        "demand_left_after_total": 3.0,
                        "capacity_added_total": 1200.0,
                        "files": {"status_preview_png": "/tmp/status.png"},
                    }
                ),
                encoding="utf-8",
            )
            row = CITY_LEVEL_MODULE.build_placement_result_row(
                city="a_city",
                summary_after_path=summary_path,
                demand_total=100.0,
                target_provision=0.9,
                baseline_provision=0.7,
            )
            self.assertEqual(row["city"], "a_city")
            self.assertEqual(row["additional_polyclinics_needed_to_0_9"], 4.0)
            self.assertAlmostEqual(float(row["achieved_provision_after"]), 0.9)
            self.assertAlmostEqual(float(row["target_provision"]), 0.9)
            self.assertEqual(row["status_preview_png"], "/tmp/status.png")

    def test_build_failed_placement_result_row_keeps_error_and_baseline_metrics(self) -> None:
        row = CITY_LEVEL_MODULE.build_failed_placement_result_row(
            city="a_city",
            error="Problem not solved: Infeasible.",
            demand_total=100.0,
            full_gap_total=30.0,
            baseline_provision=0.7,
            target_provision=0.9,
            target_unmet_total=20.0,
            target_fraction_of_full_gap=2 / 3,
        )
        self.assertEqual(row["city"], "a_city")
        self.assertEqual(row["placement_status"], "failed")
        self.assertIn("Infeasible", row["placement_error"])
        self.assertAlmostEqual(float(row["baseline_provision"]), 0.7)
        self.assertTrue(pd.isna(row["additional_polyclinics_needed_to_0_9"]))

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

    def test_transfer_street_pattern_cells_to_blocks_assigns_dominant_class(self) -> None:
        blocks = gpd.GeoDataFrame(
            {"geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ]},
            geometry="geometry",
            crs="EPSG:3857",
        )
        cells = gpd.GeoDataFrame(
            {
                "top1_class_name": ["Regular Grid", "Broken Grid", "Sparse"],
                "geometry": [
                    Polygon([(0, 0), (0.75, 0), (0.75, 1), (0, 1)]),
                    Polygon([(0.75, 0), (1, 0), (1, 1), (0.75, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:3857",
        )

        transferred = CITY_LEVEL_MODULE.transfer_street_pattern_cells_to_blocks(blocks, cells)

        self.assertEqual(transferred["block_name"].tolist(), ["0", "1"])
        self.assertEqual(
            transferred["street_pattern_dominant_class"].tolist(),
            ["Regular Grid", "Sparse"],
        )

    def test_build_city_target90_pattern_lift_rows_compares_selected_against_baselines(self) -> None:
        block_patterns = pd.DataFrame(
            {
                "block_name": ["0", "1", "2", "3", "4"],
                "street_pattern_dominant_class": ["Regular Grid", "Regular Grid", "Regular Grid", "Broken Grid", "Sparse"],
            }
        )

        rows = CITY_LEVEL_MODULE.build_city_target90_pattern_lift_rows(
            city="x",
            block_patterns=block_patterns,
            candidate_block_names=["0", "1", "3", "4"],
            selected_block_names=["3", "4"],
        )

        broken = rows[rows["street_pattern_dominant_class"] == "Broken Grid"].iloc[0]
        regular = rows[rows["street_pattern_dominant_class"] == "Regular Grid"].iloc[0]
        sparse = rows[rows["street_pattern_dominant_class"] == "Sparse"].iloc[0]

        self.assertAlmostEqual(float(broken["selected_share"]), 0.5)
        self.assertAlmostEqual(float(broken["city_share"]), 0.2)
        self.assertAlmostEqual(float(broken["candidate_share"]), 0.25)
        self.assertAlmostEqual(float(broken["placement_lift_vs_city"]), 0.3)
        self.assertAlmostEqual(float(broken["placement_lift_vs_candidates"]), 0.25)
        self.assertAlmostEqual(float(broken["placement_ratio_vs_city"]), 2.5)
        self.assertAlmostEqual(float(broken["placement_ratio_vs_candidates"]), 2.0)

        self.assertAlmostEqual(float(regular["selected_share"]), 0.0)
        self.assertAlmostEqual(float(regular["placement_lift_vs_city"]), -0.6)
        self.assertAlmostEqual(float(regular["placement_lift_vs_candidates"]), -0.5)

        self.assertAlmostEqual(float(sparse["selected_share"]), 0.5)
        self.assertAlmostEqual(float(sparse["placement_lift_vs_city"]), 0.3)
        self.assertAlmostEqual(float(sparse["placement_lift_vs_candidates"]), 0.25)

    def test_build_overall_target90_pattern_lift_rows_aggregates_counts_before_shares(self) -> None:
        detail = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "street_pattern_dominant_class": ["Regular Grid", "Broken Grid", "Regular Grid", "Broken Grid"],
                "city_count": [3.0, 1.0, 1.0, 3.0],
                "candidate_count": [2.0, 1.0, 1.0, 2.0],
                "selected_count": [0.0, 1.0, 1.0, 1.0],
            }
        )

        overall = CITY_LEVEL_MODULE.build_overall_target90_pattern_lift_rows(detail)

        broken = overall[overall["street_pattern_dominant_class"] == "Broken Grid"].iloc[0]
        regular = overall[overall["street_pattern_dominant_class"] == "Regular Grid"].iloc[0]
        self.assertAlmostEqual(float(broken["city_share"]), 0.5)
        self.assertAlmostEqual(float(broken["candidate_share"]), 0.5)
        self.assertAlmostEqual(float(broken["selected_share"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(broken["placement_lift_vs_candidates"]), (2.0 / 3.0) - 0.5)
        self.assertAlmostEqual(float(regular["selected_share"]), 1.0 / 3.0)

    def test_build_pattern_demand_supply_rows_summarizes_city_blocks(self) -> None:
        blocks = pd.DataFrame(
            {
                "street_pattern_top1_class": ["Regular Grid", "Regular Grid", "Broken Grid"],
                "demand": [10.0, 30.0, 60.0],
                "capacity": [5.0, 15.0, 10.0],
                "provision": [4.0, 10.0, 20.0],
                "demand_without": [1.0, 5.0, 20.0],
                "demand_left": [2.0, 5.0, 10.0],
            }
        )

        rows = CITY_LEVEL_MODULE.build_pattern_demand_supply_rows("x", blocks)

        regular = rows[rows["street_pattern_dominant_class"] == "Regular Grid"].iloc[0]
        broken = rows[rows["street_pattern_dominant_class"] == "Broken Grid"].iloc[0]
        self.assertEqual(int(regular["block_count"]), 2)
        self.assertAlmostEqual(float(regular["demand"]), 40.0)
        self.assertAlmostEqual(float(regular["demand_share"]), 0.4)
        self.assertAlmostEqual(float(regular["capacity_share"]), 20.0 / 30.0)
        self.assertAlmostEqual(float(regular["unmet_demand"]), 13.0)
        self.assertAlmostEqual(float(regular["coverage_proxy"]), (40.0 - 13.0) / 40.0)
        self.assertAlmostEqual(float(broken["unmet_share"]), 30.0 / 43.0)

    def test_build_overall_pattern_demand_supply_rows_aggregates_counts_before_shares(self) -> None:
        detail = pd.DataFrame(
            {
                "city": ["a", "a", "b", "b"],
                "street_pattern_dominant_class": ["Regular Grid", "Broken Grid", "Regular Grid", "Broken Grid"],
                "block_count": [1, 1, 3, 1],
                "demand": [10.0, 30.0, 30.0, 30.0],
                "capacity": [5.0, 10.0, 15.0, 0.0],
                "provision": [5.0, 10.0, 20.0, 0.0],
                "unmet_demand": [5.0, 20.0, 10.0, 30.0],
            }
        )

        overall = CITY_LEVEL_MODULE.build_overall_pattern_demand_supply_rows(detail)

        regular = overall[overall["street_pattern_dominant_class"] == "Regular Grid"].iloc[0]
        broken = overall[overall["street_pattern_dominant_class"] == "Broken Grid"].iloc[0]
        self.assertAlmostEqual(float(regular["demand_share"]), 0.4)
        self.assertAlmostEqual(float(regular["coverage_proxy"]), (40.0 - 15.0) / 40.0)
        self.assertAlmostEqual(float(broken["unmet_share"]), 50.0 / 65.0)

    def test_street_pattern_route_target_policy_keeps_non_loops(self) -> None:
        policy = ACCESS_FIRST_MODULE._street_pattern_route_target_policy(
            client_pattern="Regular Grid",
            facility_pattern="Irregular Grid",
            client_distance_to_stop=1500.0,
            facility_distance_to_stop=1200.0,
            loops_stop_distance_threshold_m=800.0,
            loops_route_target_multiplier=0.5,
        )

        self.assertTrue(policy["keep"])
        self.assertEqual(policy["reason"], "non_loops_endpoint")
        self.assertAlmostEqual(float(policy["weight_multiplier"]), 1.0)

    def test_street_pattern_route_target_policy_excludes_loops_with_bad_stop_access(self) -> None:
        policy = ACCESS_FIRST_MODULE._street_pattern_route_target_policy(
            client_pattern="Loops & Lollipops",
            facility_pattern="Regular Grid",
            client_distance_to_stop=900.0,
            facility_distance_to_stop=100.0,
            loops_stop_distance_threshold_m=800.0,
            loops_route_target_multiplier=1.0,
        )

        self.assertFalse(policy["keep"])
        self.assertEqual(policy["status"], "excluded")
        self.assertEqual(policy["reason"], "client_loops_stop_distance_gt_threshold")

    def test_street_pattern_route_target_policy_keeps_loops_when_stop_access_ok(self) -> None:
        policy = ACCESS_FIRST_MODULE._street_pattern_route_target_policy(
            client_pattern="Loops & Lollipops",
            facility_pattern="Regular Grid",
            client_distance_to_stop=300.0,
            facility_distance_to_stop=100.0,
            loops_stop_distance_threshold_m=800.0,
            loops_route_target_multiplier=0.25,
        )

        self.assertTrue(policy["keep"])
        self.assertEqual(policy["reason"], "loops_endpoint_stop_access_ok")
        self.assertAlmostEqual(float(policy["weight_multiplier"]), 0.25)

    def test_extract_block_pattern_prefers_solver_top1_class(self) -> None:
        blocks = pd.DataFrame(
            {
                "street_pattern_top1_class": ["Regular Grid"],
                "street_pattern_dominant_class": ["Sparse"],
            },
            index=["42"],
        )

        self.assertEqual(ACCESS_FIRST_MODULE._extract_block_pattern(blocks, "42"), "Regular Grid")
        self.assertIsNone(ACCESS_FIRST_MODULE._extract_block_pattern(blocks, "missing"))

    def test_build_candidate_service_target_links_uses_highest_catchment_candidates(self) -> None:
        blocks = pd.DataFrame(
            {
                "demand_without": [10.0, 20.0, 5.0, 0.0],
                "capacity": [0.0, 0.0, 0.0, 100.0],
                "service_radius_min": [15.0, 15.0, 15.0, 15.0],
            },
            index=["a", "b", "c", "existing"],
        )
        matrix = pd.DataFrame(
            {
                "a": [0.0, 4.0, 20.0, 6.0],
                "b": [4.0, 0.0, 9.0, 8.0],
                "c": [20.0, 9.0, 0.0, 8.0],
                "existing": [6.0, 8.0, 8.0, 0.0],
            },
            index=["a", "b", "c", "existing"],
        )

        links, candidates = ACCESS_FIRST_MODULE._build_candidate_service_target_links(
            blocks,
            matrix,
            top_k_candidates=1,
            max_destinations_per_client=1,
            demand_col="demand_without",
        )

        self.assertEqual(candidates["candidate_id"].tolist(), ["b"])
        self.assertAlmostEqual(float(candidates.iloc[0]["candidate_catchment_demand"]), 35.0)
        self.assertEqual(set(links["target"].astype(str)), {"b"})
        self.assertEqual(set(links["source"].astype(str)), {"a", "b", "c"})
        self.assertAlmostEqual(float(links["value"].sum()), 35.0)

    def test_build_candidate_service_target_links_splits_weight_between_destinations(self) -> None:
        blocks = pd.DataFrame(
            {
                "demand_without": [12.0, 0.0, 0.0],
                "capacity": [0.0, 0.0, 0.0],
                "service_radius_min": [15.0, 15.0, 15.0],
            },
            index=["origin", "cand1", "cand2"],
        )
        matrix = pd.DataFrame(
            {
                "origin": [0.0, 3.0, 4.0],
                "cand1": [3.0, 0.0, 1.0],
                "cand2": [4.0, 1.0, 0.0],
            },
            index=["origin", "cand1", "cand2"],
        )

        links, _candidates = ACCESS_FIRST_MODULE._build_candidate_service_target_links(
            blocks,
            matrix,
            top_k_candidates=2,
            max_destinations_per_client=2,
            demand_col="demand_without",
        )

        origin_links = links[links["source"] == "origin"].sort_values("target")
        self.assertEqual(origin_links["target"].tolist(), ["cand1", "cand2"])
        self.assertEqual(origin_links["value"].tolist(), [6.0, 6.0])

    def test_build_candidate_or_existing_target_links_uses_union_destinations(self) -> None:
        blocks = pd.DataFrame(
            {
                "demand_without": [10.0, 20.0, 0.0, 0.0],
                "capacity": [0.0, 0.0, 0.0, 100.0],
                "service_radius_min": [15.0, 15.0, 15.0, 15.0],
            },
            index=["a", "b", "candidate", "existing"],
        )
        matrix = pd.DataFrame(
            {
                "a": [0.0, 20.0, 8.0, 2.0],
                "b": [20.0, 0.0, 3.0, 9.0],
                "candidate": [8.0, 3.0, 0.0, 7.0],
                "existing": [2.0, 9.0, 7.0, 0.0],
            },
            index=["a", "b", "candidate", "existing"],
        )

        links, destinations = ACCESS_FIRST_MODULE._build_candidate_or_existing_target_links(
            blocks,
            matrix,
            top_k_candidates=1,
            max_destinations_per_client=1,
            demand_col="demand_without",
        )

        self.assertEqual(set(destinations["destination_type"]), {"candidate", "existing"})
        by_source = dict(zip(links["source"], links["target"]))
        self.assertEqual(by_source["a"], "existing")
        self.assertEqual(by_source["b"], "candidate")
        self.assertAlmostEqual(float(links["value"].sum()), 30.0)


if __name__ == "__main__":
    unittest.main()

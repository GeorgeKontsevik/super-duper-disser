from __future__ import annotations

import geopandas as gpd
import pandas as pd
import unittest
from shapely.geometry import LineString, Point, Polygon

from aggregated_spatial_pipeline.pipeline.run_street_pattern_stop_highway_filter import (
    build_class_comparison,
    normalize_highway_values,
    pick_stop_highway_types,
)


class StreetPatternStopHighwayFilterTest(unittest.TestCase):
    def test_normalize_highway_values_handles_scalars_lists_and_json_strings(self) -> None:
        values = normalize_highway_values(
            pd.Series(
                [
                    "residential",
                    ["primary", "secondary"],
                    '["tertiary", "service"]',
                    None,
                    "",
                    "residential;unclassified",
                ]
            )
        )

        self.assertEqual(
            values,
            [
                {"residential"},
                {"primary", "secondary"},
                {"tertiary", "service"},
                set(),
                set(),
                {"residential", "unclassified"},
            ],
        )


    def test_pick_stop_highway_types_uses_stop_buffer_intersections(self) -> None:
        roads = gpd.GeoDataFrame(
            {
                "road_id": [1, 2, 3],
                "highway": ["residential", "primary", "service"],
                "geometry": [
                    LineString([(0, 0), (10, 0)]),
                    LineString([(0, 20), (10, 20)]),
                    LineString([(0, 50), (10, 50)]),
                ],
            },
            crs="EPSG:3857",
        )
        stops = gpd.GeoDataFrame(
            {"stop_id": [1, 2], "geometry": [Point(5, 1), Point(5, 23)]},
            crs="EPSG:3857",
        )

        selected, matched = pick_stop_highway_types(roads, stops, stop_buffer_m=5.0)

        self.assertEqual(selected, {"primary", "residential"})
        self.assertEqual(
            matched[["road_id", "highway"]].sort_values("road_id").to_dict("records"),
            [
                {"road_id": 1, "highway": "residential"},
                {"road_id": 2, "highway": "primary"},
            ],
        )


    def test_build_class_comparison_reports_matches_drops_and_confusion(self) -> None:
        full = gpd.GeoDataFrame(
            {
                "cell_id": ["a", "b", "c"],
                "top1_class_name": ["Regular Grid", "Sparse", "Irregular Grid"],
                "prob_0": [0.1, 0.2, 0.3],
                "prob_1": [0.2, 0.3, 0.4],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                    Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
                ],
            },
            crs="EPSG:3857",
        )
        filtered = gpd.GeoDataFrame(
            {
                "cell_id": ["a", "b"],
                "top1_class_name": ["Regular Grid", "Irregular Grid"],
                "prob_0": [0.4, 0.1],
                "prob_1": [0.2, 0.7],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                ],
            },
            crs="EPSG:3857",
        )

        comparison, confusion, summary = build_class_comparison(full, filtered)

        self.assertEqual(
            summary,
            {
                "baseline_cells": 3,
                "filtered_cells": 2,
                "matched_cells": 2,
                "dropped_cells": 1,
                "changed_matched_cells": 1,
            },
        )
        by_cell = comparison.set_index("cell_id")
        self.assertTrue(by_cell.loc["b", "class_changed"])
        self.assertEqual(by_cell.loc["c", "filtered_class"], "dropped")
        self.assertEqual(
            confusion.to_dict("records"),
            [
                {
                    "full_class": "Irregular Grid",
                    "filtered_class": "dropped",
                    "cell_count": 1,
                },
                {
                    "full_class": "Regular Grid",
                    "filtered_class": "Regular Grid",
                    "cell_count": 1,
                },
                {
                    "full_class": "Sparse",
                    "filtered_class": "Irregular Grid",
                    "cell_count": 1,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()

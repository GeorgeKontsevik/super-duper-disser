from __future__ import annotations

import unittest
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from aggregated_spatial_pipeline.pipeline import run_historical_osm_download as module


class HistoricalOsmDownloadRoadsTest(unittest.TestCase):
    def test_download_layer_roads_uses_drive_graph_path(self) -> None:
        boundary = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        edges = gpd.GeoDataFrame(
            {
                "element_type": ["way", "way"],
                "osmid": [101, 202],
                "geometry": [
                    LineString([(0.1, 0.1), (0.9, 0.9)]),
                    LineString([(0.1, 0.9), (0.9, 0.1)]),
                ]
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        graph_sentinel = object()

        with patch.object(module, "_features_from_polygon_or_empty", side_effect=AssertionError("feature query used")):
            with patch.object(module.ox, "graph_from_polygon", return_value=graph_sentinel) as graph_mock:
                with patch.object(module.ox, "graph_to_gdfs", return_value=(None, edges)) as to_gdfs_mock:
                    roads = module._download_layer("roads", boundary)

        self.assertEqual(len(roads), 2)
        self.assertTrue(roads.geometry.geom_type.isin(["LineString", "MultiLineString"]).all())
        graph_mock.assert_called_once_with(
            boundary,
            network_type="drive",
            retain_all=True,
            truncate_by_edge=True,
        )
        to_gdfs_mock.assert_called_once_with(
            graph_sentinel,
            nodes=True,
            edges=True,
            node_geometry=True,
            fill_edge_geometry=True,
        )

    def test_download_layer_roads_falls_back_to_feature_query_when_graph_has_missing_nodes(self) -> None:
        boundary = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        fallback = gpd.GeoDataFrame(
            {"geometry": [LineString([(0.2, 0.2), (0.8, 0.8)])]},
            geometry="geometry",
            crs="EPSG:4326",
        )

        with patch.object(
            module,
            "_roads_from_graph_or_empty",
            side_effect=ValueError("Some edges missing nodes, possibly due to input data clipping issue."),
        ) as roads_graph_mock:
            with patch.object(module, "_features_from_polygon_or_empty", return_value=fallback) as features_mock:
                roads = module._download_layer("roads", boundary)

        self.assertEqual(len(roads), 1)
        roads_graph_mock.assert_called_once_with(boundary)
        features_mock.assert_called_once_with(boundary, module.BC_TAGS["roads"], "roads")


if __name__ == "__main__":
    unittest.main()

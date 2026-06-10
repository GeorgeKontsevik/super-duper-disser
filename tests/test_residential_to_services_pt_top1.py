from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path("/Users/gk/Code/super-duper-disser/scripts/run_residential_to_services_pt_top1.py")
SPEC = importlib.util.spec_from_file_location("run_residential_to_services_pt_top1", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ResidentialToServicesPtTop1Test(unittest.TestCase):
    def test_walk_filter_excludes_only_strictly_below_15_minutes(self) -> None:
        walk_df = pd.DataFrame(
            {
                "building_idx": [1, 2, 3, 4],
                "service_name": ["hospital", "hospital", "hospital", "hospital"],
                "walk_time_min": [14.9, 15.0, 16.2, float("inf")],
            }
        )

        kept = MODULE._eligible_buildings_for_service(walk_df, "hospital", 15.0)

        self.assertEqual(kept, [2, 3, 4])

    def test_walk_filter_can_select_only_strictly_below_15_minutes(self) -> None:
        walk_df = pd.DataFrame(
            {
                "building_idx": [1, 2, 3, 4],
                "service_name": ["hospital", "hospital", "hospital", "hospital"],
                "walk_time_min": [14.9, 15.0, 16.2, 8.0],
            }
        )

        kept = MODULE._eligible_buildings_for_service(
            walk_df,
            "hospital",
            min_walk_min=None,
            max_walk_min_exclusive=15.0,
        )

        self.assertEqual(kept, [1, 4])

    def test_multi_source_pt_result_returns_nearest_service_source(self) -> None:
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_edge(10, 20, time_min=4.0, type="walk")
        g.add_edge(20, 30, time_min=6.0, type="bus")
        g.add_edge(10, 40, time_min=20.0, type="walk")

        dist_map, source_map, path_map = MODULE._multi_source_pt_to_services(g, [30])

        self.assertEqual(source_map[10], 30)
        self.assertAlmostEqual(dist_map[10], 10.0)
        self.assertEqual(path_map[10], [10, 20, 30])

    def test_path_decomposition_splits_access_egress_transfer_and_transport_minutes(self) -> None:
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_edge(10, 20, time_min=2.0, type="walk")
        g.add_edge(20, 25, time_min=1.0, type="boarding")
        g.add_edge(25, 30, time_min=5.0, type="bus")
        g.add_edge(30, 31, time_min=1.5, type="walk")
        g.add_edge(31, 32, time_min=2.0, type="boarding")
        g.add_edge(32, 35, time_min=4.0, type="bus")
        g.add_edge(35, 40, time_min=3.0, type="walk")

        out = MODULE._decompose_pt_path(
            g,
            [10, 20, 25, 30, 31, 32, 35, 40],
            distance_home_to_graph_node_m=50.0,
            distance_service_to_graph_node_m=25.0,
        )

        self.assertAlmostEqual(out["walk_edge_time_min"], 6.5)
        self.assertAlmostEqual(out["transport_time_min"], 9.0)
        self.assertAlmostEqual(out["other_edge_time_min"], 3.0)
        self.assertAlmostEqual(out["access_walk_time_min"], 2.6)
        self.assertAlmostEqual(out["egress_walk_time_min"], 3.3)
        self.assertAlmostEqual(out["transfer_time_min"], 4.5)
        self.assertAlmostEqual(out["access_egress_walk_time_min"], 7.4)
        self.assertAlmostEqual(out["total_time_min"], 19.4)


if __name__ == "__main__":
    unittest.main()

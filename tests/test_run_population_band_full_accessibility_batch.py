from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


SCRIPT_PATH = Path("/Users/gk/Code/super-duper-disser/scripts/run_population_band_full_accessibility_batch.py")
SPEC = importlib.util.spec_from_file_location("run_population_band_full_accessibility_batch", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunPopulationBandFullAccessibilityBatchTests(unittest.TestCase):
    def test_parse_args_uses_longer_timeout_defaults(self) -> None:
        with mock.patch.object(sys, "argv", ["run_population_band_full_accessibility_batch.py"]):
            args = MODULE.parse_args()
        self.assertEqual(args.osm_timeout_s, 600.0)
        self.assertEqual(args.collection_attempt_timeout_s, 2400.0)

    def test_prioritize_candidates_prefers_stable_country_pool(self) -> None:
        candidates = [
            MODULE.CityCandidate("x1", "X1", "X1", "A", "Syria", "SYR", 2_000_000.0),
            MODULE.CityCandidate("x2", "X2", "X2", "B", "Germany", "DEU", 2_100_000.0),
            MODULE.CityCandidate("x3", "X3", "X3", "C", "Brazil", "BRA", 2_200_000.0),
        ]
        ordered = MODULE._prioritize_candidates_for_collection(candidates)
        self.assertEqual({item.iso3 for item in ordered[:2]}, {"DEU", "BRA"})
        self.assertEqual(ordered[-1].iso3, "SYR")

    def test_sample_across_countries_respects_exclusions_and_max_per_country(self) -> None:
        candidates = [
            MODULE.CityCandidate("a1", "A1", "A1", "X", "A", "AAA", 2_000_000.0),
            MODULE.CityCandidate("a2", "A2", "A2", "X", "A", "AAA", 2_100_000.0),
            MODULE.CityCandidate("a3", "A3", "A3", "X", "A", "AAA", 2_200_000.0),
            MODULE.CityCandidate("b1", "B1", "B1", "Y", "B", "BBB", 2_300_000.0),
            MODULE.CityCandidate("b2", "B2", "B2", "Y", "B", "BBB", 2_400_000.0),
            MODULE.CityCandidate("c1", "C1", "C1", "Z", "C", "CCC", 2_500_000.0),
        ]
        sampled = MODULE._sample_across_countries(
            candidates,
            sample_size=4,
            seed=7,
            max_per_country=1,
            exclude_slugs={"a1"},
        )
        self.assertEqual(len(sampled), 4)
        self.assertNotIn("a1", {item.slug for item in sampled})
        self.assertGreaterEqual(len({item.iso3 for item in sampled}), 3)

    def test_sample_across_countries_prefers_preferred_iso3_when_quota_allows(self) -> None:
        candidates = [
            MODULE.CityCandidate("bad1", "Bad1", "Bad1", "A", "Syria", "SYR", 2_100_000.0),
            MODULE.CityCandidate("bad2", "Bad2", "Bad2", "B", "Somalia", "SOM", 2_000_000.0),
            MODULE.CityCandidate("good1", "Good1", "Good1", "C", "Germany", "DEU", 2_200_000.0),
            MODULE.CityCandidate("good2", "Good2", "Good2", "D", "Brazil", "BRA", 2_300_000.0),
            MODULE.CityCandidate("good3", "Good3", "Good3", "E", "Poland", "POL", 2_400_000.0),
        ]
        sampled = MODULE._sample_across_countries(
            MODULE._prioritize_candidates_for_collection(candidates),
            sample_size=3,
            seed=42,
            max_per_country=1,
            exclude_slugs=set(),
        )
        self.assertEqual({item.iso3 for item in sampled}, {"DEU", "BRA", "POL"})

    def test_build_collection_command_uses_collect_only(self) -> None:
        cmd = MODULE._build_collection_command(
            place="Curitiba, Parana, Brazil",
            data_dir=Path("/tmp/data"),
            output_dir=Path("/tmp/out"),
            buffer_m=10000.0,
            street_grid_step=500.0,
            osm_timeout_s=180.0,
            modalities=["bus", "tram"],
            floor_ignore_missing_below_pct=0.0,
            overpass_url="https://overpass.kumi.systems/api/interpreter",
            no_cache=True,
        )
        self.assertIn("--collect-only", cmd)
        self.assertIn("--overpass-url", cmd)
        self.assertIn("--modalities", cmd)
        self.assertIn("--no-cache", cmd)

    def test_retryable_failure_detection_catches_generic_timeout(self) -> None:
        self.assertTrue(MODULE._is_retryable_run_joint_failure("Command timed out after 900 seconds"))

    def test_exception_message_includes_called_process_output(self) -> None:
        exc = subprocess.CalledProcessError(
            1,
            ["python", "worker.py"],
            output="hello\nRead timed out\nworld\n",
            stderr="",
        )
        text = MODULE._exception_message(exc)
        self.assertIn("Read timed out", text)

    def test_build_stage_command_for_pt_lt15(self) -> None:
        cmd = MODULE._build_python_stage_command(
            script_name="run_residential_to_services_pt_top1.py",
            args=[
                "--joint-inputs-root",
                "/tmp/joint_inputs",
                "--walk-root",
                "/tmp/walk",
                "--out-root",
                "/tmp/pt_lt15",
                "--cities",
                "foo_city",
                "bar_city",
                "--min-walk-min",
                "0",
                "--max-walk-min-exclusive",
                "15",
            ],
        )
        self.assertTrue(cmd[0].endswith("/.venv/bin/python"))
        self.assertEqual(cmd[1], str(SCRIPT_PATH.parents[1] / "scripts" / "run_residential_to_services_pt_top1.py"))
        self.assertIn("--max-walk-min-exclusive", cmd)
        self.assertIn("15", cmd)

    def test_collection_complete_requires_core_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            city_dir = Path(tmp) / "sample_city"
            (city_dir / "derived_layers").mkdir(parents=True)
            (city_dir / "intermodal_graph_iduedu").mkdir(parents=True)
            (city_dir / "pipeline_2" / "services_raw").mkdir(parents=True)
            (city_dir / "street_pattern" / "sample_city").mkdir(parents=True)

            pd.DataFrame({"is_living": [1, 0]}).to_parquet(city_dir / "derived_layers" / "buildings_floor_enriched.parquet", index=False)
            pd.DataFrame({"index": [0]}).to_parquet(city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet", index=False)
            pd.DataFrame({"u": [0], "v": [0], "type": ["walk"]}).to_parquet(
                city_dir / "intermodal_graph_iduedu" / "graph_edges.parquet",
                index=False,
            )
            (city_dir / "intermodal_graph_iduedu" / "graph.pkl").write_bytes(b"pickle")
            for service in MODULE.DEFAULT_SERVICES:
                pd.DataFrame({"x": []}).to_parquet(city_dir / "pipeline_2" / "services_raw" / f"{service}.parquet", index=False)
            (city_dir / "street_pattern" / "sample_city" / "predicted_cells.geojson").write_text("{}", encoding="utf-8")

            ok, details = MODULE._collection_complete(city_dir)
            self.assertTrue(ok)
            self.assertEqual(details["buildings_has_is_living"], 1)

            (city_dir / "intermodal_graph_iduedu" / "graph.pkl").unlink()
            ok2, details2 = MODULE._collection_complete(city_dir)
            self.assertFalse(ok2)
            self.assertEqual(details2["graph_pickle_exists"], 0)


if __name__ == "__main__":
    unittest.main()

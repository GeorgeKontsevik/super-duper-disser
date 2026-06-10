from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path("/Users/gk/Code/super-duper-disser/scripts/run_topup_full_joint_input_collection.py")
SPEC = importlib.util.spec_from_file_location("run_topup_full_joint_input_collection", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunTopupFullJointInputCollectionTests(unittest.TestCase):
    def test_retryable_failure_detection(self) -> None:
        self.assertTrue(MODULE._is_retryable_run_joint_failure("Read timed out"))
        self.assertTrue(MODULE._is_retryable_run_joint_failure("requests.exceptions.ConnectionError"))
        self.assertFalse(MODULE._is_retryable_run_joint_failure("FileNotFoundError: missing manifest"))

    def test_resolve_output_root_uses_batch_runs_prefix(self) -> None:
        out = MODULE._resolve_output_root(None)
        self.assertIn("aggregated_spatial_pipeline/outputs/batch_runs/topup_full_joint_inputs_", str(out))

    def test_build_run_joint_command_uses_collect_only_phase1(self) -> None:
        cmd = MODULE._build_run_joint_command(
            place="Kazan, Tatarstan, Russia",
            data_dir=Path("/tmp/data"),
            output_dir=Path("/tmp/out"),
            buffer_m=10000.0,
            street_grid_step=500.0,
            osm_timeout_s=60.0,
            modalities=["bus", "tram"],
            floor_ignore_missing_below_pct=0.0,
            overpass_url="https://overpass.kumi.systems/api/interpreter",
            no_cache=True,
        )
        self.assertIn("--collect-only", cmd)
        self.assertIn("--floor-ignore-missing-below-pct", cmd)
        self.assertIn("--overpass-url", cmd)
        self.assertIn("--modalities", cmd)
        self.assertEqual(cmd[cmd.index("--place") + 1], "Kazan, Tatarstan, Russia")
        self.assertEqual(cmd[cmd.index("--data-dir") + 1], "/tmp/data")
        self.assertEqual(cmd[cmd.index("--output-dir") + 1], "/tmp/out")
        self.assertIn("--no-cache", cmd)


if __name__ == "__main__":
    unittest.main()

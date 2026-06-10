from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aggregated_spatial_pipeline.pipeline import run_historical_osm_analysis as module


class HistoricalOsmAnalysisIntermodalRuntimeTest(unittest.TestCase):
    def test_run_intermodal_uses_runtime_helper_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            city_dir = Path(tmpdir) / "city"
            (city_dir / "analysis_territory").mkdir(parents=True)
            (city_dir / "analysis_territory" / "buffer.parquet").write_text("stub", encoding="utf-8")
            args = type("Args", (), {"no_cache": True, "osm_timeout_s": 1800, "overpass_url": None})()

            with patch.object(module, "_run_command") as run_command_mock:
                with patch.object(module, "intermodal_python", return_value=Path("/tmp/iduedu-python")):
                    module._run_intermodal(city_dir, "perth_western_australia_australia", 2025, args)

            command = run_command_mock.call_args.args[0]
            self.assertEqual(command[0], "/tmp/iduedu-python")
            self.assertEqual(command[1:3], ["-m", "aggregated_spatial_pipeline.intermodal_graph_data_pipeline.build_bundle_external"])


if __name__ == "__main__":
    unittest.main()

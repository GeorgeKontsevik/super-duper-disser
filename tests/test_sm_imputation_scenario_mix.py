from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path("/Users/gk/Code/super-duper-disser")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aggregated_spatial_pipeline.pipeline.run_sm_imputation_scenario import _parse_land_use_mix


class SmImputationScenarioMixTests(unittest.TestCase):
    def test_parse_land_use_mix_normalizes_named_weights(self) -> None:
        parsed = _parse_land_use_mix("residential=0.7,business=0.2,recreation=0.1")

        self.assertAlmostEqual(parsed["residential"], 0.7)
        self.assertAlmostEqual(parsed["business"], 0.2)
        self.assertAlmostEqual(parsed["recreation"], 0.1)
        self.assertAlmostEqual(sum(parsed.values()), 1.0)

    def test_parse_land_use_mix_rejects_unknown_land_use(self) -> None:
        with self.assertRaises(ValueError):
            _parse_land_use_mix("residential=1,park=1")


if __name__ == "__main__":
    unittest.main()

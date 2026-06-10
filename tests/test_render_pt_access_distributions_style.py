from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path("/Users/gk/Code/super-duper-disser/scripts/render_pt_access_distributions.py")
SPEC = importlib.util.spec_from_file_location("render_pt_access_distributions", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RenderPtAccessDistributionsStyleTest(unittest.TestCase):
    def test_renderer_uses_seaborn_as_default_style(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("import seaborn as sns", source)
        self.assertTrue(
            any(token in source for token in ['sns.set_theme(', 'sns.set_style(']),
            "render script should configure seaborn styling explicitly",
        )

    def test_renderer_uses_distribution_plots_instead_of_matplotlib_boxplots(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("sns.histplot(", source)
        self.assertNotIn(".boxplot(", source)
        self.assertNotIn("sns.violinplot(", source)

    def test_city_order_uses_population_density(self) -> None:
        order = MODULE._city_order_by_density(["marseille_france", "lyon_france", "bergen_norway"])
        self.assertEqual(order, ["lyon_france", "bergen_norway", "marseille_france"])

    def test_street_pattern_order_is_fixed_from_grid_to_loops(self) -> None:
        order = MODULE._street_pattern_order(
            ["Loops & Lollipops", "Broken Grid", "Regular Grid", "Sparse", "Warped Parallel"]
        )
        self.assertEqual(
            order,
            ["Regular Grid", "Warped Parallel", "Broken Grid", "Sparse", "Loops & Lollipops"],
        )

    def test_outlier_split_uses_strong_iqr_cutoff(self) -> None:
        kept, removed_count, cutoff = MODULE._split_strong_outliers([1, 2, 3, 4, 5, 30])
        self.assertEqual(kept, [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(removed_count, 1)
        self.assertGreater(cutoff, 5.0)
        self.assertLess(cutoff, 30.0)

    def test_renderer_supports_pt_to_service_split_outputs(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("HOME_SERVICE_PT15_ROOT", source)
        self.assertIn("HOME_SERVICE_PTLT15_ROOT", source)
        self.assertIn('_pt_city_hist_minutes_walk15plus.png', source)
        self.assertIn('_pt_city_hist_minutes_walklt15.png', source)
        self.assertIn('_pt_street_pattern_hist_minutes_walk15plus.png', source)
        self.assertIn('_pt_street_pattern_hist_minutes_walklt15.png', source)
        self.assertIn("access_egress_walk_time_min", source)
        self.assertIn("transport_time_min", source)
        self.assertIn("alpha=0.45", source)

    def test_renderer_uses_experiment_subdirectories(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('OUT_HOMES_TO_PT', source)
        self.assertIn('OUT_SERVICES_TO_PT', source)
        self.assertIn('OUT_HOME_TO_SERVICE_WALK', source)
        self.assertIn('OUT_HOME_TO_SERVICE_PT15', source)
        self.assertIn('OUT_HOME_TO_SERVICE_PTLT15', source)

    def test_renderer_draws_explicit_15_min_reference_line(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("ax.axvline(15", source)
        self.assertIn("_add_15_min_axis_label", source)


if __name__ == "__main__":
    unittest.main()

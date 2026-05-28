from __future__ import annotations

from pathlib import Path

import run_yana_bergen_lollipop_cost_sweep as sweep


ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = ROOT / "yana_experiments/street_pattern_route_comparison/bergen_norway/cost_sweep_lollipop_n5_20260429"

sweep.SWEEP_ROOT = SWEEP_ROOT
sweep.GEN_ROOT = SWEEP_ROOT / "generated"
sweep.CMP_ROOT = SWEEP_ROOT / "comparison"
sweep.LOG_ROOT = SWEEP_ROOT / "logs"
sweep.TABLE_PATH = SWEEP_ROOT / "metrics_n5_variants.csv"
sweep.CONFIG_PATH = SWEEP_ROOT / "variant_configs.json"
sweep.BEST_PATH = SWEEP_ROOT / "best_variant.json"


def _score(
    *,
    focus_length_share: float,
    edge_dup: float,
    focus_edge_dup: float,
    dominant_focus_share: float,
    coverage_loss: float,
    unserved: float,
) -> float:
    unserved_share = unserved / 100.0 if unserved > 1.0 else unserved
    return (
        focus_length_share
        + 0.55 * edge_dup
        + 0.75 * focus_edge_dup
        + 0.35 * dominant_focus_share
        + 20.0 * max(0.0, coverage_loss - 0.10)
        + 0.8 * max(0.0, unserved_share - 0.15)
    )


def _variant(name: str, **overrides: float | int | str) -> dict:
    config = sweep._variant(
        name,
        n_routes=5,
        min_route_len=9,
        max_route_len=25,
        n_samples=80,
        **overrides,
    )
    return config


def _variants() -> list[dict]:
    return [
        _variant(
            "n5_conn100",
            median_connectivity_weight=1.00,
            focus_class_overlap_weight=0.00,
        ),
        _variant(
            "n5_conn080_loopsdup020",
            median_connectivity_weight=0.80,
            focus_class_overlap_weight=0.20,
        ),
        _variant(
            "n5_conn060_loopsdup040",
            median_connectivity_weight=0.60,
            focus_class_overlap_weight=0.40,
        ),
        _variant(
            "n5_conn050_loopsdup060",
            median_connectivity_weight=0.50,
            focus_class_overlap_weight=0.60,
        ),
        _variant(
            "n5_conn040_loopsdup080",
            median_connectivity_weight=0.40,
            focus_class_overlap_weight=0.80,
        ),
        _variant(
            "n5_conn060_loops_share020_dup040",
            median_connectivity_weight=0.60,
            focus_class_weight=0.20,
            focus_class_overlap_weight=0.40,
        ),
        _variant(
            "n5_conn050_loops_share030_dup060",
            median_connectivity_weight=0.50,
            focus_class_weight=0.30,
            focus_class_overlap_weight=0.60,
        ),
        _variant(
            "n5_conn060_loops_presence030_dup040",
            median_connectivity_weight=0.60,
            focus_class_presence_weight=0.30,
            focus_class_presence_threshold=0.15,
            focus_class_overlap_weight=0.40,
        ),
        _variant(
            "n5_conn050_loops_dist050_dup050",
            median_connectivity_weight=0.50,
            focus_class_distribution_weight=0.50,
            focus_class_overlap_weight=0.50,
        ),
        _variant(
            "n5_conn050_targetdist050_dup050",
            median_connectivity_weight=0.50,
            street_pattern_target_distribution_weight=0.50,
            street_pattern_target_focus_multiplier=6.0,
            focus_class_overlap_weight=0.50,
        ),
        _variant(
            "n5_conn060_all_dup040",
            median_connectivity_weight=0.60,
            route_overlap_weight=0.40,
            focus_class_overlap_weight=0.40,
        ),
        _variant(
            "n5_conn050_all_dup060",
            median_connectivity_weight=0.50,
            route_overlap_weight=0.60,
            focus_class_overlap_weight=0.60,
        ),
        _variant(
            "n5_d10_r10_conn50_loops_share020_dup040",
            demand_time_weight=0.10,
            route_time_weight=0.10,
            median_connectivity_weight=0.50,
            focus_class_weight=0.20,
            focus_class_overlap_weight=0.40,
        ),
        _variant(
            "n5_d20_r10_conn50_loops_share020_dup060",
            demand_time_weight=0.20,
            route_time_weight=0.10,
            median_connectivity_weight=0.50,
            focus_class_weight=0.20,
            focus_class_overlap_weight=0.60,
        ),
        _variant(
            "n5_d10_r20_conn40_loops_share030_dup060",
            demand_time_weight=0.10,
            route_time_weight=0.20,
            median_connectivity_weight=0.40,
            focus_class_weight=0.30,
            focus_class_overlap_weight=0.60,
        ),
    ]


sweep._variants = _variants
sweep._score = _score
sweep._load_existing_rows = lambda: {}


if __name__ == "__main__":
    sweep.main()

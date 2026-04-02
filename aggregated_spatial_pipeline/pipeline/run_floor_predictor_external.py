from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregated_spatial_pipeline.pipeline.run_joint import _run_floor_predictor_preprocessing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run floor predictor preprocessing in dedicated floor-predictor runtime.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--buildings-path", required=True)
    parser.add_argument("--land-use-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--simple-bad-is-living-restore", action="store_true")
    parser.add_argument("--floor-ignore-missing-below-pct", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = _run_floor_predictor_preprocessing(
        repo_root=Path(args.repo_root).resolve(),
        buildings_path=Path(args.buildings_path).resolve(),
        land_use_path=Path(args.land_use_path).resolve(),
        output_path=Path(args.output_path).resolve(),
        simple_bad_is_living_restore=bool(args.simple_bad_is_living_restore),
        floor_ignore_missing_below_pct=float(args.floor_ignore_missing_below_pct),
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

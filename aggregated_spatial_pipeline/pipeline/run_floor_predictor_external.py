from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


LOG_FORMAT = (
    "<green>{time:DD MMM HH:mm}</green> | "
    "<level>{level: <7}</level> | "
    "<magenta>{extra[tag]}</magenta> "
    "{message}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run floor predictor preprocessing in dedicated floor-predictor runtime.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--buildings-path", required=True)
    parser.add_argument("--land-use-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--simple-bad-is-living-restore", action="store_true")
    parser.add_argument("--floor-ignore-missing-below-pct", type=float, default=2.0)
    return parser.parse_args()


def _configure_logging() -> None:
    logger.remove()
    logger.configure(extra={"tag": "[floor-predictor]"})
    logger.add(
        sys.stderr,
        level="INFO",
        format=LOG_FORMAT,
        colorize=sys.stderr.isatty(),
    )


def main() -> None:
    _configure_logging()
    args = parse_args()
    logger.info(
        "Starting dedicated floor preprocessing: buildings={}, land_use={}, output={}",
        Path(args.buildings_path).name,
        Path(args.land_use_path).name,
        Path(args.output_path).name,
    )
    from aggregated_spatial_pipeline.pipeline.run_joint import _run_floor_predictor_preprocessing

    # Some imported modules reconfigure loguru on import; enforce compact format again.
    _configure_logging()

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

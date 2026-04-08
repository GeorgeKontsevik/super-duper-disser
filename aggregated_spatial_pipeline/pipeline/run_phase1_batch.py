from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

from aggregated_spatial_pipeline.runtime_config import (
    DEFAULT_BATCH_REGIONS,
    DEFAULT_STREET_GRID_STEP_M,
    configure_logger,
    repo_mplconfigdir,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "aggregated_spatial_pipeline" / "config" / "phase1_city_batches.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phase-1 city preparation in batch mode: collection, floor, blocks, and street-pattern source."
    )
    parser.add_argument(
        "--config-path",
        default=str(CONFIG_PATH),
        help="Path to JSON file with named city batches.",
    )
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_BATCH_REGIONS))
    parser.add_argument("--limit-per-region", type=int)
    parser.add_argument("--buffer-m", type=float, default=None)
    parser.add_argument("--street-grid-step", type=float, default=DEFAULT_STREET_GRID_STEP_M)
    parser.add_argument("--street-min-road-count", type=int, default=5)
    parser.add_argument("--street-min-total-road-length", type=float, default=500.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--summary-path")
    return parser.parse_args()


def _configure_logging() -> None:
    configure_logger("[phase1-batch]")


def _load_city_batches(config_path: Path) -> dict[str, list[str]]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _summary_path(override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    return (ROOT / "aggregated_spatial_pipeline" / "outputs" / "batch_runs" / "phase1_summary.json").resolve()


def _build_command(args: argparse.Namespace, place: str) -> list[str]:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_joint",
        "--place",
        place,
        "--collect-only",
        "--street-grid-step",
        str(float(args.street_grid_step)),
        "--street-min-road-count",
        str(int(args.street_min_road_count)),
        "--street-min-total-road-length",
        str(float(args.street_min_total_road_length)),
    ]
    if args.buffer_m is not None:
        command.extend(["--buffer-m", str(float(args.buffer_m))])
    if args.no_cache:
        command.append("--no-cache")
    return command


def main() -> None:
    _configure_logging()
    args = parse_args()
    config_path = Path(args.config_path).resolve()
    batches = _load_city_batches(config_path)
    summary_path = _summary_path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    selected_regions = [region for region in args.regions if region in batches]
    if not selected_regions:
        raise ValueError(f"No known regions selected. Available: {', '.join(sorted(batches))}")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT)
    env.setdefault("MPLCONFIGDIR", repo_mplconfigdir("mpl-phase1-batch", root=ROOT))

    results: list[dict] = []
    for region in selected_regions:
        cities = list(batches.get(region, []))
        if args.limit_per_region is not None:
            cities = cities[: int(args.limit_per_region)]
        logger.info("Region {}: {} cities queued for phase 1.", region, len(cities))
        for place in cities:
            command = _build_command(args, place)
            started = time.time()
            logger.info("Phase 1 start: {}", place)
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
            )
            elapsed_s = round(time.time() - started, 1)
            result = {
                "region": region,
                "place": place,
                "returncode": int(completed.returncode),
                "elapsed_s": elapsed_s,
                "command": command,
            }
            if completed.returncode == 0:
                logger.info("Phase 1 done: {} ({}s)", place, elapsed_s)
            else:
                logger.warning("Phase 1 failed: {} (returncode={}, {}s)", place, completed.returncode, elapsed_s)
            results.append(result)
            summary = {
                "config_path": str(config_path),
                "regions": selected_regions,
                "buffer_m": args.buffer_m,
                "street_grid_step": float(args.street_grid_step),
                "street_min_road_count": int(args.street_min_road_count),
                "street_min_total_road_length": float(args.street_min_total_road_length),
                "no_cache": bool(args.no_cache),
                "results": results,
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary_path)


if __name__ == "__main__":
    main()

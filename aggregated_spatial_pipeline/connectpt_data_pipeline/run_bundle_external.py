from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from aggregated_spatial_pipeline.connectpt_data_pipeline.pipeline import build_connectpt_osm_bundle, parse_modalities


LOG_FORMAT = (
    "<green>{time:DD MMM HH:mm}</green> | "
    "<level>{level: <7}</level> | "
    "<magenta>{extra[tag]}</magenta> "
    "{message}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build connectpt bundle in dedicated connectpt runtime.")
    parser.add_argument("--place", required=True)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speed-kmh", type=float, required=True)
    parser.add_argument("--boundary-path")
    parser.add_argument("--drive-roads-path")
    parser.add_argument("--buildings-path")
    parser.add_argument("--intermodal-nodes-path")
    return parser.parse_args()


def _configure_logging() -> None:
    logger.remove()
    logger.configure(extra={"tag": "[connectpt]"})
    logger.add(
        sys.stderr,
        level="INFO",
        format=LOG_FORMAT,
        colorize=sys.stderr.isatty(),
    )


def main() -> None:
    _configure_logging()
    args = parse_args()
    manifest = build_connectpt_osm_bundle(
        place=args.place,
        modalities=parse_modalities(args.modalities),
        output_dir=args.output_dir,
        speed_kmh=float(args.speed_kmh),
        boundary_path=args.boundary_path,
        drive_roads_path=args.drive_roads_path,
        buildings_path=args.buildings_path,
        intermodal_nodes_path=args.intermodal_nodes_path,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

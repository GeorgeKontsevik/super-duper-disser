from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregated_spatial_pipeline.connectpt_data_pipeline.pipeline import build_connectpt_osm_bundle, parse_modalities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build connectpt bundle in dedicated connectpt runtime.")
    parser.add_argument("--place", required=True)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speed-kmh", type=float, required=True)
    parser.add_argument("--boundary-path")
    parser.add_argument("--drive-roads-path")
    parser.add_argument("--intermodal-nodes-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_connectpt_osm_bundle(
        place=args.place,
        modalities=parse_modalities(args.modalities),
        output_dir=args.output_dir,
        speed_kmh=float(args.speed_kmh),
        boundary_path=args.boundary_path,
        drive_roads_path=args.drive_roads_path,
        intermodal_nodes_path=args.intermodal_nodes_path,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

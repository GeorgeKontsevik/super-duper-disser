from __future__ import annotations

import argparse
import json

from aggregated_spatial_pipeline.blocksnet_data_pipeline.pipeline import build_blocksnet_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build blocksnet bundle in dedicated blocksnet runtime.")
    parser.add_argument("--place", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-path", required=True)
    parser.add_argument("--prefetched-layers-json", required=True)
    parser.add_argument("--buildings-override-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_blocksnet_bundle(
        place=args.place,
        output_dir=args.output_dir,
        boundary_path=args.boundary_path,
        prefetched_layers=json.loads(args.prefetched_layers_json),
        buildings_override_path=args.buildings_override_path,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

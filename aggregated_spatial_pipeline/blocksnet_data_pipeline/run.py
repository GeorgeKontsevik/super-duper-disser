from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import build_blocksnet_bundle, slugify_place


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cut urban blocks for a given city using BlocksNet dev environment.",
    )
    parser.add_argument("--place", required=True, help="Place name for OSM geocoding, e.g. 'Tianjin, China'.")
    parser.add_argument(
        "--output-dir",
        help="Directory where outputs will be written. Defaults to aggregated_spatial_pipeline/outputs/blocksnet/<slug>.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "aggregated_spatial_pipeline" / "outputs" / "blocksnet" / slugify_place(args.place)
    )

    manifest = build_blocksnet_bundle(args.place, output_dir=output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

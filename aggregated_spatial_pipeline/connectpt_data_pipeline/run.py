from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import build_connectpt_osm_bundle, parse_modalities, slugify_place


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download connectpt-ready OSM transport inputs for a given city.",
    )
    parser.add_argument("--place", required=True, help="Place name for OSM geocoding, e.g. 'Tianjin, China'.")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["bus"],
        help="Transport modalities to build. Supported: bus tram trolleybus.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where outputs will be written. Defaults to aggregated_spatial_pipeline/outputs/connectpt_osm/<slug>.",
    )
    parser.add_argument(
        "--speed-kmh",
        type=float,
        default=20.0,
        help="Assumed average speed for stop-to-stop time estimates.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "aggregated_spatial_pipeline" / "outputs" / "connectpt_osm" / slugify_place(args.place)
    )

    manifest = build_connectpt_osm_bundle(
        place=args.place,
        modalities=parse_modalities(args.modalities),
        output_dir=output_dir,
        speed_kmh=args.speed_kmh,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

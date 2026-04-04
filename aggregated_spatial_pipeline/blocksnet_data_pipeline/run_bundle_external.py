from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from aggregated_spatial_pipeline.blocksnet_data_pipeline.pipeline import build_blocksnet_bundle


ROOT = Path(__file__).resolve().parents[2]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build blocksnet bundle in dedicated blocksnet runtime.")
    parser.add_argument("--place")
    parser.add_argument("--joint-input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--boundary-path")
    parser.add_argument("--prefetched-layers-json")
    parser.add_argument("--buildings-override-path")
    return parser.parse_args()


def _resolve_city_dir(args: argparse.Namespace) -> Path | None:
    if args.joint_input_dir:
        return Path(args.joint_input_dir).resolve()
    if args.place:
        return (ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs" / _slugify(str(args.place))).resolve()
    return None


def _resolve_prefetched_layers(city_dir: Path) -> dict[str, str]:
    raw_dir = city_dir / "blocksnet_raw_osm"
    return {
        "boundary": str((raw_dir / "boundary.parquet").resolve()),
        "water": str((raw_dir / "water.parquet").resolve()),
        "roads": str((raw_dir / "roads.parquet").resolve()),
        "railways": str((raw_dir / "railways.parquet").resolve()),
        "land_use": str((raw_dir / "land_use.parquet").resolve()),
        "buildings": str((raw_dir / "buildings.parquet").resolve()),
    }


def main() -> None:
    args = parse_args()
    city_dir = _resolve_city_dir(args)
    place = str(args.place) if args.place else (city_dir.name if city_dir is not None else None)
    if not place:
        raise ValueError("Provide --place or --joint-input-dir.")

    output_dir = args.output_dir or (str((city_dir / "blocksnet").resolve()) if city_dir is not None else None)
    boundary_path = args.boundary_path or (
        str((city_dir / "analysis_territory" / "buffer_collection.parquet").resolve()) if city_dir is not None else None
    )
    prefetched_layers_json = args.prefetched_layers_json or (
        json.dumps(_resolve_prefetched_layers(city_dir), ensure_ascii=False) if city_dir is not None else None
    )
    buildings_override_path = args.buildings_override_path
    if buildings_override_path is None and city_dir is not None:
        candidate = city_dir / "derived_layers" / "buildings_floor_enriched.parquet"
        if candidate.exists():
            buildings_override_path = str(candidate.resolve())

    if output_dir is None or boundary_path is None or prefetched_layers_json is None:
        raise ValueError("Could not resolve blocksnet bundle inputs. Provide --joint-input-dir or explicit paths.")

    manifest = build_blocksnet_bundle(
        place=place,
        output_dir=output_dir,
        boundary_path=boundary_path,
        prefetched_layers=json.loads(prefetched_layers_json),
        buildings_override_path=buildings_override_path,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

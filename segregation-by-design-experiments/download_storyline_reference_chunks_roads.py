from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import osmnx as ox


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_FILE = REPO_ROOT / "segregation-by-design-experiments" / "storyline_reference_chunks_12.tsv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs_reference_chunks"
DEFAULT_STREET_PATTERN_DIR = "street_pattern_reference_chunks"


@dataclass
class ChunkSpec:
    slug: str
    place: str
    lat: float
    lon: float
    window_m: float


def _read_chunks(path: Path) -> list[ChunkSpec]:
    rows: list[ChunkSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 5:
            continue
        slug, place, lat, lon, window_m = parts[:5]
        rows.append(
            ChunkSpec(
                slug=slug,
                place=place,
                lat=float(lat),
                lon=float(lon),
                window_m=float(window_m),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download roads for manual storyline reference chunks.")
    parser.add_argument("--chunks-file", default=str(DEFAULT_CHUNKS_FILE), help="TSV: slug, place, lat, lon, window_m")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for per-city downloaded roads.")
    parser.add_argument(
        "--street-pattern-dir",
        default=DEFAULT_STREET_PATTERN_DIR,
        help="Per-city folder name under output root.",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        help=(
            "Optional buffer radius in meters around each chunk center. "
            "If provided, overrides per-row TSV window_m (which otherwise is interpreted as full window size)."
        ),
    )
    parser.add_argument("--network-type", default="drive", help="OSMnx network type.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download and overwrite existing roads.geojson files.")
    args = parser.parse_args()

    chunks = _read_chunks(Path(args.chunks_file).resolve())
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict[str, str | int | float]] = {}
    for spec in chunks:
        city_dir = output_root / spec.slug / str(args.street_pattern_dir) / spec.slug
        city_dir.mkdir(parents=True, exist_ok=True)
        roads_path = city_dir / "roads.geojson"

        if roads_path.exists() and not args.overwrite:
            report[spec.slug] = {"status": "skipped_existing", "roads": str(roads_path)}
            print(f"SKIP {spec.slug}: roads already exist")
            continue

        if args.buffer_m is not None:
            half = max(1.0, float(args.buffer_m))
            source = "cli_buffer_m"
        else:
            half = max(200.0, float(spec.window_m) / 2.0)
            source = "tsv_window_m"
        print(
            f"DOWNLOAD {spec.slug}: lat={spec.lat:.6f}, lon={spec.lon:.6f}, "
            f"buffer={half:.0f}m source={source}"
        )
        graph = ox.graph_from_point(
            (spec.lat, spec.lon),
            dist=half,
            dist_type="bbox",
            network_type=str(args.network_type),
            simplify=True,
        )
        edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
        edges.to_file(roads_path, driver="GeoJSON")

        centre = gpd.GeoDataFrame({"slug": [spec.slug]}, geometry=gpd.points_from_xy([spec.lon], [spec.lat]), crs=4326)
        centre.to_file(city_dir / "centre.geojson", driver="GeoJSON")

        report[spec.slug] = {
            "status": "downloaded",
            "roads": str(roads_path),
            "edge_count": int(len(edges)),
            "buffer_m": float(half),
            "window_m": float(spec.window_m),
            "buffer_source": source,
        }
        print(f"DONE {spec.slug}: edges={len(edges)}")

    report_path = output_root / f"{str(args.street_pattern_dir)}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

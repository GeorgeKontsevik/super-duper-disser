from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

import run_street_pattern_city as sp


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_FILE = REPO_ROOT / "segregation-by-design-experiments" / "storyline_reference_chunks_12_b700.tsv"
DEFAULT_JOINT_INPUTS_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs_reference_chunks_b700"
DEFAULT_STREET_PATTERN_DIR = "street_pattern_reference_chunks_b700"


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
    parser = argparse.ArgumentParser(description="Classify street pattern for pre-downloaded reference chunks.")
    parser.add_argument("--chunks-file", default=str(DEFAULT_CHUNKS_FILE))
    parser.add_argument("--joint-inputs-root", default=str(DEFAULT_JOINT_INPUTS_ROOT))
    parser.add_argument("--street-pattern-dir", default=DEFAULT_STREET_PATTERN_DIR)
    parser.add_argument("--buffer-m", type=float, default=700.0, help="Buffer radius (meters) around chunk center.")
    parser.add_argument("--grid-step", type=float, default=500.0)
    parser.add_argument("--min-road-count", type=int, default=5)
    parser.add_argument("--min-total-road-length", type=float, default=500.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "segregation-by-design-experiments" / "outputs" / "cache" / "reference_chunks"),
    )
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "segregation-by-design-experiments" / "models" / "best_model.pth"),
    )
    args = parser.parse_args()

    sp._configure_logging()
    chunks = _read_chunks(Path(args.chunks_file).resolve())
    if not chunks:
        raise ValueError("No chunk rows found")

    joint_inputs_root = Path(args.joint_inputs_root).resolve()
    joint_inputs_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = sp.resolve_model_path(
        model_path=Path(args.model_path).resolve(),
        models_dir=(REPO_ROOT / "segregation-by-design-experiments" / "models"),
    )

    succeeded: list[str] = []
    failed: list[str] = []
    report: dict[str, dict] = {}

    for spec in chunks:
        city_root = joint_inputs_root / spec.slug / str(args.street_pattern_dir)
        city_dir = city_root / spec.slug
        roads_path = city_dir / "roads.geojson"
        if not roads_path.exists():
            failed.append(spec.slug)
            report[spec.slug] = {"status": "missing_roads", "roads_path": str(roads_path)}
            continue

        centre_node = {"id": None, "lon": float(spec.lon), "lat": float(spec.lat)}
        relation = {"id": None}
        polygon = sp.build_buffer_polygon(centre_node, float(args.buffer_m))
        buffer_gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=4326)
        local_crs = buffer_gdf.estimate_utm_crs() or "EPSG:3857"
        polygon_projected = buffer_gdf.to_crs(local_crs).iloc[0].geometry
        centre_projected = gpd.GeoSeries([Point(float(spec.lon), float(spec.lat))], crs=4326).to_crs(local_crs).iloc[0]
        grid_anchor_xy = (float(centre_projected.x), float(centre_projected.y))

        try:
            snapshot = sp.classify_snapshot(
                place=spec.place,
                year=None,
                network_type="drive",
                relation=relation,
                centre_node=centre_node,
                polygon_wgs84=polygon,
                polygon_projected=polygon_projected,
                local_crs=local_crs,
                model_path=model_path,
                grid_step=float(args.grid_step),
                buffer_m=float(args.buffer_m),
                min_road_count=int(args.min_road_count),
                min_total_road_length=float(args.min_total_road_length),
                relation_id=None,
                device=str(args.device),
                no_cache=bool(args.no_cache),
                cache_dir=cache_dir,
                roads_path=roads_path,
                road_source_label="local",
                grid_anchor_xy=grid_anchor_xy,
            )

            city_dir.mkdir(parents=True, exist_ok=True)
            sp.save_city_outputs(
                city_dir=city_dir,
                maps_dir=city_root / "maps",
                place=spec.place,
                relation=relation,
                centre_node=centre_node,
                summary=snapshot["summary"],
                roads_wgs84=snapshot["roads_wgs84"],
                buffer_polygon_wgs84=polygon,
                prediction_gdf=snapshot["prediction_gdf"],
                map_coloring="multivariate",
            )
            sp.save_city_outputs(
                city_dir=city_dir,
                maps_dir=city_root / "maps",
                place=spec.place,
                relation=relation,
                centre_node=centre_node,
                summary=snapshot["summary"],
                roads_wgs84=snapshot["roads_wgs84"],
                buffer_polygon_wgs84=polygon,
                prediction_gdf=snapshot["prediction_gdf"],
                map_coloring="top1",
            )

            summary_path = city_root / f"{spec.slug}_summary.json"
            summary_path.write_text(json.dumps(snapshot["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
            succeeded.append(spec.slug)
            report[spec.slug] = {
                "status": "ok",
                "summary_path": str(summary_path),
                "num_predictions": int(snapshot["summary"]["num_predictions"]),
            }
        except Exception as exc:  # noqa: BLE001
            failed.append(spec.slug)
            report[spec.slug] = {"status": "failed", "error": str(exc)}

    report_path = joint_inputs_root / f"{str(args.street_pattern_dir)}_classification_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    for slug in failed:
        print(f"  FAIL {slug}")
    print(report_path)


if __name__ == "__main__":
    main()

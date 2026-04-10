from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

import run_street_pattern_city as sp


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = REPO_ROOT / "segregation-by-design-experiments" / "storyline_reference_chunks_12_b700.tsv"


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


def _source_city_dir(root: Path, street_pattern_dir: str, slug: str) -> Path:
    return root / slug / street_pattern_dir / slug


def _source_summary_path(root: Path, street_pattern_dir: str, slug: str) -> Path:
    return root / slug / street_pattern_dir / f"{slug}_summary.json"


def _subset_cells_to_buffer(cells: gpd.GeoDataFrame, buffer_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    local_crs = buffer_gdf.estimate_utm_crs() or "EPSG:3857"
    cells_local = cells.to_crs(local_crs)
    buffer_local = buffer_gdf.to_crs(local_crs)
    polygon = buffer_local.geometry.iloc[0]
    mask = cells_local.geometry.centroid.within(polygon)
    subset = cells.loc[mask].copy()
    if subset.empty:
        mask2 = cells_local.intersects(polygon)
        subset = cells.loc[mask2].copy()
    return subset


def _clip_roads_to_buffer(roads: gpd.GeoDataFrame, buffer_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    local_crs = buffer_gdf.estimate_utm_crs() or "EPSG:3857"
    roads_local = roads.to_crs(local_crs)
    buffer_local = buffer_gdf.to_crs(local_crs)
    poly = buffer_local.geometry.iloc[0]
    roads_local = roads_local[roads_local.intersects(poly)].copy()
    if roads_local.empty:
        return roads_local.to_crs(4326)
    roads_local["geometry"] = roads_local.geometry.intersection(poly)
    roads_local = roads_local[roads_local.geometry.notna() & ~roads_local.geometry.is_empty].copy()
    return roads_local.to_crs(4326)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive smaller-buffer chunks from a larger parent grid (same cell alignment).")
    parser.add_argument("--chunks-file", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--source-root", required=True, help="Root with parent outputs (e.g. .../joint_inputs_reference_chunks_b10000)")
    parser.add_argument("--source-dir", required=True, help="Parent street-pattern dir name (e.g. street_pattern_reference_chunks_b10000)")
    parser.add_argument("--target-root", required=True, help="Root for derived outputs (e.g. .../joint_inputs_reference_chunks_b2000)")
    parser.add_argument("--target-dir", required=True, help="Target street-pattern dir name.")
    parser.add_argument("--target-buffer-m", type=float, required=True, help="Target buffer radius in meters.")
    args = parser.parse_args()

    chunks = _read_chunks(Path(args.chunks_file).resolve())
    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()
    source_dir = str(args.source_dir)
    target_dir = str(args.target_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    for spec in chunks:
        src_city_dir = _source_city_dir(source_root, source_dir, spec.slug)
        src_cells = src_city_dir / "predicted_cells.geojson"
        src_roads = src_city_dir / "roads.geojson"
        src_centre = src_city_dir / "centre.geojson"
        src_summary = _source_summary_path(source_root, source_dir, spec.slug)

        if not src_cells.exists() or not src_roads.exists():
            report[spec.slug] = {"status": "missing_source", "src_city_dir": str(src_city_dir)}
            continue

        cells = gpd.read_file(src_cells)
        roads = gpd.read_file(src_roads)

        if src_centre.exists():
            centre_gdf = gpd.read_file(src_centre)
            if centre_gdf.empty:
                centre_node = {"id": None, "lon": float(spec.lon), "lat": float(spec.lat)}
            else:
                pt = centre_gdf.geometry.iloc[0]
                centre_node = {"id": None, "lon": float(pt.x), "lat": float(pt.y)}
        else:
            centre_node = {"id": None, "lon": float(spec.lon), "lat": float(spec.lat)}
            centre_gdf = gpd.GeoDataFrame(
                {"slug": [spec.slug]}, geometry=gpd.points_from_xy([centre_node["lon"]], [centre_node["lat"]]), crs=4326
            )

        buffer_polygon = sp.build_buffer_polygon(centre_node, float(args.target_buffer_m))
        buffer_gdf = gpd.GeoDataFrame({"geometry": [buffer_polygon]}, crs=4326)
        subset_cells = _subset_cells_to_buffer(cells, buffer_gdf)
        subset_roads = _clip_roads_to_buffer(roads, buffer_gdf)

        tgt_city_root = target_root / spec.slug / target_dir
        tgt_city_dir = tgt_city_root / spec.slug
        tgt_city_dir.mkdir(parents=True, exist_ok=True)

        subset_cells.to_file(tgt_city_dir / "predicted_cells.geojson", driver="GeoJSON")
        subset_cells.drop(columns="geometry", errors="ignore").to_csv(tgt_city_dir / "predicted_cells.csv", index=False)
        subset_roads.to_file(tgt_city_dir / "roads.geojson", driver="GeoJSON")
        buffer_gdf.to_file(tgt_city_dir / "buffer.geojson", driver="GeoJSON")
        centre_gdf.to_file(tgt_city_dir / "centre.geojson", driver="GeoJSON")

        if src_summary.exists():
            summary = json.loads(src_summary.read_text(encoding="utf-8"))
        else:
            summary = {"place": spec.place, "class_names": []}
        summary["place"] = spec.place
        summary["buffer_m"] = float(args.target_buffer_m)
        summary["num_predictions"] = int(len(subset_cells))
        summary["num_subgraphs"] = int(len(subset_cells))
        class_names = summary.get("class_names") or []
        if "class_name" in subset_cells.columns:
            counts = subset_cells["class_name"].value_counts().to_dict()
            summary["class_counts"] = {
                str(name): int(counts.get(name, 0)) for name in class_names
            } if class_names else {str(k): int(v) for k, v in counts.items()}

        (tgt_city_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (tgt_city_root / f"{spec.slug}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        report[spec.slug] = {
            "status": "ok",
            "source_cells": int(len(cells)),
            "target_cells": int(len(subset_cells)),
            "target_buffer_m": float(args.target_buffer_m),
        }

    report_path = target_root / f"{target_dir}_derived_from_{source_dir}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

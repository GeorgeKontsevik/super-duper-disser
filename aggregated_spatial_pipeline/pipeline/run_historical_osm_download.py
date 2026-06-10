from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.blocksnet_data_pipeline.pipeline import (
    BC_TAGS,
    IS_LIVING_TAGS,
    _clip_to_boundary,
    _keep_geometry_types,
    _normalize_raw_osm,
    _roads_from_graph_or_empty,
    _save_geodata,
)
from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.runtime_config import configure_logger


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "aggregated_spatial_pipeline" / "outputs" / "historical_osm_2000_2025"
DEFAULT_YEARS = list(range(2000, 2026, 5))
CORE_LAYERS = ("roads", "buildings", "services_pipeline2_raw")
# Access/is_living historical runs need land-use and amenity context for the
# existing imputer. Water/railways stay supported for broader BlocksNet-style
# runs; raw PT stops/routes stay supported for diagnostics. They are
# intentionally not downloaded by default because latest access experiments
# build historical PT through iduedu with --overpass-date.
ACCESS_SUPPORT_LAYERS = ("land_use", "amenities_floor_context")
OPTIONAL_CONTEXT_LAYERS = ("pt_stops", "pt_routes", "water", "railways")
DEFAULT_LAYERS = (*CORE_LAYERS, *ACCESS_SUPPORT_LAYERS)
SUPPORT_LAYERS = (*ACCESS_SUPPORT_LAYERS, *OPTIONAL_CONTEXT_LAYERS)
SUPPORTED_LAYERS = (*CORE_LAYERS, *SUPPORT_LAYERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download historical OSM snapshots for roads, buildings, and service POIs "
            "needed by street-pattern and service-accessibility analysis."
        )
    )
    parser.add_argument("--city-dirs", nargs="+", help="Explicit city bundle directories.")
    parser.add_argument("--joint-input-root", help="Directory containing city bundle subdirectories.")
    parser.add_argument("--limit", type=int, help="Optional limit when --joint-input-root is used.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--layers", nargs="+", choices=SUPPORTED_LAYERS, default=list(DEFAULT_LAYERS))
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--step-years", type=int, default=5)
    parser.add_argument(
        "--snapshot-month-day",
        default="12-31",
        help="Snapshot date within every year, MM-DD. Default: year-end.",
    )
    parser.add_argument(
        "--boundary-source",
        choices=("blocksnet-boundary", "analysis-buffer"),
        default="blocksnet-boundary",
        help=(
            "Geometry used for downloads. 'blocksnet-boundary' uses blocksnet_raw_osm/boundary.parquet; "
            "'analysis-buffer' uses analysis_territory/buffer.parquet."
        ),
    )
    parser.add_argument("--osm-timeout-s", type=int, default=240)
    parser.add_argument("--overpass-url", default=None)
    parser.add_argument("--sleep-s", type=float, default=2.0, help="Pause between Overpass requests.")
    parser.add_argument("--no-cache", action="store_true", help="Redownload even if parquet outputs exist.")
    return parser.parse_args()


def _configure_logging() -> None:
    configure_logger("[historical-osm]")


def _years_from_args(args: argparse.Namespace) -> list[int]:
    if args.years:
        return list(dict.fromkeys(int(year) for year in args.years))
    return list(range(int(args.start_year), int(args.end_year) + 1, int(args.step_years)))


def _resolve_city_dirs(args: argparse.Namespace) -> list[Path]:
    if args.city_dirs:
        return [Path(path).resolve() for path in args.city_dirs]
    if args.joint_input_root:
        root = Path(args.joint_input_root).resolve()
        city_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        if args.limit is not None:
            city_dirs = city_dirs[: int(args.limit)]
        return city_dirs
    raise ValueError("Provide --city-dirs or --joint-input-root.")


def _boundary_path(city_dir: Path, source: str) -> Path:
    if source == "analysis-buffer":
        return city_dir / "analysis_territory" / "buffer.parquet"
    return city_dir / "blocksnet_raw_osm" / "boundary.parquet"


def _read_boundary_polygon(city_dir: Path, source: str) -> tuple[gpd.GeoDataFrame, object, Path]:
    path = _boundary_path(city_dir, source)
    if not path.exists():
        raise FileNotFoundError(f"Missing boundary for {city_dir.name}: {path}")
    boundary = read_geodata(path)
    if boundary.empty:
        raise ValueError(f"Boundary is empty for {city_dir.name}: {path}")
    if boundary.crs is None:
        boundary = boundary.set_crs(4326)
    boundary_wgs84 = boundary.to_crs(4326)
    return boundary_wgs84, boundary_wgs84.union_all(), path


@contextmanager
def _historical_overpass_settings(snapshot_iso: str, *, timeout_s: int, overpass_url: str | None):
    original_settings = ox.settings.overpass_settings
    original_timeout = getattr(ox.settings, "timeout", None)
    original_requests_timeout = getattr(ox.settings, "requests_timeout", None)
    original_url = ox.settings.overpass_url
    try:
        if hasattr(ox.settings, "timeout"):
            ox.settings.timeout = int(timeout_s)
        if hasattr(ox.settings, "requests_timeout"):
            ox.settings.requests_timeout = int(timeout_s)
        if overpass_url:
            ox.settings.overpass_url = str(overpass_url)
        ox.settings.overpass_settings = (
            f'[out:json][timeout:{{timeout}}][date:"{snapshot_iso}"]{{maxsize}}'
        )
        yield
    finally:
        ox.settings.overpass_settings = original_settings
        if original_timeout is not None:
            ox.settings.timeout = original_timeout
        if original_requests_timeout is not None and hasattr(ox.settings, "requests_timeout"):
            ox.settings.requests_timeout = original_requests_timeout
        ox.settings.overpass_url = original_url


def _empty_layer(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def _features_from_polygon_or_empty(boundary_geom, tags: dict, layer_name: str) -> gpd.GeoDataFrame:
    started = time.time()
    logger.info("OSM request start: layer={}", layer_name)
    try:
        gdf = ox.features_from_polygon(boundary_geom, tags)
        logger.info(
            "OSM request done: layer={}, features={}, elapsed={:.1f}s",
            layer_name,
            len(gdf),
            time.time() - started,
        )
        if gdf.empty:
            return _empty_layer()
        return gdf
    except Exception as exc:  # noqa: BLE001 - OSMnx uses different empty-response classes by version.
        if exc.__class__.__name__ == "InsufficientResponseError":
            logger.warning("OSM layer empty: layer={}, elapsed={:.1f}s", layer_name, time.time() - started)
            return _empty_layer()
        raise


def _tag_contains(value, expected: set[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_tag_contains(item, expected) for item in value)
    tokens = [part.strip().lower() for part in str(value).replace("|", ";").replace(",", ";").split(";")]
    return any(token in expected for token in tokens if token)


def _prepare_layer(layer: str, raw: gpd.GeoDataFrame, boundary_geom) -> gpd.GeoDataFrame:
    if raw.empty:
        return _empty_layer()
    work = raw.copy()
    if work.crs is None:
        work = work.set_crs(4326)
    if layer != "roads":
        work = _normalize_raw_osm(work)
    work = _clip_to_boundary(work, boundary_geom)
    if layer == "roads":
        return _keep_geometry_types(work, {"LineString", "MultiLineString"}, layer_name=layer)
    if layer == "buildings":
        work = _keep_geometry_types(work, {"Polygon", "MultiPolygon"}, layer_name=layer)
        # Keep the downloaded snapshot raw. Living/non-living classification is
        # produced later by the existing imputer, not by OSM tag rules here.
        work = work.drop(columns=["is_living"], errors="ignore")
        return work
    if layer in {"services_pipeline2_raw", "pt_stops"}:
        return _keep_geometry_types(
            work,
            {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"},
            layer_name=layer,
        )
    if layer in {"pt_routes", "water"}:
        return _keep_geometry_types(
            work,
            {"LineString", "MultiLineString", "Polygon", "MultiPolygon"},
            layer_name=layer,
        )
    if layer == "railways":
        return _keep_geometry_types(work, {"LineString", "MultiLineString"}, layer_name=layer)
    if layer in {"land_use", "amenities_floor_context"}:
        return _keep_geometry_types(work, {"Polygon", "MultiPolygon"}, layer_name=layer)
    raise ValueError(f"Unsupported layer: {layer}")


def _download_layer(layer: str, boundary_geom) -> gpd.GeoDataFrame:
    if layer == "roads":
        try:
            raw = _roads_from_graph_or_empty(boundary_geom)
        except ValueError as exc:
            if "Some edges missing nodes" not in str(exc):
                raise
            logger.warning(
                "Road graph path failed due to missing nodes; falling back to feature query for this snapshot."
            )
            raw = _features_from_polygon_or_empty(boundary_geom, BC_TAGS["roads"], layer)
    elif layer == "buildings":
        raw = _features_from_polygon_or_empty(boundary_geom, {"building": True}, layer)
    elif layer == "services_pipeline2_raw":
        raw = _features_from_polygon_or_empty(boundary_geom, BC_TAGS["services_pipeline2_raw"], layer)
    elif layer == "pt_stops":
        raw = _features_from_polygon_or_empty(boundary_geom, _pt_stop_tags(), layer)
    elif layer == "pt_routes":
        raw = _features_from_polygon_or_empty(boundary_geom, _pt_route_tags(), layer)
    elif layer == "water":
        raw = _features_from_polygon_or_empty(boundary_geom, BC_TAGS["water"], layer)
    elif layer == "railways":
        raw = _features_from_polygon_or_empty(boundary_geom, BC_TAGS["railways"], layer)
    elif layer == "land_use":
        raw = _features_from_polygon_or_empty(boundary_geom, {"landuse": True}, layer)
    elif layer == "amenities_floor_context":
        raw = _features_from_polygon_or_empty(boundary_geom, BC_TAGS["amenities_floor_context"], layer)
    else:
        raise ValueError(f"Unsupported layer: {layer}")
    return _prepare_layer(layer, raw, boundary_geom)


def _pt_stop_tags() -> dict:
    return {
        "highway": "bus_stop",
        "railway": ["tram_stop", "subway_entrance", "station"],
        "public_transport": ["platform", "stop_position", "station"],
        "bus": "yes",
        "tram": "yes",
        "trolleybus": "yes",
        "subway": "yes",
    }


def _pt_route_tags() -> dict:
    return {
        "route": ["bus", "tram", "trolleybus", "subway"],
        "railway": ["tram", "subway", "light_rail"],
        "trolley_wire": "yes",
        "public_transport": ["platform", "stop_position", "station"],
    }


def _metric_length_m(gdf: gpd.GeoDataFrame) -> float:
    if gdf.empty:
        return 0.0
    local = gdf.to_crs(gdf.estimate_utm_crs() or "EPSG:3857")
    return float(local.geometry.length.sum())


def _metric_area_m2(gdf: gpd.GeoDataFrame) -> float:
    if gdf.empty:
        return 0.0
    local = gdf.to_crs(gdf.estimate_utm_crs() or "EPSG:3857")
    return float(local.geometry.area.sum())


def _service_counts(gdf: gpd.GeoDataFrame) -> dict[str, int]:
    if gdf.empty:
        return {
            "hospital_count": 0,
            "clinic_count": 0,
            "school_count": 0,
            "kindergarten_count": 0,
        }
    amenity = gdf["amenity"] if "amenity" in gdf.columns else pd.Series(None, index=gdf.index)
    healthcare = gdf["healthcare"] if "healthcare" in gdf.columns else pd.Series(None, index=gdf.index)
    return {
        "hospital_count": int(
            amenity.map(lambda value: _tag_contains(value, {"hospital"})).sum()
            + healthcare.map(lambda value: _tag_contains(value, {"hospital"})).sum()
        ),
        "clinic_count": int(
            amenity.map(lambda value: _tag_contains(value, {"clinic"})).sum()
            + healthcare.map(lambda value: _tag_contains(value, {"clinic", "centre"})).sum()
        ),
        "school_count": int(amenity.map(lambda value: _tag_contains(value, {"school"})).sum()),
        "kindergarten_count": int(amenity.map(lambda value: _tag_contains(value, {"kindergarten"})).sum()),
    }


def _pt_counts(gdf: gpd.GeoDataFrame) -> dict[str, int]:
    if gdf.empty:
        return {
            "bus_count": 0,
            "tram_count": 0,
            "trolleybus_count": 0,
            "subway_count": 0,
        }
    route = gdf["route"] if "route" in gdf.columns else pd.Series(None, index=gdf.index)
    railway = gdf["railway"] if "railway" in gdf.columns else pd.Series(None, index=gdf.index)
    return {
        "bus_count": int(route.map(lambda value: _tag_contains(value, {"bus"})).sum()),
        "tram_count": int(
            route.map(lambda value: _tag_contains(value, {"tram"})).sum()
            + railway.map(lambda value: _tag_contains(value, {"tram", "light_rail"})).sum()
        ),
        "trolleybus_count": int(route.map(lambda value: _tag_contains(value, {"trolleybus"})).sum()),
        "subway_count": int(
            route.map(lambda value: _tag_contains(value, {"subway"})).sum()
            + railway.map(lambda value: _tag_contains(value, {"subway"})).sum()
        ),
    }


def _summarize_layer(city: str, year: int, layer: str, gdf: gpd.GeoDataFrame, path: Path) -> dict:
    row = {
        "city": city,
        "year": int(year),
        "layer": layer,
        "features": int(len(gdf)),
        "path": str(path),
    }
    if layer == "roads":
        row["road_length_m"] = _metric_length_m(gdf)
    elif layer == "buildings":
        row["building_area_m2"] = _metric_area_m2(gdf)
        building = gdf.get("building", pd.Series("", index=gdf.index))
        living_tags = {tag.lower() for tag in IS_LIVING_TAGS}
        row["osm_living_tag_candidates"] = int(building.map(lambda value: _tag_contains(value, living_tags)).sum())
    elif layer == "services_pipeline2_raw":
        row.update(_service_counts(gdf))
    elif layer == "pt_stops":
        row["pt_stop_points"] = int(gdf.geometry.geom_type.isin(["Point", "MultiPoint"]).sum()) if not gdf.empty else 0
    elif layer == "pt_routes":
        row["pt_route_length_m"] = _metric_length_m(gdf)
        row.update(_pt_counts(gdf))
    return row


def _add_delta_columns(summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in summary.columns
        if col
        not in {
            "city",
            "year",
            "layer",
            "path",
        }
        and pd.api.types.is_numeric_dtype(summary[col])
    ]
    summary = summary.sort_values(["city", "layer", "year"]).reset_index(drop=True)
    for col in metric_cols:
        summary[f"{col}_delta_prev"] = summary.groupby(["city", "layer"], dropna=False)[col].diff()
    return summary


def _snapshot_iso(year: int, month_day: str) -> str:
    return f"{int(year):04d}-{month_day}T23:59:59Z"


def main() -> None:
    _configure_logging()
    args = parse_args()
    years = _years_from_args(args)
    city_dirs = _resolve_city_dirs(args)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    errors: list[dict] = []
    logger.info("Queued cities={}, years={}, layers={}", len(city_dirs), years, args.layers)

    for city_dir in city_dirs:
        city = city_dir.name
        boundary_gdf, boundary_geom, boundary_path = _read_boundary_polygon(city_dir, args.boundary_source)
        city_out = output_root / city
        city_out.mkdir(parents=True, exist_ok=True)
        _save_geodata(boundary_gdf, city_out / "boundary.parquet")
        logger.info("City start: {} boundary={}", city, boundary_path.name)

        for year in years:
            snapshot = _snapshot_iso(year, args.snapshot_month_day)
            year_out = city_out / str(year)
            year_out.mkdir(parents=True, exist_ok=True)
            year_manifest = {
                "city": city,
                "year": int(year),
                "snapshot": snapshot,
                "city_dir": str(city_dir),
                "boundary_source": str(boundary_path),
                "files": {},
                "counts": {},
            }
            logger.info("Snapshot start: {}/{} ({})", city, year, snapshot)
            with _historical_overpass_settings(
                snapshot,
                timeout_s=int(args.osm_timeout_s),
                overpass_url=args.overpass_url,
            ):
                for layer in args.layers:
                    layer_path = year_out / f"{layer}.parquet"
                    if layer_path.exists() and not args.no_cache:
                        gdf = read_geodata(layer_path)
                        logger.info("Using cached layer: {}/{}/{} features={}", city, year, layer, len(gdf))
                    else:
                        try:
                            gdf = _download_layer(layer, boundary_geom)
                            _save_geodata(gdf, layer_path)
                            if args.sleep_s > 0:
                                time.sleep(float(args.sleep_s))
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Layer failed: {}/{}/{} error={}", city, year, layer, exc)
                            errors.append(
                                {
                                    "city": city,
                                    "year": int(year),
                                    "layer": layer,
                                    "error": str(exc),
                                }
                            )
                            # Keep a complete year bundle even when OSMnx fails on
                            # a malformed historical element. The error is preserved
                            # in manifests; downstream stages can mark the year as
                            # sparse/failed instead of crashing on a missing file.
                            gdf = _empty_layer()
                            _save_geodata(gdf, layer_path)
                    year_manifest["files"][layer] = str(layer_path)
                    year_manifest["counts"][layer] = int(len(gdf))
                    rows.append(_summarize_layer(city, year, layer, gdf, layer_path))

            manifest_path = year_out / "manifest.json"
            manifest_path.write_text(json.dumps(year_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _add_delta_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()
    summary_path = output_root / "historical_osm_summary.csv"
    summary.to_csv(summary_path, index=False)

    root_manifest = {
        "output_root": str(output_root),
        "years": years,
        "layers": list(args.layers),
        "boundary_source": args.boundary_source,
        "summary_csv": str(summary_path),
        "errors": errors,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Done. Summary: {}", summary_path)
    print(summary_path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata
from blocksnet.analysis.provision import competitive_provision
from blocksnet.relations import calculate_accessibility_matrix


SUPPORTED_SERVICES = ("health", "post", "culture", "port", "airport", "marina")

# Keep matplotlib/tqdm ecosystem caches in writable workspace path.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-asp-pipeline2")

# High-level service categories used by arctic solver scenarios.
# Downloaded as raw OSM layers, then converted to capacity points.
SERVICE_TAG_QUERIES: dict[str, list[dict]] = {
    "health": [
        {"amenity": ["hospital", "clinic", "doctors", "dentist"]},
        {"healthcare": True},
    ],
    "post": [
        {"amenity": "post_office"},
    ],
    "culture": [
        {"amenity": ["theatre", "cinema", "arts_centre", "community_centre", "library"]},
        {"tourism": ["museum", "gallery"]},
    ],
    "port": [
        {"landuse": "port"},
        {"amenity": "ferry_terminal"},
        {"harbour": True},
    ],
    "airport": [
        {"aeroway": ["aerodrome", "terminal"]},
    ],
    "marina": [
        {"leisure": "marina"},
    ],
}

# Arctic/solver defaults:
# - cleanerreader default capacity when unknown is 600
# - create_blocks fallback "population" when missing is CONST_BASE_DEMAND=120
# We keep these values here for city-level fallback compatibility.
ARCTIC_DEFAULT_CAPACITY = 600.0
ARCTIC_DEFAULT_POPULATION = 120.0

# Per-service defaults intentionally follow arctic-style generic fallback.
DEFAULT_CAPACITY_BY_SERVICE = {
    "health": ARCTIC_DEFAULT_CAPACITY,
    "post": ARCTIC_DEFAULT_CAPACITY,
    "culture": ARCTIC_DEFAULT_CAPACITY,
    "port": ARCTIC_DEFAULT_CAPACITY,
    "airport": ARCTIC_DEFAULT_CAPACITY,
    "marina": ARCTIC_DEFAULT_CAPACITY,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline_2 input preparation on top of pipeline_1 territory: "
            "download OSM services, aggregate to quarters, compute accessibility matrix, "
            "save solver-ready tables."
        )
    )
    parser.add_argument(
        "--joint-input-dir",
        required=True,
        help=(
            "Path to pipeline_1 city bundle, e.g. "
            "/.../aggregated_spatial_pipeline/outputs/joint_inputs/barcelona_spain"
        ),
    )
    parser.add_argument(
        "--services",
        nargs="+",
        default=list(SUPPORTED_SERVICES),
        help=f"Services to prepare. Supported: {' '.join(SUPPORTED_SERVICES)}",
    )
    parser.add_argument(
        "--service-radius-min",
        type=float,
        default=60.0,
        help="Accessibility threshold (minutes) for demand_within/demand_without split.",
    )
    parser.add_argument(
        "--demand-per-1000",
        type=float,
        default=120.0,
        help="Demand norm per 1000 population (same meaning as arctic CONST_BASE_DEMAND).",
    )
    parser.add_argument(
        "--provision-max-depth",
        type=int,
        default=1,
        help="Max depth for blocksnet competitive_provision.",
    )
    parser.add_argument(
        "--population-default",
        type=float,
        default=ARCTIC_DEFAULT_POPULATION,
        help=(
            "Fallback population for units when no population columns are available "
            "(arctic create_blocks-compatible default)."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild all artifacts even if cached files exist.",
    )
    parser.add_argument(
        "--overpass-url",
        default=None,
        help="Optional custom Overpass endpoint for OSMnx.",
    )
    parser.add_argument(
        "--osm-timeout-s",
        type=int,
        default=180,
        help="OSMnx timeout in seconds.",
    )
    parser.add_argument(
        "--osmnx-debug",
        action="store_true",
        help="Enable verbose OSMnx logs.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    logger.info(f"[pipeline_2_prepare] {message}")


def _warn(message: str) -> None:
    logger.warning(f"[pipeline_2_prepare] {message}")


def _configure_osmnx(args: argparse.Namespace) -> None:
    ox.settings.timeout = int(args.osm_timeout_s)
    ox.settings.use_cache = True
    ox.settings.log_console = bool(args.osmnx_debug)
    if args.overpass_url:
        ox.settings.overpass_url = str(args.overpass_url)


def _to_epsg_int(crs_obj) -> int:
    if crs_obj is None:
        raise ValueError("Graph CRS is missing.")
    if isinstance(crs_obj, int):
        return int(crs_obj)
    if hasattr(crs_obj, "to_epsg"):
        epsg = crs_obj.to_epsg()
        if epsg is None:
            raise ValueError(f"Could not convert graph CRS to EPSG int: {crs_obj}")
        return int(epsg)
    raise ValueError(f"Unsupported graph CRS type: {type(crs_obj)}")


def _read_boundary(boundary_path: Path) -> gpd.GeoDataFrame:
    boundary = read_geodata(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary is empty: {boundary_path}")
    return boundary


def _read_quarters(quarters_path: Path) -> gpd.GeoDataFrame:
    quarters = read_geodata(quarters_path)
    if quarters.empty:
        raise ValueError(f"Quarters layer is empty: {quarters_path}")
    return quarters


def _read_graph_pickle(graph_path: Path) -> nx.MultiDiGraph:
    graph = pd.read_pickle(graph_path)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Expected networkx graph in {graph_path}, got {type(graph)}")
    graph = graph.copy()
    graph.graph["crs"] = _to_epsg_int(graph.graph.get("crs"))
    return graph


def _first_number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if np.isfinite(v) else None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    token = match.group(0).replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _normalize_raw_osm(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    normalized = gdf.reset_index()
    for col in ("element", "id"):
        if col not in normalized.columns:
            normalized[col] = None
    if "element_type" in normalized.columns and "osmid" in normalized.columns:
        normalized["element"] = normalized["element"].fillna(normalized["element_type"])
        normalized["id"] = normalized["id"].fillna(normalized["osmid"])
    normalized["source_uid"] = normalized["element"].astype(str) + ":" + normalized["id"].astype(str)
    normalized = normalized.drop_duplicates(subset=["source_uid"])
    return normalized


def _download_service_raw(boundary_gdf: gpd.GeoDataFrame, service: str) -> gpd.GeoDataFrame:
    boundary_wgs84 = boundary_gdf.to_crs(4326)
    polygon = boundary_wgs84.union_all().convex_hull
    queries = SERVICE_TAG_QUERIES[service]
    frames: list[gpd.GeoDataFrame] = []
    for idx, tags in enumerate(queries, start=1):
        _log(f"OSM download [{service}] query {idx}/{len(queries)}: tags={tags}")
        try:
            raw = ox.features_from_polygon(polygon, tags)
            if raw is None or raw.empty:
                _warn(f"OSM [{service}] query {idx} returned empty result.")
                continue
            norm = _normalize_raw_osm(raw)
            frames.append(norm)
            _log(f"OSM [{service}] query {idx} features={len(norm)}")
        except Exception as exc:  # noqa: BLE001 - keep full OSM failure in logs
            _warn(f"OSM [{service}] query {idx} failed: {exc}")
    if not frames:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)
    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=4326)
    merged = merged.drop_duplicates(subset=["source_uid"]).reset_index(drop=True)
    return merged


def _capacity_from_row(row: pd.Series, service: str) -> float:
    if service == "health":
        beds = _first_number(row.get("beds"))
        if beds is not None and beds > 0:
            return beds
    cap = _first_number(row.get("capacity"))
    if cap is not None and cap > 0:
        return cap
    return float(DEFAULT_CAPACITY_BY_SERVICE[service])


def _to_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = gdf.copy()
    points["geometry"] = points.geometry.representative_point()
    return points


def _aggregate_capacity_to_quarters(service_points: gpd.GeoDataFrame, quarters_gdf: gpd.GeoDataFrame) -> pd.Series:
    if service_points.empty:
        return pd.Series(0.0, index=quarters_gdf.index, dtype=float)
    points = service_points.to_crs(quarters_gdf.crs)
    quarters_geom = quarters_gdf[["geometry"]].copy()
    joined = gpd.sjoin(points[["capacity_est", "geometry"]], quarters_geom, how="inner", predicate="intersects")
    if joined.empty:
        return pd.Series(0.0, index=quarters_gdf.index, dtype=float)
    by_quarter = joined.groupby("index_right")["capacity_est"].sum()
    return by_quarter.reindex(quarters_gdf.index).fillna(0.0).astype(float)


def _detect_population_column(quarters: gpd.GeoDataFrame) -> str | None:
    for col in ("population_total", "population_proxy", "population"):
        if col in quarters.columns:
            return col
    return None


def _save_geodata(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prepare_geodata_for_parquet(gdf).to_parquet(path)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _calc_demand(population: pd.Series, demand_per_1000: float) -> pd.Series:
    scaled = np.ceil((population.fillna(0.0).astype(float) / 1000.0) * float(demand_per_1000))
    return scaled.astype(int)


def _run_with_heartbeat(label: str, func, interval_s: float = 20.0):
    done = threading.Event()
    box: dict[str, object] = {}

    def _worker() -> None:
        try:
            box["result"] = func()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    started = time.time()
    pulse = 0
    while not done.wait(float(interval_s)):
        pulse += 1
        _log(f"{label}: still running... elapsed={time.time() - started:.1f}s, pulse={pulse}")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("result")


def _ensure_services_valid(services: Iterable[str]) -> list[str]:
    normalized = [str(s).strip().lower() for s in services]
    unknown = [s for s in normalized if s not in SUPPORTED_SERVICES]
    if unknown:
        raise ValueError(f"Unsupported services: {', '.join(unknown)}")
    return normalized


def main() -> None:
    args = parse_args()
    _configure_osmnx(args)

    services = _ensure_services_valid(args.services)
    city_dir = Path(args.joint_input_dir).resolve()

    boundary_path = city_dir / "analysis_territory" / "buffer.parquet"
    quarters_path = city_dir / "derived_layers" / "quarters_clipped.parquet"
    graph_path = city_dir / "intermodal_graph_iduedu" / "graph.pkl"

    output_root = city_dir / "pipeline_2"
    raw_dir = output_root / "services_raw"
    prepared_dir = output_root / "prepared"
    solver_dir = output_root / "solver_inputs"
    manifest_path = output_root / "manifest_prepare_solver_inputs.json"

    _log(f"Starting preparation for services={services}")
    _log(f"Using city bundle: {city_dir}")
    _log(f"Territory boundary: {boundary_path}")

    boundary = _read_boundary(boundary_path)
    quarters = _read_quarters(quarters_path)
    graph = _read_graph_pickle(graph_path)

    population_col = _detect_population_column(quarters)
    if population_col is None:
        _warn(
            "No population columns found in quarters. "
            f"Using arctic default population={args.population_default} for all units."
        )
    else:
        _log(f"Population source column: {population_col}")
    _log(f"Quarters features: {len(quarters)}")
    _log(f"Intermodal graph: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

    _log("STEP data_collection: downloading/caching raw OSM service layers inside analysis territory.")
    _log("STEP capacity_aggregation: converting raw service objects to per-quarter capacities.")

    # 1) Download/cache raw service layers and aggregate capacities to quarters.
    capacity_columns: dict[str, pd.Series] = {}
    raw_stats: dict[str, dict] = {}
    for service in services:
        raw_path = raw_dir / f"{service}.parquet"
        if raw_path.exists() and (not args.no_cache):
            raw = read_geodata(raw_path)
            _log(f"Using cached raw service layer [{service}]: {raw_path} ({len(raw)} features)")
        else:
            raw = _download_service_raw(boundary, service)
            if raw.empty:
                _warn(f"Raw service layer [{service}] is empty.")
            else:
                raw["capacity_est"] = raw.apply(lambda row: _capacity_from_row(row, service), axis=1)
            _save_geodata(raw, raw_path)
            _log(f"Saved raw service layer [{service}]: {raw_path} ({len(raw)} features)")

        if "capacity_est" not in raw.columns:
            raw = raw.copy()
            raw["capacity_est"] = raw.apply(lambda row: _capacity_from_row(row, service), axis=1) if not raw.empty else 0.0
            _save_geodata(raw, raw_path)

        points = _to_points(raw) if not raw.empty else raw
        aggregated = _aggregate_capacity_to_quarters(points, quarters)
        cap_col = f"capacity_{service}"
        capacity_columns[cap_col] = aggregated

        raw_stats[service] = {
            "raw_features": int(len(raw)),
            "quarters_with_capacity": int((aggregated > 0).sum()),
            "capacity_total": float(aggregated.sum()),
            "raw_path": str(raw_path),
        }

    _log("STEP matrix_build: preparing unified spatial units for accessibility matrix.")

    # 2) Build a unified spatial-units layer for matrix + solver prep.
    if population_col is None:
        units = quarters[["geometry"]].copy()
        units["population"] = float(args.population_default)
    else:
        units = quarters[[population_col, "geometry"]].copy()
        units = units.rename(columns={population_col: "population"})
        units["population"] = pd.to_numeric(units["population"], errors="coerce").fillna(0.0)

    for cap_col, series in capacity_columns.items():
        units[cap_col] = series.reindex(units.index).fillna(0.0).astype(float)

    cap_cols = [f"capacity_{s}" for s in services]
    capacity_any = units[cap_cols].sum(axis=1) > 0
    units_mask = (units["population"] > 0) | capacity_any
    units = units[units_mask].copy()
    units["unit_name"] = units.index.astype(str)
    units["demand_base"] = _calc_demand(units["population"], args.demand_per_1000)
    _log(f"Active units for solver/matrix: {len(units)} (population>0 OR any service capacity>0)")

    units_path = prepared_dir / "units_union.parquet"
    _save_geodata(units, units_path)

    # 3) Accessibility matrix between selected units (same city territory / same graph).
    matrix_path = prepared_dir / "adj_matrix_time_min_union.parquet"
    if matrix_path.exists() and (not args.no_cache):
        matrix_union = pd.read_parquet(matrix_path)
        _log(f"Using cached accessibility matrix: {matrix_path} ({matrix_union.shape[0]}x{matrix_union.shape[1]})")
    else:
        n_units = int(len(units))
        approx_pairs = n_units * n_units
        _log(
            "Computing accessibility matrix via blocksnet.relations.calculate_accessibility_matrix "
            f"for n_units={n_units} (~{approx_pairs:,} pair entries). This can take time."
        )
        started = time.time()
        matrix_union = _run_with_heartbeat(
            "Matrix build",
            lambda: calculate_accessibility_matrix(units[["geometry"]].copy(), graph, weight_key="time_min"),
            interval_s=20.0,
        )
        _save_dataframe(matrix_union, matrix_path)
        elapsed = time.time() - started
        _log(
            f"Saved accessibility matrix: {matrix_path} "
            f"({matrix_union.shape[0]}x{matrix_union.shape[1]}), elapsed={elapsed:.1f}s"
        )

    _log("STEP solver_prep: building per-service solver-ready blocks and links.")

    # 4) Per-service solver-ready tables (demand_within/demand_without/capacity_left/provision).
    service_outputs: dict[str, dict] = {}
    for service in services:
        cap_col = f"capacity_{service}"
        blocks = units[["unit_name", "population", "demand_base", cap_col, "geometry"]].copy()
        blocks = blocks.rename(columns={"demand_base": "demand", cap_col: "capacity"})
        blocks = blocks[(blocks["population"] > 0) | (blocks["capacity"] > 0)].copy()
        if blocks.empty:
            _warn(f"No active blocks for service [{service}] after filtering. Skipping.")
            continue

        _log(f"Service [{service}] provisioning prep: blocks={len(blocks)}")
        sub_mx = matrix_union.loc[blocks.index, blocks.index].copy()
        provision_started = time.time()
        provision_df, links_df = competitive_provision(
            blocks_df=blocks[["population", "demand", "capacity", "geometry"]].copy(),
            accessibility_matrix=sub_mx,
            accessibility=int(math.ceil(args.service_radius_min)),
            demand=None,
            self_supply=True,
            max_depth=int(args.provision_max_depth),
        )
        _log(f"Service [{service}] competitive_provision finished in {time.time() - provision_started:.1f}s")
        solver_blocks = provision_df.copy()
        solver_blocks["name"] = solver_blocks.index.astype(str)
        solver_blocks["service_name"] = service
        solver_blocks["service_radius_min"] = float(args.service_radius_min)
        # Compatibility with arctic solver runner fields.
        solver_blocks["provision"] = solver_blocks["provision_strong"].fillna(0.0)

        service_dir = solver_dir / service
        blocks_path = service_dir / "blocks_solver.parquet"
        matrix_service_path = service_dir / "adj_matrix_time_min.parquet"
        links_path = service_dir / "provision_links.csv"
        summary_path = service_dir / "summary.json"

        _save_geodata(solver_blocks, blocks_path)
        _save_dataframe(sub_mx, matrix_service_path)
        links_path.parent.mkdir(parents=True, exist_ok=True)
        links_df.reset_index().to_csv(links_path, index=False)

        summary = {
            "service": service,
            "blocks_count": int(len(solver_blocks)),
            "demand_total": float(solver_blocks["demand"].sum()),
            "capacity_total": float(solver_blocks["capacity"].sum()),
            "demand_within_total": float(solver_blocks["demand_within"].sum()),
            "demand_without_total": float(solver_blocks["demand_without"].sum()),
            "provision_strong_total": float(
                solver_blocks["demand_within"].sum() / solver_blocks["demand"].sum()
                if solver_blocks["demand"].sum() > 0
                else 0.0
            ),
            "files": {
                "blocks_solver": str(blocks_path),
                "adj_matrix": str(matrix_service_path),
                "provision_links": str(links_path),
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        service_outputs[service] = {
            "summary": str(summary_path),
            "blocks_solver": str(blocks_path),
            "adj_matrix": str(matrix_service_path),
            "provision_links": str(links_path),
            "blocks_count": int(len(solver_blocks)),
        }
        _log(
            f"Prepared solver input [{service}]: blocks={len(solver_blocks)}, "
            f"capacity_total={summary['capacity_total']:.1f}, demand_total={summary['demand_total']:.1f}"
        )

    manifest = {
        "city_bundle": str(city_dir),
        "boundary": str(boundary_path),
        "quarters": str(quarters_path),
        "graph": str(graph_path),
        "services": services,
        "service_radius_min": float(args.service_radius_min),
        "demand_per_1000": float(args.demand_per_1000),
        "units_union": str(units_path),
        "adj_matrix_union": str(matrix_path),
        "raw_services": raw_stats,
        "solver_outputs": service_outputs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
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


SUPPORTED_SERVICES = ("hospital", "polyclinic", "school")
LP_BLOCK_SELECTION_POLICY = "has_living_buildings_or_service_capacity"

# Keep matplotlib/tqdm ecosystem caches in writable workspace path.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-asp-pipeline2")

@dataclass(frozen=True)
class ServiceSpec:
    tags: list[dict]
    blocksnet_name: str
    fallback_capacity: float = 600.0


SERVICE_SPECS: dict[str, ServiceSpec] = {
    "hospital": ServiceSpec(
        tags=[
            {"amenity": "hospital"},
            {"healthcare": "hospital"},
        ],
        blocksnet_name="hospital",
    ),
    "polyclinic": ServiceSpec(
        tags=[
            {"amenity": ["clinic", "doctors"]},
            {"healthcare": ["clinic", "doctor", "doctors"]},
        ],
        blocksnet_name="polyclinic",
    ),
    "school": ServiceSpec(
        tags=[
            {"amenity": "school"},
        ],
        blocksnet_name="school",
    ),
}

# Arctic/solver defaults:
# - cleanerreader default capacity when unknown is 600
# - create_blocks fallback "population" when missing is CONST_BASE_DEMAND=120
# We keep these values here for city-level fallback compatibility.
ARCTIC_DEFAULT_CAPACITY = 600.0
ARCTIC_DEFAULT_POPULATION = 120.0

# Per-service defaults intentionally follow arctic-style generic fallback.
DEFAULT_CAPACITY_BY_SERVICE = {
    service: spec.fallback_capacity for service, spec in SERVICE_SPECS.items()
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
        default=None,
        help=(
            "Path to pipeline_1 city bundle, e.g. "
            "/.../aggregated_spatial_pipeline/outputs/joint_inputs/barcelona_spain"
        ),
    )
    parser.add_argument(
        "--place",
        default=None,
        help=(
            "Optional place name used to auto-resolve "
            "aggregated_spatial_pipeline/outputs/joint_inputs/<place_slug>."
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
        default=None,
        help="Optional global override for service accessibility threshold in minutes.",
    )
    parser.add_argument(
        "--demand-per-1000",
        type=float,
        default=None,
        help="Optional global override for service demand per 1000 population.",
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


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def _resolve_city_dir(args: argparse.Namespace) -> Path:
    if args.joint_input_dir:
        return Path(args.joint_input_dir).resolve()
    if args.place:
        slug = _slugify(str(args.place))
        return (
            Path(__file__).resolve().parents[2]
            / "aggregated_spatial_pipeline"
            / "outputs"
            / "joint_inputs"
            / slug
        ).resolve()
    raise ValueError("Provide either --joint-input-dir or --place.")


def _load_blocksnet_service_defaults() -> dict[str, dict]:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "blocksnet"
        / "blocksnet"
        / "config"
        / "service_types"
        / "common"
        / "default.json"
    )
    items = json.loads(config_path.read_text(encoding="utf-8"))
    return {str(item.get("name")): item for item in items if item.get("name")}


BLOCKSNET_SERVICE_DEFAULTS = _load_blocksnet_service_defaults()
LOG_FORMAT = (
    "<green>{time:DD MMM HH:mm}</green> | "
    "<level>{level: <7}</level> | "
    "<magenta>{extra[tag]}</magenta> "
    "{message}"
)


def _log_name(path: Path | str | None) -> str:
    if path is None:
        return "none"
    try:
        return Path(path).name
    except Exception:
        return str(path)


def _log(message: str) -> None:
    logger.bind(tag="[pipeline_2_prepare]").info(message)


def _warn(message: str) -> None:
    logger.bind(tag="[pipeline_2_prepare]").warning(message)


def _configure_logging() -> None:
    logger.remove()
    logger.configure(patcher=lambda record: record["extra"].setdefault("tag", "[log]"))
    logger.add(
        sys.stderr,
        level="INFO",
        format=LOG_FORMAT,
        colorize=True,
    )


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


def _derive_has_living_from_buildings(units: gpd.GeoDataFrame, city_dir: Path) -> pd.Series:
    flags = pd.Series(False, index=units.index, dtype=bool)
    buildings_path = city_dir / "derived_layers" / "buildings_floor_enriched.parquet"
    if not buildings_path.exists():
        _warn(
            "buildings_floor_enriched.parquet not found; "
            "falling back to quarter-level residential proxies for accessibility preview mask."
        )
        return flags
    try:
        buildings = read_geodata(buildings_path)
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not read buildings_floor_enriched: {exc}")
        return flags
    if buildings.empty or "is_living" not in buildings.columns:
        _warn(
            "No is_living column in buildings_floor_enriched; "
            "falling back to quarter-level residential proxies for accessibility preview mask."
        )
        return flags

    living = pd.to_numeric(buildings["is_living"], errors="coerce")
    living_gdf = buildings.loc[living >= 0.5, ["geometry"]].copy()
    living_gdf = living_gdf[living_gdf.geometry.notna() & ~living_gdf.geometry.is_empty]
    if living_gdf.empty:
        return flags
    living_gdf["geometry"] = living_gdf.geometry.representative_point()
    if units.crs is not None and living_gdf.crs is not None and living_gdf.crs != units.crs:
        living_gdf = living_gdf.to_crs(units.crs)

    joined = gpd.sjoin(living_gdf, units[["geometry"]], how="inner", predicate="intersects")
    if joined.empty:
        return flags
    flags.loc[joined["index_right"].unique()] = True
    return flags


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
    queries = SERVICE_SPECS[service].tags
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
    if service == "hospital":
        beds = _first_number(row.get("beds"))
        if beds is not None and beds > 0:
            return beds
    cap = _first_number(row.get("capacity"))
    if cap is not None and cap > 0:
        return cap
    return float(DEFAULT_CAPACITY_BY_SERVICE[service])


def _service_accessibility_min(service: str, args: argparse.Namespace) -> float:
    if args.service_radius_min is not None:
        return float(args.service_radius_min)
    blocksnet_name = SERVICE_SPECS[service].blocksnet_name
    config = BLOCKSNET_SERVICE_DEFAULTS.get(blocksnet_name, {})
    return float(config.get("accessibility", 60.0))


def _service_demand_per_1000(service: str, args: argparse.Namespace) -> float:
    if args.demand_per_1000 is not None:
        return float(args.demand_per_1000)
    blocksnet_name = SERVICE_SPECS[service].blocksnet_name
    config = BLOCKSNET_SERVICE_DEFAULTS.get(blocksnet_name, {})
    return float(config.get("demand", ARCTIC_DEFAULT_POPULATION))


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


PIPELINE2_GALLERY_FILENAMES = {
    "accessibility_mean_time_map": "30_accessibility_mean_time_map.png",
    "hospital": "31_lp_hospital_provision_unmet.png",
    "polyclinic": "32_lp_polyclinic_provision_unmet.png",
    "school": "33_lp_school_provision_unmet.png",
}


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


def _try_load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _plot_accessibility_previews(
    units: gpd.GeoDataFrame,
    matrix_union: pd.DataFrame,
    out_dir: Path,
    *,
    boundary: gpd.GeoDataFrame | None = None,
    use_cache: bool = True,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import box

    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    access_map_path = out_dir / PIPELINE2_GALLERY_FILENAMES["accessibility_mean_time_map"]
    legacy_matrix_png = out_dir / "02_accessibility_matrix_sample_heatmap.png"
    if legacy_matrix_png.exists():
        legacy_matrix_png.unlink(missing_ok=True)
    if use_cache and access_map_path.exists():
        _log(f"Preview step: using cached accessibility map: {access_map_path.name}")
        out["accessibility_mean_time_map"] = str(access_map_path)
        return out

    boundary_plot = None
    outer_bg = None
    outer_bounds = None
    if boundary is not None and not boundary.empty:
        boundary_plot = boundary.copy()
        boundary_plot = boundary_plot[boundary_plot.geometry.notna() & ~boundary_plot.geometry.is_empty].copy()
        if not boundary_plot.empty and boundary_plot.crs is not None:
            try:
                boundary_plot = boundary_plot.to_crs("EPSG:3857")
            except Exception:
                pass
        if boundary_plot is not None and not boundary_plot.empty:
            minx, miny, maxx, maxy = boundary_plot.total_bounds
            pad_x = max((maxx - minx) * 0.08, 250.0)
            pad_y = max((maxy - miny) * 0.08, 250.0)
            outer_bounds = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
            outer_bg = gpd.GeoDataFrame({"geometry": [box(*outer_bounds)]}, crs=boundary_plot.crs)

    matrix_numeric = matrix_union.apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
    matrix_numeric = matrix_numeric.where(np.isfinite(matrix_numeric), np.nan)
    row_mean = matrix_numeric.mean(axis=1, skipna=True)

    units_plot = units[["geometry"]].copy()
    # Accessibility map policy: color only quarters with residential stock.
    residential_mask = pd.Series(False, index=units.index, dtype=bool)
    if "has_living_buildings" in units.columns:
        residential_mask = units["has_living_buildings"].fillna(False).astype(bool)
    elif "residential" in units.columns:
        residential_mask = pd.to_numeric(units["residential"], errors="coerce").fillna(0.0) > 0.0
    elif "living_area" in units.columns:
        residential_mask = pd.to_numeric(units["living_area"], errors="coerce").fillna(0.0) > 0.0
    elif "living_area_proxy" in units.columns:
        residential_mask = pd.to_numeric(units["living_area_proxy"], errors="coerce").fillna(0.0) > 0.0

    units_plot["access_time_mean_min"] = row_mean.reindex(units_plot.index).astype(float)
    units_plot["is_residential"] = residential_mask.reindex(units_plot.index).fillna(False).astype(bool)
    units_plot = units_plot[units_plot.geometry.notna() & ~units_plot.geometry.is_empty].copy()
    if not units_plot.empty:
        _log("Preview step: rendering accessibility mean-time map...")
        if units_plot.crs is not None:
            try:
                units_plot = units_plot.to_crs("EPSG:3857")
            except Exception:
                pass
        base_plot = units_plot.copy()
        res_plot = units_plot[units_plot["is_residential"]].copy()
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor("#6b6b6b")
        ax.set_facecolor("#6b6b6b")
        if outer_bg is not None and not outer_bg.empty:
            outer_bg.plot(ax=ax, facecolor="#6b6b6b", edgecolor="none", alpha=1.0, zorder=-20)
        if boundary_plot is not None and not boundary_plot.empty:
            boundary_plot.plot(ax=ax, facecolor="#f7f0dd", edgecolor="none", linewidth=0.0, alpha=1.0, zorder=-10)
        base_plot.plot(
            ax=ax,
            color="#f3f4f6",
            linewidth=0.05,
            edgecolor="#d1d5db",
            alpha=0.95,
            zorder=2,
        )
        if not res_plot.empty:
            res_plot.plot(
                ax=ax,
                column="access_time_mean_min",
                cmap="RdYlGn_r",
                linewidth=0.05,
                edgecolor="#d1d5db",
                legend=True,
                legend_kwds={"label": "mean travel time (min), higher = worse"},
                missing_kwds={"color": "#9ca3af", "label": "residential with no path"},
                zorder=3,
            )
        if boundary_plot is not None and not boundary_plot.empty:
            boundary_plot.boundary.plot(ax=ax, color="#ffffff", linewidth=1.4, zorder=4)
        if outer_bounds is not None:
            ax.set_xlim(outer_bounds[0], outer_bounds[2])
            ax.set_ylim(outer_bounds[1], outer_bounds[3])
        ax.set_title(
            "Accessibility: mean travel time (residential quarters only)",
            fontsize=19,
            fontweight="bold",
            color="#ffffff",
            pad=18,
        )
        ax.set_axis_off()
        fig.savefig(access_map_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        out["accessibility_mean_time_map"] = str(access_map_path)
        _log(f"Preview step: saved accessibility map: {access_map_path.name}")

    return out


def _coerce_solver_blocks_geodataframe(
    solver_blocks: pd.DataFrame | gpd.GeoDataFrame,
    *,
    quarters_ref: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame | None:
    gdf = solver_blocks.copy()
    if isinstance(gdf, gpd.GeoDataFrame) and "geometry" in gdf.columns:
        return gdf

    if "geometry" not in gdf.columns and quarters_ref is not None and "name" in gdf.columns:
        name_to_geom = {str(idx): geom for idx, geom in quarters_ref.geometry.items()}
        gdf["geometry"] = gdf["name"].astype(str).map(name_to_geom)

    if "geometry" not in gdf.columns:
        return None
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=(quarters_ref.crs if quarters_ref is not None else 4326))


def _plot_service_lp_preview(
    solver_blocks: pd.DataFrame | gpd.GeoDataFrame,
    service: str,
    out_path: Path,
    *,
    quarters_ref: gpd.GeoDataFrame | None = None,
    boundary: gpd.GeoDataFrame | None = None,
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import box

    if solver_blocks is None or solver_blocks.empty:
        return None
    _log(f"Preview step: rendering LP map for service [{service}]...")

    gdf = _coerce_solver_blocks_geodataframe(solver_blocks, quarters_ref=quarters_ref)
    if gdf is None:
        return None
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return None
    if gdf.crs is not None:
        try:
            gdf = gdf.to_crs("EPSG:3857")
        except Exception:
            pass
    boundary_plot = None
    outer_bg = None
    outer_bounds = None
    if boundary is not None and not boundary.empty:
        boundary_plot = boundary.copy()
        boundary_plot = boundary_plot[boundary_plot.geometry.notna() & ~boundary_plot.geometry.is_empty].copy()
        if not boundary_plot.empty and boundary_plot.crs is not None:
            try:
                boundary_plot = boundary_plot.to_crs("EPSG:3857")
            except Exception:
                pass
        if boundary_plot is not None and not boundary_plot.empty:
            minx, miny, maxx, maxy = boundary_plot.total_bounds
            pad_x = max((maxx - minx) * 0.08, 250.0)
            pad_y = max((maxy - miny) * 0.08, 250.0)
            outer_bounds = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
            outer_bg = gpd.GeoDataFrame({"geometry": [box(*outer_bounds)]}, crs=boundary_plot.crs)

    provision_col = "provision_strong" if "provision_strong" in gdf.columns else "provision" if "provision" in gdf.columns else None
    demand_without_col = "demand_without" if "demand_without" in gdf.columns else None
    if provision_col is None:
        return None

    provision = pd.to_numeric(gdf[provision_col], errors="coerce")
    gdf["provision_binary"] = np.where(provision >= 1.0, "good", "bad")
    if provision.notna().sum() > 0:
        gdf.loc[provision.isna(), "provision_binary"] = "missing"

    color_map = {"good": "#16a34a", "bad": "#dc2626", "missing": "#9ca3af"}
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 12),
        gridspec_kw={"height_ratios": [5, 1.7], "hspace": 0.18},
    )
    ax = axes[0]
    hist_ax = axes[1]
    fig.patch.set_facecolor("#6b6b6b")
    ax.set_facecolor("#6b6b6b")
    hist_ax.set_facecolor("#f7f0dd")
    if outer_bg is not None and not outer_bg.empty:
        outer_bg.plot(ax=ax, facecolor="#6b6b6b", edgecolor="none", alpha=1.0, zorder=-20)
    if boundary_plot is not None and not boundary_plot.empty:
        boundary_plot.plot(ax=ax, facecolor="#f7f0dd", edgecolor="none", linewidth=0.0, alpha=1.0, zorder=-10)
    for status in ("good", "bad", "missing"):
        part = gdf[gdf["provision_binary"] == status]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            color=color_map[status],
            linewidth=0.05,
            edgecolor="#d1d5db",
            alpha=0.9,
            zorder=2,
        )
    if boundary_plot is not None and not boundary_plot.empty:
        boundary_plot.boundary.plot(ax=ax, color="#ffffff", linewidth=1.4, zorder=3)
    if outer_bounds is not None:
        ax.set_xlim(outer_bounds[0], outer_bounds[2])
        ax.set_ylim(outer_bounds[1], outer_bounds[3])
    good_cnt = int((gdf["provision_binary"] == "good").sum())
    bad_cnt = int((gdf["provision_binary"] == "bad").sum())
    miss_cnt = int((gdf["provision_binary"] == "missing").sum())
    ax.set_title(
        f"{service}: provision status (1=good, <1=bad) | good={good_cnt}, bad={bad_cnt}, missing={miss_cnt}",
        fontsize=16,
        fontweight="bold",
        color="#ffffff",
        pad=16,
    )
    ax.set_axis_off()

    prov_hist = pd.to_numeric(gdf[provision_col], errors="coerce").dropna().clip(lower=0.0, upper=1.0)
    if not prov_hist.empty:
        bins = np.linspace(0.0, 1.0, 21)
        hist_ax.hist(prov_hist.to_numpy(dtype=float), bins=bins, color="#94a3b8", edgecolor="#334155", alpha=0.95)
        hist_ax.axvline(1.0, color="#16a34a", linestyle="--", linewidth=1.6, label="target=1.0")
        hist_ax.set_xlim(0.0, 1.0)
        hist_ax.set_xlabel("provision")
        hist_ax.set_ylabel("blocks")
        hist_ax.set_title("Provision histogram", fontsize=10)
        hist_ax.grid(alpha=0.25, axis="y")
        hist_ax.legend(loc="upper left", fontsize=8, frameon=True)
    else:
        hist_ax.text(0.5, 0.5, "no provision values", ha="center", va="center")
        hist_ax.set_axis_off()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    _log(f"Preview step: saved LP map for service [{service}]: {out_path.name}")
    return str(out_path)


def main() -> None:
    _configure_logging()
    args = parse_args()
    _configure_osmnx(args)

    services = _ensure_services_valid(args.services)
    city_dir = _resolve_city_dir(args)

    boundary_path = city_dir / "analysis_territory" / "buffer.parquet"
    quarters_path = city_dir / "derived_layers" / "quarters_clipped.parquet"
    graph_path = city_dir / "intermodal_graph_iduedu" / "graph.pkl"

    output_root = city_dir / "pipeline_2"
    raw_dir = output_root / "services_raw"
    prepared_dir = output_root / "prepared"
    solver_dir = output_root / "solver_inputs"
    preview_dir = city_dir / "preview_png" / "all_together"
    manifest_path = output_root / "manifest_prepare_solver_inputs.json"

    _log(f"Starting preparation for services={services}")
    _log(f"Using city bundle: {city_dir.name}")
    _log(f"Territory boundary: {_log_name(boundary_path)}")
    if args.no_cache:
        _warn(
            "Cache mode: disabled (--no-cache). "
            "Raw layers, matrix, and solver inputs may be rebuilt."
        )
    else:
        _log("Cache mode: enabled")

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
            _log(f"Using cached raw service layer [{service}]: {_log_name(raw_path)} ({len(raw)} features)")
        else:
            raw = _download_service_raw(boundary, service)
            if raw.empty:
                _warn(f"Raw service layer [{service}] is empty.")
            else:
                raw["capacity_est"] = raw.apply(lambda row: _capacity_from_row(row, service), axis=1)
            _save_geodata(raw, raw_path)
            _log(f"Saved raw service layer [{service}]: {_log_name(raw_path)} ({len(raw)} features)")

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
    # Keep residential context for accessibility-map styling.
    for residential_col in ("residential", "living_area", "living_area_proxy"):
        if residential_col in quarters.columns:
            units[residential_col] = quarters[residential_col].reindex(units.index)

    for cap_col, series in capacity_columns.items():
        units[cap_col] = series.reindex(units.index).fillna(0.0).astype(float)

    # Strict rule: quarters without population are excluded from all pipeline_2 calculations.
    units_mask = units["population"] > 0
    units = units[units_mask].copy()
    units["has_living_buildings"] = _derive_has_living_from_buildings(units, city_dir)
    units["unit_name"] = units.index.astype(str)
    legacy_demand_per_1000 = (
        float(args.demand_per_1000)
        if args.demand_per_1000 is not None
        else _service_demand_per_1000(services[0], args)
    )
    units["demand_base"] = _calc_demand(units["population"], legacy_demand_per_1000)
    living_units = int(units["has_living_buildings"].fillna(False).sum())
    _log(
        f"Active units for solver/matrix: {len(units)} (population>0 only); "
        f"units with is_living buildings={living_units}"
    )

    units_path = prepared_dir / "units_union.parquet"
    _save_geodata(units, units_path)

    # 3) Accessibility matrix between selected units (same city territory / same graph).
    matrix_path = prepared_dir / "adj_matrix_time_min_union.parquet"
    if matrix_path.exists() and (not args.no_cache):
        matrix_union = pd.read_parquet(matrix_path)
        _log(f"Using cached accessibility matrix: {_log_name(matrix_path)} ({matrix_union.shape[0]}x{matrix_union.shape[1]})")
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
            f"Saved accessibility matrix: {_log_name(matrix_path)} "
            f"({matrix_union.shape[0]}x{matrix_union.shape[1]}), elapsed={elapsed:.1f}s"
        )

    _log("STEP solver_prep: building per-service solver-ready blocks and links.")
    _log("STEP previews: generating PNGs for accessibility and LP outputs.")

    # 4) Per-service solver-ready tables (demand_within/demand_without/capacity_left/provision).
    service_outputs: dict[str, dict] = {}
    preview_outputs: dict[str, object] = {}
    preview_outputs.update(
        _plot_accessibility_previews(units, matrix_union, preview_dir, boundary=boundary, use_cache=(not args.no_cache))
    )
    for service in services:
        if float(raw_stats.get(service, {}).get("capacity_total", 0.0)) <= 0.0:
            _warn(f"Service [{service}] has zero total capacity in territory; skipping.")
            service_outputs[service] = {
                "skipped": True,
                "reason": "zero_capacity_in_territory",
            }
            continue

        service_dir = solver_dir / service
        blocks_path = service_dir / "blocks_solver.parquet"
        matrix_service_path = service_dir / "adj_matrix_time_min.parquet"
        links_path = service_dir / "provision_links.csv"
        summary_path = service_dir / "summary.json"
        lp_preview_target = preview_dir / PIPELINE2_GALLERY_FILENAMES[service]

        if (not args.no_cache) and blocks_path.exists() and matrix_service_path.exists() and summary_path.exists():
            cached_summary = _try_load_json(summary_path) or {}
            expected_radius = _service_accessibility_min(service, args)
            expected_demand = _service_demand_per_1000(service, args)
            if cached_summary.get("block_selection_policy") != LP_BLOCK_SELECTION_POLICY:
                _warn(
                    f"Cached solver input [{service}] is outdated for current block-selection policy "
                    f"({LP_BLOCK_SELECTION_POLICY}). Rebuilding."
                )
            elif float(cached_summary.get("service_radius_min", -1.0)) != float(expected_radius):
                _warn(
                    f"Cached solver input [{service}] uses outdated service_radius_min "
                    f"({cached_summary.get('service_radius_min')} != {expected_radius}). Rebuilding."
                )
            elif float(cached_summary.get("service_demand_per_1000", -1.0)) != float(expected_demand):
                _warn(
                    f"Cached solver input [{service}] uses outdated service_demand_per_1000 "
                    f"({cached_summary.get('service_demand_per_1000')} != {expected_demand}). Rebuilding."
                )
            else:
                _log(f"Using cached solver input [{service}]: {_log_name(summary_path)}")
                lp_preview_path = str(lp_preview_target) if lp_preview_target.exists() else None
                if lp_preview_path is None:
                    try:
                        cached_blocks = pd.read_parquet(blocks_path)
                        lp_preview_path = _plot_service_lp_preview(
                            cached_blocks,
                            service,
                            lp_preview_target,
                            quarters_ref=quarters,
                            boundary=boundary,
                        )
                    except Exception:
                        lp_preview_path = None
                service_outputs[service] = {
                    "summary": str(summary_path),
                    "blocks_solver": str(blocks_path),
                    "adj_matrix": str(matrix_service_path),
                    "provision_links": str(links_path),
                    "blocks_count": int(cached_summary.get("blocks_count", 0)),
                    "lp_preview_png": lp_preview_path,
                }
                continue

        cap_col = f"capacity_{service}"
        blocks = units[["unit_name", "population", "demand_base", cap_col, "geometry", "has_living_buildings"]].copy()
        blocks = blocks.rename(columns={cap_col: "capacity"})
        service_demand_per_1000 = _service_demand_per_1000(service, args)
        service_radius_min = _service_accessibility_min(service, args)
        blocks["demand"] = _calc_demand(blocks["population"], service_demand_per_1000)
        # LP policy: include only quarters with living buildings OR own service capacity.
        blocks["has_living_buildings"] = blocks["has_living_buildings"].fillna(False).astype(bool)
        blocks = blocks[blocks["has_living_buildings"] | (blocks["capacity"] > 0)].copy()
        if blocks.empty:
            _warn(f"No active blocks for service [{service}] after filtering. Skipping.")
            service_outputs[service] = {
                "skipped": True,
                "reason": "no_living_or_service_blocks_after_filtering",
            }
            continue

        _log(
            f"Service [{service}] provisioning prep: blocks={len(blocks)} "
            f"(living={int(blocks['has_living_buildings'].sum())}, capacity>0={int((blocks['capacity'] > 0).sum())}, "
            f"accessibility={service_radius_min:.1f} min, demand_per_1000={service_demand_per_1000:.1f})"
        )
        sub_mx = matrix_union.loc[blocks.index, blocks.index].copy()
        provision_started = time.time()
        provision_df, links_df = competitive_provision(
            blocks_df=blocks[["population", "demand", "capacity", "geometry"]].copy(),
            accessibility_matrix=sub_mx,
            accessibility=int(math.ceil(service_radius_min)),
            demand=None,
            self_supply=True,
            max_depth=int(args.provision_max_depth),
        )
        _log(f"Service [{service}] competitive_provision finished in {time.time() - provision_started:.1f}s")
        solver_blocks = provision_df.copy()
        if "geometry" not in solver_blocks.columns and "geometry" in blocks.columns:
            solver_blocks = solver_blocks.join(blocks[["geometry"]], how="left")
        if not isinstance(solver_blocks, gpd.GeoDataFrame) and "geometry" in solver_blocks.columns:
            solver_blocks = gpd.GeoDataFrame(solver_blocks, geometry="geometry", crs=blocks.crs)
        solver_blocks["name"] = solver_blocks.index.astype(str)
        solver_blocks["service_name"] = service
        solver_blocks["service_radius_min"] = float(service_radius_min)
        solver_blocks["service_demand_per_1000"] = float(service_demand_per_1000)
        # Compatibility with arctic solver runner fields.
        solver_blocks["provision"] = solver_blocks["provision_strong"].fillna(0.0)

        _save_geodata(solver_blocks, blocks_path)
        _save_dataframe(sub_mx, matrix_service_path)
        links_path.parent.mkdir(parents=True, exist_ok=True)
        links_df.reset_index().to_csv(links_path, index=False)
        if (not args.no_cache) and lp_preview_target.exists():
            lp_preview_path = str(lp_preview_target)
        else:
            lp_preview_path = _plot_service_lp_preview(
                solver_blocks,
                service,
                lp_preview_target,
                quarters_ref=quarters,
                boundary=boundary,
            )

        summary = {
            "service": service,
            "block_selection_policy": LP_BLOCK_SELECTION_POLICY,
            "blocks_count": int(len(solver_blocks)),
            "blocks_with_living": int(blocks["has_living_buildings"].sum()),
            "blocks_with_service_capacity": int((blocks["capacity"] > 0).sum()),
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
            "lp_preview_png": lp_preview_path,
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
        "service_radius_min": float(args.service_radius_min) if args.service_radius_min is not None else None,
        "demand_per_1000": float(args.demand_per_1000) if args.demand_per_1000 is not None else None,
        "units_union": str(units_path),
        "adj_matrix_union": str(matrix_path),
        "previews_dir": str(preview_dir),
        "preview_outputs": preview_outputs,
        "raw_services": raw_stats,
        "solver_outputs": service_outputs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Done. Manifest: {_log_name(manifest_path)}")


if __name__ == "__main__":
    main()

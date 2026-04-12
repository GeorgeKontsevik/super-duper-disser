from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import seaborn as sns
from esda.moran import Moran, Moran_BV, Moran_Local, Moran_Local_BV
import libpysal.weights as lw
from libpysal.weights import KNN, Queen
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib import colors as mcolors
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
import statsmodels.api as sm
import statsmodels.formula.api as smf

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata
from aggregated_spatial_pipeline.pipeline.crosswalks import build_crosswalk
from aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs import (
    SERVICE_SPECS,
    _calc_service_demand,
    _capacity_from_row as _pipeline2_capacity_from_row,
    _normalize_raw_osm,
    _plot_service_lp_preview,
    _read_graph_pickle,
    _run_arctic_lp_provision,
    _service_accessibility_min,
    _service_demand_per_1000,
    _to_points,
)
from aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters import (
    CLASS_COLORS,
    CLASS_LABELS,
)
from aggregated_spatial_pipeline.pipeline.transfers import apply_transfer_rule
from aggregated_spatial_pipeline.runtime_config import configure_logger, ensure_repo_mplconfigdir
from aggregated_spatial_pipeline.visualization import (
    apply_preview_canvas,
    footer_text,
    legend_bottom,
    normalize_preview_gdf,
    save_preview_figure,
)
from blocksnet.relations import calculate_accessibility_matrix


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT
SUBPROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SUBPROJECT_ROOT / "outputs"
DEFAULT_JOINT_INPUT_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs"
DEFAULT_CITIES = ("warsaw_poland", "berlin_germany")
PROB_COLUMNS = list(CLASS_LABELS.keys())
DEFAULT_STOP_MODALITIES = ("bus", "tram", "trolleybus")
SOCIAL_SERVICES = ("school", "polyclinic", "hospital")
CROSS_CITY_XLABEL_ROTATION = 82
CROSS_CITY_XLABEL_FONTSIZE = 7
CROSS_CITY_MIN_BUS_ROUTES = 5
CROSS_CITY_MIN_POPULATION = 150000
LOCAL_MORAN_MIN_GLOBAL_I = 0.6
EFFECT_HIGHLIGHT_MIN_ABS = 0.3
NEAREST_SERVICE_UNREACHABLE_MIN = 120.0

ensure_repo_mplconfigdir("mpl-sbd-service-accessibility", root=REPO_ROOT)


def _capacity_from_row(row: pd.Series, service: str) -> float:
    return _pipeline2_capacity_from_row(row, service, allow_default_fallback=True)


def _download_service_raw(boundary: gpd.GeoDataFrame, service: str) -> gpd.GeoDataFrame:
    boundary_wgs84 = boundary.to_crs(4326)
    polygon = boundary_wgs84.union_all()
    parts: list[gpd.GeoDataFrame] = []
    for tags in SERVICE_SPECS[service].tags:
        try:
            raw = ox.features.features_from_polygon(polygon, tags=tags)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        if not isinstance(raw, gpd.GeoDataFrame):
            raw = gpd.GeoDataFrame(raw, geometry="geometry", crs=4326)
        parts.append(raw)
    if not parts:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=boundary_wgs84.crs)
    merged = pd.concat(parts, axis=0, ignore_index=False)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=parts[0].crs or boundary_wgs84.crs)
    return merged


@dataclass(frozen=True)
class CityPaths:
    slug: str
    city_dir: Path
    output_dir: Path
    boundary_path: Path
    blocks_path: Path
    street_cells_path: Path
    graph_path: Path
    connectpt_dir: Path
    pipeline2_dir: Path
    pipeline2_prepared_dir: Path
    pipeline2_services_raw_dir: Path
    pipeline2_solver_inputs_dir: Path


def _configure_logging() -> None:
    configure_logger("[sbd-service-accessibility]")


def _log(message: str) -> None:
    logger.bind(tag="[sbd-service-accessibility]").info(message)


def _warn(message: str) -> None:
    logger.bind(tag="[sbd-service-accessibility]").warning(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run service/accessibility vs street-pattern experiments on already prepared city bundles."
        )
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=list(DEFAULT_CITIES),
        help="City slugs inside aggregated_spatial_pipeline/outputs/joint_inputs, or 'all'.",
    )
    parser.add_argument(
        "--joint-input-root",
        default=str(DEFAULT_JOINT_INPUT_ROOT),
        help="Root with prepared city bundles.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for this experiment subproject.",
    )
    parser.add_argument(
        "--services",
        nargs="+",
        default=list(SERVICE_SPECS.keys()),
        help="Services to analyse.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=199,
        help="Permutation count for Moran/BV Moran.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild prepared tables, service downloads and accessibility matrices.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Do not download missing service layers; use only already cached local data.",
    )
    parser.add_argument(
        "--cross-city-only",
        action="store_true",
        help="Skip per-city outputs and compute only cross-city accessibility-focused outputs.",
    )
    parser.add_argument(
        "--cross-city-min-coverage-pct",
        type=float,
        default=None,
        help=(
            "Optional minimum combined territory coverage (buildings + buffered roads, in percent) "
            "required for a city to be included in cross-city outputs."
        ),
    )
    parser.add_argument(
        "--cross-city-roads-buffer-m",
        type=float,
        default=8.0,
        help="Road buffer radius in meters used in cross-city coverage screening.",
    )
    parser.add_argument(
        "--osmnx-timeout-s",
        type=int,
        default=180,
        help="Timeout for OSM service download calls.",
    )
    parser.add_argument(
        "--use-sm-imputed",
        action="store_true",
        help=(
            "Use quarters_sm_imputed.parquet as blocks source when available. "
            "Default behavior uses quarters_clipped.parquet."
        ),
    )
    parser.add_argument(
        "--require-ready-data",
        action="store_true",
        help=(
            "Use only already prepared city artifacts. "
            "Do not download, infer, or rebuild intermodal/service inputs from fallback sources."
        ),
    )
    return parser.parse_args()


def _resolve_city_paths(slug: str, joint_input_root: Path, output_root: Path, *, use_sm_imputed: bool = False) -> CityPaths:
    city_dir = (joint_input_root / slug).resolve()
    if not city_dir.exists():
        raise FileNotFoundError(f"City bundle not found: {city_dir}")
    blocks_path = (
        city_dir / "derived_layers" / "quarters_sm_imputed.parquet"
        if use_sm_imputed
        else city_dir / "derived_layers" / "quarters_clipped.parquet"
    )
    if not blocks_path.exists():
        raise FileNotFoundError(f"Blocks layer not found for city {slug}: {blocks_path}")
    paths = CityPaths(
        slug=slug,
        city_dir=city_dir,
        output_dir=(output_root / slug).resolve(),
        boundary_path=city_dir / "analysis_territory" / "buffer.parquet",
        blocks_path=blocks_path,
        street_cells_path=city_dir / "street_pattern" / slug / "predicted_cells.geojson",
        graph_path=city_dir / "intermodal_graph_iduedu" / "graph.pkl",
        connectpt_dir=city_dir / "connectpt_osm",
        pipeline2_dir=city_dir / "pipeline_2",
        pipeline2_prepared_dir=city_dir / "pipeline_2" / "prepared",
        pipeline2_services_raw_dir=city_dir / "pipeline_2" / "services_raw",
        pipeline2_solver_inputs_dir=city_dir / "pipeline_2" / "solver_inputs",
    )
    for check in (paths.boundary_path, paths.street_cells_path, paths.graph_path):
        if not check.exists():
            raise FileNotFoundError(f"Required input not found for city {slug}: {check}")
    return paths


def _discover_available_city_slugs(
    joint_input_root: Path,
    output_root: Path,
    *,
    use_sm_imputed: bool = False,
) -> list[str]:
    slugs: list[str] = []
    for city_dir in sorted(p for p in joint_input_root.iterdir() if p.is_dir()):
        slug = city_dir.name
        try:
            _resolve_city_paths(slug, joint_input_root, output_root, use_sm_imputed=use_sm_imputed)
        except Exception:
            continue
        slugs.append(slug)
    return slugs


def _compute_city_coverage_pct(paths: CityPaths, roads_buffer_m: float) -> dict[str, float | int | str]:
    territory = _load_boundary(paths.boundary_path)
    local_crs = territory.estimate_utm_crs() or "EPSG:3857"
    territory_local = territory.to_crs(local_crs)
    territory_geom = territory_local.union_all()
    territory_area = float(territory_geom.area)
    if territory_area <= 0.0:
        return {
            "city_slug": paths.slug,
            "coverage_pct": np.nan,
            "territory_area_m2": 0.0,
            "buildings_coverage_area_m2": 0.0,
            "roads_coverage_area_m2": 0.0,
            "combined_coverage_area_m2": 0.0,
            "buildings_count": 0,
            "roads_count": 0,
        }

    territory_frame = gpd.GeoDataFrame({"geometry": [territory_geom]}, crs=local_crs)
    buildings_path = paths.city_dir / "blocksnet_raw_osm" / "buildings.parquet"
    roads_path = paths.city_dir / "derived_layers" / "roads_drive_osmnx.parquet"

    buildings_area = 0.0
    roads_area = 0.0
    buildings_count = 0
    roads_count = 0
    combined_geoms: list = []

    if buildings_path.exists():
        b = read_geodata(buildings_path)
        if not b.empty:
            b = b.to_crs(local_crs)
            b = b[b.geometry.notna() & ~b.geometry.is_empty].copy()
            b = b[b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            buildings_count = int(len(b))
            if not b.empty:
                b_clip = gpd.clip(b[["geometry"]], territory_frame, keep_geom_type=False)
                b_clip = b_clip[b_clip.geometry.notna() & ~b_clip.geometry.is_empty].copy()
                if not b_clip.empty:
                    b_geom = b_clip.union_all()
                    buildings_area = float(b_geom.area)
                    combined_geoms.append(b_geom)

    if roads_path.exists():
        r = read_geodata(roads_path)
        if not r.empty:
            r = r.to_crs(local_crs)
            r = r[r.geometry.notna() & ~r.geometry.is_empty].copy()
            r = r[r.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
            roads_count = int(len(r))
            if not r.empty:
                r = r.copy()
                r["geometry"] = r.geometry.buffer(float(roads_buffer_m))
                r = r[r.geometry.notna() & ~r.geometry.is_empty].copy()
                r_clip = gpd.clip(r[["geometry"]], territory_frame, keep_geom_type=False)
                r_clip = r_clip[r_clip.geometry.notna() & ~r_clip.geometry.is_empty].copy()
                if not r_clip.empty:
                    r_geom = r_clip.union_all()
                    roads_area = float(r_geom.area)
                    combined_geoms.append(r_geom)

    combined_area = float(gpd.GeoSeries(combined_geoms, crs=local_crs).union_all().area) if combined_geoms else 0.0
    return {
        "city_slug": paths.slug,
        "coverage_pct": (combined_area / territory_area) * 100.0,
        "territory_area_m2": territory_area,
        "buildings_coverage_area_m2": buildings_area,
        "roads_coverage_area_m2": roads_area,
        "combined_coverage_area_m2": combined_area,
        "buildings_count": buildings_count,
        "roads_count": roads_count,
    }


def _save_geodata(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prepare_geodata_for_parquet(gdf).to_parquet(path)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _read_dataframe_parquet_or_none(path: Path, *, label: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        _warn(f"{label}: failed to read parquet {path.name}; treating cache as invalid ({exc})")
        return None


def _load_blocks(blocks_path: Path) -> gpd.GeoDataFrame:
    blocks = read_geodata(blocks_path).copy()
    if blocks.empty:
        raise ValueError(f"Blocks layer is empty: {blocks_path}")
    blocks["block_id"] = blocks.index.astype(str)
    blocks["population"] = 0.0
    for candidate in ("population_total", "population_proxy", "population"):
        if candidate in blocks.columns:
            blocks["population"] = pd.to_numeric(blocks[candidate], errors="coerce").fillna(0.0)
            break
    return blocks


def _load_boundary(boundary_path: Path) -> gpd.GeoDataFrame:
    boundary = read_geodata(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary layer is empty: {boundary_path}")
    return boundary


def _load_street_cells(street_cells_path: Path) -> gpd.GeoDataFrame:
    cells = gpd.read_file(street_cells_path)
    if cells.empty:
        raise ValueError(f"Street-pattern predicted_cells layer is empty: {street_cells_path}")
    cells["grid_id"] = cells["cell_id"].astype(str)
    return cells


def _rename_prob_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    for prob_col, label in CLASS_LABELS.items():
        if prob_col in renamed.columns:
            slug = label.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")
            slug = slug.replace("__", "_")
            renamed[f"street_pattern_prob_{slug}"] = pd.to_numeric(renamed[prob_col], errors="coerce").fillna(0.0)
    return renamed


def _transfer_street_pattern_to_blocks(
    blocks: gpd.GeoDataFrame,
    street_cells: gpd.GeoDataFrame,
    prepared_dir: Path,
    *,
    no_cache: bool,
) -> gpd.GeoDataFrame:
    output_path = prepared_dir / "blocks_with_street_pattern.parquet"
    if output_path.exists() and not no_cache:
        return read_geodata(output_path)

    source_columns = ["grid_id", *[c for c in street_cells.columns if c not in {"geometry", "grid_id"}], "geometry"]
    source = street_cells[source_columns].copy()
    target = blocks.copy()

    polygon_types = {"Polygon", "MultiPolygon"}
    source_poly = source[source.geometry.geom_type.isin(polygon_types)].copy()
    target_poly = target[target.geometry.geom_type.isin(polygon_types)].copy()
    skipped_source = int(len(source) - len(source_poly))
    skipped_target = int(len(target) - len(target_poly))
    if skipped_source > 0 or skipped_target > 0:
        _warn(
            f"[{prepared_dir.parent.name}] street-pattern transfer: "
            f"skipping non-polygon geometries (source={skipped_source}, target={skipped_target})."
        )

    if source_poly.empty or target_poly.empty:
        transferred = target.copy()
        for prob_col in PROB_COLUMNS:
            if prob_col not in transferred.columns:
                transferred[prob_col] = 0.0
        transferred["street_pattern_class"] = "unknown"
    else:
        crosswalk = build_crosswalk(source_poly, target_poly, "grid", "block")

        transferred_poly = apply_transfer_rule(
            source_gdf=source_poly,
            target_gdf=target_poly,
            crosswalk_gdf=crosswalk,
            source_layer="grid",
            target_layer="block",
            attribute="street_pattern_probs",
            aggregation_method="weighted_mean",
            weight_field="intersection_area",
        )
        transferred_poly = apply_transfer_rule(
            source_gdf=source_poly,
            target_gdf=transferred_poly,
            crosswalk_gdf=crosswalk,
            source_layer="grid",
            target_layer="block",
            attribute="street_pattern_class",
            aggregation_method="majority_vote",
            weight_field="intersection_area",
        )
        transferred_poly = apply_transfer_rule(
            source_gdf=source_poly,
            target_gdf=transferred_poly,
            crosswalk_gdf=crosswalk,
            source_layer="grid",
            target_layer="block",
            attribute="street_pattern_class_shares",
            aggregation_method="class_area_shares",
            weight_field="intersection_area",
        )
        transferred = target.copy()
        for col in transferred_poly.columns:
            if col == "geometry":
                continue
            if col not in transferred.columns:
                transferred[col] = pd.Series(index=transferred.index, dtype=transferred_poly[col].dtype)
            transferred.loc[transferred_poly.index, col] = transferred_poly[col]

    for prob_col in PROB_COLUMNS:
        if prob_col not in transferred.columns:
            transferred[prob_col] = 0.0
        transferred[prob_col] = pd.to_numeric(transferred[prob_col], errors="coerce").fillna(0.0)
    if "street_pattern_class" not in transferred.columns:
        transferred["street_pattern_class"] = "unknown"

    transferred["street_pattern_covered_mass"] = transferred[PROB_COLUMNS].sum(axis=1)
    dominant_fallback = pd.Series("unknown", index=transferred.index, dtype=object)
    covered_mask = transferred["street_pattern_covered_mass"] > 0.0
    if covered_mask.any():
        dominant_fallback.loc[covered_mask] = (
            transferred.loc[covered_mask, PROB_COLUMNS]
            .idxmax(axis=1)
            .map(CLASS_LABELS)
            .fillna("unknown")
        )
    transferred["street_pattern_dominant_class"] = (
        transferred["street_pattern_class"]
        .fillna(dominant_fallback)
        .astype(str)
    )
    transferred.loc[transferred["street_pattern_covered_mass"] <= 0.0, "street_pattern_dominant_class"] = "unknown"
    transferred = _rename_prob_columns(transferred)
    _save_geodata(transferred, output_path)
    return transferred


def _collect_service_metrics(
    *,
    boundary: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    service: str,
    raw_dir: Path,
    reuse_raw_path: Path | None,
    fallback_raw_path: Path | None,
    fallback_buildings_path: Path | None,
    no_cache: bool,
    local_only: bool,
    require_ready_data: bool,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    raw_path = raw_dir / f"{service}.parquet"
    empty_raw = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=boundary.to_crs(4326).crs)
    empty_raw["capacity_est"] = []
    raw: gpd.GeoDataFrame | None = None

    if raw_path.exists():
        cached_raw = read_geodata(raw_path)
        if not no_cache or not cached_raw.empty:
            raw = cached_raw

    if raw is None and reuse_raw_path is not None and reuse_raw_path.exists():
        raw = read_geodata(reuse_raw_path)
        if raw_path != reuse_raw_path:
            _save_geodata(raw, raw_path)

    if raw is None and require_ready_data:
        raise FileNotFoundError(
            f"Missing prepared raw service layer for [{service}]. "
            f"Checked experiment cache {raw_path} and prepared source {reuse_raw_path}."
        )

    if raw is None and fallback_raw_path is not None and fallback_raw_path.exists():
        raw = read_geodata(fallback_raw_path)
        if raw_path != fallback_raw_path:
            _save_geodata(raw, raw_path)

    if raw is None and fallback_buildings_path is not None and fallback_buildings_path.exists():
        raw = _extract_service_raw_from_local_buildings(fallback_buildings_path, service, boundary)
        if not raw.empty:
            _save_geodata(raw, raw_path)

    if raw is None:
        if local_only:
            raw = empty_raw
        else:
            raw = _download_service_raw(boundary, service)
            if not raw.empty:
                raw = _normalize_raw_osm(raw)
                raw["capacity_est"] = raw.apply(lambda row: _capacity_from_row(row, service), axis=1)
            else:
                raw = empty_raw
            _save_geodata(raw, raw_path)

    if "capacity_est" not in raw.columns:
        raw = raw.copy()
        raw["capacity_est"] = raw.apply(lambda row: _capacity_from_row(row, service), axis=1) if not raw.empty else 0.0
        _save_geodata(raw, raw_path)

    metrics = pd.DataFrame(index=blocks.index)
    metrics[f"service_count_{service}"] = 0.0
    metrics[f"service_capacity_{service}"] = 0.0
    metrics[f"service_has_{service}"] = 0.0

    if raw.empty:
        return raw, metrics

    points = _to_points(raw)
    points = points.to_crs(blocks.crs)
    joined_intersects = gpd.sjoin(
        points[["capacity_est", "geometry"]],
        blocks[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    joined_parts = [joined_intersects]
    unmatched_idx = points.index.difference(joined_intersects.index.unique())
    if len(unmatched_idx) > 0:
        nearest = gpd.sjoin_nearest(
            points.loc[unmatched_idx, ["capacity_est", "geometry"]],
            blocks[["geometry"]],
            how="left",
            distance_col="nearest_block_distance_m",
        )
        nearest = nearest[nearest["index_right"].notna()].copy()
        if not nearest.empty:
            joined_parts.append(nearest[["capacity_est", "geometry", "index_right"]].copy())
            _log(
                f"Service [{service}] nearest-block fallback assigned "
                f"{len(nearest)} feature(s) that did not intersect any block polygon."
            )
    joined = (
        pd.concat(joined_parts, axis=0, ignore_index=False)
        if joined_parts
        else pd.DataFrame(columns=["capacity_est", "geometry", "index_right"])
    )
    if not joined.empty:
        counts = joined.groupby("index_right").size().reindex(blocks.index).fillna(0.0)
        capacities = joined.groupby("index_right")["capacity_est"].sum().reindex(blocks.index).fillna(0.0)
        metrics[f"service_count_{service}"] = counts.astype(float)
        metrics[f"service_capacity_{service}"] = capacities.astype(float)
        metrics[f"service_has_{service}"] = (counts > 0).astype(float)

    return raw, metrics


def _extract_service_raw_from_local_buildings(
    buildings_path: Path,
    service: str,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    buildings = read_geodata(buildings_path)
    if buildings.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=boundary.to_crs(4326).crs)

    def _string_series(column: str) -> pd.Series:
        if column not in buildings.columns:
            return pd.Series("", index=buildings.index, dtype=object)
        return buildings[column].fillna("").astype(str).str.strip().str.lower()

    amenity = _string_series("amenity")
    healthcare = _string_series("healthcare")

    if service == "hospital":
        mask = amenity.eq("hospital") | healthcare.eq("hospital")
    elif service == "polyclinic":
        mask = amenity.isin({"clinic", "doctors"}) | healthcare.isin({"clinic", "doctor", "doctors"})
    elif service == "school":
        school_cols = [
            col
            for col in buildings.columns
            if col == "school" or col == "school_type" or col.startswith("school:")
        ]
        school_mask = pd.Series(False, index=buildings.index)
        for col in school_cols:
            values = _string_series(col)
            school_mask = school_mask | values.ne("") & ~values.isin({"0", "false", "no", "none", "nan"})
        mask = amenity.eq("school") | school_mask
    else:
        mask = pd.Series(False, index=buildings.index)

    extracted = buildings.loc[mask].copy()
    if extracted.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=boundary.to_crs(4326).crs)
    extracted["capacity_est"] = extracted.apply(lambda row: _capacity_from_row(row, service), axis=1)
    return extracted


def _available_stop_modalities(paths: CityPaths) -> list[str]:
    modalities: list[str] = []
    for modality in DEFAULT_STOP_MODALITIES:
        stops_path = paths.connectpt_dir / modality / "aggregated_stops.parquet"
        if stops_path.exists():
            modalities.append(modality)
    return modalities


def _collect_stop_metrics(
    *,
    blocks: gpd.GeoDataFrame,
    stops_path: Path,
    modality: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    stops = read_geodata(stops_path)
    metrics = pd.DataFrame(index=blocks.index)
    metrics[f"stop_count_{modality}"] = 0.0
    metrics[f"stop_has_{modality}"] = 0.0
    if stops.empty:
        return stops, metrics
    stops_local = stops.to_crs(blocks.crs)
    joined = gpd.sjoin(
        stops_local[["geometry"]],
        blocks[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return stops, metrics
    counts = joined.groupby("index_right").size().reindex(blocks.index).fillna(0.0).astype(float)
    metrics[f"stop_count_{modality}"] = counts
    metrics[f"stop_has_{modality}"] = (counts > 0).astype(float)
    return stops, metrics


def _compute_accessibility(
    *,
    blocks: gpd.GeoDataFrame,
    graph: nx.Graph,
    prepared_dir: Path,
    no_cache: bool,
    cache_name: str,
    reuse_matrix_path: Path | None = None,
    require_ready_data: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    matrix_path = prepared_dir / cache_name
    access_mask = pd.to_numeric(blocks["population"], errors="coerce").fillna(0.0) > 0.0
    active = blocks.loc[access_mask, ["block_id", "geometry"]].copy()
    active["block_id"] = active["block_id"].astype(str)
    active = active.set_index("block_id")
    if active.empty:
        empty_series = pd.Series(np.nan, index=blocks["block_id"].astype(str), dtype=float)
        return pd.DataFrame(), empty_series, access_mask
    if matrix_path.exists() and not no_cache:
        matrix = _read_dataframe_parquet_or_none(
            matrix_path,
            label=f"Accessibility matrix cache [{cache_name}]",
        )
    else:
        matrix = None
    if matrix is None and reuse_matrix_path is not None and reuse_matrix_path.exists():
        _log(
            f"Reusing prepared accessibility matrix: {reuse_matrix_path.name} "
            f"-> {matrix_path.name}"
        )
        matrix = _read_dataframe_parquet_or_none(
            reuse_matrix_path,
            label=f"Prepared accessibility matrix reuse [{cache_name}]",
        )
        if matrix is None:
            raise ValueError(f"Prepared accessibility matrix is unreadable: {reuse_matrix_path}")
        _save_dataframe(matrix, matrix_path)
    elif matrix is None and require_ready_data:
        raise FileNotFoundError(
            f"Missing prepared accessibility matrix: {matrix_path} "
            f"(reuse source: {reuse_matrix_path})"
        )
    elif matrix is None:
        _log(f"Computing accessibility matrix for active blocks: n={len(active)}")
        matrix = calculate_accessibility_matrix(active[["geometry"]].copy(), graph, weight_key="time_min")
        _save_dataframe(matrix, matrix_path)

    matrix_numeric = matrix.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean_time = matrix_numeric.mean(axis=1, skipna=True)
    mean_time.index = mean_time.index.astype(str)
    access_series = pd.Series(np.nan, index=blocks["block_id"].astype(str), dtype=float)
    access_series.loc[mean_time.index] = mean_time.astype(float).to_numpy()
    return matrix_numeric, access_series, access_mask


def _extract_walk_only_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    walk_graph = nx.MultiDiGraph()
    walk_graph.graph.update(graph.graph)
    walk_graph.graph["type"] = "walk_only"
    for node, data in graph.nodes(data=True):
        walk_graph.add_node(node, **data)
    for u, v, key, data in graph.edges(keys=True, data=True):
        if str(data.get("type", "")).lower() != "walk":
            continue
        walk_graph.add_edge(u, v, key=key, **data)
    isolates = [node for node in walk_graph.nodes if walk_graph.degree(node) == 0]
    if isolates:
        walk_graph.remove_nodes_from(isolates)
    return walk_graph


def _compute_service_assigned_accessibility(
    *,
    blocks: gpd.GeoDataFrame,
    graph: nx.Graph,
    service: str,
    prepared_dir: Path,
    no_cache: bool,
    context_label: str,
    reuse_service_dir: Path | None = None,
    require_ready_data: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame(index=blocks.index)
    metrics[f"service_provision_{service}_{context_label}"] = np.nan
    metrics[f"assigned_service_time_{service}_{context_label}"] = np.nan
    metrics[f"assigned_service_targets_{service}_{context_label}"] = np.nan
    metrics[f"nearest_service_time_{service}_{context_label}"] = np.nan

    capacity_col = f"service_capacity_{service}"
    if capacity_col not in blocks.columns:
        return pd.DataFrame(), metrics

    population = pd.to_numeric(blocks["population"], errors="coerce").fillna(0.0)
    capacity = pd.to_numeric(blocks[capacity_col], errors="coerce").fillna(0.0)
    select_mask = (population > 0.0) | (capacity > 0.0)
    selected = blocks.loc[select_mask, ["geometry"]].copy()
    if selected.empty or float(capacity.loc[select_mask].sum()) <= 0.0:
        return pd.DataFrame(), metrics

    selected["population"] = population.loc[select_mask].to_numpy(dtype=float)
    selected["capacity"] = capacity.loc[select_mask].to_numpy(dtype=float)
    selected["demand"] = _calc_service_demand(
        selected["population"],
        service,
        _service_demand_per_1000(service, argparse.Namespace(demand_per_1000=None)),
    ).astype(float)
    common_block_ids = pd.Index(blocks.index.astype(str))

    service_dir = prepared_dir / "service_accessibility" / context_label / service
    service_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = service_dir / "adj_matrix_time_min.parquet"
    blocks_solver_path = service_dir / "blocks_solver.parquet"
    links_path = service_dir / "provision_links.csv"
    reuse_matrix_path = reuse_service_dir / "adj_matrix_time_min.parquet" if reuse_service_dir is not None else None
    reuse_blocks_solver_path = reuse_service_dir / "blocks_solver.parquet" if reuse_service_dir is not None else None
    reuse_links_path = reuse_service_dir / "provision_links.csv" if reuse_service_dir is not None else None

    cached_bundle_ready = False
    if matrix_path.exists() and blocks_solver_path.exists() and links_path.exists() and not no_cache:
        service_matrix = _read_dataframe_parquet_or_none(
            matrix_path,
            label=f"Service accessibility cache [{service}/{context_label}]",
        )
        if service_matrix is not None:
            service_matrix.index = service_matrix.index.astype(str)
            service_matrix.columns = service_matrix.columns.astype(str)
            solver_blocks = read_geodata(blocks_solver_path)
            if "solver_index" in solver_blocks.columns:
                solver_blocks = solver_blocks.set_index("solver_index")
            elif "name" in solver_blocks.columns:
                solver_blocks = solver_blocks.set_index("name")
            else:
                solver_blocks.index = solver_blocks.index.astype(str)
            solver_blocks.index = solver_blocks.index.astype(str)
            links_df = pd.read_csv(links_path)
            cached_bundle_ready = True
    if cached_bundle_ready:
        pass
    elif (
        reuse_matrix_path is not None
        and reuse_blocks_solver_path is not None
        and reuse_links_path is not None
        and reuse_matrix_path.exists()
        and reuse_blocks_solver_path.exists()
        and reuse_links_path.exists()
    ):
        _log(
            f"Reusing prepared service accessibility [{service}/{context_label}] "
            f"from {reuse_service_dir.name}"
        )
        service_matrix = _read_dataframe_parquet_or_none(
            reuse_matrix_path,
            label=f"Prepared service accessibility reuse [{service}/{context_label}]",
        )
        if service_matrix is None:
            raise ValueError(f"Prepared service accessibility matrix is unreadable: {reuse_matrix_path}")
        service_matrix.index = service_matrix.index.astype(str)
        service_matrix.columns = service_matrix.columns.astype(str)
        solver_blocks = read_geodata(reuse_blocks_solver_path)
        if "solver_index" in solver_blocks.columns:
            solver_blocks = solver_blocks.set_index("solver_index")
        elif "name" in solver_blocks.columns:
            solver_blocks = solver_blocks.set_index("name")
        else:
            solver_blocks.index = solver_blocks.index.astype(str)
        solver_blocks.index = solver_blocks.index.astype(str)
        links_df = pd.read_csv(reuse_links_path)
        _save_dataframe(service_matrix, matrix_path)
        prepare_geodata_for_parquet(gpd.GeoDataFrame(solver_blocks, geometry="geometry", crs=blocks.crs)).to_parquet(
            blocks_solver_path,
            index=False,
        )
        _normalize_provision_links(links_df).to_csv(links_path, index=False)
    elif require_ready_data:
        raise FileNotFoundError(
            f"Missing prepared service accessibility [{service}/{context_label}] "
            f"under {service_dir} and reuse source {reuse_service_dir}."
        )
    else:
        service_matrix = calculate_accessibility_matrix(selected[["geometry"]].copy(), graph, weight_key="time_min")
        service_matrix = service_matrix.apply(pd.to_numeric, errors="coerce")
        service_radius_min = _service_accessibility_min(service, argparse.Namespace(service_radius_min=None))
        service_matrix.index = selected.index.astype(str)
        service_matrix.columns = selected.index.astype(str)
        solver_selected = gpd.GeoDataFrame(
            selected[["geometry", "population", "demand", "capacity"]].copy(),
            geometry="geometry",
            crs=blocks.crs,
        )
        solver_selected["name"] = selected.index.astype(str)
        solver_blocks, links_df = _run_arctic_lp_provision(
            blocks_df=solver_selected[["name", "population", "demand", "capacity", "geometry"]].copy(),
            accessibility_matrix=service_matrix,
            service=service,
            service_radius_min=float(service_radius_min),
            service_demand_per_1000=float(
                _service_demand_per_1000(service, argparse.Namespace(demand_per_1000=None))
            ),
        )
        solver_blocks = solver_blocks.copy()
        solver_blocks["solver_index"] = solver_blocks.index.astype(str)
        prepare_geodata_for_parquet(gpd.GeoDataFrame(solver_blocks, geometry="geometry", crs=solver_selected.crs)).to_parquet(
            blocks_solver_path,
            index=False,
        )
        _save_dataframe(service_matrix, matrix_path)
        _normalize_provision_links(links_df).to_csv(links_path, index=False)
        solver_blocks.index = solver_blocks.index.astype(str)

    # Pure accessibility metric (independent from provision/assignment):
    # minimal travel time from each selected block to nearest block that has service capacity.
    service_capacity_selected = pd.to_numeric(capacity.loc[select_mask], errors="coerce").fillna(0.0)
    service_targets = service_capacity_selected[service_capacity_selected > 0.0].index.astype(str).tolist()
    if service_targets:
        matrix_numeric_for_nearest = service_matrix.apply(pd.to_numeric, errors="coerce")
        try:
            nearest = matrix_numeric_for_nearest[service_targets].min(axis=1, skipna=True)
            nearest = pd.to_numeric(nearest, errors="coerce")
            nearest.index = nearest.index.astype(str)
            metrics[f"nearest_service_time_{service}_{context_label}"] = common_block_ids.map(nearest)
        except Exception:
            pass
    # Keep this metric strictly residential/populated.
    residential = pd.to_numeric(population, errors="coerce").fillna(0.0) > 0.0
    non_residential = ~residential
    nearest_col = f"nearest_service_time_{service}_{context_label}"
    nearest_vals = pd.to_numeric(metrics[nearest_col], errors="coerce")
    nearest_vals = nearest_vals.where(np.isfinite(nearest_vals), np.nan)
    # Do not drop underserved residential blocks from city-level distributions:
    # if nearest service is unreachable/undefined, keep them with a capped high travel time.
    nearest_vals = nearest_vals.where(~residential, nearest_vals.fillna(float(NEAREST_SERVICE_UNREACHABLE_MIN)))
    nearest_vals = nearest_vals.where(~non_residential, np.nan)
    metrics[nearest_col] = nearest_vals

    solver_blocks.index = solver_blocks.index.astype(str)
    provision_map = solver_blocks.get("provision_strong", solver_blocks.get("provision")).astype(float) if not solver_blocks.empty else pd.Series(dtype=float)
    metrics[f"service_provision_{service}_{context_label}"] = common_block_ids.map(provision_map)

    if links_df.empty:
        return service_matrix, metrics

    links = _normalize_provision_links(links_df)
    if not {"source", "target"}.issubset(links.columns):
        _warn(
            f"Service assigned accessibility [{service}/{context_label}] skipped weighted assignment details "
            f"because provision links do not contain expected source/target columns: {list(links.columns)}"
        )
        return service_matrix, metrics
    links["source"] = pd.to_numeric(links["source"], errors="coerce").astype("Int64")
    links["target"] = pd.to_numeric(links["target"], errors="coerce").astype("Int64")
    links = links.dropna(subset=["source", "target"]).copy()
    links["source"] = links["source"].astype(str)
    links["target"] = links["target"].astype(str)
    matrix_numeric = service_matrix.apply(pd.to_numeric, errors="coerce")
    matrix_numeric.index = matrix_numeric.index.astype(str)
    matrix_numeric.columns = matrix_numeric.columns.astype(str)
    links["travel_time_min"] = [
        float(matrix_numeric.at[src, tgt]) if src in matrix_numeric.index and tgt in matrix_numeric.columns else np.nan
        for src, tgt in zip(links["source"], links["target"])
    ]
    links["value"] = pd.to_numeric(links["value"], errors="coerce").fillna(0.0)
    valid_links = links.dropna(subset=["travel_time_min"]).copy()
    if valid_links.empty:
        return service_matrix, metrics
    grouped = valid_links.groupby("source")
    valid_links["weighted_time_component"] = (
        pd.to_numeric(valid_links["travel_time_min"], errors="coerce").fillna(0.0)
        * pd.to_numeric(valid_links["value"], errors="coerce").fillna(0.0)
    )
    weight_sum = grouped["value"].sum()
    weighted_sum = grouped["weighted_time_component"].sum()
    fallback_mean = grouped["travel_time_min"].mean()
    weighted_time = weighted_sum.div(weight_sum.where(weight_sum > 0)).where(weight_sum > 0, fallback_mean)
    target_count = grouped["target"].nunique().astype(float)
    metrics[f"assigned_service_time_{service}_{context_label}"] = common_block_ids.map(weighted_time)
    metrics[f"assigned_service_targets_{service}_{context_label}"] = common_block_ids.map(target_count)
    return service_matrix, metrics


def _normalize_provision_links(links_df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(links_df, pd.Series):
        value_name = links_df.name if links_df.name else "value"
        links = links_df.to_frame(name=value_name)
    else:
        links = links_df.copy()

    if "source" not in links.columns or "target" not in links.columns:
        links = links.reset_index()

    if "value" not in links.columns:
        candidate_value_cols = [col for col in links.columns if col not in {"source", "target", "index", "service", "level_0", "level_1"}]
        if len(candidate_value_cols) == 1:
            links = links.rename(columns={candidate_value_cols[0]: "value"})

    rename_map: dict[str, str] = {}
    if "source" not in links.columns:
        for candidate in ("index", "level_0"):
            if candidate in links.columns:
                rename_map[candidate] = "source"
                break
    if "target" not in links.columns:
        for candidate in ("service", "level_1"):
            if candidate in links.columns:
                rename_map[candidate] = "target"
                break
    links = links.rename(columns=rename_map)

    if "source" not in links.columns or "target" not in links.columns:
        meta_cols = [col for col in links.columns if col != "value"]
        if "source" not in links.columns and len(meta_cols) >= 1:
            links = links.rename(columns={meta_cols[0]: "source"})
        if "target" not in links.columns:
            meta_cols = [col for col in links.columns if col not in {"value", "source"}]
            if len(meta_cols) >= 1:
                links = links.rename(columns={meta_cols[0]: "target"})

    ordered = [col for col in ("source", "target", "value") if col in links.columns]
    tail = [col for col in links.columns if col not in ordered]
    return links[ordered + tail].copy()


def _prepare_weights(gdf: gpd.GeoDataFrame, ids: list[str]):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The weights matrix is not fully connected*")
        weights = Queen.from_dataframe(gdf, ids=ids, use_index=False, silence_warnings=True)
    islands = list(getattr(weights, "islands", []))
    if islands:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The weights matrix is not fully connected*")
            knn_weights = KNN.from_dataframe(gdf, ids=ids, use_index=False, k=1)
            weights = lw.w_union(weights, knn_weights)
    weights.transform = "r"
    return weights


def _heatmap(
    value_table: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    significance_table: pd.DataFrame | None = None,
    fmt: str = ".2f",
    value_vmin: float | None = None,
    value_vmax: float | None = None,
) -> None:
    if value_table.empty:
        return
    values = value_table.apply(pd.to_numeric, errors="coerce")
    if values.isna().all().all():
        return
    pvals = None
    if significance_table is not None and not significance_table.empty:
        pvals = significance_table.reindex(index=values.index, columns=values.columns).apply(pd.to_numeric, errors="coerce")

    finite_values = values.to_numpy(dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return
    inferred_vmin = float(np.min(finite_values))
    inferred_vmax = float(np.max(finite_values))
    if value_vmin is None or value_vmax is None:
        if inferred_vmin >= 0.0 and inferred_vmax <= 1.0:
            inferred_vmin, inferred_vmax = 0.0, 1.0
        elif inferred_vmin >= -1.0 and inferred_vmax <= 1.0:
            inferred_vmin, inferred_vmax = -1.0, 1.0
    vmin = float(value_vmin) if value_vmin is not None else inferred_vmin
    vmax = float(value_vmax) if value_vmax is not None else inferred_vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = inferred_vmin, inferred_vmax
    cmap = plt.get_cmap("RdYlBu_r") if vmin < 0.0 else plt.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * values.shape[1]), max(4, 0.7 * values.shape[0] + 2)))
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(0, values.shape[0])
    ax.invert_yaxis()
    ax.set_aspect("auto")
    ax.set_facecolor("#f8fafc")

    def _cell_style(value: float | None, p_value: float | None) -> tuple[str, str]:
        if value is None or pd.isna(value):
            return "#f3f4f6", "#6b7280"
        if p_value is None or pd.isna(p_value) or float(p_value) >= 0.05:
            return "#e5e7eb", "#111827"
        if abs(float(value)) < EFFECT_HIGHLIGHT_MIN_ABS:
            return "#e5e7eb", "#111827"
        rgba = cmap(norm(float(value)))
        facecolor = mcolors.to_hex(rgba, keep_alpha=False)
        luminance = (0.299 * rgba[0]) + (0.587 * rgba[1]) + (0.114 * rgba[2])
        textcolor = "#111827" if luminance >= 0.53 else "#ffffff"
        return facecolor, textcolor

    def _sig_marker(p_value: float | None, value: float | None) -> str:
        if p_value is None or pd.isna(p_value):
            return ""
        if value is None or pd.isna(value):
            return ""
        if abs(float(value)) < EFFECT_HIGHLIGHT_MIN_ABS:
            return ""
        if float(p_value) < 0.001:
            return "***"
        if float(p_value) < 0.01:
            return "**"
        if float(p_value) < 0.05:
            return "*"
        return ""

    for row_idx, row_name in enumerate(values.index):
        for col_idx, col_name in enumerate(values.columns):
            value = values.loc[row_name, col_name]
            p_value = pvals.loc[row_name, col_name] if pvals is not None else np.nan
            facecolor, textcolor = _cell_style(value, p_value)
            ax.add_patch(
                Rectangle(
                    (col_idx, row_idx),
                    1.0,
                    1.0,
                    facecolor=facecolor,
                    edgecolor="#f8fafc",
                    linewidth=1.0,
                )
            )
            if pd.notna(value):
                label = format(float(value), fmt)
                marker = _sig_marker(p_value, value)
                if marker:
                    label = f"{label}\n{marker}"
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=textcolor,
                    fontweight="bold" if marker in {"*", "**", "***"} else None,
                )

    ax.set_xticks(np.arange(values.shape[1]) + 0.5)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5)
    ax.set_xticklabels(values.columns, rotation=20, ha="right")
    ax.set_yticklabels(values.index, rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    footer_text(
        fig,
        [
            f"cell color = effect size on fixed scale [{vmin:.2f}, {vmax:.2f}]",
            "only significant cells (p < 0.05) are color-emphasized",
            f"cells with |effect| < {EFFECT_HIGHLIGHT_MIN_ABS:.1f} are neutral (no color emphasis)",
            f"significance stars shown only when p < 0.05 and |effect| >= {EFFECT_HIGHLIGHT_MIN_ABS:.1f}",
        ],
        y=0.01,
        fontsize=8,
    )
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _global_moran_heatmap(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    if frame.empty:
        return
    table = frame[["response", "moran_I", "p_sim"]].copy()
    if table.empty:
        return
    value_table = table.set_index("response")[["moran_I"]]
    significance_table = table.set_index("response")[["p_sim"]]
    _heatmap(
        value_table,
        output_path,
        title=title,
        significance_table=significance_table,
        fmt=".3f",
    )


def _plot_dominant_class_map(blocks: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, output_path: Path) -> None:
    boundary_norm = normalize_preview_gdf(boundary)
    blocks_norm = normalize_preview_gdf(blocks, boundary_norm)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    apply_preview_canvas(fig, ax, boundary_norm, title="Street Pattern Dominant Class")
    handles: list[Patch] = []
    present_classes = [c for c in list(CLASS_COLORS.keys()) if c != "unknown" and c in set(blocks_norm["street_pattern_dominant_class"])]
    for class_name in present_classes:
        part = blocks_norm[blocks_norm["street_pattern_dominant_class"] == class_name]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            color=CLASS_COLORS[class_name],
            edgecolor="#ffffff",
            linewidth=0.12,
            alpha=0.95,
            zorder=5,
        )
        handles.append(Patch(facecolor=CLASS_COLORS[class_name], edgecolor="none", label=class_name))
    unknown = blocks_norm[blocks_norm["street_pattern_dominant_class"] == "unknown"]
    if not unknown.empty:
        unknown.plot(ax=ax, color=CLASS_COLORS["unknown"], edgecolor="#ffffff", linewidth=0.1, alpha=0.9, zorder=4)
        handles.append(Patch(facecolor=CLASS_COLORS["unknown"], edgecolor="none", label="unknown"))
    legend_bottom(ax, handles, max_cols=3, fontsize=8)
    footer_text(fig, ["Dominant class after grid-to-block transfer."], y=0.015)
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_accessibility_map(blocks: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, output_path: Path) -> None:
    valid = blocks[blocks["accessibility_time_mean_pt"].notna()].copy()
    if valid.empty:
        return
    boundary_norm = normalize_preview_gdf(boundary)
    valid_norm = normalize_preview_gdf(valid, boundary_norm)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    apply_preview_canvas(fig, ax, boundary_norm, title="Accessibility Mean Time")
    valid_norm.plot(
        ax=ax,
        column="accessibility_time_mean_pt",
        cmap="viridis_r",
        legend=True,
        linewidth=0.12,
        edgecolor="#ffffff",
        alpha=0.97,
        zorder=5,
        legend_kwds={"label": "mean travel time, min", "shrink": 0.75},
    )
    excluded = blocks[blocks["accessibility_time_mean_pt"].isna()].copy()
    if not excluded.empty:
        excluded_norm = normalize_preview_gdf(excluded, boundary_norm)
        excluded_norm.plot(ax=ax, color="#d1d5db", edgecolor="#ffffff", linewidth=0.08, alpha=0.65, zorder=4)
    footer_text(fig, ["Accessibility is computed only for blocks with population > 0."], y=0.015)
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_grouped_metric_bars(
    grouped: pd.DataFrame,
    output_path: Path,
    *,
    value_kind: str,
    title: str,
    ylabel: str,
) -> None:
    if grouped.empty:
        return
    melt_cols = [c for c in grouped.columns if c.startswith(value_kind)]
    if not melt_cols:
        return
    data = grouped.reset_index().melt(
        id_vars="street_pattern_dominant_class",
        value_vars=melt_cols,
        var_name="metric",
        value_name="value",
    )
    data["metric"] = data["metric"].str.replace(f"{value_kind}_", "", regex=False)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=data, x="street_pattern_dominant_class", y="value", hue="metric", ax=ax)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_accessibility_by_class(access_df: pd.DataFrame, output_path: Path) -> None:
    if access_df.empty:
        return
    plot_df = access_df.copy()
    plot_df["street_pattern_dominant_class"] = plot_df["street_pattern_dominant_class"].astype(str)
    plot_df["accessibility_time_mean_pt"] = pd.to_numeric(plot_df["accessibility_time_mean_pt"], errors="coerce")
    plot_df["accessibility_time_mean_pt"] = plot_df["accessibility_time_mean_pt"].replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna(subset=["accessibility_time_mean_pt"]).copy()
    if plot_df.empty:
        return

    class_order = [
        class_name
        for class_name in CLASS_LABELS.values()
        if class_name in set(plot_df["street_pattern_dominant_class"])
    ]
    if not class_order:
        class_order = sorted(plot_df["street_pattern_dominant_class"].unique().tolist())
    grouped_pairs = []
    for class_name in class_order:
        values = (
            plot_df.loc[plot_df["street_pattern_dominant_class"] == class_name, "accessibility_time_mean_pt"]
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values) > 0:
            grouped_pairs.append((class_name, values))
    class_order = [label for label, _ in grouped_pairs]
    grouped_values = [values for _, values in grouped_pairs]
    if not grouped_values:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(
        grouped_values,
        tick_labels=class_order,
        patch_artist=True,
        showfliers=True,
        flierprops={"markersize": 2, "markerfacecolor": "#60a5fa", "markeredgecolor": "#60a5fa"},
    )
    for patch in ax.artists:
        patch.set_facecolor("#60a5fa")
        patch.set_alpha(0.7)
        patch.set_edgecolor("#1e3a8a")
    for median in ax.lines[4::6]:
        median.set_color("#0f172a")
        median.set_linewidth(1.4)
    ax.set_title("Accessibility By Dominant Street Pattern Class", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("mean travel time, min")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_local_bivariate_moran_map(
    gdf: gpd.GeoDataFrame,
    *,
    response_col: str,
    feature_col: str,
    boundary: gpd.GeoDataFrame,
    permutations: int,
    output_path: Path,
) -> None:
    subset = gdf[["block_id", response_col, feature_col, "geometry"]].copy()
    subset[response_col] = pd.to_numeric(subset[response_col], errors="coerce")
    subset[feature_col] = pd.to_numeric(subset[feature_col], errors="coerce")
    subset = subset.dropna(subset=[response_col, feature_col]).copy()
    subset = subset[subset.geometry.notna() & ~subset.geometry.is_empty].copy()
    subset = subset[subset.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if len(subset) < 4 or subset[response_col].nunique() < 2 or subset[feature_col].nunique() < 2:
        return

    weights = _prepare_weights(subset, subset["block_id"].astype(str).tolist())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The weights matrix is not fully connected*")
        lisa = Moran_Local_BV(
            subset[response_col].to_numpy(dtype=float),
            subset[feature_col].to_numpy(dtype=float),
            weights,
            permutations=permutations,
        )

    cluster_labels = {
        1: "high-high",
        2: "low-high",
        3: "low-low",
        4: "high-low",
    }
    cluster_colors = {
        "high-high": "#b91c1c",
        "low-high": "#f59e0b",
        "low-low": "#2563eb",
        "high-low": "#7c3aed",
        "not_significant": "#d1d5db",
    }

    plot_gdf = subset.copy()
    significant = pd.Series(lisa.p_sim, index=plot_gdf.index) < 0.05
    quadrants = pd.Series(lisa.q, index=plot_gdf.index).map(cluster_labels)
    plot_gdf["moran_cluster"] = np.where(significant, quadrants.fillna("not_significant"), "not_significant")

    boundary_norm = normalize_preview_gdf(boundary)
    plot_norm = normalize_preview_gdf(plot_gdf, boundary_norm)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    pretty_feature = _friendly_feature_name(feature_col)
    title = f"Local Bivariate Moran: {response_col} vs {pretty_feature}"
    apply_preview_canvas(fig, ax, boundary_norm, title=title)

    handles: list[Patch] = []
    for label in ("high-high", "high-low", "low-high", "low-low", "not_significant"):
        part = plot_norm[plot_norm["moran_cluster"] == label]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            color=cluster_colors[label],
            edgecolor="#ffffff",
            linewidth=0.12,
            alpha=0.95 if label != "not_significant" else 0.7,
            zorder=5 if label != "not_significant" else 4,
        )
        handles.append(Patch(facecolor=cluster_colors[label], edgecolor="none", label=label.replace("_", " ")))

    legend_bottom(ax, handles, max_cols=3, fontsize=8)
    footer_text(
        fig,
        [
            "local bivariate Moran clusters at p < 0.05",
            f"global relation shown here is: {response_col} vs W * {pretty_feature}",
            "W * feature = spatial lag of the street-pattern feature in neighboring blocks",
        ],
        y=0.015,
    )
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_local_lisa_map(
    gdf: gpd.GeoDataFrame,
    *,
    response_col: str,
    boundary: gpd.GeoDataFrame,
    permutations: int,
    output_path: Path,
) -> None:
    subset = gdf[["block_id", response_col, "geometry"]].copy()
    subset[response_col] = pd.to_numeric(subset[response_col], errors="coerce")
    subset = subset.dropna(subset=[response_col]).copy()
    subset = subset[subset.geometry.notna() & ~subset.geometry.is_empty].copy()
    subset = subset[subset.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if len(subset) < 4 or subset[response_col].nunique() < 2:
        return

    weights = _prepare_weights(subset, subset["block_id"].astype(str).tolist())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The weights matrix is not fully connected*")
        lisa = Moran_Local(
            subset[response_col].to_numpy(dtype=float),
            weights,
            permutations=permutations,
        )

    cluster_labels = {
        1: "high-high",
        2: "low-high",
        3: "low-low",
        4: "high-low",
    }
    cluster_colors = {
        "high-high": "#b91c1c",
        "low-high": "#f59e0b",
        "low-low": "#2563eb",
        "high-low": "#7c3aed",
        "not_significant": "#d1d5db",
    }

    plot_gdf = subset.copy()
    significant = pd.Series(lisa.p_sim, index=plot_gdf.index) < 0.05
    quadrants = pd.Series(lisa.q, index=plot_gdf.index).map(cluster_labels)
    plot_gdf["lisa_cluster"] = np.where(significant, quadrants.fillna("not_significant"), "not_significant")

    boundary_norm = normalize_preview_gdf(boundary)
    plot_norm = normalize_preview_gdf(plot_gdf, boundary_norm)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    apply_preview_canvas(fig, ax, boundary_norm, title=f"Local Moran LISA: {response_col}")

    handles: list[Patch] = []
    for label in ("high-high", "high-low", "low-high", "low-low", "not_significant"):
        part = plot_norm[plot_norm["lisa_cluster"] == label]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            color=cluster_colors[label],
            edgecolor="#ffffff",
            linewidth=0.12,
            alpha=0.95 if label != "not_significant" else 0.7,
            zorder=5 if label != "not_significant" else 4,
        )
        handles.append(Patch(facecolor=cluster_colors[label], edgecolor="none", label=label.replace("_", " ")))

    legend_bottom(ax, handles, max_cols=3, fontsize=8)
    footer_text(
        fig,
        [
            "local univariate Moran (LISA) clusters at p < 0.05",
            f"response={response_col}",
        ],
        y=0.015,
    )
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _response_columns(services: list[str], stop_modalities: list[str]) -> tuple[list[str], list[str], list[str]]:
    service_response_cols = []
    for service in services:
        service_response_cols.extend([f"service_has_{service}", f"service_capacity_{service}"])
    stop_response_cols = []
    for modality in stop_modalities:
        stop_response_cols.extend([f"stop_has_{modality}", f"stop_count_{modality}"])
    accessibility_cols = [
        "accessibility_time_mean_pt_intermodal",
        "accessibility_time_mean_pt_walk",
    ]
    for service in services:
        accessibility_cols.extend(
            [
                f"assigned_service_time_{service}_intermodal",
                f"assigned_service_time_{service}_walk",
            ]
        )
    return service_response_cols, stop_response_cols, accessibility_cols


def _sanitize_numeric_inf_inplace(frame: pd.DataFrame) -> None:
    for col in frame.columns:
        if col == "geometry":
            continue
        series = frame[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            frame[col] = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _spearman_table(frame: pd.DataFrame, response_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for response_col in response_cols:
        for feature_col in feature_cols:
            subset = (
                frame[[response_col, feature_col]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(subset) < 3 or subset[response_col].nunique() < 2 or subset[feature_col].nunique() < 2:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(subset[response_col], subset[feature_col], nan_policy="omit")
            rows.append(
                {
                    "response": response_col,
                    "street_pattern_feature": feature_col,
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "n": int(len(subset)),
                }
            )
    return pd.DataFrame(rows)


def _moran_tables(
    gdf: gpd.GeoDataFrame,
    *,
    response_cols: list[str],
    feature_cols: list[str],
    permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    univariate_rows: list[dict[str, object]] = []
    bivariate_rows: list[dict[str, object]] = []
    base = gdf.copy()
    base = base[base.geometry.notna() & ~base.geometry.is_empty].copy()
    base = base[base.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    base["block_id"] = base["block_id"].astype(str)

    for response_col in response_cols:
        subset = base[["block_id", response_col, "geometry"]].copy()
        subset[response_col] = pd.to_numeric(subset[response_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna(subset=[response_col]).copy()
        if len(subset) < 4 or subset[response_col].nunique() < 2:
            continue
        weights = _prepare_weights(subset, subset["block_id"].tolist())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The weights matrix is not fully connected*")
            moran = Moran(subset[response_col].to_numpy(dtype=float), weights, permutations=permutations)
        univariate_rows.append(
            {
                "response": response_col,
                "moran_I": float(moran.I),
                "p_sim": float(moran.p_sim),
                "z_norm": float(moran.z_norm),
                "n": int(len(subset)),
                "n_components": int(getattr(weights, "n_components", 1)),
            }
        )

    for response_col in response_cols:
        for feature_col in feature_cols:
            subset = base[["block_id", response_col, feature_col, "geometry"]].copy()
            subset[response_col] = pd.to_numeric(subset[response_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            subset[feature_col] = pd.to_numeric(subset[feature_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            subset = subset.dropna(subset=[response_col, feature_col]).copy()
            if len(subset) < 4 or subset[response_col].nunique() < 2 or subset[feature_col].nunique() < 2:
                continue
            weights = _prepare_weights(subset, subset["block_id"].tolist())
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The weights matrix is not fully connected*")
                moran_bv = Moran_BV(
                    subset[response_col].to_numpy(dtype=float),
                    subset[feature_col].to_numpy(dtype=float),
                    weights,
                    permutations=permutations,
                )
            bivariate_rows.append(
                {
                    "response": response_col,
                    "street_pattern_feature": feature_col,
                    "moran_bv_I": float(moran_bv.I),
                    "p_sim": float(moran_bv.p_sim),
                    "z_sim": float(moran_bv.z_sim),
                    "n": int(len(subset)),
                    "n_components": int(getattr(weights, "n_components", 1)),
                }
            )
    return pd.DataFrame(univariate_rows), pd.DataFrame(bivariate_rows)


def _friendly_feature_name(col: str) -> str:
    for prob_col, label in CLASS_LABELS.items():
        slug = f"street_pattern_prob_{label.lower().replace(' & ', '_').replace(' ', '_').replace('-', '_').replace('__', '_')}"
        if col == slug:
            return label
    return col


def _class_feature_slug(label: str) -> str:
    return f"street_pattern_prob_{label.lower().replace(' & ', '_').replace(' ', '_').replace('-', '_').replace('__', '_')}"


def _group_summary(frame: pd.DataFrame, services: list[str], stop_modalities: list[str]) -> pd.DataFrame:
    value_cols = ["accessibility_time_mean_pt"]
    for service in services:
        value_cols.extend([f"service_has_{service}", f"service_capacity_{service}", f"service_count_{service}"])
    for modality in stop_modalities:
        value_cols.extend([f"stop_has_{modality}", f"stop_count_{modality}"])
    grouped = (
        frame.groupby("street_pattern_dominant_class")[value_cols]
        .mean(numeric_only=True)
        .sort_index()
    )
    counts = frame.groupby("street_pattern_dominant_class").size().rename("block_count")
    grouped = grouped.join(counts, how="left")
    grouped.index.name = "street_pattern_dominant_class"
    return grouped.reset_index()


def _cross_city_response_columns(
    services: list[str],
    stop_modalities: list[str],
    *,
    accessibility_only: bool = False,
) -> list[str]:
    response_cols = ["accessibility_time_mean_pt_intermodal", "accessibility_time_mean_pt_walk"]
    for service in services:
        response_cols.append(f"assigned_service_time_{service}_intermodal")
        response_cols.append(f"assigned_service_time_{service}_walk")
    if not accessibility_only:
        for service in services:
            response_cols.append(f"service_has_{service}")
        for modality in stop_modalities:
            response_cols.append(f"stop_has_{modality}")
    return response_cols


def _prepare_cross_city_frame(city_frames: dict[str, gpd.GeoDataFrame], services: list[str]) -> tuple[gpd.GeoDataFrame, list[str], list[str]]:
    common_crs = "EPSG:3857"
    feature_cols = [
        f"street_pattern_prob_{label.lower().replace(' & ', '_').replace(' ', '_').replace('-', '_').replace('__', '_')}"
        for label in CLASS_LABELS.values()
    ]
    all_stop_modalities = sorted(
        {
            col.removeprefix("stop_has_")
            for frame in city_frames.values()
            for col in frame.columns
            if col.startswith("stop_has_")
        }
    )
    response_cols = _cross_city_response_columns(services, all_stop_modalities)

    prepared_frames: list[gpd.GeoDataFrame] = []
    expected_cols = list(dict.fromkeys(["city", "block_id", "geometry", "population", "street_pattern_dominant_class", *feature_cols, *response_cols]))
    for city, frame in city_frames.items():
        prepared = frame.copy().to_crs(common_crs)
        prepared["city"] = city
        prepared["block_id"] = prepared["block_id"].astype(str)
        for col in feature_cols:
            if col not in prepared.columns:
                prepared[col] = 0.0
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0.0)
        for col in response_cols:
            if col not in prepared.columns:
                prepared[col] = np.nan if (col.startswith("accessibility_time_mean_pt") or col.startswith("assigned_service_time_")) else 0.0
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        prepared["population"] = pd.to_numeric(prepared.get("population"), errors="coerce").fillna(0.0)
        prepared["street_pattern_dominant_class"] = prepared.get("street_pattern_dominant_class", "unknown").fillna("unknown").astype(str)
        prepared_frames.append(prepared[expected_cols].copy())

    combined = gpd.GeoDataFrame(pd.concat(prepared_frames, ignore_index=True), geometry="geometry", crs=common_crs)
    combined["block_area"] = combined.geometry.area.astype(float)
    combined["log_block_area"] = np.log1p(np.clip(combined["block_area"], a_min=0.0, a_max=None))
    combined["log_population"] = np.log1p(np.clip(combined["population"], a_min=0.0, a_max=None))
    return combined, feature_cols, all_stop_modalities


def _distribution_tests(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    cities = sorted(frame["city"].dropna().unique().tolist())
    rows: list[dict[str, object]] = []
    for i, city_a in enumerate(cities):
        for city_b in cities[i + 1 :]:
            subset_a = frame[frame["city"] == city_a]
            subset_b = frame[frame["city"] == city_b]
            for feature_col in feature_cols:
                values_a = pd.to_numeric(subset_a[feature_col], errors="coerce").dropna()
                values_b = pd.to_numeric(subset_b[feature_col], errors="coerce").dropna()
                if len(values_a) < 3 or len(values_b) < 3:
                    continue
                ks_stat, ks_p = ks_2samp(values_a, values_b)
                rows.append(
                    {
                        "city_a": city_a,
                        "city_b": city_b,
                        "street_pattern_feature": feature_col,
                        "n_a": int(len(values_a)),
                        "n_b": int(len(values_b)),
                        "mean_a": float(values_a.mean()),
                        "mean_b": float(values_b.mean()),
                        "median_a": float(values_a.median()),
                        "median_b": float(values_b.median()),
                        "ks_stat": float(ks_stat),
                        "ks_p_value": float(ks_p),
                        "wasserstein_distance": float(wasserstein_distance(values_a, values_b)),
                    }
                )
    return pd.DataFrame(rows)


def _dominant_class_composition(frame: pd.DataFrame) -> pd.DataFrame:
    counts = (
        frame.groupby(["city", "street_pattern_dominant_class"])
        .size()
        .rename("block_count")
        .reset_index()
    )
    totals = counts.groupby("city")["block_count"].transform("sum")
    counts["block_share"] = np.where(totals > 0, counts["block_count"] / totals, np.nan)
    return counts


def _variance_decomposition(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature_col in feature_cols:
        subset = frame[["city", feature_col]].copy()
        subset[feature_col] = pd.to_numeric(subset[feature_col], errors="coerce")
        subset = subset.dropna(subset=[feature_col]).copy()
        if len(subset) < 3:
            continue
        overall_var = float(subset[feature_col].var(ddof=1)) if len(subset) > 1 else np.nan
        city_stats = subset.groupby("city")[feature_col].agg(["mean", "var", "count"]).reset_index()
        if city_stats.empty:
            continue
        grand_mean = float(subset[feature_col].mean())
        weights = city_stats["count"].astype(float)
        between_var = float(np.average((city_stats["mean"] - grand_mean) ** 2, weights=weights)) if weights.sum() > 0 else np.nan
        within_var = float(np.average(city_stats["var"].fillna(0.0), weights=weights)) if weights.sum() > 0 else np.nan
        total_for_share = between_var + within_var if pd.notna(between_var) and pd.notna(within_var) else np.nan
        rows.append(
            {
                "street_pattern_feature": feature_col,
                "overall_mean": grand_mean,
                "overall_var": overall_var,
                "between_city_var": between_var,
                "within_city_var": within_var,
                "between_city_share": float(between_var / total_for_share) if total_for_share and total_for_share > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _response_family(frame: pd.DataFrame, response_col: str) -> str:
    values = pd.to_numeric(frame[response_col], errors="coerce").dropna()
    if not values.empty and set(values.unique()).issubset({0.0, 1.0}):
        return "binomial"
    return "gaussian"


def _fit_formula_model(frame: pd.DataFrame, response_col: str, rhs: str):
    family = _response_family(frame, response_col)
    formula = f"{response_col} ~ {rhs}"
    if family == "binomial":
        return smf.glm(formula=formula, data=frame, family=sm.families.Binomial()).fit()
    return smf.ols(formula=formula, data=frame).fit()


def _model_score(result, family: str) -> float:
    if family == "binomial":
        null_deviance = getattr(result, "null_deviance", np.nan)
        deviance = getattr(result, "deviance", np.nan)
        if pd.notna(null_deviance) and float(null_deviance) > 0:
            return float(1.0 - float(deviance) / float(null_deviance))
        return np.nan
    return float(getattr(result, "rsquared", np.nan))


def _fit_cross_city_models(frame: pd.DataFrame, response_cols: list[str], feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    if len(feature_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    pattern_terms = " + ".join(feature_cols[:-1])
    control_terms = "log_population + log_block_area"
    rhs_city = "C(city)"
    rhs_pattern = f"{pattern_terms} + {control_terms}"
    rhs_full = f"C(city) + {pattern_terms} + {control_terms}"

    for response_col in response_cols:
        required_cols = [response_col, "city", "log_population", "log_block_area", *feature_cols]
        subset = frame[required_cols].copy()
        for col in required_cols:
            if col != "city":
                subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset.dropna(subset=[response_col, "city", "log_population", "log_block_area", *feature_cols]).copy()
        if len(subset) < 12 or subset["city"].nunique() < 2 or subset[response_col].nunique() < 2:
            continue
        family = _response_family(subset, response_col)
        try:
            model_city = _fit_formula_model(subset, response_col, rhs_city)
            model_pattern = _fit_formula_model(subset, response_col, rhs_pattern)
            model_full = _fit_formula_model(subset, response_col, rhs_full)
        except Exception:
            continue

        score_city = _model_score(model_city, family)
        score_pattern = _model_score(model_pattern, family)
        score_full = _model_score(model_full, family)
        city_after_pattern = float(score_full - score_pattern) if pd.notna(score_full) and pd.notna(score_pattern) else np.nan
        pattern_after_city = float(score_full - score_city) if pd.notna(score_full) and pd.notna(score_city) else np.nan
        city_effect_reduction = np.nan
        if pd.notna(score_city) and float(score_city) >= 0.01 and pd.notna(city_after_pattern):
            city_effect_reduction = float(1.0 - city_after_pattern / float(score_city))
        model_rows.append(
            {
                "response": response_col,
                "family": family,
                "n": int(len(subset)),
                "city_only_score": score_city,
                "pattern_only_score": score_pattern,
                "city_plus_pattern_score": score_full,
                "city_after_pattern_increment": city_after_pattern,
                "pattern_after_city_increment": pattern_after_city,
                "city_effect_reduction_pct": city_effect_reduction,
            }
        )
        for model_name, result in (("city_only", model_city), ("pattern_only", model_pattern), ("city_plus_pattern", model_full)):
            params = result.params
            pvalues = result.pvalues
            bse = result.bse
            for term, estimate in params.items():
                coef_rows.append(
                    {
                        "response": response_col,
                        "family": family,
                        "model": model_name,
                        "term": str(term),
                        "coef": float(estimate) if pd.notna(estimate) else np.nan,
                        "std_err": float(bse.get(term, np.nan)) if hasattr(bse, "get") else np.nan,
                        "p_value": float(pvalues.get(term, np.nan)) if hasattr(pvalues, "get") else np.nan,
                        "n": int(len(subset)),
                    }
                )

    return pd.DataFrame(model_rows), pd.DataFrame(coef_rows)


def _fit_between_within_models(frame: pd.DataFrame, response_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame["city"].nunique() < 2:
        return pd.DataFrame()

    for feature_col in feature_cols:
        feature_frame = frame[["city", response_cols[0] if response_cols else feature_col]].copy()  # placeholder to keep loop structure simple
        _ = feature_frame  # silence linters if any
        city_mean_col = f"{feature_col}_city_mean"
        within_col = f"{feature_col}_within"
        base = frame.copy()
        base[city_mean_col] = base.groupby("city")[feature_col].transform("mean")
        base[within_col] = base[feature_col] - base[city_mean_col]
        for response_col in response_cols:
            required_cols = [response_col, city_mean_col, within_col, "log_population", "log_block_area"]
            subset = base[required_cols].copy()
            for col in required_cols:
                subset[col] = pd.to_numeric(subset[col], errors="coerce")
            subset["city"] = base["city"].values
            subset = subset.dropna(subset=required_cols).copy()
            if len(subset) < 12 or subset[response_col].nunique() < 2:
                continue
            family = _response_family(subset, response_col)
            formula = f"{response_col} ~ {city_mean_col} + {within_col} + log_population + log_block_area"
            try:
                if family == "binomial":
                    result = smf.glm(formula=formula, data=subset, family=sm.families.Binomial()).fit()
                else:
                    result = smf.ols(formula=formula, data=subset).fit()
            except Exception:
                continue
            rows.append(
                {
                    "response": response_col,
                    "street_pattern_feature": feature_col,
                    "family": family,
                    "n": int(len(subset)),
                    "between_coef": float(result.params.get(city_mean_col, np.nan)),
                    "between_p_value": float(result.pvalues.get(city_mean_col, np.nan)),
                    "within_coef": float(result.params.get(within_col, np.nan)),
                    "within_p_value": float(result.pvalues.get(within_col, np.nan)),
                    "model_score": _model_score(result, family),
                }
            )
    return pd.DataFrame(rows)


def _plot_feature_distributions(frame: pd.DataFrame, feature_cols: list[str], output_path: Path) -> None:
    if frame.empty or not feature_cols:
        return
    plot_df = frame[["city", *feature_cols]].melt(
        id_vars="city",
        value_vars=feature_cols,
        var_name="street_pattern_feature",
        value_name="value",
    )
    summary = (
        plot_df.groupby(["street_pattern_feature", "city"])["value"]
        .agg(
            q25=lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.25)),
            median=lambda s: float(pd.to_numeric(s, errors="coerce").median()),
            q75=lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.75)),
        )
        .reset_index()
    )
    if summary.empty:
        return
    summary["street_pattern_feature"] = summary["street_pattern_feature"].map(_friendly_feature_name)

    n_panels = len(feature_cols)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.5, max(5.8, 3.2 * nrows)), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, feature_name in zip(axes, [_friendly_feature_name(col) for col in feature_cols]):
        feature_summary = summary[summary["street_pattern_feature"] == feature_name].copy()
        if feature_summary.empty:
            ax.axis("off")
            continue
        feature_summary = feature_summary.sort_values("median", ascending=False).reset_index(drop=True)
        y_positions = np.arange(len(feature_summary), dtype=float)
        for y, q25, q75 in zip(y_positions, feature_summary["q25"], feature_summary["q75"]):
            ax.plot([q25, q75], [y, y], color="#94a3b8", linewidth=2.2, solid_capstyle="round", zorder=1)
        ax.scatter(
            feature_summary["median"],
            y_positions,
            s=34,
            color="#1d4ed8",
            edgecolor="#ffffff",
            linewidth=0.7,
            zorder=3,
        )
        ax.set_title(feature_name, fontsize=12, fontweight="bold")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(feature_summary["city"])
        ax.set_xlim(0, 1)
        ax.grid(True, axis="x", alpha=0.18, linewidth=0.6)
        ax.set_xlabel("share")
        ax.set_ylabel("")

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("Street Pattern Distributions By City", fontsize=16, fontweight="bold", y=0.995)
    footer_text(
        fig,
        [
            "each panel shows one street-pattern feature",
            "dot = city median, segment = interquartile range",
        ],
        y=0.012,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_dominant_class_composition(composition: pd.DataFrame, output_path: Path) -> None:
    if composition.empty:
        return
    pivot = composition.pivot(index="city", columns="street_pattern_dominant_class", values="block_share").fillna(0.0)
    if "Irregular Grid" in pivot.columns:
        pivot = pivot.sort_values("Irregular Grid", ascending=False)
    color_map = {class_name: CLASS_COLORS.get(class_name, "#9ca3af") for class_name in pivot.columns}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(pivot), dtype=float)
    for class_name in pivot.columns:
        values = pivot[class_name].to_numpy(dtype=float)
        ax.bar(pivot.index, values, bottom=bottom, color=color_map[class_name], label=class_name)
        bottom = bottom + values
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of blocks")
    ax.set_title("Dominant Street Pattern Composition By City", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", rotation=CROSS_CITY_XLABEL_ROTATION, labelsize=CROSS_CITY_XLABEL_FONTSIZE)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_variance_decomposition(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    plot_df = frame.copy()
    plot_df["street_pattern_feature"] = plot_df["street_pattern_feature"].map(_friendly_feature_name)
    plot_df = plot_df.sort_values("between_city_share", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.7 * len(plot_df) + 1)))
    sns.barplot(data=plot_df, x="between_city_share", y="street_pattern_feature", color="#2563eb", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("between-city share of variance")
    ax.set_ylabel("")
    ax.set_title("Street Pattern Variance Decomposition", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_model_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    plot_df = frame[["response", "city_only_score", "pattern_only_score", "city_plus_pattern_score"]].melt(
        id_vars="response",
        var_name="model",
        value_name="score",
    )
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.6 * plot_df["response"].nunique() + 1)))
    sns.barplot(data=plot_df, x="score", y="response", hue="model", orient="h", ax=ax)
    ax.set_title("City vs Street Pattern Model Fit", fontsize=16, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("fit score (R² / deviance-based pseudo-R²)")
    ax.set_ylabel("")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_effect_reduction(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    plot_df = frame.dropna(subset=["city_effect_reduction_pct"]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.6 * len(plot_df) + 1)))
    sns.barplot(data=plot_df.sort_values("city_effect_reduction_pct", ascending=False), x="city_effect_reduction_pct", y="response", color="#ea580c", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of city effect reduced after adding street pattern")
    ax.set_ylabel("")
    ax.set_title("City Effect Reduction After Street Pattern Controls", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _accessibility_city_profiles(
    frame: pd.DataFrame,
    feature_cols: list[str],
    accessibility_response_cols: list[str],
    city_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    subset = frame.copy()
    tracked_access_cols = [
        col
        for col in accessibility_response_cols
        if col in subset.columns
    ]
    for col in tracked_access_cols:
        subset[col] = pd.to_numeric(subset[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    subset = subset[
        subset[tracked_access_cols].notna().any(axis=1)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    grouped = (
        subset.groupby("city")[list(dict.fromkeys([*tracked_access_cols, *feature_cols]))]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = subset.groupby("city").size().rename("accessibility_block_count").reset_index()
    base_median_cols = [
        col
        for col in ["accessibility_time_mean_pt_intermodal", "accessibility_time_mean_pt_walk"]
        if col in subset.columns
    ]
    assigned_median_cols = sorted(
        [
            col
            for col in tracked_access_cols
            if col.startswith("assigned_service_time_")
        ]
    )
    median_cols = list(dict.fromkeys([*base_median_cols, *assigned_median_cols]))
    medians = (
        subset.groupby("city")[median_cols]
        .median(numeric_only=True)
        .rename(columns={col: f"accessibility_median_{col}" for col in median_cols})
        .reset_index()
    )
    result = grouped.merge(counts, on="city", how="left").merge(medians, on="city", how="left")
    if city_context is not None and not city_context.empty:
        result = result.merge(city_context, on="city", how="left")
    if {
        "accessibility_median_accessibility_time_mean_pt_intermodal",
        "accessibility_median_accessibility_time_mean_pt_walk",
    }.issubset(result.columns):
        result["accessibility_median_delta_walk_minus_intermodal"] = (
            pd.to_numeric(result["accessibility_median_accessibility_time_mean_pt_walk"], errors="coerce")
            - pd.to_numeric(result["accessibility_median_accessibility_time_mean_pt_intermodal"], errors="coerce")
        )
    return result


def _plot_accessibility_city_profiles(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    class_composition: pd.DataFrame | None = None,
    distribution_frame: pd.DataFrame | None = None,
) -> None:
    if frame.empty:
        return
    required = [
        "accessibility_median_accessibility_time_mean_pt_intermodal",
        "accessibility_median_accessibility_time_mean_pt_walk",
    ]
    if not all(col in frame.columns for col in required):
        return
    canvas_gray = "#7a7a7a"
    plot_df = frame.copy()
    if "bus_route_count" in plot_df.columns:
        route_counts = pd.to_numeric(plot_df["bus_route_count"], errors="coerce")
        plot_df = plot_df[route_counts >= CROSS_CITY_MIN_BUS_ROUTES].copy()
    if "population_total" in plot_df.columns:
        population_total = pd.to_numeric(plot_df["population_total"], errors="coerce")
        plot_df = plot_df[population_total >= CROSS_CITY_MIN_POPULATION].copy()
    if plot_df.empty:
        return
    sorted_by_irregular = False
    if class_composition is not None and not class_composition.empty:
        comp = class_composition.copy()
        comp = comp[comp["city"].astype(str).isin(plot_df["city"].astype(str))].copy()
        if not comp.empty:
            comp_pivot = (
                comp.pivot(index="city", columns="street_pattern_dominant_class", values="block_share")
                .fillna(0.0)
            )
            if "Irregular Grid" in comp_pivot.columns:
                city_order = (
                    pd.to_numeric(comp_pivot["Irregular Grid"], errors="coerce")
                    .sort_values(ascending=False)
                    .index.astype(str)
                    .tolist()
                )
                if city_order:
                    order_rank = {city: idx for idx, city in enumerate(city_order)}
                    plot_df["_city_order_rank"] = plot_df["city"].astype(str).map(order_rank).fillna(len(order_rank))
                    plot_df = plot_df.sort_values("_city_order_rank", ascending=True).drop(columns="_city_order_rank")
                    sorted_by_irregular = True
    if not sorted_by_irregular:
        irregular_col = "street_pattern_prob_irregular_grid"
        if irregular_col in plot_df.columns:
            irregular_values = pd.to_numeric(plot_df[irregular_col], errors="coerce")
            plot_df = plot_df.assign(_irregular_sort=irregular_values).sort_values("_irregular_sort", ascending=False).drop(columns="_irregular_sort")
        elif "blocks_total" in plot_df.columns:
            plot_df = plot_df.sort_values("blocks_total", ascending=False)
        else:
            plot_df = plot_df.sort_values("accessibility_median_accessibility_time_mean_pt_intermodal")
    if {"population_total", "blocks_total"}.issubset(plot_df.columns):
        population_total = pd.to_numeric(plot_df["population_total"], errors="coerce")
        blocks_total = pd.to_numeric(plot_df["blocks_total"], errors="coerce")
        blocks_total = blocks_total.where(blocks_total > 0, np.nan)
        plot_df["population_per_block"] = population_total / blocks_total
    if "bus_routes_per_100k_population" in plot_df.columns:
        routes_per_100k = pd.to_numeric(plot_df["bus_routes_per_100k_population"], errors="coerce")
        routes_per_100k = routes_per_100k.where(routes_per_100k > 0, np.nan)
        log_routes_per_100k = np.log1p(routes_per_100k)
        log_routes_per_100k = log_routes_per_100k.where(log_routes_per_100k > 0, np.nan)
        plot_df["bus_routes_log1p_per_100k_population"] = log_routes_per_100k
    metric_specs: list[tuple[str, str]] = [
        ("blocks_total", "Blocks"),
        ("population_total", "Population"),
        ("population_per_block", "Population per block"),
        ("bus_route_count", "Bus routes"),
        ("bus_routes_per_100k_population", "Bus routes per 100k"),
        ("bus_routes_log1p_per_100k_population", "log1p(routes per 100k)"),
        ("bus_stop_count", "Bus stops"),
        ("accessibility_median_accessibility_time_mean_pt_intermodal", "Intermodal median"),
        ("accessibility_median_accessibility_time_mean_pt_walk", "Walk median"),
        ("accessibility_median_delta_walk_minus_intermodal", "Walk - intermodal delta"),
    ]
    service_metric_specs = sorted(
        [
            (
                col,
                _transport_response_label(col.removeprefix("accessibility_median_")),
            )
            for col in plot_df.columns
            if col.startswith("accessibility_median_assigned_service_time_") and not col.endswith("_per_log_bus_routes_per_100k")
        ],
        key=lambda item: item[0],
    )
    metric_specs.extend(service_metric_specs)
    available_specs = [spec for spec in metric_specs if spec[0] in plot_df.columns]
    if not available_specs:
        return
    metric_color_map: dict[str, str] = {
        "blocks_total": "#0f766e",
        "population_total": "#b45309",
        "population_per_block": "#b91c1c",
        "bus_route_count": "#dc2626",
        "bus_routes_per_100k_population": "#be185d",
        "bus_routes_log1p_per_100k_population": "#9d174d",
        "bus_stop_count": "#4f46e5",
        "accessibility_median_accessibility_time_mean_pt_intermodal": "#2563eb",
        "accessibility_median_accessibility_time_mean_pt_walk": "#ea580c",
        "accessibility_median_delta_walk_minus_intermodal": "#a78bfa",
    }
    fallback_cycle = [
        "#22c55e",
        "#f59e0b",
        "#06b6d4",
        "#ef4444",
        "#8b5cf6",
        "#14b8a6",
        "#eab308",
        "#f97316",
    ]

    panel_defs: list[tuple[str, tuple[str, str] | None]] = []
    has_composition_panel = class_composition is not None and not class_composition.empty
    has_distribution_panel = (
        distribution_frame is not None
        and not distribution_frame.empty
        and {"city", "accessibility_time_mean_pt_intermodal", "accessibility_time_mean_pt_walk"}.issubset(distribution_frame.columns)
    )
    if has_composition_panel:
        panel_defs.append(("composition", None))
    if has_distribution_panel:
        panel_defs.append(("distribution", None))
    panel_defs.extend([("metric", spec) for spec in available_specs])

    nrows = len(panel_defs)
    height_ratios = [1.5 if kind in {"composition", "distribution"} else 1.0 for kind, _ in panel_defs]
    fig_height = max(8.0, 2.1 * sum(height_ratios))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(13.5, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor(canvas_gray)
    x = np.arange(len(plot_df), dtype=float)
    city_labels = plot_df["city"].tolist()

    city_order = city_labels
    for ax, (kind, payload) in zip(axes, panel_defs):
        ax.set_facecolor(canvas_gray)
        if kind == "composition":
            comp = class_composition.copy() if class_composition is not None else pd.DataFrame()
            comp = comp[comp["city"].astype(str).isin(city_order)].copy()
            if comp.empty:
                ax.axis("off")
                continue
            pivot = (
                comp.pivot(index="city", columns="street_pattern_dominant_class", values="block_share")
                .reindex(city_order)
                .fillna(0.0)
            )
            bottom = np.zeros(len(pivot), dtype=float)
            for class_name in pivot.columns:
                values = pd.to_numeric(pivot[class_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                ax.bar(
                    np.arange(len(pivot), dtype=float),
                    values,
                    bottom=bottom,
                    color=CLASS_COLORS.get(class_name, "#9ca3af"),
                    width=0.72,
                    zorder=3,
                )
                bottom = bottom + values
            ax.set_ylim(0.0, 1.0)
            ax.set_title("Dominant Street Pattern Composition", fontsize=12, fontweight="bold", color="#ffffff", loc="left", pad=6)
            ax.grid(True, axis="y", alpha=0.18, linewidth=0.6, color="#e5e7eb")
            ax.tick_params(axis="y", colors="#ffffff")
        elif kind == "distribution":
            dist = distribution_frame.copy() if distribution_frame is not None else pd.DataFrame()
            dist = dist[dist["city"].astype(str).isin(city_order)].copy()
            subset = dist[
                ["city", "accessibility_time_mean_pt_intermodal", "accessibility_time_mean_pt_walk"]
            ].melt(
                id_vars="city",
                value_vars=["accessibility_time_mean_pt_intermodal", "accessibility_time_mean_pt_walk"],
                var_name="graph_type",
                value_name="accessibility_time_mean_pt",
            )
            subset["accessibility_time_mean_pt"] = pd.to_numeric(
                subset["accessibility_time_mean_pt"], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan)
            subset = subset.dropna(subset=["accessibility_time_mean_pt"]).copy()
            if subset.empty:
                ax.axis("off")
                continue
            sns.boxplot(
                data=subset,
                x="city",
                y="accessibility_time_mean_pt",
                hue="graph_type",
                order=city_order,
                fliersize=1.8,
                ax=ax,
            )
            ax.set_ylim(0, 90)
            handles, labels = ax.get_legend_handles_labels()
            mapped_labels = [_transport_response_label(label) for label in labels]
            ax.legend(handles, mapped_labels, loc="upper right", frameon=False, fontsize=8)
            ax.set_title("Accessibility Distribution (Intermodal vs Walk)", fontsize=12, fontweight="bold", color="#ffffff", loc="left", pad=6)
            ax.set_ylabel("mean travel time, min")
            ax.grid(True, axis="y", alpha=0.18, linewidth=0.6, color="#e5e7eb")
            ax.tick_params(axis="y", colors="#ffffff")
        else:
            assert payload is not None
            column, label = payload
            values = pd.to_numeric(plot_df[column], errors="coerce").to_numpy(dtype=float)
            finite_mask = np.isfinite(values)
            if column == "accessibility_median_delta_walk_minus_intermodal":
                ax.axhline(0.0, color="#e5e7eb", linewidth=1.0, linestyle="--", zorder=1)
            color = metric_color_map.get(column, fallback_cycle[hash(column) % len(fallback_cycle)])
            ax.bar(x[finite_mask], values[finite_mask], color=color, width=0.72, zorder=3)
            ax.set_title(label, fontsize=12, fontweight="bold", color="#ffffff", loc="left", pad=6)
            ax.grid(True, axis="y", alpha=0.18, linewidth=0.6, color="#e5e7eb")
            ax.tick_params(axis="y", colors="#ffffff")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_title("City Accessibility And Transport Context", fontsize=16, fontweight="bold", color="#ffffff", loc="left", pad=12)
    pattern_handles = [
        Patch(facecolor=CLASS_COLORS.get(label, "#9ca3af"), edgecolor="none", label=label)
        for label in [*CLASS_LABELS.values(), "unknown"]
    ]
    fig.legend(
        handles=pattern_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=4,
        frameon=False,
        fontsize=8,
        labelcolor="#ffffff",
    )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        city_labels,
        rotation=CROSS_CITY_XLABEL_ROTATION,
        ha="right",
        color="#ffffff",
        fontsize=CROSS_CITY_XLABEL_FONTSIZE,
    )
    for ax in axes[:-1]:
        ax.tick_params(axis="x", colors="#ffffff", labelbottom=False)
    axes[-1].tick_params(axis="x", colors="#ffffff")
    footer_text(
        fig,
        [
            "each row is one metric; x-axis is city order sorted by irregular-grid share",
            "bar color = metric (each row has its own color)",
            "normalized route metric: bus routes per 100k population",
            "delta = median walk travel time - median intermodal travel time; accessibility medians use populated blocks only",
        ],
        y=0.012,
        color="#f3f4f6",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.955))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _count_routes_in_graph(graph_path: Path, modality: str) -> int:
    graph = _read_graph_pickle(graph_path)
    routes = {
        str(data.get("route"))
        for _u, _v, _k, data in graph.edges(keys=True, data=True)
        if data.get("type") == modality and data.get("route") is not None
    }
    return len(routes)


def _cross_city_context(
    city_blocks: dict[str, gpd.GeoDataFrame],
    city_paths: dict[str, CityPaths],
    city_summaries: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for city, blocks in city_blocks.items():
        summary = city_summaries.get(city, {})
        stop_modalities = summary.get("stop_modalities", {}) if isinstance(summary, dict) else {}
        bus_stop_count = np.nan
        if isinstance(stop_modalities, dict) and "bus" in stop_modalities:
            bus_stop_count = float(stop_modalities["bus"].get("aggregated_stops", np.nan))
        elif "stop_count_bus" in blocks.columns:
            bus_stop_count = float(pd.to_numeric(blocks["stop_count_bus"], errors="coerce").fillna(0.0).sum())
        population_total = float(pd.to_numeric(blocks.get("population"), errors="coerce").fillna(0.0).sum()) if "population" in blocks.columns else 0.0
        bus_route_count = int(_count_routes_in_graph(city_paths[city].graph_path, "bus"))
        bus_routes_per_100k_population = np.nan
        if population_total > 0:
            bus_routes_per_100k_population = (bus_route_count / population_total) * 100000.0
        rows.append(
            {
                "city": city,
                "blocks_total": int(len(blocks)),
                "population_total": population_total,
                "bus_route_count": bus_route_count,
                "bus_routes_per_100k_population": bus_routes_per_100k_population,
                "bus_stop_count": bus_stop_count,
            }
        )
    return pd.DataFrame(rows)


def _plot_accessibility_between_city_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    panels: list[tuple[str, list[str]]] = []
    base_cols = [
        col
        for col in ["accessibility_time_mean_pt_intermodal", "accessibility_time_mean_pt_walk"]
        if col in frame.columns
    ]
    if base_cols:
        panels.append(("Overall accessibility", base_cols))

    service_names = sorted(
        {
            col.removeprefix("assigned_service_time_").removesuffix("_intermodal")
            for col in frame.columns
            if col.startswith("assigned_service_time_") and col.endswith("_intermodal")
        }
    )
    nearest_service_names = sorted(
        {
            col.removeprefix("nearest_service_time_").removesuffix("_intermodal")
            for col in frame.columns
            if col.startswith("nearest_service_time_") and col.endswith("_intermodal")
        }
    )
    if nearest_service_names:
        service_names = sorted(set(service_names).union(set(nearest_service_names)))
    for service_name in service_names:
        # Figure 15 uses pure accessibility to nearest service (no provision/solver assignment).
        service_cols = [col for col in [f"nearest_service_time_{service_name}_intermodal", f"nearest_service_time_{service_name}_walk"] if col in frame.columns]
        if not service_cols:
            service_cols = [
                col
                for col in [
                    f"assigned_service_time_{service_name}_intermodal",
                    f"assigned_service_time_{service_name}_walk",
                ]
                if col in frame.columns
            ]
        if service_cols:
            panels.append((f"Nearest {service_name} accessibility", service_cols))

    if not panels:
        return

    nrows = len(panels)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(13.5, max(6.0, 4.2 * nrows)),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    did_plot = False

    for idx, ((panel_title, value_cols), ax) in enumerate(zip(panels, axes)):
        subset = frame[["city", *value_cols]].melt(
            id_vars="city",
            value_vars=value_cols,
            var_name="graph_type",
            value_name="accessibility_time_mean_pt",
        )
        subset["accessibility_time_mean_pt"] = pd.to_numeric(
            subset["accessibility_time_mean_pt"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna(subset=["accessibility_time_mean_pt"]).copy()
        if subset.empty:
            ax.axis("off")
            continue
        did_plot = True
        primary_col = value_cols[0]
        city_order = (
            subset[subset["graph_type"] == primary_col]
            .groupby("city")["accessibility_time_mean_pt"]
            .median()
            .sort_values()
            .index
            .tolist()
        )
        if not city_order:
            city_order = sorted(subset["city"].unique().tolist())
        sns.boxplot(
            data=subset,
            x="city",
            y="accessibility_time_mean_pt",
            hue="graph_type",
            order=city_order,
            fliersize=2,
            ax=ax,
        )
        ymax = subset["accessibility_time_mean_pt"].quantile(0.98)
        if pd.isna(ymax):
            ymax = 90.0
        ax.set_ylim(0, min(float(ymax) * 1.1, 120.0))
        ax.set_title(panel_title, fontsize=13, fontweight="bold", loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("mean travel time, min")
        handles, labels = ax.get_legend_handles_labels()
        mapped_labels = [_transport_response_label(label) for label in labels]
        ax.legend(handles, mapped_labels, loc="upper right", frameon=False)
        if idx < nrows - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.tick_params(axis="x", rotation=CROSS_CITY_XLABEL_ROTATION, labelsize=CROSS_CITY_XLABEL_FONTSIZE)
            for label in ax.get_xticklabels():
                label.set_ha("right")

    if not did_plot:
        plt.close(fig)
        return
    fig.suptitle("Accessibility Distribution By City", fontsize=16, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _city_level_accessibility_correlations(
    frame: pd.DataFrame,
    feature_cols: list[str],
    accessibility_response_cols: list[str],
) -> pd.DataFrame:
    expected_columns = ["response", "street_pattern_feature", "spearman_rho", "p_value", "n_cities"]
    if frame.empty:
        return pd.DataFrame(columns=expected_columns)
    rows: list[dict[str, object]] = []
    for response_col in accessibility_response_cols:
        if response_col not in frame.columns:
            continue
        for feature_col in feature_cols:
            if feature_col not in frame.columns:
                continue
            subset = frame[[response_col, feature_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(subset) < 4 or subset[response_col].nunique() < 2 or subset[feature_col].nunique() < 2:
                continue
            rho, p_value = spearmanr(subset[response_col], subset[feature_col], nan_policy="omit")
            rows.append(
                {
                    "response": response_col,
                    "street_pattern_feature": feature_col,
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "n_cities": int(len(subset)),
                }
            )
    return pd.DataFrame(rows, columns=expected_columns)


def _plot_accessibility_between_city_effects(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Cross-City Spearman: Accessibility vs Street Pattern",
) -> None:
    subset = frame.copy()
    if subset.empty:
        return
    value_table = (
        subset.assign(street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name))
        .pivot(index="response", columns="street_pattern_feature", values="spearman_rho")
    )
    p_table = (
        subset.assign(street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name))
        .pivot(index="response", columns="street_pattern_feature", values="p_value")
    )
    _heatmap(
        value_table,
        output_path,
        title=title,
        significance_table=p_table,
        fmt=".3f",
    )


def _plot_accessibility_cross_city_scatter(
    frame: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path,
    *,
    response_col: str,
    response_label: str,
    response_color: str,
    size_by_population_per_block: bool = False,
) -> None:
    if frame.empty or not feature_cols:
        return
    if response_col not in frame.columns:
        return

    n_panels = len(feature_cols)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    marker_sizes_by_city: dict[str, float] = {}
    city_ppb: pd.Series | None = None
    if size_by_population_per_block and {"city", "population_total", "blocks_total"}.issubset(frame.columns):
        city_stats = frame[["city", "population_total", "blocks_total"]].copy()
        city_stats["population_total"] = pd.to_numeric(city_stats["population_total"], errors="coerce")
        city_stats["blocks_total"] = pd.to_numeric(city_stats["blocks_total"], errors="coerce")
        city_stats["blocks_total"] = city_stats["blocks_total"].where(city_stats["blocks_total"] > 0, np.nan)
        city_stats["population_per_block"] = city_stats["population_total"] / city_stats["blocks_total"]
        city_stats = city_stats.dropna(subset=["population_per_block"]).copy()
        if not city_stats.empty:
            city_ppb = city_stats.groupby("city")["population_per_block"].median()
            low = float(city_ppb.quantile(0.05))
            high = float(city_ppb.quantile(0.95))
            if not np.isfinite(low):
                low = float(city_ppb.min())
            if not np.isfinite(high):
                high = float(city_ppb.max())
            if high <= low:
                low, high = float(city_ppb.min()), float(city_ppb.max())
            if high <= low:
                marker_sizes_by_city = {str(city): 60.0 for city in city_ppb.index}
            else:
                for city, value in city_ppb.items():
                    clipped = min(max(float(value), low), high)
                    norm = (clipped - low) / (high - low)
                    marker_sizes_by_city[str(city)] = 30.0 + (norm * 130.0)

    for ax, feature_col in zip(axes, feature_cols):
        feature_label = _friendly_feature_name(feature_col)
        subset = frame[[feature_col, response_col, "city"]].copy()
        subset[feature_col] = pd.to_numeric(subset[feature_col], errors="coerce")
        subset[response_col] = pd.to_numeric(subset[response_col], errors="coerce")
        subset = subset.dropna(subset=[feature_col, response_col]).copy()
        if subset.empty:
            ax.axis("off")
            continue
        if size_by_population_per_block and marker_sizes_by_city:
            marker_sizes = subset["city"].astype(str).map(marker_sizes_by_city).fillna(60.0).to_numpy(dtype=float)
        else:
            marker_sizes = np.full(len(subset), 46.0, dtype=float)
        ax.scatter(
            subset[feature_col],
            subset[response_col],
            s=marker_sizes,
            color=response_color,
            alpha=0.88,
            edgecolor="#ffffff",
            linewidth=0.7,
        )
        if subset[feature_col].nunique() >= 2 and subset[response_col].nunique() >= 2:
            coeffs = np.polyfit(subset[feature_col].to_numpy(dtype=float), subset[response_col].to_numpy(dtype=float), deg=1)
            x_grid = np.linspace(float(subset[feature_col].min()), float(subset[feature_col].max()), 50)
            y_grid = coeffs[0] * x_grid + coeffs[1]
            ax.plot(x_grid, y_grid, color=response_color, linewidth=1.5, alpha=0.92)
            if len(subset) >= 4:
                rho, p_value = spearmanr(subset[feature_col], subset[response_col], nan_policy="omit")
                if pd.notna(rho) and pd.notna(p_value):
                    is_significant = bool(float(p_value) < 0.05 and abs(float(rho)) >= EFFECT_HIGHLIGHT_MIN_ABS)
                    ax.text(
                        0.02,
                        0.98,
                        (
                            f"{'significant correlation' if is_significant else 'not significant'}\n"
                            f"rho={rho:.2f}, p={p_value:.3f}"
                        ),
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="#166534" if is_significant else "#991b1b",
                        bbox={
                            "facecolor": "#dcfce7" if is_significant else "#fee2e2",
                            "edgecolor": "#16a34a" if is_significant else "#dc2626",
                            "alpha": 0.95,
                            "boxstyle": "round,pad=0.25",
                        },
                    )
        ax.set_title(feature_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("city mean feature share")
        ax.set_ylabel("city mean accessibility, min")
        ax.grid(True, alpha=0.18, linewidth=0.6)

    for ax in axes[n_panels:]:
        ax.axis("off")

    handles = [Patch(facecolor=response_color, edgecolor="none", label=response_label)]
    if size_by_population_per_block and marker_sizes_by_city and city_ppb is not None and not city_ppb.empty:
        q_vals = city_ppb.quantile([0.2, 0.5, 0.8]).to_numpy(dtype=float)
        low = float(np.nanmin(city_ppb.to_numpy(dtype=float)))
        high = float(np.nanmax(city_ppb.to_numpy(dtype=float)))
        if np.isfinite(low) and np.isfinite(high) and high > low:
            size_handles: list[Line2D] = []
            for qv in q_vals:
                clipped = min(max(float(qv), low), high)
                norm = (clipped - low) / (high - low)
                ms = np.sqrt(30.0 + (norm * 130.0))
                size_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor="#60a5fa",
                        markeredgecolor="#ffffff",
                        markeredgewidth=0.7,
                        markersize=ms,
                        label=f"ppb≈{int(round(qv))}",
                    )
                )
            handles.extend(size_handles)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=min(4, max(2, len(handles))),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(f"Cross-City {response_label} Accessibility vs Street Pattern", fontsize=16, fontweight="bold", y=0.995)
    if response_col.startswith("assigned_service_time_"):
        parts = response_col.split("_")
        if len(parts) >= 5:
            service_name = parts[3]
            fig.text(
                0.5,
                0.968,
                f"Service: {service_name}",
                ha="center",
                va="top",
                fontsize=11,
                color="#2563eb",
                fontweight="bold",
            )
    footer_text(
        fig,
        [
            "each point is one city; line = linear trend",
            "point size reflects population per block" if size_by_population_per_block else "point size is constant",
            "annotation shows city-level Spearman rho and p-value",
        ],
        y=0.012,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _transport_response_label(response_col: str) -> str:
    labels = {
        "accessibility_time_mean_pt_intermodal": "Intermodal accessibility",
        "accessibility_time_mean_pt_walk": "Walk accessibility",
        "assigned_service_time_hospital_intermodal": "Assigned hospital accessibility (intermodal)",
        "assigned_service_time_hospital_walk": "Assigned hospital accessibility (walk)",
        "assigned_service_time_polyclinic_intermodal": "Assigned polyclinic accessibility (intermodal)",
        "assigned_service_time_polyclinic_walk": "Assigned polyclinic accessibility (walk)",
        "assigned_service_time_school_intermodal": "Assigned school accessibility (intermodal)",
        "assigned_service_time_school_walk": "Assigned school accessibility (walk)",
    }
    if response_col.startswith("nearest_service_time_"):
        suffix = response_col.removeprefix("nearest_service_time_")
        parts = suffix.rsplit("_", 1)
        if len(parts) == 2:
            service, context = parts
            return f"Nearest {service} accessibility ({context})"
    return labels.get(response_col, response_col)


def _parse_assigned_response(response_col: str) -> tuple[str, str] | None:
    if not response_col.startswith("assigned_service_time_"):
        return None
    suffix = response_col.removeprefix("assigned_service_time_")
    parts = suffix.rsplit("_", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _plot_cross_city_accessibility_service_atlas(
    *,
    city_blocks: dict[str, gpd.GeoDataFrame],
    city_paths: dict[str, CityPaths],
    accessibility_between: pd.DataFrame,
    output_path: Path,
    services: list[str],
    eligible_cities: list[str] | None = None,
) -> None:
    if accessibility_between.empty:
        return

    target_features = {
        _class_feature_slug("Sparse"),
        _class_feature_slug("Broken Grid"),
    }
    effects = accessibility_between.copy()
    effects["spearman_rho"] = pd.to_numeric(effects.get("spearman_rho"), errors="coerce")
    effects["p_value"] = pd.to_numeric(effects.get("p_value"), errors="coerce")
    effects = effects[
        effects["street_pattern_feature"].astype(str).isin(target_features)
        & effects["spearman_rho"].notna()
        & effects["p_value"].notna()
    ].copy()
    effects = effects[
        (effects["p_value"] < 0.05)
        & (effects["spearman_rho"].abs() >= EFFECT_HIGHLIGHT_MIN_ABS)
    ].copy()
    if effects.empty:
        _warn(
            "Cross-city atlas skipped: no significant accessibility links for Sparse/Broken Grid "
            f"(abs(rho)>={EFFECT_HIGHLIGHT_MIN_ABS}, p<0.05)."
        )
        return

    effects["is_assigned"] = effects["response"].astype(str).str.startswith("assigned_service_time_")
    effects["abs_rho"] = effects["spearman_rho"].abs()
    effects = effects.sort_values(
        by=["is_assigned", "abs_rho", "p_value"],
        ascending=[False, False, True],
    )
    chosen = effects.iloc[0]
    response_col = str(chosen["response"])
    feature_col = str(chosen["street_pattern_feature"])
    feature_label = _friendly_feature_name(feature_col)
    response_label = _transport_response_label(response_col)
    parsed = _parse_assigned_response(response_col)
    selected_service = parsed[0] if parsed is not None else None
    selected_context = parsed[1] if parsed is not None else None
    provision_col = (
        f"service_provision_{selected_service}_{selected_context}"
        if selected_service is not None and selected_context is not None
        else None
    )
    city_pool = sorted(set(eligible_cities or city_blocks.keys()))
    available_cities: list[str] = []
    for city in city_pool:
        blocks = city_blocks.get(city)
        if blocks is None or blocks.empty:
            continue
        if response_col not in blocks.columns:
            continue
        response_values = pd.to_numeric(blocks[response_col], errors="coerce")
        if int(response_values.notna().sum()) <= 0:
            continue
        available_cities.append(city)
    if not available_cities:
        _warn(f"Cross-city atlas skipped: no cities with non-empty '{response_col}'.")
        return

    sampled_cities = available_cities

    all_values: list[np.ndarray] = []
    for city in sampled_cities:
        values = pd.to_numeric(city_blocks[city].get(response_col), errors="coerce")
        values = values.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            all_values.append(finite)
    if not all_values:
        _warn(f"Cross-city atlas skipped: no finite values for '{response_col}'.")
        return
    stacked_values = np.concatenate(all_values)
    vmin = float(np.nanquantile(stacked_values, 0.05))
    vmax = float(np.nanquantile(stacked_values, 0.95))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(stacked_values))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(stacked_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return
    if vmax <= vmin:
        vmax = vmin + 1e-6

    ncols = 4
    nrows = int(np.ceil(len(sampled_cities) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    service_colors = {
        "school": "#2563eb",
        "polyclinic": "#f59e0b",
        "hospital": "#ef4444",
    }

    for idx, ax in enumerate(axes):
        if idx >= len(sampled_cities):
            ax.axis("off")
            continue
        city = str(sampled_cities[idx])
        blocks = city_blocks[city].copy()
        boundary = _load_boundary(city_paths[city].boundary_path)

        boundary_norm = normalize_preview_gdf(boundary)
        apply_preview_canvas(fig, ax, boundary_norm, title=city)
        ax.set_title(city, fontsize=10, color="#ffffff", pad=8)

        values_primary = pd.to_numeric(blocks.get(response_col), errors="coerce")
        values = values_primary.copy()
        selected_mask = blocks.geometry.notna() & ~blocks.geometry.is_empty
        valid_mask = selected_mask & values.notna()
        valid = blocks.loc[valid_mask].copy()
        if not valid.empty:
            valid["plot_accessibility_time"] = values.loc[valid.index].to_numpy(dtype=float)
            valid_norm = normalize_preview_gdf(valid, boundary_norm)
            valid_norm.plot(
                ax=ax,
                column="plot_accessibility_time",
                cmap="viridis_r",
                vmin=vmin,
                vmax=vmax,
                linewidth=0.05,
                edgecolor="#ffffff",
                alpha=0.96,
                zorder=4,
            )
        missing = blocks.loc[selected_mask & values.isna()].copy()
        if not missing.empty:
            missing_norm = normalize_preview_gdf(missing, boundary_norm)
            missing_norm.plot(
                ax=ax,
                color="#d1d5db",
                edgecolor="#ffffff",
                linewidth=0.03,
                alpha=0.55,
                zorder=3,
            )

        if provision_col is not None and provision_col in blocks.columns:
            provision = pd.to_numeric(blocks.get(provision_col), errors="coerce")
            unmet = blocks.loc[selected_mask & provision.notna() & (provision < 0.999)].copy()
            if not unmet.empty:
                unmet_norm = normalize_preview_gdf(unmet, boundary_norm)
                unmet_norm.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="#f43f5e",
                    linewidth=0.0,
                    hatch="////",
                    alpha=0.95,
                    zorder=7,
                )
                unmet_norm.boundary.plot(
                    ax=ax,
                    color="#f43f5e",
                    linewidth=0.32,
                    alpha=0.96,
                    zorder=8,
                )

        for service in services:
            raw_path = city_paths[city].output_dir / "raw_services" / f"{service}.parquet"
            if not raw_path.exists():
                continue
            raw = read_geodata(raw_path)
            if raw.empty:
                continue
            points = _to_points(raw)
            if points.empty:
                continue
            points_norm = normalize_preview_gdf(points, boundary_norm)
            points_norm.plot(
                ax=ax,
                color=service_colors.get(service, "#0ea5e9"),
                markersize=10.5 if service == selected_service else 7.8,
                edgecolor="#ffffff",
                linewidth=0.85,
                alpha=0.9,
                zorder=8,
            )

        population = pd.to_numeric(blocks.get("population"), errors="coerce").fillna(0.0)
        dominant = blocks.get("street_pattern_dominant_class")
        if dominant is not None:
            pop_by_class = (
                pd.DataFrame(
                    {
                        "street_pattern_dominant_class": dominant.astype(str).fillna("unknown"),
                        "population": population,
                    }
                )
                .groupby("street_pattern_dominant_class", as_index=False)["population"]
                .sum()
            )
            pop_by_class = pop_by_class[pop_by_class["population"] > 0].copy()
            total_population = float(pop_by_class["population"].sum())
            if total_population > 0 and not pop_by_class.empty:
                pop_by_class["population_share"] = pop_by_class["population"] / total_population
                pop_by_class = pop_by_class.sort_values("population_share", ascending=False).head(5).copy()
                inset = ax.inset_axes([0.61, 0.04, 0.35, 0.26], zorder=20)
                inset.set_facecolor((1.0, 1.0, 1.0, 0.88))
                inset.barh(
                    pop_by_class["street_pattern_dominant_class"],
                    pop_by_class["population_share"],
                    color=[
                        CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])
                        for class_name in pop_by_class["street_pattern_dominant_class"]
                    ],
                    edgecolor="#ffffff",
                    linewidth=0.4,
                )
                inset.invert_yaxis()
                inset.set_xlim(0.0, max(0.35, float(pop_by_class["population_share"].max()) * 1.12))
                inset.set_title("population by pattern", fontsize=6.6, pad=1.5)
                inset.tick_params(axis="x", labelsize=5.5, length=1.5)
                inset.tick_params(axis="y", labelsize=5.3, length=0.0, pad=0.8)
                inset.grid(True, axis="x", alpha=0.14, linewidth=0.4)
                inset.set_xlabel("")
                inset.set_ylabel("")
                for spine in inset.spines.values():
                    spine.set_color("#cbd5e1")
                    spine.set_linewidth(0.5)
        ax.set_axis_off()

    for ax in axes[len(sampled_cities):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap="viridis_r")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.018, pad=0.015, orientation="horizontal")
    if parsed is not None:
        cbar.set_label("assigned accessibility time, min")
    else:
        cbar.set_label("accessibility time, min")

    legend_handles: list = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=service_colors.get(service, "#0ea5e9"), markersize=6, label=service)
        for service in services
    ]
    if provision_col is not None:
        legend_handles.append(
            Patch(
                facecolor="none",
                edgecolor="#f43f5e",
                hatch="////",
                linewidth=1.0,
                label="not fully provided blocks",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(5, max(3, len(legend_handles))),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        (
            f"Cross-City Atlas: {response_label}\n"
            f"Selected by significant link with {feature_label} "
            f"(rho={float(chosen['spearman_rho']):.2f}, p={float(chosen['p_value']):.3f}); "
            f"{len(sampled_cities)} cities"
        ),
        fontsize=14,
        fontweight="bold",
        y=1.035,
    )
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.03, right=0.98, hspace=0.15, wspace=0.06)
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_response_by_dominant_class_by_city(
    frame: pd.DataFrame,
    *,
    response_col: str,
    output_path: Path,
    title: str,
) -> None:
    subset = frame[["city", "street_pattern_dominant_class", response_col]].copy()
    subset[response_col] = pd.to_numeric(subset[response_col], errors="coerce")
    subset["street_pattern_dominant_class"] = subset["street_pattern_dominant_class"].fillna("unknown").astype(str)
    subset = subset.dropna(subset=[response_col]).copy()
    if subset.empty:
        return

    class_order = [label for label in CLASS_LABELS.values() if label in set(subset["street_pattern_dominant_class"])]
    if "unknown" in set(subset["street_pattern_dominant_class"]):
        class_order.append("unknown")
    city_order = (
        subset.groupby("city")[response_col]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    n_panels = len(city_order)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.4 * ncols, max(4.5, 3.6 * nrows)), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    y_cap = subset[response_col].quantile(0.98)
    if response_col.startswith("accessibility_time_mean_pt"):
        y_cap = min(float(y_cap), 120.0) if pd.notna(y_cap) else 120.0

    for ax, city in zip(axes, city_order):
        city_df = subset[subset["city"] == city].copy()
        sns.boxplot(
            data=city_df,
            x="street_pattern_dominant_class",
            y=response_col,
            order=class_order,
            color="#60a5fa",
            fliersize=1.8,
            ax=ax,
        )
        ax.set_title(city, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("minutes")
        ax.tick_params(axis="x", rotation=18)
        if pd.notna(y_cap) and float(y_cap) > 0:
            ax.set_ylim(0, float(y_cap))
        ax.grid(True, axis="y", alpha=0.16, linewidth=0.6)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    footer_text(fig, ["per-city boxplots by dominant street-pattern class"], y=0.012)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _fit_transport_fixed_effects(
    frame: pd.DataFrame,
    *,
    response_cols: list[str],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    if len(feature_cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    pattern_terms = " + ".join(feature_cols[:-1])
    rhs = f"C(city) + {pattern_terms} + log_population + log_block_area"
    for response_col in response_cols:
        required = ["city", response_col, "log_population", "log_block_area", *feature_cols]
        subset = frame[required].copy()
        for col in required:
            if col != "city":
                subset[col] = pd.to_numeric(subset[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        subset = subset.dropna(subset=required).copy()
        if len(subset) < 20 or subset["city"].nunique() < 2 or subset[response_col].nunique() < 2:
            continue
        family = _response_family(subset, response_col)
        try:
            result = _fit_formula_model(subset, response_col, rhs)
        except Exception:
            continue
        summary_rows.append(
            {
                "response": response_col,
                "response_label": _transport_response_label(response_col),
                "family": family,
                "n": int(len(subset)),
                "n_cities": int(subset["city"].nunique()),
                "model_score": _model_score(result, family),
            }
        )
        for term, estimate in result.params.items():
            if term not in feature_cols:
                continue
            coef_rows.append(
                {
                    "response": response_col,
                    "response_label": _transport_response_label(response_col),
                    "street_pattern_feature": term,
                    "street_pattern_label": _friendly_feature_name(term),
                    "coef": float(estimate) if pd.notna(estimate) else np.nan,
                    "std_err": float(result.bse.get(term, np.nan)),
                    "p_value": float(result.pvalues.get(term, np.nan)),
                    "n": int(len(subset)),
                    "n_cities": int(subset["city"].nunique()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(coef_rows)


def _plot_transport_coefficient_effects(
    coef_df: pd.DataFrame,
    *,
    response_col: str,
    output_path: Path,
) -> None:
    subset = coef_df[coef_df["response"] == response_col].copy()
    if subset.empty:
        return
    subset["ci_low"] = subset["coef"] - 1.96 * subset["std_err"]
    subset["ci_high"] = subset["coef"] + 1.96 * subset["std_err"]
    subset["is_significant"] = pd.to_numeric(subset["p_value"], errors="coerce") < 0.05
    subset = subset.sort_values("coef")
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.8 * len(subset) + 1)))
    y_positions = np.arange(len(subset), dtype=float)
    ax.axvline(0.0, color="#94a3b8", linewidth=1.2, linestyle="--", zorder=1)
    colors = [
        "#b91c1c" if (sig and coef > 0) else "#2563eb" if (sig and coef < 0) else "#d1d5db"
        for sig, coef in zip(subset["is_significant"], subset["coef"])
    ]
    ax.hlines(y_positions, subset["ci_low"], subset["ci_high"], color=colors, linewidth=2.0, zorder=2)
    ax.scatter(subset["coef"], y_positions, color=colors, s=44, edgecolor="#ffffff", linewidth=0.7, zorder=3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(subset["street_pattern_label"])
    ax.set_xlabel("fixed-effects coefficient")
    ax.set_ylabel("")
    ax.set_title(_transport_response_label(response_col), fontsize=15, fontweight="bold")
    footer_text(
        fig,
        [
            "pooled model: response ~ city fixed effects + street pattern + log population + log block area",
            "red = higher travel time, blue = lower travel time, gray = not significant",
        ],
        y=0.012,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _write_transport_story_outputs(
    *,
    combined: gpd.GeoDataFrame,
    output_root: Path,
    services: list[str],
    feature_cols: list[str],
) -> None:
    story_dir = output_root / "_transport_pattern_story"
    stats_dir = story_dir / "stats"
    preview_dir = story_dir / "preview_png"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    base_responses = [
        "accessibility_time_mean_pt_intermodal",
        "accessibility_time_mean_pt_walk",
    ]
    assigned_responses = [
        f"assigned_service_time_{service}_{context}"
        for service in services
        for context in ("intermodal", "walk")
        if f"assigned_service_time_{service}_{context}" in combined.columns
    ]
    all_responses = [*base_responses, *assigned_responses]

    availability_rows: list[dict[str, object]] = []
    for response_col in all_responses:
        values = pd.to_numeric(combined.get(response_col), errors="coerce").replace([np.inf, -np.inf], np.nan)
        cities_with_data = int(combined.loc[values.notna(), "city"].nunique()) if values is not None else 0
        availability_rows.append(
            {
                "response": response_col,
                "response_label": _transport_response_label(response_col),
                "nonnull_blocks": int(values.notna().sum()) if values is not None else 0,
                "cities_with_data": cities_with_data,
            }
        )
    availability_df = pd.DataFrame(availability_rows)

    fixed_summary, fixed_coefs = _fit_transport_fixed_effects(
        combined,
        response_cols=all_responses,
        feature_cols=feature_cols,
    )

    (stats_dir / "response_availability.csv").write_text(availability_df.to_csv(index=False), encoding="utf-8")
    (stats_dir / "fixed_effect_model_summary.csv").write_text(fixed_summary.to_csv(index=False), encoding="utf-8")
    (stats_dir / "fixed_effect_coefficients.csv").write_text(fixed_coefs.to_csv(index=False), encoding="utf-8")

    _plot_response_by_dominant_class_by_city(
        combined,
        response_col="accessibility_time_mean_pt_intermodal",
        output_path=preview_dir / "01_intermodal_accessibility_by_class_by_city.png",
        title="Intermodal Accessibility By Dominant Street Pattern Class",
    )
    _plot_response_by_dominant_class_by_city(
        combined,
        response_col="accessibility_time_mean_pt_walk",
        output_path=preview_dir / "02_walk_accessibility_by_class_by_city.png",
        title="Walk Accessibility By Dominant Street Pattern Class",
    )
    _plot_transport_coefficient_effects(
        fixed_coefs,
        response_col="accessibility_time_mean_pt_intermodal",
        output_path=preview_dir / "03_intermodal_fixed_effect_coefficients.png",
    )
    _plot_transport_coefficient_effects(
        fixed_coefs,
        response_col="accessibility_time_mean_pt_walk",
        output_path=preview_dir / "04_walk_fixed_effect_coefficients.png",
    )

    assigned_plot_index = 5
    for response_col in assigned_responses:
        available_cities = int(availability_df.loc[availability_df["response"] == response_col, "cities_with_data"].max()) if not availability_df.empty else 0
        nonnull_blocks = int(availability_df.loc[availability_df["response"] == response_col, "nonnull_blocks"].max()) if not availability_df.empty else 0
        if nonnull_blocks <= 0:
            continue
        slug = response_col.removeprefix("assigned_service_time_")
        _plot_response_by_dominant_class_by_city(
            combined,
            response_col=response_col,
            output_path=preview_dir / f"{assigned_plot_index:02d}_{slug}_by_class_by_city.png",
            title=f"{_transport_response_label(response_col)} By Dominant Street Pattern Class",
        )
        assigned_plot_index += 1
        if available_cities >= 2:
            _plot_transport_coefficient_effects(
                fixed_coefs,
                response_col=response_col,
                output_path=preview_dir / f"{assigned_plot_index:02d}_{slug}_fixed_effect_coefficients.png",
            )
            assigned_plot_index += 1


def _write_social_access_story_outputs(
    *,
    combined: gpd.GeoDataFrame,
    output_root: Path,
    services: list[str],
    feature_cols: list[str],
) -> None:
    story_dir = output_root / "_social_access_pattern_story"
    stats_dir = story_dir / "stats"
    preview_dir = story_dir / "preview_png"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    selected_services = [service for service in SOCIAL_SERVICES if service in services]
    response_cols = [
        f"assigned_service_time_{service}_{context}"
        for service in selected_services
        for context in ("intermodal", "walk")
        if f"assigned_service_time_{service}_{context}" in combined.columns
    ]
    if not response_cols:
        return

    availability_rows: list[dict[str, object]] = []
    for response_col in response_cols:
        values = pd.to_numeric(combined.get(response_col), errors="coerce").replace([np.inf, -np.inf], np.nan)
        availability_rows.append(
            {
                "response": response_col,
                "response_label": _transport_response_label(response_col),
                "nonnull_blocks": int(values.notna().sum()),
                "cities_with_data": int(combined.loc[values.notna(), "city"].nunique()),
            }
        )
    availability_df = pd.DataFrame(availability_rows)
    fixed_summary, fixed_coefs = _fit_transport_fixed_effects(
        combined,
        response_cols=response_cols,
        feature_cols=feature_cols,
    )

    (stats_dir / "response_availability.csv").write_text(availability_df.to_csv(index=False), encoding="utf-8")
    (stats_dir / "fixed_effect_model_summary.csv").write_text(fixed_summary.to_csv(index=False), encoding="utf-8")
    (stats_dir / "fixed_effect_coefficients.csv").write_text(fixed_coefs.to_csv(index=False), encoding="utf-8")

    summary_lines = [
        "# Social Accessibility Story",
        "",
        "Этот слой читает accessibility не как среднее время до всех кварталов, а как доступность до конкретной социалки.",
        "",
        "Фокусированные destinations:",
        "",
    ]
    for service in selected_services:
        summary_lines.append(f"- `{service}`")
    summary_lines.extend(
        [
            "",
            "Здесь основная переменная результата:",
            "",
            "- `assigned_service_time_<service>_intermodal`",
            "- `assigned_service_time_<service>_walk`",
            "",
            "То есть время от квартала до назначенного сервиса в соответствующем контексте, а не средняя доступность \"вообще\".",
        ]
    )
    (story_dir / "narrative_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    plot_index = 1
    for service in selected_services:
        for context in ("intermodal", "walk"):
            response_col = f"assigned_service_time_{service}_{context}"
            if response_col not in response_cols:
                continue
            values = pd.to_numeric(combined.get(response_col), errors="coerce").replace([np.inf, -np.inf], np.nan)
            if int(values.notna().sum()) <= 0:
                continue
            label = _transport_response_label(response_col)
            safe_slug = f"{service}_{context}"
            _plot_response_by_dominant_class_by_city(
                combined,
                response_col=response_col,
                output_path=preview_dir / f"{plot_index:02d}_{safe_slug}_by_class_by_city.png",
                title=f"{label} By Dominant Street Pattern Class",
            )
            plot_index += 1
            available_cities = int(availability_df.loc[availability_df["response"] == response_col, "cities_with_data"].max())
            if available_cities >= 2:
                _plot_transport_coefficient_effects(
                    fixed_coefs,
                    response_col=response_col,
                    output_path=preview_dir / f"{plot_index:02d}_{safe_slug}_fixed_effect_coefficients.png",
                )
                plot_index += 1


def _write_cross_city_outputs(
    *,
    city_blocks: dict[str, gpd.GeoDataFrame],
    city_paths: dict[str, CityPaths],
    city_summaries: dict[str, dict[str, object]],
    output_root: Path,
    services: list[str],
    accessibility_only: bool = False,
) -> None:
    if len(city_blocks) < 2:
        return
    cross_dir = output_root / "_cross_city"
    stats_dir = cross_dir / "stats"
    preview_dir = cross_dir / "preview_png"
    prepared_dir = cross_dir / "prepared"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    combined, feature_cols, stop_modalities = _prepare_cross_city_frame(city_blocks, services)
    response_cols = _cross_city_response_columns(services, stop_modalities, accessibility_only=accessibility_only)
    city_context = _cross_city_context(city_blocks, city_paths, city_summaries)
    eligible_cities: list[str] | None = None
    if not city_context.empty:
        route_counts = pd.to_numeric(city_context.get("bus_route_count"), errors="coerce")
        population_total = pd.to_numeric(city_context.get("population_total"), errors="coerce")
        eligible_mask = (route_counts >= CROSS_CITY_MIN_BUS_ROUTES) & (population_total >= CROSS_CITY_MIN_POPULATION)
        eligible_cities = city_context.loc[eligible_mask, "city"].astype(str).tolist()
        excluded_count = int((~eligible_mask.fillna(False)).sum())
        _log(
            "Cross-city eligibility filter applied: "
            f"bus_route_count>={CROSS_CITY_MIN_BUS_ROUTES}, population_total>={CROSS_CITY_MIN_POPULATION}. "
            f"Kept {len(eligible_cities)} city(ies), excluded {excluded_count}."
        )
        if len(eligible_cities) < 2:
            _warn("Cross-city output skipped after eligibility filtering: fewer than 2 eligible cities.")
            return
        combined = combined[combined["city"].astype(str).isin(eligible_cities)].copy()
        city_context = city_context[city_context["city"].astype(str).isin(eligible_cities)].copy()
    if combined.empty or combined["city"].nunique() < 2:
        _warn("Cross-city output skipped after eligibility filtering: not enough city data.")
        return

    distribution_tests = _distribution_tests(combined, feature_cols)
    class_composition = _dominant_class_composition(combined)
    variance_decomp = _variance_decomposition(combined, feature_cols)
    model_summary, model_coefs = _fit_cross_city_models(combined, response_cols, feature_cols)
    between_within = (
        pd.DataFrame()
        if accessibility_only
        else _fit_between_within_models(combined, response_cols, feature_cols)
    )
    accessibility_response_cols = [
        col
        for col in response_cols
        if col.startswith("accessibility_time_mean_pt") or col.startswith("assigned_service_time_")
    ]
    accessibility_profiles = _accessibility_city_profiles(
        combined,
        feature_cols,
        accessibility_response_cols,
        city_context=city_context,
    )
    accessibility_between = _city_level_accessibility_correlations(
        accessibility_profiles,
        feature_cols,
        accessibility_response_cols,
    )

    _save_geodata(combined, prepared_dir / "blocks_experiment_cross_city.parquet")
    for name, frame in {
        "distribution_tests.csv": distribution_tests,
        "dominant_class_composition.csv": class_composition,
        "street_pattern_variance_decomposition.csv": variance_decomp,
        "response_model_summary.csv": model_summary,
        "response_model_full_coefficients.csv": model_coefs,
        "between_within_models.csv": between_within,
        "accessibility_city_profiles.csv": accessibility_profiles,
        "accessibility_between_city_effects.csv": accessibility_between,
    }.items():
        (stats_dir / name).write_text(frame.to_csv(index=False), encoding="utf-8")

    _plot_feature_distributions(combined, feature_cols, preview_dir / "10_street_pattern_feature_distributions_by_city.png")
    _plot_dominant_class_composition(class_composition, preview_dir / "11_dominant_class_composition_by_city.png")
    _plot_variance_decomposition(variance_decomp, preview_dir / "12_street_pattern_variance_decomposition.png")
    _plot_model_comparison(model_summary, preview_dir / "13_city_vs_pattern_model_fit.png")
    _plot_effect_reduction(model_summary, preview_dir / "14_city_effect_reduction_after_pattern.png")
    _plot_accessibility_between_city_distribution(combined, preview_dir / "15_accessibility_distribution_by_city.png")
    _plot_accessibility_city_profiles(
        accessibility_profiles,
        preview_dir / "16_accessibility_city_profile.png",
        class_composition=class_composition,
        distribution_frame=combined,
    )
    _plot_accessibility_between_city_effects(accessibility_between, preview_dir / "17_accessibility_between_city_effects_heatmap.png")
    base_effects = accessibility_between[
        accessibility_between["response"].isin(
            ["accessibility_time_mean_pt_intermodal", "accessibility_time_mean_pt_walk"]
        )
    ].copy()
    _plot_accessibility_between_city_effects(
        base_effects,
        preview_dir / "17_accessibility_between_city_effects_heatmap_intermodal_walk.png",
        title="Cross-City Spearman: Overall Accessibility vs Street Pattern",
    )
    for service in sorted(set(services)):
        service_effects = accessibility_between[
            accessibility_between["response"].str.startswith(f"assigned_service_time_{service}_", na=False)
        ].copy()
        _plot_accessibility_between_city_effects(
            service_effects,
            preview_dir / f"17_accessibility_between_city_effects_heatmap_{service}.png",
            title=f"Cross-City Spearman: Assigned {service.title()} Accessibility vs Street Pattern",
        )
    scatter_response_order = [
        "accessibility_time_mean_pt_intermodal",
        "accessibility_time_mean_pt_walk",
        *sorted([col for col in accessibility_response_cols if col.startswith("assigned_service_time_")]),
    ]
    scatter_palette = {
        "accessibility_time_mean_pt_intermodal": "#2563eb",
        "accessibility_time_mean_pt_walk": "#ea580c",
    }
    for idx, response_col in enumerate(scatter_response_order, start=20):
        if response_col not in accessibility_profiles.columns:
            continue
        response_label = _transport_response_label(response_col)
        safe_suffix = (
            response_col
            .replace("assigned_service_time_", "")
            .replace("accessibility_time_mean_pt_", "")
            .replace("_", "-")
        )
        if not safe_suffix:
            safe_suffix = "metric"
        _plot_accessibility_cross_city_scatter(
            accessibility_profiles,
            feature_cols,
            preview_dir / f"{idx:02d}_accessibility_between_city_effects_{safe_suffix}.png",
            response_col=response_col,
            response_label=response_label,
            response_color=scatter_palette.get(response_col, "#60a5fa"),
            size_by_population_per_block=True,
        )
    _plot_cross_city_accessibility_service_atlas(
        city_blocks=city_blocks,
        city_paths=city_paths,
        accessibility_between=accessibility_between,
        output_path=preview_dir / "28_accessibility_service_city_atlas_sparse_broken.png",
        services=services,
        eligible_cities=eligible_cities,
    )
    _write_transport_story_outputs(
        combined=combined,
        output_root=output_root,
        services=services,
        feature_cols=feature_cols,
    )
    _write_social_access_story_outputs(
        combined=combined,
        output_root=output_root,
        services=services,
        feature_cols=feature_cols,
    )



def _prepare_city_dataset(
    paths: CityPaths,
    *,
    services: list[str],
    no_cache: bool,
    local_only: bool,
    require_ready_data: bool,
    include_stop_metrics: bool = True,
) -> tuple[gpd.GeoDataFrame, dict[str, str | int | float], bool]:
    prepared_dir = paths.output_dir / "prepared"
    raw_services_dir = paths.output_dir / "raw_services"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    raw_services_dir.mkdir(parents=True, exist_ok=True)

    final_path = prepared_dir / "blocks_experiment.parquet"
    summary_path = paths.output_dir / "summary.json"
    if final_path.exists() and summary_path.exists() and not no_cache:
        cached_blocks = read_geodata(final_path)
        cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        required_cached_cols = {
            "accessibility_time_mean_pt_intermodal",
            "accessibility_time_mean_pt_walk",
            *{f"assigned_service_time_{service}_{context}" for service in services for context in ("intermodal", "walk")},
            *{f"service_provision_{service}_{context}" for service in services for context in ("intermodal", "walk")},
            *{f"nearest_service_time_{service}_{context}" for service in services for context in ("intermodal", "walk")},
        }
        cached_access = pd.to_numeric(
            cached_blocks.get("accessibility_time_mean_pt_intermodal"),
            errors="coerce",
        )
        cached_walk_access = pd.to_numeric(
            cached_blocks.get("accessibility_time_mean_pt_walk"),
            errors="coerce",
        )
        cached_walk_has_inf = bool(np.isinf(cached_walk_access).any()) if cached_walk_access is not None else True
        if cached_walk_has_inf:
            for col in ("accessibility_time_mean_pt_walk", "accessibility_time_mean_pt_intermodal", "accessibility_time_mean_pt"):
                if col in cached_blocks.columns:
                    cached_blocks[col] = pd.to_numeric(cached_blocks[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            _log(f"[{paths.slug}] cached accessibility contains inf values; normalized to NaN and reusing cache.")
        _sanitize_numeric_inf_inplace(cached_blocks)
        assigned_cache_complete = True
        for service in services:
            capacity_col = f"service_capacity_{service}"
            if capacity_col not in cached_blocks.columns:
                continue
            service_capacity = pd.to_numeric(cached_blocks[capacity_col], errors="coerce").fillna(0.0)
            if float(service_capacity.sum()) <= 0.0:
                continue
            for context in ("intermodal", "walk"):
                assigned_col = f"assigned_service_time_{service}_{context}"
                provision_col = f"service_provision_{service}_{context}"
                assigned_values = pd.to_numeric(cached_blocks.get(assigned_col), errors="coerce")
                provision_values = pd.to_numeric(cached_blocks.get(provision_col), errors="coerce")
                if assigned_values is None or provision_values is None:
                    assigned_cache_complete = False
                    break
                if int(assigned_values.notna().sum()) <= 0 or int(provision_values.notna().sum()) <= 0:
                    assigned_cache_complete = False
                    break
            if not assigned_cache_complete:
                break
        stop_cache_complete = True
        if include_stop_metrics:
            for modality in _available_stop_modalities(paths):
                required_stop_cols = {f"stop_has_{modality}", f"stop_count_{modality}"}
                if not required_stop_cols.issubset(set(cached_blocks.columns)):
                    stop_cache_complete = False
                    break
        if (
            required_cached_cols.issubset(set(cached_blocks.columns))
            and cached_access is not None
            and int(cached_access.notna().sum()) > 0
            and assigned_cache_complete
            and stop_cache_complete
        ):
            if cached_walk_has_inf:
                _save_geodata(cached_blocks, final_path)
            return cached_blocks, cached_summary, True
        _log(f"[{paths.slug}] cached experiment dataset is incomplete for accessibility or assigned-service metrics, rebuilding prepared dataset.")

    blocks = _load_blocks(paths.blocks_path)
    boundary = _load_boundary(paths.boundary_path)
    street_cells = _load_street_cells(paths.street_cells_path)
    graph = _read_graph_pickle(paths.graph_path)

    blocks = _transfer_street_pattern_to_blocks(blocks, street_cells, prepared_dir, no_cache=no_cache)

    service_stats: dict[str, dict[str, object]] = {}
    for service in services:
        _log(f"[{paths.slug}] collecting service layer: {service}")
        raw, metrics = _collect_service_metrics(
            boundary=boundary,
            blocks=blocks,
            service=service,
            raw_dir=raw_services_dir,
            reuse_raw_path=paths.pipeline2_services_raw_dir / f"{service}.parquet",
            fallback_raw_path=paths.city_dir / "pipeline_2" / "services_raw" / f"{service}.parquet",
            fallback_buildings_path=paths.city_dir / "blocksnet_raw_osm" / "buildings.parquet",
            no_cache=no_cache,
            local_only=local_only,
            require_ready_data=require_ready_data,
        )
        for col in metrics.columns:
            blocks[col] = metrics[col].reindex(blocks.index).fillna(0.0).astype(float)
        service_stats[service] = {
            "raw_features": int(len(raw)),
            "blocks_with_service": int((blocks[f"service_has_{service}"] > 0).sum()),
            "capacity_total": float(blocks[f"service_capacity_{service}"].sum()),
        }

    stop_stats: dict[str, dict[str, object]] = {}
    if include_stop_metrics:
        stop_modalities = _available_stop_modalities(paths)
        for modality in stop_modalities:
            stops_path = paths.connectpt_dir / modality / "aggregated_stops.parquet"
            _log(f"[{paths.slug}] collecting stop metrics from connectpt: {modality}")
            stops, metrics = _collect_stop_metrics(
                blocks=blocks,
                stops_path=stops_path,
                modality=modality,
            )
            for col in metrics.columns:
                blocks[col] = metrics[col].reindex(blocks.index).fillna(0.0).astype(float)
            stop_stats[modality] = {
                "aggregated_stops": int(len(stops)),
                "blocks_with_stop": int((blocks[f"stop_has_{modality}"] > 0).sum()),
                "stop_count_total": float(blocks[f"stop_count_{modality}"].sum()),
            }

    walk_graph = _extract_walk_only_graph(graph)
    matrix_intermodal, access_series_intermodal, access_mask = _compute_accessibility(
        blocks=blocks,
        graph=graph,
        prepared_dir=prepared_dir,
        no_cache=no_cache,
        cache_name="accessibility_matrix_intermodal.parquet",
        reuse_matrix_path=paths.pipeline2_prepared_dir / "adj_matrix_time_min_union.parquet",
        require_ready_data=require_ready_data,
    )
    _, access_series_walk, _ = _compute_accessibility(
        blocks=blocks,
        graph=walk_graph,
        prepared_dir=prepared_dir,
        no_cache=no_cache,
        cache_name="accessibility_matrix_walk.parquet",
    )
    accessibility_mean_frame = (
        pd.DataFrame(
            {
                "block_id": blocks["block_id"].astype(str),
                "accessibility_time_mean_pt_intermodal": blocks["block_id"].astype(str).map(access_series_intermodal),
                "accessibility_time_mean_pt_walk": blocks["block_id"].astype(str).map(access_series_walk),
            }
        )
        .reset_index(drop=True)
    )
    blocks["block_id"] = blocks["block_id"].astype(str)
    blocks = blocks.drop(
        columns=[
            "accessibility_time_mean_pt",
            "accessibility_time_mean_pt_intermodal",
            "accessibility_time_mean_pt_walk",
        ],
        errors="ignore",
    ).merge(
        accessibility_mean_frame,
        on="block_id",
        how="left",
    )
    blocks["accessibility_time_mean_pt"] = blocks["accessibility_time_mean_pt_intermodal"]
    blocks["accessibility_block_selected"] = access_mask.reindex(blocks.index).fillna(False).astype(bool)

    service_accessibility_stats: dict[str, dict[str, object]] = {}
    for service in services:
        context_stats: dict[str, object] = {}
        for context_label, context_graph in (("intermodal", graph), ("walk", walk_graph)):
            _, service_metrics = _compute_service_assigned_accessibility(
                blocks=blocks,
                graph=context_graph,
                service=service,
                prepared_dir=prepared_dir,
                no_cache=no_cache,
                context_label=context_label,
                reuse_service_dir=(
                    paths.pipeline2_solver_inputs_dir / service
                    if context_label == "intermodal"
                    else None
                ),
                require_ready_data=bool(require_ready_data and context_label == "intermodal"),
            )
            for col in service_metrics.columns:
                blocks[col] = service_metrics[col].reindex(blocks.index)
            assigned_col = f"assigned_service_time_{service}_{context_label}"
            provision_col = f"service_provision_{service}_{context_label}"
            nearest_col = f"nearest_service_time_{service}_{context_label}"
            context_stats[context_label] = {
                "assigned_time_nonnull_blocks": int(pd.to_numeric(blocks[assigned_col], errors="coerce").notna().sum()),
                "assigned_time_median": float(pd.to_numeric(blocks[assigned_col], errors="coerce").dropna().median()) if pd.to_numeric(blocks[assigned_col], errors="coerce").notna().any() else np.nan,
                "provision_mean": float(pd.to_numeric(blocks[provision_col], errors="coerce").dropna().mean()) if pd.to_numeric(blocks[provision_col], errors="coerce").notna().any() else np.nan,
                "nearest_time_nonnull_blocks": int(pd.to_numeric(blocks[nearest_col], errors="coerce").notna().sum()) if nearest_col in blocks.columns else 0,
                "nearest_time_median": float(pd.to_numeric(blocks[nearest_col], errors="coerce").dropna().median()) if (nearest_col in blocks.columns and pd.to_numeric(blocks[nearest_col], errors="coerce").notna().any()) else np.nan,
            }
        service_accessibility_stats[service] = context_stats

    _sanitize_numeric_inf_inplace(blocks)
    _save_geodata(blocks, final_path)
    summary = {
        "city_slug": paths.slug,
        "blocks_total": int(len(blocks)),
        "blocks_with_population_positive": int(access_mask.sum()),
        "street_cells_total": int(len(street_cells)),
        "street_pattern_covered_blocks": int((blocks["street_pattern_covered_mass"] > 0).sum()),
        "services": service_stats,
        "stop_modalities": stop_stats,
        "service_assigned_accessibility": service_accessibility_stats,
        "files": {
            "blocks_experiment": str(final_path),
            "accessibility_matrix_intermodal": str(prepared_dir / "accessibility_matrix_intermodal.parquet"),
            "accessibility_matrix_walk": str(prepared_dir / "accessibility_matrix_walk.parquet"),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return blocks, summary, False


def _city_outputs_exist(paths: CityPaths) -> bool:
    summary_path = paths.output_dir / "summary.json"
    stats_dir = paths.output_dir / "stats"
    preview_dir = paths.output_dir / "preview_png"
    if not summary_path.exists() or not stats_dir.exists() or not preview_dir.exists():
        return False
    if not any(stats_dir.glob("*.csv")):
        return False
    if not any(preview_dir.glob("*.png")):
        return False
    return True


def _write_city_outputs(
    paths: CityPaths,
    *,
    blocks: gpd.GeoDataFrame,
    summary: dict[str, object],
    services: list[str],
    permutations: int,
) -> None:
    stats_dir = paths.output_dir / "stats"
    preview_dir = paths.output_dir / "preview_png"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    boundary = _load_boundary(paths.boundary_path)
    stop_modalities = sorted(
        col.removeprefix("stop_has_")
        for col in blocks.columns
        if col.startswith("stop_has_")
    )

    feature_cols = [
        f"street_pattern_prob_{label.lower().replace(' & ', '_').replace(' ', '_').replace('-', '_').replace('__', '_')}"
        for label in CLASS_LABELS.values()
        if f"street_pattern_prob_{label.lower().replace(' & ', '_').replace(' ', '_').replace('-', '_').replace('__', '_')}" in blocks.columns
    ]
    service_response_cols, stop_response_cols, accessibility_cols = _response_columns(services, stop_modalities)
    service_response_cols = [col for col in service_response_cols if col in blocks.columns]
    stop_response_cols = [col for col in stop_response_cols if col in blocks.columns]
    accessibility_cols = [col for col in accessibility_cols if col in blocks.columns]

    service_corr = _spearman_table(blocks, service_response_cols, feature_cols)
    service_uni, service_bv = _moran_tables(blocks, response_cols=service_response_cols, feature_cols=feature_cols, permutations=permutations)
    stop_corr = _spearman_table(blocks, stop_response_cols, feature_cols)
    stop_uni, stop_bv = _moran_tables(blocks, response_cols=stop_response_cols, feature_cols=feature_cols, permutations=permutations)

    access_blocks = blocks[blocks["accessibility_block_selected"]].copy()
    access_corr = _spearman_table(access_blocks, accessibility_cols, feature_cols)
    access_uni, access_bv = _moran_tables(access_blocks, response_cols=accessibility_cols, feature_cols=feature_cols, permutations=permutations)

    dominant_summary = _group_summary(blocks, services, stop_modalities)

    for name, frame in {
        "service_correlations.csv": service_corr,
        "service_spatial_moran.csv": service_uni,
        "service_spatial_bivariate_moran.csv": service_bv,
        "stop_correlations.csv": stop_corr,
        "stop_spatial_moran.csv": stop_uni,
        "stop_spatial_bivariate_moran.csv": stop_bv,
        "accessibility_correlations.csv": access_corr,
        "accessibility_spatial_moran.csv": access_uni,
        "accessibility_spatial_bivariate_moran.csv": access_bv,
        "dominant_class_summary.csv": dominant_summary,
    }.items():
        if frame.empty and len(frame.columns) == 0:
            frame = pd.DataFrame()
        (stats_dir / name).write_text(frame.to_csv(index=False), encoding="utf-8")

    _plot_dominant_class_map(blocks, boundary, preview_dir / "01_street_pattern_dominant_class.png")
    _plot_accessibility_map(blocks, boundary, preview_dir / "05_accessibility_mean_time_map.png")
    for service in services:
        for context_label in ("intermodal", "walk"):
            solver_blocks_path = (
                paths.output_dir
                / "prepared"
                / "service_accessibility"
                / context_label
                / service
                / "blocks_solver.parquet"
            )
            if not solver_blocks_path.exists():
                continue
            try:
                solver_blocks = read_geodata(solver_blocks_path)
            except Exception:
                continue
            _plot_service_lp_preview(
                solver_blocks,
                service,
                preview_dir / f"lp_{service}_provision_unmet_{context_label}.png",
                blocks_ref=blocks,
                boundary=boundary,
            )
    grouped = dominant_summary.set_index("street_pattern_dominant_class")
    _plot_grouped_metric_bars(
        grouped,
        preview_dir / "02_service_presence_by_dominant_class.png",
        value_kind="service_has",
        title="Service Presence By Dominant Street Pattern Class",
        ylabel="share of blocks with service",
    )
    _plot_grouped_metric_bars(
        grouped,
        preview_dir / "03_service_capacity_by_dominant_class.png",
        value_kind="service_capacity",
        title="Service Capacity By Dominant Street Pattern Class",
        ylabel="mean capacity",
    )
    _plot_grouped_metric_bars(
        grouped,
        preview_dir / "03b_stop_presence_by_dominant_class.png",
        value_kind="stop_has",
        title="Stop Presence By Dominant Street Pattern Class",
        ylabel="share of blocks with stop",
    )
    _plot_grouped_metric_bars(
        grouped,
        preview_dir / "03c_stop_count_by_dominant_class.png",
        value_kind="stop_count",
        title="Stop Count By Dominant Street Pattern Class",
        ylabel="mean stop count",
    )
    _plot_accessibility_by_class(access_blocks, preview_dir / "04_accessibility_by_dominant_class.png")

    if not service_corr.empty:
        service_heat = (
            service_corr.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="spearman_rho")
            .sort_index(axis=0)
        )
        service_p_heat = (
            service_corr.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="p_value")
            .sort_index(axis=0)
        )
        _heatmap(
            service_heat,
            preview_dir / "06_service_spearman_heatmap.png",
            title="Service vs Street Pattern Probabilities (Spearman)",
            significance_table=service_p_heat,
        )

    if not service_bv.empty:
        service_bv_heat = (
            service_bv.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="moran_bv_I")
            .sort_index(axis=0)
        )
        service_bv_p_heat = (
            service_bv.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="p_sim")
            .sort_index(axis=0)
        )
        _heatmap(
            service_bv_heat,
            preview_dir / "07_service_bivariate_moran_heatmap.png",
            title="Service vs Street Pattern Probabilities (Bivariate Moran)",
            significance_table=service_bv_p_heat,
        )
        best_service_features = (
            service_bv.assign(abs_moran=lambda df: df["moran_bv_I"].abs())
            .sort_values(["response", "abs_moran"], ascending=[True, False])
            .drop_duplicates(subset=["response"])
        )
        for row in best_service_features.itertuples(index=False):
            if (
                pd.notna(row.p_sim)
                and float(row.p_sim) < 0.05
                and pd.notna(row.moran_bv_I)
                and float(row.moran_bv_I) >= LOCAL_MORAN_MIN_GLOBAL_I
            ):
                _plot_local_bivariate_moran_map(
                    blocks,
                    response_col=str(row.response),
                    feature_col=str(row.street_pattern_feature),
                    boundary=boundary,
                    permutations=permutations,
                    output_path=preview_dir / f"moran_local_{row.response}.png",
                )

    if not service_uni.empty:
        _global_moran_heatmap(
            service_uni,
            preview_dir / "06b_service_global_moran_heatmap.png",
            title="Service Global Moran I",
        )
        for row in service_uni.itertuples(index=False):
            if (
                pd.notna(row.p_sim)
                and float(row.p_sim) < 0.05
                and pd.notna(row.moran_I)
                and float(row.moran_I) >= LOCAL_MORAN_MIN_GLOBAL_I
            ):
                _plot_local_lisa_map(
                    blocks,
                    response_col=str(row.response),
                    boundary=boundary,
                    permutations=permutations,
                    output_path=preview_dir / f"lisa_local_{row.response}.png",
                )

    if not stop_corr.empty:
        stop_heat = (
            stop_corr.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="spearman_rho")
            .sort_index(axis=0)
        )
        stop_p_heat = (
            stop_corr.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="p_value")
            .sort_index(axis=0)
        )
        _heatmap(
            stop_heat,
            preview_dir / "07b_stop_spearman_heatmap.png",
            title="Stops vs Street Pattern Probabilities (Spearman)",
            significance_table=stop_p_heat,
        )

    if not stop_bv.empty:
        stop_bv_heat = (
            stop_bv.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="moran_bv_I")
            .sort_index(axis=0)
        )
        stop_bv_p_heat = (
            stop_bv.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="p_sim")
            .sort_index(axis=0)
        )
        _heatmap(
            stop_bv_heat,
            preview_dir / "07c_stop_bivariate_moran_heatmap.png",
            title="Stops vs Street Pattern Probabilities (Bivariate Moran)",
            significance_table=stop_bv_p_heat,
        )
        best_stop_features = (
            stop_bv.assign(abs_moran=lambda df: df["moran_bv_I"].abs())
            .sort_values(["response", "abs_moran"], ascending=[True, False])
            .drop_duplicates(subset=["response"])
        )
        for row in best_stop_features.itertuples(index=False):
            if (
                pd.notna(row.p_sim)
                and float(row.p_sim) < 0.05
                and pd.notna(row.moran_bv_I)
                and float(row.moran_bv_I) >= LOCAL_MORAN_MIN_GLOBAL_I
            ):
                _plot_local_bivariate_moran_map(
                    blocks,
                    response_col=str(row.response),
                    feature_col=str(row.street_pattern_feature),
                    boundary=boundary,
                    permutations=permutations,
                    output_path=preview_dir / f"moran_local_{row.response}.png",
                )

    if not stop_uni.empty:
        _global_moran_heatmap(
            stop_uni,
            preview_dir / "07d_stop_global_moran_heatmap.png",
            title="Stops Global Moran I",
        )
        for row in stop_uni.itertuples(index=False):
            if (
                pd.notna(row.p_sim)
                and float(row.p_sim) < 0.05
                and pd.notna(row.moran_I)
                and float(row.moran_I) >= LOCAL_MORAN_MIN_GLOBAL_I
            ):
                _plot_local_lisa_map(
                    blocks,
                    response_col=str(row.response),
                    boundary=boundary,
                    permutations=permutations,
                    output_path=preview_dir / f"lisa_local_{row.response}.png",
                )

    if not access_corr.empty:
        access_heat = (
            access_corr.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="spearman_rho")
        )
        access_p_heat = (
            access_corr.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="p_value")
        )
        _heatmap(
            access_heat,
            preview_dir / "08_accessibility_spearman_heatmap.png",
            title="Accessibility vs Street Pattern Probabilities (Spearman)",
            significance_table=access_p_heat,
        )

    if not access_bv.empty:
        access_bv_heat = (
            access_bv.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="moran_bv_I")
        )
        access_bv_p_heat = (
            access_bv.assign(
                street_pattern_feature=lambda df: df["street_pattern_feature"].map(_friendly_feature_name),
            )
            .pivot(index="response", columns="street_pattern_feature", values="p_sim")
        )
        _heatmap(
            access_bv_heat,
            preview_dir / "09_accessibility_bivariate_moran_heatmap.png",
            title="Accessibility vs Street Pattern Probabilities (Bivariate Moran)",
            significance_table=access_bv_p_heat,
        )
        best_access_features = (
            access_bv.assign(abs_moran=lambda df: df["moran_bv_I"].abs())
            .sort_values(["response", "abs_moran"], ascending=[True, False])
            .drop_duplicates(subset=["response"])
        )
        for row in best_access_features.itertuples(index=False):
            if (
                pd.notna(row.p_sim)
                and float(row.p_sim) < 0.05
                and pd.notna(row.moran_bv_I)
                and float(row.moran_bv_I) >= LOCAL_MORAN_MIN_GLOBAL_I
            ):
                _plot_local_bivariate_moran_map(
                    access_blocks,
                    response_col=str(row.response),
                    feature_col=str(row.street_pattern_feature),
                    boundary=boundary,
                    permutations=permutations,
                    output_path=preview_dir / f"moran_local_{row.response}.png",
                )

    if not access_uni.empty:
        _global_moran_heatmap(
            access_uni,
            preview_dir / "08b_accessibility_global_moran_heatmap.png",
            title="Accessibility Global Moran I",
        )
        for row in access_uni.itertuples(index=False):
            if (
                pd.notna(row.p_sim)
                and float(row.p_sim) < 0.05
                and pd.notna(row.moran_I)
                and float(row.moran_I) >= LOCAL_MORAN_MIN_GLOBAL_I
            ):
                _plot_local_lisa_map(
                    access_blocks,
                    response_col=str(row.response),
                    boundary=boundary,
                    permutations=permutations,
                    output_path=preview_dir / f"lisa_local_{row.response}.png",
                )

    summary["stats_files"] = {
        "service_correlations": str(stats_dir / "service_correlations.csv"),
        "service_spatial_bivariate_moran": str(stats_dir / "service_spatial_bivariate_moran.csv"),
        "stop_correlations": str(stats_dir / "stop_correlations.csv"),
        "stop_spatial_bivariate_moran": str(stats_dir / "stop_spatial_bivariate_moran.csv"),
        "accessibility_correlations": str(stats_dir / "accessibility_correlations.csv"),
        "accessibility_spatial_bivariate_moran": str(stats_dir / "accessibility_spatial_bivariate_moran.csv"),
        "dominant_class_summary": str(stats_dir / "dominant_class_summary.csv"),
    }
    summary["preview_dir"] = str(preview_dir)
    (paths.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _configure_osmnx(timeout_s: int) -> None:
    ox.settings.timeout = int(timeout_s)
    ox.settings.use_cache = True
    ox.settings.log_console = False


def main() -> None:
    args = parse_args()
    _configure_logging()
    _configure_osmnx(args.osmnx_timeout_s)

    joint_input_root = Path(args.joint_input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    city_list = list(args.cities)
    if len(city_list) == 1 and city_list[0].lower() == "all":
        city_list = _discover_available_city_slugs(
            joint_input_root,
            output_root,
            use_sm_imputed=bool(args.use_sm_imputed),
        )

    _log(
        "Running service/accessibility vs street-pattern experiments for "
        f"{', '.join(city_list)}"
    )

    city_blocks: dict[str, gpd.GeoDataFrame] = {}
    city_paths_map: dict[str, CityPaths] = {}
    city_summaries: dict[str, dict[str, object]] = {}
    city_coverage_rows: list[dict[str, object]] = []
    for slug in city_list:
        paths = _resolve_city_paths(slug, joint_input_root, output_root, use_sm_imputed=bool(args.use_sm_imputed))
        _log(f"[{slug}] preparing experiment dataset from loaded territory.")
        blocks, summary, used_cached_dataset = _prepare_city_dataset(
            paths,
            services=list(args.services),
            no_cache=bool(args.no_cache),
            local_only=bool(args.local_only),
            require_ready_data=bool(args.require_ready_data),
            include_stop_metrics=True,
        )
        if not args.cross_city_only:
            if used_cached_dataset and (not bool(args.no_cache)) and _city_outputs_exist(paths):
                _log(f"[{slug}] cached dataset and city outputs already exist; skipping stats/preview rewrite.")
            else:
                _log(f"[{slug}] writing stats and previews.")
                _write_city_outputs(
                    paths,
                    blocks=blocks,
                    summary=summary,
                    services=list(args.services),
                    permutations=int(args.permutations),
                )
        city_blocks[slug] = blocks
        city_paths_map[slug] = paths
        city_summaries[slug] = summary
        if args.cross_city_min_coverage_pct is not None:
            try:
                coverage_row = _compute_city_coverage_pct(paths, roads_buffer_m=float(args.cross_city_roads_buffer_m))
            except Exception as exc:
                coverage_row = {
                    "city_slug": slug,
                    "coverage_pct": np.nan,
                    "coverage_error": str(exc),
                    "territory_area_m2": np.nan,
                    "buildings_coverage_area_m2": np.nan,
                    "roads_coverage_area_m2": np.nan,
                    "combined_coverage_area_m2": np.nan,
                    "buildings_count": np.nan,
                    "roads_count": np.nan,
                }
            city_coverage_rows.append(coverage_row)
        _log(f"[{slug}] done: {paths.output_dir}")

    cross_city_blocks = city_blocks
    cross_city_paths = city_paths_map
    cross_city_summaries = city_summaries
    _log(f"Cross-city input pool: {len(city_blocks)} city(ies).")
    if args.cross_city_min_coverage_pct is not None:
        coverage_df = pd.DataFrame(city_coverage_rows)
        if not coverage_df.empty:
            coverage_df = coverage_df.sort_values("coverage_pct", na_position="first").reset_index(drop=True)
        stats_dir = output_root / "_cross_city" / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        coverage_report = stats_dir / "city_coverage_screening.csv"
        coverage_df.to_csv(coverage_report, index=False)
        threshold = float(args.cross_city_min_coverage_pct)
        eligible = set(
            coverage_df.loc[pd.to_numeric(coverage_df.get("coverage_pct"), errors="coerce") >= threshold, "city_slug"].astype(str)
            if not coverage_df.empty else []
        )
        excluded = sorted(set(city_blocks.keys()) - eligible)
        _log(
            f"Cross-city coverage screening threshold={threshold:.2f}% "
            f"(roads_buffer_m={float(args.cross_city_roads_buffer_m):.1f}): "
            f"passed={len(eligible)}, excluded={len(excluded)}."
        )
        if excluded:
            _warn(
                f"Cross-city coverage screening excluded {len(excluded)} city(ies) below "
                f"{threshold:.2f}% combined coverage: {', '.join(excluded)}"
            )
        else:
            _log(f"Cross-city coverage screening: all cities passed threshold {threshold:.2f}%.")
        cross_city_blocks = {k: v for k, v in city_blocks.items() if k in eligible}
        cross_city_paths = {k: v for k, v in city_paths_map.items() if k in eligible}
        cross_city_summaries = {k: v for k, v in city_summaries.items() if k in eligible}

    if len(cross_city_blocks) >= 2:
        selected_city_slugs = sorted(cross_city_blocks.keys())
        _log(
            "Writing cross-city pilot outputs for "
            f"{len(selected_city_slugs)} city(ies): {', '.join(selected_city_slugs)}"
        )
        _write_cross_city_outputs(
            city_blocks=cross_city_blocks,
            city_paths=cross_city_paths,
            city_summaries=cross_city_summaries,
            output_root=output_root,
            services=list(args.services),
            accessibility_only=bool(args.cross_city_only),
        )
        _log(f"Cross-city pilot outputs ready: {output_root / '_cross_city'}")
    else:
        _warn(
            "Cross-city outputs skipped: fewer than 2 eligible cities after screening "
            f"(eligible={len(cross_city_blocks)})."
        )


if __name__ == "__main__":
    main()

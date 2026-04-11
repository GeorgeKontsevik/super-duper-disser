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
from aggregated_spatial_pipeline.runtime_config import configure_logger, ensure_repo_mplconfigdir
from aggregated_spatial_pipeline.visualization import (
    apply_preview_canvas,
    footer_text,
    legend_bottom,
    normalize_preview_gdf,
    save_preview_figure,
)


SUPPORTED_SERVICES = ("hospital", "polyclinic", "school", "kindergarten")
LP_BLOCK_SELECTION_POLICY = "has_living_buildings_or_service_capacity"
PROVISION_ENGINE_NAME = "arctic_lp_provision"
PLACEMENT_ENGINE_NAME = "solver_flp_optimize_placement"

# Keep matplotlib/tqdm ecosystem caches in writable workspace path.
ensure_repo_mplconfigdir("mpl-asp-pipeline2")

@dataclass(frozen=True)
class ServiceSpec:
    tags: list[dict]
    fallback_capacity: float = 600.0


SERVICE_SPECS: dict[str, ServiceSpec] = {
    "hospital": ServiceSpec(
        tags=[
            {"amenity": "hospital"},
            {"healthcare": "hospital"},
        ],
    ),
    "polyclinic": ServiceSpec(
        tags=[
            {"amenity": "clinic"},
            {"healthcare": ["clinic", "centre"]},
        ],
    ),
    "school": ServiceSpec(
        tags=[
            {"amenity": "school"},
        ],
    ),
    "kindergarten": ServiceSpec(
        tags=[
            {"amenity": "kindergarten"},
        ],
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
DEFAULT_ACCESSIBILITY_MIN_BY_SERVICE = {
    "hospital": 60.0,
    "polyclinic": 10.0,
    "school": 15.0,
    "kindergarten": 15.0,
}
DEFAULT_DEMAND_PER_1000_BY_SERVICE = {
    "hospital": 9.0,
    "polyclinic": 13.0,
    "school": 120.0,
    "kindergarten": 120.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline_2 input preparation on top of pipeline_1 territory: "
            "load shared raw services, aggregate to blocks, compute accessibility matrix, "
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
        help="Reserved option (kept for CLI compatibility; ignored in arctic lp_coverage mode).",
    )
    parser.add_argument(
        "--capacity-default-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use fallback capacity defaults when OSM objects miss explicit capacity/beds "
            "(default: enabled). Set --no-capacity-default-fallback for strict fail-fast."
        ),
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
        "--placement-exact",
        action="store_true",
        help="Run exact service placement after solver-input preparation and save after-placement outputs/previews.",
    )
    parser.add_argument(
        "--placement-genetic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use genetic search inside solver_flp optimize_placement for exact placement "
            "(default: disabled / deterministic non-genetic path)."
        ),
    )
    parser.add_argument(
        "--placement-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show solver_flp tqdm progress for exact placement (default: enabled).",
    )
    parser.add_argument(
        "--placement-prefer-existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer existing facilities in placement objective (default: disabled).",
    )
    parser.add_argument(
        "--placement-allow-existing-expansion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow expanding existing capacities during placement (default: disabled).",
    )
    parser.add_argument(
        "--placement-capacity-mode",
        choices=("service_min", "fixed_mean"),
        default="fixed_mean",
        help=(
            "Capacity policy for newly opened facilities: "
            "'fixed_mean' (default, constant rounded mean of existing capacities) or "
            "'service_min' (use service-specific minimum capacity floor)."
        ),
    )
    parser.add_argument(
        "--placement-genetic-population-size",
        type=int,
        default=50,
        help="Population size for genetic placement mode (default: 50).",
    )
    parser.add_argument(
        "--placement-genetic-generations",
        type=int,
        default=20,
        help="Number of generations for genetic placement mode (default: 20).",
    )
    parser.add_argument(
        "--placement-genetic-mutation-rate",
        type=float,
        default=0.7,
        help="Mutation rate for genetic placement mode (default: 0.7).",
    )
    parser.add_argument(
        "--placement-genetic-num-parents",
        type=int,
        default=10,
        help="Number of parents for genetic placement mode (default: 10).",
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


DEFAULT_POPULATION_SHARE_BY_SERVICE = {
    "hospital": 1.0,
    "polyclinic": 1.0,
    "school": 1.0,
}
DEFAULT_MIN_NEW_CAPACITY_BY_SERVICE = {
    "hospital": 50.0,
    "polyclinic": 50.0,
    "school": 1500.0,
}
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
    configure_logger("[pipeline_2_prepare]")


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


def _read_blocks(blocks_path: Path) -> gpd.GeoDataFrame:
    blocks = read_geodata(blocks_path)
    if blocks.empty:
        raise ValueError(f"Blocks layer is empty: {blocks_path}")
    return blocks


def _derive_has_living_from_buildings(units: gpd.GeoDataFrame, city_dir: Path) -> pd.Series:
    flags = pd.Series(False, index=units.index, dtype=bool)
    buildings_path = city_dir / "derived_layers" / "buildings_floor_enriched.parquet"
    if not buildings_path.exists():
        raise FileNotFoundError(
            "Missing required buildings layer for strict living-mask derivation: "
            f"{buildings_path}"
        )
    try:
        buildings = read_geodata(buildings_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not read required buildings_floor_enriched: {exc}") from exc
    if buildings.empty or "is_living" not in buildings.columns:
        raise ValueError(
            "Required column is_living is missing in buildings_floor_enriched.parquet; "
            "strict mode does not allow fallback proxies."
        )

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


def _calculate_accessibility_matrix_native(
    units: gpd.GeoDataFrame,
    graph: nx.Graph,
    *,
    weight_key: str = "time_min",
) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    work = units[["geometry"]].copy()
    if work.crs is None:
        work = work.set_crs(4326)
    graph_crs = graph.graph.get("crs")
    if graph_crs is None:
        raise ValueError("Graph CRS is missing for native accessibility matrix build.")
    work = work.to_crs(graph_crs)
    pts = work.geometry.representative_point()
    node_ids = ox.distance.nearest_nodes(graph, X=pts.x.to_numpy(), Y=pts.y.to_numpy())
    idx = list(units.index)
    node_by_unit = {unit_idx: node_ids[pos] for pos, unit_idx in enumerate(idx)}
    unique_nodes = sorted(set(node_ids))
    lengths_by_source: dict = {}
    for source in unique_nodes:
        lengths_by_source[source] = nx.single_source_dijkstra_path_length(
            graph,
            source,
            weight=weight_key,
        )

    matrix = pd.DataFrame(np.inf, index=idx, columns=idx, dtype=float)
    np.fill_diagonal(matrix.values, 0.0)
    for src_unit in idx:
        src_node = node_by_unit[src_unit]
        lengths = lengths_by_source[src_node]
        for dst_unit in idx:
            dst_node = node_by_unit[dst_unit]
            distance = lengths.get(dst_node)
            if distance is not None:
                matrix.loc[src_unit, dst_unit] = float(distance)
    return matrix


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


def _match_tag_value(value, expected_values: set[str]) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_match_tag_value(v, expected_values) for v in value)
    text = str(value).strip().lower()
    if not text:
        return False
    if text in expected_values:
        return True
    for token in re.split(r"[;,\|]", text):
        if token.strip().lower() in expected_values:
            return True
    return False


def _read_common_services_raw(city_dir: Path) -> tuple[gpd.GeoDataFrame, Path]:
    common_path = city_dir / "blocksnet_raw_osm" / "services_pipeline2_raw.parquet"
    if not common_path.exists():
        raise ValueError(
            "Common raw services layer is missing. "
            f"Expected: {common_path}. Rebuild shared collection (run run_joint with --no-cache)."
        )
    common_raw = read_geodata(common_path)
    if common_raw.crs is None:
        common_raw = common_raw.set_crs(4326)
    if "source_uid" not in common_raw.columns and not common_raw.empty:
        common_raw = _normalize_raw_osm(common_raw)
    return common_raw, common_path


def _extract_service_raw_from_common(common_raw: gpd.GeoDataFrame, service: str) -> gpd.GeoDataFrame:
    if common_raw.empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=common_raw.crs)

    queries = SERVICE_SPECS[service].tags
    mask_total = pd.Series(False, index=common_raw.index, dtype=bool)
    for tags in queries:
        qmask = pd.Series(True, index=common_raw.index, dtype=bool)
        for tag_key, expected in tags.items():
            expected_values = expected if isinstance(expected, list) else [expected]
            expected_set = {str(v).strip().lower() for v in expected_values}
            if tag_key not in common_raw.columns:
                qmask = pd.Series(False, index=common_raw.index, dtype=bool)
                break
            qmask = qmask & common_raw[tag_key].map(lambda val: _match_tag_value(val, expected_set))
        mask_total = mask_total | qmask

    filtered = common_raw.loc[mask_total].copy()
    if "source_uid" in filtered.columns:
        filtered = filtered.drop_duplicates(subset=["source_uid"]).reset_index(drop=True)
    else:
        filtered = filtered.reset_index(drop=True)
    filtered = _filter_non_private_kindergarten(filtered, service)
    return filtered


def _filter_non_private_kindergarten(raw: gpd.GeoDataFrame, service: str) -> gpd.GeoDataFrame:
    if service != "kindergarten" or raw.empty:
        return raw
    private_cols = ("operator:type", "access", "ownership")
    private_mask = pd.Series(False, index=raw.index, dtype=bool)
    for col in private_cols:
        if col not in raw.columns:
            continue
        values = raw[col].astype("string").str.lower()
        private_mask = private_mask | values.str.contains(r"(^|[;,\s])private($|[;,\s])", regex=True, na=False)
    removed = int(private_mask.sum())
    if removed > 0:
        _log(f"Kindergarten filter: removed private features={removed}")
    return raw.loc[~private_mask].copy()


def _capacity_from_row(
    row: pd.Series,
    service: str,
    *,
    allow_default_fallback: bool,
) -> float:
    if service == "hospital":
        beds = _first_number(row.get("beds"))
        if beds is not None and beds > 0:
            return beds
    cap = _first_number(row.get("capacity"))
    if cap is not None and cap > 0:
        return cap
    if allow_default_fallback:
        return float(DEFAULT_CAPACITY_BY_SERVICE[service])
    source_uid = str(row.get("source_uid", "<unknown>"))
    raise ValueError(
        f"Missing explicit capacity for service [{service}] at OSM object {source_uid}. "
        "Fail-fast mode is active (no fallback defaults)."
    )


def _service_accessibility_min(service: str, args: argparse.Namespace) -> float:
    if args.service_radius_min is not None:
        return float(args.service_radius_min)
    return float(DEFAULT_ACCESSIBILITY_MIN_BY_SERVICE.get(service, 60.0))


def _service_demand_per_1000(service: str, args: argparse.Namespace) -> float:
    if args.demand_per_1000 is not None:
        return float(args.demand_per_1000)
    return float(DEFAULT_DEMAND_PER_1000_BY_SERVICE.get(service, ARCTIC_DEFAULT_POPULATION))


def _service_population_share(service: str) -> float:
    return float(DEFAULT_POPULATION_SHARE_BY_SERVICE.get(service, 1.0))


def _service_min_new_capacity(service: str) -> float:
    return float(DEFAULT_MIN_NEW_CAPACITY_BY_SERVICE.get(service, 50.0))


def _service_fixed_mean_capacity(service: str, solver_blocks: pd.DataFrame | gpd.GeoDataFrame) -> float:
    capacities = pd.to_numeric(solver_blocks.get("capacity", 0.0), errors="coerce").fillna(0.0)
    positive = capacities[capacities > 0.0]
    if positive.empty:
        return float(_service_min_new_capacity(service))
    return float(max(1, int(round(float(positive.mean())))))


def _to_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = gdf.copy()
    points["geometry"] = points.geometry.representative_point()
    return points


def _aggregate_capacity_to_blocks(service_points: gpd.GeoDataFrame, blocks_gdf: gpd.GeoDataFrame) -> pd.Series:
    if service_points.empty:
        return pd.Series(0.0, index=blocks_gdf.index, dtype=float)
    points = service_points.to_crs(blocks_gdf.crs)
    blocks_geom = blocks_gdf[["geometry"]].copy()
    joined = gpd.sjoin(points[["capacity_est", "geometry"]], blocks_geom, how="inner", predicate="intersects")
    if joined.empty:
        return pd.Series(0.0, index=blocks_gdf.index, dtype=float)
    by_block = joined.groupby("index_right")["capacity_est"].sum()
    return by_block.reindex(blocks_gdf.index).fillna(0.0).astype(float)


def _detect_population_column(blocks: gpd.GeoDataFrame) -> str | None:
    candidates = [col for col in ("population_total", "population_proxy", "population") if col in blocks.columns]
    if not candidates:
        return None
    # Prefer the column with the best non-zero coverage to avoid stale zero-filled fields.
    best_col = None
    best_score = (-1, -1.0)  # (positive_count, total_sum)
    for col in candidates:
        series = pd.to_numeric(blocks[col], errors="coerce").fillna(0.0)
        positive_count = int((series > 0).sum())
        total_sum = float(series.sum())
        score = (positive_count, total_sum)
        if score > best_score:
            best_col = col
            best_score = score
    if best_col is not None:
        return best_col
    return None


def _load_arctic_calculate_provision():
    repo_root = Path(__file__).resolve().parents[2]
    arctic_path = repo_root / "arctic_access"
    if str(arctic_path) not in sys.path:
        sys.path.insert(0, str(arctic_path))
    from scripts.model.model import calculate_provision  # type: ignore

    return calculate_provision


def _build_city_graph_dict_from_matrix(matrix: pd.DataFrame) -> dict:
    graph: dict = {}
    for source in matrix.index:
        source_key = source.item() if hasattr(source, "item") else source
        edges = {}
        for target in matrix.columns:
            if source == target:
                continue
            weight = pd.to_numeric(matrix.loc[source, target], errors="coerce")
            if pd.isna(weight):
                continue
            target_key = target.item() if hasattr(target, "item") else target
            edges[target_key] = {"weight": float(weight)}
        graph[source_key] = edges
    return graph


def _assignment_matrix_to_links(assignments: pd.DataFrame) -> pd.DataFrame:
    if assignments is None or assignments.empty:
        return pd.DataFrame(columns=["source", "target", "value"])
    links: list[dict] = []
    for source in assignments.index:
        row = assignments.loc[source]
        for target, value in row.items():
            flow = pd.to_numeric(value, errors="coerce")
            if pd.isna(flow) or float(flow) <= 0.0:
                continue
            links.append(
                {
                    "source": source.item() if hasattr(source, "item") else source,
                    "target": target.item() if hasattr(target, "item") else target,
                    "value": float(flow),
                }
            )
    return pd.DataFrame(links, columns=["source", "target", "value"])


def _run_arctic_lp_provision(
    blocks_df: gpd.GeoDataFrame,
    accessibility_matrix: pd.DataFrame,
    *,
    service: str,
    service_radius_min: float,
    service_demand_per_1000: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    def _safe_float(value) -> float:
        parsed = pd.to_numeric(value, errors="coerce")
        if pd.isna(parsed):
            return 0.0
        return float(parsed)

    calculate_provision = _load_arctic_calculate_provision()
    work = blocks_df.copy()
    if "name" in work.columns:
        work["name"] = work["name"].astype(str)
        work = work.set_index("name", drop=False)
    else:
        work["name"] = work.index.astype(str)
        work = work.set_index("name", drop=False)

    if "demand" not in work.columns:
        work["demand"] = 0.0
    if "capacity" not in work.columns:
        work["capacity"] = 0.0
    work["demand"] = pd.to_numeric(work["demand"], errors="coerce").fillna(0.0).astype(float)
    work["capacity"] = pd.to_numeric(work["capacity"], errors="coerce").fillna(0.0).astype(float)
    if "population" not in work.columns:
        work["population"] = 0.0
    work["population"] = pd.to_numeric(work["population"], errors="coerce").fillna(0.0).astype(float)

    if work.crs is None:
        work = work.set_crs(4326)
    epsg = _to_epsg_int(work.crs)
    matrix = accessibility_matrix.copy()
    matrix.index = work.index
    matrix.columns = work.index
    service_capacity_col = f"capacity_{service}"
    city_model = {
        "epsg": epsg,
        "blocks": [
            {
                "id": str(name),
                "name": str(name),
                "geometry": row.geometry,
                "population": _safe_float(row.get("population")),
                "demand": _safe_float(row.get("demand")),
                service_capacity_col: _safe_float(row.get("capacity")),
            }
            for name, row in work.iterrows()
        ],
        "graph": _build_city_graph_dict_from_matrix(matrix),
        "service_types": {
            service: {
                "accessibility": float(service_radius_min),
                "demand": float(service_demand_per_1000),
            }
        },
    }
    provision_df, assignments = calculate_provision(
        city_model=city_model,
        service_type=service,
        method="lp",
    )
    provision_df = provision_df.copy()
    provision_df.index = provision_df.index.astype(str)
    work = work.copy()
    work.index = work.index.astype(str)
    for col in ("demand_within", "demand_without", "capacity_left", "provision"):
        if col in provision_df.columns:
            work[col] = pd.to_numeric(provision_df[col], errors="coerce").fillna(0.0)
        else:
            work[col] = 0.0
    work["demand_left"] = (
        pd.to_numeric(work["demand"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["demand_within"], errors="coerce").fillna(0.0)
        - pd.to_numeric(work["demand_without"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    work["capacity_within"] = 0.0
    work["capacity_without"] = 0.0
    links_df = _assignment_matrix_to_links(assignments)
    result = gpd.GeoDataFrame(work, geometry="geometry", crs=work.crs)
    return result, links_df


def _save_geodata(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prepare_geodata_for_parquet(gdf).to_parquet(path)


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


PIPELINE2_GALLERY_FILENAMES = {
    "services_raw_all": "29_services_raw_all_categories.png",
    "accessibility_mean_time_map": "30_accessibility_mean_time_map.png",
    "hospital": "31_lp_hospital_provision_unmet.png",
    "polyclinic": "32_lp_polyclinic_provision_unmet.png",
    "school": "33_lp_school_provision_unmet.png",
    "kindergarten": "40_lp_kindergarten_provision_unmet.png",
}

PIPELINE2_SELECTION_GALLERY_FILENAMES = {
    "accessibility": "accessibility_block_selection_status.png",
    "hospital": "lp_hospital_block_selection_status.png",
    "polyclinic": "lp_polyclinic_block_selection_status.png",
    "school": "lp_school_block_selection_status.png",
    "kindergarten": "lp_kindergarten_block_selection_status.png",
}

PIPELINE2_PLACEMENT_GALLERY_FILENAMES = {
    "hospital": {
        "status": "34_exact_hospital_placement_status.png",
        "after": "35_exact_hospital_provision_after.png",
    },
    "polyclinic": {
        "status": "36_exact_polyclinic_placement_status.png",
        "after": "37_exact_polyclinic_provision_after.png",
    },
    "school": {
        "status": "38_exact_school_placement_status.png",
        "after": "39_exact_school_provision_after.png",
    },
    "kindergarten": {
        "status": "41_exact_kindergarten_placement_status.png",
        "after": "42_exact_kindergarten_provision_after.png",
    },
}


def _plot_services_raw_overview(
    raw_service_points: dict[str, gpd.GeoDataFrame],
    out_path: Path,
    *,
    boundary: gpd.GeoDataFrame | None = None,
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not raw_service_points:
        return None

    service_colors = {
        "hospital": "#dc2626",
        "polyclinic": "#2563eb",
        "school": "#16a34a",
        "kindergarten": "#f59e0b",
    }
    boundary_plot = normalize_preview_gdf(boundary, target_crs="EPSG:3857")
    fig, ax = plt.subplots(figsize=(12, 10))
    apply_preview_canvas(fig, ax, boundary_plot, title="Pipeline_2 Raw Services (all categories)")
    legend_handles: list[Line2D] = []

    for service in SUPPORTED_SERVICES:
        raw = raw_service_points.get(service)
        if raw is None or raw.empty:
            continue
        points = _to_points(raw)
        if points.empty:
            continue
        points_plot = normalize_preview_gdf(points[["geometry"]], boundary_plot, target_crs="EPSG:3857")
        if points_plot is None or points_plot.empty:
            continue
        color = service_colors.get(service, "#334155")
        points_plot.plot(ax=ax, color=color, markersize=12, alpha=0.9)
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=7, label=service)
        )

    if not legend_handles:
        plt.close(fig)
        return None

    legend_bottom(ax, legend_handles, max_cols=4, fontsize=10)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview_figure(fig, out_path)
    plt.close(fig)
    _log(f"Preview step: saved raw-services overview map: {out_path.name}")
    return str(out_path)


def _calc_demand(population: pd.Series, demand_per_1000: float) -> pd.Series:
    scaled = np.ceil((population.fillna(0.0).astype(float) / 1000.0) * float(demand_per_1000))
    return scaled.astype(int)


def _calc_service_demand(population: pd.Series, service: str, demand_per_1000: float) -> pd.Series:
    effective_population = population.fillna(0.0).astype(float) * _service_population_share(service)
    return _calc_demand(effective_population, demand_per_1000)


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

    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    access_map_path = out_dir / PIPELINE2_GALLERY_FILENAMES["accessibility_mean_time_map"]
    if use_cache and access_map_path.exists():
        _log(f"Preview step: using cached accessibility map: {access_map_path.name}")
        out["accessibility_mean_time_map"] = str(access_map_path)
        return out

    boundary_plot = normalize_preview_gdf(boundary, target_crs="EPSG:3857")

    matrix_numeric = matrix_union.apply(pd.to_numeric, errors="coerce").astype(np.float32, copy=False)
    matrix_numeric = matrix_numeric.where(np.isfinite(matrix_numeric), np.nan)
    row_mean = matrix_numeric.mean(axis=1, skipna=True)

    units_plot = units[["geometry"]].copy()
    # Accessibility map policy: color only blocks with residential stock.
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
        units_plot = normalize_preview_gdf(units_plot, boundary_plot, target_crs="EPSG:3857")
        base_plot = units_plot.copy()
        res_plot = units_plot[units_plot["is_residential"]].copy()
        fig, ax = plt.subplots(figsize=(12, 10))
        apply_preview_canvas(fig, ax, boundary_plot)
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
        ax.set_title(
            "Accessibility: mean travel time (residential blocks only)",
            fontsize=19,
            fontweight="bold",
            color="#ffffff",
            pad=18,
        )
        ax.set_axis_off()
        save_preview_figure(fig, access_map_path)
        plt.close(fig)
        out["accessibility_mean_time_map"] = str(access_map_path)
        _log(f"Preview step: saved accessibility map: {access_map_path.name}")

    return out


def _plot_block_selection_status(
    blocks: gpd.GeoDataFrame,
    out_path: Path,
    *,
    title: str,
    status_column: str,
    color_map: dict[str, str],
    label_map: dict[str, str],
    footer_lines: list[str] | None = None,
    boundary: gpd.GeoDataFrame | None = None,
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if blocks is None or blocks.empty or status_column not in blocks.columns:
        return None

    gdf = blocks.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return None
    boundary_plot = normalize_preview_gdf(boundary, target_crs="EPSG:3857")
    gdf = normalize_preview_gdf(gdf, boundary_plot, target_crs="EPSG:3857")

    fig, ax = plt.subplots(figsize=(12, 10))
    apply_preview_canvas(fig, ax, boundary_plot, title=title)

    statuses = gdf[status_column].astype("string").fillna("unknown")
    legend_handles = []
    for status, color in color_map.items():
        part = gdf[statuses == status]
        if part.empty:
            continue
        part.plot(ax=ax, color=color, linewidth=0.05, edgecolor="#d1d5db", alpha=0.92, zorder=2)
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=label_map.get(status, status)))
    if legend_handles:
        legend_bottom(ax, legend_handles, max_cols=4, fontsize=10)
    footer_text(fig, footer_lines)
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview_figure(fig, out_path)
    plt.close(fig)
    _log(f"Preview step: saved selection-status map: {out_path.name}")
    return str(out_path)


def _coerce_solver_blocks_geodataframe(
    solver_blocks: pd.DataFrame | gpd.GeoDataFrame,
    *,
    blocks_ref: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame | None:
    gdf = solver_blocks.copy()
    if isinstance(gdf, gpd.GeoDataFrame) and "geometry" in gdf.columns:
        return gdf

    if "geometry" not in gdf.columns and blocks_ref is not None and "name" in gdf.columns:
        name_to_geom = {str(idx): geom for idx, geom in blocks_ref.geometry.items()}
        gdf["geometry"] = gdf["name"].astype(str).map(name_to_geom)

    if "geometry" not in gdf.columns:
        return None
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=(blocks_ref.crs if blocks_ref is not None else 4326))


def _plot_service_lp_preview(
    solver_blocks: pd.DataFrame | gpd.GeoDataFrame,
    service: str,
    out_path: Path,
    *,
    blocks_ref: gpd.GeoDataFrame | None = None,
    boundary: gpd.GeoDataFrame | None = None,
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if solver_blocks is None or solver_blocks.empty:
        return None
    _log(f"Preview step: rendering LP map for service [{service}]...")

    gdf = _coerce_solver_blocks_geodataframe(solver_blocks, blocks_ref=blocks_ref)
    if gdf is None:
        return None
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return None
    boundary_plot = normalize_preview_gdf(boundary, target_crs="EPSG:3857")
    gdf = normalize_preview_gdf(gdf, boundary_plot, target_crs="EPSG:3857")

    provision_col = "provision" if "provision" in gdf.columns else "provision_strong" if "provision_strong" in gdf.columns else None
    if provision_col is None:
        return None

    provision = pd.to_numeric(gdf[provision_col], errors="coerce")
    demand = pd.to_numeric(gdf.get("demand", 0.0), errors="coerce").fillna(0.0)
    demand_without = pd.to_numeric(gdf.get("demand_without", 0.0), errors="coerce").fillna(0.0)
    demand_left = pd.to_numeric(gdf.get("demand_left", 0.0), errors="coerce").fillna(0.0)
    has_access_gap = demand_without > 1e-9
    has_capacity_gap = demand_left > 1e-9

    gdf["gap_type"] = "no_gap"
    gdf.loc[has_access_gap & ~has_capacity_gap, "gap_type"] = "accessibility_gap"
    gdf.loc[~has_access_gap & has_capacity_gap, "gap_type"] = "capacity_gap"
    gdf.loc[has_access_gap & has_capacity_gap, "gap_type"] = "both_gaps"
    gdf.loc[provision.isna(), "gap_type"] = "missing_data"

    color_map = {
        "no_gap": "#16a34a",
        "capacity_gap": "#dc2626",
        "accessibility_gap": "#2563eb",
        "both_gaps": "#a21caf",
        "missing_data": "#9ca3af",
    }
    label_map = {
        "no_gap": "no unmet demand",
        "capacity_gap": "capacity gap (demand_left)",
        "accessibility_gap": "accessibility gap (demand_without)",
        "both_gaps": "both gaps",
        "missing_data": "missing provision data",
    }
    status_order = ("no_gap", "capacity_gap", "accessibility_gap", "both_gaps", "missing_data")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 12),
        gridspec_kw={"height_ratios": [5, 1.7], "hspace": 0.18},
    )
    ax = axes[0]
    hist_ax = axes[1]
    apply_preview_canvas(fig, ax, boundary_plot)
    hist_ax.set_facecolor("#f7f0dd")
    legend_handles = []
    for status in status_order:
        part = gdf[gdf["gap_type"] == status]
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
        legend_handles.append(Patch(facecolor=color_map[status], edgecolor="none", label=label_map[status]))
    if legend_handles:
        legend_bottom(ax, legend_handles, max_cols=2, fontsize=10)
    no_gap_cnt = int((gdf["gap_type"] == "no_gap").sum())
    cap_gap_cnt = int((gdf["gap_type"] == "capacity_gap").sum())
    access_gap_cnt = int((gdf["gap_type"] == "accessibility_gap").sum())
    both_gap_cnt = int((gdf["gap_type"] == "both_gaps").sum())
    ax.set_title(
        (
            f"{service}: unmet-demand type | "
            f"no_gap={no_gap_cnt}, cap_gap={cap_gap_cnt}, "
            f"access_gap={access_gap_cnt}, both={both_gap_cnt}"
        ),
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

    demand_total = float(demand.sum())
    access_total = float(demand_without.sum())
    capacity_total = float(demand_left.sum())
    footer_text(
        fig,
        [
            (
                f"unmet totals: accessibility={access_total:.0f} "
                f"({(access_total / demand_total * 100.0) if demand_total > 0 else 0.0:.2f}%), "
                f"capacity={capacity_total:.0f} "
                f"({(capacity_total / demand_total * 100.0) if demand_total > 0 else 0.0:.2f}%)"
            )
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview_figure(fig, out_path)
    plt.close(fig)
    _log(f"Preview step: saved LP map for service [{service}]: {out_path.name}")
    return str(out_path)


def _load_solver_flp_optimize():
    repo_root = Path(__file__).resolve().parents[2]
    solver_src = repo_root / "solver_flp" / "src"
    if str(solver_src) not in sys.path:
        sys.path.insert(0, str(solver_src))
    from method import optimize_placement  # type: ignore

    return optimize_placement


def _resolve_placement_demand_column(solver_blocks: pd.DataFrame) -> str:
    for candidate in ("demand_without", "demand_left", "demand"):
        if candidate in solver_blocks.columns:
            values = pd.to_numeric(solver_blocks[candidate], errors="coerce").fillna(0.0)
            if candidate == "demand" or float(values.sum()) > 0.0:
                return candidate
    raise ValueError("Could not resolve placement demand column.")


def _build_placement_target_demand(solver_blocks: pd.DataFrame) -> pd.Series:
    demand_left = pd.to_numeric(solver_blocks.get("demand_left", 0.0), errors="coerce").fillna(0.0)
    demand_without = pd.to_numeric(solver_blocks.get("demand_without", 0.0), errors="coerce").fillna(0.0)
    target = demand_left + demand_without
    if float(target.sum()) > 0.0:
        return target
    demand = pd.to_numeric(solver_blocks.get("demand", 0.0), errors="coerce").fillna(0.0)
    return demand


def _build_assignment_links(res_id: dict, blocks_after: pd.DataFrame) -> pd.DataFrame:
    demand_by_id = {}
    if "target_unmet_demand" in blocks_after.columns and "name" in blocks_after.columns:
        demand_by_id = {
            str(row["name"]): float(row["target_unmet_demand"])
            for _, row in blocks_after[["name", "target_unmet_demand"]].iterrows()
        }
    rows = []
    for facility_id, client_ids in res_id.items():
        for client_id in client_ids:
            rows.append(
                {
                    "facility_id": str(facility_id),
                    "client_id": str(client_id),
                    "client_demand_target": float(demand_by_id.get(str(client_id), 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _plot_placement_status_preview(
    blocks_after: pd.DataFrame | gpd.GeoDataFrame,
    service: str,
    out_path: Path,
    *,
    blocks_ref: gpd.GeoDataFrame | None = None,
    boundary: gpd.GeoDataFrame | None = None,
) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import box

    gdf = _coerce_solver_blocks_geodataframe(blocks_after, blocks_ref=blocks_ref)
    if gdf is None or gdf.empty:
        return None
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return None
    _log(f"Preview step: rendering exact placement-status map for service [{service}]...")
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

    if "placement_status" not in gdf.columns:
        return None
    color_map = {
        "existing": "#2563eb",
        "expanded": "#7c3aed",
        "new": "#dc2626",
        "inactive": "#d1d5db",
    }
    label_map = {
        "existing": "existing service kept",
        "expanded": "existing service expanded",
        "new": "new service added",
        "inactive": "no service in block",
    }
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#6b6b6b")
    ax.set_facecolor("#6b6b6b")
    if outer_bg is not None and not outer_bg.empty:
        outer_bg.plot(ax=ax, facecolor="#6b6b6b", edgecolor="none", alpha=1.0, zorder=-20)
    if boundary_plot is not None and not boundary_plot.empty:
        boundary_plot.plot(ax=ax, facecolor="#f7f0dd", edgecolor="none", linewidth=0.0, alpha=1.0, zorder=-10)
    active_statuses = [status for status in ("inactive", "existing", "expanded", "new") if status in set(gdf["placement_status"])]
    for status in active_statuses:
        part = gdf[gdf["placement_status"] == status]
        if part.empty:
            continue
        part.plot(
            ax=ax,
            color=color_map[status],
            linewidth=0.05,
            edgecolor="#d1d5db",
            alpha=0.92,
            zorder=2,
        )
    if boundary_plot is not None and not boundary_plot.empty:
        boundary_plot.boundary.plot(ax=ax, color="#ffffff", linewidth=1.4, zorder=3)
    if outer_bounds is not None:
        ax.set_xlim(outer_bounds[0], outer_bounds[2])
        ax.set_ylim(outer_bounds[1], outer_bounds[3])
    counts = {status: int((gdf["placement_status"] == status).sum()) for status in color_map}
    title_bits = [
        f"existing={counts['existing']}",
        f"new={counts['new']}",
    ]
    if counts["expanded"] > 0:
        title_bits.append(f"expanded={counts['expanded']}")
    ax.set_title(
        f"{service}: exact placement status | " + ", ".join(title_bits),
        fontsize=16,
        fontweight="bold",
        color="#ffffff",
        pad=16,
    )
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color_map[s], edgecolor="none", label=label_map[s]) for s in active_statuses if counts[s] > 0]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            frameon=True,
            facecolor="#f7f0dd",
            edgecolor="#9ca3af",
            fontsize=10,
        )
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    _log(f"Preview step: saved exact placement-status map for service [{service}]: {out_path.name}")
    return str(out_path)


def _plot_placement_after_preview(
    blocks_after: pd.DataFrame | gpd.GeoDataFrame,
    service: str,
    out_path: Path,
    *,
    blocks_ref: gpd.GeoDataFrame | None = None,
    boundary: gpd.GeoDataFrame | None = None,
) -> str | None:
    after = blocks_after.copy()
    if "provision_after" in after.columns:
        after["provision"] = after["provision_after"]
    elif "provision_strong_after" in after.columns:
        # Backward compatibility with old cached outputs.
        after["provision"] = after["provision_strong_after"]
    if "demand_without_after" in after.columns:
        after["demand_without"] = after["demand_without_after"]
    return _plot_service_lp_preview(after, f"{service} exact-after", out_path, blocks_ref=blocks_ref, boundary=boundary)


def _run_exact_placement_for_service(
    solver_blocks: gpd.GeoDataFrame | pd.DataFrame,
    sub_mx: pd.DataFrame,
    service: str,
    output_dir: Path,
    *,
    preview_dir: Path,
    blocks_ref: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    use_genetic: bool = False,
    progress: bool = True,
    prefer_existing: bool = False,
    allow_existing_expansion: bool = False,
    capacity_mode: str = "fixed_mean",
    genetic_population_size: int = 50,
    genetic_generations: int = 20,
    genetic_mutation_rate: float = 0.7,
    genetic_num_parents: int = 10,
    use_cache: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_after_path = output_dir / "blocks_solver_after.parquet"
    summary_after_path = output_dir / "summary_after.json"
    assignment_links_path = output_dir / "assignment_links_after.csv"
    gallery_targets = PIPELINE2_PLACEMENT_GALLERY_FILENAMES.get(service, {})

    cached_summary = _try_load_json(summary_after_path) if summary_after_path.exists() else {}
    min_new_capacity = float(_service_min_new_capacity(service))
    fixed_new_capacity = None
    if capacity_mode == "fixed_mean":
        fixed_new_capacity = float(_service_fixed_mean_capacity(service, solver_blocks))
        min_new_capacity = float(fixed_new_capacity)

    cache_mode_matches = bool(
        cached_summary
        and bool(cached_summary.get("use_genetic", False)) == bool(use_genetic)
        and bool(cached_summary.get("prefer_existing", False)) == bool(prefer_existing)
        and bool(cached_summary.get("allow_existing_expansion", False)) == bool(allow_existing_expansion)
        and str(cached_summary.get("capacity_mode", "service_min")) == str(capacity_mode)
        and float(cached_summary.get("min_new_capacity", -1.0)) == float(min_new_capacity)
        and int(cached_summary.get("genetic_population_size", 50)) == int(genetic_population_size)
        and int(cached_summary.get("genetic_generations", 20)) == int(genetic_generations)
        and float(cached_summary.get("genetic_mutation_rate", 0.7)) == float(genetic_mutation_rate)
        and int(cached_summary.get("genetic_num_parents", 10)) == int(genetic_num_parents)
        and (
            (cached_summary.get("fixed_new_capacity") is None and fixed_new_capacity is None)
            or float(cached_summary.get("fixed_new_capacity", -1.0)) == float(fixed_new_capacity)
        )
    )
    manifest_cache_ok = (
        use_cache
        and blocks_after_path.exists()
        and summary_after_path.exists()
        and assignment_links_path.exists()
        and cache_mode_matches
    )
    if manifest_cache_ok:
        return {
            "summary_after": str(summary_after_path),
            "blocks_after": str(blocks_after_path),
            "assignment_links_after": str(assignment_links_path),
            "status_preview_png": str(preview_dir / gallery_targets["status"]) if gallery_targets.get("status") and (preview_dir / gallery_targets["status"]).exists() else None,
            "after_preview_png": str(preview_dir / gallery_targets["after"]) if gallery_targets.get("after") and (preview_dir / gallery_targets["after"]).exists() else None,
            "selected_count": int(cached_summary.get("selected_count", 0)),
            "new_count": int(cached_summary.get("new_count", 0)),
            "expanded_count": int(cached_summary.get("expanded_count", 0)),
            "use_genetic": bool(cached_summary.get("use_genetic", False)),
        }

    optimize_placement = _load_solver_flp_optimize()
    demand_column = "target_unmet_demand"
    target_demand_full = _build_placement_target_demand(solver_blocks)
    provision_series = pd.to_numeric(solver_blocks.get("provision", 0.0), errors="coerce").fillna(0.0)
    unmet_mask = target_demand_full > 0.0
    if "provision" in solver_blocks.columns:
        unmet_mask = unmet_mask | (provision_series < 1.0)
    work = solver_blocks[unmet_mask].copy()
    work[demand_column] = target_demand_full.loc[work.index].astype(float)
    ids = [str(idx) for idx in work.index]
    if not ids:
        raise ValueError(f"No active blocks for exact placement [{service}].")
    matrix_full = sub_mx.copy()
    matrix_full.index = matrix_full.index.map(str)
    matrix_full.columns = matrix_full.columns.map(str)
    missing_ids = [idx for idx in ids if idx not in matrix_full.index or idx not in matrix_full.columns]
    if missing_ids:
        raise ValueError(
            f"Accessibility matrix does not contain {len(missing_ids)} placement ids "
            f"for service [{service}] (sample={missing_ids[:5]})."
        )
    matrix = matrix_full.loc[ids, ids].copy()
    work = work.reset_index(drop=True)
    matrix = matrix.reset_index(drop=True)
    matrix.columns = matrix.index

    _log(
        f"Exact placement [{service}]: blocks={len(work)}, demand_column={demand_column}, "
        f"demand_sum={float(pd.to_numeric(work[demand_column], errors='coerce').fillna(0.0).sum()):.1f}, "
        f"use_genetic={bool(use_genetic)}, capacity_mode={capacity_mode}, "
        f"fixed_new_capacity={fixed_new_capacity if fixed_new_capacity is not None else 'none'}, "
        f"genetic(pop={int(genetic_population_size)}, gen={int(genetic_generations)})"
    )
    started = time.time()
    optimization = optimize_placement(
        matrix=matrix,
        df=work,
        service_radius=float(work["service_radius_min"].iloc[0]),
        id_matrix=ids,
        use_genetic=bool(use_genetic),
        demand_column=demand_column,
        population_size=int(genetic_population_size),
        num_generations=int(genetic_generations),
        mutation_rate=float(genetic_mutation_rate),
        num_parents=int(genetic_num_parents),
        prefer_existing=bool(prefer_existing),
        keep_existing_capacity=True,
        allow_existing_expansion=bool(allow_existing_expansion),
        min_new_capacity=float(min_new_capacity),
        fixed_new_capacity=fixed_new_capacity,
        heartbeat_interval_sec=1.0,
        verbose=bool(progress),
    )
    elapsed = time.time() - started

    blocks_after = solver_blocks.copy()
    original_capacity = pd.to_numeric(blocks_after["capacity"], errors="coerce").fillna(0.0)
    optimized_capacity_total = original_capacity.copy()
    optimized_capacity_total.loc[ids] = np.asarray(optimization["capacities"], dtype=float)
    blocks_after["optimized_capacity_total"] = optimized_capacity_total.reindex(blocks_after.index).fillna(0.0)
    blocks_after["optimized_capacity_added"] = np.maximum(
        blocks_after["optimized_capacity_total"] - original_capacity,
        0.0,
    )
    statuses = pd.Series("inactive", index=blocks_after.index, dtype=object)
    selected_mask = blocks_after["optimized_capacity_total"] > 0.0
    existing_mask = original_capacity > 0.0
    statuses.loc[existing_mask & selected_mask] = "existing"
    statuses.loc[existing_mask & (blocks_after["optimized_capacity_added"] > 0.0)] = "expanded"
    statuses.loc[~existing_mask & selected_mask] = "new"
    blocks_after["placement_status"] = statuses
    target_demand = target_demand_full.reindex(blocks_after.index).fillna(0.0)
    blocks_after["target_unmet_demand"] = target_demand

    # Re-evaluate post-placement provision using the same arctic lp_coverage solver.
    reprovision_input = blocks_after[["name", "population", "demand", "geometry"]].copy()
    reprovision_input["capacity"] = pd.to_numeric(blocks_after["optimized_capacity_total"], errors="coerce").fillna(0.0)
    provision_after_df, links_after_df = _run_arctic_lp_provision(
        blocks_df=gpd.GeoDataFrame(reprovision_input, geometry="geometry", crs=blocks_after.crs),
        accessibility_matrix=sub_mx,
        service=service,
        service_radius_min=float(solver_blocks["service_radius_min"].iloc[0]),
        service_demand_per_1000=float(solver_blocks["service_demand_per_1000"].iloc[0]),
    )
    if "provision" not in provision_after_df.columns and "provision_strong" in provision_after_df.columns:
        provision_after_df = provision_after_df.copy()
        provision_after_df["provision"] = provision_after_df["provision_strong"]
    for before_col, after_col in (
        ("demand_within", "demand_within_after"),
        ("demand_without", "demand_without_after"),
        ("demand_left", "demand_left_after"),
        ("capacity_left", "capacity_left_after"),
        ("capacity_within", "capacity_within_after"),
        ("capacity_without", "capacity_without_after"),
        ("provision", "provision_after"),
    ):
        if before_col in provision_after_df.columns:
            blocks_after[after_col] = pd.to_numeric(provision_after_df[before_col], errors="coerce").reindex(blocks_after.index)
    _save_geodata(blocks_after, blocks_after_path)

    assignment_links = _build_assignment_links(optimization["res_id"], blocks_after)
    assignment_links_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_links.to_csv(assignment_links_path, index=False)
    provision_links_after_path = output_dir / "provision_links_after.csv"
    provision_links_after_path.parent.mkdir(parents=True, exist_ok=True)
    links_after_df.to_csv(provision_links_after_path, index=False)

    status_png = None
    after_png = None
    if gallery_targets.get("status"):
        status_png = _plot_placement_status_preview(
            blocks_after,
            service,
            preview_dir / gallery_targets["status"],
            blocks_ref=blocks_ref,
            boundary=boundary,
        )
    if gallery_targets.get("after"):
        after_png = _plot_placement_after_preview(
            blocks_after,
            service,
            preview_dir / gallery_targets["after"],
            blocks_ref=blocks_ref,
            boundary=boundary,
        )

    total_demand = pd.to_numeric(blocks_after.get("demand", 0.0), errors="coerce").fillna(0.0)
    summary_after = {
        "service": service,
        "mode": "exact",
        "use_genetic": bool(use_genetic),
        "prefer_existing": bool(prefer_existing),
        "allow_existing_expansion": bool(allow_existing_expansion),
        "capacity_mode": str(capacity_mode),
        "genetic_population_size": int(genetic_population_size),
        "genetic_generations": int(genetic_generations),
        "genetic_mutation_rate": float(genetic_mutation_rate),
        "genetic_num_parents": int(genetic_num_parents),
        "placement_engine": PLACEMENT_ENGINE_NAME,
        "provision_engine_recompute": PROVISION_ENGINE_NAME,
        "demand_column": demand_column,
        "elapsed_sec": float(elapsed),
        "min_new_capacity": float(min_new_capacity),
        "fixed_new_capacity": float(fixed_new_capacity) if fixed_new_capacity is not None else None,
        "blocks_count": int(len(blocks_after)),
        "selected_count": int((blocks_after["optimized_capacity_total"] > 0.0).sum()),
        "new_count": int((blocks_after["placement_status"] == "new").sum()),
        "expanded_count": int((blocks_after["placement_status"] == "expanded").sum()),
        "existing_count": int((blocks_after["placement_status"] == "existing").sum()),
        "capacity_total_before": float(original_capacity.sum()),
        "capacity_total_after": float(blocks_after["optimized_capacity_total"].sum()),
        "capacity_added_total": float(blocks_after["optimized_capacity_added"].sum()),
        "demand_target_total": float(target_demand.sum()),
        "demand_without_after_total": float(pd.to_numeric(blocks_after["demand_without_after"], errors="coerce").fillna(0.0).sum()),
        "demand_left_after_total": float(pd.to_numeric(blocks_after["demand_left_after"], errors="coerce").fillna(0.0).sum()),
        "provision_total_after": float(
            pd.to_numeric(blocks_after["demand_within_after"], errors="coerce").fillna(0.0).sum() / total_demand.sum()
            if total_demand.sum() > 0
            else 0.0
        ),
        "files": {
            "blocks_after": str(blocks_after_path),
            "assignment_links_after": str(assignment_links_path),
            "provision_links_after": str(provision_links_after_path),
            "status_preview_png": status_png,
            "after_preview_png": after_png,
        },
    }
    summary_after_path.write_text(json.dumps(summary_after, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "summary_after": str(summary_after_path),
        "blocks_after": str(blocks_after_path),
        "assignment_links_after": str(assignment_links_path),
        "status_preview_png": status_png,
        "after_preview_png": after_png,
        "selected_count": summary_after["selected_count"],
        "new_count": summary_after["new_count"],
        "expanded_count": summary_after["expanded_count"],
        "use_genetic": bool(use_genetic),
        "capacity_mode": str(capacity_mode),
    }


def main() -> None:
    _configure_logging()
    args = parse_args()
    _configure_osmnx(args)

    services = _ensure_services_valid(args.services)
    city_dir = _resolve_city_dir(args)

    boundary_path = city_dir / "analysis_territory" / "buffer.parquet"
    blocks_layer_path = city_dir / "derived_layers" / "blocks_clipped.parquet"
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
    blocks = _read_blocks(blocks_layer_path)
    graph = _read_graph_pickle(graph_path)

    population_col = _detect_population_column(blocks)
    if population_col is None:
        raise ValueError(
            "No population columns found in blocks. Strict mode requires one of: "
            "population_total, population_proxy, population."
        )
    else:
        _log(f"Population source column: {population_col}")
    _log(f"Blocks features: {len(blocks)}")
    _log(f"Intermodal graph: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

    _log("STEP data_collection: preparing raw service layers from shared collection cache.")
    _log("STEP capacity_aggregation: converting raw service objects to per-block capacities.")

    # 1) Build per-service raw layers from shared collection and aggregate capacities to blocks.
    common_services_raw, common_services_path = _read_common_services_raw(city_dir)
    _log(
        "Using shared raw services layer: "
        f"{_log_name(common_services_path)} ({len(common_services_raw)} features)"
    )
    capacity_columns: dict[str, pd.Series] = {}
    raw_stats: dict[str, dict] = {}
    raw_service_points: dict[str, gpd.GeoDataFrame] = {}
    for service in services:
        raw_path = raw_dir / f"{service}.parquet"
        if raw_path.exists() and (not args.no_cache):
            raw = read_geodata(raw_path)
            _log(f"Using cached raw service layer [{service}]: {_log_name(raw_path)} ({len(raw)} features)")
        else:
            raw = _extract_service_raw_from_common(common_services_raw, service)
            if raw.empty:
                _warn(f"Raw service layer [{service}] is empty.")
            else:
                raw["capacity_est"] = raw.apply(
                    lambda row: _capacity_from_row(
                        row,
                        service,
                        allow_default_fallback=bool(args.capacity_default_fallback),
                    ),
                    axis=1,
                )
            _save_geodata(raw, raw_path)
            _log(f"Saved raw service layer [{service}]: {_log_name(raw_path)} ({len(raw)} features)")

        if "capacity_est" not in raw.columns:
            raw = raw.copy()
            raw["capacity_est"] = (
                raw.apply(
                    lambda row: _capacity_from_row(
                        row,
                        service,
                        allow_default_fallback=bool(args.capacity_default_fallback),
                    ),
                    axis=1,
                )
                if not raw.empty
                else 0.0
            )
            _save_geodata(raw, raw_path)

        points = _to_points(raw) if not raw.empty else raw
        if points is not None and not points.empty:
            raw_service_points[service] = points.copy()
        aggregated = _aggregate_capacity_to_blocks(points, blocks)
        cap_col = f"capacity_{service}"
        capacity_columns[cap_col] = aggregated

        raw_stats[service] = {
            "raw_features": int(len(raw)),
            "blocks_with_capacity": int((aggregated > 0).sum()),
            "capacity_total": float(aggregated.sum()),
            "raw_path": str(raw_path),
        }

    _log("STEP matrix_build: preparing unified spatial units for accessibility matrix.")

    # 2) Build a unified spatial-units layer for matrix + solver prep.
    units_all = blocks[[population_col, "geometry"]].copy()
    units_all = units_all.rename(columns={population_col: "population"})
    units_all["population"] = pd.to_numeric(units_all["population"], errors="coerce").fillna(0.0)
    # Keep residential context for accessibility-map styling.
    for residential_col in ("residential", "living_area", "living_area_proxy"):
        if residential_col in blocks.columns:
            units_all[residential_col] = blocks[residential_col].reindex(units_all.index)

    for cap_col, series in capacity_columns.items():
        units_all[cap_col] = series.reindex(units_all.index).fillna(0.0).astype(float)

    # Strict rule: blocks without population are excluded from all pipeline_2 calculations.
    units_mask = units_all["population"] > 0
    units = units_all[units_mask].copy()
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

    accessibility_selection = units_all[["geometry", "population"]].copy()
    accessibility_selection["selection_status"] = np.where(
        units_mask.reindex(units_all.index).fillna(False),
        "included_population_positive",
        "excluded_zero_population",
    )
    accessibility_selection["has_living_buildings"] = False
    accessibility_selection.loc[units.index, "has_living_buildings"] = units["has_living_buildings"].fillna(False).astype(bool)
    accessibility_selection["selection_detail"] = np.where(
        accessibility_selection["selection_status"].eq("excluded_zero_population"),
        "excluded_zero_population",
        np.where(
            accessibility_selection["has_living_buildings"],
            "included_with_living_buildings",
            "included_population_only",
        ),
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
            "Computing accessibility matrix via native graph shortest-path routine "
            f"for n_units={n_units} (~{approx_pairs:,} pair entries). This can take time."
        )
        started = time.time()
        matrix_union = _run_with_heartbeat(
            "Matrix build",
            lambda: _calculate_accessibility_matrix_native(units[["geometry"]].copy(), graph, weight_key="time_min"),
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
    _log(f"Provision engine: {PROVISION_ENGINE_NAME}")
    _log(
        "Placement engine: "
        f"{PLACEMENT_ENGINE_NAME} "
        f"({'enabled exact mode' if args.placement_exact else 'disabled'}, "
        f"use_genetic={bool(args.placement_genetic)}, progress={bool(args.placement_progress)}, "
        f"prefer_existing={bool(args.placement_prefer_existing)}, "
        f"allow_existing_expansion={bool(args.placement_allow_existing_expansion)}, "
        f"capacity_mode={str(args.placement_capacity_mode)}, "
        f"genetic(pop={int(args.placement_genetic_population_size)}, "
        f"gen={int(args.placement_genetic_generations)}, "
        f"mutation={float(args.placement_genetic_mutation_rate):.2f}, "
        f"parents={int(args.placement_genetic_num_parents)}))"
    )

    # 4) Per-service solver-ready tables (demand_within/demand_without/capacity_left/provision).
    service_outputs: dict[str, dict] = {}
    placement_outputs: dict[str, dict] = {}
    preview_outputs: dict[str, object] = {}
    placement_root_name = "placement_exact_genetic" if bool(args.placement_genetic) else "placement_exact"
    services_raw_overview_png = _plot_services_raw_overview(
        raw_service_points,
        preview_dir / PIPELINE2_GALLERY_FILENAMES["services_raw_all"],
        boundary=boundary,
    )
    if services_raw_overview_png is not None:
        preview_outputs["services_raw_all"] = services_raw_overview_png
    preview_outputs.update(
        _plot_accessibility_previews(units, matrix_union, preview_dir, boundary=boundary, use_cache=(not args.no_cache))
    )
    accessibility_selection_png = _plot_block_selection_status(
        accessibility_selection,
        preview_dir / PIPELINE2_SELECTION_GALLERY_FILENAMES["accessibility"],
        title="Accessibility Input Selection",
        status_column="selection_detail",
        color_map={
            "included_with_living_buildings": "#2563eb",
            "included_population_only": "#93c5fd",
            "excluded_zero_population": "#d1d5db",
        },
        label_map={
            "included_with_living_buildings": "included: population>0 and living buildings",
            "included_population_only": "included: population>0 but no living buildings",
            "excluded_zero_population": "excluded: zero population",
        },
        footer_lines=[
            "matrix/accessibility rule: include only blocks with population > 0",
            f"included={int(units_mask.sum())}, excluded={int((~units_mask).sum())}, included_with_living={int(living_units)}",
        ],
        boundary=boundary,
    )
    if accessibility_selection_png is not None:
        preview_outputs["accessibility_block_selection_status"] = accessibility_selection_png

    for service in services:
        cap_col = f"capacity_{service}"
        service_selection = units[["geometry", "has_living_buildings"]].copy()
        service_selection["capacity"] = pd.to_numeric(units.get(cap_col, 0.0), errors="coerce").fillna(0.0)
        service_selection["selection_status"] = "excluded_no_living_no_capacity"
        living_mask = service_selection["has_living_buildings"].fillna(False).astype(bool)
        capacity_mask = service_selection["capacity"] > 0.0
        service_selection.loc[living_mask & ~capacity_mask, "selection_status"] = "included_living_only"
        service_selection.loc[~living_mask & capacity_mask, "selection_status"] = "included_capacity_only"
        service_selection.loc[living_mask & capacity_mask, "selection_status"] = "included_living_and_capacity"
        selection_preview_target = preview_dir / PIPELINE2_SELECTION_GALLERY_FILENAMES[service]
        selection_preview_path = _plot_block_selection_status(
            service_selection,
            selection_preview_target,
            title=f"{service}: LP Input Selection",
            status_column="selection_status",
            color_map={
                "included_living_and_capacity": "#7c3aed",
                "included_living_only": "#2563eb",
                "included_capacity_only": "#f59e0b",
                "excluded_no_living_no_capacity": "#d1d5db",
            },
            label_map={
                "included_living_and_capacity": "included: living buildings and service capacity",
                "included_living_only": "included: living buildings only",
                "included_capacity_only": "included: service capacity only",
                "excluded_no_living_no_capacity": "excluded: no living buildings and no service capacity",
            },
            footer_lines=[
                "LP rule: include blocks with living buildings OR own service capacity",
                f"living_only={int((living_mask & ~capacity_mask).sum())}, capacity_only={int((~living_mask & capacity_mask).sum())}, living_and_capacity={int((living_mask & capacity_mask).sum())}, excluded={int((~living_mask & ~capacity_mask).sum())}",
            ],
            boundary=boundary,
        )
        if float(raw_stats.get(service, {}).get("capacity_total", 0.0)) <= 0.0:
            _warn(f"Service [{service}] has zero total capacity in territory; skipping.")
            service_outputs[service] = {
                "skipped": True,
                "reason": "zero_capacity_in_territory",
                "selection_preview_png": selection_preview_path,
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
            expected_population_share = _service_population_share(service)
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
            elif float(cached_summary.get("service_population_share", -1.0)) != float(expected_population_share):
                _warn(
                    f"Cached solver input [{service}] uses outdated service_population_share "
                    f"({cached_summary.get('service_population_share')} != {expected_population_share}). Rebuilding."
                )
            else:
                _log(f"Using cached solver input [{service}]: {_log_name(summary_path)}")
                lp_preview_path = str(lp_preview_target) if lp_preview_target.exists() else None
                if lp_preview_path is None:
                    try:
                        cached_blocks = gpd.read_parquet(blocks_path)
                        lp_preview_path = _plot_service_lp_preview(
                            cached_blocks,
                            service,
                            lp_preview_target,
                            blocks_ref=blocks,
                            boundary=boundary,
                        )
                    except Exception:
                        lp_preview_path = None
                service_outputs[service] = {
                    "provision_engine": PROVISION_ENGINE_NAME,
                    "summary": str(summary_path),
                    "blocks_solver": str(blocks_path),
                    "adj_matrix": str(matrix_service_path),
                    "provision_links": str(links_path),
                    "blocks_count": int(cached_summary.get("blocks_count", 0)),
                    "lp_preview_png": lp_preview_path,
                    "selection_preview_png": selection_preview_path,
                }
                if args.placement_exact:
                    try:
                        cached_blocks = gpd.read_parquet(blocks_path)
                        cached_matrix = pd.read_parquet(matrix_service_path)
                        placement_outputs[service] = _run_exact_placement_for_service(
                            cached_blocks,
                            cached_matrix,
                            service,
                            output_root / placement_root_name / service,
                            preview_dir=preview_dir,
                            blocks_ref=blocks,
                            boundary=boundary,
                            use_genetic=bool(args.placement_genetic),
                            progress=bool(args.placement_progress),
                            prefer_existing=bool(args.placement_prefer_existing),
                            allow_existing_expansion=bool(args.placement_allow_existing_expansion),
                            capacity_mode=str(args.placement_capacity_mode),
                            genetic_population_size=int(args.placement_genetic_population_size),
                            genetic_generations=int(args.placement_genetic_generations),
                            genetic_mutation_rate=float(args.placement_genetic_mutation_rate),
                            genetic_num_parents=int(args.placement_genetic_num_parents),
                            use_cache=(not args.no_cache),
                        )
                    except Exception as exc:  # noqa: BLE001
                        _warn(f"Exact placement [{service}] failed on cached inputs: {exc}")
                continue

        blocks = units[["unit_name", "population", "demand_base", cap_col, "geometry", "has_living_buildings"]].copy()
        blocks = blocks.rename(columns={cap_col: "capacity"})
        service_demand_per_1000 = _service_demand_per_1000(service, args)
        service_radius_min = _service_accessibility_min(service, args)
        blocks["demand"] = _calc_service_demand(blocks["population"], service, service_demand_per_1000)
        # LP policy: include only blocks with living buildings OR own service capacity.
        blocks["has_living_buildings"] = blocks["has_living_buildings"].fillna(False).astype(bool)
        blocks = blocks[blocks["has_living_buildings"] | (blocks["capacity"] > 0)].copy()
        if blocks.empty:
            _warn(f"No active blocks for service [{service}] after filtering. Skipping.")
            service_outputs[service] = {
                "skipped": True,
                "reason": "no_living_or_service_blocks_after_filtering",
                "selection_preview_png": selection_preview_path,
            }
            continue

        _log(
            f"Service [{service}] provisioning prep: blocks={len(blocks)} "
            f"(living={int(blocks['has_living_buildings'].sum())}, capacity>0={int((blocks['capacity'] > 0).sum())}, "
            f"accessibility={service_radius_min:.1f} min, demand_per_1000={service_demand_per_1000:.1f}, "
            f"population_share={_service_population_share(service):.2f})"
        )
        sub_mx = matrix_union.loc[blocks.index, blocks.index].copy()
        provision_started = time.time()
        provision_df, links_df = _run_arctic_lp_provision(
            blocks_df=gpd.GeoDataFrame(
                blocks[["population", "demand", "capacity", "geometry"]].copy().assign(name=blocks["unit_name"].astype(str)),
                geometry="geometry",
                crs=blocks.crs,
            ),
            accessibility_matrix=sub_mx,
            service=service,
            service_radius_min=float(service_radius_min),
            service_demand_per_1000=float(service_demand_per_1000),
        )
        _log(f"Service [{service}] arctic lp_coverage provision finished in {time.time() - provision_started:.1f}s")
        solver_blocks = provision_df.copy()
        if "geometry" not in solver_blocks.columns and "geometry" in blocks.columns:
            solver_blocks = solver_blocks.join(blocks[["geometry"]], how="left")
        if not isinstance(solver_blocks, gpd.GeoDataFrame) and "geometry" in solver_blocks.columns:
            solver_blocks = gpd.GeoDataFrame(solver_blocks, geometry="geometry", crs=blocks.crs)
        solver_blocks["name"] = solver_blocks.index.astype(str)
        solver_blocks["service_name"] = service
        solver_blocks["service_radius_min"] = float(service_radius_min)
        solver_blocks["service_demand_per_1000"] = float(service_demand_per_1000)
        solver_blocks["service_population_share"] = float(_service_population_share(service))
        if "provision" not in solver_blocks.columns and "provision_strong" in solver_blocks.columns:
            # Backward compatibility with old cached solver files.
            solver_blocks["provision"] = solver_blocks["provision_strong"].fillna(0.0)
        elif "provision" in solver_blocks.columns:
            solver_blocks["provision"] = pd.to_numeric(solver_blocks["provision"], errors="coerce").fillna(0.0)

        _save_geodata(solver_blocks, blocks_path)
        _save_dataframe(sub_mx, matrix_service_path)
        links_path.parent.mkdir(parents=True, exist_ok=True)
        links_df.to_csv(links_path, index=False)
        if (not args.no_cache) and lp_preview_target.exists():
            lp_preview_path = str(lp_preview_target)
        else:
            lp_preview_path = _plot_service_lp_preview(
                solver_blocks,
                service,
                lp_preview_target,
                blocks_ref=blocks,
                boundary=boundary,
            )
        summary = {
            "service": service,
            "provision_engine": PROVISION_ENGINE_NAME,
            "block_selection_policy": LP_BLOCK_SELECTION_POLICY,
            "service_radius_min": float(service_radius_min),
            "service_demand_per_1000": float(service_demand_per_1000),
            "service_population_share": float(_service_population_share(service)),
            "blocks_count": int(len(solver_blocks)),
            "blocks_with_living": int(blocks["has_living_buildings"].sum()),
            "blocks_with_service_capacity": int((blocks["capacity"] > 0).sum()),
            "blocks_living_only": int((living_mask & ~capacity_mask).sum()),
            "blocks_capacity_only": int((~living_mask & capacity_mask).sum()),
            "blocks_living_and_capacity": int((living_mask & capacity_mask).sum()),
            "blocks_excluded_no_living_no_capacity": int((~living_mask & ~capacity_mask).sum()),
            "demand_total": float(solver_blocks["demand"].sum()),
            "capacity_total": float(solver_blocks["capacity"].sum()),
            "demand_within_total": float(solver_blocks["demand_within"].sum()),
            "demand_without_total": float(solver_blocks["demand_without"].sum()),
            "provision_total": float(
                solver_blocks["demand_within"].sum() / solver_blocks["demand"].sum()
                if solver_blocks["demand"].sum() > 0
                else 0.0
            ),
            "files": {
                "blocks_solver": str(blocks_path),
                "adj_matrix": str(matrix_service_path),
                "provision_links": str(links_path),
                "selection_preview_png": selection_preview_path,
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        service_outputs[service] = {
            "provision_engine": PROVISION_ENGINE_NAME,
            "summary": str(summary_path),
            "blocks_solver": str(blocks_path),
            "adj_matrix": str(matrix_service_path),
            "provision_links": str(links_path),
            "blocks_count": int(len(solver_blocks)),
            "lp_preview_png": lp_preview_path,
            "selection_preview_png": selection_preview_path,
        }
        if args.placement_exact:
            try:
                placement_outputs[service] = _run_exact_placement_for_service(
                    solver_blocks,
                    sub_mx,
                    service,
                    output_root / placement_root_name / service,
                    preview_dir=preview_dir,
                    blocks_ref=blocks,
                    boundary=boundary,
                    use_genetic=bool(args.placement_genetic),
                    progress=bool(args.placement_progress),
                    prefer_existing=bool(args.placement_prefer_existing),
                    allow_existing_expansion=bool(args.placement_allow_existing_expansion),
                    capacity_mode=str(args.placement_capacity_mode),
                    genetic_population_size=int(args.placement_genetic_population_size),
                    genetic_generations=int(args.placement_genetic_generations),
                    genetic_mutation_rate=float(args.placement_genetic_mutation_rate),
                    genetic_num_parents=int(args.placement_genetic_num_parents),
                    use_cache=(not args.no_cache),
                )
            except Exception as exc:  # noqa: BLE001
                _warn(f"Exact placement [{service}] failed: {exc}")
        _log(
            f"Prepared solver input [{service}]: blocks={len(solver_blocks)}, "
            f"capacity_total={summary['capacity_total']:.1f}, demand_total={summary['demand_total']:.1f}"
        )

    manifest = {
        "city_bundle": str(city_dir),
        "boundary": str(boundary_path),
        "blocks": str(blocks_layer_path),
        "graph": str(graph_path),
        "services": services,
        "service_radius_min": float(args.service_radius_min) if args.service_radius_min is not None else None,
        "demand_per_1000": float(args.demand_per_1000) if args.demand_per_1000 is not None else None,
        "units_union": str(units_path),
        "adj_matrix_union": str(matrix_path),
        "accessibility_selection": {
            "selection_policy": "population_positive_only",
            "blocks_total_before_filter": int(len(units_all)),
            "blocks_included": int(units_mask.sum()),
            "blocks_excluded_zero_population": int((~units_mask).sum()),
            "blocks_included_with_living_buildings": int(living_units),
            "selection_preview_png": accessibility_selection_png,
        },
        "previews_dir": str(preview_dir),
        "preview_outputs": preview_outputs,
        "raw_services": raw_stats,
        "provision_engine": {
            "name": PROVISION_ENGINE_NAME,
            "scope": "baseline provision assessment and post-placement recompute",
        },
        "placement_engine": {
            "name": PLACEMENT_ENGINE_NAME,
            "enabled": bool(args.placement_exact),
            "mode": ("exact_genetic" if (args.placement_exact and args.placement_genetic) else ("exact" if args.placement_exact else None)),
            "use_genetic": bool(args.placement_genetic),
            "progress": bool(args.placement_progress),
            "prefer_existing": bool(args.placement_prefer_existing),
            "allow_existing_expansion": bool(args.placement_allow_existing_expansion),
            "capacity_mode": str(args.placement_capacity_mode),
            "genetic_population_size": int(args.placement_genetic_population_size),
            "genetic_generations": int(args.placement_genetic_generations),
            "genetic_mutation_rate": float(args.placement_genetic_mutation_rate),
            "genetic_num_parents": int(args.placement_genetic_num_parents),
        },
        "solver_outputs": service_outputs,
        "placement_exact_enabled": bool(args.placement_exact),
        "placement_exact_outputs": placement_outputs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Done. Manifest: {_log_name(manifest_path)}")


if __name__ == "__main__":
    main()

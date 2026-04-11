from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from matplotlib.patches import Patch
from shapely import Polygon
from shapely.geometry import Point, box
from tqdm.auto import tqdm

from aggregated_spatial_pipeline.geodata_io import (
    prepare_geodata_for_parquet,
    read_geodata,
)
from aggregated_spatial_pipeline.runtime_paths import (
    blocksnet_python,
    connectpt_python,
    floor_predictor_python,
    repo_root,
    street_pattern_python,
)
from aggregated_spatial_pipeline.runtime_config import configure_logger, repo_mplconfigdir
from aggregated_spatial_pipeline.spec import CONFIG_DIR, PipelineSpec
from aggregated_spatial_pipeline.visualization import (
    apply_preview_canvas,
    clip_to_preview_boundary,
    footer_text,
    legend_bottom,
    normalize_preview_gdf,
    save_preview_figure,
)

from .crosswalks import build_crosswalk, save_crosswalk
from .io import load_layer, save_layer
from .scenarios import run_scenarios


JOINT_SCENARIO_ID = "joint_optimization"
TQDM_DISABLE = not sys.stderr.isatty()


@dataclass(frozen=True)
class PreparedInputs:
    layer_inputs: dict[str, Path]
    city_label: str | None
    downloaded_in_this_run: bool
    source_details: dict[str, dict]


def _format_crs_for_log(crs) -> str:
    if crs is None:
        return "unknown"
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg is not None:
        return f"EPSG:{epsg}"
    name = getattr(crs, "name", None)
    if name:
        return str(name)
    return str(crs)


def _log_name(path: Path | str | None) -> str:
    if path is None:
        return "none"
    try:
        return Path(path).name
    except Exception:
        return str(path)


def _tqdm_kwargs(*, leave: bool = False) -> dict:
    return {
        "disable": TQDM_DISABLE,
        "leave": leave,
        "ascii": True,
        "dynamic_ncols": True,
        "mininterval": 0.5,
    }


def _repo_mplconfigdir(root: Path, name: str) -> str:
    return repo_mplconfigdir(name, root=root)


def _run_external_json_command(
    *,
    python_path: Path,
    module: str,
    repo_root: Path,
    args: list[str],
    mplconfigdir: str,
) -> dict:
    if not python_path.exists():
        raise FileNotFoundError(
            f"Required runtime was not found: {python_path}. "
            "Create dedicated per-repository environments first via bootstrap."
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    env.setdefault("MPLCONFIGDIR", mplconfigdir)
    command = [str(python_path), "-m", module, *args]
    completed = subprocess.run(
        command,
        check=True,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    stdout = (completed.stdout or "").strip()
    if not stdout:
        return {}
    last_line = stdout.splitlines()[-1]
    return json.loads(last_line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the joint aggregated pipeline with optional data collection and clear progress output."
    )
    parser.add_argument("--blocks", help="Path to block polygons.")
    parser.add_argument("--street-grid", help="Path to street-grid polygons and morphology attributes.")
    parser.add_argument("--climate-grid", help="Path to climate-grid polygons and environmental attributes.")
    parser.add_argument("--cities", help="Path to city polygons.")
    parser.add_argument("--place", help="City/place for automatic data collection (e.g. 'Tianjin, China').")
    parser.add_argument(
        "--city-centers-csv",
        default=None,
        help=(
            "Path to CSV with city centres (SimpleMaps worldcities format). "
            "Default: <repo>/simplemaps_worldcities_basicv1/worldcities.csv."
        ),
    )
    parser.add_argument(
        "--center-node-id",
        "--centre-node-id",
        dest="center_node_id",
        type=int,
        help="Optional OSM node id of city centre. If provided, buffer is built from this node directly.",
    )
    parser.add_argument(
        "--relation-id",
        dest="relation_id",
        type=int,
        help=(
            "Optional OSM relation id used as analysis territory directly. "
            "If provided, the main territory is the relation polygon itself and no outer margin is added."
        ),
    )
    parser.add_argument(
        "--relation-geometry",
        dest="relation_geometry",
        help=(
            "Optional local GeoJSON/Parquet/GPKG file with territory polygon. "
            "If provided together with --relation-id, the file is used as the territory geometry "
            "without any network requests."
        ),
    )
    parser.add_argument(
        "--city",
        help="Optional city label shown in logs/manifest (overrides auto-detection).",
    )
    parser.add_argument(
        "--data-dir",
        help=(
            "Directory for collected input artifacts. "
            "Default: aggregated_spatial_pipeline/outputs/joint_inputs/<place_slug>."
        ),
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["bus", "tram", "trolleybus"],
        help="Transport modalities for connectpt collection. Default: all supported (bus tram trolleybus).",
    )
    parser.add_argument(
        "--speed-kmh",
        type=float,
        default=20.0,
        help="Assumed average speed for connectpt stop-to-stop estimates.",
    )
    parser.add_argument(
        "--osm-timeout-s",
        type=float,
        default=60.0,
        help="Timeout (seconds) for OSMnx/Overpass requests during data collection.",
    )
    parser.add_argument(
        "--overpass-url",
        default=None,
        help="Optional Overpass endpoint URL (e.g. https://overpass.kumi.systems/api/interpreter).",
    )
    parser.add_argument(
        "--osmnx-debug",
        action="store_true",
        help="Enable OSMnx console logs via OSMnx settings.",
    )
    parser.add_argument(
        "--intermodal-python",
        default=None,
        help=(
            "Optional path to dedicated python executable used only for intermodal graph collection "
            "(e.g. iduedu-fork/.venv/bin/python)."
        ),
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=None,
        help=(
            "Optional buffer radius in meters used for circle-mode. "
            "If omitted for --place, the pipeline uses the relation polygon; "
            "if omitted with --center-node-id, the default circle radius is kept."
        ),
    )
    parser.add_argument(
        "--street-grid-step",
        type=float,
        default=500.0,
        help="Grid step in meters for street-pattern classification. Default: 500.",
    )
    parser.add_argument(
        "--street-min-road-count",
        type=int,
        default=5,
        help="Skip street-grid cells with fewer clipped road segments than this.",
    )
    parser.add_argument(
        "--street-min-total-road-length",
        type=float,
        default=500.0,
        help="Skip street-grid cells whose total clipped road length is below this threshold.",
    )
    parser.add_argument(
        "--floor-min-buffer-m",
        type=float,
        default=25000.0,
        help="Recommended minimum clipping radius (meters) for floor/is_living restoration context.",
    )
    parser.add_argument(
        "--floor-ignore-missing-below-pct",
        type=float,
        default=2.0,
        help=(
            "Skip heavy storey model inference when share of buildings requiring prediction "
            "is below this threshold (in percent). Default: 2.0"
        ),
    )
    parser.add_argument(
        "--pt-street-pattern-dependency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run PT x street-pattern dependency step on collected iduedu graph "
            "(bus/tram/trolleybus in modality analysis + subway stops buffer context)."
        ),
    )
    parser.add_argument(
        "--pt-dependency-top-routes",
        type=int,
        default=30,
        help="Number of top routes to keep in PT x street-pattern manifest preview.",
    )
    parser.add_argument(
        "--pt-subway-stop-buffer-m",
        type=float,
        default=500.0,
        help="Subway stop buffer radius in meters for PT x street-pattern context.",
    )
    parser.add_argument(
        "--is-living-model-path",
        default=None,
        help=(
            "Optional path to trained floor_predictior osm_living model (.pkl). "
            "Default: floor-predictor/floor_predictior/model/model_dt.pkl"
        ),
    )
    parser.add_argument(
        "--climate-grid-step-m",
        type=float,
        default=5000.0,
        help="Cell size in meters for derived climate-grid layer.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force full data recollection/rebuild, ignoring cached manifests and derived layers.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Run only phase 1 (data collection/preparation) and stop before any joint calculations.",
    )
    parser.add_argument(
        "--roads-coverage-buffer-m",
        type=float,
        default=8.0,
        help=(
            "Buffer radius in meters around road centerlines for data-coverage proxy "
            "(used in buildings+roads territory coverage checks)."
        ),
    )
    parser.add_argument(
        "--min-data-coverage-pct",
        type=float,
        default=5.0,
        help=(
            "Warning threshold for combined buildings+roads area coverage (percent of analysis territory). "
            "Run continues even if below this threshold."
        ),
    )
    parser.add_argument(
        "--spec-dir",
        default=str(CONFIG_DIR),
        help="Directory containing aggregated spatial pipeline JSON specs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where joint outputs will be written. "
            "Default: aggregated_spatial_pipeline/outputs/joint/<place_slug>."
        ),
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.center_node_id is not None and args.relation_id is not None:
        raise SystemExit("Use at most one territory override: --center-node-id or --relation-id.")
    if args.relation_geometry and args.relation_id is None:
        raise SystemExit("--relation-geometry requires --relation-id to label outputs.")

    has_auto_source = bool(args.place or args.center_node_id is not None or args.relation_id is not None)
    has_min_explicit = all([args.blocks, args.street_grid, args.cities])
    if has_auto_source:
        if not args.place:
            raise SystemExit(
                "Automatic data collection now requires --place because city centres are resolved only "
                "from SimpleMaps worldcities.csv."
            )
        return
    if has_min_explicit:
        if args.collect_only:
            raise SystemExit("--collect-only is supported only with automatic data collection mode.")
        return
    raise SystemExit(
        "Provide --place (optionally with --center-node-id/--relation-id override) for automatic data collection OR explicit layer paths: "
        "--blocks --street-grid --cities [--climate-grid]."
    )


def _prune_climate_from_spec(spec: PipelineSpec) -> tuple[PipelineSpec, dict]:
    pruned = copy.deepcopy(spec)

    climate_crosswalk_ids = {
        crosswalk["crosswalk_id"]
        for crosswalk in pruned.crosswalks["crosswalks"]
        if "climate_grid" in (crosswalk["source_layer"], crosswalk["target_layer"])
    }

    pruned.layers["layers"] = [
        layer for layer in pruned.layers["layers"] if layer["layer_id"] != "climate_grid"
    ]
    pruned.crosswalks["crosswalks"] = [
        crosswalk
        for crosswalk in pruned.crosswalks["crosswalks"]
        if crosswalk["crosswalk_id"] not in climate_crosswalk_ids
    ]

    removed_rule_ids = {
        rule["rule_id"]
        for rule in pruned.transfer_rules["rules"]
        if rule["crosswalk_id"] in climate_crosswalk_ids
        or rule["attribute"] in {"temperature_mean", "flood_risk_index"}
    }
    pruned.transfer_rules["rules"] = [
        rule for rule in pruned.transfer_rules["rules"] if rule["rule_id"] not in removed_rule_ids
    ]

    for scenario in pruned.scenarios["scenarios"]:
        scenario["operations"] = [
            operation
            for operation in scenario["operations"]
            if not (
                operation.get("kind") == "attribute_transfer"
                and operation.get("rule_id") in removed_rule_ids
            )
        ]

    meta = {
        "climate_enabled": False,
        "removed_crosswalk_ids": sorted(climate_crosswalk_ids),
        "removed_rule_ids": sorted(removed_rule_ids),
    }
    return pruned, meta


def _log(message: str) -> None:
    logger.bind(tag="[joint-pipeline]").info(message)


def _warn(message: str) -> None:
    logger.bind(tag="[joint-pipeline]").warning(message)


def _log_data_sources_summary(source_details: dict[str, dict]) -> None:
    if not source_details:
        return
    lines = ["Data sources summary:"]
    for layer_id, detail in source_details.items():
        input_path = detail.get("input_path") or "n/a"
        try:
            input_label = Path(str(input_path)).name
        except Exception:
            input_label = str(input_path)
        parts = [f"{layer_id}={input_label}"]
        origin = detail.get("origin")
        if origin:
            parts.append(f"origin={origin}")
        manifest_path = detail.get("manifest_path")
        if manifest_path:
            try:
                manifest_label = Path(str(manifest_path)).name
            except Exception:
                manifest_label = str(manifest_path)
            parts.append(f"manifest={manifest_label}")
        lines.append(" | ".join(parts))
    _log("\n".join(lines))


def _configure_logging() -> None:
    configure_logger("[joint-pipeline]")


def _configure_osm_requests(timeout_s: float, *, debug: bool = False, overpass_url: str | None = None) -> None:
    try:
        import osmnx as ox
    except Exception:
        return
    timeout_int = max(10, int(float(timeout_s)))
    ox.settings.requests_timeout = timeout_int
    # Keep other Overpass defaults, but force timeout in the query header.
    ox.settings.overpass_settings = f"[out:json][timeout:{timeout_int}]"
    ox.settings.overpass_rate_limit = True
    if overpass_url:
        ox.settings.overpass_url = str(overpass_url).strip()
    ox.settings.log_console = bool(debug)
    if hasattr(ox.settings, "log_level"):
        ox.settings.log_level = 20
    if debug:
        endpoint = getattr(ox.settings, "overpass_url", "default")
        _log(
            "OSMnx debug logs enabled "
            f"(timeout={timeout_int}s, overpass_rate_limit=True, overpass_url={endpoint})."
        )


def _validate_layer_input_path(path: Path, layer_id: str) -> None:
    raw = str(path).strip()
    if raw in {"...", ""} or "..." in raw:
        raise SystemExit(
            f"Invalid path for layer {layer_id!r}: {raw!r}. "
            "Replace placeholders with a real Parquet/GeoJSON/GPKG path."
        )
    if not path.exists():
        raise SystemExit(
            f"Input file for layer {layer_id!r} does not exist: {path}"
        )
    if path.is_dir():
        raise SystemExit(
            f"Input path for layer {layer_id!r} is a directory, expected a geospatial file: {path}"
        )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def _normalize_lookup_token(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_COUNTRY_ALIASES = {
    "us": "united states",
    "usa": "united states",
    "u s a": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "u k": "united kingdom",
    "uae": "united arab emirates",
    "u a e": "united arab emirates",
    "czech republic": "czechia",
}


def _normalize_country_token(value: str | None) -> str:
    key = _normalize_lookup_token(value)
    if not key:
        return ""
    return _COUNTRY_ALIASES.get(key, key)


def _parse_place_tokens(place: str) -> tuple[str, str, str, set[str]]:
    parts = [part.strip() for part in str(place).split(",") if part.strip()]
    if not parts:
        return "", "", "", set()

    city_key = _normalize_lookup_token(parts[0])
    admin_key = ""
    country_key = ""
    region_tokens: set[str] = set()
    if len(parts) >= 3:
        admin_key = _normalize_lookup_token(parts[-2])
        country_key = _normalize_country_token(parts[-1])
        region_tokens.add(_normalize_lookup_token(parts[-1]))
        region_tokens.add(_normalize_country_token(parts[-1]))
    elif len(parts) == 2:
        admin_key = _normalize_lookup_token(parts[-1])
        country_key = _normalize_country_token(parts[-1])
        region_tokens.add(_normalize_lookup_token(parts[-1]))
        region_tokens.add(_normalize_country_token(parts[-1]))

    region_tokens.discard("")
    return city_key, admin_key, country_key, region_tokens


def _resolve_city_center_from_worldcities(
    *,
    place: str,
    city_centers_csv: Path,
) -> dict:
    if not city_centers_csv.exists():
        raise FileNotFoundError(
            f"City centres CSV was not found: {city_centers_csv}. "
            "Expected SimpleMaps worldcities.csv."
        )
    required_columns = [
        "city",
        "city_ascii",
        "lat",
        "lng",
        "country",
        "iso2",
        "iso3",
        "admin_name",
        "capital",
        "population",
        "id",
    ]
    cities = pd.read_csv(city_centers_csv, usecols=required_columns, dtype=str)
    if cities.empty:
        raise ValueError(f"City centres CSV is empty: {city_centers_csv}")

    cities["lat_num"] = pd.to_numeric(cities["lat"], errors="coerce")
    cities["lng_num"] = pd.to_numeric(cities["lng"], errors="coerce")
    cities = cities[
        cities["lat_num"].between(-90.0, 90.0)
        & cities["lng_num"].between(-180.0, 180.0)
    ].copy()
    if cities.empty:
        raise ValueError(f"City centres CSV has no valid coordinate rows: {city_centers_csv}")

    city_source = cities["city_ascii"].fillna("").astype(str).str.strip()
    city_fallback = cities["city"].fillna("").astype(str).str.strip()
    cities["city_key"] = city_source.where(city_source.ne(""), city_fallback).map(_normalize_lookup_token)
    cities["admin_key"] = cities["admin_name"].map(_normalize_lookup_token)
    cities["country_key"] = cities["country"].map(_normalize_country_token)
    cities["iso2_key"] = cities["iso2"].map(_normalize_lookup_token)
    cities["iso3_key"] = cities["iso3"].map(_normalize_lookup_token)
    cities["population_num"] = pd.to_numeric(cities["population"], errors="coerce").fillna(-1.0)

    city_key, admin_key, country_key, region_tokens = _parse_place_tokens(place)
    if not city_key:
        raise ValueError(f"Cannot parse city token from place: {place!r}")

    candidates = cities[cities["city_key"] == city_key].copy()
    if candidates.empty:
        raise ValueError(
            "City centre lookup in worldcities.csv failed: "
            f"no city match for {place!r}."
        )

    candidates["score"] = 0
    if country_key:
        candidates.loc[candidates["country_key"] == country_key, "score"] += 50
    if region_tokens:
        candidates.loc[candidates["country_key"].isin(region_tokens), "score"] += 30
        candidates.loc[candidates["iso2_key"].isin(region_tokens), "score"] += 30
        candidates.loc[candidates["iso3_key"].isin(region_tokens), "score"] += 30
    if admin_key:
        candidates.loc[candidates["admin_key"] == admin_key, "score"] += 20

    capital_rank = (
        candidates["capital"]
        .fillna("")
        .str.strip()
        .str.lower()
        .map({"primary": 2, "admin": 1})
        .fillna(0)
    )
    candidates["capital_rank"] = pd.to_numeric(capital_rank, errors="coerce").fillna(0).astype(int)

    candidates = candidates.sort_values(
        by=["score", "population_num", "capital_rank"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top = candidates.iloc[0]
    return {
        "place_query": place,
        "matched_city": str(top.get("city_ascii") or top.get("city") or "").strip(),
        "matched_admin": str(top.get("admin_name") or "").strip(),
        "matched_country": str(top.get("country") or "").strip(),
        "matched_row_id": str(top.get("id") or "").strip(),
        "lon": float(top["lng_num"]),
        "lat": float(top["lat_num"]),
        "source_csv": str(city_centers_csv),
    }


def _resolve_joint_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if args.place:
        slug = _slugify(str(args.place))
    elif args.center_node_id is not None:
        slug = f"osm_node_{int(args.center_node_id)}"
    elif args.relation_id is not None:
        slug = f"osm_relation_{int(args.relation_id)}"
    else:
        raise ValueError("Provide --output-dir explicitly when running run_joint without --place/--center-node-id/--relation-id.")
    return (repo_root / "aggregated_spatial_pipeline" / "outputs" / "joint" / slug).resolve()


def _collect_required_scenarios(spec: PipelineSpec, scenario_id: str) -> list[str]:
    scenario_by_id = {scenario["scenario_id"]: scenario for scenario in spec.scenarios["scenarios"]}
    if scenario_id not in scenario_by_id:
        available = ", ".join(sorted(scenario_by_id))
        raise KeyError(f"Scenario {scenario_id!r} is missing in spec. Available scenarios: {available}")

    required: list[str] = []

    def visit(current_scenario_id: str) -> None:
        if current_scenario_id in required:
            return
        scenario = scenario_by_id[current_scenario_id]
        for operation in scenario["operations"]:
            if operation.get("kind") == "copy_from":
                visit(operation["scenario_id"])
        required.append(current_scenario_id)

    visit(scenario_id)
    return required


def _try_load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_manifest(path: Path, max_levels: int = 5) -> tuple[Path, dict] | None:
    current = path.resolve()
    if current.is_file():
        current = current.parent

    for _ in range(max_levels):
        candidate = current / "manifest.json"
        if candidate.exists():
            payload = _try_load_json(candidate)
            if payload is not None:
                return candidate, payload
        if current.parent == current:
            break
        current = current.parent
    return None


def _detect_city_from_inputs(layer_inputs: dict[str, Path]) -> tuple[str | None, dict[str, dict]]:
    city_candidates: list[str] = []
    source_details: dict[str, dict] = {}

    for layer_id, path in layer_inputs.items():
        detail = {
            "input_path": str(path.resolve()),
            "manifest_path": None,
            "place": None,
            "slug": None,
            "downloaded_in_this_run": False,
            "cached": True,
        }
        discovered = _discover_manifest(path)
        if discovered is not None:
            manifest_path, payload = discovered
            detail["manifest_path"] = str(manifest_path)
            detail["place"] = payload.get("place")
            detail["slug"] = payload.get("slug")
            if isinstance(payload.get("place"), str) and payload["place"].strip():
                city_candidates.append(payload["place"].strip())
        source_details[layer_id] = detail

    if not city_candidates:
        return None, source_details
    detected_city = max(set(city_candidates), key=city_candidates.count)
    return detected_city, source_details


def _create_regular_grid(boundary_gdf: gpd.GeoDataFrame, step_m: float, cell_prefix: str) -> gpd.GeoDataFrame:
    boundary_local = boundary_gdf.to_crs(boundary_gdf.estimate_utm_crs() or "EPSG:3857")
    unified = boundary_local.union_all()
    minx, miny, maxx, maxy = unified.bounds

    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append(box(x, y, x + step_m, y + step_m))
            y += step_m
        x += step_m

    grid = gpd.GeoDataFrame({"geometry": cells}, crs=boundary_local.crs)
    grid = gpd.overlay(grid, boundary_local[["geometry"]], how="intersection", keep_geom_type=False)
    grid = grid[grid.geometry.notna() & ~grid.geometry.is_empty].reset_index(drop=True)
    grid["cell_id"] = [f"{cell_prefix}_{idx}" for idx in range(len(grid))]
    return grid


def _ensure_street_grid_from_repo(
    *,
    place: str,
    repo_root: Path,
    data_root: Path,
    no_cache: bool,
    buffer_m: float,
    grid_step: float,
    min_road_count: int,
    min_total_road_length: float,
    relation_id: int | None = None,
    center_node_id: int | None = None,
    boundary_path: Path | None = None,
    roads_path: Path | None = None,
) -> tuple[Path, Path, bool]:
    experiments_dir = repo_root / "segregation-by-design-experiments"
    city_slug = _slugify(place)

    summary_output = data_root / "street_pattern" / f"{city_slug}_summary.json"
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    predicted_cells_path = summary_output.parent / city_slug / "predicted_cells.geojson"

    if predicted_cells_path.exists() and summary_output.exists() and not no_cache:
        summary_payload = _try_load_json(summary_output)
        cached_buffer = None
        cached_grid_step = None
        cached_min_road_count = None
        cached_min_total_road_length = None
        if isinstance(summary_payload, dict):
            cached_buffer = summary_payload.get("buffer_m")
            cached_grid_step = summary_payload.get("grid_step")
            cached_min_road_count = summary_payload.get("min_road_count")
            cached_min_total_road_length = summary_payload.get("min_total_road_length")
        buffer_matches = cached_buffer is not None and abs(float(cached_buffer) - float(buffer_m)) <= 1e-6
        grid_matches = cached_grid_step is not None and abs(float(cached_grid_step) - float(grid_step)) <= 1e-6
        road_count_matches = (
            cached_min_road_count is not None and int(cached_min_road_count) == int(min_road_count)
        )
        road_length_matches = (
            cached_min_total_road_length is not None
            and abs(float(cached_min_total_road_length) - float(min_total_road_length)) <= 1e-6
        )
        if buffer_matches and grid_matches and road_count_matches and road_length_matches:
            _log(f"Using cached street grid from repo outputs: {_log_name(predicted_cells_path)}")
            return predicted_cells_path, summary_output, False
        _warn(
            "Cached street-grid summary does not match current request "
            f"(buffer cached={cached_buffer}, requested={float(buffer_m)}; "
            f"grid_step cached={cached_grid_step}, requested={float(grid_step)}; "
            f"min_road_count cached={cached_min_road_count}, requested={int(min_road_count)}; "
            f"min_total_road_length cached={cached_min_total_road_length}, "
            f"requested={float(min_total_road_length)}). "
            "Rebuilding classification."
        )

    script_path = experiments_dir / "run_street_pattern_city.py"
    street_python = street_pattern_python(repo_root)
    if not street_python.exists():
        raise FileNotFoundError(
            f"Street-pattern runtime was not found: {street_python}. "
            "Create the dedicated segregation-by-design-experiments environment first."
        )
    command = [
        str(street_python),
        str(script_path),
        "--place",
        place,
        "--device",
        "cpu",
        "--output",
        str(summary_output),
        "--buffer-m",
        str(float(buffer_m)),
        "--grid-step",
        str(float(grid_step)),
        "--min-road-count",
        str(int(min_road_count)),
        "--min-total-road-length",
        str(float(min_total_road_length)),
    ]
    if roads_path is not None and roads_path.exists():
        roads_for_street_pattern = roads_path
        if roads_path.suffix.lower() == ".parquet":
            roads_for_street_pattern = roads_path.with_name(f"{roads_path.stem}_street_pattern.geojson")
            if not roads_for_street_pattern.exists():
                roads_local = read_geodata(roads_path)
                roads_for_street_pattern.parent.mkdir(parents=True, exist_ok=True)
                roads_local.to_file(roads_for_street_pattern, driver="GeoJSON")
            _log(
                "Street-pattern compatibility export prepared: "
                f"{_log_name(roads_for_street_pattern)} (source parquet: {_log_name(roads_path)})"
            )
        command.extend(["--road-source", "local", "--roads", str(roads_for_street_pattern)])
    if center_node_id is not None:
        command.extend(["--center-node-id", str(int(center_node_id))])
    if relation_id is not None:
        command.extend(["--relation-id", str(int(relation_id))])
    if boundary_path is not None and boundary_path.exists():
        command.extend(["--boundary-path", str(boundary_path)])
    if no_cache:
        command.append("--no-cache")

    _log("Classification module start: segregation-by-design street-pattern inference.")
    _log(
        "Classification config: "
        f"place={place}, road_source={'local' if roads_path is not None and roads_path.exists() else 'osm'}, "
        f"buffer_m={buffer_m:.1f}, grid_step={grid_step:.1f}, "
        f"min_road_count={int(min_road_count)}, min_total_road_length={float(min_total_road_length):.1f}, "
        f"output={summary_output.name}, roads={_log_name(roads_for_street_pattern) if roads_path is not None and roads_path.exists() else 'osm'}"
    )
    started = time.time()
    subprocess.run(command, check=True, cwd=str(repo_root))
    _log(
        "Classification module finished in "
        f"{time.time() - started:.1f}s. Summary: {_log_name(summary_output)}"
    )

    summary_data = _try_load_json(summary_output) or {}
    num_predictions = int(pd.to_numeric(summary_data.get("num_predictions"), errors="coerce") or 0)
    num_subgraphs = int(pd.to_numeric(summary_data.get("num_subgraphs"), errors="coerce") or 0)
    if num_predictions <= 0:
        raise RuntimeError(
            "Street-pattern pipeline finished with zero predicted cells "
            f"(num_subgraphs={num_subgraphs}, num_predictions={num_predictions}, summary={summary_output}). "
            "This usually means the current territory/filtering is too sparse for classification; "
            "try a larger buffer or lower street-pattern thresholds."
        )

    if not predicted_cells_path.exists():
        raise RuntimeError(
            "Street-pattern pipeline finished, but predicted_cells.geojson was not found at "
            f"{predicted_cells_path} even though num_predictions={num_predictions}."
        )
    _log(f"Classification artifact: {_log_name(predicted_cells_path)}")
    return predicted_cells_path, summary_output, True


def _run_pt_street_pattern_dependency_step(
    *,
    repo_root: Path,
    city_dir: Path,
    top_routes: int,
    subway_stop_buffer_m: float,
) -> tuple[Path, dict]:
    output_dir = city_dir / "pt_street_pattern_dependency"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    env.setdefault("MPLCONFIGDIR", _repo_mplconfigdir(repo_root, "mpl-pt-street-dependency"))

    command = [
        sys.executable,
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_pt_street_pattern_dependency",
        "--joint-input-dir",
        str(city_dir),
        "--output-dir",
        str(output_dir),
        "--top-routes",
        str(int(top_routes)),
        "--subway-stop-buffer-m",
        str(float(subway_stop_buffer_m)),
    ]
    _log(
        "PT x street-pattern dependency step start: "
        f"city_dir={_log_name(city_dir)}, top_routes={int(top_routes)}, "
        f"subway_stop_buffer_m={float(subway_stop_buffer_m):.1f}"
    )
    started = time.time()
    subprocess.run(command, check=True, cwd=str(repo_root), env=env)
    _log(f"PT x street-pattern dependency step finished in {time.time() - started:.1f}s.")

    manifest = _try_load_json(manifest_path)
    if manifest is None:
        raise RuntimeError(f"Cannot read PT x street-pattern dependency manifest: {manifest_path}")
    return manifest_path, manifest


def _ensure_shared_drive_roads(
    *,
    buffer_path: Path,
    output_path: Path,
    no_cache: bool,
) -> tuple[Path, int, bool]:
    compat_geojson = output_path.with_name(f"{output_path.stem}_street_pattern.geojson")
    if output_path.exists() and (not no_cache):
        if not compat_geojson.exists():
            cached_for_compat = read_geodata(output_path)
            compat_geojson.parent.mkdir(parents=True, exist_ok=True)
            cached_for_compat.to_file(compat_geojson, driver="GeoJSON")
        cached = read_geodata(output_path)
        return output_path, int(len(cached)), False

    import osmnx as ox

    boundary_gdf = read_geodata(buffer_path)
    if boundary_gdf.empty:
        raise RuntimeError(f"Analysis buffer is empty: {buffer_path}")
    boundary_geom = boundary_gdf.union_all()
    if boundary_gdf.crs is not None and str(boundary_gdf.crs) != "EPSG:4326":
        boundary_geom = gpd.GeoSeries([boundary_geom], crs=boundary_gdf.crs).to_crs(4326).iloc[0]

    graph = ox.graph_from_polygon(
        boundary_geom,
        network_type="drive",
        retain_all=True,
        truncate_by_edge=True,
    )
    _, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)
    edges = edges[edges.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    boundary_clip = gpd.GeoDataFrame({"geometry": [boundary_geom]}, crs=4326)
    if edges.crs != boundary_clip.crs:
        boundary_clip = boundary_clip.to_crs(edges.crs)
    edges = edges.clip(boundary_clip)
    edges = edges[edges.geometry.notna() & ~edges.geometry.is_empty].reset_index(drop=True)
    for col in [c for c in edges.columns if c != "geometry"]:
        edges[col] = edges[col].map(
            lambda v: json.dumps(list(v), ensure_ascii=False)
            if isinstance(v, (list, tuple, set))
            else (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_geodata_for_parquet(edges).to_parquet(output_path)
    edges.to_file(compat_geojson, driver="GeoJSON")
    return output_path, int(len(edges)), True


def _compute_data_coverage_metrics(
    *,
    territory_path: Path,
    buildings_path: Path,
    roads_path: Path,
    roads_buffer_m: float,
) -> dict:
    territory = read_geodata(territory_path)
    if territory.empty:
        raise RuntimeError(f"Coverage check failed: empty analysis territory ({territory_path}).")

    local_crs = territory.estimate_utm_crs() or "EPSG:3857"
    territory_local = territory.to_crs(local_crs)
    territory_geom = territory_local.union_all()
    territory_area_m2 = float(territory_geom.area)
    if territory_area_m2 <= 0.0:
        raise RuntimeError(f"Coverage check failed: non-positive territory area ({territory_path}).")
    territory_frame = gpd.GeoDataFrame({"geometry": [territory_geom]}, crs=local_crs)

    buildings_clip = gpd.GeoDataFrame(geometry=[], crs=local_crs)
    buildings_raw = read_geodata(buildings_path)
    if buildings_raw.empty:
        buildings_count = 0
        buildings_area_m2 = 0.0
    else:
        buildings_local = buildings_raw.to_crs(local_crs)
        buildings_local = buildings_local[buildings_local.geometry.notna() & ~buildings_local.geometry.is_empty].copy()
        buildings_local = buildings_local[
            buildings_local.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()
        buildings_count = int(len(buildings_local))
        if buildings_local.empty:
            buildings_area_m2 = 0.0
        else:
            buildings_clip = gpd.clip(buildings_local[["geometry"]], territory_frame, keep_geom_type=False)
            buildings_clip = buildings_clip[buildings_clip.geometry.notna() & ~buildings_clip.geometry.is_empty].copy()
            buildings_area_m2 = float(buildings_clip.union_all().area) if not buildings_clip.empty else 0.0

    roads_buffer_clip = gpd.GeoDataFrame(geometry=[], crs=local_crs)
    roads_raw = read_geodata(roads_path)
    if roads_raw.empty:
        roads_count = 0
        roads_area_m2 = 0.0
    else:
        roads_local = roads_raw.to_crs(local_crs)
        roads_local = roads_local[roads_local.geometry.notna() & ~roads_local.geometry.is_empty].copy()
        roads_local = roads_local[roads_local.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        roads_count = int(len(roads_local))
        if roads_local.empty:
            roads_area_m2 = 0.0
        else:
            roads_buffer = roads_local.copy()
            roads_buffer["geometry"] = roads_buffer.geometry.buffer(float(roads_buffer_m))
            roads_buffer = roads_buffer[roads_buffer.geometry.notna() & ~roads_buffer.geometry.is_empty].copy()
            roads_buffer_clip = gpd.clip(roads_buffer[["geometry"]], territory_frame, keep_geom_type=False)
            roads_buffer_clip = roads_buffer_clip[
                roads_buffer_clip.geometry.notna() & ~roads_buffer_clip.geometry.is_empty
            ].copy()
            roads_area_m2 = float(roads_buffer_clip.union_all().area) if not roads_buffer_clip.empty else 0.0

    combined_geoms = []
    if not buildings_clip.empty:
        combined_geoms.append(buildings_clip.union_all())
    if not roads_buffer_clip.empty:
        combined_geoms.append(roads_buffer_clip.union_all())
    if combined_geoms:
        combined_area_m2 = float(gpd.GeoSeries(combined_geoms, crs=local_crs).union_all().area)
    else:
        combined_area_m2 = 0.0

    return {
        "territory_path": str(territory_path),
        "buildings_path": str(buildings_path),
        "roads_path": str(roads_path),
        "roads_buffer_m": float(roads_buffer_m),
        "territory_area_m2": territory_area_m2,
        "buildings_count": int(buildings_count),
        "roads_count": int(roads_count),
        "buildings_coverage_area_m2": buildings_area_m2,
        "roads_coverage_area_m2": roads_area_m2,
        "combined_data_coverage_area_m2": combined_area_m2,
        "buildings_coverage_pct": (buildings_area_m2 / territory_area_m2) * 100.0,
        "roads_coverage_pct": (roads_area_m2 / territory_area_m2) * 100.0,
        "combined_data_coverage_pct": (combined_area_m2 / territory_area_m2) * 100.0,
    }


def _clip_street_grid_to_buffer(
    *,
    street_grid_path: Path,
    buffer_path: Path,
    output_path: Path,
) -> tuple[Path, int]:
    street_grid = read_geodata(street_grid_path)
    buffer_gdf = read_geodata(buffer_path)
    if street_grid.empty:
        raise RuntimeError(f"Street grid is empty: {street_grid_path}")
    if street_grid.crs != buffer_gdf.crs:
        buffer_gdf = buffer_gdf.to_crs(street_grid.crs)
    clipped = gpd.overlay(street_grid, buffer_gdf[["geometry"]], how="intersection", keep_geom_type=False)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].reset_index(drop=True)
    if clipped.empty:
        raise RuntimeError(
            "Buffer clipping produced empty street-grid layer. "
            "Increase --buffer-m or check classification source."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_parquet(output_path)
    return output_path, len(clipped)


def _relation_get(api, relation_id: int) -> dict:
    if hasattr(api, "relation_get"):
        return api.relation_get(relation_id)
    return api.RelationGet(relation_id)


def _node_get(api, node_id: int) -> dict:
    if hasattr(api, "node_get"):
        return api.node_get(node_id)
    return api.NodeGet(node_id)


def _pick_relation_center_node(api, relation_id: int) -> dict | None:
    relation = _relation_get(api, int(relation_id))
    members = relation.get("member") or relation.get("members") or []
    preferred_roles = ("admin_centre", "admin_center", "label", "capital")
    for role in preferred_roles:
        node_member = next(
            (
                member
                for member in members
                if member.get("type") == "node" and member.get("role") == role
            ),
            None,
        )
        if node_member is not None:
            return _node_get(api, int(node_member["ref"]))
    return None


def _get_relation_boundary_via_overpass(relation_id: int):
    try:
        import osmnx as ox
        from shapely.ops import unary_union

        gdf = ox.geocode_to_gdf(f"R{int(relation_id)}", by_osmid=True)
        if gdf.empty or "geometry" not in gdf:
            raise ValueError(f"osmnx returned empty geometry for relation {relation_id}")
        geoms = gdf.geometry.dropna().tolist()
        if not geoms:
            raise ValueError(f"osmnx returned no valid geometries for relation {relation_id}")
        merged = unary_union(geoms)
        if merged.is_empty:
            raise ValueError(f"osmnx produced empty union for relation {relation_id}")
        return merged
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not resolve relation boundary via osmnx "
            f"(relation_id={relation_id})."
        ) from exc


def _resolve_relation_id_for_place(place: str | None) -> int | None:
    if not place:
        return None
    try:
        import osmnx as ox

        geocoded = ox.geocode_to_gdf(place)
        if geocoded.empty:
            return None
        row = geocoded.iloc[0]
        if str(row.get("osm_type", "")).lower() != "relation":
            return None
        relation_id = row.get("osm_id")
        if relation_id is None:
            return None
        return int(relation_id)
    except Exception:
        return None


def _resolve_analysis_buffer_from_osm(
    *,
    place: str | None,
    buffer_m: float,
    output_path: Path,
    city_centers_csv: Path,
    center_node_id: int | None = None,
    relation_id: int | None = None,
    relation_geometry: Path | None = None,
) -> Path:
    import osmnx as ox
    if not place or not str(place).strip():
        raise ValueError(
            "City centre lookup requires --place because centres are resolved from worldcities.csv."
        )
    city_center = _resolve_city_center_from_worldcities(
        place=str(place).strip(),
        city_centers_csv=city_centers_csv,
    )
    centre_lon = float(city_center["lon"])
    centre_lat = float(city_center["lat"])
    center_point = Point(centre_lon, centre_lat)
    centre_node_id = None
    territory_mode = "node_buffer"
    if relation_geometry is not None:
        relation_gdf = read_geodata(relation_geometry)
        if relation_gdf.empty:
            raise ValueError(f"Relation geometry file is empty: {relation_geometry}")
        relation_geom = relation_gdf.geometry.iloc[0]
        if relation_geom.is_empty:
            raise ValueError(f"Relation geometry is empty: {relation_geometry}")
        if float(buffer_m) > 0.0:
            relation_gdf = gpd.GeoDataFrame({"geometry": [relation_geom]}, crs=4326)
            buffered = relation_gdf.to_crs(3857).buffer(float(buffer_m))
            buffer_geom = gpd.GeoSeries(buffered, crs=3857).to_crs(4326).iloc[0]
            territory_mode = "relation_buffer"
        else:
            buffer_geom = relation_geom
            territory_mode = "relation"
    elif relation_id is not None:
        relation_geom = _get_relation_boundary_via_overpass(int(relation_id))
        if relation_geom.is_empty:
            raise ValueError(f"Empty boundary for relation {relation_id}")
        if float(buffer_m) > 0.0:
            point_gdf = gpd.GeoDataFrame({"geometry": [center_point]}, crs=4326)
            buffered = point_gdf.to_crs(3857).buffer(float(buffer_m))
            buffer_geom = gpd.GeoSeries(buffered, crs=3857).to_crs(4326).iloc[0]
            territory_mode = "place_center_buffer" if place else "relation_buffer"
        else:
            buffer_geom = relation_geom
            territory_mode = "relation"
    elif center_node_id is not None:
        relation_id = None
        point_gdf = gpd.GeoDataFrame({"geometry": [center_point]}, crs=4326)
        buffer_geom = gpd.GeoSeries(point_gdf.to_crs(3857).buffer(float(buffer_m)), crs=3857).to_crs(4326).iloc[0]
        territory_mode = "place_center_buffer"
    else:
        if float(buffer_m) > 0.0:
            # For explicit buffer mode we do not require place->relation geocoding.
            # City center from worldcities.csv is enough to build the analysis territory.
            relation_id = None
            point_gdf = gpd.GeoDataFrame({"geometry": [center_point]}, crs=4326)
            buffer_geom = gpd.GeoSeries(point_gdf.to_crs(3857).buffer(float(buffer_m)), crs=3857).to_crs(4326).iloc[0]
            territory_mode = "place_center_buffer"
        else:
            geocoded = ox.geocode_to_gdf(place)
            if geocoded.empty:
                raise ValueError(f"Could not geocode place: {place}")

            row = geocoded.iloc[0]
            relation_id = row.get("osm_id")
            osm_type = str(row.get("osm_type", "")).lower()
            if relation_id is None or osm_type != "relation":
                raise ValueError(
                    f"Expected a relation for {place}, got osm_type={osm_type!r}, osm_id={relation_id!r}"
                )

            relation_geom = _get_relation_boundary_via_overpass(int(relation_id))
            buffer_geom = relation_geom
            territory_mode = "relation"

    buffer_gdf = gpd.GeoDataFrame(
        [
            {
                "place": place or (f"osm_relation_{int(relation_id)}" if relation_id is not None else f"osm_node_{int(center_node_id)}"),
                "territory_mode": territory_mode,
                "relation_id": int(relation_id) if relation_id is not None else None,
                "centre_node_id": centre_node_id,
                "centre_lon": centre_lon,
                "centre_lat": centre_lat,
                "centre_source": "simplemaps_worldcities_basicv1/worldcities.csv",
                "centre_source_row_id": city_center["matched_row_id"],
                "centre_source_city": city_center["matched_city"],
                "centre_source_admin": city_center["matched_admin"],
                "centre_source_country": city_center["matched_country"],
                "buffer_m": float(buffer_m),
                "geometry": buffer_geom,
            }
        ],
        geometry="geometry",
        crs=4326,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer_gdf.to_parquet(output_path)
    return output_path


def _analysis_buffer_matches(path: Path, expected_buffer_m: float) -> bool:
    if not path.exists():
        return False
    try:
        gdf = read_geodata(path)
        if gdf.empty:
            return False
        if "buffer_m" not in gdf.columns:
            return False
        value = pd.to_numeric(gdf["buffer_m"], errors="coerce").dropna()
        if value.empty:
            return False
        return abs(float(value.iloc[0]) - float(expected_buffer_m)) <= 1e-6
    except Exception:
        return False
def _build_climate_grid(
    *,
    boundary_path: Path,
    output_path: Path,
    step_m: float,
) -> Path:
    boundary = read_geodata(boundary_path)
    grid = _create_regular_grid(boundary, step_m, "climate")
    local = grid.to_crs(grid.estimate_utm_crs() or "EPSG:3857").copy()
    centroids = local.geometry.centroid
    x = centroids.x
    y = centroids.y

    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else pd.Series(0.5, index=local.index)
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else pd.Series(0.5, index=local.index)
    dist = ((x_norm - 0.5) ** 2 + (y_norm - 0.5) ** 2) ** 0.5
    dist_norm = dist / dist.max() if dist.max() > 0 else pd.Series(0.0, index=local.index)

    local["temperature_mean"] = -5.0 + 15.0 * y_norm
    local["flood_risk_index"] = (0.2 + 0.6 * (1.0 - dist_norm)).clip(0.0, 1.0)

    climate_grid = local.to_crs(boundary.crs)[["temperature_mean", "flood_risk_index", "geometry"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    climate_grid.to_parquet(output_path)
    return output_path


def _build_buffered_quarters(
    *,
    blocks_path: Path,
    buffer_path: Path,
    output_path: Path,
) -> tuple[Path, int]:
    blocks = read_geodata(blocks_path)
    buffer_gdf = read_geodata(buffer_path)

    if blocks.crs != buffer_gdf.crs:
        buffer_gdf = buffer_gdf.to_crs(blocks.crs)

    clipped = gpd.overlay(blocks, buffer_gdf[["geometry"]], how="intersection", keep_geom_type=False)
    clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].reset_index(drop=True)
    if clipped.empty:
        raise RuntimeError("Buffer clipping produced empty blocks layer. Increase --buffer-m or check source data.")

    # Normalize expected quarter-level attributes for downstream city aggregation rules.
    population_proxy = pd.to_numeric(clipped.get("population_proxy"), errors="coerce")
    if isinstance(population_proxy, pd.Series) and population_proxy.notna().any():
        # Prefer post-floor proxy population propagated from blocks processing.
        clipped["population_total"] = population_proxy.fillna(0.0)
    elif "population_total" in clipped.columns:
        clipped["population_total"] = pd.to_numeric(clipped["population_total"], errors="coerce").fillna(0.0)
    else:
        clipped["population_total"] = 0.0

    if "service_capacity_total" not in clipped.columns:
        clipped["service_capacity_total"] = 0.0
    else:
        clipped["service_capacity_total"] = pd.to_numeric(
            clipped["service_capacity_total"], errors="coerce"
        ).fillna(0.0)

    if "accessibility_time_mean" not in clipped.columns:
        clipped["accessibility_time_mean"] = 0.0
    else:
        clipped["accessibility_time_mean"] = pd.to_numeric(
            clipped["accessibility_time_mean"], errors="coerce"
        ).fillna(0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_parquet(output_path)
    return output_path, len(clipped)


def _to_numeric_storey(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _floor_output_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        gdf = read_geodata(path)
    except Exception:
        return False
    required_columns = {"is_living_source", "storey_source"}
    return required_columns.issubset(set(gdf.columns))


def _empty_floor_roads(local_crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "osmid": pd.Series(dtype="object"),
            "highway": pd.Series(dtype="string"),
        },
        geometry=gpd.GeoSeries([], crs=local_crs),
        crs=local_crs,
    )


def _empty_floor_amenities(local_crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"amenity_key": pd.Series(dtype="string")},
        geometry=gpd.GeoSeries([], crs=local_crs),
        crs=local_crs,
    )


def _load_local_roads_for_floor(
    *,
    roads_path: Path,
    local_crs: str,
) -> gpd.GeoDataFrame:
    roads_raw = read_geodata(roads_path)
    if roads_raw.empty:
        return _empty_floor_roads(local_crs)
    roads = roads_raw[roads_raw.geometry.notna() & ~roads_raw.geometry.is_empty].copy()
    if roads.empty:
        return _empty_floor_roads(local_crs)
    if roads.crs is None:
        roads = roads.set_crs(4326, allow_override=True)
    roads = roads.to_crs(local_crs)
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if roads.empty:
        return _empty_floor_roads(local_crs)
    if "highway" not in roads.columns:
        roads["highway"] = "service"
    roads["highway"] = roads["highway"].astype("string").fillna("service")
    if "osmid" not in roads.columns:
        roads["osmid"] = pd.Series([pd.NA] * len(roads), index=roads.index, dtype="object")
    keep = [c for c in ["osmid", "highway", "geometry"] if c in roads.columns]
    return roads[keep].reset_index(drop=True)


def _load_local_amenities_for_floor(
    *,
    amenities_path: Path,
    local_crs: str,
) -> gpd.GeoDataFrame:
    amenities_raw = read_geodata(amenities_path)
    if amenities_raw.empty:
        return _empty_floor_amenities(local_crs)
    amenities = amenities_raw[amenities_raw.geometry.notna() & ~amenities_raw.geometry.is_empty].copy()
    if amenities.empty:
        return _empty_floor_amenities(local_crs)
    if amenities.crs is None:
        amenities = amenities.set_crs(4326, allow_override=True)
    amenities = amenities.to_crs(local_crs)
    if "amenity_key" not in amenities.columns:
        if "amenity" in amenities.columns:
            amenities["amenity_key"] = amenities["amenity"].astype("string")
        else:
            amenities["amenity_key"] = pd.Series([pd.NA] * len(amenities), index=amenities.index, dtype="string")
    keep = [c for c in ["amenity_key", "geometry"] if c in amenities.columns]
    return amenities[keep].reset_index(drop=True)


def _model_based_is_living_restore(
    *,
    gdf: gpd.GeoDataFrame,
    model_path: Path,
    overpass_url: str | None,
    osm_timeout_s: int,
    local_roads_path: Path | None = None,
    local_amenities_path: Path | None = None,
) -> gpd.GeoDataFrame:
    import osmnx as ox
    from shapely.ops import unary_union

    from floor_predictior.osm_living_predictor import (
        AmenityProcessor,
        FeatureBuilder,
        ModelHandler,
        RoadProcessor,
    )

    started = time.time()
    out = gdf.copy().reset_index(drop=True)
    if out.crs is None:
        out = out.set_crs(4326)
    local_crs = out.estimate_utm_crs() or "EPSG:3857"
    bounds = unary_union(out.to_crs(4326).geometry.values)
    # Align with original buildings_classification notebooks:
    # build all_tags/all_tag_keys from raw OSM-like columns when absent.
    if "all_tags" not in out.columns or "all_tag_keys" not in out.columns:
        raw_tag_cols = [
            "building",
            "landuse",
            "land_building",
            "source",
            "addr:street",
            "name",
            "building:levels",
            "roof:shape",
            "amenity",
            "brand",
            "website",
            "shop",
        ]
        available_raw = [c for c in raw_tag_cols if c in out.columns]
        if available_raw:
            t0 = time.time()
            tag_frame = out[available_raw].copy()

            def _norm_val(v):
                if pd.isna(v):
                    return None
                if isinstance(v, str):
                    s = v.strip()
                    return s if s else None
                return v

            out["all_tags"] = tag_frame.apply(
                lambda row: {k: vv for k, vv in ((k, _norm_val(row[k])) for k in available_raw) if vv is not None},
                axis=1,
            )
            out["all_tag_keys"] = out["all_tags"].map(lambda d: list(d.keys()) if isinstance(d, dict) else [])
            _log(
                "Floor step: osm_living tag dictionaries prepared from raw columns: "
                f"cols={len(available_raw)} in {time.time() - t0:.1f}s"
            )
    _log(
        "Floor step: osm_living restore init: "
        f"buildings={len(out)}, timeout={int(osm_timeout_s)}s, "
        f"overpass={'default' if not overpass_url else overpass_url}, "
        "context=local_inputs_required"
    )

    ox.settings.timeout = int(osm_timeout_s)
    ox.settings.overpass_rate_limit = True
    ox.settings.use_cache = True
    if overpass_url:
        if hasattr(ox.settings, "overpass_url"):
            ox.settings.overpass_url = str(overpass_url)
        if hasattr(ox.settings, "overpass_endpoint"):
            ox.settings.overpass_endpoint = str(overpass_url)

    if local_roads_path is None:
        raise RuntimeError("Floor step requires local roads input; got no local_roads_path.")
    if not local_roads_path.exists():
        raise RuntimeError(f"Floor step local roads file not found: {local_roads_path}")

    rp = RoadProcessor(bounds=bounds, buildings=out.copy(), radius_list=[30, 60, 90], local_crs=local_crs)
    _log(f"Floor step: osm_living roads: loading local roads ({_log_name(local_roads_path)})...")
    t0 = time.time()
    roads = _load_local_roads_for_floor(roads_path=local_roads_path, local_crs=local_crs)
    rp.roads = roads
    _log(f"Floor step: osm_living roads loaded from local input: {len(roads)} in {time.time() - t0:.1f}s")

    _log("Floor step: osm_living roads: building buffered road features...")
    t0 = time.time()
    rp._buffer_roads()
    rp._join_buildings_to_buffers()
    features_df = rp.build_feature_table()
    _log(f"Floor step: osm_living roads features ready in {time.time() - t0:.1f}s")
    raw_tag_probe = ["source", "addr:street", "name", "amenity", "brand", "website", "shop", "building:levels"]
    present_raw_tags = [c for c in raw_tag_probe if c in features_df.columns]
    if present_raw_tags:
        _log(
            "Floor step: osm_living raw tag columns present: "
            + ", ".join(f"{c}={int(features_df[c].notna().sum())}" for c in present_raw_tags)
        )

    if local_amenities_path is None:
        raise RuntimeError("Floor step requires local amenities input; got no local_amenities_path.")
    if not local_amenities_path.exists():
        raise RuntimeError(f"Floor step local amenities file not found: {local_amenities_path}")

    ap = AmenityProcessor(bounds=bounds, base_gdf=features_df, radii=[30, 60, 90], local_crs=local_crs)
    _log(f"Floor step: osm_living amenities: loading local amenities ({_log_name(local_amenities_path)})...")
    t0 = time.time()
    amenities = _load_local_amenities_for_floor(amenities_path=local_amenities_path, local_crs=local_crs)
    ap.amenities = amenities
    _log(f"Floor step: osm_living amenities loaded from local input: {len(amenities)} in {time.time() - t0:.1f}s")

    _log("Floor step: osm_living amenities: aggregating parking features...")
    t0 = time.time()
    bins = ap._classify_parking_bins(n_classes=3)
    ap._build_parking_buffers(bins)
    feat_with_amenities = ap._join_parking_buffers()
    for radius in (30, 60, 90):
        for parking_bin in (1, 2, 3):
            col = f"near_parking_bin{parking_bin}_r{radius}"
            if col not in feat_with_amenities.columns:
                feat_with_amenities[col] = 0
    _log(f"Floor step: osm_living amenity features ready in {time.time() - t0:.1f}s")

    # floor_predictior FeatureBuilder uses numpy set ops over categorical uniques;
    # sanitize pandas NA upfront to avoid NAType sorting/comparison crashes.
    if "landuse" in feat_with_amenities.columns:
        missing_landuse = int(pd.isna(feat_with_amenities["landuse"]).sum())
        if missing_landuse > 0:
            _log(f"Floor step: osm_living features: filling missing landuse={missing_landuse} -> no_landuse")
            feat_with_amenities["landuse"] = (
                feat_with_amenities["landuse"].astype("string").fillna("no_landuse")
            )
    if "building" in feat_with_amenities.columns:
        missing_building = int(pd.isna(feat_with_amenities["building"]).sum())
        if missing_building > 0:
            _log(f"Floor step: osm_living features: filling missing building={missing_building} -> yes")
            feat_with_amenities["building"] = (
                feat_with_amenities["building"].astype("string").fillna("yes")
            )

    _log("Floor step: osm_living features: building model feature table...")
    t0 = time.time()
    builder = FeatureBuilder(feat_with_amenities)
    model_features = builder.build_features()
    _log(f"Floor step: osm_living model features ready in {time.time() - t0:.1f}s")
    probe_cols = [
        "tag_source",
        "tag_addr:street",
        "tag_name",
        "tag_amenity",
        "tag_brand",
        "tag_website",
        "tag_shop",
        "building:levels",
    ]
    present_probe_cols = [c for c in probe_cols if c in model_features.columns]
    if present_probe_cols:
        _log("Floor step: osm_living key model features present: " + ", ".join(present_probe_cols))

    _log(f"Floor step: osm_living model: loading {_log_name(model_path)}...")
    t0 = time.time()
    handler = ModelHandler(str(model_path), df=model_features.copy(), target_col="is_living")
    handler.load_model_from_file()
    _log(f"Floor step: osm_living model loaded in {time.time() - t0:.1f}s")
    _log("Floor step: osm_living model: predicting is_living...")
    t0 = time.time()
    predicted_df = handler.predict(model_features, map_labels=False)
    if "predicted" not in predicted_df.columns:
        raise RuntimeError("osm_living_predictor did not return 'predicted' column.")
    pred = pd.to_numeric(predicted_df["predicted"], errors="coerce")
    out["is_living"] = out["is_living"].fillna(pred.reindex(out.index))
    _log(
        "Floor step: osm_living prediction finished in "
        f"{time.time() - t0:.1f}s; restored={int(out['is_living'].notna().sum())}/{len(out)}; "
        f"total_elapsed={time.time() - started:.1f}s"
    )
    return out


def _run_floor_predictor_preprocessing(
    *,
    repo_root: Path,
    buildings_path: Path,
    land_use_path: Path,
    output_path: Path,
    local_roads_path: Path | None = None,
    local_amenities_path: Path | None = None,
    is_living_model_path: str | None = None,
    overpass_url: str | None = None,
    osm_timeout_s: int = 180,
    floor_ignore_missing_below_pct: float = 2.0,
    is_living_only: bool = False,
) -> dict:
    started_total = time.time()
    _log("Floor step: initializing dedicated floor-predictor runtime...")
    floor_repo = repo_root / "floor-predictor"
    floor_pkg = floor_repo / "floor_predictior"
    if not floor_pkg.exists():
        raise RuntimeError(f"floor-predictor package not found: {floor_pkg}")

    if str(floor_repo) not in sys.path:
        sys.path.insert(0, str(floor_repo))

    from floor_predictior.osm_height_predictor.geo import (
        GeometryFeatureGenerator,
        SpatialNeighborhoodAnalyzer,
        StoreyModelTrainer,
    )
    _log("Floor step: floor-predictor modules imported.")

    _log(f"Floor step: reading buildings layer: {_log_name(buildings_path)}")
    t0 = time.time()
    buildings = read_geodata(buildings_path)
    _log(f"Floor step: buildings loaded ({len(buildings)} rows) in {time.time() - t0:.1f}s")
    _log(f"Floor step: reading land-use layer: {_log_name(land_use_path)}")
    t0 = time.time()
    land_use = read_geodata(land_use_path)
    _log(f"Floor step: land-use loaded ({len(land_use)} rows) in {time.time() - t0:.1f}s")
    if buildings.empty:
        raise RuntimeError(f"BlocksNet buildings layer is empty: {buildings_path}")

    if buildings.crs is None:
        buildings = buildings.set_crs(4326)

    gdf = buildings.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise RuntimeError("No polygon buildings available for floor-predictor preprocessing.")
    gdf = gdf.reset_index(drop=True)
    _log(f"Floor step: valid polygon buildings for processing: {len(gdf)}")

    # 1) Restore missing is_living using the floor_predictior model-based path.
    if "is_living" not in gdf.columns:
        gdf["is_living"] = pd.Series([pd.NA] * len(gdf), dtype="Float64")
    else:
        gdf["is_living"] = pd.to_numeric(gdf["is_living"], errors="coerce")
    gdf["is_living_source"] = pd.Series("missing", index=gdf.index, dtype="string")
    gdf.loc[gdf["is_living"].notna(), "is_living_source"] = "original_is_living"
    missing_living_mask_before = gdf["is_living"].isna()
    missing_living_before = int(gdf["is_living"].isna().sum())
    resolved_is_living_model_path = (
        Path(is_living_model_path).resolve()
        if is_living_model_path
        else (floor_repo / "floor_predictior" / "model" / "model_dt.pkl").resolve()
    )
    if missing_living_before > 0:
        if not resolved_is_living_model_path.exists():
            raise RuntimeError(
                "Missing trained is_living model for floor_predictior path: "
                f"{resolved_is_living_model_path}"
            )
        _log(
            "Floor step: running model-based is_living restoration "
            f"(model={_log_name(resolved_is_living_model_path)})..."
        )
        t0 = time.time()
        gdf = _model_based_is_living_restore(
            gdf=gdf,
            model_path=resolved_is_living_model_path,
            overpass_url=overpass_url,
            osm_timeout_s=int(osm_timeout_s),
            local_roads_path=local_roads_path,
            local_amenities_path=local_amenities_path,
        )
        _log(f"Floor step: model-based is_living restoration finished in {time.time() - t0:.1f}s")
    local_crs = gdf.estimate_utm_crs() or "EPSG:3857"
    missing_living_after = int(gdf["is_living"].isna().sum())
    restored_living_mask = missing_living_mask_before & gdf["is_living"].notna()
    gdf.loc[restored_living_mask, "is_living_source"] = "osm_living_model"
    gdf["is_living_restored"] = (gdf["is_living_source"] == "osm_living_model").astype(int)
    original_living_count = int((gdf["is_living_source"] == "original_is_living").sum())
    original_living_from_building_tag_count = 0
    restored_living_count = int((gdf["is_living_source"] == "osm_living_model").sum())
    _log(
        "Floor step: is_living source after preprocessing: "
        f"original_is_living={int((gdf['is_living_source'] == 'original_is_living').sum())}, "
        f"osm_living_model={restored_living_count}, "
        f"still_missing={missing_living_after}"
    )

    # 2) Storey path (can be disabled for is_living-only runs).
    model_path = floor_repo / "floor_predictior" / "model" / "StoreyModelTrainer.joblib"
    storey_model_info: dict | None = None
    predicted_storey_count = 0
    prediction_skipped_by_threshold = False
    target_missing_count = 0
    target_missing_pct = 0.0

    if "storey" not in gdf.columns:
        gdf["storey"] = pd.Series([pd.NA] * len(gdf), dtype="Float64")
    gdf["storey"] = pd.to_numeric(gdf["storey"], errors="coerce")
    gdf["storey_source"] = pd.Series("missing", index=gdf.index, dtype="string")
    gdf.loc[gdf["storey"].notna(), "storey_source"] = "original_storey"
    original_storey_count = int((gdf["storey_source"] == "original_storey").sum())
    filled_from_building_levels_count = 0

    if is_living_only:
        _log("Floor step: is_living-only mode enabled, skipping storey enrichment/model inference.")
        missing_storey_before_model = int(gdf["storey"].isna().sum())
        missing_storey_after_model = missing_storey_before_model
        gdf["storey_restored"] = 0
        prediction_skipped_by_threshold = True
    else:
        if "building:levels" in gdf.columns:
            _log("Floor step: normalizing storey from building:levels...")
            parsed_levels = _to_numeric_storey(gdf["building:levels"])
            fill_from_levels_mask = gdf["storey"].isna() & parsed_levels.notna()
            gdf.loc[fill_from_levels_mask, "storey"] = parsed_levels.loc[fill_from_levels_mask]
            gdf.loc[fill_from_levels_mask, "storey_source"] = "osm_building_levels"
            filled_from_building_levels_count = int(fill_from_levels_mask.sum())

        missing_storey_before_model = int(gdf["storey"].isna().sum())
        _log(
            "Floor step: storey source before model: "
            f"original_storey={original_storey_count}, "
            f"osm_building_levels={filled_from_building_levels_count}, "
            f"still_missing={missing_storey_before_model}"
        )

        # 3) Predict missing storey for living buildings using pre-trained height model.
        target_mask = gdf["storey"].isna() & (gdf["is_living"].fillna(0) >= 0.5)
        target_missing_count = int(target_mask.sum())
        target_missing_pct = (100.0 * target_missing_count / max(1, len(gdf)))
        _log(
            "Floor step: missing storey requiring model prediction: "
            f"{target_missing_count}/{len(gdf)} ({target_missing_pct:.2f}%)."
        )
        if float(floor_ignore_missing_below_pct) > 0 and target_missing_pct < float(floor_ignore_missing_below_pct):
            _warn(
                "Floor step: skipping storey model inference because missing share is below threshold "
                f"({target_missing_pct:.2f}% < {float(floor_ignore_missing_below_pct):.2f}%)."
            )
            prediction_skipped_by_threshold = True

        if model_path.exists():
            if not prediction_skipped_by_threshold:
                _log(f"Floor step: loading storey model: {_log_name(model_path)}")
                model = StoreyModelTrainer.load_model(str(model_path))
                storey_model_info = getattr(model, "info", None)
                _log("Floor step: computing geometry features for storey model...")
                t0 = time.time()
                features_df = GeometryFeatureGenerator(gdf.copy()).compute_geometry_features()
                _log(f"Floor step: geometry features ready in {time.time() - t0:.1f}s")
                analyzer = SpatialNeighborhoodAnalyzer(features_df, radius=500)
                _log("Floor step: computing neighborhood metrics (this can be slow on large areas)...")
                t0 = time.time()
                features_df, _ = analyzer.compute_neighborhood_metrics(plot=False, show_progress=True)
                _log(f"Floor step: neighborhood metrics ready in {time.time() - t0:.1f}s")

                if target_missing_count > 0:
                    _log(f"Floor step: predicting missing storey for {target_missing_count} buildings...")
                    t0 = time.time()
                    pred_values = model.predict(features_df.loc[target_mask].copy())
                    pred_series = pd.Series(pred_values, index=gdf.index[target_mask], dtype="float64")
                    pred_series = pred_series.round().clip(lower=1)
                    gdf.loc[target_mask, "storey"] = pred_series
                    gdf.loc[target_mask, "storey_source"] = "model_predicted"
                    predicted_storey_count = target_missing_count
                    _log(f"Floor step: storey prediction complete in {time.time() - t0:.1f}s")
        else:
            _warn(f"Floor step: storey model not found, skipping prediction: {_log_name(model_path)}")

        missing_storey_after_model = int(gdf["storey"].isna().sum())
        gdf["storey_restored"] = (gdf["storey_source"] == "model_predicted").astype(int)

    storey_missing_mask_after = gdf["storey"].isna() | gdf["storey"].le(0)
    living_numeric_after = pd.to_numeric(gdf.get("is_living"), errors="coerce")
    living_mask_after = living_numeric_after.ge(0.5).fillna(False)
    non_living_mask_after = living_numeric_after.lt(0.5).fillna(False)
    unknown_living_mask_after = living_numeric_after.isna()
    storey_missing_after_model_living = int((storey_missing_mask_after & living_mask_after).sum())
    storey_missing_after_model_non_living = int((storey_missing_mask_after & non_living_mask_after).sum())
    storey_missing_after_model_unknown_living = int((storey_missing_mask_after & unknown_living_mask_after).sum())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Floor step: writing enriched buildings layer: {_log_name(output_path)}")
    gdf.to_parquet(output_path)
    _log(f"Floor step: preprocessing finished in {time.time() - started_total:.1f}s")

    return {
        "output_path": str(output_path),
        "model_path": str(model_path),
        "rows_total": int(len(gdf)),
        "is_living_missing_before": missing_living_before,
        "is_living_missing_after": missing_living_after,
        "is_living_original_count": original_living_count,
        "is_living_original_from_building_tag_count": original_living_from_building_tag_count,
        "is_living_restored_count": int(gdf["is_living_restored"].sum()),
        "is_living_restore_method": "osm_living_model" if int(gdf["is_living_restored"].sum()) > 0 else "none",
        "storey_original_count": original_storey_count,
        "storey_filled_from_building_levels_count": filled_from_building_levels_count,
        "storey_missing_before_model": missing_storey_before_model,
        "storey_prediction_target_count": target_missing_count,
        "storey_prediction_target_pct": target_missing_pct,
        "storey_prediction_skipped_by_threshold": prediction_skipped_by_threshold,
        "storey_prediction_skip_threshold_pct": float(floor_ignore_missing_below_pct),
        "storey_predicted_by_model": predicted_storey_count,
        "storey_missing_after_model": missing_storey_after_model,
        "storey_missing_after_model_living": storey_missing_after_model_living,
        "storey_missing_after_model_non_living": storey_missing_after_model_non_living,
        "storey_missing_after_model_unknown_living": storey_missing_after_model_unknown_living,
        "storey_restored_count": int(gdf["storey_restored"].sum()),
        "storey_model_info": storey_model_info,
        "is_living_only": bool(is_living_only),
        "floor_context_roads_input_path": str(local_roads_path) if local_roads_path else None,
        "floor_context_amenities_input_path": str(local_amenities_path) if local_amenities_path else None,
    }


def _save_collection_previews(
    *,
    data_root: Path,
    buffer_path: Path,
    raw_files: dict,
    connectpt_manifest_path: Path,
    intermodal_manifest_path: Path | None,
    blocks_manifest_path: Path,
    buffered_blocks_path: Path,
    street_grid_path: Path,
    floor_enriched_path: Path,
    sm_imputed_path: Path | None = None,
    floor_metrics: dict | None = None,
    sm_imputation_metrics: dict | None = None,
    stage_label: str = "stage",
) -> list[Path]:
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from shapely.geometry import box

    def _visual_only_shrink_buffer(
        buffer_layer: gpd.GeoDataFrame | None,
        shrink_m: float = 5.0,
    ) -> gpd.GeoDataFrame | None:
        if buffer_layer is None or buffer_layer.empty:
            return buffer_layer
        try:
            work = buffer_layer.copy()
            if work.crs is None:
                work = work.set_crs(4326)
            local = work.to_crs(work.estimate_utm_crs() or "EPSG:3857")
            local["geometry"] = local.geometry.buffer(-abs(float(shrink_m)))
            local = local[local.geometry.notna() & ~local.geometry.is_empty].copy()
            if local.empty:
                return buffer_layer
            return local.to_crs(work.crs)
        except Exception:
            return buffer_layer

    def _read(path: Path) -> gpd.GeoDataFrame | None:
        try:
            if not path.exists():
                return None
            gdf = read_geodata(path)
            if gdf.empty:
                return None
            return gdf
        except Exception:
            return None

    def _build_buffer_circle_and_center(buffer_layer: gpd.GeoDataFrame | None) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
        if buffer_layer is None or buffer_layer.empty:
            return None, None
        lon_col = "centre_lon" if "centre_lon" in buffer_layer.columns else "center_lon" if "center_lon" in buffer_layer.columns else None
        lat_col = "centre_lat" if "centre_lat" in buffer_layer.columns else "center_lat" if "center_lat" in buffer_layer.columns else None
        if lon_col is None or lat_col is None or "buffer_m" not in buffer_layer.columns:
            return None, None
        row = buffer_layer.iloc[0]
        try:
            lon = float(row[lon_col])
            lat = float(row[lat_col])
            radius_m = float(row["buffer_m"])
        except Exception:
            return None, None
        if pd.isna(lon) or pd.isna(lat) or pd.isna(radius_m):
            return None, None

        center = gpd.GeoDataFrame(
            [{"geometry": Point(lon, lat)}],
            geometry="geometry",
            crs=4326,
        )
        try:
            local = center.to_crs(3857)
            circle_geom = gpd.GeoSeries(local.geometry.buffer(radius_m), crs=3857).to_crs(4326).iloc[0]
            circle = gpd.GeoDataFrame([{"geometry": circle_geom}], geometry="geometry", crs=4326)
            return circle, center
        except Exception:
            return None, center

    def _legend_bottom(ax, handles: list) -> None:
        legend_bottom(ax, handles)

    def _footer_text(fig, lines: list[str] | None) -> None:
        footer_text(fig, lines)

    def _apply_preview_theme(
        fig,
        ax,
        boundary_layer: gpd.GeoDataFrame | None,
        *,
        title: str | None = None,
    ) -> None:
        apply_preview_canvas(fig, ax, boundary_layer, title=title)

    def _clip_to_preview_boundary(
        gdf: gpd.GeoDataFrame | None,
        boundary_layer: gpd.GeoDataFrame | None,
    ) -> gpd.GeoDataFrame | None:
        return clip_to_preview_boundary(gdf, boundary_layer)

    def _plot(
        output_path: Path,
        *,
        layers: list[tuple[gpd.GeoDataFrame | None, dict]],
        title: str | None = None,
    ) -> Path | None:
        valid_layers = [(gdf, style) for gdf, style in layers if gdf is not None and not gdf.empty]
        if not valid_layers:
            return None
        # Normalize all layers to a single CRS for correct overlay/scale in previews.
        target_crs = "EPSG:3857"
        boundary_norm = normalize_preview_gdf(buffer_gdf, target_crs=target_crs)
        normalized_layers: list[tuple[gpd.GeoDataFrame, dict]] = []
        for gdf, style in valid_layers:
            normalized = normalize_preview_gdf(gdf, boundary_norm, target_crs=target_crs)
            if normalized is not None and not normalized.empty:
                normalized_layers.append((normalized, style))
        fig, ax = plt.subplots(figsize=(12, 12))
        legend_handles = []
        for gdf, style in normalized_layers:
            plot_style = dict(style)
            label = plot_style.pop("label", None)
            geom_types = set(gdf.geom_type.astype(str))
            if any("Polygon" in geom for geom in geom_types):
                plot_style.setdefault("edgecolor", "#d1d5db")
                plot_style.setdefault("linewidth", 0.05)
            gdf.plot(ax=ax, **plot_style)
            if label:
                if any("Point" in g for g in geom_types):
                    marker_color = plot_style.get("color", "#111111")
                    legend_handles.append(
                        Line2D([0], [0], marker="o", color="none", markerfacecolor=marker_color, markersize=7, label=label)
                    )
                elif any("Line" in g for g in geom_types):
                    line_color = plot_style.get("color", "#111111")
                    legend_handles.append(Line2D([0], [0], color=line_color, linewidth=2, label=label))
                else:
                    patch_color = plot_style.get("color", plot_style.get("facecolor", "#777777"))
                    legend_handles.append(Patch(facecolor=patch_color, edgecolor="none", label=label, alpha=0.6))
        _apply_preview_theme(fig, ax, boundary_norm, title=title)
        _legend_bottom(ax, legend_handles)
        ax.set_axis_off()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_preview_figure(fig, output_path)
        plt.close(fig)
        return output_path

    def _single_layer(
        output_stem: str,
        gdf: gpd.GeoDataFrame | None,
        *,
        color: str,
        linewidth: float = 0.4,
        alpha: float = 0.8,
        markersize: float = 8.0,
        title: str | None = None,
    ) -> Path | None:
        if gdf is None or gdf.empty:
            return None
        style: dict = {"color": color, "alpha": alpha}
        geom_type = str(gdf.geom_type.iloc[0]) if "geom_type" in gdf else ""
        if "Point" in geom_type:
            style["markersize"] = markersize
        else:
            if "Polygon" in geom_type:
                style["linewidth"] = 0.05
                style["edgecolor"] = "#d1d5db"
                style["alpha"] = 0.92
            else:
                style["linewidth"] = linewidth
        return _plot(
            _preview_path(output_stem),
            layers=[
                (gdf, {**style, "label": title or output_stem}),
                (buffer_gdf, {"facecolor": "none", "edgecolor": "#111111", "linewidth": 1.1, "label": "analysis buffer"}),
            ],
            title=title,
        )

    def _filter_accessibility_blocks(gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
        if gdf is None or gdf.empty:
            return gdf
        for column in ("population_total", "population", "population_proxy", "pop_total", "residents", "res_population"):
            if column in gdf.columns:
                population = pd.to_numeric(gdf[column], errors="coerce").fillna(0.0)
                selected = gdf[population > 0].copy()
                return selected if not selected.empty else gdf
        return gdf

    def _plot_status(
        output_stem: str,
        *,
        title: str,
        groups: list[tuple[gpd.GeoDataFrame | None, str, str]],
        footer_lines: list[str] | None = None,
    ) -> Path | None:
        valid = [(g, color, label) for g, color, label in groups if g is not None and not g.empty]
        if not valid:
            return None
        target_crs = "EPSG:3857"
        normalized_valid: list[tuple[gpd.GeoDataFrame, str, str]] = []
        for gdf, color, label in valid:
            try:
                if gdf.crs is None:
                    normalized_valid.append((gdf, color, label))
                else:
                    normalized_valid.append((gdf.to_crs(target_crs), color, label))
            except Exception:
                normalized_valid.append((gdf, color, label))
        buffer_norm = buffer_gdf
        if buffer_gdf is not None and not buffer_gdf.empty and buffer_gdf.crs is not None:
            try:
                buffer_norm = buffer_gdf.to_crs(target_crs)
            except Exception:
                buffer_norm = buffer_gdf
        normalized_valid = [(_clip_to_preview_boundary(gdf, buffer_norm), color, label) for gdf, color, label in normalized_valid]
        fig, ax = plt.subplots(figsize=(12, 12))
        legend_handles = []
        for gdf, color, label in normalized_valid:
            gdf.plot(ax=ax, color=color, alpha=0.92, linewidth=0.05, edgecolor="#d1d5db")
            legend_handles.append(Patch(facecolor=color, edgecolor="none", label=label))
        if buffer_norm is not None and not buffer_norm.empty:
            _apply_preview_theme(fig, ax, buffer_norm, title=title)
        else:
            _apply_preview_theme(fig, ax, None, title=title)
        _legend_bottom(ax, legend_handles)
        ax.set_axis_off()
        _footer_text(fig, footer_lines)
        out = _preview_path(output_stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return out

    preview_dir = data_root / "preview_png"
    all_together_dir = preview_dir / "all_together"
    stage_dir = preview_dir / "stages" / _slugify(stage_label)
    preview_dir.mkdir(parents=True, exist_ok=True)
    all_together_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_groups = {
        "raw_collection": {"raw", "intermodal", "connectpt"},
        "floor_enrichment": {"floor"},
        "blocks_ready": {"blocks", "services"},
        "street_pattern_ready": {"street_pattern"},
    }
    enabled_groups = stage_groups.get(
        stage_label,
        {"raw", "intermodal", "connectpt", "floor", "blocks", "services", "street_pattern"},
    )

    def _should_render(group: str) -> bool:
        return group in enabled_groups

    def _preview_path(stem: str) -> Path:
        return all_together_dir / f"{stem}.png"

    saved: list[Path] = []

    def _remember_preview(path: Path | None, label: str) -> None:
        if path is None:
            return
        saved.append(path)
        try:
            shutil.copy2(path, stage_dir / path.name)
        except Exception:
            pass
        _log(f"Preview step: saved {label}: {path.name}")

    buffer_gdf_full = _read(buffer_path)
    buffer_circle_gdf, center_point_gdf = _build_buffer_circle_and_center(buffer_gdf_full)
    buffer_gdf = _visual_only_shrink_buffer(buffer_gdf_full, shrink_m=5.0)
    water_gdf = _read(Path(raw_files["water"]))
    roads_gdf = _read(Path(raw_files["roads"]))
    shared_roads_gdf = _read(data_root / "derived_layers" / "roads_drive_osmnx.parquet")
    roads_base_gdf = shared_roads_gdf if shared_roads_gdf is not None and not shared_roads_gdf.empty else roads_gdf
    railways_gdf = _read(Path(raw_files["railways"]))
    land_use_gdf = _read(Path(raw_files["land_use"]))
    buildings_gdf = _read(Path(raw_files["buildings"]))

    if _should_render("raw"):
        raw_png = _plot(
            _preview_path("overview_raw_osm_layers"),
            layers=[
                (water_gdf, {"color": "#67b7dc", "linewidth": 0.3, "alpha": 0.6, "label": "water"}),
                (land_use_gdf, {"color": "#b8d8a8", "alpha": 0.3, "linewidth": 0.1, "label": "land use"}),
                (roads_gdf, {"color": "#666666", "linewidth": 0.4, "alpha": 0.7, "label": "roads"}),
                (railways_gdf, {"color": "#8d6e63", "linewidth": 0.6, "alpha": 0.8, "label": "railways"}),
                (buildings_gdf, {"color": "#f59e0b", "alpha": 0.25, "linewidth": 0.05, "label": "buildings"}),
                (buffer_gdf, {"facecolor": "none", "edgecolor": "#111111", "linewidth": 1.2, "label": "analysis buffer"}),
            ],
            title="Raw OSM Layers",
        )
        _remember_preview(raw_png, "raw OSM composite")

    blocks_manifest = _try_load_json(blocks_manifest_path) or {}
    blocks_files = blocks_manifest.get("files", {})
    blocks_gdf = _read(Path(blocks_files.get("blocks", ""))) if blocks_files.get("blocks") else None
    clipped_blocks_gdf = _read(buffered_blocks_path)
    active_blocks_gdf = _filter_accessibility_blocks(clipped_blocks_gdf)
    street_grid_gdf = _read(street_grid_path)
    if _should_render("blocks"):
        prep_png = _plot(
            _preview_path("overview_prepared_blocks_and_street_grid"),
            layers=[
                (active_blocks_gdf, {"color": "#93c5fd", "alpha": 0.45, "linewidth": 0.1, "label": "blocks (accessibility set)"}),
                (street_grid_gdf, {"color": "#ef4444", "alpha": 0.25, "linewidth": 0.1, "label": "street grid"}),
                (blocks_gdf, {"facecolor": "none", "edgecolor": "#1f2937", "linewidth": 0.2, "alpha": 0.7, "label": "blocks"}),
                (buffer_gdf, {"facecolor": "none", "edgecolor": "#111111", "linewidth": 1.1, "label": "analysis buffer"}),
            ],
            title="Prepared Blocks + Street Grid",
        )
        _remember_preview(prep_png, "prepared blocks + street grid")

    intermodal_manifest = _try_load_json(intermodal_manifest_path) if intermodal_manifest_path else None
    if _should_render("intermodal") and isinstance(intermodal_manifest, dict):
        intermodal_files = intermodal_manifest.get("files") or {}
        intermodal_boundary = _read(Path(intermodal_files["boundary"])) if intermodal_files.get("boundary") else None
        intermodal_nodes = _read(Path(intermodal_files["graph_nodes"])) if intermodal_files.get("graph_nodes") else None
        intermodal_edges = _read(Path(intermodal_files["graph_edges"])) if intermodal_files.get("graph_edges") else None
        if (intermodal_nodes is not None and not intermodal_nodes.empty) or (intermodal_edges is not None and not intermodal_edges.empty):
            edges_plot = intermodal_edges.copy() if intermodal_edges is not None and not intermodal_edges.empty else None
            nodes_plot = intermodal_nodes.copy() if intermodal_nodes is not None and not intermodal_nodes.empty else None
            territory_plot = intermodal_boundary.copy() if intermodal_boundary is not None and not intermodal_boundary.empty else None
            if territory_plot is None:
                territory_plot = buffer_gdf_full.copy() if buffer_gdf_full is not None and not buffer_gdf_full.empty else None
            circle_plot = buffer_circle_gdf.copy() if buffer_circle_gdf is not None and not buffer_circle_gdf.empty else None
            center_plot = center_point_gdf.copy() if center_point_gdf is not None and not center_point_gdf.empty else None

            if edges_plot is not None and edges_plot.crs is not None:
                try:
                    edges_plot = edges_plot.to_crs("EPSG:3857")
                except Exception:
                    pass
            if nodes_plot is not None and nodes_plot.crs is not None:
                try:
                    nodes_plot = nodes_plot.to_crs("EPSG:3857")
                except Exception:
                    pass
            if territory_plot is not None and not territory_plot.empty and territory_plot.crs is not None:
                try:
                    territory_plot = territory_plot.to_crs("EPSG:3857")
                except Exception:
                    pass
            if circle_plot is not None and not circle_plot.empty and circle_plot.crs is not None:
                try:
                    circle_plot = circle_plot.to_crs("EPSG:3857")
                except Exception:
                    pass
            if center_plot is not None and not center_plot.empty and center_plot.crs is not None:
                try:
                    center_plot = center_plot.to_crs("EPSG:3857")
                except Exception:
                    pass

            fig, ax = plt.subplots(figsize=(12, 12))
            legend_handles = []
            footer_lines: list[str] = []

            territory_union = territory_plot.union_all() if territory_plot is not None and not territory_plot.empty else None

            if edges_plot is not None and not edges_plot.empty:
                mode_col = next((c for c in ("type", "transport_type", "mode", "route_type") if c in edges_plot.columns), None)
                if mode_col is None:
                    edges_plot.plot(ax=ax, color="#0f766e", linewidth=0.65, alpha=0.8)
                    legend_handles.append(Line2D([0], [0], color="#0f766e", linewidth=2, label="intermodal edges"))
                else:
                    mode_values = edges_plot[mode_col].astype("string").fillna("unknown")
                    top_modes = mode_values.value_counts().head(8).index.tolist()
                    palette = ["#0f766e", "#0ea5e9", "#8b5cf6", "#f97316", "#16a34a", "#dc2626", "#334155", "#eab308"]
                    for idx, mode_name in enumerate(top_modes):
                        color = palette[idx % len(palette)]
                        part = edges_plot[mode_values == mode_name]
                        if part.empty:
                            continue
                        part.plot(ax=ax, color=color, linewidth=0.7, alpha=0.85)
                        legend_handles.append(Line2D([0], [0], color=color, linewidth=2, label=str(mode_name)))
                    other = edges_plot[~mode_values.isin(top_modes)]
                    if not other.empty:
                        other.plot(ax=ax, color="#9ca3af", linewidth=0.5, alpha=0.5)
                        legend_handles.append(Line2D([0], [0], color="#9ca3af", linewidth=2, label="other"))
                if territory_union is not None:
                    try:
                        outside_edges = int((~edges_plot.geometry.intersects(territory_union)).sum())
                        footer_lines.append(f"edges outside territory: {outside_edges}")
                    except Exception:
                        pass

            if nodes_plot is not None and not nodes_plot.empty:
                nodes_plot.plot(ax=ax, color="#111827", markersize=5, alpha=0.7)
                legend_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markersize=6, label="nodes"))
                if territory_union is not None:
                    try:
                        outside_nodes = int((~nodes_plot.geometry.within(territory_union)).sum())
                        footer_lines.append(f"nodes outside territory: {outside_nodes}")
                    except Exception:
                        pass

            if territory_plot is not None and not territory_plot.empty:
                territory_plot.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.5)
                legend_handles.append(Line2D([0], [0], color="#111111", linewidth=2.2, label="analysis territory boundary"))
            if circle_plot is not None and not circle_plot.empty:
                circle_plot.plot(ax=ax, facecolor="none", edgecolor="#dc2626", linewidth=1.3, linestyle="--")
                legend_handles.append(Line2D([0], [0], color="#dc2626", linestyle="--", linewidth=2, label="buffer circle"))
            if center_plot is not None and not center_plot.empty:
                center_plot.plot(ax=ax, color="#dc2626", markersize=28, marker="*")
                legend_handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor="#dc2626", markersize=10, label="buffer center"))

            _apply_preview_theme(fig, ax, territory_plot, title="Intermodal Transport Graph (all PT modes)")
            _legend_bottom(ax, legend_handles)
            ax.set_axis_off()
            _footer_text(fig, footer_lines)
            intermodal_png = _preview_path("pt_intermodal_graph_modes")
            fig.savefig(intermodal_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            _remember_preview(intermodal_png, "intermodal graph modes")

    # Pipeline_2 raw services preview (single combined map).
    services_raw_dir = data_root / "pipeline_2" / "services_raw"
    if _should_render("services") and services_raw_dir.exists():
        service_order = ["health", "post", "culture", "port", "airport", "marina"]
        service_colors = {
            "health": "#dc2626",
            "post": "#2563eb",
            "culture": "#7c3aed",
            "port": "#0f766e",
            "airport": "#d97706",
            "marina": "#0891b2",
        }
        service_layers: list[tuple[str, gpd.GeoDataFrame, str]] = []
        for service_name in service_order:
            service_path = services_raw_dir / f"{service_name}.parquet"
            service_gdf = _read(service_path)
            if service_gdf is None or service_gdf.empty:
                continue
            service_plot = service_gdf.copy()
            # Render services as points for a unified, readable map.
            non_point_mask = ~service_plot.geometry.geom_type.isin(["Point", "MultiPoint"])
            if non_point_mask.any():
                service_plot.loc[non_point_mask, "geometry"] = service_plot.loc[non_point_mask, "geometry"].representative_point()
            if service_plot.crs is not None:
                try:
                    service_plot = service_plot.to_crs("EPSG:3857")
                except Exception:
                    pass
            service_layers.append((service_name, service_plot, service_colors.get(service_name, "#334155")))

        if service_layers:
            fig, ax = plt.subplots(figsize=(12, 12))
            legend_handles = []

            roads_plot = roads_gdf
            if roads_plot is not None and not roads_plot.empty and roads_plot.crs is not None:
                try:
                    roads_plot = roads_plot.to_crs("EPSG:3857")
                except Exception:
                    pass
            if roads_plot is not None and not roads_plot.empty:
                roads_plot.plot(ax=ax, color="#d1d5db", linewidth=0.35, alpha=0.6)
                legend_handles.append(Line2D([0], [0], color="#9ca3af", linewidth=2, label="roads"))

            for service_name, service_plot, color in service_layers:
                service_plot.plot(ax=ax, color=color, markersize=10, alpha=0.9)
                legend_handles.append(
                    Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=7, label=service_name)
                )

            territory_for_services = _read(Path(raw_files["boundary"]))
            if territory_for_services is not None and not territory_for_services.empty and territory_for_services.crs is not None:
                try:
                    territory_for_services = territory_for_services.to_crs("EPSG:3857")
                except Exception:
                    pass

            if territory_for_services is not None and not territory_for_services.empty:
                territory_for_services.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.4)
                legend_handles.append(Line2D([0], [0], color="#111111", linewidth=2, label="city boundary"))

            _apply_preview_theme(fig, ax, territory_for_services, title="Pipeline_2 Raw Services (all categories)")
            _legend_bottom(ax, legend_handles)
            ax.set_axis_off()
            services_png = _preview_path("services_pipeline2_raw_all")
            fig.savefig(services_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            _remember_preview(services_png, "pipeline_2 raw services")

    if _should_render("street_pattern") and street_grid_gdf is not None and not street_grid_gdf.empty:
        street_plot = street_grid_gdf.copy()
        roads_plot = roads_base_gdf.copy() if roads_base_gdf is not None and not roads_base_gdf.empty else None
        buffer_plot = buffer_gdf
        if street_plot.crs is not None:
            try:
                street_plot = street_plot.to_crs("EPSG:3857")
            except Exception:
                pass
        if roads_plot is not None and roads_plot.crs is not None:
            try:
                roads_plot = roads_plot.to_crs("EPSG:3857")
            except Exception:
                pass
        if buffer_plot is not None and not buffer_plot.empty and buffer_plot.crs is not None:
            try:
                buffer_plot = buffer_plot.to_crs("EPSG:3857")
            except Exception:
                pass

        if "top1_class_name" in street_plot.columns:
            fig, ax = plt.subplots(figsize=(12, 12))
            legend_handles = []
            if roads_plot is not None and not roads_plot.empty:
                roads_plot.plot(ax=ax, color="#d1d5db", linewidth=0.35, alpha=0.65)
                legend_handles.append(Line2D([0], [0], color="#9ca3af", linewidth=2, label="roads"))
            classes = street_plot["top1_class_name"].astype("string").fillna("unknown")
            class_order = [v for v in classes.value_counts().index.tolist() if v][:8]
            palette = ["#0f766e", "#0ea5e9", "#8b5cf6", "#f97316", "#16a34a", "#dc2626", "#334155", "#eab308"]
            for idx, class_name in enumerate(class_order):
                color = palette[idx % len(palette)]
                part = street_plot[classes == class_name]
                if part.empty:
                    continue
                part.plot(ax=ax, color=color, alpha=0.55, linewidth=0.1)
                legend_handles.append(Patch(facecolor=color, edgecolor="none", label=str(class_name)))
            if buffer_plot is not None and not buffer_plot.empty:
                buffer_plot.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.1)
                legend_handles.append(Line2D([0], [0], color="#111111", linewidth=2, label="analysis buffer"))
            _apply_preview_theme(fig, ax, buffer_plot, title="Street Pattern Top-1 Classification")
            _legend_bottom(ax, legend_handles)
            ax.set_axis_off()
            street_top1_png = _preview_path("street_pattern_top1")
            fig.savefig(street_top1_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            _remember_preview(street_top1_png, "street-pattern top1")

        if "multivariate_color" in street_plot.columns:
            fig, ax = plt.subplots(figsize=(12, 12))
            legend_handles = []
            if roads_plot is not None and not roads_plot.empty:
                roads_plot.plot(ax=ax, color="#d1d5db", linewidth=0.35, alpha=0.65)
                legend_handles.append(Line2D([0], [0], color="#9ca3af", linewidth=2, label="roads"))
            multi = street_plot[street_plot["multivariate_color"].notna()].copy()
            if not multi.empty:
                multi.plot(
                    ax=ax,
                    color=multi["multivariate_color"].astype("string"),
                    alpha=0.65,
                    linewidth=0.1,
                )
                legend_handles.append(Patch(facecolor="#98c5d7", edgecolor="none", label="multivariate coloring"))
            if buffer_plot is not None and not buffer_plot.empty:
                buffer_plot.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.1)
                legend_handles.append(Line2D([0], [0], color="#111111", linewidth=2, label="analysis buffer"))
            _apply_preview_theme(fig, ax, buffer_plot, title="Street Pattern Multivariate Classification")
            _legend_bottom(ax, legend_handles)
            ax.set_axis_off()
            street_multi_png = _preview_path("street_pattern_multivariate")
            fig.savefig(street_multi_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            _remember_preview(street_multi_png, "street-pattern multivariate")

    connectpt_manifest = _try_load_json(connectpt_manifest_path) or {}
    if _should_render("connectpt"):
        for modality in connectpt_manifest.get("modalities", []):
            modality_name = modality.get("modality", "unknown")
            files = modality.get("files") or {}
            lines = _read(Path(files["lines"])) if files.get("lines") else None
            stops = _read(Path(files["aggregated_stops"])) if files.get("aggregated_stops") else None
            projected_lines = _read(Path(files["projected_lines"])) if files.get("projected_lines") else None
            has_modality_content = any(
                g is not None and not g.empty
                for g in (projected_lines, lines, stops)
            )
            if has_modality_content:
                if projected_lines is not None and not projected_lines.empty:
                    proj = projected_lines.copy()
                    if proj.crs is not None:
                        proj = proj.to_crs("EPSG:3857")
                else:
                    proj = None
                base_lines = lines.copy().to_crs("EPSG:3857") if lines is not None and not lines.empty and lines.crs is not None else lines
                stops_plot = stops.copy().to_crs("EPSG:3857") if stops is not None and not stops.empty and stops.crs is not None else stops
                buf_plot = buffer_gdf.copy().to_crs("EPSG:3857") if buffer_gdf is not None and not buffer_gdf.empty and buffer_gdf.crs is not None else buffer_gdf

                fig, ax = plt.subplots(figsize=(12, 12))
                legend_handles = []
                if base_lines is not None and not base_lines.empty:
                    base_lines.plot(ax=ax, color="#9ca3af", linewidth=0.35, alpha=0.6)
                    legend_handles.append(Line2D([0], [0], color="#9ca3af", linewidth=2, label="roads/lines base"))
                if proj is not None and not proj.empty:
                    route_col = next((c for c in ("ref", "route", "name", "line_id") if c in proj.columns), None)
                    if route_col is None:
                        proj.plot(ax=ax, color="#0f766e", linewidth=0.8, alpha=0.9)
                        legend_handles.append(Line2D([0], [0], color="#0f766e", linewidth=2, label="routes"))
                    else:
                        values = proj[route_col].astype("string").fillna("unknown")
                        top_values = values.value_counts().head(6).index.tolist()
                        palette = ["#0f766e", "#0ea5e9", "#8b5cf6", "#f97316", "#16a34a", "#dc2626", "#334155"]
                        for idx, value in enumerate(top_values):
                            color = palette[idx % len(palette)]
                            part = proj[values == value]
                            if part.empty:
                                continue
                            part.plot(ax=ax, color=color, linewidth=0.9, alpha=0.95)
                            legend_handles.append(Line2D([0], [0], color=color, linewidth=2, label=f"route {value}"))
                        other = proj[~values.isin(top_values)]
                        if not other.empty:
                            other.plot(ax=ax, color=palette[-1], linewidth=0.7, alpha=0.65)
                            legend_handles.append(Line2D([0], [0], color=palette[-1], linewidth=2, label="other routes"))
                if stops_plot is not None and not stops_plot.empty:
                    stops_plot.plot(ax=ax, color="#111827", markersize=7, alpha=0.95)
                    legend_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markersize=7, label="stops"))
                if buf_plot is not None and not buf_plot.empty:
                    buf_plot.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.1)
                    legend_handles.append(Line2D([0], [0], color="#111111", linewidth=2, label="analysis buffer"))
                _apply_preview_theme(fig, ax, buf_plot, title=f"ConnectPT {modality_name}")
                _legend_bottom(ax, legend_handles)
                ax.set_axis_off()
                modality_png = _preview_path(f"pt_connectpt_{modality_name}")
                fig.savefig(modality_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
                plt.close(fig)
                _remember_preview(modality_png, f"connectpt {modality_name}")

            graph_nodes = _read(Path(files["graph_nodes"])) if files.get("graph_nodes") else None
            graph_edges = _read(Path(files["graph_edges"])) if files.get("graph_edges") else None
            if (graph_nodes is None or graph_nodes.empty) and (graph_edges is None or graph_edges.empty):
                continue
            graph_png = _plot(
                _preview_path(f"pt_connectpt_graph_{modality_name}"),
                layers=[
                    (graph_edges, {"color": "#0b7285", "linewidth": 0.4, "alpha": 0.7, "label": "graph edges"}),
                    (graph_nodes, {"color": "#e03131", "markersize": 6, "alpha": 0.9, "label": "graph nodes"}),
                    (buffer_gdf, {"facecolor": "none", "edgecolor": "#111111", "linewidth": 1.1, "label": "analysis buffer"}),
                ],
                title=f"ConnectPT Graph {modality_name}",
            )
            _remember_preview(graph_png, f"connectpt graph {modality_name}")

    if _should_render("raw"):
        for item in [
            _single_layer("raw_water", water_gdf, color="#0284c7", linewidth=0.5, alpha=0.8, title="Raw Water"),
            _single_layer("raw_roads", roads_gdf, color="#4b5563", linewidth=0.35, alpha=0.85, title="Raw Roads"),
            _single_layer("raw_railways", railways_gdf, color="#6b4f4f", linewidth=0.8, alpha=0.9, title="Raw Railways"),
            _single_layer("raw_land_use", land_use_gdf, color="#65a30d", linewidth=0.15, alpha=0.45, title="Raw Land Use"),
            _single_layer("raw_buildings", buildings_gdf, color="#f97316", linewidth=0.05, alpha=0.35, title="Raw Buildings"),
        ]:
            _remember_preview(item, item.stem if item is not None else "")
    if _should_render("blocks"):
        for item in [
            _single_layer("blocksnet_blocks", blocks_gdf, color="#334155", linewidth=0.2, alpha=0.75, title="BlocksNet Blocks"),
            _single_layer("blocks_clipped", active_blocks_gdf, color="#2563eb", linewidth=0.15, alpha=0.55, title="Blocks Used For Accessibility"),
        ]:
            _remember_preview(item, item.stem if item is not None else "")

    sm_imputed_gdf = _read(sm_imputed_path) if sm_imputed_path is not None else None
    if _should_render("sm_imputation") and sm_imputed_gdf is not None and not sm_imputed_gdf.empty:
        status_footer: list[str] = []
        if isinstance(sm_imputation_metrics, dict):
            status_footer.append(
                "targeted="
                f"{int(sm_imputation_metrics.get('target_rows', 0))}, "
                f"imputed={int(sm_imputation_metrics.get('imputed_rows', 0))}, "
                f"known={int(sm_imputation_metrics.get('known_rows', 0))}"
            )
            if sm_imputation_metrics.get("skipped"):
                status_footer.append(f"skipped: {sm_imputation_metrics.get('skip_reason') or 'unknown'}")
        status_png = _plot_status(
            "sm_imputation_status",
            title="SM Imputation Status",
            groups=[
                (sm_imputed_gdf[pd.to_numeric(sm_imputed_gdf.get("sm_imputation_used"), errors="coerce").fillna(0) >= 0.5], "#f59e0b", "imputed"),
                (
                    sm_imputed_gdf[
                        (pd.to_numeric(sm_imputed_gdf.get("sm_imputation_target"), errors="coerce").fillna(0) >= 0.5)
                        & (pd.to_numeric(sm_imputed_gdf.get("sm_imputation_used"), errors="coerce").fillna(0) < 0.5)
                    ],
                    "#dc2626",
                    "targeted but unresolved",
                ),
                (
                    sm_imputed_gdf[pd.to_numeric(sm_imputed_gdf.get("sm_imputation_target"), errors="coerce").fillna(0) < 0.5],
                    "#93c5fd",
                    "existing built form",
                ),
            ],
            footer_lines=status_footer,
        )
        _remember_preview(status_png, "sm-imputation status")

        for column_name, output_stem, title, cmap_name in [
            ("sm_fsi_final", "sm_imputation_fsi_final", "SM Imputation Final FSI", "YlOrRd"),
            ("sm_gsi_final", "sm_imputation_gsi_final", "SM Imputation Final GSI", "PuBuGn"),
        ]:
            if column_name not in sm_imputed_gdf.columns:
                continue
            plot_gdf = sm_imputed_gdf.copy()
            plot_gdf[column_name] = pd.to_numeric(plot_gdf[column_name], errors="coerce")
            plot_gdf = plot_gdf[plot_gdf[column_name].notna() & plot_gdf[column_name].gt(0)].copy()
            if plot_gdf.empty:
                continue
            if plot_gdf.crs is not None:
                try:
                    plot_gdf = plot_gdf.to_crs("EPSG:3857")
                except Exception:
                    pass
            buffer_plot = buffer_gdf
            if buffer_plot is not None and not buffer_plot.empty and buffer_plot.crs is not None:
                try:
                    buffer_plot = buffer_plot.to_crs("EPSG:3857")
                except Exception:
                    pass
            fig, ax = plt.subplots(figsize=(12, 12))
            plot_gdf.plot(
                ax=ax,
                column=column_name,
                cmap=cmap_name,
                linewidth=0.08,
                edgecolor="#f8fafc",
                alpha=0.85,
                legend=True,
                legend_kwds={"shrink": 0.72, "label": column_name},
            )
            if buffer_plot is not None and not buffer_plot.empty:
                buffer_plot.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.1)
            footer_lines = [
                f"positive blocks: {int(plot_gdf[column_name].gt(0).sum())}",
                f"median={float(plot_gdf[column_name].median()):.3f}",
            ]
            _apply_preview_theme(fig, ax, buffer_plot, title=title)
            ax.set_axis_off()
            _footer_text(fig, footer_lines)
            out = _preview_path(output_stem)
            fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            _remember_preview(out, output_stem)

    # Land-use categorical legend preview.
    if _should_render("raw") and land_use_gdf is not None and not land_use_gdf.empty:
        landuse_col = "landuse" if "landuse" in land_use_gdf.columns else "functional_zone" if "functional_zone" in land_use_gdf.columns else None
        if landuse_col is not None:
            fig, ax = plt.subplots(figsize=(12, 12))
            land_plot = land_use_gdf.copy()
            land_plot[landuse_col] = land_plot[landuse_col].astype("string")
            if land_plot.crs is not None:
                try:
                    land_plot = land_plot.to_crs("EPSG:3857")
                except Exception:
                    pass
            buffer_land_plot = buffer_gdf
            if buffer_gdf is not None and not buffer_gdf.empty and buffer_gdf.crs is not None:
                try:
                    buffer_land_plot = buffer_gdf.to_crs("EPSG:3857")
                except Exception:
                    buffer_land_plot = buffer_gdf
            land_plot.plot(
                ax=ax,
                column=landuse_col,
                cmap="tab20",
                legend=False,
                alpha=0.5,
                linewidth=0.1,
            )
            legend_handles = []
            unique_values = [v for v in land_plot[landuse_col].dropna().astype("string").unique().tolist() if v]
            unique_values = sorted(unique_values)[:8]
            cmap = plt.get_cmap("tab20")
            for i, value in enumerate(unique_values):
                color = cmap(i % 20)
                part = land_plot[land_plot[landuse_col].astype("string") == value]
                if part.empty:
                    continue
                part.plot(ax=ax, color=color, alpha=0.55, linewidth=0.1)
                legend_handles.append(Patch(facecolor=color, edgecolor="none", label=str(value)))
            if buffer_land_plot is not None and not buffer_land_plot.empty:
                buffer_land_plot.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.1)
                legend_handles.append(Line2D([0], [0], color="#111111", linewidth=2, label="analysis buffer"))
            _apply_preview_theme(fig, ax, buffer_land_plot, title="Raw Land Use (categorical)")
            _legend_bottom(ax, legend_handles)
            ax.set_axis_off()
            landuse_png = _preview_path("raw_land_use_categorical")
            fig.savefig(landuse_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            _remember_preview(landuse_png, "raw land-use categorical")

    floor_enriched_gdf = _read(floor_enriched_path)
    storey_model_footer_lines: list[str] = []
    if _should_render("floor") and floor_enriched_gdf is not None and not floor_enriched_gdf.empty:
        floor_base = floor_enriched_gdf.copy()
        try:
            floor_base["is_living"] = pd.to_numeric(floor_base.get("is_living"), errors="coerce")
            if "is_living_restored" not in floor_base.columns:
                floor_base["is_living_restored"] = 0
            if "is_living_source" not in floor_base.columns:
                floor_base["is_living_source"] = "missing"
                floor_base.loc[floor_base["is_living"].notna(), "is_living_source"] = "original_is_living"
                floor_base.loc[pd.to_numeric(floor_base["is_living_restored"], errors="coerce").fillna(0) >= 0.5, "is_living_source"] = "osm_living_model"
            if "storey_restored" not in floor_base.columns:
                floor_base["storey_restored"] = 0
            if "storey_source" not in floor_base.columns:
                floor_base["storey_source"] = "missing"
                floor_base.loc[floor_base["storey"].notna(), "storey_source"] = "original_storey"
                floor_base.loc[pd.to_numeric(floor_base["storey_restored"], errors="coerce").fillna(0) >= 0.5, "storey_source"] = "model_predicted"
            floor_base["is_living_restored"] = pd.to_numeric(floor_base["is_living_restored"], errors="coerce").fillna(0)
            floor_base["storey_restored"] = pd.to_numeric(floor_base["storey_restored"], errors="coerce").fillna(0)
            floor_base["is_living_source"] = floor_base["is_living_source"].astype("string").fillna("missing")
            floor_base["storey_source"] = floor_base["storey_source"].astype("string").fillna("missing")
            living_known = floor_base[floor_base["is_living"].notna()].copy()
            if not living_known.empty:
                living_distribution_png = _plot_status(
                    "buildings_is_living_distribution",
                    title="Buildings is_living distribution by source",
                    groups=[
                        (living_known[(living_known["is_living"] >= 0.5) & (living_known["is_living_source"] == "osm_living_model")], "#4ade80", "restored living"),
                        (living_known[(living_known["is_living"] < 0.5) & (living_known["is_living_source"] == "osm_living_model")], "#f87171", "restored non-living"),
                        (living_known[(living_known["is_living"] >= 0.5) & (living_known["is_living_source"] == "original_is_living")], "#16a34a", "original living"),
                        (living_known[(living_known["is_living"] < 0.5) & (living_known["is_living_source"] == "original_is_living")], "#ef4444", "original non-living"),
                        (floor_base[floor_base["is_living"].isna()], "#6b7280", "missing"),
                    ],
                )
                _remember_preview(living_distribution_png, "buildings is_living distribution")

            if "is_living_source" in floor_base.columns:
                living_status_png = _plot_status(
                    "buildings_is_living_restoration_status",
                    title="Buildings is_living restoration status",
                    groups=[
                        (floor_base[floor_base["is_living_source"] == "osm_living_model"], "#f59e0b", "osm_living_model"),
                        (floor_base[floor_base["is_living_source"] == "original_is_living"], "#16a34a", "original is_living"),
                        (floor_base[floor_base["is_living"].isna()], "#dc2626", "missing"),
                    ],
                )
                _remember_preview(living_status_png, "buildings is_living status")

            floor_base["storey"] = pd.to_numeric(floor_base.get("storey"), errors="coerce")
            known_storey = floor_base[floor_base["storey"].notna()]
            if not known_storey.empty:
                known_storey_plot = known_storey.copy()
                if known_storey_plot.crs is not None:
                    try:
                        known_storey_plot = known_storey_plot.to_crs("EPSG:3857")
                    except Exception:
                        pass
                buffer_storey_plot = buffer_gdf
                if buffer_gdf is not None and not buffer_gdf.empty and buffer_gdf.crs is not None:
                    try:
                        buffer_storey_plot = buffer_gdf.to_crs("EPSG:3857")
                    except Exception:
                        buffer_storey_plot = buffer_gdf
                # Storey quantiles for all known buildings.
                fig, ax = plt.subplots(figsize=(12, 12))
                q_all = min(5, max(1, int(known_storey_plot["storey"].nunique())))
                bins_all = pd.qcut(known_storey_plot["storey"], q=q_all, duplicates="drop")
                known_storey_plot["storey_q"] = bins_all.astype("string")
                labels_all = [v for v in known_storey_plot["storey_q"].dropna().unique().tolist() if v]
                labels_all = sorted(labels_all)
                colors_all = plt.get_cmap("viridis")(pd.Series(range(len(labels_all))) / max(1, len(labels_all) - 1))
                legend_handles_all = []
                for idx, label in enumerate(labels_all):
                    part = known_storey_plot[known_storey_plot["storey_q"] == label]
                    color = colors_all[idx]
                    part.plot(ax=ax, color=color, linewidth=0.03, alpha=0.75)
                    legend_handles_all.append(Patch(facecolor=color, edgecolor="none", label=str(label)))
                if buffer_storey_plot is not None and not buffer_storey_plot.empty:
                    buffer_storey_plot.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.1)
                    legend_handles_all.append(Line2D([0], [0], color="#111111", linewidth=2, label="analysis buffer"))
                _apply_preview_theme(fig, ax, buffer_storey_plot, title="Buildings storey quantiles (all known storey)")
                _legend_bottom(ax, legend_handles_all)
                ax.set_axis_off()
                _footer_text(fig, storey_model_footer_lines)
                storey_png_all = _preview_path("buildings_storey_quantiles_all")
                fig.savefig(storey_png_all, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
                plt.close(fig)
                _remember_preview(storey_png_all, "buildings storey quantiles all")

                # Storey quantiles only for model-predicted buildings.
                predicted_known = known_storey_plot[known_storey_plot["storey_source"] == "model_predicted"].copy()
                if not predicted_known.empty:
                    fig, ax = plt.subplots(figsize=(12, 12))
                    q_rest = min(5, max(1, int(predicted_known["storey"].nunique())))
                    bins_rest = pd.qcut(predicted_known["storey"], q=q_rest, duplicates="drop")
                    predicted_known["storey_q"] = bins_rest.astype("string")
                    labels_rest = [v for v in predicted_known["storey_q"].dropna().unique().tolist() if v]
                    labels_rest = sorted(labels_rest)
                    colors_rest = plt.get_cmap("plasma")(pd.Series(range(len(labels_rest))) / max(1, len(labels_rest) - 1))
                    legend_handles_rest = []
                    for idx, label in enumerate(labels_rest):
                        part = predicted_known[predicted_known["storey_q"] == label]
                        color = colors_rest[idx]
                        part.plot(ax=ax, color=color, linewidth=0.03, alpha=0.85)
                        legend_handles_rest.append(Patch(facecolor=color, edgecolor="none", label=str(label)))
                    if buffer_storey_plot is not None and not buffer_storey_plot.empty:
                        buffer_storey_plot.plot(ax=ax, facecolor="none", edgecolor="#111111", linewidth=1.1)
                        legend_handles_rest.append(Line2D([0], [0], color="#111111", linewidth=2, label="analysis buffer"))
                    _apply_preview_theme(fig, ax, buffer_storey_plot, title="Buildings storey quantiles (model predicted only)")
                    _legend_bottom(ax, legend_handles_rest)
                    ax.set_axis_off()
                    _footer_text(fig, storey_model_footer_lines)
                    storey_png_restored = _preview_path("buildings_storey_quantiles_model_predicted")
                    fig.savefig(storey_png_restored, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
                    plt.close(fig)
                    _remember_preview(storey_png_restored, "buildings storey quantiles model predicted")

            if "storey_source" in floor_base.columns:
                known_non_living = floor_base["is_living"].notna() & (floor_base["is_living"] < 0.5)
                known_living = floor_base["is_living"].notna() & (floor_base["is_living"] >= 0.5)
                known_storey = floor_base["storey"].notna()
                missing_storey_living = floor_base["storey"].isna() & known_living
                missing_storey_non_living = floor_base["storey"].isna() & known_non_living
                known_storey_non_living = known_storey & known_non_living
                known_storey_living_original = known_storey & known_living & floor_base["storey_source"].isin(["original_storey", "osm_building_levels"])
                known_storey_living_predicted = known_storey & known_living & (floor_base["storey_source"] == "model_predicted")
                missing_storey_unknown = floor_base["storey"].isna() & floor_base["is_living"].isna()
                storey_status_png = _plot_status(
                    "buildings_storey_restoration_status",
                    title="Buildings storey restoration status",
                    groups=[
                        (floor_base[known_storey_non_living], "#111111", "non-living with known storey"),
                        (floor_base[known_storey_living_predicted], "#f59e0b", "living with model-predicted storey"),
                        (floor_base[known_storey_living_original], "#0ea5e9", "living with known storey"),
                        (floor_base[missing_storey_non_living], "#a3a3a3", "non-living (storey skipped)"),
                        (floor_base[missing_storey_living], "#dc2626", "missing living"),
                        (floor_base[missing_storey_unknown], "#6b7280", "missing / unknown living"),
                    ],
                    footer_lines=storey_model_footer_lines,
                )
                _remember_preview(storey_status_png, "buildings storey restoration status")
        except Exception:
            pass

    return saved


def _prepare_inputs_from_place(args: argparse.Namespace) -> PreparedInputs:
    # Lazy imports so explicit file mode does not require collection dependencies at import time.
    from aggregated_spatial_pipeline.blocksnet_data_pipeline.pipeline import (
        collect_blocksnet_raw_osm_bundle,
        slugify_place,
    )
    from aggregated_spatial_pipeline.connectpt_data_pipeline.pipeline import parse_modalities
    from aggregated_spatial_pipeline.intermodal_graph_data_pipeline.pipeline import build_intermodal_graph_bundle

    # Some imported modules may reconfigure loguru on import; enforce compact format again.
    _configure_logging()
    _configure_osm_requests(args.osm_timeout_s, overpass_url=args.overpass_url)

    place = str(args.place or "").strip()
    if not place:
        raise ValueError(
            "Automatic input preparation requires --place because city centres are resolved only from worldcities.csv."
        )

    repo_root = Path(__file__).resolve().parents[2]
    city_centers_csv = (
        Path(args.city_centers_csv).resolve()
        if args.city_centers_csv
        else (repo_root / "simplemaps_worldcities_basicv1" / "worldcities.csv").resolve()
    )
    slug = slugify_place(place)
    data_root = Path(args.data_dir).resolve() if args.data_dir else repo_root / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs" / slug
    blocks_raw_dir = data_root / "blocksnet_raw_osm"
    blocks_dir = data_root / "blocksnet"
    connectpt_dir = data_root / "connectpt_osm"
    intermodal_dir = data_root / "intermodal_graph_iduedu"
    derived_dir = data_root / "derived_layers"

    blocks_raw_manifest_path = blocks_raw_dir / "manifest.json"
    blocks_manifest_path = blocks_dir / "manifest.json"
    connectpt_manifest_path = connectpt_dir / "manifest.json"
    intermodal_manifest_path = intermodal_dir / "manifest.json"
    derived_manifest_path = derived_dir / "manifest.json"

    downloaded_in_this_run = False

    _log(f"Auto-collection enabled for place: {place}")
    _log(f"City centres source: {_log_name(city_centers_csv)}")
    _log(f"Cache mode: {'disabled (--no-cache)' if args.no_cache else 'enabled'}")
    detected_relation_id = args.relation_id
    relation_mode = detected_relation_id is not None or (args.place is not None and args.buffer_m is None)
    if relation_mode and detected_relation_id is None and args.place:
        detected_relation_id = _resolve_relation_id_for_place(args.place)

    if relation_mode:
        effective_buffer_m = 0.0
    else:
        effective_buffer_m = float(args.buffer_m) if args.buffer_m is not None else 20000.0
    if relation_mode:
        relation_log_id = f"relation_id={int(detected_relation_id)}" if detected_relation_id is not None else "relation_id=unknown"
        _log(
            "Analysis territory policy: relation polygon with no outer margin "
            f"({relation_log_id})."
        )
    else:
        _log(
            "Analysis territory policy: exact fixed buffer with no outer margin "
            f"({effective_buffer_m}m)."
        )
    floor_context_radius_m = None if relation_mode else effective_buffer_m
    if floor_context_radius_m is not None and floor_context_radius_m < float(args.floor_min_buffer_m):
        _warn(
            "Floor restoration context is usually unstable for small buffers. "
            f"Current effective radius={floor_context_radius_m}m, recommended minimum={float(args.floor_min_buffer_m)}m "
            "(tune via --floor-min-buffer-m or increase --buffer-m)."
        )

    analysis_dir = data_root / "analysis_territory"
    analysis_buffer_path = analysis_dir / "buffer.parquet"
    collection_buffer_path = analysis_dir / "buffer_collection.parquet"
    main_territory_buffer_m = 0.0 if relation_mode else effective_buffer_m
    buffer_matches = _analysis_buffer_matches(analysis_buffer_path, main_territory_buffer_m)
    collection_buffer_matches = _analysis_buffer_matches(collection_buffer_path, main_territory_buffer_m)
    if (not args.no_cache) and analysis_buffer_path.exists() and (not buffer_matches):
        _warn(
            "Cached analysis buffer radius does not match current --buffer-m "
            f"(requested={main_territory_buffer_m}m). Rebuilding analysis buffer."
        )
    if (not relation_mode) and (not args.no_cache) and collection_buffer_path.exists() and (not collection_buffer_matches):
        _warn(
            "Cached collection territory radius does not match current exact --buffer-m "
            f"(requested={main_territory_buffer_m}m). Rebuilding collection territory."
        )
    if args.no_cache or not analysis_buffer_path.exists() or (not buffer_matches):
        if relation_mode:
            _log("OSM territory step: resolving relation geometry and building main analysis territory.")
        else:
            _log("OSM territory step: resolving city centre and building main analysis buffer.")
        started = time.time()
        _resolve_analysis_buffer_from_osm(
            place=args.place,
            buffer_m=main_territory_buffer_m,
            output_path=analysis_buffer_path,
            city_centers_csv=city_centers_csv,
            center_node_id=args.center_node_id,
            relation_id=detected_relation_id,
            relation_geometry=Path(args.relation_geometry).resolve() if args.relation_geometry else None,
        )
        if relation_mode:
            _log(f"Relation territory built in {time.time() - started:.1f}s: {analysis_buffer_path.name}")
        else:
            _log(f"Analysis buffer built in {time.time() - started:.1f}s: {analysis_buffer_path.name}")
        downloaded_in_this_run = True
    else:
        if relation_mode:
            _log(f"Using cached relation territory: {analysis_buffer_path.name}")
        else:
            _log(f"Using cached analysis buffer: {analysis_buffer_path.name}")
    if relation_mode:
        _log("OSM territory step: relation geometry reused for collection territory.")
        if args.no_cache or not collection_buffer_path.exists():
            analysis_buffer_gdf = read_geodata(analysis_buffer_path)
            analysis_buffer_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_buffer_gdf.to_parquet(collection_buffer_path)
            _log(f"Relation territory reused in 0.0s: {collection_buffer_path.name}")
        else:
            _log(f"Using cached relation territory reuse: {collection_buffer_path.name}")
    else:
        _log("OSM territory step: exact analysis buffer reused for collection territory.")
        if args.no_cache or not collection_buffer_path.exists() or (not collection_buffer_matches):
            analysis_buffer_gdf = read_geodata(analysis_buffer_path)
            analysis_buffer_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_buffer_gdf.to_parquet(collection_buffer_path)
            _log(f"Exact collection territory reused in 0.0s: {collection_buffer_path.name}")
        else:
            _log(f"Using cached exact collection territory reuse: {collection_buffer_path.name}")
    buffer_path = analysis_buffer_path
    collection_boundary_path = collection_buffer_path
    shared_roads_path = derived_dir / "roads_drive_osmnx.parquet"
    if args.no_cache or not shared_roads_path.exists():
        _log(
            "Preparing shared drive roads from OSMnx drive graph "
            "(this may take time before the next network log appears)..."
        )
    shared_roads_path, shared_roads_count, shared_roads_rebuilt = _ensure_shared_drive_roads(
        buffer_path=collection_boundary_path,
        output_path=shared_roads_path,
        no_cache=args.no_cache,
    )
    if shared_roads_rebuilt:
        _log(
            "Shared roads prepared from OSMnx drive graph: "
            f"{shared_roads_count} edges ({shared_roads_path.name})"
        )
        downloaded_in_this_run = True
    else:
        _log(f"Using cached shared drive roads: {shared_roads_path.name} ({shared_roads_count} edges)")

    if args.no_cache or not blocks_raw_manifest_path.exists():
        _log("Collecting raw OSM layers for blocks/buildings preprocessing...")
        _log(
            "OSM download (raw): "
            "requesting water/railways/landuse/buildings plus floor-context amenities and pipeline_2 services via Overpass "
            "inside the exact analysis territory, "
            "while reusing the already prepared shared drive roads layer."
        )
        started = time.time()
        collect_blocksnet_raw_osm_bundle(
            place,
            output_dir=blocks_raw_dir,
            boundary_path=collection_boundary_path,
            roads_override_path=shared_roads_path,
        )
        _log(f"Raw OSM bundle collected in {time.time() - started:.1f}s: {_log_name(blocks_raw_manifest_path)}")
        downloaded_in_this_run = True
    else:
        _log("Using cached raw OSM bundle.")

    blocks_raw_manifest = _try_load_json(blocks_raw_manifest_path)
    if blocks_raw_manifest is None:
        raise RuntimeError(f"Cannot read raw OSM manifest: {blocks_raw_manifest_path}")

    raw_files = blocks_raw_manifest.get("files", {})
    boundary_path = Path(raw_files["boundary"]).resolve()
    buildings_path = Path(raw_files["buildings"]).resolve()
    land_use_path = Path(raw_files["land_use"]).resolve()
    amenities_raw = raw_files.get("amenities")
    if not amenities_raw:
        raise RuntimeError(
            "Raw OSM manifest does not contain 'amenities'. "
            "Rebuild raw collection for this city (run with --no-cache)."
        )
    floor_amenities_path = Path(amenities_raw).resolve()
    if not floor_amenities_path.exists():
        raise RuntimeError(
            f"Raw amenities layer is missing: {floor_amenities_path}. "
            "Rebuild raw collection for this city (run with --no-cache)."
        )
    data_coverage_metrics: dict | None = None
    try:
        data_coverage_metrics = _compute_data_coverage_metrics(
            territory_path=buffer_path,
            buildings_path=buildings_path,
            roads_path=shared_roads_path,
            roads_buffer_m=float(args.roads_coverage_buffer_m),
        )
        _log(
            "Data coverage check: "
            f"buildings={data_coverage_metrics['buildings_coverage_pct']:.2f}%, "
            f"roads(buffer)={data_coverage_metrics['roads_coverage_pct']:.2f}%, "
            f"combined={data_coverage_metrics['combined_data_coverage_pct']:.2f}% "
            f"(threshold={float(args.min_data_coverage_pct):.2f}%)."
        )
        if float(data_coverage_metrics["combined_data_coverage_pct"]) < float(args.min_data_coverage_pct):
            _warn(
                "Low data coverage detected for selected territory: "
                f"combined buildings+roads coverage is {data_coverage_metrics['combined_data_coverage_pct']:.2f}% "
                f"(< {float(args.min_data_coverage_pct):.2f}%). "
                "This often indicates wrong place relation/centre or incomplete source layers."
            )
    except Exception as exc:
        data_coverage_metrics = {"error": str(exc)}
        _warn(f"Data coverage check skipped due to error: {exc}")

    parsed_modalities = parse_modalities(args.modalities)
    if args.no_cache or not intermodal_manifest_path.exists():
        _log("Collecting intermodal transport graph bundle...")
        _log(
            "IduEdu download (intermodal): "
            "requesting one intermodal transport graph with all supported PT modalities "
            "(tram, bus, trolleybus, subway) for the specified analysis territory via dedicated iduedu 1.2.1 runtime; "
            "this bundle is cached and intended for quarter-to-quarter accessibility in pipeline_2."
        )
        started = time.time()
        build_intermodal_graph_bundle(
            place=place,
            output_dir=intermodal_dir,
            boundary_path=collection_boundary_path,
            python_executable=args.intermodal_python,
            repo_root=repo_root,
        )
        _log(
            "Intermodal graph bundle collected in "
            f"{time.time() - started:.1f}s: {_log_name(intermodal_manifest_path)}"
        )
        downloaded_in_this_run = True
    else:
        _log("Using cached intermodal transport graph bundle.")

    intermodal_nodes_path = intermodal_dir / "graph_nodes.parquet"
    if args.no_cache or not connectpt_manifest_path.exists():
        _log("Collecting connectpt bundle...")
        _log(
            "ConnectPT stop policy: derive modality stops from intermodal iduedu graph first, "
            "aggregate them with connectpt logic, store raw->aggregated mapping, "
            "and use the intermodal-expanded stop layer as the only stop source. "
            "PT lines are still collected per modality; bus road geometry reuses the shared drive graph."
        )
        started = time.time()
        _run_external_json_command(
            python_path=connectpt_python(repo_root),
            module="aggregated_spatial_pipeline.connectpt_data_pipeline.run_bundle_external",
            repo_root=repo_root,
            args=[
                "--place",
                place,
                "--modalities",
                *[m.value for m in parsed_modalities],
                "--output-dir",
                str(connectpt_dir),
                "--speed-kmh",
                str(float(args.speed_kmh)),
                "--boundary-path",
                str(collection_boundary_path),
                "--drive-roads-path",
                str(shared_roads_path),
                "--buildings-path",
                str(buildings_path),
                "--intermodal-nodes-path",
                str(intermodal_nodes_path),
            ],
            mplconfigdir=_repo_mplconfigdir(repo_root, "mpl-connectpt"),
        )
        _log(f"ConnectPT bundle collected in {time.time() - started:.1f}s: {_log_name(connectpt_manifest_path)}")
        downloaded_in_this_run = True
    else:
        _log("Using cached connectpt bundle.")

    floor_output_path = derived_dir / "buildings_floor_enriched.parquet"
    buffered_blocks_path = derived_dir / "blocks_clipped.parquet"
    clipped_street_grid_path = derived_dir / "street_grid_buffered.parquet"

    def _refresh_collection_previews(
        stage_label: str,
        *,
        floor_metrics_for_preview: dict | None = None,
    ) -> None:
        started = time.time()
        preview_paths = _save_collection_previews(
            data_root=data_root,
            buffer_path=buffer_path,
            raw_files=raw_files,
            connectpt_manifest_path=connectpt_manifest_path,
            intermodal_manifest_path=intermodal_manifest_path,
            blocks_manifest_path=blocks_manifest_path,
            buffered_blocks_path=buffered_blocks_path,
            street_grid_path=clipped_street_grid_path,
            floor_enriched_path=floor_output_path,
            floor_metrics=floor_metrics_for_preview,
            stage_label=stage_label,
        )
        if preview_paths:
            _log(
                f"Preview refresh [{stage_label}] finished in {time.time() - started:.1f}s "
                f"({len(preview_paths)} files, stage snapshots saved under preview_png/stages/{_slugify(stage_label)})."
            )
        else:
            _log(f"Preview refresh [{stage_label}] skipped (no readable layers yet).")

    _log("Refreshing previews after raw/intermodal/connectpt collection...")
    _refresh_collection_previews("raw_collection")

    floor_cache_current = _floor_output_is_current(floor_output_path)
    floor_rebuilt = False
    if floor_output_path.exists() and (not floor_cache_current) and (not args.no_cache):
        _warn(
            "Cached floor-preprocessing output is outdated for current source-tracking logic. "
            "Rebuilding buildings_floor_enriched."
        )
    if args.no_cache or not floor_output_path.exists() or not floor_cache_current:
        _log(
            "Floor-predictor preprocessing: "
            "restoring missing is_living with model-based osm_living_predictor path and "
            "then preparing storey fields."
        )
        floor_args = [
            "--repo-root",
            str(repo_root),
            "--buildings-path",
            str(buildings_path),
            "--land-use-path",
            str(land_use_path),
            "--roads-path",
            str(shared_roads_path),
            "--amenities-path",
            str(floor_amenities_path),
            "--output-path",
            str(floor_output_path),
            "--floor-ignore-missing-below-pct",
            str(float(args.floor_ignore_missing_below_pct)),
            "--osm-timeout-s",
            str(int(args.osm_timeout_s)),
        ]
        if args.overpass_url:
            floor_args.extend(["--overpass-url", str(args.overpass_url)])
        if args.is_living_model_path:
            floor_args.extend(["--is-living-model-path", str(Path(args.is_living_model_path).resolve())])
        floor_metrics = _run_external_json_command(
            python_path=floor_predictor_python(repo_root),
            module="aggregated_spatial_pipeline.pipeline.run_floor_predictor_external",
            repo_root=repo_root,
            args=floor_args,
            mplconfigdir=_repo_mplconfigdir(repo_root, "mpl-floor-predictor"),
        )
        _log(
            "Floor-predictor done: "
            f"is_living missing {floor_metrics['is_living_missing_before']} -> {floor_metrics['is_living_missing_after']}, "
            f"storey missing {floor_metrics['storey_missing_before_model']} -> {floor_metrics['storey_missing_after_model']} "
            f"(predicted {floor_metrics['storey_predicted_by_model']}); "
            f"storey_missing_after split: living={floor_metrics.get('storey_missing_after_model_living')}, "
            f"non_living={floor_metrics.get('storey_missing_after_model_non_living')} "
            f"(usually acceptable), unknown_living={floor_metrics.get('storey_missing_after_model_unknown_living')}"
        )
        downloaded_in_this_run = True
        floor_rebuilt = True
        _log("Refreshing previews after floor/building enrichment...")
        _refresh_collection_previews("floor_enrichment", floor_metrics_for_preview=floor_metrics)
    else:
        _log(f"Using cached floor-preprocessing output: {_log_name(floor_output_path)}")
        floor_enriched = read_geodata(floor_output_path)
        floor_is_living_missing = int(pd.to_numeric(floor_enriched.get("is_living"), errors="coerce").isna().sum())
        floor_storey_missing = int(pd.to_numeric(floor_enriched.get("storey"), errors="coerce").isna().sum())
        floor_metrics = {
            "output_path": str(floor_output_path),
            "model_path": str((repo_root / "floor-predictor" / "floor_predictior" / "model" / "StoreyModelTrainer.joblib")),
            "rows_total": int(len(floor_enriched)),
            "is_living_missing_before": None,
            "is_living_missing_after": floor_is_living_missing,
            "storey_missing_before_model": None,
            "storey_predicted_by_model": None,
            "storey_missing_after_model": floor_storey_missing,
        }

    def _ensure_blocks_ready_at_end() -> tuple[Path, Path, int]:
        nonlocal downloaded_in_this_run

        blocks_stale_by_floor = (
            blocks_manifest_path.exists()
            and floor_output_path.exists()
            and floor_output_path.stat().st_mtime > blocks_manifest_path.stat().st_mtime
        )
        if blocks_stale_by_floor and (not args.no_cache):
            _warn(
                "Cached BlocksNet bundle is older than buildings_floor_enriched. "
                "Rebuilding blocks at final stage to keep building-level living/resident context -> block aggregation order."
            )
        blocks_rebuilt = False
        if args.no_cache or floor_rebuilt or blocks_stale_by_floor or not blocks_manifest_path.exists():
            _log("BlocksNet processing step (final): building blocks from pre-collected OSM layers + enriched buildings.")
            started = time.time()
            _run_external_json_command(
                python_path=blocksnet_python(repo_root),
                module="aggregated_spatial_pipeline.blocksnet_data_pipeline.run_bundle_external",
                repo_root=repo_root,
                args=[
                    "--place",
                    place,
                    "--output-dir",
                    str(blocks_dir),
                    "--boundary-path",
                    str(collection_boundary_path),
                    "--prefetched-layers-json",
                    json.dumps({k: str(v) for k, v in raw_files.items()}, ensure_ascii=False),
                    "--buildings-override-path",
                    str(floor_output_path),
                ],
                mplconfigdir=_repo_mplconfigdir(repo_root, "mpl-blocksnet"),
            )
            _log(f"BlocksNet bundle built in {time.time() - started:.1f}s: {_log_name(blocks_manifest_path)}")
            downloaded_in_this_run = True
            blocks_rebuilt = True
        else:
            _log("Using cached BlocksNet processed bundle.")

        blocks_manifest = _try_load_json(blocks_manifest_path)
        if blocks_manifest is None:
            raise RuntimeError(f"Cannot read blocksnet manifest: {blocks_manifest_path}")
        files = blocks_manifest.get("files", {})
        blocks_path_local = Path(files["blocks"]).resolve()
        buffered_blocks_path_local = buffered_blocks_path
        buffered_blocks_stale = (
            buffered_blocks_path_local.exists()
            and blocks_path_local.exists()
            and blocks_path_local.stat().st_mtime > buffered_blocks_path_local.stat().st_mtime
        )
        if buffered_blocks_stale and (not args.no_cache):
            _warn(
                "Cached clipped blocks are older than blocks.parquet. "
                "Rebuilding clipped blocks to keep population/living attributes current."
            )
        if args.no_cache or blocks_rebuilt or buffered_blocks_stale or not buffered_blocks_path_local.exists():
            _log("Clipping blocks to analysis buffer...")
            buffered_blocks_path_local, buffered_blocks_count_local = _build_buffered_quarters(
                blocks_path=blocks_path_local,
                buffer_path=buffer_path,
                output_path=buffered_blocks_path_local,
            )
            _log(f"Clipped blocks ready: {buffered_blocks_count_local} features")
        else:
            buffered_blocks_count_local = len(read_geodata(buffered_blocks_path_local))
            _log(
                f"Using cached clipped blocks: {_log_name(buffered_blocks_path_local)} "
                f"({buffered_blocks_count_local} features)"
            )
        return blocks_path_local, buffered_blocks_path_local, buffered_blocks_count_local

    _log("Classification step: building street-pattern grid for the same territory policy.")
    street_relation_id = detected_relation_id
    street_center_node_id = args.center_node_id if street_relation_id is None else None
    if (street_relation_id is None) and analysis_buffer_path.exists():
        try:
            analysis_buffer_gdf = read_geodata(analysis_buffer_path)
            if not analysis_buffer_gdf.empty and "relation_id" in analysis_buffer_gdf.columns:
                relation_values = pd.to_numeric(analysis_buffer_gdf["relation_id"], errors="coerce").dropna()
                if not relation_values.empty:
                    street_relation_id = int(relation_values.iloc[0])
                    street_center_node_id = None
            if not analysis_buffer_gdf.empty and "centre_node_id" in analysis_buffer_gdf.columns:
                centre_values = pd.to_numeric(analysis_buffer_gdf["centre_node_id"], errors="coerce").dropna()
                if (street_relation_id is None) and (not centre_values.empty):
                    street_center_node_id = int(centre_values.iloc[0])
        except Exception:
            street_relation_id = args.relation_id
            street_center_node_id = args.center_node_id if street_relation_id is None else None
    street_grid_source_path, street_summary_path, street_rebuilt = _ensure_street_grid_from_repo(
        place=place,
        repo_root=repo_root,
        data_root=data_root,
        no_cache=args.no_cache,
        buffer_m=effective_buffer_m,
        grid_step=float(args.street_grid_step),
        min_road_count=int(args.street_min_road_count),
        min_total_road_length=float(args.street_min_total_road_length),
        relation_id=street_relation_id,
        center_node_id=street_center_node_id,
        boundary_path=collection_boundary_path,
        roads_path=shared_roads_path,
    )
    if args.no_cache or street_rebuilt or not clipped_street_grid_path.exists():
        _log("Clipping street-grid to analysis buffer...")
        clipped_street_grid_path, clipped_street_grid_count = _clip_street_grid_to_buffer(
            street_grid_path=street_grid_source_path,
            buffer_path=buffer_path,
            output_path=clipped_street_grid_path,
        )
        _log(f"Street-grid clipped by analysis buffer: {clipped_street_grid_count} features")
    else:
        clipped_street_grid_count = len(read_geodata(clipped_street_grid_path))
        _log(
            f"Using cached clipped street-grid: {_log_name(clipped_street_grid_path)} "
            f"({clipped_street_grid_count} features)"
        )
    downloaded_in_this_run = downloaded_in_this_run or street_rebuilt
    _log("Refreshing previews after street-pattern preparation...")
    _refresh_collection_previews("street_pattern_ready", floor_metrics_for_preview=floor_metrics)

    pt_street_dependency_manifest_path: Path | None = None
    pt_street_dependency_manifest: dict = {
        "enabled": False,
        "reason": "disabled_by_flag",
    }
    if bool(args.pt_street_pattern_dependency):
        pt_street_dependency_manifest_path, pt_street_dependency_manifest = _run_pt_street_pattern_dependency_step(
            repo_root=repo_root,
            city_dir=data_root,
            top_routes=int(args.pt_dependency_top_routes),
            subway_stop_buffer_m=float(args.pt_subway_stop_buffer_m),
        )
        dep_counts = pt_street_dependency_manifest.get("counts", {}) if isinstance(pt_street_dependency_manifest, dict) else {}
        _log(
            "PT x street-pattern dependency ready: "
            f"edges={int(pd.to_numeric(dep_counts.get('pt_edges_filtered'), errors='coerce') or 0)}, "
            f"overlay_segments={int(pd.to_numeric(dep_counts.get('overlay_segments'), errors='coerce') or 0)}, "
            f"routes={int(pd.to_numeric(dep_counts.get('routes_total'), errors='coerce') or 0)}"
        )
    else:
        _log("PT x street-pattern dependency step is disabled (--no-pt-street-pattern-dependency).")

    climate_grid_path = derived_dir / "climate_grid.parquet"

    climate_enabled = bool(args.climate_grid)
    if climate_enabled:
        if args.no_cache or not climate_grid_path.exists():
            _log("Building derived climate-grid layer...")
            _build_climate_grid(
                boundary_path=buffer_path,
                output_path=climate_grid_path,
                step_m=args.climate_grid_step_m,
            )
        else:
            _log(f"Using cached climate-grid layer: {_log_name(climate_grid_path)}")
    else:
        _log("Climate layer is disabled for this run (no --climate-grid was requested).")

    _log(
        "Final block aggregation stage: first ensure residents/living context on buildings, "
        "then aggregate to blocks."
    )
    blocks_path, buffered_blocks_path, buffered_blocks_count = _ensure_blocks_ready_at_end()
    _log("Refreshing previews after final block preparation...")
    _refresh_collection_previews("blocks_ready", floor_metrics_for_preview=floor_metrics)

    derived_dir.mkdir(parents=True, exist_ok=True)
    derived_manifest = {
        "place": place,
        "slug": slug,
        "generated_by": "aggregated_spatial_pipeline.pipeline.run_joint",
        "buffer_m": effective_buffer_m,
        "street_grid_step": args.street_grid_step,
        "street_min_road_count": args.street_min_road_count,
        "street_min_total_road_length": args.street_min_total_road_length,
        "effective_buffer_m": effective_buffer_m,
        "climate_grid_step_m": args.climate_grid_step_m,
        "files": {
            "blocks": str(buffered_blocks_path),
            "cities": str(buffer_path),
            "cities_collection": str(collection_boundary_path),
            "blocks_processed": str(blocks_path),
            "city_boundary_raw": str(boundary_path),
            "buildings_raw": str(buildings_path),
            "land_use_raw": str(land_use_path),
            "buildings_floor_enriched": str(floor_output_path),
            "street_grid": str(clipped_street_grid_path),
            "street_grid_source": str(street_grid_source_path),
            "climate_grid": str(climate_grid_path) if climate_enabled else None,
            "blocksnet_raw_manifest": str(blocks_raw_manifest_path),
            "blocksnet_manifest": str(blocks_manifest_path),
            "connectpt_manifest": str(connectpt_manifest_path),
            "intermodal_graph_manifest": str(intermodal_manifest_path),
            "street_pattern_summary": str(street_summary_path),
            "street_pattern_buffer": str(collection_boundary_path),
            "shared_drive_roads": str(shared_roads_path),
            "pt_street_pattern_dependency_manifest": (
                str(pt_street_dependency_manifest_path) if pt_street_dependency_manifest_path is not None else None
            ),
        },
        "floor_predictor": floor_metrics,
        "pt_street_pattern_dependency": pt_street_dependency_manifest,
        "data_coverage_check": {
            "min_data_coverage_pct_threshold": float(args.min_data_coverage_pct),
            "metrics": data_coverage_metrics,
        },
    }
    derived_manifest_path.write_text(json.dumps(derived_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    preview_dir = data_root / "preview_png" / "all_together"
    preview_paths = sorted(preview_dir.glob("*.png"))
    if preview_paths:
        _log(f"Preview PNG files are up to date: {len(preview_paths)} in {_log_name(preview_dir)}")
    else:
        _log("Preview PNG files were not generated yet (no readable non-empty layers).")

    layer_inputs = {
        "quarters": buffered_blocks_path,
        "street_grid": clipped_street_grid_path,
        **({"climate_grid": climate_grid_path} if climate_enabled else {}),
        "cities": buffer_path,
    }
    source_details = {
        "quarters": {
            "input_path": str(buffered_blocks_path),
            "manifest_path": str(derived_manifest_path),
            "place": place,
            "slug": slug,
            "downloaded_in_this_run": downloaded_in_this_run,
            "cached": (not args.no_cache and buffered_blocks_path.exists()),
            "origin": "blocks clipped to relation polygon" if relation_mode else f"blocks clipped to {effective_buffer_m}m buffer",
        },
        "street_grid": {
            "input_path": str(clipped_street_grid_path),
            "manifest_path": str(street_summary_path),
            "place": place,
            "slug": slug,
            "downloaded_in_this_run": downloaded_in_this_run,
            "cached": (not args.no_cache and clipped_street_grid_path.exists()),
            "origin": (
                "segregation-by-design-experiments predicted_cells.geojson built on relation polygon"
                if relation_mode
                else f"segregation-by-design-experiments predicted_cells.geojson built and clipped on the same exact {effective_buffer_m}m analysis buffer"
            ),
        },
        **(
            {
                "climate_grid": {
                    "input_path": str(climate_grid_path),
                    "manifest_path": str(derived_manifest_path),
                    "place": place,
                    "slug": slug,
                    "downloaded_in_this_run": downloaded_in_this_run,
                    "cached": (not args.no_cache and climate_grid_path.exists()),
                }
            }
            if climate_enabled
            else {}
        ),
        "cities": {
            "input_path": str(buffer_path),
            "manifest_path": str(street_summary_path),
            "place": place,
            "slug": slug,
            "center_source_csv": str(city_centers_csv),
            "downloaded_in_this_run": downloaded_in_this_run,
            "cached": (not args.no_cache and buffer_path.exists()),
            "origin": "main analysis relation polygon" if relation_mode else f"main analysis buffer polygon ({effective_buffer_m}m)",
            "data_coverage_check": {
                "min_data_coverage_pct_threshold": float(args.min_data_coverage_pct),
                "metrics": data_coverage_metrics,
            },
        },
    }
    return PreparedInputs(
        layer_inputs=layer_inputs,
        city_label=place,
        downloaded_in_this_run=downloaded_in_this_run,
        source_details=source_details,
    )


def _prepare_inputs(args: argparse.Namespace) -> PreparedInputs:
    if args.place or args.center_node_id is not None or args.relation_id is not None:
        prepared = _prepare_inputs_from_place(args)
        return PreparedInputs(
            layer_inputs=prepared.layer_inputs,
            city_label=args.city or prepared.city_label,
            downloaded_in_this_run=prepared.downloaded_in_this_run,
            source_details=prepared.source_details,
        )

    layer_inputs = {
        "quarters": Path(args.blocks),
        "street_grid": Path(args.street_grid),
        "cities": Path(args.cities),
        **({"climate_grid": Path(args.climate_grid)} if args.climate_grid else {}),
    }
    detected_city, source_details = _detect_city_from_inputs(layer_inputs)
    return PreparedInputs(
        layer_inputs=layer_inputs,
        city_label=args.city or detected_city,
        downloaded_in_this_run=False,
        source_details=source_details,
    )


def main() -> None:
    _configure_logging()
    args = parse_args()
    _configure_osm_requests(args.osm_timeout_s, debug=args.osmnx_debug, overpass_url=args.overpass_url)
    output_dir = _resolve_joint_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("Starting joint pipeline.")

    with tqdm(total=6, desc="Joint Pipeline", unit="step", **_tqdm_kwargs(leave=True)) as steps:
        steps.set_description("Joint Pipeline: validate spec")
        runtime_spec = PipelineSpec.load(Path(args.spec_dir))
        issues = runtime_spec.validate()
        if issues:
            raise SystemExit("Invalid pipeline spec:\n- " + "\n- ".join(issues))
        required_scenarios = _collect_required_scenarios(runtime_spec, JOINT_SCENARIO_ID)
        _log(f"Spec is valid. Runtime spec references {len(required_scenarios)} scenario id(s).")
        steps.update(1)

        steps.set_description("Joint Pipeline: prepare inputs")
        _log("PHASE 1/2: DATA COLLECTION & PREPARATION")
        prepared = _prepare_inputs(args)
        _log(f"Data download in this run: {'YES' if prepared.downloaded_in_this_run else 'NO'}")
        _log(f"Detected city: {prepared.city_label or 'unknown'}")
        _log_data_sources_summary(prepared.source_details)
        climate_meta = {"climate_enabled": "climate_grid" in prepared.layer_inputs}
        if not climate_meta["climate_enabled"]:
            runtime_spec, pruned_meta = _prune_climate_from_spec(runtime_spec)
            climate_meta.update(pruned_meta)
            required_scenarios = _collect_required_scenarios(runtime_spec, JOINT_SCENARIO_ID)
            _log("Climate is disabled: pruned climate crosswalks/rules from runtime spec.")
            _log(
                f"Removed crosswalks: {', '.join(climate_meta['removed_crosswalk_ids']) or 'none'}; "
                f"removed rules: {', '.join(climate_meta['removed_rule_ids']) or 'none'}"
            )
        steps.update(1)

        if args.collect_only:
            collect_manifest_path = output_dir / "manifest_collection.json"
            collect_manifest = {
                "phase": "collection_only",
                "detected_city": prepared.city_label,
                "downloaded_in_this_run": prepared.downloaded_in_this_run,
                "layers": {layer_id: str(path) for layer_id, path in prepared.layer_inputs.items()},
                "layer_sources": prepared.source_details,
                "climate": climate_meta,
            }
            collect_manifest_path.write_text(
                json.dumps(collect_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _log(f"Collection phase complete. Saved manifest: {_log_name(collect_manifest_path)}")
            _log("Stopping after phase 1 (--collect-only).")
            return

        steps.set_description("Joint Pipeline: load layers")
        _log("PHASE 2/2: JOINT CALCULATIONS")
        layers: dict[str, gpd.GeoDataFrame] = {}
        for layer_id, path in tqdm(prepared.layer_inputs.items(), desc="Loading layers", unit="layer", **_tqdm_kwargs()):
            _validate_layer_input_path(path, layer_id)
            try:
                gdf = load_layer(path, layer_id)
            except Exception as exc:
                raise SystemExit(
                    f"Failed to read layer {layer_id!r} from {path}: {exc}\n"
                    "Expected a valid Parquet/GeoJSON/GPKG with geometry and CRS."
                ) from exc
            layers[layer_id] = gdf
            _log(f"Loaded {layer_id}: {len(gdf)} features, CRS={_format_crs_for_log(gdf.crs)}")
        steps.update(1)

        steps.set_description("Joint Pipeline: build crosswalks")
        crosswalks = {}
        crosswalks_path = output_dir / "crosswalks.gpkg"
        for crosswalk in tqdm(runtime_spec.crosswalks["crosswalks"], desc="Building crosswalks", unit="cw", **_tqdm_kwargs()):
            crosswalk_id = crosswalk["crosswalk_id"]
            crosswalk_gdf = build_crosswalk(
                source_gdf=layers[crosswalk["source_layer"]],
                target_gdf=layers[crosswalk["target_layer"]],
                source_layer=crosswalk["source_layer"],
                target_layer=crosswalk["target_layer"],
            )
            crosswalks[crosswalk_id] = crosswalk_gdf
            save_crosswalk(crosswalk_gdf, crosswalks_path, crosswalk_id)
            _log(f"Built crosswalk {crosswalk_id}: {len(crosswalk_gdf)} intersections")
        steps.update(1)

        steps.set_description("Joint Pipeline: apply spec scenarios")
        all_results = run_scenarios(spec=runtime_spec, layers=layers, crosswalks=crosswalks)
        _log(
            "Spec scenario pass complete: "
            f"{len(all_results)} scenario id(s) materialized from runtime spec."
        )
        steps.update(1)

        steps.set_description("Joint Pipeline: save outputs")
        manifest = {
            "pipeline": "joint",
            "target_scenario": JOINT_SCENARIO_ID,
            "required_scenarios": required_scenarios,
            "downloaded_in_this_run": prepared.downloaded_in_this_run,
            "detected_city": prepared.city_label,
            "layers": {layer_id: str(path) for layer_id, path in prepared.layer_inputs.items()},
            "layer_sources": prepared.source_details,
            "crosswalk_layers": list(crosswalks.keys()),
            "climate": climate_meta,
            "scenarios": {},
        }
        for scenario_id in tqdm(required_scenarios, desc="Saving scenarios", unit="scenario", **_tqdm_kwargs()):
            result = all_results[scenario_id]
            scenario_dir = output_dir / scenario_id
            scenario_dir.mkdir(parents=True, exist_ok=True)
            blocks_path = scenario_dir / "blocks.parquet"
            cities_path = scenario_dir / "cities.parquet"
            metadata_path = scenario_dir / "metadata.json"

            save_layer(result.quarters, blocks_path)
            save_layer(result.cities, cities_path)
            metadata_path.write_text(
                json.dumps(result.metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _log(
                "Saved scenario "
                f"{scenario_id}: blocks={len(result.quarters)}, cities={len(result.cities)}, "
                f"pending_ops={len(result.metadata.get('pending_operations', []))}"
            )
            manifest["scenarios"][scenario_id] = {
                "blocks": str(blocks_path),
                "cities": str(cities_path),
                "metadata": str(metadata_path),
            }

        manifest_path = output_dir / "manifest_joint.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"Finished. Manifest: {_log_name(manifest_path)}")
        steps.update(1)


if __name__ == "__main__":
    main()

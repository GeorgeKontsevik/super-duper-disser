from __future__ import annotations

import argparse
import json
import math
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from loguru import logger
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata
from aggregated_spatial_pipeline.pandana_bridge import (
    build_graph_node_matrix_pandana_external,
    build_pairs_shortest_paths_pandana_external,
)
from aggregated_spatial_pipeline.pipeline.crosswalks import build_crosswalk
from aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs import _read_graph_pickle
from aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters import (
    CLASS_COLORS,
    CLASS_LABELS,
)
from aggregated_spatial_pipeline.pipeline.transfers import apply_transfer_rule
from aggregated_spatial_pipeline.runtime_config import configure_logger, ensure_repo_mplconfigdir
from aggregated_spatial_pipeline.visualization import (
    apply_preview_canvas,
    footer_text,
    normalize_preview_gdf,
    save_preview_figure,
)


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT
SUBPROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SUBPROJECT_ROOT / "outputs"
DEFAULT_JOINT_INPUT_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs"
DEFAULT_CITY_FALLBACK_ROOT = (
    REPO_ROOT / "segregation-by-design-experiments" / "service_accessibility_street_pattern" / "outputs"
)
DEFAULT_CITIES = ("warsaw_poland", "berlin_germany")
DEFAULT_MODALITIES = ("bus", "tram", "trolleybus")
DEFAULT_ROUTE_CRS = "EPSG:3857"
MIN_TERMINAL_BASELINE_M = 250.0
PROB_COLUMNS = list(CLASS_LABELS.keys())
RENAMED_PROB_COLUMNS = {
    prob_col: f"street_pattern_prob_{label.lower().replace(' & ', '_').replace(' ', '_').replace('-', '_').replace('__', '_')}"
    for prob_col, label in CLASS_LABELS.items()
}
MODEL_FEATURE_COLUMNS = list(RENAMED_PROB_COLUMNS.values())
BASELINE_FEATURE = next(
    RENAMED_PROB_COLUMNS[prob_col]
    for prob_col, label in CLASS_LABELS.items()
    if label == "Broken Grid"
)

ensure_repo_mplconfigdir("mpl-route-directness-street-pattern", root=REPO_ROOT)


@dataclass(frozen=True)
class CityPaths:
    slug: str
    city_dir: Path
    output_dir: Path
    boundary_path: Path
    blocks_path: Path
    street_cells_path: Path
    graph_path: Path
    service_accessibility_blocks_path: Path | None


def _configure_logging() -> None:
    configure_logger("[route-directness-street-pattern]")


def _log(message: str) -> None:
    logger.bind(tag="[route-directness-street-pattern]").info(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse existing PT route variants from intermodal_graph_iduedu against "
            "street-pattern context along route corridors."
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
        "--service-accessibility-root",
        default=str(DEFAULT_CITY_FALLBACK_ROOT),
        help="Optional root with prepared blocks_experiment.parquet from service_accessibility_street_pattern.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for this experiment subproject.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        help="PT modalities to analyse.",
    )
    parser.add_argument(
        "--corridor-buffer-m",
        type=float,
        default=300.0,
        help="Buffer around route geometry when aggregating street-pattern context.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild prepared route variants and stats.",
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
            "Use only already prepared experiment inputs. "
            "Do not rebuild block street-pattern features when prepared outputs are missing."
        ),
    )
    return parser.parse_args()


def _resolve_city_paths(
    slug: str,
    joint_input_root: Path,
    output_root: Path,
    service_accessibility_root: Path,
    *,
    use_sm_imputed: bool = False,
) -> CityPaths:
    city_dir = (joint_input_root / slug).resolve()
    if not city_dir.exists():
        raise FileNotFoundError(f"City bundle not found: {city_dir}")
    blocks_path = (
        city_dir / "derived_layers" / "quarters_sm_imputed.parquet"
        if use_sm_imputed
        else city_dir / "derived_layers" / "quarters_clipped.parquet"
    )
    paths = CityPaths(
        slug=slug,
        city_dir=city_dir,
        output_dir=(output_root / slug).resolve(),
        boundary_path=city_dir / "analysis_territory" / "buffer.parquet",
        blocks_path=blocks_path,
        street_cells_path=city_dir / "street_pattern" / slug / "predicted_cells.geojson",
        graph_path=city_dir / "intermodal_graph_iduedu" / "graph.pkl",
        service_accessibility_blocks_path=(
            (service_accessibility_root / slug / "prepared" / "blocks_experiment.parquet").resolve()
            if service_accessibility_root is not None
            else None
        ),
    )
    for check in (paths.boundary_path, paths.blocks_path, paths.street_cells_path, paths.graph_path):
        if not check.exists():
            raise FileNotFoundError(f"Required input not found for city {slug}: {check}")
    return paths


def _discover_available_city_slugs(
    joint_input_root: Path,
    output_root: Path,
    service_accessibility_root: Path,
    *,
    use_sm_imputed: bool = False,
) -> list[str]:
    slugs: list[str] = []
    for city_dir in sorted(p for p in joint_input_root.iterdir() if p.is_dir()):
        slug = city_dir.name
        try:
            _resolve_city_paths(
                slug,
                joint_input_root,
                output_root,
                service_accessibility_root,
                use_sm_imputed=use_sm_imputed,
            )
        except Exception:
            continue
        slugs.append(slug)
    return slugs


def _save_geodata(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prepare_geodata_for_parquet(gdf).to_parquet(path)


def _rename_prob_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    for prob_col, new_col in RENAMED_PROB_COLUMNS.items():
        if prob_col in renamed.columns:
            renamed[new_col] = pd.to_numeric(renamed[prob_col], errors="coerce").fillna(0.0)
    return renamed


def _load_boundary(boundary_path: Path) -> gpd.GeoDataFrame:
    boundary = read_geodata(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary layer is empty: {boundary_path}")
    return boundary


def _load_blocks(blocks_path: Path) -> gpd.GeoDataFrame:
    blocks = read_geodata(blocks_path).copy()
    if blocks.empty:
        raise ValueError(f"Blocks layer is empty: {blocks_path}")
    blocks["block_id"] = blocks.index.astype(str)
    return blocks


def _load_street_cells(street_cells_path: Path) -> gpd.GeoDataFrame:
    cells = gpd.read_file(street_cells_path)
    if cells.empty:
        raise ValueError(f"Street-pattern predicted_cells layer is empty: {street_cells_path}")
    cells["grid_id"] = cells["cell_id"].astype(str)
    return cells


def _transfer_street_pattern_to_blocks(
    blocks: gpd.GeoDataFrame,
    street_cells: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    source_columns = ["grid_id", *[c for c in street_cells.columns if c not in {"geometry", "grid_id"}], "geometry"]
    source = street_cells[source_columns].copy()
    target = blocks.copy()
    crosswalk = build_crosswalk(source, target, "grid", "block")

    transferred = apply_transfer_rule(
        source_gdf=source,
        target_gdf=target,
        crosswalk_gdf=crosswalk,
        source_layer="grid",
        target_layer="block",
        attribute="street_pattern_probs",
        aggregation_method="weighted_mean",
        weight_field="intersection_area",
    )
    transferred = apply_transfer_rule(
        source_gdf=source,
        target_gdf=transferred,
        crosswalk_gdf=crosswalk,
        source_layer="grid",
        target_layer="block",
        attribute="street_pattern_class",
        aggregation_method="majority_vote",
        weight_field="intersection_area",
    )
    transferred["street_pattern_covered_mass"] = transferred[PROB_COLUMNS].sum(axis=1)
    covered_mask = transferred["street_pattern_covered_mass"] > 0.0
    dominant = pd.Series("unknown", index=transferred.index, dtype=object)
    if covered_mask.any():
        dominant.loc[covered_mask] = (
            transferred.loc[covered_mask, PROB_COLUMNS].idxmax(axis=1).map(CLASS_LABELS).fillna("unknown")
        )
    transferred["street_pattern_dominant_class"] = dominant
    transferred = _rename_prob_columns(transferred)
    return transferred


def _load_blocks_with_street_pattern(
    paths: CityPaths,
    prepared_dir: Path,
    *,
    no_cache: bool,
    require_ready_data: bool,
) -> gpd.GeoDataFrame:
    output_path = prepared_dir / "blocks_with_street_pattern.parquet"
    if output_path.exists() and not no_cache:
        _log(f"[{paths.slug}] reusing prepared blocks_with_street_pattern.parquet")
        return read_geodata(output_path)

    if (
        paths.service_accessibility_blocks_path is not None
        and paths.service_accessibility_blocks_path.exists()
        and not no_cache
    ):
        _log(f"[{paths.slug}] reusing block street-pattern features from service_accessibility output")
        blocks = read_geodata(paths.service_accessibility_blocks_path)
        if not blocks.empty and {"block_id", "street_pattern_dominant_class", *MODEL_FEATURE_COLUMNS}.issubset(blocks.columns):
            _save_geodata(blocks, output_path)
            return blocks

    if require_ready_data:
        raise FileNotFoundError(
            "Missing prepared block-level street-pattern features. "
            f"Checked {output_path} and {paths.service_accessibility_blocks_path}."
        )

    blocks = _load_blocks(paths.blocks_path)
    street_cells = _load_street_cells(paths.street_cells_path)
    blocks = _transfer_street_pattern_to_blocks(blocks, street_cells)
    _save_geodata(blocks, output_path)
    return blocks


def _edge_line_from_nodes(node_u: dict, node_v: dict) -> LineString:
    return LineString([(float(node_u["x"]), float(node_u["y"])), (float(node_v["x"]), float(node_v["y"]))])


def _canonical_edge_attrs(
    graph: nx.MultiDiGraph,
    component_nodes: set[int],
    route: str,
    modality: str,
) -> list[tuple[int, int, dict]]:
    deduped: dict[tuple[int, int], dict] = {}
    for u, v, _key, data in graph.edges(component_nodes, keys=True, data=True):
        if u not in component_nodes or v not in component_nodes:
            continue
        if data.get("type") != modality or str(data.get("route")) != route:
            continue
        pair = tuple(sorted((int(u), int(v))))
        candidate = data.copy()
        candidate["length_meter"] = float(candidate.get("length_meter") or 0.0)
        candidate["time_min"] = float(candidate.get("time_min") or 0.0)
        if candidate.get("geometry") is None:
            candidate["geometry"] = _edge_line_from_nodes(graph.nodes[u], graph.nodes[v])
        if pair not in deduped or candidate["length_meter"] > deduped[pair]["length_meter"]:
            deduped[pair] = candidate
    return [(u, v, attrs) for (u, v), attrs in deduped.items()]


def _line_merge_safe(geometries: list) -> object:
    non_empty = [geom for geom in geometries if geom is not None and not geom.is_empty]
    if not non_empty:
        return None
    merged = unary_union(non_empty)
    try:
        merged = linemerge(merged)
    except Exception:
        pass
    return merged


def _choose_terminal_pair(graph: nx.Graph) -> tuple[int, int, float, float, list[int]]:
    if graph.number_of_nodes() < 2:
        raise ValueError("Need at least two nodes to choose terminals.")
    degree_one_nodes = [node for node, degree in graph.degree() if degree == 1]
    candidates = degree_one_nodes if len(degree_one_nodes) >= 2 else list(graph.nodes())
    best_pair: tuple[int, int] | None = None
    best_time = -1.0
    best_length = -1.0
    best_path: list[int] = []
    with tempfile.TemporaryDirectory(prefix="pandana-route-directness-", dir="/tmp") as tmp_dir:
        tmp_root = Path(tmp_dir)
        graph_path = tmp_root / "component_graph.pkl"
        matrix_path = tmp_root / "component_matrix.parquet"
        with graph_path.open("wb") as handle:
            pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        matrix = build_graph_node_matrix_pandana_external(
            graph_pickle_path=graph_path,
            output_path=matrix_path,
            weight_key="time_min",
            repo_root_path=REPO_ROOT,
        )
        for idx, source in enumerate(candidates):
            for target in candidates[idx + 1 :]:
                if source not in matrix.index or target not in matrix.columns:
                    continue
                time_val = float(matrix.loc[source, target])
                if not np.isfinite(time_val) or time_val <= best_time:
                    continue
                best_pair = (source, target)
                best_time = time_val
        if best_pair is not None:
            path_df = build_pairs_shortest_paths_pandana_external(
                graph_pickle_path=graph_path,
                pairs_df=pd.DataFrame([{"source": best_pair[0], "target": best_pair[1]}]),
                weight_key="time_min",
                repo_root_path=REPO_ROOT,
            )
            path = list(path_df.iloc[0]["path"])
            best_path = path
            best_length = 0.0
            for u, v in zip(path[:-1], path[1:]):
                best_length += float(graph[u][v].get("length_meter") or 0.0)
    if best_pair is None:
        nodes = list(graph.nodes())
        fallback = nodes[:2] if len(nodes) >= 2 else nodes
        return nodes[0], nodes[-1], 0.0, 0.0, fallback
    return best_pair[0], best_pair[1], best_time, best_length, best_path


def _build_component_geometry(graph: nx.Graph, path_nodes: list[int]) -> object:
    path_geometries = [graph[u][v].get("geometry") for u, v in zip(path_nodes[:-1], path_nodes[1:])]
    merged = _line_merge_safe(path_geometries)
    if merged is not None and not merged.is_empty:
        return merged
    coords = [(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])) for node in path_nodes]
    return LineString(coords)


def _extract_route_variants(
    graph: nx.MultiDiGraph,
    city_slug: str,
    modalities: list[str],
) -> gpd.GeoDataFrame:
    crs = graph.graph.get("crs")
    records: list[dict] = []
    for modality in modalities:
        route_labels = sorted(
            {
                str(data.get("route"))
                for _u, _v, _k, data in graph.edges(keys=True, data=True)
                if data.get("type") == modality and data.get("route") is not None
            }
        )
        _log(f"[{city_slug}] route extraction: modality={modality} routes={len(route_labels)}")
        for route in route_labels:
            nodes_in_route: set[int] = set()
            for u, v, _k, data in graph.edges(keys=True, data=True):
                if data.get("type") == modality and str(data.get("route")) == route:
                    nodes_in_route.add(int(u))
                    nodes_in_route.add(int(v))
            if len(nodes_in_route) < 2:
                continue
            component_edges = _canonical_edge_attrs(graph, nodes_in_route, route, modality)
            if not component_edges:
                continue
            undirected = nx.Graph()
            for u, v, attrs in component_edges:
                undirected.add_node(u, **graph.nodes[u])
                undirected.add_node(v, **graph.nodes[v])
                undirected.add_edge(u, v, **attrs)
            components = sorted(nx.connected_components(undirected), key=len, reverse=True)
            for component_rank, component_nodes in enumerate(components, start=1):
                component_graph = undirected.subgraph(component_nodes).copy()
                if component_graph.number_of_edges() == 0 or component_graph.number_of_nodes() < 2:
                    continue
                total_length_m = float(
                    sum(float(attrs.get("length_meter") or 0.0) for *_edge, attrs in component_graph.edges(data=True))
                )
                total_time_min = float(
                    sum(float(attrs.get("time_min") or 0.0) for *_edge, attrs in component_graph.edges(data=True))
                )
                start_node, end_node, spine_time_min, spine_length_m, path_nodes = _choose_terminal_pair(component_graph)
                spine_geometry = _build_component_geometry(component_graph, path_nodes)
                component_geometry = _line_merge_safe([attrs.get("geometry") for *_edge, attrs in component_graph.edges(data=True)])
                if component_geometry is None or component_geometry.is_empty:
                    component_geometry = spine_geometry
                x1 = float(component_graph.nodes[start_node]["x"])
                y1 = float(component_graph.nodes[start_node]["y"])
                x2 = float(component_graph.nodes[end_node]["x"])
                y2 = float(component_graph.nodes[end_node]["y"])
                terminal_straight_m = float(Point(x1, y1).distance(Point(x2, y2)))
                terminal_baseline_ok = terminal_straight_m >= MIN_TERMINAL_BASELINE_M
                branch_length_m = max(total_length_m - float(spine_length_m), 0.0)
                branch_time_min = max(total_time_min - float(spine_time_min), 0.0)
                branch_ratio = (
                    float(total_time_min) / float(spine_time_min)
                    if spine_time_min and np.isfinite(spine_time_min) and spine_time_min > 0
                    else np.nan
                )
                spine_detour_ratio = (
                    float(spine_length_m) / terminal_straight_m
                    if terminal_baseline_ok and np.isfinite(terminal_straight_m) and terminal_straight_m > 0
                    else np.nan
                )
                terminal_count = int(sum(1 for _node, degree in component_graph.degree() if degree == 1))
                records.append(
                    {
                        "city": city_slug,
                        "modality": modality,
                        "route": route,
                        "route_variant": f"{modality}:{route}:{component_rank}",
                        "route_component_rank": component_rank,
                        "route_component_count": len(components),
                        "node_count": component_graph.number_of_nodes(),
                        "edge_count": component_graph.number_of_edges(),
                        "terminal_count": terminal_count,
                        "route_length_m": total_length_m,
                        "route_time_min": total_time_min,
                        "spine_length_m": float(spine_length_m),
                        "spine_time_min": float(spine_time_min),
                        "branch_length_m": branch_length_m,
                        "branch_time_min": branch_time_min,
                        "branch_ratio": branch_ratio,
                        "terminal_straight_m": terminal_straight_m,
                        "terminal_baseline_ok": terminal_baseline_ok,
                        "spine_detour_ratio": spine_detour_ratio,
                        "route_time_per_km": total_time_min / max(total_length_m / 1000.0, 1e-6),
                        "route_time_per_terminal_km": (
                            total_time_min / max(terminal_straight_m / 1000.0, 1e-6) if terminal_baseline_ok else np.nan
                        ),
                        "geometry": component_geometry,
                        "spine_wkt": spine_geometry.wkt if spine_geometry is not None and not spine_geometry.is_empty else None,
                    }
                )
    if not records:
        return gpd.GeoDataFrame(
            {
                "city": pd.Series(dtype=str),
                "modality": pd.Series(dtype=str),
                "route": pd.Series(dtype=str),
                "route_variant": pd.Series(dtype=str),
                "geometry": gpd.GeoSeries([], crs=crs),
            },
            geometry="geometry",
            crs=crs,
        )
    variants = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    _log(f"[{city_slug}] route extraction complete: variants={len(variants)}")
    return variants


def _aggregate_route_corridor_pattern(
    routes_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
    *,
    corridor_buffer_m: float,
) -> gpd.GeoDataFrame:
    records: list[dict] = []
    blocks = blocks_gdf.copy()
    for route_row in routes_gdf.itertuples():
        corridor = route_row.geometry.buffer(corridor_buffer_m)
        intersecting = blocks[blocks.geometry.intersects(corridor)].copy()
        row: dict[str, object] = {"route_variant": route_row.route_variant}
        if intersecting.empty:
            for feature in MODEL_FEATURE_COLUMNS:
                row[feature] = 0.0
            row["corridor_dominant_class"] = "unknown"
            row["corridor_block_count"] = 0
            row["corridor_population_sum"] = 0.0
            row["corridor_area_m2"] = float(corridor.area)
            records.append(row)
            continue
        weights = intersecting.geometry.intersection(corridor).area.astype(float)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            weights = pd.Series(np.full(len(intersecting), 1.0 / len(intersecting)), index=intersecting.index)
        else:
            weights = weights / total_weight
        for feature in MODEL_FEATURE_COLUMNS:
            values = pd.to_numeric(intersecting.get(feature, 0.0), errors="coerce").fillna(0.0)
            row[feature] = float((values * weights).sum())
        dominant_scores = (
            weights.groupby(intersecting["street_pattern_dominant_class"].fillna("unknown")).sum().sort_values(ascending=False)
        )
        row["corridor_dominant_class"] = str(dominant_scores.index[0]) if not dominant_scores.empty else "unknown"
        row["corridor_block_count"] = int(len(intersecting))
        if "population" in intersecting.columns:
            row["corridor_population_sum"] = float(
                (pd.to_numeric(intersecting["population"], errors="coerce").fillna(0.0) * weights).sum()
            )
        else:
            row["corridor_population_sum"] = 0.0
        row["corridor_area_m2"] = float(corridor.area)
        records.append(row)
    metrics = pd.DataFrame.from_records(records)
    return routes_gdf.merge(metrics, on="route_variant", how="left")


def _compute_route_correlations(routes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    metric_columns = [
        "route_time_min",
        "branch_time_min",
        "branch_ratio",
        "spine_detour_ratio",
        "route_time_per_terminal_km",
    ]
    rows: list[dict] = []
    for modality in sorted(routes_gdf["modality"].dropna().unique()):
        subset = routes_gdf[routes_gdf["modality"] == modality].copy()
        for metric in metric_columns:
            for feature in MODEL_FEATURE_COLUMNS:
                valid = subset[[metric, feature]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) < 3 or valid[metric].nunique() < 2 or valid[feature].nunique() < 2:
                    rows.append(
                        {
                            "modality": modality,
                            "metric": metric,
                            "street_pattern_feature": feature,
                            "rho": np.nan,
                            "p_value": np.nan,
                            "n": len(valid),
                        }
                    )
                    continue
                rho, p_value = spearmanr(valid[metric], valid[feature])
                rows.append(
                    {
                        "modality": modality,
                        "metric": metric,
                        "street_pattern_feature": feature,
                        "rho": float(rho),
                        "p_value": float(p_value),
                        "n": len(valid),
                    }
                )
    return pd.DataFrame(rows)


def _fit_pooled_models(routes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = routes_df.copy()
    for column in ("route_time_min", "branch_ratio", "terminal_straight_m"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan)
    working = working[working["terminal_straight_m"] > 0].copy()
    working["log_terminal_straight_m"] = np.log(working["terminal_straight_m"])
    working["route_time_min_log"] = np.log1p(working["route_time_min"])
    working["branch_ratio_log"] = np.log(working["branch_ratio"].clip(lower=1.0))

    formulas = {
        "route_time_min_log": "route_time_min_log ~ C(city) + C(modality) + log_terminal_straight_m + " + " + ".join(
            feature for feature in MODEL_FEATURE_COLUMNS if feature != BASELINE_FEATURE
        ),
        "branch_ratio_log": "branch_ratio_log ~ C(city) + C(modality) + log_terminal_straight_m + " + " + ".join(
            feature for feature in MODEL_FEATURE_COLUMNS if feature != BASELINE_FEATURE
        ),
    }
    summary_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    for response, formula in formulas.items():
        dataset = working[[response, "city", "modality", "log_terminal_straight_m", *MODEL_FEATURE_COLUMNS]].dropna()
        if len(dataset) < 10:
            continue
        model = smf.ols(formula, data=dataset).fit()
        summary_rows.append(
            {
                "response": response,
                "n": int(model.nobs),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "baseline_feature": BASELINE_FEATURE,
            }
        )
        conf = model.conf_int()
        for term, coef in model.params.items():
            coefficient_rows.append(
                {
                    "response": response,
                    "term": term,
                    "coef": float(coef),
                    "std_err": float(model.bse[term]),
                    "p_value": float(model.pvalues[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "is_pattern_term": term in MODEL_FEATURE_COLUMNS,
                    "baseline_feature": BASELINE_FEATURE,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(coefficient_rows)


def _plot_city_route_metrics(routes_gdf: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame, preview_dir: Path) -> None:
    plot_gdf = routes_gdf.copy()
    plot_gdf["branch_ratio_plot"] = plot_gdf["branch_ratio"].clip(upper=np.nanquantile(plot_gdf["branch_ratio"].dropna(), 0.95))
    for metric, filename, title in (
        ("branch_ratio_plot", "01_route_branch_ratio_by_corridor_class.png", "Route branch ratio by corridor class"),
        ("route_time_per_terminal_km", "02_route_time_per_terminal_km_by_corridor_class.png", "Route time per terminal km"),
    ):
        subset = plot_gdf[plot_gdf["corridor_dominant_class"] != "unknown"].copy()
        if subset.empty:
            continue
        order = [label for label in CLASS_LABELS.values() if label in set(subset["corridor_dominant_class"])]
        if not order:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(
            data=subset,
            x="corridor_dominant_class",
            y=metric,
            hue="modality",
            order=order,
            ax=ax,
            showfliers=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        save_preview_figure(fig, preview_dir / filename)
        plt.close(fig)

    if routes_gdf.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 9))
    boundary_plot = normalize_preview_gdf(boundary)
    route_plot = normalize_preview_gdf(routes_gdf[["branch_ratio", "modality", "geometry"]], boundary_plot)
    apply_preview_canvas(fig, ax, boundary_plot, title="Existing PT route variants by branch ratio")
    if route_plot is not None and not route_plot.empty:
        route_plot.plot(
            ax=ax,
            column="branch_ratio",
            cmap="magma_r",
            linewidth=2.0,
            legend=True,
            zorder=5,
        )
    ax.set_axis_off()
    footer_text(
        fig,
        [
            "branch_ratio = total route time / main terminal-to-terminal spine time",
            "route_time_min aligns with connectpt route_cost; branch_ratio is a candidate directness penalty.",
        ],
        bbox={"facecolor": "#f7f0dd", "alpha": 0.95, "edgecolor": "#d6c7a1"},
    )
    save_preview_figure(fig, preview_dir / "03_route_variants_branch_ratio_map.png")
    plt.close(fig)


def _plot_pooled_coefficients(coefficients: pd.DataFrame, preview_dir: Path) -> None:
    if coefficients.empty:
        return
    response_specs = [
        ("route_time_min_log", "10_pooled_route_time_coefficients.png", "Pooled route-time effects"),
        ("branch_ratio_log", "11_pooled_branch_ratio_coefficients.png", "Pooled branch-ratio effects"),
    ]
    for response, filename, title in response_specs:
        subset = coefficients[(coefficients["response"] == response) & (coefficients["is_pattern_term"])].copy()
        if subset.empty:
            continue
        subset["label"] = subset["term"].map(
            {
                RENAMED_PROB_COLUMNS[key]: label
                for key, label in CLASS_LABELS.items()
            }
        )
        subset = subset.sort_values("coef")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = np.where(subset["p_value"] < 0.05, "#0f766e", "#dc2626")
        for row, color in zip(subset.itertuples(), colors):
            ax.errorbar(
                row.coef,
                row.label,
                xerr=[[row.coef - row.ci_low], [row.ci_high - row.coef]],
                fmt="none",
                ecolor=color,
                elinewidth=1.7,
                capsize=3,
                zorder=2,
            )
            ax.scatter(row.coef, row.label, color=color, s=46, zorder=3)
        ax.axvline(0.0, color="#4b5563", linestyle="--", linewidth=1.0, zorder=1)
        ax.set_title(title)
        ax.set_xlabel("Coefficient")
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#0f766e", label="p < 0.05", markersize=8),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc2626", label="not significant", markersize=8),
        ]
        ax.legend(handles=legend_handles, loc="lower right", frameon=True)
        fig.tight_layout()
        save_preview_figure(fig, preview_dir / filename)
        plt.close(fig)


def _write_city_outputs(
    city_paths: CityPaths,
    routes_gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    correlations: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    prepared_dir = city_paths.output_dir / "prepared"
    stats_dir = city_paths.output_dir / "stats"
    preview_dir = city_paths.output_dir / "preview_png"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    _save_geodata(routes_gdf, prepared_dir / "route_variants.parquet")
    correlations.to_csv(stats_dir / "route_pattern_correlations.csv", index=False)
    (stats_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _plot_city_route_metrics(routes_gdf, boundary, preview_dir)


def _write_cross_city_outputs(
    output_root: Path,
    all_routes: gpd.GeoDataFrame,
    model_summary: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    cross_dir = output_root / "_cross_city"
    stats_dir = cross_dir / "stats"
    preview_dir = cross_dir / "preview_png"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    _save_geodata(all_routes, cross_dir / "route_variants_all_cities.parquet")
    model_summary.to_csv(stats_dir / "pooled_model_summary.csv", index=False)
    coefficients.to_csv(stats_dir / "pooled_model_coefficients.csv", index=False)
    _plot_pooled_coefficients(coefficients, preview_dir)

    if not all_routes.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ordered = (
            all_routes.groupby("city")["branch_ratio"].median().sort_values().index.tolist()
        )
        sns.boxplot(
            data=all_routes.replace([np.inf, -np.inf], np.nan).dropna(subset=["branch_ratio"]),
            x="city",
            y="branch_ratio",
            hue="modality",
            order=ordered,
            showfliers=False,
            ax=ax,
        )
        ax.set_title("Existing PT route branch ratio by city")
        ax.set_xlabel("")
        ax.set_ylabel("branch_ratio")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        save_preview_figure(fig, preview_dir / "12_branch_ratio_by_city.png")
        plt.close(fig)


def main() -> None:
    _configure_logging()
    args = parse_args()
    joint_input_root = Path(args.joint_input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    service_accessibility_root = Path(args.service_accessibility_root).expanduser().resolve()
    if args.cities == ["all"]:
        cities = _discover_available_city_slugs(
            joint_input_root,
            output_root,
            service_accessibility_root,
            use_sm_imputed=bool(args.use_sm_imputed),
        )
    else:
        cities = args.cities

    _log(
        "Running route directness vs street-pattern experiments for "
        + ", ".join(cities)
    )

    all_routes: list[gpd.GeoDataFrame] = []
    for slug in cities:
        paths = _resolve_city_paths(
            slug,
            joint_input_root,
            output_root,
            service_accessibility_root,
            use_sm_imputed=bool(args.use_sm_imputed),
        )
        prepared_dir = paths.output_dir / "prepared"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        route_variants_path = prepared_dir / "route_variants.parquet"
        boundary = _load_boundary(paths.boundary_path)
        if route_variants_path.exists() and not args.no_cache:
            routes_gdf = read_geodata(route_variants_path)
        else:
            _log(f"[{slug}] loading blocks and street-pattern context.")
            blocks = _load_blocks_with_street_pattern(
                paths,
                prepared_dir,
                no_cache=args.no_cache,
                require_ready_data=bool(args.require_ready_data),
            )
            graph = _read_graph_pickle(paths.graph_path)
            _log(f"[{slug}] extracting PT route variants from intermodal graph.")
            routes_gdf = _extract_route_variants(graph, slug, list(args.modalities))
            if routes_gdf.empty:
                logger.bind(tag="[route-directness-street-pattern]").warning(
                    f"[{slug}] no PT route variants found for modalities {args.modalities}; skipping city."
                )
                continue
            blocks = blocks.to_crs(routes_gdf.crs)
            routes_gdf = _aggregate_route_corridor_pattern(
                routes_gdf,
                blocks,
                corridor_buffer_m=float(args.corridor_buffer_m),
            )
        correlations = _compute_route_correlations(routes_gdf)
        summary = {
            "city": slug,
            "modalities": list(sorted(routes_gdf["modality"].dropna().unique())),
            "route_variant_count": int(len(routes_gdf)),
            "median_branch_ratio": float(pd.to_numeric(routes_gdf["branch_ratio"], errors="coerce").median()),
            "median_route_time_min": float(pd.to_numeric(routes_gdf["route_time_min"], errors="coerce").median()),
            "alignment_notes": {
                "route_time_min": "Direct analogue of connectpt total_route_time / route_cost component.",
                "branch_time_min": "Extra route time beyond the main terminal-to-terminal spine; candidate directness penalty.",
                "branch_ratio": "Normalized branch penalty = route_time_min / spine_time_min.",
            },
        }
        _log(f"[{slug}] writing route-level outputs.")
        _write_city_outputs(paths, routes_gdf, boundary, correlations, summary)
        all_routes.append(routes_gdf.to_crs(DEFAULT_ROUTE_CRS))

    if all_routes:
        merged = pd.concat(all_routes, ignore_index=True)
        all_routes_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=all_routes[0].crs)
        model_summary, coefficients = _fit_pooled_models(pd.DataFrame(all_routes_gdf.drop(columns="geometry")))
        _write_cross_city_outputs(output_root, all_routes_gdf, model_summary, coefficients)


if __name__ == "__main__":
    main()

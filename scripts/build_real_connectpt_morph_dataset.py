#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter, deque
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
CONNECTPT_ROOT = ROOT / "connectpt"
for candidate in (ROOT, CONNECTPT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aggregated_spatial_pipeline.geodata_io import read_geodata  # noqa: E402
from aggregated_spatial_pipeline.connectpt_data_pipeline.run_route_generator_external import (  # noqa: E402
    _build_node_locs,
    _build_stops_gdf_from_graph,
    _build_street_adj,
    _compute_od_matrix,
    _ensure_length_m_edges,
    _prepare_blocks_for_demand,
)
from aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters import CLASS_LABELS  # noqa: E402
from connectpt.routes_generator.citygraph_dataset import (  # noqa: E402
    RAW_GRAPH_FILENAME,
    STOP_KEY,
    CityGraphData,
    CityGraphDataset,
)


DEFAULT_CITY_ROOTS = [
    ROOT / "aggregated_spatial_pipeline" / "outputs" / "active_19_good_cities_20260412" / "joint_inputs",
    ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "old"
    / "cities_with_street_grid_and_routes_20260412"
    / "batch_runs"
    / "random50_pop200k_10km"
    / "joint_inputs",
]
DEFAULT_OUTPUT_DIR = ROOT / "connectpt" / "datasets" / "real_morph_50nodes_local"

CLASS_NAMES = [CLASS_LABELS[key] for key in sorted(CLASS_LABELS)]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
FOCUS_CLASS_NAME = "Loops & Lollipops"
FOCUS_CLASS_ID = CLASS_TO_ID[FOCUS_CLASS_NAME]
UNKNOWN_CLASS_ID = -1


class SkipCity(RuntimeError):
    """City cannot produce a valid non-fallback training sample."""


class SkipSample(RuntimeError):
    """Sample cannot produce a valid non-fallback training example."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ConnectPT pickle training dataset from real simplified graph bundles, "
            "street-pattern labels, and OD demand when blocks are available."
        )
    )
    parser.add_argument(
        "--city-root",
        action="append",
        default=None,
        help="Root containing per-city joint input dirs. Can be repeated.",
    )
    parser.add_argument("--city-dir", action="append", default=None, help="Explicit city dir. Can be repeated.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--modality", default="bus")
    parser.add_argument(
        "--graph-source",
        choices=("auto", "connectpt", "roads"),
        default="auto",
        help=(
            "auto/connectpt use only a ConnectPT modality graph large enough for target-nodes; "
            "roads is an explicit separate experiment using derived drive roads."
        ),
    )
    parser.add_argument("--speed-kmh", type=float, default=20.0)
    parser.add_argument("--target-nodes", type=int, default=50)
    parser.add_argument("--samples-per-city", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-focus-class-share",
        type=float,
        default=0.25,
        help=(
            "Minimum Loops & Lollipops node share required in every sampled graph. "
            "Set 0 to disable loop-heavy oversampling."
        ),
    )
    parser.add_argument(
        "--sample-search-attempts",
        type=int,
        default=500,
        help="Maximum random connected-crop attempts per accepted sample.",
    )
    parser.add_argument(
        "--demand-source",
        choices=("auto", "gravity", "synthetic"),
        default="auto",
        help=(
            "auto uses blocks-based ConnectPT OD and skips cities/samples without usable gravity OD; "
            "synthetic must be requested explicitly."
        ),
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Also instantiate CityGraphDataset to write processed/collated.pt.",
    )
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[real-morph-dataset] {message}", flush=True)


def _resolve_city_dirs(args: argparse.Namespace) -> list[Path]:
    explicit = [Path(p).resolve() for p in (args.city_dir or [])]
    if args.city_root is not None:
        roots = [Path(p).resolve() for p in args.city_root]
    elif explicit:
        roots = []
    else:
        roots = list(DEFAULT_CITY_ROOTS)
    city_dirs: list[Path] = []
    city_dirs.extend(explicit)
    for root in roots:
        if not root.exists():
            continue
        city_dirs.extend(path for path in sorted(root.iterdir()) if path.is_dir())

    seen: set[Path] = set()
    result: list[Path] = []
    for city_dir in city_dirs:
        if city_dir in seen:
            continue
        seen.add(city_dir)
        result.append(city_dir)
    return result


def _street_cells_path(city_dir: Path) -> Path | None:
    candidates = [
        city_dir / "derived_layers" / "street_grid_buffered.parquet",
        city_dir / "derived_layers" / "street_grid_clipped.parquet",
        city_dir / "street_pattern" / city_dir.name / "predicted_cells.geojson",
    ]
    candidates.extend(sorted((city_dir / "street_pattern").glob("*/predicted_cells.geojson")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _blocks_path(city_dir: Path) -> Path | None:
    candidates = [
        city_dir / "derived_layers" / "blocks_clipped.parquet",
        city_dir / "derived_layers" / "blocks_sm_imputed.parquet",
        city_dir / "blocksnet" / "blocks.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_connectpt_graph(city_dir: Path, modality: str):
    graph_path = city_dir / "connectpt_osm" / modality / "graph.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    with graph_path.open("rb") as fh:
        graph = pickle.load(fh)
    _ensure_length_m_edges(graph)
    return graph


def _node_key(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), 2), round(float(y), 2))


def _iter_lines(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
    elif geometry.geom_type == "MultiLineString":
        yield from geometry.geoms


def _load_roads_graph(city_dir: Path, speed_kmh: float):
    roads_path = city_dir / "derived_layers" / "roads_drive_osmnx.parquet"
    if not roads_path.exists():
        raise FileNotFoundError(roads_path)
    roads = read_geodata(roads_path)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if roads.empty:
        raise ValueError(f"Roads layer is empty: {roads_path}")
    if roads.crs is None:
        roads = roads.set_crs(4326)
    if roads.crs.is_geographic:
        local_crs = roads.estimate_utm_crs() or "EPSG:3857"
        roads = roads.to_crs(local_crs)

    graph = nx.Graph()
    graph.graph["crs"] = roads.crs
    key_to_node: dict[tuple[float, float], int] = {}

    def _get_node(x: float, y: float) -> int:
        key = _node_key(x, y)
        node_id = key_to_node.get(key)
        if node_id is None:
            node_id = len(key_to_node)
            key_to_node[key] = node_id
            graph.add_node(node_id, x=float(key[0]), y=float(key[1]), is_stop=True)
        return node_id

    meters_per_minute = float(speed_kmh) * 1000.0 / 60.0
    for geom in roads.geometry:
        for line in _iter_lines(geom):
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            u = _get_node(coords[0][0], coords[0][1])
            v = _get_node(coords[-1][0], coords[-1][1])
            if u == v:
                continue
            length_m = float(line.length)
            if length_m <= 0.0:
                continue
            time_min = length_m / meters_per_minute if meters_per_minute > 0 else length_m
            if graph.has_edge(u, v):
                if float(graph[u][v].get("length_m", float("inf"))) <= length_m:
                    continue
            graph.add_edge(
                u,
                v,
                weight=length_m,
                length_m=length_m,
                time_min=time_min,
                geometry=line,
            )

    if graph.number_of_edges() == 0:
        raise ValueError(f"Could not build road graph from {roads_path}")
    largest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def _load_training_graph(city_dir: Path, modality: str, graph_source: str, target_nodes: int, speed_kmh: float):
    if graph_source in {"auto", "connectpt"}:
        try:
            graph = _load_connectpt_graph(city_dir, modality)
            if graph.number_of_nodes() >= target_nodes and graph.number_of_edges() > 0:
                return graph, "connectpt"
            raise ValueError(
                f"ConnectPT {modality} graph too small for target_nodes={target_nodes}: "
                f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Skipping city because usable ConnectPT graph is missing: {exc}") from exc
    graph = _load_roads_graph(city_dir, speed_kmh)
    return graph, "roads"


def _pick_class_column(cells: gpd.GeoDataFrame) -> str:
    for column in ("top1_class_name", "class_name", "street_pattern_class"):
        if column in cells.columns:
            return column
    raise ValueError("Street-pattern cells do not contain a known class column.")


def _assign_street_pattern_classes(city_dir: Path, stops: gpd.GeoDataFrame) -> tuple[torch.Tensor, dict]:
    cells_path = _street_cells_path(city_dir)
    if cells_path is None:
        return torch.full((len(stops),), UNKNOWN_CLASS_ID, dtype=torch.long), {
            "enabled": False,
            "reason": "missing_cells",
            "cells_path": None,
            "matched_stops": 0,
            "total_stops": int(len(stops)),
        }

    cells = read_geodata(cells_path)
    if cells.empty:
        return torch.full((len(stops),), UNKNOWN_CLASS_ID, dtype=torch.long), {
            "enabled": False,
            "reason": "empty_cells",
            "cells_path": str(cells_path),
            "matched_stops": 0,
            "total_stops": int(len(stops)),
        }

    class_col = _pick_class_column(cells)
    cells = cells[[class_col, "geometry"]].copy()
    cells = cells[cells.geometry.notna() & ~cells.geometry.is_empty].copy()
    if cells.crs is None and stops.crs is not None:
        cells = cells.set_crs(stops.crs)
    if stops.crs is not None and cells.crs is not None and cells.crs != stops.crs:
        cells = cells.to_crs(stops.crs)

    joined = stops[["graph_node_id", "geometry"]].sjoin(
        cells[[class_col, "geometry"]],
        how="left",
        predicate="within",
    )
    if joined.index.duplicated().any():
        joined = joined[~joined.index.duplicated(keep="first")]

    labels = pd.Series(UNKNOWN_CLASS_ID, index=stops.index, dtype="int64")
    matched = joined[class_col].dropna().astype(str).map(CLASS_TO_ID)
    labels.loc[matched.index] = matched.fillna(UNKNOWN_CLASS_ID).astype("int64")
    tensor = torch.tensor(labels.to_numpy(), dtype=torch.long)
    return tensor, {
        "enabled": True,
        "cells_path": str(cells_path),
        "class_col": class_col,
        "class_to_id": CLASS_TO_ID,
        "focus_class_name": FOCUS_CLASS_NAME,
        "focus_class_id": FOCUS_CLASS_ID,
        "matched_stops": int((tensor >= 0).sum().item()),
        "total_stops": int(len(tensor)),
        "unmatched_stops": int((tensor < 0).sum().item()),
    }


def _synthetic_demand(node_locs: torch.Tensor, class_ids: torch.Tensor, rng: random.Random) -> torch.Tensor:
    coords = node_locs.detach().cpu().numpy().astype("float64")
    n_nodes = coords.shape[0]
    if n_nodes <= 1:
        return torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    positive = dists[dists > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1.0)

    ids = class_ids.detach().cpu().numpy().astype("int64")
    class_weight = np.ones(n_nodes, dtype="float64")
    class_weight[ids == CLASS_TO_ID["Regular Grid"]] = 1.25
    class_weight[ids == CLASS_TO_ID["Irregular Grid"]] = 1.05
    class_weight[ids == CLASS_TO_ID["Warped Parallel"]] = 1.00
    class_weight[ids == CLASS_TO_ID["Broken Grid"]] = 0.85
    class_weight[ids == CLASS_TO_ID["Sparse"]] = 0.65
    class_weight[ids == FOCUS_CLASS_ID] = 0.70
    class_weight[ids < 0] = 0.80

    center = coords.mean(axis=0)
    centrality = np.exp(-np.linalg.norm(coords - center, axis=1) / (scale * 1.5))
    activity = class_weight * (0.65 + centrality)
    noise = np.array([rng.lognormvariate(0.0, 0.25) for _ in range(n_nodes)])
    activity *= noise

    demand = activity[:, None] * activity[None, :] * np.exp(-dists / (scale * 1.15))
    demand = (demand + demand.T) / 2.0
    np.fill_diagonal(demand, 0.0)
    positive_mask = demand > 0
    if positive_mask.any():
        demand = demand / demand[positive_mask].max()
        demand = demand * (800.0 - 60.0) + 60.0
        demand[~positive_mask] = 0.0
    return torch.tensor(demand, dtype=torch.float32)


def _connected_sample_nodes(
    graph: nx.Graph,
    target_nodes: int,
    rng: random.Random,
    *,
    full_node_to_idx: dict[int, int] | None = None,
    full_classes: torch.Tensor | None = None,
    min_focus_nodes: int = 0,
    max_attempts: int = 100,
) -> list[int] | None:
    undirected = graph.to_undirected()
    components = [list(c) for c in nx.connected_components(undirected)]
    components = [c for c in components if len(c) >= target_nodes]
    if not components:
        return None
    component = max(components, key=len)

    def _is_focus(node) -> bool:
        if full_node_to_idx is None or full_classes is None:
            return False
        idx = full_node_to_idx.get(node)
        if idx is None:
            return False
        return int(full_classes[idx].item()) == FOCUS_CLASS_ID

    def _focus_count(sampled: list) -> int:
        if min_focus_nodes <= 0:
            return 0
        return sum(1 for node in sampled if _is_focus(node))

    def _accept(sampled: list) -> bool:
        if len(sampled) != target_nodes:
            return False
        if not nx.is_connected(undirected.subgraph(sampled)):
            return False
        return min_focus_nodes <= 0 or _focus_count(sampled) >= min_focus_nodes

    focus_nodes = [node for node in component if _is_focus(node)]
    if min_focus_nodes > 0 and len(focus_nodes) < min_focus_nodes:
        return None

    best_sampled: list | None = None
    best_focus_count = -1
    for _ in range(max(1, int(max_attempts))):
        if min_focus_nodes > 0 and focus_nodes and rng.random() < 0.85:
            seed = rng.choice(focus_nodes)
        else:
            seed = rng.choice(component)
        seen = {seed}
        queue: deque[int] = deque([seed])
        while queue and len(seen) < target_nodes:
            node = queue.popleft()
            neighbors = list(undirected.neighbors(node))
            rng.shuffle(neighbors)
            if min_focus_nodes > 0:
                neighbors.sort(key=lambda candidate: _is_focus(candidate), reverse=True)
            for neighbor in neighbors:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
                if len(seen) >= target_nodes:
                    break
        if len(seen) >= target_nodes:
            sampled = list(seen)
            focus_count = _focus_count(sampled)
            if focus_count > best_focus_count:
                best_sampled = sampled
                best_focus_count = focus_count
            if _accept(sampled):
                return sorted(sampled)

    seed = rng.choice(component)
    lengths = nx.single_source_shortest_path_length(undirected.subgraph(component), seed)
    sampled = [node for node, _ in sorted(lengths.items(), key=lambda item: (item[1], str(item[0])))[:target_nodes]]
    if _accept(sampled):
        return sorted(sampled)
    if best_sampled is not None and _accept(best_sampled):
        return sorted(best_sampled)
    return None


def _build_city_od(
    *,
    city_dir: Path,
    graph,
    nodes: list[int],
    stops: gpd.GeoDataFrame,
    demand_source: str,
    graph_source: str,
) -> tuple[pd.DataFrame | None, str, str | None]:
    if demand_source == "synthetic":
        return None, "synthetic", "explicit_synthetic"
    blocks_path = _blocks_path(city_dir)
    if blocks_path is None:
        raise SkipCity("missing_blocks")
    try:
        blocks = _prepare_blocks_for_demand(blocks_path)
        od_matrix = _compute_od_matrix(blocks, stops, graph)
        od_matrix = od_matrix.reindex(index=range(len(nodes)), columns=range(len(nodes)), fill_value=0.0)
        od_matrix = od_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
        if float(od_matrix.to_numpy().sum()) <= 0.0:
            raise SkipCity("zero_gravity_od")
        return od_matrix, "gravity", str(blocks_path)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, SkipCity):
            raise
        raise SkipCity(f"gravity_failed: {exc}") from exc


def _make_sample(
    *,
    graph,
    sample_nodes: list[int],
    full_node_to_idx: dict[int, int],
    full_od: pd.DataFrame | None,
    full_classes: torch.Tensor,
    rng: random.Random,
) -> tuple[CityGraphData, dict]:
    subgraph = graph.subgraph(sample_nodes).copy()
    nodes, node_to_idx, _ = _build_stops_gdf_from_graph(subgraph, subgraph.graph.get("crs"))
    full_indices = [full_node_to_idx[node] for node in nodes]

    node_locs = _build_node_locs(subgraph, nodes)
    street_adj = _build_street_adj(subgraph, nodes, node_to_idx)
    class_ids = full_classes[full_indices].clone()

    if full_od is None:
        demand = _synthetic_demand(node_locs, class_ids, rng)
        sample_demand_source = "synthetic"
    else:
        demand_values = full_od.to_numpy(dtype="float32")
        demand = torch.tensor(demand_values[np.ix_(full_indices, full_indices)], dtype=torch.float32)
        if float(demand.sum().item()) <= 0.0:
            raise SkipSample("zero_gravity_od_slice")
        else:
            sample_demand_source = "gravity"

    tensors = {
        "node_locs": node_locs,
        "street_adj": street_adj,
        "demand": demand,
        "street_pattern_classes": class_ids,
        "focus_class_id": torch.tensor([FOCUS_CLASS_ID], dtype=torch.long),
        "street_pattern_class_weights": torch.ones((len(CLASS_NAMES),), dtype=torch.float32),
    }
    data = CityGraphData.from_tensors_with_transformations(
        tensors,
        scale_dynamically=False,
        extra_node_feats=True,
        fully_connected_demand=True,
        center_nodes=True,
    )
    # CityGraphDataset applies InsertPosFeatures at load time; keep raw x to
    # non-pos node features so the processed model input stays 4-dimensional.
    data[STOP_KEY].x = data[STOP_KEY].x[:, 2:].clone()
    meta = {
        "node_count": int(len(nodes)),
        "edge_count": int(subgraph.number_of_edges()),
        "demand_source": sample_demand_source,
        "demand_sum": float(demand.sum().item()),
        "focus_class_count": int((class_ids == FOCUS_CLASS_ID).sum().item()),
        "focus_class_share": float((class_ids == FOCUS_CLASS_ID).float().mean().item()) if len(class_ids) else 0.0,
        "class_counts": {
            CLASS_NAMES[class_id]: int((class_ids == class_id).sum().item())
            for class_id in range(len(CLASS_NAMES))
        },
        "unknown_class_count": int((class_ids < 0).sum().item()),
    }
    return data, meta


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    raw_path = output_dir / RAW_GRAPH_FILENAME
    manifest_path = output_dir / "manifest.json"
    if args.no_overwrite and (raw_path.exists() or manifest_path.exists()):
        raise FileExistsError(f"Output already exists: {output_dir}")

    rng = random.Random(int(args.seed))
    city_dirs = _resolve_city_dirs(args)
    rng.shuffle(city_dirs)
    min_focus_class_share = max(0.0, min(1.0, float(args.min_focus_class_share)))
    min_focus_nodes = int(np.ceil(int(args.target_nodes) * min_focus_class_share)) if min_focus_class_share > 0 else 0

    output_dir.mkdir(parents=True, exist_ok=True)
    data_list: list[CityGraphData] = []
    city_rows: list[dict] = []
    sample_rows: list[dict] = []
    global_class_counts: Counter[str] = Counter()
    demand_sources: Counter[str] = Counter()

    _log(
        f"source_cities={len(city_dirs)}, target_nodes={int(args.target_nodes)}, "
        f"samples_per_city={int(args.samples_per_city)}, max_samples={int(args.max_samples)}, "
        f"min_focus_class_share={min_focus_class_share:.2f}, min_focus_nodes={min_focus_nodes}"
    )

    for city_dir in city_dirs:
        if len(data_list) >= int(args.max_samples):
            break
        row = {
            "city": city_dir.name,
            "city_dir": str(city_dir),
            "status": "ok",
            "samples": 0,
            "graph_nodes": None,
            "graph_edges": None,
            "graph_source": None,
            "street_pattern": None,
            "demand_source": None,
            "error": None,
        }
        try:
            graph, graph_source = _load_training_graph(
                city_dir,
                str(args.modality),
                str(args.graph_source),
                int(args.target_nodes),
                float(args.speed_kmh),
            )
            nodes, node_to_idx, stops = _build_stops_gdf_from_graph(graph, graph.graph.get("crs"))
            row["graph_nodes"] = int(graph.number_of_nodes())
            row["graph_edges"] = int(graph.number_of_edges())
            row["graph_source"] = graph_source
            if graph.number_of_nodes() < int(args.target_nodes):
                row["status"] = "skipped_too_small"
                city_rows.append(row)
                continue

            full_classes, street_meta = _assign_street_pattern_classes(city_dir, stops)
            row["street_pattern"] = street_meta
            graph_focus_nodes = int((full_classes == FOCUS_CLASS_ID).sum().item())
            row["focus_class_nodes"] = graph_focus_nodes
            row["focus_class_share"] = float(graph_focus_nodes / max(len(full_classes), 1))
            row["min_focus_class_nodes"] = min_focus_nodes
            if min_focus_nodes > 0 and graph_focus_nodes < min_focus_nodes:
                row["status"] = "skipped_insufficient_focus_class_nodes"
                city_rows.append(row)
                _log(f"{city_dir.name}: {row['status']} samples={row['samples']}")
                continue
            full_od, city_demand_source, demand_note = _build_city_od(
                city_dir=city_dir,
                graph=graph,
                nodes=nodes,
                stops=stops,
                demand_source=str(args.demand_source),
                graph_source=graph_source,
            )
            row["demand_source"] = city_demand_source
            if demand_note:
                row["demand_note"] = demand_note

            per_city_samples = 0
            sample_skip_reasons: Counter[str] = Counter()
            for _ in range(int(args.samples_per_city) * 3):
                if per_city_samples >= int(args.samples_per_city) or len(data_list) >= int(args.max_samples):
                    break
                sample_nodes = _connected_sample_nodes(
                    graph,
                    int(args.target_nodes),
                    rng,
                    full_node_to_idx=node_to_idx,
                    full_classes=full_classes,
                    min_focus_nodes=min_focus_nodes,
                    max_attempts=int(args.sample_search_attempts),
                )
                if sample_nodes is None:
                    row["status"] = (
                        "skipped_no_focus_heavy_connected_sample"
                        if min_focus_nodes > 0
                        else "skipped_no_connected_sample"
                    )
                    break
                try:
                    data, sample_meta = _make_sample(
                        graph=graph,
                        sample_nodes=sample_nodes,
                        full_node_to_idx=node_to_idx,
                        full_od=full_od,
                        full_classes=full_classes,
                        rng=rng,
                    )
                    if min_focus_nodes > 0 and int(sample_meta["focus_class_count"]) < min_focus_nodes:
                        raise SkipSample("below_min_focus_class_share")
                except SkipSample as exc:
                    sample_skip_reasons.update([str(exc)])
                    continue
                data_list.append(data)
                sample_rows.append({"city": city_dir.name, **sample_meta})
                per_city_samples += 1
                demand_sources.update([sample_meta["demand_source"]])
                global_class_counts.update(sample_meta["class_counts"])
                if sample_meta["unknown_class_count"]:
                    global_class_counts.update({"unknown": sample_meta["unknown_class_count"]})

            row["samples"] = int(per_city_samples)
            if sample_skip_reasons:
                row["sample_skip_reasons"] = dict(sorted(sample_skip_reasons.items()))
            if per_city_samples == 0 and row["status"] == "ok":
                row["status"] = "skipped_no_samples"
        except SkipCity as exc:
            row["status"] = "skipped_no_gravity_od"
            row["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = str(exc)
        city_rows.append(row)
        _log(f"{city_dir.name}: {row['status']} samples={row['samples']}")

    if not data_list:
        raise RuntimeError("No samples were produced.")

    with raw_path.open("wb") as fh:
        pickle.dump(data_list, fh, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "dataset": "real_connectpt_morph",
        "output_dir": str(output_dir),
        "raw_graphs": str(raw_path),
        "processed_collated": str(output_dir / "processed" / "collated.pt"),
        "modality": str(args.modality),
        "graph_source_requested": str(args.graph_source),
        "target_nodes": int(args.target_nodes),
        "sample_count": int(len(data_list)),
        "source_city_count": int(len(city_rows)),
        "successful_city_count": int(sum(1 for row in city_rows if row["samples"])),
        "seed": int(args.seed),
        "min_focus_class_share": min_focus_class_share,
        "min_focus_class_nodes": min_focus_nodes,
        "sample_search_attempts": int(args.sample_search_attempts),
        "demand_source_requested": str(args.demand_source),
        "sample_demand_sources": dict(sorted(demand_sources.items())),
        "street_pattern_class_to_id": CLASS_TO_ID,
        "focus_class_name": FOCUS_CLASS_NAME,
        "focus_class_id": FOCUS_CLASS_ID,
        "sample_class_counts": dict(sorted(global_class_counts.items())),
        "cities": city_rows,
        "samples": sample_rows,
        "train_config_hint": {
            "config": "ppo_50nodes",
            "override": f"dataset.kwargs.path={output_dir}",
            "note": "Hydra entrypoint currently needs config_path='cfg' in inductive_route_learning.py before training.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.process:
        _log("processing pickle through CityGraphDataset -> processed/collated.pt")
        _ = CityGraphDataset(str(output_dir))

    _log(f"wrote {len(data_list)} samples -> {raw_path}")
    _log(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pandana as pdna
from shapely.geometry import LineString
from shapely.ops import linemerge


def _coerce_graph_crs(graph: nx.Graph) -> nx.Graph:
    crs = graph.graph.get("crs")
    if crs is not None and not isinstance(crs, int):
        if hasattr(crs, "to_epsg") and crs.to_epsg() is not None:
            graph.graph["crs"] = int(crs.to_epsg())
    return graph


def _pandana_shortest_path_lengths(net: pdna.Network, sources, targets, *, imp_name: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unsigned integer: shortest path distance is trying to be calculated.*",
            category=UserWarning,
        )
        return np.asarray(net.shortest_path_lengths(sources, targets, imp_name=imp_name), dtype=float)


def _repeat_node_id(value, count: int) -> np.ndarray:
    arr = np.empty(count, dtype=object)
    arr[:] = [value] * count
    return arr


def _sorted_nodes(graph: nx.Graph) -> list:
    try:
        return sorted(graph.nodes())
    except Exception:
        return sorted(graph.nodes(), key=lambda value: str(value))


def _node_xy(node_id, data: dict, *, node_x: str = "x", node_y: str = "y") -> tuple[float, float]:
    x = data.get(node_x)
    y = data.get(node_y)
    if x is None or y is None:
        if isinstance(node_id, tuple) and len(node_id) >= 2:
            x = node_id[0]
            y = node_id[1]
    if x is None or y is None:
        raise ValueError(f"Node {node_id!r} is missing coordinates.")
    return float(x), float(y)


def _graph_to_pandana_network(
    graph: nx.Graph,
    *,
    weight_key: str,
    node_x: str = "x",
    node_y: str = "y",
) -> tuple[pdna.Network, pd.DataFrame, dict, dict]:
    nodes = _sorted_nodes(graph)
    node_to_pdna = {node_id: idx for idx, node_id in enumerate(nodes)}
    pdna_to_node = {idx: node_id for node_id, idx in node_to_pdna.items()}
    node_rows = []
    for node_id in nodes:
        data = graph.nodes[node_id]
        x, y = _node_xy(node_id, data, node_x=node_x, node_y=node_y)
        node_rows.append({"node_id": node_to_pdna[node_id], "x": x, "y": y})
    nodes_df = pd.DataFrame(node_rows).set_index("node_id")

    edge_rows = []
    if graph.is_multigraph():
        for u, v, _key, data in graph.edges(keys=True, data=True):
            weight = pd.to_numeric(data.get(weight_key), errors="coerce")
            if pd.isna(weight) or not np.isfinite(float(weight)) or float(weight) < 0.0:
                continue
            edge_rows.append({"u": node_to_pdna[u], "v": node_to_pdna[v], weight_key: float(weight)})
    else:
        for u, v, data in graph.edges(data=True):
            weight = pd.to_numeric(data.get(weight_key), errors="coerce")
            if pd.isna(weight) or not np.isfinite(float(weight)) or float(weight) < 0.0:
                continue
            edge_rows.append({"u": node_to_pdna[u], "v": node_to_pdna[v], weight_key: float(weight)})
    if not edge_rows:
        raise ValueError(f"Graph has no valid edges with weight [{weight_key}].")
    edges_df = pd.DataFrame(edge_rows)

    net = pdna.Network(
        nodes_df["x"],
        nodes_df["y"],
        edges_df["u"],
        edges_df["v"],
        edges_df[[weight_key]],
        twoway=not nx.is_directed(graph),
    )
    return net, nodes_df, node_to_pdna, pdna_to_node


def _build_units_matrix(
    *,
    units_path: Path,
    graph_pickle_path: Path,
    output_path: Path,
    weight_key: str,
) -> dict:
    units = gpd.read_parquet(units_path)
    graph = _coerce_graph_crs(pd.read_pickle(graph_pickle_path))
    if units.empty:
        matrix = pd.DataFrame()
        matrix.to_parquet(output_path)
        return {"rows": 0, "cols": 0}

    work = units[["geometry"]].copy()
    if work.crs is None:
        work = work.set_crs(4326)
    graph_crs = graph.graph.get("crs")
    if graph_crs is None:
        raise ValueError("Graph CRS is missing for pandana units-matrix build.")
    work = work.to_crs(graph_crs)
    pts = work.geometry.representative_point()
    idx = list(units.index)

    net, _nodes_df, _node_to_pdna, _pdna_to_node = _graph_to_pandana_network(graph, weight_key=weight_key)
    unit_node_ids = np.asarray(
        net.get_node_ids(pts.x.to_numpy(dtype=float), pts.y.to_numpy(dtype=float)),
    )

    matrix = pd.DataFrame(np.inf, index=idx, columns=idx, dtype=float)
    for i in range(len(idx)):
        matrix.iat[i, i] = 0.0
    for i, src in enumerate(unit_node_ids):
        lengths = _pandana_shortest_path_lengths(
            net,
            _repeat_node_id(src, len(unit_node_ids)),
            unit_node_ids,
            imp_name=weight_key,
        )
        lengths[~np.isfinite(lengths)] = np.inf
        matrix.iloc[i, :] = lengths

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(output_path)
    return {"rows": int(matrix.shape[0]), "cols": int(matrix.shape[1])}


def _build_graph_node_matrix(
    *,
    graph_pickle_path: Path,
    output_path: Path,
    weight_key: str,
) -> dict:
    graph = _coerce_graph_crs(pd.read_pickle(graph_pickle_path))
    nodes = _sorted_nodes(graph)
    net, _nodes_df, node_to_pdna, _pdna_to_node = _graph_to_pandana_network(graph, weight_key=weight_key)
    node_ids = np.asarray([node_to_pdna[node] for node in nodes], dtype=int)

    matrix = pd.DataFrame(np.inf, index=nodes, columns=nodes, dtype=float)
    for i in range(len(nodes)):
        matrix.iat[i, i] = 0.0
    for i, src in enumerate(node_ids):
        lengths = _pandana_shortest_path_lengths(
            net,
            _repeat_node_id(src, len(node_ids)),
            node_ids,
            imp_name=weight_key,
        )
        lengths[~np.isfinite(lengths)] = np.inf
        matrix.iloc[i, :] = lengths

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(output_path)
    return {"rows": int(matrix.shape[0]), "cols": int(matrix.shape[1])}


def _build_pairs_shortest_paths(
    *,
    graph_pickle_path: Path,
    pairs_pickle_path: Path,
    output_pickle_path: Path,
    weight_key: str,
) -> dict:
    graph = _coerce_graph_crs(pd.read_pickle(graph_pickle_path))
    pairs = pd.read_pickle(pairs_pickle_path)
    if pairs.empty:
        pd.DataFrame(columns=["source", "target", "length", "path"]).to_pickle(output_pickle_path)
        return {"pairs": 0}

    net, _nodes_df, node_to_pdna, pdna_to_node = _graph_to_pandana_network(graph, weight_key=weight_key)
    sources = [node_to_pdna[source] for source in pairs["source"].tolist()]
    targets = [node_to_pdna[target] for target in pairs["target"].tolist()]
    lengths = _pandana_shortest_path_lengths(net, sources, targets, imp_name=weight_key)
    paths = net.shortest_paths(sources, targets, imp_name=weight_key)
    result = pairs.copy()
    result["length"] = lengths
    result["path"] = [[pdna_to_node[node] for node in path] for path in paths]
    output_pickle_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(output_pickle_path)
    return {"pairs": int(len(result))}


def _best_edge_data(graph: nx.Graph, u, v, *, weight_key: str) -> dict | None:
    edge_data = graph.get_edge_data(u, v)
    if edge_data is None:
        return None
    if graph.is_multigraph():
        best = None
        best_weight = math.inf
        for data in edge_data.values():
            weight = pd.to_numeric(data.get(weight_key), errors="coerce")
            if pd.isna(weight):
                continue
            weight_val = float(weight)
            if weight_val < best_weight:
                best_weight = weight_val
                best = data
        return best
    return edge_data


def _build_segment_geometry(graph: nx.Graph, u, v, *, weight_key: str, node_x: str, node_y: str):
    edge_data = _best_edge_data(graph, u, v, weight_key=weight_key) or {}
    geom = edge_data.get("geometry")
    if geom is not None:
        return geom
    ux, uy = _node_xy(u, graph.nodes[u], node_x=node_x, node_y=node_y)
    vx, vy = _node_xy(v, graph.nodes[v], node_x=node_x, node_y=node_y)
    return LineString([(ux, uy), (vx, vy)])


def _stop_complete_then_prune(
    *,
    graph_pickle_path: Path,
    output_graph_path: Path,
    stop_flag: str,
    weight_attr: str,
    node_x: str,
    node_y: str,
    min_weight: float | None,
    max_weight: float | None,
    speed_kmh: float | None,
) -> dict:
    graph = _coerce_graph_crs(pd.read_pickle(graph_pickle_path))

    def is_stop(node) -> bool:
        return bool(graph.nodes[node].get(stop_flag, False))

    stops = [node for node in _sorted_nodes(graph) if is_stop(node)]
    simplified = nx.Graph()
    if "crs" in graph.graph:
        simplified.graph["crs"] = graph.graph["crs"]
    for stop in stops:
        simplified.add_node(stop, **graph.nodes[stop])

    if not stops:
        output_graph_path.parent.mkdir(parents=True, exist_ok=True)
        with output_graph_path.open("wb") as handle:
            pickle.dump(nx.convert_node_labels_to_integers(simplified), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return {"stop_count": 0, "edge_count": 0}

    net, _nodes_df, node_to_pdna, pdna_to_node = _graph_to_pandana_network(
        graph,
        weight_key=weight_attr,
        node_x=node_x,
        node_y=node_y,
    )
    stop_pdna_ids = [node_to_pdna[stop] for stop in stops]

    for i, source in enumerate(stops):
        source_pdna = stop_pdna_ids[i]
        targets = stops[i + 1 :]
        target_pdna_ids = stop_pdna_ids[i + 1 :]
        if not targets:
            continue
        lengths = _pandana_shortest_path_lengths(
            net,
            _repeat_node_id(source_pdna, len(targets)),
            np.asarray(target_pdna_ids, dtype=int),
            imp_name=weight_attr,
        )
        valid_targets = []
        valid_lengths = []
        for target, distance in zip(targets, lengths):
            if not np.isfinite(distance):
                continue
            if min_weight is not None and float(distance) < float(min_weight):
                continue
            if max_weight is not None and float(distance) > float(max_weight):
                continue
            valid_targets.append(target)
            valid_lengths.append(float(distance))
        if not valid_targets:
            continue

        paths = net.shortest_paths(
            _repeat_node_id(source_pdna, len(valid_targets)),
            np.asarray([node_to_pdna[target] for target in valid_targets], dtype=int),
            imp_name=weight_attr,
        )
        for target, distance, path in zip(valid_targets, valid_lengths, paths):
            path_nodes = [pdna_to_node[node] for node in path]
            stop_count = sum(1 for node in path_nodes if is_stop(node))
            if stop_count != 2:
                continue

            geoms = []
            total_check = 0.0
            ok = True
            for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                edge_data = _best_edge_data(graph, u, v, weight_key=weight_attr)
                if edge_data is None:
                    ok = False
                    break
                weight_val = pd.to_numeric(edge_data.get(weight_attr), errors="coerce")
                if pd.isna(weight_val):
                    ok = False
                    break
                total_check += float(weight_val)
                geoms.append(_build_segment_geometry(graph, u, v, weight_key=weight_attr, node_x=node_x, node_y=node_y))
            if not ok:
                continue

            attrs = {"weight": total_check, "original_path": path_nodes}
            if speed_kmh:
                attrs["time_min"] = total_check * 60.0 / (1000.0 * float(speed_kmh))
            if geoms:
                try:
                    attrs["geometry"] = linemerge(geoms)
                except Exception:
                    pass
            simplified.add_edge(source, target, **attrs)

    simplified = nx.convert_node_labels_to_integers(simplified)
    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with output_graph_path.open("wb") as handle:
        pickle.dump(simplified, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {"stop_count": int(len(stops)), "edge_count": int(simplified.number_of_edges())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pandana backend tasks for path calculations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    units_matrix = subparsers.add_parser("units-matrix")
    units_matrix.add_argument("--units-path", required=True)
    units_matrix.add_argument("--graph-pickle-path", required=True)
    units_matrix.add_argument("--output-path", required=True)
    units_matrix.add_argument("--weight-key", default="time_min")

    graph_matrix = subparsers.add_parser("graph-node-matrix")
    graph_matrix.add_argument("--graph-pickle-path", required=True)
    graph_matrix.add_argument("--output-path", required=True)
    graph_matrix.add_argument("--weight-key", default="weight")

    pairs_paths = subparsers.add_parser("pairs-shortest-paths")
    pairs_paths.add_argument("--graph-pickle-path", required=True)
    pairs_paths.add_argument("--pairs-pickle-path", required=True)
    pairs_paths.add_argument("--output-pickle-path", required=True)
    pairs_paths.add_argument("--weight-key", default="weight")

    stop_graph = subparsers.add_parser("stop-complete-then-prune")
    stop_graph.add_argument("--graph-pickle-path", required=True)
    stop_graph.add_argument("--output-graph-path", required=True)
    stop_graph.add_argument("--stop-flag", default="is_stop")
    stop_graph.add_argument("--weight-attr", default="mm_len")
    stop_graph.add_argument("--node-x", default="x")
    stop_graph.add_argument("--node-y", default="y")
    stop_graph.add_argument("--min-weight", type=float, default=None)
    stop_graph.add_argument("--max-weight", type=float, default=None)
    stop_graph.add_argument("--speed-kmh", type=float, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "units-matrix":
        result = _build_units_matrix(
            units_path=Path(args.units_path).resolve(),
            graph_pickle_path=Path(args.graph_pickle_path).resolve(),
            output_path=Path(args.output_path).resolve(),
            weight_key=str(args.weight_key),
        )
    elif args.command == "graph-node-matrix":
        result = _build_graph_node_matrix(
            graph_pickle_path=Path(args.graph_pickle_path).resolve(),
            output_path=Path(args.output_path).resolve(),
            weight_key=str(args.weight_key),
        )
    elif args.command == "pairs-shortest-paths":
        result = _build_pairs_shortest_paths(
            graph_pickle_path=Path(args.graph_pickle_path).resolve(),
            pairs_pickle_path=Path(args.pairs_pickle_path).resolve(),
            output_pickle_path=Path(args.output_pickle_path).resolve(),
            weight_key=str(args.weight_key),
        )
    else:
        result = _stop_complete_then_prune(
            graph_pickle_path=Path(args.graph_pickle_path).resolve(),
            output_graph_path=Path(args.output_graph_path).resolve(),
            stop_flag=str(args.stop_flag),
            weight_attr=str(args.weight_attr),
            node_x=str(args.node_x),
            node_y=str(args.node_y),
            min_weight=args.min_weight,
            max_weight=args.max_weight,
            speed_kmh=args.speed_kmh,
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

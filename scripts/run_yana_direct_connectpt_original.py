from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONNECTPT_ROOT = ROOT / "connectpt"
if str(CONNECTPT_ROOT) not in sys.path:
    sys.path.insert(0, str(CONNECTPT_ROOT))

from aggregated_spatial_pipeline.connectpt_data_pipeline.run_route_generator_external import (  # noqa: E402
    _build_node_locs,
    _build_street_adj,
    _extract_metric,
    _load_external_od_matrix,
    _route_sequences,
    _unique_route_count,
)
from connectpt.routes_generator.citygraph_dataset import get_dataset_from_config  # noqa: E402
from connectpt.routes_generator.eval_route_generator import eval_model  # noqa: E402
from connectpt.routes_generator.torch_utils import dump_routes  # noqa: E402
from connectpt.routes_generator.utils import get_eval_cfg  # noqa: E402
import connectpt.routes_generator.utils as lrnu  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ConnectPT generator directly on an original graph + OD pair.")
    parser.add_argument("--dataset", default="tartu")
    parser.add_argument("--graph-path", type=Path, default=ROOT / "connectpt/examples/data/tartu/bus_graph_Tartu.pkl")
    parser.add_argument("--od-path", type=Path, default=ROOT / "connectpt/examples/data/tartu/OD_matrix_Tartu.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-routes", type=int, required=True)
    parser.add_argument("--min-route-len", type=int, required=True)
    parser.add_argument("--max-route-len", type=int, required=True)
    parser.add_argument("--demand-time-weight", type=float, default=0.3)
    parser.add_argument("--route-time-weight", type=float, default=0.3)
    parser.add_argument("--median-connectivity-weight", type=float, default=0.3)
    parser.add_argument("--street-pattern-weight", type=float, default=0.0)
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=ROOT / "connectpt/examples/data/model_weights/inductive_random_graphs_weighted_connectivity.pt",
    )
    return parser.parse_args()


def _load_graph(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if float(args.street_pattern_weight) > 0.0:
        raise ValueError(
            "direct_connectpt_original does not have city street-pattern cells. "
            "Use aggregated_spatial_pipeline.connectpt_data_pipeline.run_route_generator_external "
            "for street-pattern-aware city bundle experiments."
        )

    graph = _load_graph(args.graph_path)
    nodes = sorted(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    od_matrix = _load_external_od_matrix(args.od_path, nodes)
    street_adj = _build_street_adj(graph, nodes, node_to_idx)
    node_locs = _build_node_locs(graph, nodes)
    demand = torch.tensor(od_matrix.to_numpy(), dtype=torch.float32)

    params = {
        "dataset_name": "tensor",
        "n_routes": int(args.n_routes),
        "min_route_len": int(args.min_route_len),
        "max_route_len": int(args.max_route_len),
        "demand_time_weight": float(args.demand_time_weight),
        "route_time_weight": float(args.route_time_weight),
        "median_connectivity_weight": float(args.median_connectivity_weight),
        "street_pattern_weight": float(args.street_pattern_weight),
        "run_name": f"direct_original_{args.dataset}",
        "model_weights": str(args.weights_path.resolve()),
    }
    cfg_dir = str(ROOT / "connectpt/connectpt/routes_generator/cfg")
    cfg = get_eval_cfg(cfg_dir=cfg_dir, base_cfg_name="eval_model_mumford", params=params)
    tensors = {"street_adj": street_adj, "demand": demand, "node_locs": node_locs}
    test_ds = get_dataset_from_config(cfg.eval.dataset, tensors=tensors)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)
    device, run_name, _, cost_obj, model = lrnu.process_standard_experiment_cfg(
        cfg,
        run_name_prefix="lc_",
        weights_required=True,
    )
    _, _, metrics, routes = eval_model(
        model,
        test_dl,
        cfg.eval,
        cost_obj,
        n_samples=cfg.get("n_samples", 1),
        return_routes=True,
        silent=True,
        device=device,
    )
    dump_routes(run_name, routes.cpu(), out_dir=output_dir)
    od_out = output_dir / "bus_od_matrix.csv"
    od_matrix.to_csv(od_out)

    route_sequences = _route_sequences(routes)
    summary = {
        "runner": "direct_connectpt_original",
        "dataset": args.dataset,
        "graph_path": str(args.graph_path.resolve()),
        "od_path": str(args.od_path.resolve()),
        "weights_path": str(args.weights_path.resolve()),
        "objective_weights": {
            "demand_time_weight": float(args.demand_time_weight),
            "route_time_weight": float(args.route_time_weight),
            "median_connectivity_weight": float(args.median_connectivity_weight),
            "street_pattern_weight": float(args.street_pattern_weight),
        },
        "n_routes_requested": int(args.n_routes),
        "min_route_len": int(args.min_route_len),
        "max_route_len": int(args.max_route_len),
        "route_count": len(route_sequences),
        "unique_route_count": _unique_route_count(route_sequences),
        "route_lengths": [len(route) for route in route_sequences],
        "graph_node_count": int(graph.number_of_nodes()),
        "graph_edge_count": int(graph.number_of_edges()),
        "demand_sum": float(demand.sum().item()),
        "positive_od_pairs": int((demand > 0).sum().item()),
        "cost": _extract_metric(metrics, "cost"),
        "att": _extract_metric(metrics, "ATT"),
        "unserved_demand_pct": _extract_metric(metrics, "$d_{un}$"),
        "median_connectivity": _extract_metric(metrics, "median_connectivity"),
        "street_pattern_class_count": _extract_metric(metrics, "street_pattern_class_count"),
        "street_pattern_penalty_value": _extract_metric(metrics, "street_pattern_penalty"),
        "routes_shape": list(routes.shape),
        "routes_tensor": routes.cpu().tolist(),
        "files": {
            "od_matrix": str(od_out),
            "routes_pickle": str(output_dir / f"{run_name}_routes.pkl"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("route_count", "unique_route_count", "route_lengths", "demand_sum", "unserved_demand_pct")}, indent=2))


if __name__ == "__main__":
    main()

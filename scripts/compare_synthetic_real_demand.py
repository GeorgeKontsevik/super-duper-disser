#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
CONNECTPT_ROOT = ROOT / "connectpt"
for candidate in (ROOT, CONNECTPT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import connectpt.routes_generator.citygraph_dataset as cgd  # noqa: E402
from connectpt.routes_generator.citygraph_dataset import (  # noqa: E402
    MIXED,
    RAW_GRAPH_FILENAME,
    STOP_KEY,
    DynamicCityGraphDataset,
)


DEFAULT_REAL_DATASET = ROOT / "connectpt/datasets/real_morph_10cities_bus50_heavy"
CANVAS = "#f7f3ed"
INK = "#1f2937"
GRID = "#ddd6c9"
COLORS = {
    "synthetic_original": "#2563eb",
    "real_sampled_gravity": "#f97316",
}
LABELS = {
    "synthetic_original": "Original synthetic",
    "real_sampled_gravity": "Sampled real-network gravity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original ConnectPT synthetic demand with sampled real-network gravity demand."
    )
    parser.add_argument("--real-dataset-dir", type=Path, default=DEFAULT_REAL_DATASET)
    parser.add_argument("--synthetic-pickle", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-synthetic", type=int, default=None)
    parser.add_argument("--synthetic-nodes", type=int, default=50)
    parser.add_argument("--edge-keep-prob", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260429)
    return parser.parse_args()


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": CANVAS,
            "savefig.facecolor": CANVAS,
            "axes.facecolor": "#fffdf9",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
        }
    )


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_real_graphs(dataset_dir: Path) -> tuple[list, dict]:
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = Path(manifest.get("raw_graphs") or dataset_dir / RAW_GRAPH_FILENAME)
    if not raw_path.is_absolute():
        raw_path = dataset_dir / raw_path
    with raw_path.open("rb") as fh:
        graphs = pickle.load(fh)
    return graphs, manifest


def _install_knn_fallback_if_needed() -> bool:
    try:
        import torch_cluster  # noqa: F401

        return False
    except ImportError:
        pass

    def _fallback_build_knn_graph(
        n_nodes: int,
        knn: int,
        edge_keep_prob: float = 1.0,
        flow: str = "target_to_source",
        directed: bool = True,
    ):
        while True:
            locs = torch.rand((n_nodes, 2)) * 2 - 1
            distances = torch.cdist(locs, locs)
            distances.fill_diagonal_(float("inf"))
            neighbors = distances.topk(knn, largest=False).indices
            targets = torch.arange(n_nodes).repeat_interleave(knn)
            sources = neighbors.reshape(-1)
            if flow == "source_to_target":
                edge_index = torch.stack((targets, sources), dim=0)
            else:
                edge_index = torch.stack((sources, targets), dim=0)
            if not directed:
                edge_index = cgd.pygu.to_undirected(edge_index, num_nodes=n_nodes)
            edge_index, _ = cgd.pygu.coalesce(edge_index, None, n_nodes)
            street_graph = cgd.Data(pos=locs, edge_index=edge_index)
            cgd.drop_edges(street_graph, edge_keep_prob, directed)
            nx_graph = cgd.pygu.to_networkx(street_graph, to_undirected=not directed)
            if cgd.is_strongly_connected(nx_graph):
                return street_graph

    cgd.build_knn_graph = _fallback_build_knn_graph
    return True


def _load_or_generate_synthetic(args: argparse.Namespace, count: int) -> tuple[list, dict]:
    if args.synthetic_pickle is not None:
        with args.synthetic_pickle.open("rb") as fh:
            graphs = pickle.load(fh)
        return graphs[:count], {
            "source": str(args.synthetic_pickle),
            "mode": "pickle",
            "count": min(count, len(graphs)),
        }

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    used_knn_fallback = _install_knn_fallback_if_needed()
    dataset = DynamicCityGraphDataset(
        min_nodes=int(args.synthetic_nodes),
        max_nodes=int(args.synthetic_nodes),
        edge_keep_prob=float(args.edge_keep_prob),
        data_type=MIXED,
        directed=False,
        fully_connected_demand=True,
        mumford_style=True,
    )
    graphs = [dataset.generate_graph(n_nodes=int(args.synthetic_nodes)) for _ in range(count)]
    return graphs, {
        "source": "DynamicCityGraphDataset.generate_graph",
        "mode": "generated_reference",
        "count": int(count),
        "graph_type": MIXED,
        "nodes": int(args.synthetic_nodes),
        "edge_keep_prob": float(args.edge_keep_prob),
        "mumford_style": True,
        "seed": int(args.seed),
        "used_local_knn_fallback": bool(used_knn_fallback),
        "note": "Reference generated from ConnectPT original synthetic generator because datasets/mixed_50 is not present locally.",
    }


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    values = values - values.min()
    total = values.sum()
    if total <= 0:
        return 0.0
    values = np.sort(values)
    n = values.size
    index = np.arange(1, n + 1, dtype=float)
    return float(((2 * index - n - 1) * values).sum() / (n * total))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return 0.0
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cutoff = float(quantile) * float(cdf[-1])
    return float(values[np.searchsorted(cdf, cutoff, side="left")])


def _upper_edge_count(street_adj: np.ndarray) -> int:
    finite = np.isfinite(street_adj) & (street_adj > 0)
    finite &= ~np.eye(street_adj.shape[0], dtype=bool)
    return int(np.triu(finite, k=1).sum())


def _graph_arrays(graph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demand = graph.demand.detach().cpu().numpy().astype(float)
    drive_times = graph.drive_times.detach().cpu().numpy().astype(float)
    street_adj = graph.street_adj.detach().cpu().numpy().astype(float)
    n_nodes = int(graph[STOP_KEY].pos.shape[0])
    demand = demand[:n_nodes, :n_nodes]
    drive_times = drive_times[:n_nodes, :n_nodes]
    street_adj = street_adj[:n_nodes, :n_nodes]
    return demand, drive_times, street_adj


def _distance_percentile(distances: np.ndarray) -> np.ndarray:
    order = np.argsort(distances, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = (np.arange(len(distances), dtype=float) + 0.5) / max(len(distances), 1)
    return ranks


def _metrics_for_graphs(graphs: list, dataset_name: str, *, max_pair_rows: int, rng: random.Random):
    sample_rows: list[dict] = []
    bin_rows: list[dict] = []
    pair_rows: list[dict] = []
    pair_reservoir: list[dict] = []

    for sample_idx, graph in enumerate(graphs):
        demand, drive_times, street_adj = _graph_arrays(graph)
        n_nodes = demand.shape[0]
        offdiag = ~np.eye(n_nodes, dtype=bool)
        finite = np.isfinite(drive_times) & offdiag
        positive = finite & (demand > 0)
        positive_values = demand[positive]
        finite_demands = demand[finite]
        finite_distances = drive_times[finite]
        total_demand = float(positive_values.sum())

        activity = demand.sum(axis=0) + demand.sum(axis=1)
        activity_sorted = np.sort(activity)[::-1]
        top5_share = float(activity_sorted[:5].sum() / activity_sorted.sum()) if activity_sorted.sum() else 0.0

        dist_percentiles = _distance_percentile(finite_distances)
        pos_dist_percentiles = dist_percentiles[finite_demands > 0]
        pos_distances = finite_distances[finite_demands > 0]
        pos_demands = finite_demands[finite_demands > 0]
        demand_share_by_bin = np.zeros(10, dtype=float)
        pair_share_by_bin = np.zeros(10, dtype=float)
        if finite_demands.size:
            bins = np.clip((dist_percentiles * 10).astype(int), 0, 9)
            for bin_idx in range(10):
                in_bin = bins == bin_idx
                pair_share_by_bin[bin_idx] = float(in_bin.mean())
                if total_demand > 0:
                    demand_share_by_bin[bin_idx] = float(finite_demands[in_bin].sum() / total_demand)

        for bin_idx in range(10):
            bin_rows.append(
                {
                    "dataset": dataset_name,
                    "sample_idx": sample_idx,
                    "network_distance_decile": bin_idx + 1,
                    "pair_share": pair_share_by_bin[bin_idx],
                    "demand_share": demand_share_by_bin[bin_idx],
                    "demand_lift_vs_pairs": (
                        demand_share_by_bin[bin_idx] / pair_share_by_bin[bin_idx]
                        if pair_share_by_bin[bin_idx] > 0
                        else 0.0
                    ),
                }
            )

        sample_rows.append(
            {
                "dataset": dataset_name,
                "sample_idx": sample_idx,
                "node_count": n_nodes,
                "edge_count": _upper_edge_count(street_adj),
                "total_demand": total_demand,
                "positive_od_share": float(positive.sum() / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0,
                "positive_demand_p50": float(np.median(positive_values)) if positive_values.size else 0.0,
                "positive_demand_p90": float(np.quantile(positive_values, 0.9)) if positive_values.size else 0.0,
                "positive_demand_max": float(positive_values.max()) if positive_values.size else 0.0,
                "positive_demand_norm_p90": (
                    float(np.quantile(positive_values / positive_values.sum(), 0.9))
                    if positive_values.sum() > 0
                    else 0.0
                ),
                "od_demand_gini": _gini(positive_values),
                "activity_gini": _gini(activity),
                "top5_activity_share": top5_share,
                "demand_weighted_network_time": (
                    float((pos_demands * pos_distances).sum() / pos_demands.sum()) if pos_demands.sum() > 0 else 0.0
                ),
                "demand_weighted_network_percentile": (
                    float((pos_demands * pos_dist_percentiles).sum() / pos_demands.sum())
                    if pos_demands.sum() > 0
                    else 0.0
                ),
                "demand_network_percentile_p50": _weighted_quantile(pos_dist_percentiles, pos_demands, 0.5),
                "demand_network_percentile_p90": _weighted_quantile(pos_dist_percentiles, pos_demands, 0.9),
            }
        )

        if max_pair_rows > 0 and pos_demands.size:
            norm_demands = pos_demands / pos_demands.sum() if pos_demands.sum() else pos_demands
            for demand_value, norm_demand, dist_pct in zip(pos_demands, norm_demands, pos_dist_percentiles):
                item = {
                    "dataset": dataset_name,
                    "sample_idx": sample_idx,
                    "demand": float(demand_value),
                    "demand_share": float(norm_demand),
                    "network_distance_percentile": float(dist_pct),
                }
                if len(pair_reservoir) < max_pair_rows:
                    pair_reservoir.append(item)
                else:
                    j = rng.randrange(sample_idx * max(1, pos_demands.size) + 1)
                    if j < max_pair_rows:
                        pair_reservoir[j] = item

    pair_rows.extend(pair_reservoir)
    return pd.DataFrame(sample_rows), pd.DataFrame(bin_rows), pd.DataFrame(pair_rows)


def _boxplot(ax, df: pd.DataFrame, column: str, title: str, ylabel: str, *, log: bool = False) -> None:
    data = [
        df[df["dataset"] == "synthetic_original"][column].dropna().to_numpy(),
        df[df["dataset"] == "real_sampled_gravity"][column].dropna().to_numpy(),
    ]
    box = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=[LABELS["synthetic_original"], LABELS["real_sampled_gravity"]],
    )
    for patch, color in zip(box["boxes"], [COLORS["synthetic_original"], COLORS["real_sampled_gravity"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.74)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=8)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, alpha=0.55)


def _plot_sample_metrics(sample_metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    _boxplot(axes[0, 0], sample_metrics, "total_demand", "Total Demand", "sum", log=True)
    _boxplot(axes[0, 1], sample_metrics, "positive_demand_p90", "P90 Positive OD Demand", "raw demand", log=True)
    _boxplot(axes[0, 2], sample_metrics, "positive_od_share", "Positive OD Pair Share", "share")
    _boxplot(axes[1, 0], sample_metrics, "od_demand_gini", "OD Pair Demand Gini", "gini")
    _boxplot(axes[1, 1], sample_metrics, "activity_gini", "Node Activity Gini", "gini")
    _boxplot(axes[1, 2], sample_metrics, "top5_activity_share", "Top-5 Node Activity Share", "share")
    _finish(fig, output_dir / "01_sample_demand_metrics.png")


def _plot_network_distance(bin_metrics: pd.DataFrame, sample_metrics: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        bin_metrics.groupby(["dataset", "network_distance_decile"], as_index=False)
        .agg(
            demand_share_mean=("demand_share", "mean"),
            demand_share_p10=("demand_share", lambda x: float(np.quantile(x, 0.1))),
            demand_share_p90=("demand_share", lambda x: float(np.quantile(x, 0.9))),
            lift_mean=("demand_lift_vs_pairs", "mean"),
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for dataset_name in ["synthetic_original", "real_sampled_gravity"]:
        part = grouped[grouped["dataset"] == dataset_name]
        x = part["network_distance_decile"].to_numpy(dtype=float)
        y = part["demand_share_mean"].to_numpy(dtype=float)
        axes[0].plot(x, y, marker="o", color=COLORS[dataset_name], label=LABELS[dataset_name])
        axes[0].fill_between(
            x,
            part["demand_share_p10"].to_numpy(dtype=float),
            part["demand_share_p90"].to_numpy(dtype=float),
            color=COLORS[dataset_name],
            alpha=0.16,
        )
        axes[1].plot(x, part["lift_mean"].to_numpy(dtype=float), marker="o", color=COLORS[dataset_name], label=LABELS[dataset_name])

    axes[0].axhline(0.1, color="#64748b", linestyle="--", linewidth=1.0)
    axes[0].set_title("Demand Share By Network-Distance Decile")
    axes[0].set_xlabel("shortest-path distance decile, per graph")
    axes[0].set_ylabel("share of total demand")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(color=GRID, alpha=0.55)
    axes[0].legend(frameon=True)

    axes[1].axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    axes[1].set_title("Demand Lift Vs Pair Count")
    axes[1].set_xlabel("shortest-path distance decile, per graph")
    axes[1].set_ylabel("demand share / pair share")
    axes[1].grid(color=GRID, alpha=0.55)

    _boxplot(
        axes[2],
        sample_metrics,
        "demand_weighted_network_percentile",
        "Demand-Weighted Network Percentile",
        "0=short OD, 1=long OD",
    )
    _finish(fig, output_dir / "02_network_distance_demand.png")


def _plot_pair_scatter(pair_metrics: pd.DataFrame, output_dir: Path) -> None:
    if pair_metrics.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for dataset_name in ["synthetic_original", "real_sampled_gravity"]:
        part = pair_metrics[pair_metrics["dataset"] == dataset_name]
        axes[0].scatter(
            part["network_distance_percentile"],
            part["demand"],
            s=4,
            alpha=0.18,
            color=COLORS[dataset_name],
            label=LABELS[dataset_name],
            rasterized=True,
        )
        axes[1].scatter(
            part["network_distance_percentile"],
            part["demand_share"],
            s=4,
            alpha=0.18,
            color=COLORS[dataset_name],
            label=LABELS[dataset_name],
            rasterized=True,
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Raw OD Demand Vs Network Distance")
    axes[0].set_xlabel("shortest-path distance percentile")
    axes[0].set_ylabel("raw demand, log scale")
    axes[1].set_yscale("log")
    axes[1].set_title("Within-Sample Demand Share Vs Network Distance")
    axes[1].set_xlabel("shortest-path distance percentile")
    axes[1].set_ylabel("OD share of sample demand, log scale")
    for ax in axes:
        ax.grid(color=GRID, alpha=0.55)
        ax.legend(frameon=True)
    _finish(fig, output_dir / "03_pair_demand_vs_network_distance.png")


def _png_checks(output_dir: Path) -> dict[str, dict[str, object]]:
    checks: dict[str, dict[str, object]] = {}
    for path in sorted(output_dir.glob("*.png")):
        image = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(image)
        checks[path.name] = {
            "size": list(image.size),
            "channel_var": [round(float(value), 3) for value in stat.var],
            "nonblank": bool(max(stat.var) > 1.0),
        }
    return checks


def main() -> None:
    args = parse_args()
    _setup_style()

    real_dataset_dir = args.real_dataset_dir.resolve()
    output_dir = (args.output_dir or real_dataset_dir / "analysis" / "demand_synthetic_vs_real").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_graphs, real_manifest = _load_real_graphs(real_dataset_dir)
    count = int(args.n_synthetic or len(real_graphs))
    synthetic_graphs, synthetic_meta = _load_or_generate_synthetic(args, count)
    real_graphs = real_graphs[: len(synthetic_graphs)]

    rng = random.Random(int(args.seed) + 17)
    syn_samples, syn_bins, syn_pairs = _metrics_for_graphs(
        synthetic_graphs,
        "synthetic_original",
        max_pair_rows=35000,
        rng=rng,
    )
    real_samples, real_bins, real_pairs = _metrics_for_graphs(
        real_graphs,
        "real_sampled_gravity",
        max_pair_rows=35000,
        rng=rng,
    )

    sample_metrics = pd.concat([syn_samples, real_samples], ignore_index=True)
    bin_metrics = pd.concat([syn_bins, real_bins], ignore_index=True)
    pair_metrics = pd.concat([syn_pairs, real_pairs], ignore_index=True)

    sample_metrics.to_csv(output_dir / "sample_demand_metrics.csv", index=False)
    bin_metrics.to_csv(output_dir / "network_distance_decile_metrics.csv", index=False)
    pair_metrics.to_csv(output_dir / "od_pair_sample_metrics.csv", index=False)

    _plot_sample_metrics(sample_metrics, output_dir)
    _plot_network_distance(bin_metrics, sample_metrics, output_dir)
    _plot_pair_scatter(pair_metrics, output_dir)

    summary_by_dataset = (
        sample_metrics.groupby("dataset")
        .agg(
            samples=("sample_idx", "count"),
            total_demand_median=("total_demand", "median"),
            total_demand_p05=("total_demand", lambda x: float(np.quantile(x, 0.05))),
            total_demand_p95=("total_demand", lambda x: float(np.quantile(x, 0.95))),
            positive_od_share_median=("positive_od_share", "median"),
            od_demand_gini_median=("od_demand_gini", "median"),
            activity_gini_median=("activity_gini", "median"),
            top5_activity_share_median=("top5_activity_share", "median"),
            demand_weighted_network_percentile_median=("demand_weighted_network_percentile", "median"),
            demand_network_percentile_p90_median=("demand_network_percentile_p90", "median"),
        )
        .reset_index()
    )
    summary_by_dataset.to_csv(output_dir / "summary_by_dataset.csv", index=False)

    decile_summary = (
        bin_metrics.groupby(["dataset", "network_distance_decile"], as_index=False)
        .agg(demand_share_mean=("demand_share", "mean"), demand_lift_mean=("demand_lift_vs_pairs", "mean"))
    )
    decile_summary.to_csv(output_dir / "summary_network_distance_deciles.csv", index=False)

    summary = {
        "real_dataset_dir": str(real_dataset_dir),
        "output_dir": str(output_dir),
        "real_sample_count": int(len(real_graphs)),
        "real_manifest_sample_demand_sources": real_manifest.get("sample_demand_sources"),
        "synthetic_reference": synthetic_meta,
        "summary_by_dataset": summary_by_dataset.to_dict(orient="records"),
        "png_checks": _png_checks(output_dir),
        "files": {
            "sample_demand_metrics": str(output_dir / "sample_demand_metrics.csv"),
            "network_distance_decile_metrics": str(output_dir / "network_distance_decile_metrics.csv"),
            "summary_by_dataset": str(output_dir / "summary_by_dataset.csv"),
            "summary_network_distance_deciles": str(output_dir / "summary_network_distance_deciles.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

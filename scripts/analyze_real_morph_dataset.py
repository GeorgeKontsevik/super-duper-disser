#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
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

from connectpt.routes_generator.citygraph_dataset import STOP_KEY  # noqa: E402


DEFAULT_DATASET_DIR = ROOT / "connectpt/datasets/real_morph_10cities_bus50_heavy"

CLASS_COLORS = {
    "Irregular Grid": "#a86700",
    "Loops & Lollipops": "#0f7f73",
    "Regular Grid": "#16a34a",
    "Warped Parallel": "#f97316",
    "Sparse": "#7c3aed",
    "Broken Grid": "#dc2626",
    "unknown": "#cbd5e1",
}
FOCUS_CLASS = "Loops & Lollipops"
CANVAS = "#f7f3ed"
INK = "#1f2937"
GRID = "#ddd6c9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a real-morph ConnectPT training dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_graphs(dataset_dir: Path):
    manifest = _load_manifest(dataset_dir)
    path = Path(manifest.get("raw_graphs") or dataset_dir / "raw_graphs_1000.pkl")
    if not path.is_absolute():
        path = dataset_dir / path
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as fh:
        return pickle.load(fh), path


def _class_names(manifest: dict) -> list[str]:
    mapping = manifest["street_pattern_class_to_id"]
    return [name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]


def _short_city_label(city: str) -> str:
    special = {
        "aix_en_provence_provence_alpes_c_te_d_azur_france": "Aix-en-Provence",
        "pristina_prishtin_kosovo": "Pristina",
        "skopje_skopje_north_macedonia": "Skopje",
    }
    if city in special:
        return special[city]
    return city.split("_")[0].replace("-", " ").title()


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


def _finite_upper_edges(street_adj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(street_adj) & (street_adj > 0)
    finite &= ~np.eye(street_adj.shape[0], dtype=bool)
    upper = np.triu(finite, k=1)
    u, v = np.where(upper)
    return u, v, street_adj[u, v]


def _euclidean_distance_matrix(pos: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)


def _node_activity(demand: np.ndarray) -> np.ndarray:
    return demand.sum(axis=0) + demand.sum(axis=1)


def _metric_rows(raw_graphs: list, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_names = _class_names(manifest)
    id_to_class = {idx: name for idx, name in enumerate(class_names)}
    sample_meta = manifest.get("samples", [])
    rows: list[dict] = []
    class_rows: list[dict] = []
    demand_class_rows: list[dict] = []

    for idx, data in enumerate(raw_graphs):
        meta = sample_meta[idx] if idx < len(sample_meta) else {}
        city = str(meta.get("city", f"sample_{idx}"))
        demand_source = str(meta.get("demand_source", "unknown"))

        street_adj = data.street_adj.detach().cpu().numpy()
        demand = data.demand.detach().cpu().numpy()
        pos = data[STOP_KEY].pos.detach().cpu().numpy()
        class_ids = data.street_pattern_classes.detach().cpu().numpy().astype(int)
        n_nodes = int(demand.shape[0])

        u, v, edge_values = _finite_upper_edges(street_adj)
        degrees = np.bincount(np.concatenate([u, v]) if u.size else np.array([], dtype=int), minlength=n_nodes)
        density = float((2 * len(u)) / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
        edge_p50 = float(np.median(edge_values)) if edge_values.size else 0.0
        edge_p90 = float(np.quantile(edge_values, 0.9)) if edge_values.size else 0.0

        positive_demand = demand[demand > 0]
        demand_sum = float(demand.sum())
        positive_pairs = int(positive_demand.size)
        positive_share = float(positive_pairs / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0
        demand_p50 = float(np.median(positive_demand)) if positive_demand.size else 0.0
        demand_p90 = float(np.quantile(positive_demand, 0.9)) if positive_demand.size else 0.0
        demand_max = float(positive_demand.max()) if positive_demand.size else 0.0
        activity = _node_activity(demand)
        activity_sorted = np.sort(activity)[::-1]
        top5_activity_share = float(activity_sorted[:5].sum() / activity_sorted.sum()) if activity_sorted.sum() else 0.0

        distances = _euclidean_distance_matrix(pos)
        distance_positive = distances[demand > 0]
        demand_weighted_distance = (
            float((distances * demand).sum() / demand_sum) if demand_sum > 0 else 0.0
        )
        median_positive_od_distance = float(np.median(distance_positive)) if distance_positive.size else 0.0

        known_class_mask = class_ids >= 0
        focus_nodes = class_ids == manifest.get("focus_class_id", 0)
        unknown_nodes = class_ids < 0

        row = {
            "sample_idx": idx,
            "city": city,
            "demand_source": demand_source,
            "node_count": n_nodes,
            "edge_count": int(len(u)),
            "density": density,
            "degree_mean": float(degrees.mean()) if degrees.size else 0.0,
            "degree_p90": float(np.quantile(degrees, 0.9)) if degrees.size else 0.0,
            "degree_max": float(degrees.max()) if degrees.size else 0.0,
            "edge_value_p50": edge_p50,
            "edge_value_p90": edge_p90,
            "demand_sum": demand_sum,
            "positive_od_pairs": positive_pairs,
            "positive_od_share": positive_share,
            "demand_p50": demand_p50,
            "demand_p90": demand_p90,
            "demand_max": demand_max,
            "activity_gini": _gini(activity),
            "top5_activity_share": top5_activity_share,
            "demand_weighted_distance": demand_weighted_distance,
            "median_positive_od_distance": median_positive_od_distance,
            "known_class_share": float(known_class_mask.mean()) if n_nodes else 0.0,
            "unknown_class_share": float(unknown_nodes.mean()) if n_nodes else 0.0,
            "loops_node_share": float(focus_nodes.mean()) if n_nodes else 0.0,
        }
        rows.append(row)

        for class_id, class_name in id_to_class.items():
            mask = class_ids == class_id
            node_share = float(mask.mean()) if n_nodes else 0.0
            origin_demand = float(demand[mask, :].sum()) if mask.any() else 0.0
            dest_demand = float(demand[:, mask].sum()) if mask.any() else 0.0
            touch_demand = float(demand[mask, :].sum() + demand[:, mask].sum()) if mask.any() else 0.0
            class_rows.append(
                {
                    "sample_idx": idx,
                    "city": city,
                    "demand_source": demand_source,
                    "class": class_name,
                    "node_count": int(mask.sum()),
                    "node_share": node_share,
                }
            )
            demand_class_rows.append(
                {
                    "sample_idx": idx,
                    "city": city,
                    "demand_source": demand_source,
                    "class": class_name,
                    "node_share": node_share,
                    "origin_demand_share": origin_demand / demand_sum if demand_sum else 0.0,
                    "dest_demand_share": dest_demand / demand_sum if demand_sum else 0.0,
                    "touch_demand_share": touch_demand / (2 * demand_sum) if demand_sum else 0.0,
                }
            )
        if unknown_nodes.any():
            class_rows.append(
                {
                    "sample_idx": idx,
                    "city": city,
                    "demand_source": demand_source,
                    "class": "unknown",
                    "node_count": int(unknown_nodes.sum()),
                    "node_share": float(unknown_nodes.mean()),
                }
            )
            origin_demand = float(demand[unknown_nodes, :].sum())
            dest_demand = float(demand[:, unknown_nodes].sum())
            demand_class_rows.append(
                {
                    "sample_idx": idx,
                    "city": city,
                    "demand_source": demand_source,
                    "class": "unknown",
                    "node_share": float(unknown_nodes.mean()),
                    "origin_demand_share": origin_demand / demand_sum if demand_sum else 0.0,
                    "dest_demand_share": dest_demand / demand_sum if demand_sum else 0.0,
                    "touch_demand_share": (origin_demand + dest_demand) / (2 * demand_sum) if demand_sum else 0.0,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(class_rows), pd.DataFrame(demand_class_rows)


def _city_metrics(sample_metrics: pd.DataFrame, class_metrics: pd.DataFrame, demand_class_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "edge_count",
        "density",
        "degree_mean",
        "demand_sum",
        "positive_od_share",
        "activity_gini",
        "top5_activity_share",
        "demand_weighted_distance",
        "loops_node_share",
        "unknown_class_share",
    ]
    city = sample_metrics.groupby("city", as_index=False).agg(
        samples=("sample_idx", "count"),
        demand_source=("demand_source", lambda x: ",".join(sorted(set(map(str, x))))),
        **{f"{col}_mean": (col, "mean") for col in numeric_cols},
    )
    loop_demand = (
        demand_class_metrics[demand_class_metrics["class"] == FOCUS_CLASS]
        .groupby("city", as_index=False)
        .agg(loops_touch_demand_share_mean=("touch_demand_share", "mean"))
    )
    city = city.merge(loop_demand, on="city", how="left")
    return city


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
            "axes.titlesize": 13,
        }
    )


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _hist(ax, values: pd.Series, title: str, xlabel: str, *, bins: int = 30, color: str = "#0f766e") -> None:
    ax.hist(values.dropna().to_numpy(), bins=bins, color=color, alpha=0.86, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("samples")
    ax.grid(axis="y", color=GRID, alpha=0.55)


def _plot_city_mix(sample_metrics: pd.DataFrame, city_metrics: pd.DataFrame, output_dir: Path) -> None:
    city_order = city_metrics.sort_values("loops_node_share_mean", ascending=False)["city"].tolist()
    city = city_metrics.set_index("city").reindex(city_order)
    labels = [_short_city_label(name) for name in city_order]
    y = np.arange(len(city_order))
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1.25, 1.25, 1.0]})

    axes[0].barh(y - 0.18, city["loops_node_share_mean"], height=0.34, color=CLASS_COLORS[FOCUS_CLASS], label="node share")
    axes[0].barh(y + 0.18, city["loops_touch_demand_share_mean"], height=0.34, color="#f97316", label="demand touch")
    axes[0].set_title("Loops & Lollipops By City")
    axes[0].set_xlabel("share")
    axes[0].set_xlim(0, max(0.55, float(city[["loops_node_share_mean", "loops_touch_demand_share_mean"]].max().max()) * 1.12))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", color=GRID, alpha=0.55)
    axes[0].legend(frameon=True)

    axes[1].barh(y - 0.18, city["density_mean"], height=0.34, color="#2563eb", label="graph density")
    axes[1].barh(y + 0.18, city["activity_gini_mean"], height=0.34, color="#7c3aed", label="activity gini")
    axes[1].set_title("Network And Demand Shape")
    axes[1].set_xlabel("mean value")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", color=GRID, alpha=0.55)
    axes[1].legend(frameon=True)

    axes[2].barh(y, city["demand_sum_mean"], color="#a16207")
    axes[2].set_title("Mean Total Demand")
    axes[2].set_xlabel("sum, log scale")
    axes[2].set_xscale("log")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", color=GRID, alpha=0.55)
    _finish(fig, output_dir / "01_city_mix_and_structure.png")


def _plot_network(sample_metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _hist(axes[0, 0], sample_metrics["edge_count"], "Street Edges Per 50-Node Sample", "undirected edges", color="#0f766e")
    _hist(axes[0, 1], sample_metrics["density"], "Graph Density", "2E / N(N-1)", color="#2563eb")
    _hist(axes[0, 2], sample_metrics["degree_mean"], "Mean Degree", "degree", color="#16a34a")
    _hist(axes[1, 0], sample_metrics["degree_p90"], "P90 Node Degree", "degree", color="#a16207")
    _hist(axes[1, 1], sample_metrics["edge_value_p50"], "Median Edge Value", "edge value", color="#7c3aed")
    _hist(axes[1, 2], sample_metrics["edge_value_p90"], "P90 Edge Value", "edge value", color="#dc2626")
    _finish(fig, output_dir / "02_network_structure_distributions.png")


def _plot_demand(sample_metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _hist(axes[0, 0], sample_metrics["demand_sum"], "Total Demand", "sum", color="#2563eb")
    _hist(axes[0, 1], sample_metrics["positive_od_share"], "Positive OD Pair Share", "share", color="#0f766e")
    _hist(axes[0, 2], sample_metrics["demand_p90"], "P90 Positive OD Demand", "demand", color="#f97316")
    _hist(axes[1, 0], sample_metrics["activity_gini"], "Node Activity Gini", "gini", color="#7c3aed")
    _hist(axes[1, 1], sample_metrics["top5_activity_share"], "Top-5 Node Activity Share", "share", color="#a16207")
    _hist(axes[1, 2], sample_metrics["demand_weighted_distance"], "Demand-Weighted Euclidean Distance", "distance", color="#dc2626")
    _finish(fig, output_dir / "03_demand_distributions.png")


def _plot_class_shares(class_metrics: pd.DataFrame, output_dir: Path) -> None:
    classes = [c for c in CLASS_COLORS if c in set(class_metrics["class"])]
    fig, ax = plt.subplots(figsize=(14, 7))
    data = [class_metrics[class_metrics["class"] == klass]["node_share"].to_numpy() for klass in classes]
    positions = np.arange(len(classes))
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True, widths=0.85)
    for body, klass in zip(parts["bodies"], classes):
        body.set_facecolor(CLASS_COLORS.get(klass, "#64748b"))
        body.set_edgecolor("white")
        body.set_alpha(0.82)
    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color(INK)
            parts[key].set_linewidth(1.0)
    ax.set_title("Street-Pattern Node Share Distribution Across Samples")
    ax.set_ylabel("share of sample nodes")
    ax.set_xticks(positions)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.grid(axis="y", color=GRID, alpha=0.55)
    _finish(fig, output_dir / "04_street_pattern_class_share_distributions.png")


def _plot_city_class_heatmap(class_metrics: pd.DataFrame, output_dir: Path) -> None:
    pivot = (
        class_metrics.groupby(["city", "class"], as_index=False)["node_share"].mean()
        .pivot(index="city", columns="class", values="node_share")
        .fillna(0.0)
    )
    if FOCUS_CLASS in pivot.columns:
        pivot = pivot.sort_values(FOCUS_CLASS, ascending=False)
    class_order = [c for c in CLASS_COLORS if c in pivot.columns]
    pivot = pivot[class_order]
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.01, float(pivot.max().max())))
    ax.set_title("Mean Street-Pattern Node Shares By City")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([_short_city_label(city) for city in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean node share")
    _finish(fig, output_dir / "05_city_street_pattern_heatmap.png")


def _plot_demand_by_class(demand_class_metrics: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        demand_class_metrics.groupby("class", as_index=False)
        .agg(
            node_share=("node_share", "mean"),
            origin_demand_share=("origin_demand_share", "mean"),
            dest_demand_share=("dest_demand_share", "mean"),
            touch_demand_share=("touch_demand_share", "mean"),
        )
    )
    order = [c for c in CLASS_COLORS if c in set(summary["class"])]
    summary = summary.set_index("class").reindex(order).reset_index()
    x = np.arange(len(summary))
    width = 0.22
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, summary["node_share"], width, label="node share", color="#64748b")
    ax.bar(x, summary["origin_demand_share"], width, label="origin demand share", color="#2563eb")
    ax.bar(x + width, summary["touch_demand_share"], width, label="touch demand share", color="#f97316")
    ax.set_title("Demand Exposure By Street-Pattern Class")
    ax.set_ylabel("mean share")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["class"], rotation=20, ha="right")
    ax.grid(axis="y", color=GRID, alpha=0.55)
    ax.legend(frameon=True)
    _finish(fig, output_dir / "06_demand_by_street_pattern_class.png")


def _plot_scatter(sample_metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = sample_metrics["demand_source"].map({"gravity": "#2563eb", "synthetic": "#f97316"}).fillna("#64748b")
    axes[0].scatter(sample_metrics["density"], sample_metrics["positive_od_share"], c=colors, s=16, alpha=0.72)
    axes[0].set_title("Graph Density vs OD Coverage")
    axes[0].set_xlabel("density")
    axes[0].set_ylabel("positive OD share")
    axes[1].scatter(sample_metrics["loops_node_share"], sample_metrics["activity_gini"], c=colors, s=16, alpha=0.72)
    axes[1].set_title("Loops Share vs Demand Concentration")
    axes[1].set_xlabel("loops node share")
    axes[1].set_ylabel("activity gini")
    axes[2].scatter(sample_metrics["loops_node_share"], sample_metrics["demand_weighted_distance"], c=colors, s=16, alpha=0.72)
    axes[2].set_title("Loops Share vs Demand Distance")
    axes[2].set_xlabel("loops node share")
    axes[2].set_ylabel("weighted distance")
    for ax in axes:
        ax.grid(color=GRID, alpha=0.55)
    _finish(fig, output_dir / "07_structure_vs_demand_scatter.png")


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
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_dir or dataset_dir / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()

    manifest = _load_manifest(dataset_dir)
    raw_graphs, raw_path = _load_raw_graphs(dataset_dir)
    sample_metrics, class_metrics, demand_class_metrics = _metric_rows(raw_graphs, manifest)
    city_metrics = _city_metrics(sample_metrics, class_metrics, demand_class_metrics)

    sample_metrics.to_csv(output_dir / "sample_metrics.csv", index=False)
    class_metrics.to_csv(output_dir / "sample_class_shares.csv", index=False)
    demand_class_metrics.to_csv(output_dir / "sample_demand_by_class.csv", index=False)
    city_metrics.to_csv(output_dir / "city_metrics.csv", index=False)

    _plot_city_mix(sample_metrics, city_metrics, output_dir)
    _plot_network(sample_metrics, output_dir)
    _plot_demand(sample_metrics, output_dir)
    _plot_class_shares(class_metrics, output_dir)
    _plot_city_class_heatmap(class_metrics, output_dir)
    _plot_demand_by_class(demand_class_metrics, output_dir)
    _plot_scatter(sample_metrics, output_dir)

    summary = {
        "dataset_dir": str(dataset_dir),
        "raw_graphs": str(raw_path),
        "sample_count": int(len(sample_metrics)),
        "city_count": int(sample_metrics["city"].nunique()),
        "manifest_city_status_counts": dict(
            sorted(Counter(str(row.get("status", "unknown")) for row in manifest.get("cities", [])).items())
        ),
        "skipped_cities": [
            {
                "city": str(row.get("city")),
                "status": str(row.get("status")),
                "error": str(row.get("error")),
            }
            for row in manifest.get("cities", [])
            if int(row.get("samples") or 0) == 0
        ],
        "demand_source_counts": sample_metrics["demand_source"].value_counts().sort_index().to_dict(),
        "key_quantiles": {
            col: {
                "p05": float(sample_metrics[col].quantile(0.05)),
                "p50": float(sample_metrics[col].quantile(0.50)),
                "p95": float(sample_metrics[col].quantile(0.95)),
            }
            for col in [
                "edge_count",
                "density",
                "demand_sum",
                "positive_od_share",
                "activity_gini",
                "loops_node_share",
                "unknown_class_share",
            ]
        },
        "class_node_share_mean": (
            class_metrics.groupby("class")["node_share"].mean().sort_values(ascending=False).to_dict()
        ),
        "class_demand_touch_share_mean": (
            demand_class_metrics.groupby("class")["touch_demand_share"].mean().sort_values(ascending=False).to_dict()
        ),
        "png_checks": _png_checks(output_dir),
        "files": {
            "sample_metrics": str(output_dir / "sample_metrics.csv"),
            "city_metrics": str(output_dir / "city_metrics.csv"),
            "sample_class_shares": str(output_dir / "sample_class_shares.csv"),
            "sample_demand_by_class": str(output_dir / "sample_demand_by_class.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

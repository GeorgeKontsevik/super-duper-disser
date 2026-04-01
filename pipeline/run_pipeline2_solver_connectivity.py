"""
Pipeline 2 launcher: solver-based connectivity scenario with caching and PNG exports.

Flow:
1) run_base_osm -> baseline model
2) run_solver -> FLP optimization
3) run_connectivity.apply_connectivity -> connectivity intervention rerun
4) save cache artifacts + PNG diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

# Keep matplotlib cache inside project workspace for stable non-interactive runs.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parents[1] / ".cache" / "matplotlib").resolve()),
)

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from pipeline import run_base, run_base_osm, run_connectivity, run_solver
from pipeline.config import SERVICE_NAME, SETTL_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline_2 solver connectivity scenario with caching and PNG exports.")
    parser.add_argument("--settl-name", default=SETTL_NAMES[0], help="Settlement key from pipeline/config.py")
    parser.add_argument("--service-name", default=SERVICE_NAME, help="Service name (health/post/culture/...)")
    parser.add_argument(
        "--output-dir",
        default="/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/pipeline_2/solver_connectivity",
        help="Directory for cache + pictures",
    )
    parser.add_argument(
        "--base-runner",
        choices=["run_base", "run_base_osm"],
        default="run_base",
        help="Baseline source for pipeline_2: local processed graph (run_base) or OSM/iduedu graph (run_base_osm).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Force recompute even if summary already exists.")
    return parser.parse_args()


def _average_fitness(history: list) -> list[float]:
    result: list[float] = []
    for generation in history or []:
        if not generation:
            continue
        result.append(float(sum(generation) / len(generation)))
    return result


def _save_fitness_plot(history: list, output_path: Path) -> bool:
    avg = _average_fitness(history)
    if not avg:
        return False
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(avg) + 1), avg, color="#2563eb", linewidth=2)
    ax.set_title("Solver Fitness (average per generation)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average fitness")
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_delta_bar(df_compare: pd.DataFrame, output_path: Path, top_n: int = 20) -> bool:
    if df_compare is None or df_compare.empty or "delta" not in df_compare.columns:
        return False
    data = df_compare.copy().sort_values("delta", ascending=False).head(top_n)
    if data.empty:
        return False
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in data["delta"]]
    ax.bar(data.index.astype(str), data["delta"], color=colors)
    ax.set_title(f"Provision delta (top {top_n})")
    ax.set_xlabel("Block")
    ax.set_ylabel("Delta")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _save_connectivity_map(
    blocks_gdf: gpd.GeoDataFrame,
    df_compare: pd.DataFrame,
    improved_edges: list[tuple[str, str, float, float]],
    output_path: Path,
) -> bool:
    if blocks_gdf is None or blocks_gdf.empty:
        return False

    blocks = blocks_gdf.copy()
    if "name" not in blocks.columns:
        return False
    if blocks.crs is None:
        blocks = blocks.set_crs(4326)
    try:
        blocks = blocks.to_crs("EPSG:3857")
    except Exception:
        pass

    compare = df_compare.copy() if df_compare is not None else pd.DataFrame()
    if not compare.empty:
        compare = compare.reset_index().rename(columns={"index": "name"})
        blocks = blocks.merge(compare[["name", "delta"]], on="name", how="left")
    if "delta" not in blocks.columns:
        blocks["delta"] = 0.0
    blocks["delta"] = pd.to_numeric(blocks["delta"], errors="coerce").fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 12))
    vmax = float(np.nanmax(np.abs(blocks["delta"]))) if len(blocks) else 1.0
    vmax = vmax if vmax > 0 else 1.0
    blocks.plot(
        ax=ax,
        column="delta",
        cmap="RdYlGn",
        linewidth=0.15,
        edgecolor="#9ca3af",
        legend=False,
        vmin=-vmax,
        vmax=vmax,
        alpha=0.85,
    )

    centroids = blocks.set_index("name").geometry.centroid.to_dict()
    drawn = 0
    if improved_edges:
        diffs = [max(0.0, float(old_t) - float(new_t)) for _, _, old_t, new_t in improved_edges]
        max_diff = max(diffs) if diffs else 1.0
        max_diff = max_diff if max_diff > 0 else 1.0
        norm = Normalize(vmin=0.0, vmax=max_diff)
        cmap = plt.get_cmap("coolwarm")
        for name_i, name_j, old_t, new_t in improved_edges:
            p1 = centroids.get(name_i)
            p2 = centroids.get(name_j)
            if p1 is None or p2 is None:
                continue
            diff = max(0.0, float(old_t) - float(new_t))
            ax.plot(
                [p1.x, p2.x],
                [p1.y, p2.y],
                color=cmap(norm(diff)),
                linewidth=1.4,
                alpha=0.8,
            )
            drawn += 1

    ax.set_title("Pipeline 2: connectivity scenario (provision delta + improved links)")
    ax.set_axis_off()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return drawn > 0 or not blocks.empty


def _compact_baseline_payload(all_results: dict, base_ctx: dict, settl_name: str, service_name: str) -> dict:
    graphs = all_results[settl_name][service_name]["stats"].graphs
    return {
        "settl_name": settl_name,
        "service_name": service_name,
        "graphs": graphs,
        "blocks_gdf": base_ctx.get("blocks_gdf"),
        "G_undirected": base_ctx.get("G_undirected"),
    }


def _restore_baseline_payload(compact_payload: dict) -> tuple[dict, dict]:
    settl_name = compact_payload["settl_name"]
    service_name = compact_payload["service_name"]
    graphs = compact_payload["graphs"]
    stats_obj = SimpleNamespace(graphs=graphs)

    all_results = {
        settl_name: {
            service_name: {
                "stats": stats_obj,
            }
        }
    }
    base_ctx = {
        "settl_name": settl_name,
        "service_name": service_name,
        "blocks_gdf": compact_payload.get("blocks_gdf"),
        "G_undirected": compact_payload.get("G_undirected"),
    }
    return all_results, base_ctx


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    solver_result_path = output_dir / "solver_result.pkl"
    compare_csv_path = output_dir / "compare_connectivity.csv"
    improved_edges_csv_path = output_dir / "improved_edges.csv"
    img_fitness = output_dir / "01_solver_fitness.png"
    img_delta = output_dir / "02_provision_delta_top.png"
    img_map = output_dir / "03_connectivity_links_map.png"
    baseline_compact_cache_path = output_dir / "baseline_compact.pkl"

    if summary_path.exists() and (not args.no_cache):
        print(f"[pipeline_2] Cache hit: {summary_path}")
        print("[pipeline_2] Reuse existing artifacts (use --no-cache to recompute).")
        return

    if (not args.no_cache) and baseline_compact_cache_path.exists():
        print(f"[pipeline_2] Reusing cached baseline: {baseline_compact_cache_path}")
        compact_payload = pd.read_pickle(baseline_compact_cache_path)
        all_results, base_ctx = _restore_baseline_payload(compact_payload)
    else:
        baseline_runner = run_base.run if args.base_runner == "run_base" else run_base_osm.run
        print(f"[pipeline_2] Running baseline ({args.base_runner})...")
        all_results, base_ctx = baseline_runner(
            settl_names=[args.settl_name],
            service_name=args.service_name,
        )
        compact_payload = _compact_baseline_payload(all_results, base_ctx, args.settl_name, args.service_name)
        pd.to_pickle(compact_payload, baseline_compact_cache_path)
        all_results, base_ctx = _restore_baseline_payload(compact_payload)
        print(f"[pipeline_2] Baseline cached: {baseline_compact_cache_path}")

    print("[pipeline_2] Running solver (run_solver)...")
    solver_result = run_solver.run(
        all_results,
        settl_name=args.settl_name,
        service_name=args.service_name,
    )

    print("[pipeline_2] Applying connectivity scenario (run_connectivity.apply_connectivity)...")
    if isinstance(solver_result.get("best_candidate"), pd.DataFrame):
        all_results, connectivity_payload = run_connectivity.apply_connectivity(
            all_results,
            solver_result,
            base_ctx,
        )
    else:
        print("[pipeline_2] Solver returned no candidate matrix; connectivity rerun skipped.")
        connectivity_payload = {"df_compare": pd.DataFrame(), "improved_edges": []}

    # Cache tables/artifacts
    pd.to_pickle(solver_result, solver_result_path)
    df_compare = connectivity_payload.get("df_compare")
    if isinstance(df_compare, pd.DataFrame):
        df_compare.to_csv(compare_csv_path, index=True)

    improved_edges = connectivity_payload.get("improved_edges") or []
    pd.DataFrame(
        improved_edges,
        columns=["name_i", "name_j", "old_time_min", "new_time_min"],
    ).to_csv(improved_edges_csv_path, index=False)

    # Pictures
    fitness_written = _save_fitness_plot(solver_result.get("fitness_history", []), img_fitness)
    delta_written = _save_delta_bar(df_compare, img_delta)
    map_written = _save_connectivity_map(base_ctx.get("blocks_gdf"), df_compare, improved_edges, img_map)

    summary = {
        "pipeline": "pipeline_2",
        "settl_name": args.settl_name,
        "service_name": args.service_name,
        "cache_used": False,
        "solver": {
            "uncovered_count": int(len(solver_result.get("uncovered_ids", []))),
            "new_capacity_blocks": int(sum(1 for c in solver_result.get("capacities", []) if c and c > 0)),
            "fitness_generations": int(len(solver_result.get("fitness_history", []))),
        },
        "connectivity": {
            "improved_edges_count": int(len(improved_edges)),
        },
        "artifacts": {
            "baseline_compact_pickle": str(baseline_compact_cache_path),
            "solver_result_pickle": str(solver_result_path),
            "compare_csv": str(compare_csv_path),
            "improved_edges_csv": str(improved_edges_csv_path),
            "fitness_png": str(img_fitness) if fitness_written else None,
            "delta_png": str(img_delta) if delta_written else None,
            "map_png": str(img_map) if map_written else None,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pipeline_2] Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()

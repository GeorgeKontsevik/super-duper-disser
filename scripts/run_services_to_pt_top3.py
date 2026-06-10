#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


DEFAULT_SERVICES = ["hospital", "polyclinic", "school", "kindergarten"]


def _city_dirs(base: Path) -> list[Path]:
    return sorted([p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")])


def _service_points(service_path: Path, service_name: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(service_path)
    if gdf.empty:
        return gdf
    gdf = gdf.reset_index(drop=False).rename(columns={"index": "service_idx"})
    gdf["service_name"] = service_name
    gdf["rep_point"] = gdf.geometry.representative_point()
    return gdf


def _attach_street_pattern(points: gpd.GeoDataFrame, street_cells_path: Path) -> gpd.GeoDataFrame:
    if points.empty or not street_cells_path.exists():
        return points
    cells = gpd.read_file(street_cells_path)
    if "top1_class_name" not in cells.columns:
        return points
    join_points = gpd.GeoDataFrame(
        points[["service_idx"]].copy(),
        geometry=points["rep_point"],
        crs=points.crs,
    )
    joined = gpd.sjoin(
        join_points,
        cells[["top1_class_name", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    return points.merge(joined[["service_idx", "top1_class_name"]], on="service_idx", how="left")


def _graph_inputs(intermodal_dir: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame, ig.Graph, np.ndarray, pd.DataFrame]:
    nodes = gpd.read_parquet(intermodal_dir / "graph_nodes.parquet")
    edges = pd.read_parquet(intermodal_dir / "graph_edges.parquet")
    walk_edges = edges[edges["type"] == "walk"].copy()
    if walk_edges.empty:
        raise ValueError("no_walk_edges")

    n_vertices = int(max(nodes["index"].max(), walk_edges["u"].max(), walk_edges["v"].max())) + 1
    graph = ig.Graph(n=n_vertices, directed=False)
    graph.add_edges(list(zip(walk_edges["u"].astype(int), walk_edges["v"].astype(int), strict=False)))
    weights = walk_edges["length_meter"].astype(float).to_numpy()

    platform_nodes = nodes[nodes["type"] == "platform"][["index", "stop_transport_type"]].copy()
    platform_nodes["index"] = platform_nodes["index"].astype(int)
    if platform_nodes.empty:
        raise ValueError("no_platform_nodes")
    return nodes, walk_edges, graph, weights, platform_nodes


def _compute_service_top3_for_city(city_dir: Path, out_root: Path, services: list[str]) -> list[dict]:
    city = city_dir.name
    intermodal = city_dir / "intermodal_graph_iduedu"
    services_root = city_dir / "pipeline_2" / "services_raw"
    street_cells = city_dir / "street_pattern" / city / "predicted_cells.geojson"

    nodes, _walk_edges, graph, weights, platform_nodes = _graph_inputs(intermodal)
    targets = platform_nodes["index"].to_list()
    stop_type_map = platform_nodes.set_index("index")["stop_transport_type"].to_dict()

    node_xy = nodes.sort_values("index")[["x", "y"]].to_numpy()
    kd = cKDTree(node_xy)
    dist_target_to_all = np.asarray(graph.distances(source=targets, target=None, weights=weights))
    target_arr = np.array(targets, dtype=int)

    summaries: list[dict] = []
    for service_name in services:
        service_path = services_root / f"{service_name}.parquet"
        if not service_path.exists():
            summaries.append({"city": city, "service_name": service_name, "status": "missing_service_file"})
            continue

        points = _service_points(service_path, service_name)
        if points.empty:
            summaries.append({"city": city, "service_name": service_name, "service_points": 0, "rows_out": 0, "status": "empty"})
            continue

        points = _attach_street_pattern(points, street_cells)
        points_proj = gpd.GeoDataFrame(
            points[["service_idx", "rep_point"]].copy(),
            geometry="rep_point",
            crs=points.crs,
        ).to_crs(nodes.crs)

        pxy = np.c_[points_proj.geometry.x.to_numpy(), points_proj.geometry.y.to_numpy()]
        nearest_node_dist_m, nearest_nodes = kd.query(pxy, k=1)
        nearest_nodes = nearest_nodes.astype(int)

        dist_matrix = dist_target_to_all[:, nearest_nodes].T
        k = min(3, dist_matrix.shape[1])
        nearest_idx = np.argpartition(dist_matrix, kth=k - 1, axis=1)[:, :k]

        rows = []
        for i in range(dist_matrix.shape[0]):
            local = nearest_idx[i]
            local_sorted = local[np.argsort(dist_matrix[i, local])]
            for rank, j in enumerate(local_sorted, start=1):
                node_id = int(target_arr[j])
                rows.append(
                    {
                        "service_idx": int(points.iloc[i]["service_idx"]),
                        "service_name": service_name,
                        "source_uid": points.iloc[i].get("source_uid"),
                        "name": points.iloc[i].get("name"),
                        "rank": rank,
                        "walk_distance_m": float(dist_matrix[i, j]),
                        "distance_to_nearest_graph_node_m": float(nearest_node_dist_m[i]),
                        "source_graph_node": int(nearest_nodes[i]),
                        "stop_graph_node": node_id,
                        "stop_modality": stop_type_map.get(node_id),
                        "street_pattern_class": points.iloc[i].get("top1_class_name"),
                    }
                )

        out_city = out_root / city
        out_city.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(rows)
        out_df.to_parquet(out_city / f"{service_name}_to_pt_top3.parquet", index=False)

        finite_dist = pd.to_numeric(out_df["walk_distance_m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        summary = {
            "city": city,
            "service_name": service_name,
            "service_points": int(len(points)),
            "rows_out": int(len(out_df)),
            "reachable_rank1_points": int(np.isfinite(out_df.loc[out_df["rank"] == 1, "walk_distance_m"]).sum()),
            "mean_distance_m_reachable": float(finite_dist.mean()) if finite_dist.notna().any() else None,
            "median_distance_m_reachable": float(finite_dist.median()) if finite_dist.notna().any() else None,
            "status": "ok",
        }
        (out_city / f"{service_name}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary)

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-inputs-root",
        type=Path,
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/services_to_pt_top3"),
    )
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--services", nargs="*", default=DEFAULT_SERVICES)
    args = parser.parse_args()

    city_dirs = _city_dirs(args.joint_inputs_root)
    if args.cities:
        wanted = set(args.cities)
        city_dirs = [c for c in city_dirs if c.name in wanted]

    results: list[dict] = []
    for city_dir in tqdm(city_dirs, desc="Cities", unit="city"):
        try:
            city_results = _compute_service_top3_for_city(city_dir, args.out_root, list(args.services))
        except Exception as exc:  # pragma: no cover
            city_results = [{"city": city_dir.name, "service_name": "*", "status": "error", "error": str(exc)}]
        for res in city_results:
            print(f"{res['city']} {res['service_name']}: {res['status']}")
            results.append(res)

    args.out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out_root / "_run_report.tsv", sep="\t", index=False)
    (args.out_root / "_run_report.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

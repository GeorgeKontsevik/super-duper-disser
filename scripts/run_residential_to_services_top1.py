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
WALK_MIN_PER_M = 0.012


def _city_dirs(base: Path) -> list[Path]:
    return sorted([p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")])


def _living_buildings(buildings_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(buildings_path)
    living_flag = pd.to_numeric(gdf["is_living"], errors="coerce").fillna(0).astype(float) > 0.0
    living = gdf[living_flag].copy()
    if living.empty:
        return living
    living = living.reset_index(drop=False).rename(columns={"index": "building_idx"})
    living["rep_point"] = living.geometry.representative_point()
    return living


def _buildings_access_path(derived: Path) -> Path:
    is_living_path = derived / "buildings_is_living_enriched.parquet"
    if is_living_path.exists():
        return is_living_path
    return derived / "buildings_floor_enriched.parquet"


def _service_points(service_path: Path, service_name: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(service_path)
    if gdf.empty:
        return gdf
    gdf = gdf.reset_index(drop=False).rename(columns={"index": "service_idx"})
    gdf["service_name"] = service_name
    gdf["rep_point"] = gdf.geometry.representative_point()
    return gdf


def _attach_street_pattern(points: gpd.GeoDataFrame, id_col: str, street_cells_path: Path) -> gpd.GeoDataFrame:
    if points.empty or not street_cells_path.exists():
        return points
    cells = gpd.read_file(street_cells_path)
    if "top1_class_name" not in cells.columns:
        return points
    join_points = gpd.GeoDataFrame(points[[id_col]].copy(), geometry=points["rep_point"], crs=points.crs)
    joined = gpd.sjoin(
        join_points,
        cells[["top1_class_name", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    return points.merge(joined[[id_col, "top1_class_name"]], on=id_col, how="left")


def _graph_inputs(intermodal_dir: Path) -> tuple[gpd.GeoDataFrame, ig.Graph, np.ndarray]:
    nodes = gpd.read_parquet(intermodal_dir / "graph_nodes.parquet")
    edges = pd.read_parquet(intermodal_dir / "graph_edges.parquet")
    walk_edges = edges[edges["type"] == "walk"].copy()
    if walk_edges.empty:
        raise ValueError("no_walk_edges")

    n_vertices = int(max(nodes["index"].max(), walk_edges["u"].max(), walk_edges["v"].max())) + 1
    graph = ig.Graph(n=n_vertices, directed=False)
    graph.add_edges(list(zip(walk_edges["u"].astype(int), walk_edges["v"].astype(int), strict=False)))
    weights = walk_edges["length_meter"].astype(float).to_numpy()
    return nodes, graph, weights


def _compute_city(city_dir: Path, out_root: Path, services: list[str]) -> list[dict]:
    city = city_dir.name
    derived = city_dir / "derived_layers"
    intermodal = city_dir / "intermodal_graph_iduedu"
    services_root = city_dir / "pipeline_2" / "services_raw"
    street_cells = city_dir / "street_pattern" / city / "predicted_cells.geojson"

    homes = _living_buildings(_buildings_access_path(derived))
    if homes.empty:
        return [{"city": city, "service_name": "*", "status": "no_living_buildings"}]

    homes = _attach_street_pattern(homes, "building_idx", street_cells)
    nodes, graph, weights = _graph_inputs(intermodal)

    node_xy = nodes.sort_values("index")[["x", "y"]].to_numpy()
    kd = cKDTree(node_xy)

    home_points = gpd.GeoDataFrame(
        homes[["building_idx", "rep_point"]].copy(),
        geometry="rep_point",
        crs=homes.crs,
    ).to_crs(nodes.crs)
    home_xy = np.c_[home_points.geometry.x.to_numpy(), home_points.geometry.y.to_numpy()]
    home_snap_m, home_nodes = kd.query(home_xy, k=1)
    home_nodes = home_nodes.astype(int)

    city_rows = []
    summaries = []

    for service_name in services:
        service_path = services_root / f"{service_name}.parquet"
        if not service_path.exists():
            summaries.append({"city": city, "service_name": service_name, "status": "missing_service_file"})
            continue

        svc = _service_points(service_path, service_name)
        if svc.empty:
            summaries.append({"city": city, "service_name": service_name, "service_points": 0, "rows_out": 0, "status": "empty"})
            continue

        svc = _attach_street_pattern(svc, "service_idx", street_cells)
        svc_points = gpd.GeoDataFrame(
            svc[["service_idx", "rep_point"]].copy(),
            geometry="rep_point",
            crs=svc.crs,
        ).to_crs(nodes.crs)
        svc_xy = np.c_[svc_points.geometry.x.to_numpy(), svc_points.geometry.y.to_numpy()]
        svc_snap_m, svc_nodes = kd.query(svc_xy, k=1)
        svc_nodes = svc_nodes.astype(int)

        dist_service_to_all = np.asarray(graph.distances(source=svc_nodes.tolist(), target=None, weights=weights))
        dist_matrix = dist_service_to_all[:, home_nodes].T
        nearest_service_local = np.argmin(dist_matrix, axis=1)

        for i in range(dist_matrix.shape[0]):
            service_local = int(nearest_service_local[i])
            walk_distance_m = float(dist_matrix[i, service_local])
            svc_row = svc.iloc[service_local]
            city_rows.append(
                {
                    "building_idx": int(homes.iloc[i]["building_idx"]),
                    "service_name": service_name,
                    "walk_distance_m": walk_distance_m,
                    "walk_time_min": walk_distance_m * WALK_MIN_PER_M,
                    "distance_home_to_graph_node_m": float(home_snap_m[i]),
                    "distance_service_to_graph_node_m": float(svc_snap_m[service_local]),
                    "home_graph_node": int(home_nodes[i]),
                    "service_graph_node": int(svc_nodes[service_local]),
                    "nearest_service_idx": int(svc_row["service_idx"]),
                    "nearest_service_source_uid": svc_row.get("source_uid"),
                    "nearest_service_name": svc_row.get("name"),
                    "home_street_pattern_class": homes.iloc[i].get("top1_class_name"),
                    "service_street_pattern_class": svc_row.get("top1_class_name"),
                }
            )

        rank1_dist = pd.Series(dist_matrix[np.arange(dist_matrix.shape[0]), nearest_service_local]).replace([np.inf, -np.inf], np.nan)
        summaries.append(
            {
                "city": city,
                "service_name": service_name,
                "service_points": int(len(svc)),
                "rows_out": int(len(homes)),
                "reachable_home_count": int(rank1_dist.notna().sum()),
                "mean_distance_m_reachable": float(rank1_dist.mean()) if rank1_dist.notna().any() else None,
                "median_distance_m_reachable": float(rank1_dist.median()) if rank1_dist.notna().any() else None,
                "mean_time_min_reachable": float((rank1_dist * WALK_MIN_PER_M).mean()) if rank1_dist.notna().any() else None,
                "median_time_min_reachable": float((rank1_dist * WALK_MIN_PER_M).median()) if rank1_dist.notna().any() else None,
                "status": "ok",
            }
        )

    out_city = out_root / city
    out_city.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(city_rows)
    if not out_df.empty:
        out_df.to_parquet(out_city / "residential_to_services_top1.parquet", index=False)
    for summary in summaries:
        service_name = summary["service_name"]
        (out_city / f"{service_name}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_top1"),
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
            city_results = _compute_city(city_dir, args.out_root, list(args.services))
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

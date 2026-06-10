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


def _city_dirs(base: Path) -> list[Path]:
    return sorted([p for p in base.iterdir() if p.is_dir() and not p.name.startswith("_")])


def _living_buildings(buildings_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(buildings_path)
    if "is_living" not in gdf.columns:
        raise ValueError(f"Missing is_living in {buildings_path}")
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


def _attach_street_pattern(living: gpd.GeoDataFrame, street_cells_path: Path) -> gpd.GeoDataFrame:
    if living.empty:
        return living
    cells = gpd.read_file(street_cells_path)
    if "top1_class_name" not in cells.columns:
        return living
    join_points = gpd.GeoDataFrame(
        living[["building_idx"]].copy(),
        geometry=living["rep_point"],
        crs=living.crs,
    )
    joined = gpd.sjoin(
        join_points,
        cells[["top1_class_name", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    living = living.merge(joined[["building_idx", "top1_class_name"]], on="building_idx", how="left")
    return living


def _compute_top3_for_city(city_dir: Path, out_root: Path) -> dict:
    city = city_dir.name
    derived = city_dir / "derived_layers"
    intermodal = city_dir / "intermodal_graph_iduedu"
    street_cells = city_dir / "street_pattern" / city / "predicted_cells.geojson"

    living = _living_buildings(_buildings_access_path(derived))
    if living.empty:
        return {"city": city, "living_buildings": 0, "rows_out": 0, "status": "no_living_buildings"}

    if street_cells.exists():
        living = _attach_street_pattern(living, street_cells)

    nodes = gpd.read_parquet(intermodal / "graph_nodes.parquet")
    edges = pd.read_parquet(intermodal / "graph_edges.parquet")

    walk_edges = edges[edges["type"] == "walk"].copy()
    if walk_edges.empty:
        return {"city": city, "living_buildings": len(living), "rows_out": 0, "status": "no_walk_edges"}

    n_vertices = int(max(nodes["index"].max(), walk_edges["u"].max(), walk_edges["v"].max())) + 1
    g = ig.Graph(n=n_vertices, directed=False)
    g.add_edges(list(zip(walk_edges["u"].astype(int), walk_edges["v"].astype(int), strict=False)))
    weights = walk_edges["length_meter"].astype(float).to_numpy()

    platform_nodes = nodes[nodes["type"] == "platform"][["index", "stop_transport_type"]].copy()
    platform_nodes["index"] = platform_nodes["index"].astype(int)
    targets = platform_nodes["index"].to_list()
    if not targets:
        return {"city": city, "living_buildings": len(living), "rows_out": 0, "status": "no_platform_nodes"}

    living_points = gpd.GeoDataFrame(
        living[["building_idx", "rep_point"]].copy(),
        geometry="rep_point",
        crs=living.crs,
    ).to_crs(nodes.crs)

    node_xy = nodes.sort_values("index")[["x", "y"]].to_numpy()
    kd = cKDTree(node_xy)
    bxy = np.c_[living_points.geometry.x.to_numpy(), living_points.geometry.y.to_numpy()]
    nearest_node_dist_m, nearest_nodes = kd.query(bxy, k=1)
    nearest_nodes = nearest_nodes.astype(int)

    # Faster than per-building shortest paths: compute once from each platform to all graph nodes.
    dist_target_to_all = np.asarray(g.distances(source=targets, target=None, weights=weights))
    dist_matrix = dist_target_to_all[:, nearest_nodes].T
    k = min(3, dist_matrix.shape[1])
    nearest_idx = np.argpartition(dist_matrix, kth=k - 1, axis=1)[:, :k]

    rows = []
    target_arr = np.array(targets, dtype=int)
    stop_type_map = platform_nodes.set_index("index")["stop_transport_type"].to_dict()
    for i in range(dist_matrix.shape[0]):
        local = nearest_idx[i]
        local_sorted = local[np.argsort(dist_matrix[i, local])]
        for rank, j in enumerate(local_sorted, start=1):
            node_id = int(target_arr[j])
            d = float(dist_matrix[i, j])
            rows.append(
                {
                    "building_idx": int(living.iloc[i]["building_idx"]),
                    "rank": rank,
                    "walk_distance_m": d,
                    "distance_to_nearest_graph_node_m": float(nearest_node_dist_m[i]),
                    "source_graph_node": int(nearest_nodes[i]),
                    "stop_graph_node": node_id,
                    "stop_modality": stop_type_map.get(node_id),
                    "street_pattern_class": living.iloc[i].get("top1_class_name"),
                }
            )

    out_city = out_root / city
    out_city.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_city / "residential_to_pt_top3.parquet", index=False)
    summary = {
        "city": city,
        "living_buildings": int(len(living)),
        "rows_out": int(len(out_df)),
        "mean_distance_m": float(pd.to_numeric(out_df["walk_distance_m"], errors="coerce").mean()),
        "median_distance_m": float(pd.to_numeric(out_df["walk_distance_m"], errors="coerce").median()),
        "status": "ok",
    }
    (out_city / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-inputs-root",
        type=Path,
        default=Path(
            "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_pt_top3"
        ),
    )
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()

    results = []
    city_dirs = _city_dirs(args.joint_inputs_root)
    if args.cities:
        wanted = set(args.cities)
        city_dirs = [c for c in city_dirs if c.name in wanted]
    for city_dir in tqdm(city_dirs, desc="Cities", unit="city"):
        try:
            res = _compute_top3_for_city(city_dir, args.out_root)
        except Exception as exc:  # pragma: no cover
            res = {"city": city_dir.name, "status": "error", "error": str(exc)}
        print(f"{res['city']}: {res['status']}")
        results.append(res)

    args.out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out_root / "_run_report.tsv", sep="\t", index=False)
    (args.out_root / "_run_report.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

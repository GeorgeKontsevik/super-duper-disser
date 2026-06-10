#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


DEFAULT_SERVICES = ["hospital", "polyclinic", "school", "kindergarten"]
DEFAULT_MIN_WALK_MIN = 15.0


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


def _eligible_buildings_for_service(
    walk_df: pd.DataFrame,
    service_name: str,
    min_walk_min: float | None = None,
    max_walk_min_exclusive: float | None = None,
) -> list[int]:
    subset = walk_df[walk_df["service_name"] == service_name].copy()
    if subset.empty:
        return []
    walk_time = pd.to_numeric(subset["walk_time_min"], errors="coerce")
    mask = walk_time.notna()
    if min_walk_min is not None:
        mask &= walk_time >= min_walk_min
        mask |= np.isinf(walk_time.to_numpy())
    if max_walk_min_exclusive is not None:
        mask &= walk_time < max_walk_min_exclusive
    return subset.loc[mask, "building_idx"].astype(int).tolist()


def _multi_source_pt_to_services(
    graph: nx.MultiDiGraph,
    service_nodes: list[int],
) -> tuple[dict[int, float], dict[int, int], dict[int, list[int]]]:
    reverse_graph = graph.reverse(copy=False)
    distances, paths = nx.multi_source_dijkstra(reverse_graph, service_nodes, weight="time_min")
    source_map = {node: int(path[0]) for node, path in paths.items() if path}
    distance_map = {int(node): float(dist) for node, dist in distances.items()}
    path_map = {int(node): [int(p) for p in reversed(path)] for node, path in paths.items() if path}
    return distance_map, source_map, path_map


def _best_edge_data(graph: nx.MultiDiGraph, u: int, v: int) -> dict:
    edge_bundle = graph.get_edge_data(u, v)
    if not edge_bundle:
        raise KeyError(f"missing edge for path step {u}->{v}")
    return min(edge_bundle.values(), key=lambda data: float(data.get("time_min", np.inf)))


def _decompose_pt_path(
    graph: nx.MultiDiGraph,
    path_nodes: list[int],
    distance_home_to_graph_node_m: float,
    distance_service_to_graph_node_m: float,
) -> dict[str, float]:
    walk_edge_time_min = 0.0
    transport_time_min = 0.0
    other_edge_time_min = 0.0
    edges: list[dict[str, float | str]] = []

    for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False):
        edge_data = _best_edge_data(graph, int(u), int(v))
        edge_time_min = float(edge_data.get("time_min", 0.0) or 0.0)
        edge_type = str(edge_data.get("type", "") or "").lower()
        edges.append({"time_min": edge_time_min, "type": edge_type})
        if edge_type == "walk":
            walk_edge_time_min += edge_time_min
        elif edge_type in {"boarding"}:
            other_edge_time_min += edge_time_min
        else:
            transport_time_min += edge_time_min

    home_snap_walk_time_min = float(distance_home_to_graph_node_m) * 0.012
    service_snap_walk_time_min = float(distance_service_to_graph_node_m) * 0.012
    transport_positions = [idx for idx, edge in enumerate(edges) if edge["type"] not in {"walk", "boarding"}]

    access_walk_edge_time_min = 0.0
    egress_walk_edge_time_min = 0.0
    transfer_walk_edge_time_min = 0.0
    if transport_positions:
        first_transport_idx = min(transport_positions)
        last_transport_idx = max(transport_positions)
        access_walk_edge_time_min = sum(
            float(edge["time_min"])
            for idx, edge in enumerate(edges)
            if idx < first_transport_idx and edge["type"] == "walk"
        )
        egress_walk_edge_time_min = sum(
            float(edge["time_min"])
            for idx, edge in enumerate(edges)
            if idx > last_transport_idx and edge["type"] == "walk"
        )
        transfer_walk_edge_time_min = sum(
            float(edge["time_min"])
            for idx, edge in enumerate(edges)
            if first_transport_idx < idx < last_transport_idx and edge["type"] == "walk"
        )
    else:
        access_walk_edge_time_min = walk_edge_time_min

    access_walk_time_min = home_snap_walk_time_min + access_walk_edge_time_min
    egress_walk_time_min = service_snap_walk_time_min + egress_walk_edge_time_min
    transfer_time_min = other_edge_time_min + transfer_walk_edge_time_min
    access_egress_walk_time_min = walk_edge_time_min + home_snap_walk_time_min + service_snap_walk_time_min
    total_time_min = access_walk_time_min + egress_walk_time_min + transfer_time_min + transport_time_min
    return {
        "walk_edge_time_min": walk_edge_time_min,
        "transport_time_min": transport_time_min,
        "other_edge_time_min": other_edge_time_min,
        "home_snap_walk_time_min": home_snap_walk_time_min,
        "service_snap_walk_time_min": service_snap_walk_time_min,
        "access_walk_edge_time_min": access_walk_edge_time_min,
        "egress_walk_edge_time_min": egress_walk_edge_time_min,
        "transfer_walk_edge_time_min": transfer_walk_edge_time_min,
        "access_walk_time_min": access_walk_time_min,
        "egress_walk_time_min": egress_walk_time_min,
        "transfer_time_min": transfer_time_min,
        "access_egress_walk_time_min": access_egress_walk_time_min,
        "has_transport_segment": float(bool(transport_positions)),
        "total_time_min": total_time_min,
    }


def _graph_inputs(intermodal_dir: Path) -> tuple[gpd.GeoDataFrame, nx.MultiDiGraph]:
    nodes = gpd.read_parquet(intermodal_dir / "graph_nodes.parquet")
    with (intermodal_dir / "graph.pkl").open("rb") as fh:
        graph = pickle.load(fh)
    return nodes, graph


def _compute_city(
    city_dir: Path,
    out_root: Path,
    walk_root: Path,
    services: list[str],
    min_walk_min: float | None,
    max_walk_min_exclusive: float | None,
) -> list[dict]:
    city = city_dir.name
    derived = city_dir / "derived_layers"
    intermodal = city_dir / "intermodal_graph_iduedu"
    services_root = city_dir / "pipeline_2" / "services_raw"
    street_cells = city_dir / "street_pattern" / city / "predicted_cells.geojson"

    walk_path = walk_root / city / "residential_to_services_top1.parquet"
    if not walk_path.exists():
        return [{"city": city, "service_name": "*", "status": "missing_walk_baseline"}]
    walk_df = pd.read_parquet(
        walk_path,
        columns=[
            "building_idx",
            "service_name",
            "walk_time_min",
        ],
    )

    homes = _living_buildings(_buildings_access_path(derived))
    if homes.empty:
        return [{"city": city, "service_name": "*", "status": "no_living_buildings"}]
    homes = _attach_street_pattern(homes, "building_idx", street_cells)

    nodes, graph = _graph_inputs(intermodal)
    node_xy = nodes.sort_values("index")[["x", "y"]].to_numpy()
    kd = cKDTree(node_xy)

    home_points = gpd.GeoDataFrame(
        homes[["building_idx", "rep_point"]].copy(),
        geometry="rep_point",
        crs=homes.crs,
    ).to_crs(nodes.crs)
    home_xy = np.c_[home_points.geometry.x.to_numpy(), home_points.geometry.y.to_numpy()]
    home_snap_m, home_nodes = kd.query(home_xy, k=1)
    homes["home_graph_node"] = home_nodes.astype(int)
    homes["distance_home_to_graph_node_m"] = home_snap_m.astype(float)

    city_rows: list[dict] = []
    summaries: list[dict] = []

    for service_name in services:
        service_path = services_root / f"{service_name}.parquet"
        if not service_path.exists():
            summaries.append({"city": city, "service_name": service_name, "status": "missing_service_file"})
            continue

        eligible_ids = set(
            _eligible_buildings_for_service(
                walk_df,
                service_name,
                min_walk_min=min_walk_min,
                max_walk_min_exclusive=max_walk_min_exclusive,
            )
        )
        if not eligible_ids:
            summaries.append(
                {
                    "city": city,
                    "service_name": service_name,
                    "eligible_homes": 0,
                    "rows_out": 0,
                    "status": "no_homes_after_walk_filter",
                }
            )
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
        svc["service_graph_node"] = svc_nodes.astype(int)
        svc["distance_service_to_graph_node_m"] = svc_snap_m.astype(float)

        distance_map, source_map, path_map = _multi_source_pt_to_services(
            graph, svc["service_graph_node"].astype(int).tolist()
        )
        svc_by_node = svc.drop_duplicates(subset=["service_graph_node"]).set_index("service_graph_node")

        eligible_homes = homes[homes["building_idx"].isin(eligible_ids)].copy()
        for _, home_row in eligible_homes.iterrows():
            home_node = int(home_row["home_graph_node"])
            pt_time_min = float(distance_map.get(home_node, np.inf))
            nearest_service_node = source_map.get(home_node)
            service_row = svc_by_node.loc[nearest_service_node] if nearest_service_node in svc_by_node.index else None
            distance_service_to_graph_node_m = (
                float(service_row["distance_service_to_graph_node_m"]) if service_row is not None else np.nan
            )
            path_components = _decompose_pt_path(
                graph,
                path_map.get(home_node, [home_node]),
                distance_home_to_graph_node_m=float(home_row["distance_home_to_graph_node_m"]),
                distance_service_to_graph_node_m=distance_service_to_graph_node_m,
            )
            city_rows.append(
                {
                    "building_idx": int(home_row["building_idx"]),
                    "service_name": service_name,
                    "pt_time_min": pt_time_min,
                    "distance_home_to_graph_node_m": float(home_row["distance_home_to_graph_node_m"]),
                    "home_graph_node": home_node,
                    "nearest_service_graph_node": int(nearest_service_node) if nearest_service_node is not None else None,
                    "distance_service_to_graph_node_m": distance_service_to_graph_node_m,
                    "nearest_service_idx": int(service_row["service_idx"]) if service_row is not None else None,
                    "nearest_service_source_uid": service_row.get("source_uid") if service_row is not None else None,
                    "nearest_service_name": service_row.get("name") if service_row is not None else None,
                    "home_street_pattern_class": home_row.get("top1_class_name"),
                    "service_street_pattern_class": service_row.get("top1_class_name") if service_row is not None else None,
                    "walk_edge_time_min": path_components["walk_edge_time_min"],
                    "transport_time_min": path_components["transport_time_min"],
                    "other_edge_time_min": path_components["other_edge_time_min"],
                    "home_snap_walk_time_min": path_components["home_snap_walk_time_min"],
                    "service_snap_walk_time_min": path_components["service_snap_walk_time_min"],
                    "access_walk_edge_time_min": path_components["access_walk_edge_time_min"],
                    "egress_walk_edge_time_min": path_components["egress_walk_edge_time_min"],
                    "transfer_walk_edge_time_min": path_components["transfer_walk_edge_time_min"],
                    "access_walk_time_min": path_components["access_walk_time_min"],
                    "egress_walk_time_min": path_components["egress_walk_time_min"],
                    "transfer_time_min": path_components["transfer_time_min"],
                    "access_egress_walk_time_min": path_components["access_egress_walk_time_min"],
                    "has_transport_segment": bool(path_components["has_transport_segment"]),
                    "pt_total_decomposed_time_min": path_components["total_time_min"],
                    "walk_filter_min": float(min_walk_min) if min_walk_min is not None else None,
                    "walk_filter_max_exclusive": (
                        float(max_walk_min_exclusive) if max_walk_min_exclusive is not None else None
                    ),
                }
            )

        service_rows = [row for row in city_rows if row["service_name"] == service_name]
        service_df = pd.DataFrame(service_rows)
        pt_series = pd.to_numeric(service_df.get("pt_time_min"), errors="coerce").replace([np.inf, -np.inf], np.nan)
        access_series = pd.to_numeric(service_df.get("access_egress_walk_time_min"), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        transport_series = pd.to_numeric(service_df.get("transport_time_min"), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        access_series_split = pd.to_numeric(service_df.get("access_walk_time_min"), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        egress_series_split = pd.to_numeric(service_df.get("egress_walk_time_min"), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        transfer_series = pd.to_numeric(service_df.get("transfer_time_min"), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        summaries.append(
            {
                "city": city,
                "service_name": service_name,
                "service_points": int(len(svc)),
                "eligible_homes": int(len(eligible_homes)),
                "rows_out": int(len(eligible_homes)),
                "reachable_home_count": int(pt_series.notna().sum()),
                "mean_pt_time_min_reachable": float(pt_series.mean()) if pt_series.notna().any() else None,
                "median_pt_time_min_reachable": float(pt_series.median()) if pt_series.notna().any() else None,
                "mean_access_egress_walk_time_min_reachable": (
                    float(access_series[pt_series.notna()].mean()) if pt_series.notna().any() else None
                ),
                "median_access_egress_walk_time_min_reachable": (
                    float(access_series[pt_series.notna()].median()) if pt_series.notna().any() else None
                ),
                "mean_transport_time_min_reachable": (
                    float(transport_series[pt_series.notna()].mean()) if pt_series.notna().any() else None
                ),
                "median_transport_time_min_reachable": (
                    float(transport_series[pt_series.notna()].median()) if pt_series.notna().any() else None
                ),
                "mean_access_walk_time_min_reachable": (
                    float(access_series_split[pt_series.notna()].mean()) if pt_series.notna().any() else None
                ),
                "median_access_walk_time_min_reachable": (
                    float(access_series_split[pt_series.notna()].median()) if pt_series.notna().any() else None
                ),
                "mean_egress_walk_time_min_reachable": (
                    float(egress_series_split[pt_series.notna()].mean()) if pt_series.notna().any() else None
                ),
                "median_egress_walk_time_min_reachable": (
                    float(egress_series_split[pt_series.notna()].median()) if pt_series.notna().any() else None
                ),
                "mean_transfer_time_min_reachable": (
                    float(transfer_series[pt_series.notna()].mean()) if pt_series.notna().any() else None
                ),
                "median_transfer_time_min_reachable": (
                    float(transfer_series[pt_series.notna()].median()) if pt_series.notna().any() else None
                ),
                "walk_filter_min": float(min_walk_min) if min_walk_min is not None else None,
                "walk_filter_max_exclusive": (
                    float(max_walk_min_exclusive) if max_walk_min_exclusive is not None else None
                ),
                "status": "ok",
            }
        )

    out_city = out_root / city
    out_city.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(city_rows)
    if not out_df.empty:
        out_df.to_parquet(out_city / "residential_to_services_pt_top1.parquet", index=False)
    for summary in summaries:
        (out_city / f"{summary['service_name']}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joint-inputs-root",
        type=Path,
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"),
    )
    parser.add_argument(
        "--walk-root",
        type=Path,
        default=Path("/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_top1"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_pt_top1_walk15plus"
        ),
    )
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--services", nargs="*", default=DEFAULT_SERVICES)
    parser.add_argument("--min-walk-min", type=float, default=DEFAULT_MIN_WALK_MIN)
    parser.add_argument("--max-walk-min-exclusive", type=float, default=None)
    args = parser.parse_args()

    city_dirs = _city_dirs(args.joint_inputs_root)
    if args.cities:
        wanted = set(args.cities)
        city_dirs = [c for c in city_dirs if c.name in wanted]

    results: list[dict] = []
    for city_dir in tqdm(city_dirs, desc="Cities", unit="city"):
        try:
            city_results = _compute_city(
                city_dir,
                args.out_root,
                args.walk_root,
                list(args.services),
                float(args.min_walk_min) if args.min_walk_min is not None else None,
                float(args.max_walk_min_exclusive) if args.max_walk_min_exclusive is not None else None,
            )
        except Exception as exc:  # pragma: no cover
            city_results = [{"city": city_dir.name, "service_name": "*", "status": "error", "error": str(exc)}]
        for res in city_results:
            print(f"{res['city']} {res['service_name']}: {res['status']}")
            results.append(res)

    args.out_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out_root / "_run_report.tsv", sep="\t", index=False)
    (args.out_root / "_run_report.json").write_text(
        df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

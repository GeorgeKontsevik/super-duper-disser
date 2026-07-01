#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd


def build_city(city_root: Path, thermal_root: Path, out_root: Path, city: str) -> dict:
    city_dir = city_root / city
    heat_city_root = out_root / city
    out_city = heat_city_root / "intermodal_graph_iduedu"
    out_city.mkdir(parents=True, exist_ok=True)

    with (city_dir / "intermodal_graph_iduedu/graph.pkl").open("rb") as fh:
        graph = pickle.load(fh)

    walk = gpd.read_parquet(thermal_root / city / "tables/pedestrian_links_utci.parquet")
    walk["u"] = walk["u"].astype(int)
    walk["v"] = walk["v"].astype(int)
    walk["length_meter"] = pd.to_numeric(walk["length_meter"], errors="coerce")
    walk["time_min"] = pd.to_numeric(walk["time_min"], errors="coerce")
    walk["cost_factor"] = pd.to_numeric(walk["cost_factor"], errors="coerce").fillna(1.0)

    pair_factor = {}
    for row in walk.itertuples(index=False):
        key = tuple(sorted((int(row.u), int(row.v))))
        prev = pair_factor.get(key)
        cand = (float(row.length_meter), float(row.time_min), float(row.cost_factor))
        if prev is None or cand[0] > prev[0]:
            pair_factor[key] = cand

    updated = 0
    missing = 0
    for u, v, _k, d in graph.edges(keys=True, data=True):
        if str(d.get("type", "")).lower() != "walk":
            continue
        key = tuple(sorted((int(u), int(v))))
        match = pair_factor.get(key)
        if match is None:
            missing += 1
            continue
        _len, base_time, factor = match
        d["time_min"] = float(base_time) * float(factor)
        d["thermal_cost_factor"] = float(factor)
        updated += 1

    with (out_city / "graph.pkl").open("wb") as fh:
        pickle.dump(graph, fh)
    for rel in ["graph_nodes.parquet", "graph_edges.parquet", "boundary.parquet", "manifest.json"]:
        src = city_dir / "intermodal_graph_iduedu" / rel
        if src.exists():
            shutil.copy2(src, out_city / rel)
    for rel in [
        "derived_layers",
        "pipeline_2",
        "street_pattern",
        "blocksnet",
        "analysis_territory",
        "blocksnet_raw_osm",
        "connectpt_osm",
    ]:
        src = city_dir / rel
        dst = heat_city_root / rel
        if not src.exists() or dst.exists():
            continue
        try:
            dst.symlink_to(src.resolve(), target_is_directory=True)
        except Exception:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    return {"city": city, "updated_walk_edges": updated, "missing_walk_edges": missing, "walk_pairs": len(pair_factor)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city-root", type=Path, required=True)
    parser.add_argument("--thermal-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()

    cities = args.cities or sorted([p.name for p in args.city_root.iterdir() if p.is_dir() and not p.name.startswith("_")])
    for city in cities:
        print(build_city(args.city_root, args.thermal_root, args.out_root, city))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import geopandas as gpd
import iduedu
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata
LOG_FORMAT = (
    "<green>{time:DD MMM HH:mm}</green> | "
    "<level>{level: <7}</level> | "
    "<magenta>{extra[tag]}</magenta> "
    "{message}"
)

CONNECTPT_COMPAT_EXTRA_STOP_TAGS = {
    "bus": [{"highway": "bus_stop"}],
    "tram": [{"railway": "tram_stop"}],
    "trolleybus": [{"highway": "bus_stop", "trolleybus": "yes"}],
}


def slugify_place(place: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")
    return slug or "place"


def _save_geodata(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        prepare_geodata_for_parquet(gdf).to_parquet(path)
    elif path.suffix.lower() == ".gpkg":
        gdf.to_file(path, driver="GPKG")
    else:
        gdf.to_file(path, driver="GeoJSON")


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached intermodal graph bundle with isolated iduedu runtime.")
    parser.add_argument("--place", required=True)
    parser.add_argument("--boundary-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _configure_logging() -> None:
    logger.remove()
    logger.configure(patcher=lambda record: record["extra"].setdefault("tag", "[log]"))
    logger.add(
        sys.stderr,
        level="INFO",
        format=LOG_FORMAT,
        colorize=True,
    )
    try:
        iduedu.config.logger.configure(patcher=lambda record: record["extra"].setdefault("tag", "[iduedu]"))
        iduedu.config.logger.remove()
        iduedu.config.logger.add(
            sys.stderr,
            level="INFO",
            format=LOG_FORMAT,
            colorize=True,
            backtrace=True,
            diagnose=False,
        )
        iduedu.config.set_enable_tqdm(False)
    except Exception:
        pass


def _log(message: str) -> None:
    logger.bind(tag="[intermodal-builder]").info(message)


def main() -> None:
    _configure_logging()
    args = parse_args()
    place = str(args.place)
    boundary_path = Path(args.boundary_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    territory_gdf = read_geodata(boundary_path)
    if territory_gdf.empty:
        raise ValueError(f"Intermodal graph territory is empty: {boundary_path}")

    transport_types = ["tram", "bus", "trolleybus", "subway"]
    _log(
        "Building intermodal graph with iduedu 1.2.1 "
        f"for modes={','.join(transport_types)} (clip_by_territory=True)"
    )
    graph = iduedu.get_intermodal_graph(
        territory=territory_gdf,
        clip_by_territory=True,
        pt_kwargs={
            "transport_types": transport_types,
            "extra_stop_tags": CONNECTPT_COMPAT_EXTRA_STOP_TAGS,
        },
    )

    nodes_gdf = iduedu.graph_to_gdf(graph, nodes=True, edges=False)
    edges_gdf = iduedu.graph_to_gdf(graph, nodes=False, edges=True, restore_edge_geom=True)

    graph_pickle_path = output_dir / "graph.pkl"
    nodes_path = output_dir / "graph_nodes.parquet"
    edges_path = output_dir / "graph_edges.parquet"
    boundary_copy_path = output_dir / "boundary.parquet"

    _save_pickle(graph, graph_pickle_path)
    _save_geodata(nodes_gdf.reset_index(drop=False), nodes_path)
    _save_geodata(edges_gdf.reset_index(drop=False), edges_path)
    _save_geodata(territory_gdf, boundary_copy_path)

    manifest = {
        "place": place,
        "slug": slugify_place(place),
        "graph_type": "intermodal",
        "graph_provider": "iduedu",
        "iduedu_version": "1.2.1",
        "clip_by_territory": True,
        "transport_types": transport_types,
        "extra_stop_tags": CONNECTPT_COMPAT_EXTRA_STOP_TAGS,
        "extra_stop_stats": graph.graph.get("extra_stop_stats", {}),
        "boundary_source": str(boundary_path),
        "files": {
            "boundary": str(boundary_copy_path),
            "graph_pickle": str(graph_pickle_path),
            "graph_nodes": str(nodes_path),
            "graph_edges": str(edges_path),
        },
        "stats": {
            "node_count": int(graph.number_of_nodes()),
            "edge_count": int(graph.number_of_edges()),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _log(f"Bundle ready: {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm

from run_street_pattern_canada_centres import (
    EXPERIMENTS_DIR,
    REPO_ROOT,
    BlockDataset,
    _cache_prefix,
    _load_pickle,
    _save_pickle,
    build_city_prediction_gdf,
    classify_blocks,
    class_names,
    prepare_city_roads,
    project_cached_city_roads,
    resolve_city_centre_node,
    resolve_model_path,
    save_city_outputs,
    split_roads_by_grid_for_polygon,
    summarize_city,
)


TOP_30_US_CITIES_2020 = [
    "New York",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "Jacksonville",
    "Austin",
    "Fort Worth",
    "San Jose",
    "Columbus",
    "Charlotte",
    "Indianapolis",
    "San Francisco",
    "Seattle",
    "Denver",
    "Oklahoma City",
    "Nashville",
    "El Paso",
    "Washington",
    "Las Vegas",
    "Boston",
    "Detroit",
    "Portland",
    "Louisville",
    "Memphis",
    "Baltimore",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the street-pattern classifier on local U.S. road data, "
            "restricted to buffers around city centres resolved from OSM."
        )
    )
    parser.add_argument(
        "--roads",
        default=str(
            REPO_ROOT
            / "data_all_cities"
            / "hotosm_usa_roads_lines_gpkg"
            / "hotosm_usa_roads_lines_gpkg.gpkg"
        ),
        help="Path to the U.S. roads GeoPackage.",
    )
    parser.add_argument(
        "--places",
        nargs="+",
        help='City names to process, for example "Chicago" "Houston".',
    )
    parser.add_argument(
        "--top-30-usa",
        action="store_true",
        help=(
            "Process 30 large U.S. cities using a fixed 2020 Census-era ranking: "
            + ", ".join(TOP_30_US_CITIES_2020)
            + "."
        ),
    )
    parser.add_argument(
        "--place-suffix",
        default="USA",
        help='Suffix appended to every place when geocoding, for example "USA".',
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=20_000,
        help="Buffer radius around each city centre, in meters.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=2_000,
        help="Grid cell size in projected CRS units.",
    )
    parser.add_argument(
        "--min-road-count",
        type=int,
        default=5,
        help="Skip grid cells with fewer clipped road segments than this.",
    )
    parser.add_argument(
        "--min-total-road-length",
        type=float,
        default=500.0,
        help="Skip grid cells whose total clipped road length is below this threshold.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for subgraph preparation.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device. Use "cpu" by default.',
    )
    parser.add_argument(
        "--model-path",
        default=str(EXPERIMENTS_DIR / "models" / "best_model.pth"),
        help="Local path to the classifier weights. Download is used only if this file is missing.",
    )
    parser.add_argument(
        "--output",
        default=str(EXPERIMENTS_DIR / "outputs" / "usa_city_centre_predictions.json"),
        help="Where to save the prediction summary JSON.",
    )
    parser.add_argument(
        "--geo-output",
        default=str(EXPERIMENTS_DIR / "outputs" / "usa_city_centre_predictions.geojson"),
        help="Where to save predicted grid cells as GeoJSON.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(EXPERIMENTS_DIR / "outputs" / "cache" / "usa_city_centre"),
        help="Directory for per-city pickle caches of expensive intermediate results.",
    )
    parser.add_argument(
        "--per-city-dir",
        default=str(EXPERIMENTS_DIR / "outputs" / "usa_city_centre" / "cities"),
        help="Directory for per-city outputs such as GeoJSON, CSV, and JSON summaries.",
    )
    parser.add_argument(
        "--maps-dir",
        default=str(EXPERIMENTS_DIR / "outputs" / "usa_city_centre" / "maps"),
        help="Directory for rendered per-city PNG maps.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing intermediate pickle caches.",
    )
    args = parser.parse_args()
    if not args.places and not args.top_30_usa:
        parser.error("Provide --places or use --top-30-usa.")
    return args


def _resolve_places(args: argparse.Namespace) -> list[str]:
    if args.top_30_usa:
        return TOP_30_US_CITIES_2020
    return args.places


def main() -> None:
    args = parse_args()
    places = _resolve_places(args)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geo_output_path = Path(args.geo_output).resolve()
    geo_output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve()
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    per_city_dir = Path(args.per_city_dir).resolve()
    per_city_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = Path(args.maps_dir).resolve()
    maps_dir.mkdir(parents=True, exist_ok=True)
    models_dir = EXPERIMENTS_DIR / "models"
    model_path = Path(args.model_path).resolve()

    roads_path = Path(args.roads).resolve()
    if not roads_path.exists():
        raise FileNotFoundError(
            f"U.S. road GeoPackage not found: {roads_path}. Pass --roads with your local USA roads file."
        )

    stage_bar = tqdm(total=2 + len(places), desc="USA street-pattern pipeline", unit="stage")

    stage_bar.set_postfix_str("resolve model")
    model_path = resolve_model_path(model_path=model_path, models_dir=models_dir)
    stage_bar.update(1)

    stage_bar.set_postfix_str("ready")
    stage_bar.update(1)

    city_results = []
    failures = []
    city_prediction_gdfs = []

    for raw_place in places:
        place_query = raw_place if args.place_suffix.strip() == "" else f"{raw_place}, {args.place_suffix}"
        stage_bar.set_postfix_str(f"process {raw_place}")

        try:
            relation, centre_node = resolve_city_centre_node(place_query)
            cache_prefix = _cache_prefix(
                cache_dir=cache_dir,
                place_query=place_query,
                buffer_m=args.buffer_m,
                grid_step=args.grid_step,
                min_road_count=args.min_road_count,
                min_total_road_length=args.min_total_road_length,
            )
            roads_cache_path = cache_prefix.with_name(cache_prefix.name + "__roads.pkl")
            subgraphs_cache_path = cache_prefix.with_name(cache_prefix.name + "__subgraphs.pkl")
            dataset_cache_path = cache_prefix.with_name(cache_prefix.name + "__dataset.pkl")

            if not args.no_cache and roads_cache_path.exists():
                roads_cache = _load_pickle(roads_cache_path)
                required_cache_keys = {"roads_wgs84", "buffer_polygon_wgs84"}
                if required_cache_keys.issubset(roads_cache):
                    roads_wgs84 = roads_cache["roads_wgs84"]
                    buffer_polygon_wgs84 = roads_cache["buffer_polygon_wgs84"]
                    roads_projected, polygon_projected = project_cached_city_roads(
                        roads_wgs84=roads_wgs84,
                        buffer_polygon_wgs84=buffer_polygon_wgs84,
                    )
                else:
                    roads_projected, roads_wgs84, buffer_polygon_wgs84, polygon_projected = prepare_city_roads(
                        roads_path=roads_path,
                        centre_node=centre_node,
                        buffer_m=args.buffer_m,
                    )
                    if not args.no_cache:
                        _save_pickle(
                            roads_cache_path,
                            {
                                "roads_wgs84": roads_wgs84,
                                "buffer_polygon_wgs84": buffer_polygon_wgs84,
                            },
                        )
            else:
                roads_projected, roads_wgs84, buffer_polygon_wgs84, polygon_projected = prepare_city_roads(
                    roads_path=roads_path,
                    centre_node=centre_node,
                    buffer_m=args.buffer_m,
                )
                if not args.no_cache:
                    _save_pickle(
                        roads_cache_path,
                        {
                            "roads_wgs84": roads_wgs84,
                            "buffer_polygon_wgs84": buffer_polygon_wgs84,
                        },
                    )

            if not args.no_cache and subgraphs_cache_path.exists():
                subgraphs = _load_pickle(subgraphs_cache_path)
            else:
                subgraphs = split_roads_by_grid_for_polygon(
                    roads_projected,
                    polygon_projected,
                    grid_step=args.grid_step,
                    min_road_count=args.min_road_count,
                    min_total_road_length=args.min_total_road_length,
                )
                if not subgraphs:
                    raise ValueError("No non-empty subgraphs were produced.")
                if not args.no_cache:
                    _save_pickle(subgraphs_cache_path, subgraphs)

            if not subgraphs:
                raise ValueError("No non-empty subgraphs were produced.")

            if not args.no_cache and dataset_cache_path.exists():
                dataset = _load_pickle(dataset_cache_path)
            else:
                dataset = BlockDataset(subgraphs, workers=args.workers)
                if len(dataset) == 0:
                    raise ValueError("Dataset contains no valid subgraphs.")
                if not args.no_cache:
                    _save_pickle(dataset_cache_path, dataset)

            if len(dataset) == 0:
                raise ValueError("Dataset contains no valid subgraphs.")

            predictions, probabilities = classify_blocks(
                dataset,
                model_path=model_path,
                device=args.device,
            )

            city_summary = summarize_city(
                place_query=place_query,
                relation=relation,
                centre_node=centre_node,
                subgraphs=subgraphs,
                predictions=predictions,
                probabilities=probabilities,
                buffer_m=args.buffer_m,
                grid_step=args.grid_step,
            )
            city_results.append(city_summary)
            city_prediction_gdf = build_city_prediction_gdf(
                place_query=place_query,
                relation=relation,
                centre_node=centre_node,
                subgraphs=subgraphs,
                predictions=predictions,
                probabilities=probabilities,
            )
            if not city_prediction_gdf.empty:
                city_prediction_gdfs.append(city_prediction_gdf)
            save_city_outputs(
                city_dir=per_city_dir / raw_place.lower().replace(" ", "_"),
                maps_dir=maps_dir,
                raw_place=raw_place,
                place_query=place_query,
                relation=relation,
                centre_node=centre_node,
                city_summary=city_summary,
                roads_wgs84=roads_wgs84,
                buffer_polygon_wgs84=buffer_polygon_wgs84,
                prediction_gdf=city_prediction_gdf,
            )
        except Exception as exc:
            failures.append({"place": place_query, "error": str(exc)})
        finally:
            stage_bar.update(1)

    stage_bar.close()

    summary = {
        "roads": str(roads_path),
        "device": args.device,
        "buffer_m": args.buffer_m,
        "grid_step": args.grid_step,
        "min_road_count": args.min_road_count,
        "min_total_road_length": args.min_total_road_length,
        "workers": args.workers,
        "model_path": str(model_path),
        "cache_dir": str(cache_dir),
        "cache_enabled": not args.no_cache,
        "per_city_dir": str(per_city_dir),
        "maps_dir": str(maps_dir),
        "places_requested": places,
        "class_names": class_names,
        "cities_processed": len(city_results),
        "cities_failed": len(failures),
        "results": city_results,
        "failures": failures,
    }

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if city_prediction_gdfs:
        prediction_gdf = gpd.GeoDataFrame(
            pd.concat(city_prediction_gdfs, ignore_index=True),
            geometry="geometry",
            crs=city_prediction_gdfs[0].crs,
        )
        prediction_gdf.to_file(geo_output_path, driver="GeoJSON")
        print(f"Saved predicted cells to {geo_output_path}")
    print(f"Saved summary to {output_path}")
    print(f"Processed cities: {len(city_results)}")
    print(f"Failed cities: {len(failures)}")


if __name__ == "__main__":
    main()

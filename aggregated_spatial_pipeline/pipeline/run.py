from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregated_spatial_pipeline.spec import CONFIG_DIR, PipelineSpec

from .crosswalks import build_crosswalk, save_crosswalk
from .io import load_layer, save_layer
from .scenarios import run_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the aggregated spatial pipeline without modifying upstream modules."
    )
    parser.add_argument("--quarters", required=True, help="Path to quarter polygons.")
    parser.add_argument("--street-grid", required=True, help="Path to street-grid polygons and morphology attributes.")
    parser.add_argument("--climate-grid", required=True, help="Path to climate-grid polygons and environmental attributes.")
    parser.add_argument("--cities", required=True, help="Path to city polygons.")
    parser.add_argument(
        "--spec-dir",
        default=str(CONFIG_DIR),
        help="Directory containing aggregated spatial pipeline JSON specs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where crosswalks, harmonized layers, and scenario outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = PipelineSpec.load(Path(args.spec_dir))
    issues = spec.validate()
    if issues:
        raise SystemExit("Invalid pipeline spec:\n- " + "\n- ".join(issues))

    layers = {
        "quarters": load_layer(Path(args.quarters), "quarters"),
        "street_grid": load_layer(Path(args.street_grid), "street_grid"),
        "climate_grid": load_layer(Path(args.climate_grid), "climate_grid"),
        "cities": load_layer(Path(args.cities), "cities"),
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    crosswalks = {}
    crosswalks_path = output_dir / "crosswalks.gpkg"
    for crosswalk in spec.crosswalks["crosswalks"]:
        crosswalk_gdf = build_crosswalk(
            source_gdf=layers[crosswalk["source_layer"]],
            target_gdf=layers[crosswalk["target_layer"]],
            source_layer=crosswalk["source_layer"],
            target_layer=crosswalk["target_layer"],
        )
        crosswalks[crosswalk["crosswalk_id"]] = crosswalk_gdf
        save_crosswalk(crosswalk_gdf, crosswalks_path, crosswalk["crosswalk_id"])

    scenario_results = run_scenarios(spec=spec, layers=layers, crosswalks=crosswalks)

    manifest = {
        "layers": {layer_id: str(path) for layer_id, path in {
            "quarters": args.quarters,
            "street_grid": args.street_grid,
            "climate_grid": args.climate_grid,
            "cities": args.cities,
        }.items()},
        "crosswalk_layers": list(crosswalks.keys()),
        "scenarios": {},
    }

    for scenario_id, result in scenario_results.items():
        scenario_dir = output_dir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        save_layer(result.quarters, scenario_dir / "quarters.geojson")
        save_layer(result.cities, scenario_dir / "cities.geojson")
        (scenario_dir / "metadata.json").write_text(
            json.dumps(result.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["scenarios"][scenario_id] = {
            "quarters": str(scenario_dir / "quarters.geojson"),
            "cities": str(scenario_dir / "cities.geojson"),
            "metadata": str(scenario_dir / "metadata.json"),
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved pipeline outputs to {output_dir}")


if __name__ == "__main__":
    main()

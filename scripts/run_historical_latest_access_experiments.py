from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOINT_INPUT_ROOT = (
    ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "batch_runs"
    / "population_band_15m_5m_full_access_20260530"
    / "joint_inputs"
)
DEFAULT_DOWNLOAD_ROOT = (
    ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "historical_osm_latest_access_downloads_2000_2025"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "historical_osm_latest_access_experiments_2000_2025"
)
DEFAULT_CITIES = [
    "perth_western_australia_australia",
    "taipei_taipei_taiwan",
    "bucharest_bucure_ti_romania",
    "barcelona_catalonia_spain",
    "cincinnati_ohio_united_states",
]
DEFAULT_YEARS = [2025, 2020, 2015, 2010, 2005, 2000]
DEFAULT_DOWNLOAD_LAYERS = [
    "roads",
    "buildings",
    "services_pipeline2_raw",
    "land_use",
    "amenities_floor_context",
]
DEFAULT_SERVICES = ["hospital", "polyclinic", "school", "kindergarten"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the historical OSM pipeline for the latest residential/service/PT access experiments. "
            "Default downloads only the layers needed for street pattern, service access, and "
            "is_living imputation; it does not download water/railways."
        )
    )
    parser.add_argument("--joint-input-root", default=str(DEFAULT_JOINT_INPUT_ROOT))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument("--download-root", default=str(DEFAULT_DOWNLOAD_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--services", nargs="+", default=list(DEFAULT_SERVICES))
    parser.add_argument("--boundary-source", choices=("analysis-buffer", "blocksnet-boundary"), default="analysis-buffer")
    parser.add_argument("--osm-timeout-s", type=int, default=900)
    parser.add_argument("--sleep-s", type=float, default=1.0)
    parser.add_argument("--overpass-url", default=None)
    parser.add_argument("--street-grid-step", type=float, default=500.0)
    parser.add_argument("--street-min-road-count", type=int, default=5)
    parser.add_argument("--street-min-total-road-length", type=float, default=500.0)
    parser.add_argument(
        "--no-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Redownload/recompute every city-year. Use --no-no-cache to allow cached artifacts.",
    )
    parser.add_argument(
        "--download-raw-pt",
        action="store_true",
        help=(
            "Also download raw OSM pt_stops/pt_routes snapshots for diagnostics. "
            "Latest access experiments do not need this; they build PT through iduedu."
        ),
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--skip-experiment-stages",
        action="store_true",
        help="Prepare city-year bundles only; do not run final access experiment scripts.",
    )
    return parser.parse_args()


def _resolve_city_dirs(joint_input_root: Path, city_names: list[str]) -> list[Path]:
    city_dirs: list[Path] = []
    missing: list[Path] = []
    for city in city_names:
        city_dir = joint_input_root / city
        if city_dir.exists():
            city_dirs.append(city_dir.resolve())
        else:
            missing.append(city_dir)
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing city input directories:\n{missing_text}")
    return city_dirs


def _run_command(command: list[str], *, mplconfig_name: str) -> float:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT)
    env.setdefault("MPLCONFIGDIR", f"/tmp/{mplconfig_name}")
    started = time.time()
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)
    return round(time.time() - started, 1)


def _download_command(args: argparse.Namespace, city_dirs: list[Path], years: list[int]) -> list[str]:
    layers = list(DEFAULT_DOWNLOAD_LAYERS)
    if args.download_raw_pt:
        layers.extend(["pt_stops", "pt_routes"])

    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_historical_osm_download",
        "--city-dirs",
        *[str(path) for path in city_dirs],
        "--output-root",
        str(Path(args.download_root).resolve()),
        "--years",
        *[str(year) for year in years],
        "--boundary-source",
        str(args.boundary_source),
        "--osm-timeout-s",
        str(int(args.osm_timeout_s)),
        "--sleep-s",
        str(float(args.sleep_s)),
        "--layers",
        *layers,
    ]
    if args.overpass_url:
        command.extend(["--overpass-url", str(args.overpass_url)])
    if args.no_cache:
        command.append("--no-cache")
    return command


def _analysis_command(args: argparse.Namespace, cities: list[str], years: list[int]) -> list[str]:
    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_historical_osm_access_experiments",
        "--historical-root",
        str(Path(args.download_root).resolve()),
        "--city-names",
        *cities,
        "--years",
        *[str(year) for year in years],
        "--output-root",
        str(Path(args.output_root).resolve()),
        "--services",
        *list(args.services),
        "--street-grid-step",
        str(float(args.street_grid_step)),
        "--street-min-road-count",
        str(int(args.street_min_road_count)),
        "--street-min-total-road-length",
        str(float(args.street_min_total_road_length)),
        "--osm-timeout-s",
        str(int(args.osm_timeout_s)),
    ]
    if args.overpass_url:
        command.extend(["--overpass-url", str(args.overpass_url)])
    if args.no_cache:
        command.append("--no-cache")
    if args.skip_experiment_stages:
        command.append("--skip-experiment-stages")
    return command


def main() -> None:
    args = parse_args()
    if args.download_only and args.analysis_only:
        raise ValueError("Use only one of --download-only or --analysis-only.")

    joint_input_root = Path(args.joint_input_root).resolve()
    download_root = Path(args.download_root).resolve()
    output_root = Path(args.output_root).resolve()
    years = list(dict.fromkeys(int(year) for year in args.years))
    cities = list(args.cities)
    city_dirs = _resolve_city_dirs(joint_input_root, cities)

    manifest = {
        "mode": "historical_latest_access_experiments",
        "joint_input_root": str(joint_input_root),
        "download_root": str(download_root),
        "output_root": str(output_root),
        "cities": cities,
        "years": years,
        "download_layers": list(DEFAULT_DOWNLOAD_LAYERS) + (["pt_stops", "pt_routes"] if args.download_raw_pt else []),
        "no_cache": bool(args.no_cache),
        "stages": [],
    }

    if not args.analysis_only:
        download_elapsed = _run_command(
            _download_command(args, city_dirs, years),
            mplconfig_name="mpl-historical-latest-download",
        )
        manifest["stages"].append({"stage": "download", "elapsed_s": download_elapsed, "status": "ok"})

    if not args.download_only:
        analysis_elapsed = _run_command(
            _analysis_command(args, cities, years),
            mplconfig_name="mpl-historical-latest-access",
        )
        manifest["stages"].append({"stage": "analysis", "elapsed_s": analysis_elapsed, "status": "ok"})

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "latest_access_runner_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()

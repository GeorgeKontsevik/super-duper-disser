#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregated_spatial_pipeline.blocksnet_data_pipeline.pipeline import slugify_place  # noqa: E402
from connectpt_dataset_prep.city_sets import TOPUP_CITY_PLACES  # noqa: E402


DEFAULT_MODALITIES = ["bus", "tram", "trolleybus"]
DEFAULT_OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect full joint-input bundles for the top-up city list using run_joint phase 1 only. "
            "This produces the artifacts needed for is_living restoration, street pattern, PT, "
            "services_raw, and downstream accessibility plots."
        )
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-cities", type=int, default=None)
    parser.add_argument("--buffer-m", type=float, default=10_000.0)
    parser.add_argument("--street-grid-step", type=float, default=500.0)
    parser.add_argument("--osm-timeout-s", type=float, default=60.0)
    parser.add_argument("--overpass-url", default=DEFAULT_OVERPASS_URL)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument(
        "--floor-ignore-missing-below-pct",
        type=float,
        default=0.0,
        help="Set to 0 to avoid skipping floor/is_living restoration on low-missing cases.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[topup-full-joint-inputs] {message}", flush=True)


def _resolve_output_root(override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        ROOT
        / "aggregated_spatial_pipeline"
        / "outputs"
        / "batch_runs"
        / f"topup_full_joint_inputs_{stamp}"
    ).resolve()


def _env() -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = [str(ROOT), str(ROOT / "connectpt")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl-topup-full-joint-inputs"))
    return env


def _is_retryable_run_joint_failure(message: str) -> bool:
    text = str(message or "").lower()
    markers = [
        "read timed out",
        "connectionerror",
        "readtimeouterror",
        "httpsconnectionpool",
        "remote end closed connection",
        "temporarily unavailable",
    ]
    return any(marker in text for marker in markers)


def _build_run_joint_command(
    *,
    place: str,
    data_dir: Path,
    output_dir: Path,
    buffer_m: float,
    street_grid_step: float,
    osm_timeout_s: float,
    modalities: list[str],
    floor_ignore_missing_below_pct: float,
    overpass_url: str | None,
    no_cache: bool,
) -> list[str]:
    py = ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(py)
    cmd = [
        str(py),
        "-m",
        "aggregated_spatial_pipeline.pipeline.run_joint",
        "--place",
        place,
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--collect-only",
        "--buffer-m",
        str(float(buffer_m)),
        "--street-grid-step",
        str(float(street_grid_step)),
        "--osm-timeout-s",
        str(float(osm_timeout_s)),
        "--floor-ignore-missing-below-pct",
        str(float(floor_ignore_missing_below_pct)),
        "--modalities",
        *modalities,
    ]
    if overpass_url:
        cmd.extend(["--overpass-url", str(overpass_url)])
    if no_cache:
        cmd.append("--no-cache")
    return cmd


def main() -> None:
    args = parse_args()
    output_root = _resolve_output_root(args.output_root)
    joint_inputs_root = output_root / "joint_inputs"
    joint_root = output_root / "joint"
    summary_path = output_root / "summary.json"
    output_root.mkdir(parents=True, exist_ok=True)
    joint_inputs_root.mkdir(parents=True, exist_ok=True)
    joint_root.mkdir(parents=True, exist_ok=True)

    places = list(TOPUP_CITY_PLACES)
    if args.max_cities is not None:
        places = places[: int(args.max_cities)]

    summary = {
        "output_root": str(output_root),
        "joint_inputs_root": str(joint_inputs_root),
        "joint_root": str(joint_root),
        "places": places,
        "buffer_m": float(args.buffer_m),
        "street_grid_step": float(args.street_grid_step),
        "osm_timeout_s": float(args.osm_timeout_s),
        "overpass_url": str(args.overpass_url) if args.overpass_url else None,
        "modalities": list(args.modalities),
        "floor_ignore_missing_below_pct": float(args.floor_ignore_missing_below_pct),
        "dry_run": bool(args.dry_run),
        "max_retries": int(args.max_retries),
        "results": [],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = _env()
    for index, place in enumerate(places, start=1):
        slug = slugify_place(place)
        city_input_dir = joint_inputs_root / slug
        city_joint_dir = joint_root / slug
        city_input_dir.mkdir(parents=True, exist_ok=True)
        city_joint_dir.mkdir(parents=True, exist_ok=True)
        command = _build_run_joint_command(
            place=place,
            data_dir=city_input_dir,
            output_dir=city_joint_dir,
            buffer_m=float(args.buffer_m),
            street_grid_step=float(args.street_grid_step),
            osm_timeout_s=float(args.osm_timeout_s),
            modalities=list(args.modalities),
            floor_ignore_missing_below_pct=float(args.floor_ignore_missing_below_pct),
            overpass_url=args.overpass_url,
            no_cache=bool(args.no_cache),
        )
        row = {
            "index": int(index),
            "place": place,
            "slug": slug,
            "city_dir": str(city_input_dir),
            "joint_output_dir": str(city_joint_dir),
            "status": "ok",
            "elapsed_s": None,
            "command": command,
        }
        started = time.time()
        _log(f"[{index}/{len(places)}] {place}")
        try:
            if args.dry_run:
                row["status"] = "dry_run"
            else:
                attempts = max(1, int(args.max_retries) + 1)
                for attempt in range(1, attempts + 1):
                    row["attempt"] = int(attempt)
                    try:
                        subprocess.run(command, cwd=str(ROOT), env=env, check=True)
                        break
                    except Exception as exc:  # noqa: BLE001
                        message = str(exc)
                        row["error"] = message
                        if attempt >= attempts or not _is_retryable_run_joint_failure(message):
                            raise
                        _log(f"retry {attempt}/{attempts - 1} for {slug} after retryable network failure")
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["error"] = str(exc)
            _log(f"FAIL {slug}: {exc}")
            if args.fail_fast:
                row["elapsed_s"] = round(time.time() - started, 1)
                summary["results"].append(row)
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                raise
        finally:
            row["elapsed_s"] = round(time.time() - started, 1)
            summary["results"].append(row)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _log(f"{slug}: {row['status']} elapsed={row['elapsed_s']}s")

    _log(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()

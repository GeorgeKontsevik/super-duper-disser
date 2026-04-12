from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger

from aggregated_spatial_pipeline.runtime_config import configure_logger, ensure_repo_mplconfigdir
from aggregated_spatial_pipeline.visualization import save_preview_figure


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT
SUBPROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "experiments_active19_20260412" / "pt_street_pattern_dependency"
)
DEFAULT_CITIES = ("warsaw_poland", "berlin_germany")

ensure_repo_mplconfigdir("mpl-pt-street-pattern-cross-city", root=REPO_ROOT)


def _configure_logging() -> None:
    configure_logger("[pt-x-street-pattern-cross-city]")


def _log(message: str) -> None:
    logger.bind(tag="[pt-x-street-pattern-cross-city]").info(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate PT x street-pattern per-city outputs into pooled cross-city summaries."
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root with per-city pt_street_pattern_dependency outputs.",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=list(DEFAULT_CITIES),
        help="City slugs inside output-root, or 'all'.",
    )
    return parser.parse_args()


def _plot_city_class_heatmap(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    pivot = frame.pivot(index="city", columns="street_pattern_class", values="pt_length_share").fillna(0.0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.45 * len(pivot.index) + 1.5)))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.3, ax=ax, vmin=0.0, vmax=1.0)
    ax.set_title("PT Length Share By Street Pattern And City", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_city_modality_heatmap(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    plot_df = frame.copy()
    plot_df["city_modality"] = plot_df["city"].astype(str) + " | " + plot_df["type"].astype(str)
    pivot = plot_df.pivot(index="city_modality", columns="street_pattern_class", values="modality_class_share").fillna(0.0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.28 * len(pivot.index) + 1.5)))
    sns.heatmap(pivot, cmap="PuBuGn", annot=False, linewidths=0.2, ax=ax, vmin=0.0, vmax=1.0)
    ax.set_title("PT Length Share By Street Pattern, City, And Modality", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_dominant_class_heatmap(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    counts = (
        frame.groupby(["city", "dominant_street_pattern_class"], as_index=False)
        .size()
        .rename(columns={"size": "route_count"})
    )
    totals = counts.groupby("city")["route_count"].transform("sum")
    counts["route_share"] = counts["route_count"] / totals.where(totals > 0, pd.NA)
    pivot = counts.pivot(index="city", columns="dominant_street_pattern_class", values="route_share").fillna(0.0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.45 * len(pivot.index) + 1.5)))
    sns.heatmap(pivot, cmap="mako", annot=True, fmt=".2f", linewidths=0.3, ax=ax, vmin=0.0, vmax=1.0)
    ax.set_title("Share Of Routes Dominated By Each Street Pattern", fontsize=16, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    _configure_logging()
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if not output_root.exists():
        raise FileNotFoundError(f"PT dependency output root not found: {output_root}")

    if list(args.cities) == ["all"]:
        cities = sorted(p.name for p in output_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    else:
        cities = [str(city) for city in args.cities]

    class_summary_frames: list[pd.DataFrame] = []
    class_modality_frames: list[pd.DataFrame] = []
    route_stats_frames: list[pd.DataFrame] = []
    used_cities: list[str] = []
    for city in cities:
        city_dir = output_root / city
        class_summary_path = city_dir / "class_dependency_summary.csv"
        class_modality_path = city_dir / "class_modality_length.csv"
        route_stats_path = city_dir / "route_stats.csv"
        if not class_summary_path.exists() or not class_modality_path.exists() or not route_stats_path.exists():
            _log(f"[{city}] missing per-city ptdep outputs; skipping from cross-city.")
            continue
        class_summary = pd.read_csv(class_summary_path)
        class_summary["city"] = city
        class_modality = pd.read_csv(class_modality_path)
        class_modality["city"] = city
        route_stats = pd.read_csv(route_stats_path)
        route_stats["city"] = city
        class_summary_frames.append(class_summary)
        class_modality_frames.append(class_modality)
        route_stats_frames.append(route_stats)
        used_cities.append(city)

    if len(used_cities) < 2:
        raise ValueError(f"Need at least 2 cities with ptdep outputs for cross-city aggregation, got {len(used_cities)}.")

    cross_dir = output_root / "_cross_city"
    stats_dir = cross_dir / "stats"
    preview_dir = cross_dir / "preview_png"
    stats_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    class_summary_all = pd.concat(class_summary_frames, ignore_index=True)
    class_modality_all = pd.concat(class_modality_frames, ignore_index=True)
    route_stats_all = pd.concat(route_stats_frames, ignore_index=True)

    class_summary_all.to_csv(stats_dir / "class_dependency_summary_all_cities.csv", index=False)
    class_modality_all.to_csv(stats_dir / "class_modality_length_all_cities.csv", index=False)
    route_stats_all.to_csv(stats_dir / "route_stats_all_cities.csv", index=False)

    _plot_city_class_heatmap(class_summary_all, preview_dir / "01_city_class_pt_length_share_heatmap.png")
    _plot_city_modality_heatmap(class_modality_all, preview_dir / "02_city_modality_class_pt_length_share_heatmap.png")
    _plot_dominant_class_heatmap(route_stats_all, preview_dir / "03_city_route_dominant_class_share_heatmap.png")

    summary = {
        "city_count": len(used_cities),
        "cities": used_cities,
        "files": {
            "class_dependency_summary_all_cities": str((stats_dir / "class_dependency_summary_all_cities.csv").resolve()),
            "class_modality_length_all_cities": str((stats_dir / "class_modality_length_all_cities.csv").resolve()),
            "route_stats_all_cities": str((stats_dir / "route_stats_all_cities.csv").resolve()),
            "city_class_pt_length_share_heatmap": str((preview_dir / "01_city_class_pt_length_share_heatmap.png").resolve()),
            "city_modality_class_pt_length_share_heatmap": str((preview_dir / "02_city_modality_class_pt_length_share_heatmap.png").resolve()),
            "city_route_dominant_class_share_heatmap": str((preview_dir / "03_city_route_dominant_class_share_heatmap.png").resolve()),
        },
    }
    (cross_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Done. Cross-city summary: {cross_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

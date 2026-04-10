from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Point


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_FILE = REPO_ROOT / "segregation-by-design-experiments" / "storyline_reference_chunks_12.tsv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "segregation-by-design-experiments"
    / "outputs"
    / "storyline_reference_chunks_atlas_b5000_g500.png"
)
DEFAULT_JOINT_INPUTS_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs_storyline_b5000_g500"


@dataclass
class ChunkSpec:
    slug: str
    place: str
    lat: float
    lon: float
    window_m: float


def _display_name(place: str) -> str:
    return place.split(",")[0].strip().upper()


def _roads_path(root: Path, street_pattern_dir: str, slug: str) -> Path:
    return root / slug / street_pattern_dir / slug / "roads.geojson"


def _read_chunks(path: Path) -> list[ChunkSpec]:
    rows: list[ChunkSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 5:
            continue
        slug, place, lat, lon, window_m = parts[:5]
        rows.append(
            ChunkSpec(
                slug=slug,
                place=place,
                lat=float(lat),
                lon=float(lon),
                window_m=float(window_m),
            )
        )
    return rows


def _plot_chunk(
    ax: plt.Axes,
    *,
    spec: ChunkSpec,
    joint_inputs_root: Path,
    street_pattern_dir: str,
    roads_color: str,
    bg_color: str,
    roads_lw: float,
) -> None:
    roads_path = _roads_path(joint_inputs_root, street_pattern_dir, spec.slug)
    ax.set_facecolor(bg_color)
    if not roads_path.exists():
        ax.text(0.5, 0.5, "missing", color="#ff9f9f", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_title(_display_name(spec.place), fontsize=12, fontweight="bold")
        ax.set_axis_off()
        return

    roads = gpd.read_file(roads_path)
    if roads.empty:
        ax.text(0.5, 0.5, "empty", color="#ffcc80", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_title(_display_name(spec.place), fontsize=12, fontweight="bold")
        ax.set_axis_off()
        return
    if roads.crs is None:
        roads.set_crs(4326, inplace=True)

    local_crs = roads.estimate_utm_crs() or "EPSG:3857"
    roads_local = roads.to_crs(local_crs)
    point_local = gpd.GeoSeries([Point(spec.lon, spec.lat)], crs=4326).to_crs(local_crs).iloc[0]
    half = float(spec.window_m) / 2.0

    xmin, xmax = point_local.x - half, point_local.x + half
    ymin, ymax = point_local.y - half, point_local.y + half
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            2.0 * half,
            2.0 * half,
            facecolor=bg_color,
            edgecolor="none",
            zorder=0,
        )
    )
    roads_local.plot(ax=ax, color=roads_color, linewidth=roads_lw, alpha=1.0, zorder=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(_display_name(spec.place), fontsize=12, fontweight="bold", pad=3)
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render specific reference-like road chunks per city.")
    parser.add_argument("--chunks-file", default=str(DEFAULT_CHUNKS_FILE), help="TSV: slug, place, lat, lon, window_m")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument("--joint-inputs-root", default=str(DEFAULT_JOINT_INPUTS_ROOT), help="Root with city outputs.")
    parser.add_argument("--street-pattern-dir", default="street_pattern_b5000_g500", help="Street-pattern subfolder.")
    parser.add_argument("--ncols", type=int, default=4, help="Atlas columns.")
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI.")
    parser.add_argument("--roads-color", default="#f3f3f3", help="Road line color.")
    parser.add_argument("--bg-color", default="#2a2d31", help="Background color.")
    parser.add_argument("--roads-lw", type=float, default=1.25, help="Road line width.")
    args = parser.parse_args()

    chunks_file = Path(args.chunks_file).resolve()
    output = Path(args.output).resolve()
    joint_inputs_root = Path(args.joint_inputs_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    specs = _read_chunks(chunks_file)
    if not specs:
        raise ValueError(f"No valid rows in {chunks_file}")

    ncols = max(1, int(args.ncols))
    nrows = int(math.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.15 * ncols, 4.15 * nrows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.patch.set_facecolor("#f0f0f0")

    for idx, spec in enumerate(specs):
        _plot_chunk(
            axes_list[idx],
            spec=spec,
            joint_inputs_root=joint_inputs_root,
            street_pattern_dir=str(args.street_pattern_dir),
            roads_color=str(args.roads_color),
            bg_color=str(args.bg_color),
            roads_lw=float(args.roads_lw),
        )

    for ax in axes_list[len(specs):]:
        ax.set_axis_off()

    fig.tight_layout(pad=1.05)
    fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

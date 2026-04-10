from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP_FILE = REPO_ROOT / "segregation-by-design-experiments" / "storyline_street_pattern_places_reference_12.tsv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "segregation-by-design-experiments" / "outputs" / "storyline_reference_roads_atlas_b5000_g500.png"
)
DEFAULT_JOINT_INPUTS_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs_storyline_b5000_g500"


def _read_places(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    return rows


def _display_name(place: str) -> str:
    return place.split(",")[0].strip().upper()


def _roads_path(root: Path, street_pattern_dir: str, slug: str) -> Path:
    return root / slug / street_pattern_dir / slug / "roads.geojson"


def _buffer_path(root: Path, street_pattern_dir: str, slug: str) -> Path:
    return root / slug / street_pattern_dir / slug / "buffer.geojson"


def _plot_city(
    ax: plt.Axes,
    *,
    slug: str,
    place: str,
    joint_inputs_root: Path,
    street_pattern_dir: str,
    roads_color: str,
    bg_color: str,
    roads_lw: float,
) -> None:
    roads_path = _roads_path(joint_inputs_root, street_pattern_dir, slug)
    buffer_path = _buffer_path(joint_inputs_root, street_pattern_dir, slug)
    ax.set_facecolor(bg_color)

    if not roads_path.exists() or not buffer_path.exists():
        ax.text(0.5, 0.5, "missing", color="#ff9f9f", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_title(_display_name(place), fontsize=12, fontweight="bold")
        ax.set_axis_off()
        return

    roads = gpd.read_file(roads_path)
    buffer_gdf = gpd.read_file(buffer_path)
    if roads.empty or buffer_gdf.empty:
        ax.text(0.5, 0.5, "empty", color="#ffcc80", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_title(_display_name(place), fontsize=12, fontweight="bold")
        ax.set_axis_off()
        return

    if buffer_gdf.crs is None:
        buffer_gdf.set_crs(4326, inplace=True)
    if roads.crs is None:
        roads.set_crs(buffer_gdf.crs, inplace=True)

    local_crs = buffer_gdf.estimate_utm_crs() or "EPSG:3857"
    buffer_local = buffer_gdf.to_crs(local_crs)
    roads_local = roads.to_crs(local_crs)

    minx, miny, maxx, maxy = buffer_local.total_bounds
    width = maxx - minx
    height = maxy - miny
    size = max(width, height)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    half = size / 2.0

    roads_local.plot(ax=ax, color=roads_color, linewidth=roads_lw, alpha=1.0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(_display_name(place), fontsize=12, fontweight="bold", pad=3)
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render reference-style black/white roads atlas with fixed 5 km scale.")
    parser.add_argument("--map-file", default=str(DEFAULT_MAP_FILE), help="TSV file with '<slug>\\t<place>' rows.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument("--joint-inputs-root", default=str(DEFAULT_JOINT_INPUTS_ROOT), help="Root folder with city outputs.")
    parser.add_argument("--street-pattern-dir", default="street_pattern_b5000_g500", help="Street-pattern subfolder name.")
    parser.add_argument("--ncols", type=int, default=4, help="Number of columns.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI.")
    parser.add_argument("--roads-color", default="#e8e8e8", help="Road line color.")
    parser.add_argument("--bg-color", default="#25282c", help="Tile background color.")
    parser.add_argument("--roads-lw", type=float, default=1.0, help="Road line width.")
    args = parser.parse_args()

    map_file = Path(args.map_file).resolve()
    output = Path(args.output).resolve()
    joint_inputs_root = Path(args.joint_inputs_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    cities = _read_places(map_file)
    if not cities:
        raise ValueError(f"No valid cities in {map_file}")

    ncols = max(1, int(args.ncols))
    nrows = int(math.ceil(len(cities) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.15 * ncols, 4.15 * nrows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.patch.set_facecolor("#f0f0f0")

    for idx, (slug, place) in enumerate(cities):
        _plot_city(
            axes_list[idx],
            slug=slug,
            place=place,
            joint_inputs_root=joint_inputs_root,
            street_pattern_dir=str(args.street_pattern_dir),
            roads_color=str(args.roads_color),
            bg_color=str(args.bg_color),
            roads_lw=float(args.roads_lw),
        )

    for ax in axes_list[len(cities):]:
        ax.set_axis_off()

    fig.tight_layout(pad=1.05)
    fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

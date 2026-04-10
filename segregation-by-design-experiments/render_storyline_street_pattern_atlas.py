from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP_FILE = REPO_ROOT / "segregation-by-design-experiments" / "storyline_street_pattern_places.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "segregation-by-design-experiments" / "outputs" / "storyline_street_pattern_atlas.png"
JOINT_INPUTS_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs"
FALLBACK_PALETTE = ["#72d6c9", "#9fbce8", "#c9ef8f", "#f6ab8c", "#f7e0a6", "#a79aac"]


def _read_places(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) < 2:
            continue
        rows.append((parts[0], parts[1]))
    return rows


def _cells_path(joint_inputs_root: Path, street_pattern_dir: str, slug: str) -> Path:
    return joint_inputs_root / slug / street_pattern_dir / slug / "predicted_cells.geojson"


def _roads_path(joint_inputs_root: Path, street_pattern_dir: str, slug: str) -> Path:
    return joint_inputs_root / slug / street_pattern_dir / slug / "roads.geojson"


def _summary_path(joint_inputs_root: Path, street_pattern_dir: str, slug: str) -> Path:
    return joint_inputs_root / slug / street_pattern_dir / f"{slug}_summary.json"


def _display_name(place: str) -> str:
    return place.split(",")[0].strip().upper()


def _format_distance_m(value_m: float) -> str:
    if value_m >= 1000.0:
        return f"{value_m / 1000.0:.1f} km"
    return f"{int(round(value_m))} m"


def _infer_title(
    cities: list[tuple[str, str]],
    *,
    joint_inputs_root: Path,
    street_pattern_dir: str,
) -> str:
    buffers: set[float] = set()
    steps: set[float] = set()
    for slug, _ in cities:
        p = _summary_path(joint_inputs_root, street_pattern_dir, slug)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if "buffer_m" in data:
                buffers.add(float(data["buffer_m"]))
            if "grid_step" in data:
                steps.add(float(data["grid_step"]))
        except Exception:
            continue

    if len(buffers) == 1 and len(steps) == 1:
        b = next(iter(buffers))
        s = next(iter(steps))
        return f"Street-Pattern Grid Atlas (buffer {_format_distance_m(b)}, grid {_format_distance_m(s)})"
    if len(buffers) == 1:
        b = next(iter(buffers))
        return f"Street-Pattern Grid Atlas (buffer {_format_distance_m(b)})"
    return "Street-Pattern Grid Atlas"


def _build_class_color_map(
    cities: list[tuple[str, str]],
    *,
    joint_inputs_root: Path,
    street_pattern_dir: str,
) -> dict[str, str]:
    classes: set[str] = set()
    for slug, _ in cities:
        path = _cells_path(joint_inputs_root, street_pattern_dir, slug)
        if not path.exists():
            continue
        gdf = gpd.read_file(path)
        if "class_name" not in gdf.columns:
            continue
        classes.update(str(v) for v in gdf["class_name"].dropna().unique().tolist())
    ordered = sorted(classes)
    return {name: FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)] for idx, name in enumerate(ordered)}


def _plot_city(
    ax: plt.Axes,
    *,
    slug: str,
    place: str,
    class_colors: dict[str, str],
    overlay_uds: bool,
    cell_coloring: str,
    joint_inputs_root: Path,
    street_pattern_dir: str,
) -> None:
    path = _cells_path(joint_inputs_root, street_pattern_dir, slug)
    if not path.exists():
        ax.set_facecolor("#111111")
        ax.text(0.5, 0.5, "missing", color="#ff9f9f", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_title(_display_name(place), fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    gdf = gpd.read_file(path)
    if gdf.empty:
        ax.set_facecolor("#111111")
        ax.text(0.5, 0.5, "empty", color="#ffcc80", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_title(_display_name(place), fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if cell_coloring == "multivariate":
        colors = gdf.get("multivariate_color")
    else:
        colors = None

    if colors is None or colors.isna().all():
        classes = gdf.get("class_name")
        if classes is None:
            colors = "#72d6c9"
        else:
            colors = classes.map(lambda val: class_colors.get(str(val), "#34495e"))

    gdf.plot(ax=ax, color=colors, linewidth=0)
    if overlay_uds:
        roads_path = _roads_path(joint_inputs_root, street_pattern_dir, slug)
        if roads_path.exists():
            roads = gpd.read_file(roads_path)
            if not roads.empty:
                if roads.crs is not None and gdf.crs is not None and str(roads.crs) != str(gdf.crs):
                    roads = roads.to_crs(gdf.crs)
                roads.plot(ax=ax, color="#f3f3f3", linewidth=0.45, alpha=0.92, zorder=5)

    ax.set_facecolor("#1d1f22")
    ax.set_title(_display_name(place), fontsize=10, fontweight="bold")
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one combined street-pattern atlas for storyline cities.")
    parser.add_argument("--map-file", default=str(DEFAULT_MAP_FILE), help="TSV with '<slug>\\t<place>' rows.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument("--ncols", type=int, default=5, help="Number of columns in the atlas grid.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution.")
    parser.add_argument(
        "--joint-inputs-root",
        default=str(JOINT_INPUTS_ROOT),
        help="Root folder with city outputs (for example .../outputs/joint_inputs_storyline_b5000_g500).",
    )
    parser.add_argument(
        "--street-pattern-dir",
        default="street_pattern",
        help="Per-city street-pattern folder name (for example street_pattern or street_pattern_b5000_g500).",
    )
    parser.add_argument(
        "--cell-coloring",
        choices=("multivariate", "top1"),
        default="multivariate",
        help="Cell fill mode: multivariate blend or top1 class color.",
    )
    parser.add_argument("--overlay-uds", action="store_true", help="Overlay roads (UDS) on top of cells.")
    parser.add_argument("--legend-bottom", action="store_true", help="Draw bottom legend with class colors.")
    parser.add_argument("--title", default="", help="Optional custom title. If empty, title is inferred from summaries.")
    args = parser.parse_args()

    map_file = Path(args.map_file).resolve()
    output = Path(args.output).resolve()
    joint_inputs_root = Path(args.joint_inputs_root).resolve()
    street_pattern_dir = str(args.street_pattern_dir)
    output.parent.mkdir(parents=True, exist_ok=True)

    cities = _read_places(map_file)
    if not cities:
        raise ValueError(f"No valid city rows found in {map_file}")
    class_colors = _build_class_color_map(
        cities,
        joint_inputs_root=joint_inputs_root,
        street_pattern_dir=street_pattern_dir,
    )
    title_text = str(args.title).strip() or _infer_title(
        cities,
        joint_inputs_root=joint_inputs_root,
        street_pattern_dir=street_pattern_dir,
    )

    ncols = max(1, int(args.ncols))
    nrows = int(math.ceil(len(cities) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.1 * ncols, 4.1 * nrows))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (slug, place) in enumerate(cities):
        _plot_city(
            axes_list[idx],
            slug=slug,
            place=place,
            class_colors=class_colors,
            overlay_uds=bool(args.overlay_uds),
            cell_coloring=str(args.cell_coloring),
            joint_inputs_root=joint_inputs_root,
            street_pattern_dir=street_pattern_dir,
        )

    for ax in axes_list[len(cities):]:
        ax.set_axis_off()

    fig.suptitle(title_text, fontsize=16, y=0.995)
    if args.legend_bottom and class_colors:
        handles = [Patch(facecolor=color, edgecolor="none", label=name) for name, color in class_colors.items()]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(3, len(handles)),
            frameon=False,
            fontsize=10,
            title="Street-pattern classes",
            title_fontsize=11,
        )
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.97))
    else:
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))

    fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

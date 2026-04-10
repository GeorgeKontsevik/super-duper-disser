from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PKL = REPO_ROOT / "corr_transport_street_pattern" / "gdf_smallest_with_predictions_len_10k.pkl"
DEFAULT_INTERSECTION_REPORT = (
    REPO_ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "joint_inputs_reference_chunks_b10000_x_pkl"
    / "street_pattern_reference_chunks_b10000_x_pkl_intersection_report.csv"
)
DEFAULT_CHUNKS = REPO_ROOT / "segregation-by-design-experiments" / "storyline_reference_chunks_12_b700.tsv"
DEFAULT_JOINT_INPUTS_ROOT = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs_reference_chunks_b10000"
DEFAULT_STREET_PATTERN_DIR = "street_pattern_reference_chunks_b10000"
DEFAULT_OUTPUT = (
    REPO_ROOT / "segregation-by-design-experiments" / "outputs" / "storyline_reference_chunks_pkl_vs_grid_top1.png"
)

FALLBACK_PALETTE = ["#72d6c9", "#9fbce8", "#c9ef8f", "#f6ab8c", "#f7e0a6", "#a79aac"]


def _read_chunks(path: Path) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 5:
            continue
        slug, place, lat, lon, window_m = parts[:5]
        rows[slug] = {
            "place": place,
            "lat": float(lat),
            "lon": float(lon),
            "window_m": float(window_m),
        }
    return rows


def _load_class_info(summary_path: Path) -> tuple[list[str], dict[int, str]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    class_names = [str(v) for v in summary.get("class_names", [])]
    class_by_id = {idx: name for idx, name in enumerate(class_names)}
    return class_names, class_by_id


def _safe_prediction_to_class_name(value: object, class_by_id: dict[int, str]) -> str:
    try:
        idx = int(value)
    except Exception:
        return "Unknown"
    return class_by_id.get(idx, f"Class {idx}")


def _plot_empty(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("#111111")
    ax.text(0.5, 0.5, "empty", color="#ffcc80", ha="center", va="center", fontsize=10, transform=ax.transAxes)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _mix_with_white(color: str, strength: float, *, min_strength: float = 0.16) -> str:
    strength = max(0.0, min(1.0, float(strength)))
    k = min_strength + (1.0 - min_strength) * strength
    r, g, b = to_rgb(color)
    rr = (1.0 - k) + r * k
    gg = (1.0 - k) + g * k
    bb = (1.0 - k) + b * k
    return to_hex((rr, gg, bb))


def _compute_cell_match_ratio(cells: gpd.GeoDataFrame, pkl_city: gpd.GeoDataFrame) -> pd.Series:
    if cells.empty or pkl_city.empty:
        return pd.Series(index=cells.index, dtype=float)
    if "class_name" not in cells.columns:
        return pd.Series(index=cells.index, dtype=float)

    local_crs = cells.estimate_utm_crs() or "EPSG:3857"
    cells_local = cells[["class_name", "geometry"]].copy().to_crs(local_crs)
    pkl_local = pkl_city[["class_name", "geometry"]].copy().to_crs(local_crs).rename(columns={"class_name": "pkl_class"})
    cells_local["cell_idx"] = cells_local.index

    inter = gpd.overlay(
        cells_local[["cell_idx", "class_name", "geometry"]],
        pkl_local[["pkl_class", "geometry"]],
        how="intersection",
    )
    if inter.empty:
        return pd.Series(index=cells.index, dtype=float)

    inter["area"] = inter.geometry.area
    totals = inter.groupby("cell_idx")["area"].sum()
    matches = inter[inter["class_name"].astype(str) == inter["pkl_class"].astype(str)].groupby("cell_idx")["area"].sum()
    ratio = matches.div(totals).fillna(0.0)
    return ratio


def _blend_color_from_probabilities(
    probabilities: list[float],
    *,
    class_names: list[str],
    class_colors: dict[str, str],
) -> str:
    rgb = [0.0, 0.0, 0.0]
    probs = [max(0.0, float(v)) for v in probabilities]
    total = sum(probs)
    if total <= 0:
        return "#808080"
    weights = [p / total for p in probs]
    for idx, weight in enumerate(weights):
        if idx >= len(class_names):
            continue
        class_name = class_names[idx]
        class_color = class_colors.get(class_name, "#808080")
        r, g, b = to_rgb(class_color)
        rgb[0] += weight * r
        rgb[1] += weight * g
        rgb[2] += weight * b
    return to_hex(tuple(rgb))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one comparison figure: PKL street-pattern polygons vs grid classification for intersected cities."
    )
    parser.add_argument("--pkl-path", default=str(DEFAULT_PKL))
    parser.add_argument("--intersection-report", default=str(DEFAULT_INTERSECTION_REPORT))
    parser.add_argument("--chunks-file", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--joint-inputs-root", default=str(DEFAULT_JOINT_INPUTS_ROOT))
    parser.add_argument("--street-pattern-dir", default=DEFAULT_STREET_PATTERN_DIR)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--grid-color-mode",
        choices=("intensity", "solid"),
        default="intensity",
        help="How to color GRID panels: intensity by class-match ratio or original solid colors.",
    )
    parser.add_argument("--title", default="Street Pattern: PKL vs Grid (intersected cities only)")
    args = parser.parse_args()

    pkl_path = Path(args.pkl_path).resolve()
    report_path = Path(args.intersection_report).resolve()
    chunks_file = Path(args.chunks_file).resolve()
    joint_inputs_root = Path(args.joint_inputs_root).resolve()
    street_pattern_dir = str(args.street_pattern_dir)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    chunks = _read_chunks(chunks_file)
    report_df = pd.read_csv(report_path)
    report_df = report_df[(report_df["status"] == "ok") & (report_df["after"] > 0)].copy()
    if report_df.empty:
        raise ValueError("No intersected cities with non-empty coverage in intersection report.")

    city_slugs = report_df["slug"].astype(str).tolist()
    city_places = {slug: str(chunks.get(slug, {}).get("place", slug)) for slug in city_slugs}

    # Load class list once from first available summary.
    class_names: list[str] = []
    class_by_id: dict[int, str] = {}
    for slug in city_slugs:
        summary_path = joint_inputs_root / slug / street_pattern_dir / f"{slug}_summary.json"
        if summary_path.exists():
            class_names, class_by_id = _load_class_info(summary_path)
            if class_names:
                break
    if not class_names:
        class_names = [f"Class {idx}" for idx in range(6)]
        class_by_id = {idx: name for idx, name in enumerate(class_names)}
    class_colors = {name: FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)] for idx, name in enumerate(class_names)}
    class_colors.setdefault("Unknown", "#808080")

    pkl_df = pd.read_pickle(pkl_path)
    if not isinstance(pkl_df, gpd.GeoDataFrame):
        if isinstance(pkl_df, pd.DataFrame) and "geometry" in pkl_df.columns:
            pkl_df = gpd.GeoDataFrame(pkl_df, geometry="geometry", crs=getattr(pkl_df, "crs", None))
        else:
            raise ValueError("PKL does not contain geometry.")
    if pkl_df.crs is None:
        raise ValueError("PKL GeoDataFrame has no CRS.")
    pkl_df = pkl_df.to_crs(4326)
    if "prediction" not in pkl_df.columns:
        raise ValueError("PKL is missing 'prediction' column.")
    pkl_df = pkl_df.copy()
    pkl_df["class_name"] = pkl_df["prediction"].map(lambda x: _safe_prediction_to_class_name(x, class_by_id))
    prob_columns = [f"prob_{idx}" for idx in range(len(class_names))]
    missing_prob_columns = [col for col in prob_columns if col not in pkl_df.columns]
    if missing_prob_columns:
        raise ValueError(f"PKL is missing required probability columns: {missing_prob_columns}")
    prob_frame = pkl_df[prob_columns]
    invalid_mask = prob_frame.isna().any(axis=1)
    pkl_df["_prob_invalid"] = invalid_mask
    if bool(invalid_mask.any()):
        null_counts = prob_frame.isna().sum()
        bad_cols = {col: int(cnt) for col, cnt in null_counts.items() if int(cnt) > 0}
        bad_rows = int(invalid_mask.sum())
        print(f"WARNING: PKL rows with null probabilities: {bad_rows}; columns={bad_cols}")
    pkl_df["multivariate_color"] = "#606060"
    valid_prob_mask = ~invalid_mask
    pkl_df.loc[valid_prob_mask, "multivariate_color"] = pkl_df.loc[valid_prob_mask, prob_columns].apply(
        lambda row: _blend_color_from_probabilities(
            [float(v) for v in row.tolist()],
            class_names=class_names,
            class_colors=class_colors,
        ),
        axis=1,
    )

    nrows = len(city_slugs)
    fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(21.0, 4.2 * nrows))
    axes_2d = axes if nrows > 1 else [axes]

    for row_idx, slug in enumerate(city_slugs):
        place = city_places[slug]
        left_ax, center_left_ax, center_right_ax, right_ax = axes_2d[row_idx]

        grid_city_dir = joint_inputs_root / slug / street_pattern_dir / slug
        cells_path = grid_city_dir / "predicted_cells.geojson"
        roads_path = grid_city_dir / "roads.geojson"
        buffer_path = grid_city_dir / "buffer.geojson"
        if not cells_path.exists() or not buffer_path.exists():
            _plot_empty(left_ax, f"{place.split(',')[0]} — GRID (multi)")
            _plot_empty(center_left_ax, f"{place.split(',')[0]} — PKL (prob)")
            _plot_empty(center_right_ax, f"{place.split(',')[0]} — PKL (top1)")
            _plot_empty(right_ax, f"{place.split(',')[0]} — GRID (top1)")
            continue

        cells = gpd.read_file(cells_path)
        buffer_gdf = gpd.read_file(buffer_path).to_crs(4326)
        roads = gpd.read_file(roads_path).to_crs(4326) if roads_path.exists() else gpd.GeoDataFrame(geometry=[], crs=4326)
        poly = buffer_gdf.geometry.iloc[0]
        xmin, ymin, xmax, ymax = poly.bounds

        pkl_city = pkl_df[pkl_df.intersects(poly)].copy()
        if not pkl_city.empty:
            pkl_city["geometry"] = pkl_city.geometry.intersection(poly)
            pkl_city = pkl_city[pkl_city.geometry.notna() & ~pkl_city.geometry.is_empty].copy()
        pkl_invalid = pkl_city[pkl_city.get("_prob_invalid", False)].copy() if not pkl_city.empty else pkl_city
        cell_match_ratio = _compute_cell_match_ratio(cells, pkl_city) if args.grid_color_mode == "intensity" else pd.Series(index=cells.index, dtype=float)

        if cells.empty:
            _plot_empty(left_ax, f"{place.split(',')[0]} — GRID (multi)")
        else:
            base_multi = cells.get("multivariate_color")
            if base_multi is None or base_multi.isna().all():
                if "class_name" in cells.columns:
                    base_multi = cells["class_name"].map(lambda v: class_colors.get(str(v), "#808080"))
                else:
                    base_multi = pd.Series(["#72d6c9"] * len(cells), index=cells.index)
            if not isinstance(base_multi, pd.Series):
                base_multi = pd.Series(base_multi, index=cells.index)
            multi_colors = []
            for idx, base_color in base_multi.items():
                if args.grid_color_mode == "solid":
                    multi_colors.append(str(base_color))
                else:
                    ratio = float(cell_match_ratio.get(idx, 0.0)) if idx in cell_match_ratio.index else 0.0
                    if idx in cell_match_ratio.index:
                        multi_colors.append(_mix_with_white(str(base_color), ratio))
                    else:
                        multi_colors.append(_mix_with_white(str(base_color), 0.0, min_strength=0.06))
            cells.plot(
                ax=left_ax,
                color=multi_colors,
                linewidth=0,
            )
            if not roads.empty:
                roads.plot(ax=left_ax, color="#f3f3f3", linewidth=0.35, alpha=0.5, zorder=5)
            left_ax.set_facecolor("#1d1f22")
            left_ax.set_title(f"{place.split(',')[0]} — GRID (multi)", fontsize=10, fontweight="bold")
            left_ax.set_axis_off()

        if pkl_city.empty:
            _plot_empty(center_left_ax, f"{place.split(',')[0]} — PKL (prob)")
            _plot_empty(center_right_ax, f"{place.split(',')[0]} — PKL (top1)")
        else:
            # PKL panel with probability-distribution blend in class_names order.
            pkl_city.plot(
                ax=center_left_ax,
                color=pkl_city["multivariate_color"],
                linewidth=0,
            )
            if not roads.empty:
                roads.plot(ax=center_left_ax, color="#f3f3f3", linewidth=0.35, alpha=0.5, zorder=5)
            center_left_ax.set_facecolor("#1d1f22")
            center_left_ax.set_title(f"{place.split(',')[0]} — PKL (prob)", fontsize=10, fontweight="bold")
            center_left_ax.set_axis_off()
            if not pkl_invalid.empty:
                pkl_invalid.boundary.plot(ax=center_left_ax, color="#ff2d2d", linewidth=1.0, zorder=9)

            # PKL panel with top-1 class color.
            pkl_top1_colors = pkl_city["class_name"].map(lambda v: class_colors.get(str(v), "#808080"))
            pkl_city.plot(
                ax=center_right_ax,
                color=pkl_top1_colors,
                linewidth=0,
            )
            if not roads.empty:
                roads.plot(ax=center_right_ax, color="#f3f3f3", linewidth=0.35, alpha=0.5, zorder=5)
            center_right_ax.set_facecolor("#1d1f22")
            center_right_ax.set_title(f"{place.split(',')[0]} — PKL (top1)", fontsize=10, fontweight="bold")
            center_right_ax.set_axis_off()
            if not pkl_invalid.empty:
                pkl_invalid.boundary.plot(ax=center_right_ax, color="#ff2d2d", linewidth=1.0, zorder=9)

        if cells.empty:
            _plot_empty(right_ax, f"{place.split(',')[0]} — GRID (top1)")
        else:
            if "class_name" in cells.columns:
                base_top1 = cells["class_name"].map(lambda v: class_colors.get(str(v), "#808080"))
            else:
                base_top1 = pd.Series(["#72d6c9"] * len(cells), index=cells.index)
            top1_colors = []
            for idx, base_color in base_top1.items():
                if args.grid_color_mode == "solid":
                    top1_colors.append(str(base_color))
                else:
                    ratio = float(cell_match_ratio.get(idx, 0.0)) if idx in cell_match_ratio.index else 0.0
                    if idx in cell_match_ratio.index:
                        top1_colors.append(_mix_with_white(str(base_color), ratio))
                    else:
                        top1_colors.append(_mix_with_white(str(base_color), 0.0, min_strength=0.06))
            cells.plot(
                ax=right_ax,
                color=top1_colors,
                linewidth=0,
            )
            if not roads.empty:
                roads.plot(ax=right_ax, color="#f3f3f3", linewidth=0.35, alpha=0.5, zorder=5)
            right_ax.set_facecolor("#1d1f22")
            right_ax.set_title(f"{place.split(',')[0]} — GRID (top1)", fontsize=10, fontweight="bold")
            right_ax.set_axis_off()

        # Draw census-tract polygon borders on all panels for direct shape comparison.
        if not pkl_city.empty:
            pkl_city.boundary.plot(ax=left_ax, color="#000000", linewidth=0.45, zorder=8)
            pkl_city.boundary.plot(ax=center_left_ax, color="#000000", linewidth=0.45, zorder=8)
            pkl_city.boundary.plot(ax=center_right_ax, color="#000000", linewidth=0.45, zorder=8)
            pkl_city.boundary.plot(ax=right_ax, color="#000000", linewidth=0.45, zorder=8)
            if not pkl_invalid.empty:
                pkl_invalid.boundary.plot(ax=left_ax, color="#ff2d2d", linewidth=0.9, zorder=9)
                pkl_invalid.boundary.plot(ax=right_ax, color="#ff2d2d", linewidth=0.9, zorder=9)

        left_ax.set_xlim(xmin, xmax)
        left_ax.set_ylim(ymin, ymax)
        center_left_ax.set_xlim(xmin, xmax)
        center_left_ax.set_ylim(ymin, ymax)
        center_right_ax.set_xlim(xmin, xmax)
        center_right_ax.set_ylim(ymin, ymax)
        right_ax.set_xlim(xmin, xmax)
        right_ax.set_ylim(ymin, ymax)

    handles = [Patch(facecolor=class_colors[name], edgecolor="none", label=name) for name in class_names if name in class_colors]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False, fontsize=8, title="Street-pattern classes")
    if args.grid_color_mode == "intensity":
        fig.text(
            0.5,
            0.045,
            "Grid color intensity = share of census area in the cell with the same class (brighter = higher match)",
            ha="center",
            va="center",
            fontsize=8,
            color="#333333",
        )
    fig.suptitle(str(args.title), fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.08, 1, 0.985])
    fig.savefig(output, dpi=int(args.dpi))
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

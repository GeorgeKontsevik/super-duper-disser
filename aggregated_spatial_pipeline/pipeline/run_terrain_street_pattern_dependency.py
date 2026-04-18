from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.colors import ListedColormap
from loguru import logger
from rasterstats import point_query, zonal_stats
from scipy.stats import spearmanr

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata
from aggregated_spatial_pipeline.runtime_config import configure_logger
from aggregated_spatial_pipeline.visualization import (
    apply_preview_canvas,
    normalize_preview_gdf,
    order_street_pattern_classes,
    save_preview_figure,
)

from .crosswalks import build_crosswalk
from .run_pipeline3_street_pattern_to_quarters import CLASS_LABELS
from .transfers import apply_transfer_rule


PROB_COLUMNS = list(CLASS_LABELS.keys())
CLASS_COLUMN_CANDIDATES = ("top1_class_name", "class_name", "predicted_class", "street_pattern_class")


def _configure_logging() -> None:
    configure_logger("[terrain_x_street_pattern]")


def _log(message: str) -> None:
    logger.bind(tag="[terrain_x_street_pattern]").info(message)


def _warn(message: str) -> None:
    logger.bind(tag="[terrain_x_street_pattern]").warning(message)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DEM-derived terrain stats (elevation/slope) per quarter and correlate "
            "them with street-pattern classes/probabilities."
        )
    )
    parser.add_argument("--joint-input-dir", default=None)
    parser.add_argument("--place", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dem-path", required=True, help="Path to DEM GeoTIFF.")
    parser.add_argument("--dem-band", type=int, default=1, help="DEM band index. Default: 1.")
    parser.add_argument(
        "--blocks-path",
        default=None,
        help=(
            "Optional explicit path to quarter polygons. "
            "Default: <joint-input-dir>/derived_layers/quarters_clipped.parquet "
            "(fallback: quarters_sm_imputed.parquet)."
        ),
    )
    parser.add_argument(
        "--street-pattern-cells",
        default=None,
        help=(
            "Optional explicit path to street-pattern cells. "
            "Default: <joint-input-dir>/derived_layers/street_grid_clipped.parquet "
            "(fallback: street_pattern/<slug>/predicted_cells.geojson)."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild outputs even if cached artifacts exist.",
    )
    return parser.parse_args()


def _resolve_city_dir(args: argparse.Namespace) -> Path:
    if args.joint_input_dir:
        return Path(args.joint_input_dir).resolve()
    if args.place:
        slug = _slugify(str(args.place))
        return (
            Path(__file__).resolve().parents[2]
            / "aggregated_spatial_pipeline"
            / "outputs"
            / "joint_inputs"
            / slug
        ).resolve()
    raise ValueError("Provide either --joint-input-dir or --place.")


def _resolve_output_dir(city_dir: Path, args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).resolve()
    return (city_dir / "terrain_street_pattern_dependency").resolve()


def _resolve_blocks_path(city_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Blocks layer does not exist: {path}")
        return path
    primary = city_dir / "derived_layers" / "quarters_clipped.parquet"
    if primary.exists():
        return primary.resolve()
    fallback = city_dir / "derived_layers" / "quarters_sm_imputed.parquet"
    if fallback.exists():
        _warn(f"Using fallback blocks layer: {fallback.name}")
        return fallback.resolve()
    raise FileNotFoundError(
        f"Missing blocks layer. Checked: {primary} and {fallback}"
    )


def _resolve_street_cells_path(city_dir: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Street-pattern cells file does not exist: {path}")
        return path
    primary = city_dir / "derived_layers" / "street_grid_clipped.parquet"
    if primary.exists():
        return primary.resolve()
    fallback = city_dir / "street_pattern" / city_dir.name / "predicted_cells.geojson"
    if fallback.exists():
        _warn(f"Using fallback street-pattern cells: {fallback.name}")
        return fallback.resolve()
    raise FileNotFoundError(
        f"Missing street-pattern cells. Checked: {primary} and {fallback}"
    )


def _pick_class_column(cells: gpd.GeoDataFrame) -> str:
    for candidate in CLASS_COLUMN_CANDIDATES:
        if candidate in cells.columns:
            return candidate
    raise KeyError(f"Could not detect class column in street-pattern cells. Checked: {CLASS_COLUMN_CANDIDATES}")


def _class_slug(label: str) -> str:
    slug = label.lower().replace(" & ", "_").replace(" ", "_").replace("-", "_")
    return slug.replace("__", "_")


def _rename_prob_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    for prob_col, label in CLASS_LABELS.items():
        if prob_col in renamed.columns:
            slug = _class_slug(label)
            renamed[f"street_pattern_prob_{slug}"] = pd.to_numeric(renamed[prob_col], errors="coerce").fillna(0.0)
    return renamed


def _prepare_street_cells(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    class_col = _pick_class_column(cells)
    work = cells.copy()
    work = work[work.geometry.notna() & ~work.geometry.is_empty].copy()
    work = work[work.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if work.empty:
        raise ValueError("Street-pattern cells contain no polygon features after cleanup.")
    if "cell_id" in work.columns:
        work["grid_id"] = work["cell_id"].astype(str)
    else:
        work["grid_id"] = work.index.astype(str)

    if "class_name" not in work.columns:
        work["class_name"] = work[class_col].astype("string").fillna("unknown")
    work["street_pattern_dominant_class"] = work["class_name"].astype("string").fillna("unknown").astype(str)

    for prob_col in PROB_COLUMNS:
        if prob_col not in work.columns:
            work[prob_col] = 0.0
        work[prob_col] = pd.to_numeric(work[prob_col], errors="coerce").fillna(0.0)
    work["street_pattern_covered_mass"] = work[PROB_COLUMNS].sum(axis=1)
    work.loc[work["street_pattern_covered_mass"] <= 0.0, "street_pattern_dominant_class"] = "unknown"
    work = _rename_prob_columns(work)
    return work


def _transfer_street_pattern_to_blocks(
    *,
    blocks: gpd.GeoDataFrame,
    cells: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    class_col = _pick_class_column(cells)
    source_columns = ["geometry", class_col, *[c for c in PROB_COLUMNS if c in cells.columns]]
    source = cells[source_columns].copy()
    target = blocks.copy()
    source["grid_id"] = source.index.astype(str)
    if "class_name" not in source.columns:
        source["class_name"] = source[class_col].astype("string").fillna("unknown")
    if "block_id" not in target.columns:
        target["block_id"] = target.index.astype(str)

    polygon_types = {"Polygon", "MultiPolygon"}
    source_poly = source[source.geometry.geom_type.isin(polygon_types)].copy()
    target_poly = target[target.geometry.geom_type.isin(polygon_types)].copy()

    if source_poly.empty or target_poly.empty:
        raise ValueError("Street-pattern transfer requires non-empty polygon geometries in both layers.")

    crosswalk = build_crosswalk(source_poly, target_poly, "grid", "block")
    transferred_poly = apply_transfer_rule(
        source_gdf=source_poly,
        target_gdf=target_poly,
        crosswalk_gdf=crosswalk,
        source_layer="grid",
        target_layer="block",
        attribute="street_pattern_probs",
        aggregation_method="weighted_mean",
        weight_field="intersection_area",
    )
    transferred_poly = apply_transfer_rule(
        source_gdf=source_poly,
        target_gdf=transferred_poly,
        crosswalk_gdf=crosswalk,
        source_layer="grid",
        target_layer="block",
        attribute="street_pattern_class",
        aggregation_method="majority_vote",
        weight_field="intersection_area",
    )

    transferred = target.copy()
    for col in transferred_poly.columns:
        if col == "geometry":
            continue
        transferred[col] = pd.Series(index=transferred.index, dtype=transferred_poly[col].dtype)
        transferred.loc[transferred_poly.index, col] = transferred_poly[col]

    for prob_col in PROB_COLUMNS:
        if prob_col not in transferred.columns:
            transferred[prob_col] = 0.0
        transferred[prob_col] = pd.to_numeric(transferred[prob_col], errors="coerce").fillna(0.0)

    transferred["street_pattern_covered_mass"] = transferred[PROB_COLUMNS].sum(axis=1)
    fallback = pd.Series("unknown", index=transferred.index, dtype=object)
    covered_mask = transferred["street_pattern_covered_mass"] > 0.0
    if covered_mask.any():
        fallback.loc[covered_mask] = (
            transferred.loc[covered_mask, PROB_COLUMNS]
            .idxmax(axis=1)
            .map(CLASS_LABELS)
            .fillna("unknown")
        )
    if "street_pattern_class" not in transferred.columns:
        transferred["street_pattern_class"] = fallback
    transferred["street_pattern_dominant_class"] = (
        transferred["street_pattern_class"].fillna(fallback).astype(str)
    )
    transferred.loc[~covered_mask, "street_pattern_dominant_class"] = "unknown"
    transferred = _rename_prob_columns(transferred)
    return transferred


def _degrees_to_meters_scale(latitude_deg: float) -> tuple[float, float]:
    lat_rad = math.radians(latitude_deg)
    meters_per_deg_lat = (
        111132.92
        - 559.82 * math.cos(2.0 * lat_rad)
        + 1.175 * math.cos(4.0 * lat_rad)
        - 0.0023 * math.cos(6.0 * lat_rad)
    )
    meters_per_deg_lon = (
        111412.84 * math.cos(lat_rad)
        - 93.5 * math.cos(3.0 * lat_rad)
        + 0.118 * math.cos(5.0 * lat_rad)
    )
    return max(meters_per_deg_lon, 0.001), max(meters_per_deg_lat, 0.001)


def _build_slope_raster(
    *,
    dem_path: Path,
    band: int,
) -> tuple[Path, dict]:
    with rasterio.open(dem_path) as src:
        dem = src.read(band, masked=True).astype("float64")
        if dem.size == 0:
            raise ValueError(f"DEM is empty for band={band}: {dem_path}")

        xres = float(abs(src.transform.a))
        yres = float(abs(src.transform.e))
        if xres <= 0 or yres <= 0:
            raise ValueError(f"Invalid DEM pixel size in transform: {src.transform}")

        if src.crs is not None and getattr(src.crs, "is_geographic", False):
            lat_center = float((src.bounds.bottom + src.bounds.top) * 0.5)
            mx, my = _degrees_to_meters_scale(lat_center)
            xres = xres * mx
            yres = yres * my
            _warn(
                "DEM CRS is geographic; slope scale converted from degrees to meters "
                f"using latitude={lat_center:.4f}."
            )

        filled = dem.filled(np.nan)
        dz_dy, dz_dx = np.gradient(filled, yres, xres)
        slope_deg = np.degrees(np.arctan(np.sqrt(np.square(dz_dx) + np.square(dz_dy))))
        invalid = np.isnan(slope_deg) | dem.mask
        slope_out = np.where(invalid, -9999.0, slope_deg).astype("float32")

        profile = src.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")

        temp_dir = Path(tempfile.mkdtemp(prefix="asp-slope-"))
        slope_path = temp_dir / f"{dem_path.stem}_slope_deg.tif"
        with rasterio.open(slope_path, "w", **profile) as dst:
            dst.write(slope_out, 1)

    return slope_path, {"slope_nodata": -9999.0}


def _zonal_stats_table(
    *,
    polygons: gpd.GeoDataFrame,
    raster_path: Path,
    raster_band: int,
    prefix: str,
) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        work = polygons.to_crs(src.crs) if polygons.crs is not None and src.crs is not None else polygons.copy()
        stats = zonal_stats(
            vectors=work.geometry,
            raster=str(raster_path),
            band=int(raster_band),
            stats=["min", "max", "mean", "median"],
            nodata=src.nodata,
            all_touched=False,
            geojson_out=False,
        )
    frame = pd.DataFrame(stats)
    frame = frame.rename(columns={k: f"{prefix}_{k}" for k in ("min", "max", "mean", "median")})
    return frame


def _centroid_samples(
    *,
    polygons: gpd.GeoDataFrame,
    dem_path: Path,
    dem_band: int,
) -> gpd.GeoDataFrame:
    with rasterio.open(dem_path) as src:
        work = polygons.to_crs(src.crs) if polygons.crs is not None and src.crs is not None else polygons.copy()
        points = work.copy()
        points["geometry"] = points.geometry.representative_point()
        elev = point_query(
            vectors=points.geometry,
            raster=str(dem_path),
            band=dem_band,
            nodata=src.nodata,
            interpolate="nearest",
            geojson_out=False,
        )
        points["elevation_point_m"] = pd.to_numeric(pd.Series(elev, index=points.index), errors="coerce")
    return points


def _spearman_table(frame: pd.DataFrame, response_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for response_col in response_cols:
        for feature_col in feature_cols:
            if response_col not in frame.columns or feature_col not in frame.columns:
                continue
            subset = frame[[response_col, feature_col]].replace([np.inf, -np.inf], np.nan).dropna()
            n = int(len(subset))
            if n < 3:
                rows.append(
                    {
                        "response": response_col,
                        "street_pattern_feature": feature_col,
                        "spearman_rho": np.nan,
                        "p_value": np.nan,
                        "n": n,
                    }
                )
                continue
            rho, p_value = spearmanr(subset[response_col], subset[feature_col], nan_policy="omit")
            rows.append(
                {
                    "response": response_col,
                    "street_pattern_feature": feature_col,
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def _plot_metric_map(
    *,
    blocks: gpd.GeoDataFrame,
    metric_col: str,
    title: str,
    output_path: Path,
    boundary: gpd.GeoDataFrame | None = None,
    roads: gpd.GeoDataFrame | None = None,
) -> None:
    plot = blocks[[metric_col, "geometry"]].copy()
    plot = plot[plot.geometry.notna() & ~plot.geometry.is_empty].copy()
    plot[metric_col] = pd.to_numeric(plot[metric_col], errors="coerce")
    plot = plot[plot[metric_col].notna()].copy()
    if plot.empty:
        return
    boundary_plot = normalize_preview_gdf(boundary, target_crs="EPSG:3857")
    plot = normalize_preview_gdf(plot, boundary_plot, target_crs="EPSG:3857")
    if boundary_plot is None or boundary_plot.empty:
        boundary_plot = plot[["geometry"]].copy()
        try:
            boundary_union = boundary_plot.union_all()
            boundary_plot = gpd.GeoDataFrame({"geometry": [boundary_union]}, crs=plot.crs)
        except Exception:
            boundary_plot = None

    fig, ax = plt.subplots(figsize=(12, 12))
    apply_preview_canvas(fig, ax, boundary_plot, title=title)
    roads_plot = normalize_preview_gdf(roads, boundary_plot, target_crs="EPSG:3857")
    if roads_plot is not None and not roads_plot.empty:
        roads_plot = roads_plot[roads_plot.geometry.notna() & ~roads_plot.geometry.is_empty].copy()
        roads_plot = roads_plot[roads_plot.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
        if not roads_plot.empty:
            roads_plot.plot(
                ax=ax,
                color="#6b7280",
                linewidth=0.32,
                alpha=0.45,
                zorder=2,
            )
    plot.plot(
        ax=ax,
        column=metric_col,
        cmap="terrain",
        linewidth=0.05,
        edgecolor="#cbd5e1",
        alpha=0.72,
        legend=True,
        legend_kwds={"shrink": 0.6, "label": metric_col},
        zorder=3,
    )
    if roads_plot is not None and not roads_plot.empty:
        roads_plot.plot(
            ax=ax,
            color="#334155",
            linewidth=0.18,
            alpha=0.35,
            zorder=4,
        )
    ax.set_axis_off()
    save_preview_figure(fig, output_path)
    plt.close(fig)


def _plot_corr_heatmap(corr: pd.DataFrame, output_path: Path) -> None:
    if corr.empty:
        return
    pivot_rho = corr.pivot(index="response", columns="street_pattern_feature", values="spearman_rho")
    pivot_p = corr.pivot(index="response", columns="street_pattern_feature", values="p_value")
    if pivot_rho.empty:
        return
    annot = pd.DataFrame("", index=pivot_rho.index, columns=pivot_rho.columns, dtype=object)
    for ridx in pivot_rho.index:
        for cidx in pivot_rho.columns:
            rho = pivot_rho.loc[ridx, cidx]
            p_val = pivot_p.loc[ridx, cidx] if cidx in pivot_p.columns and ridx in pivot_p.index else np.nan
            if pd.isna(rho):
                annot.loc[ridx, cidx] = "n/a"
                continue
            if pd.isna(p_val):
                sig = "ns"
                p_str = "n/a"
            elif float(p_val) < 0.001:
                sig = "***"
                p_str = "<0.001"
            elif float(p_val) < 0.01:
                sig = "**"
                p_str = f"{float(p_val):.3f}"
            elif float(p_val) < 0.05:
                sig = "*"
                p_str = f"{float(p_val):.3f}"
            else:
                sig = "ns"
                p_str = f"{float(p_val):.3f}"
            annot.loc[ridx, cidx] = f"{float(rho):.2f}\np={p_str} {sig}"

    sig_mask = (pivot_p < 0.05) & pivot_rho.notna()
    ns_mask = ~sig_mask

    fig, ax = plt.subplots(figsize=(13.2, 5.8))
    # Base layer: non-significant cells in neutral gray.
    sns.heatmap(
        pd.DataFrame(np.zeros_like(pivot_rho, dtype=float), index=pivot_rho.index, columns=pivot_rho.columns),
        cmap=ListedColormap(["#d1d5db"]),
        cbar=False,
        mask=~ns_mask,
        linewidths=0.4,
        linecolor="#f8fafc",
        ax=ax,
    )
    # Overlay: significant cells with correlation colors.
    sns.heatmap(
        pivot_rho,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        mask=~sig_mask,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 8},
        linewidths=0.4,
        linecolor="#f8fafc",
        cbar_kws={"label": "Spearman rho"},
        ax=ax,
    )
    # Add annotations for non-significant cells too (overlay heatmap annotates only significant layer).
    for i, ridx in enumerate(pivot_rho.index):
        for j, cidx in enumerate(pivot_rho.columns):
            if bool(sig_mask.loc[ridx, cidx]):
                continue
            ax.text(
                j + 0.5,
                i + 0.5,
                str(annot.loc[ridx, cidx]),
                ha="center",
                va="center",
                fontsize=8,
                color="#374151",
            )
    ax.set_title("Terrain vs Street-Pattern Probabilities (Spearman; rho + p-value + significance)")
    ax.set_xlabel("Street-pattern feature")
    ax.set_ylabel("Terrain response")
    fig.text(
        0.5,
        0.01,
        "Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#334155",
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_dominant_class_bars(dominant: pd.DataFrame, output_path: Path) -> None:
    if dominant.empty:
        return
    work = dominant.copy()
    class_order = order_street_pattern_classes(set(work["street_pattern_dominant_class"].astype(str).tolist()))
    work["street_pattern_dominant_class"] = pd.Categorical(
        work["street_pattern_dominant_class"].astype(str),
        categories=class_order,
        ordered=True,
    )
    work = work.sort_values("street_pattern_dominant_class")

    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = ["elevation_m_mean", "slope_deg_mean"]
    melt = work[["street_pattern_dominant_class", *metrics]].melt(
        id_vars="street_pattern_dominant_class",
        var_name="metric",
        value_name="value",
    )
    sns.barplot(
        data=melt,
        x="street_pattern_dominant_class",
        y="value",
        hue="metric",
        ax=ax,
    )
    ax.set_title("Terrain Means By Dominant Street Pattern")
    ax.set_xlabel("Dominant street-pattern class")
    ax.set_ylabel("Mean value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right", frameon=True, title="Metric")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _weighted_multivariate_summary(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    metrics = ["elevation_m_mean", "elevation_m_median", "elevation_m_range", "slope_deg_mean", "slope_deg_median"]
    feature_to_class: dict[str, str] = {
        f"street_pattern_prob_{_class_slug(label)}": label for label in CLASS_LABELS.values()
    }
    rows: list[dict[str, float | str]] = []
    for feature_col in feature_cols:
        class_name = feature_to_class.get(feature_col)
        if class_name is None or feature_col not in frame.columns:
            continue
        weights = pd.to_numeric(frame[feature_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if float(weights.sum()) <= 0.0:
            rows.append(
                {
                    "street_pattern_class": class_name,
                    "effective_count": 0.0,
                    **{m: np.nan for m in metrics},
                }
            )
            continue
        row: dict[str, float | str] = {"street_pattern_class": class_name, "effective_count": float(weights.sum())}
        for metric in metrics:
            values = pd.to_numeric(frame[metric], errors="coerce")
            valid = values.notna() & weights.gt(0.0)
            if not valid.any():
                row[metric] = np.nan
                continue
            w = weights[valid]
            v = values[valid]
            row[metric] = float((v * w).sum() / w.sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    class_order = order_street_pattern_classes(set(summary["street_pattern_class"].astype(str).tolist()))
    summary["sort_key"] = summary["street_pattern_class"].map({name: idx for idx, name in enumerate(class_order)})
    summary = summary.sort_values(["sort_key", "street_pattern_class"]).drop(columns=["sort_key"]).reset_index(drop=True)
    return summary


def _plot_multivariate_class_bars(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    work = summary.copy()
    class_order = order_street_pattern_classes(set(work["street_pattern_class"].astype(str).tolist()))
    work["street_pattern_class"] = pd.Categorical(
        work["street_pattern_class"].astype(str),
        categories=class_order,
        ordered=True,
    )
    work = work.sort_values("street_pattern_class")
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = ["elevation_m_mean", "slope_deg_mean"]
    melt = work[["street_pattern_class", *metrics]].melt(
        id_vars="street_pattern_class",
        var_name="metric",
        value_name="value",
    )
    sns.barplot(
        data=melt,
        x="street_pattern_class",
        y="value",
        hue="metric",
        ax=ax,
    )
    ax.set_title("Terrain Means By Street Pattern (Probability-Weighted)")
    ax.set_xlabel("Street-pattern class")
    ax.set_ylabel("Probability-weighted mean value")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right", frameon=True, title="Metric")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    _configure_logging()
    args = parse_args()

    city_dir = _resolve_city_dir(args)
    output_dir = _resolve_output_dir(city_dir, args)
    prepared_dir = output_dir / "prepared"
    stats_dir = output_dir / "stats"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    blocks_out_path = prepared_dir / "blocks_terrain_street_pattern.parquet"
    points_out_path = prepared_dir / "block_points_elevation.parquet"
    grid_out_path = prepared_dir / "street_grid_terrain_street_pattern.parquet"
    grid_points_out_path = prepared_dir / "street_grid_points_elevation.parquet"
    corr_out_path = stats_dir / "terrain_correlations.csv"
    dominant_out_path = stats_dir / "dominant_class_terrain_summary.csv"
    multivariate_out_path = stats_dir / "multivariate_class_terrain_summary.csv"
    grid_corr_out_path = stats_dir / "grid_terrain_correlations.csv"
    grid_dominant_out_path = stats_dir / "grid_dominant_class_terrain_summary.csv"
    grid_multivariate_out_path = stats_dir / "grid_multivariate_class_terrain_summary.csv"
    preview_dir = output_dir / "preview_png"
    preview_dir.mkdir(parents=True, exist_ok=True)
    elev_map_path = preview_dir / "terrain_elevation_mean_map.png"
    slope_map_path = preview_dir / "terrain_slope_mean_map.png"
    heatmap_path = preview_dir / "terrain_street_pattern_spearman_heatmap.png"
    dominant_bar_path = preview_dir / "terrain_by_dominant_street_pattern.png"
    multivariate_bar_path = preview_dir / "terrain_by_multivariate_street_pattern.png"
    grid_elev_map_path = preview_dir / "grid_terrain_elevation_mean_map.png"
    grid_slope_map_path = preview_dir / "grid_terrain_slope_mean_map.png"
    grid_heatmap_path = preview_dir / "grid_terrain_street_pattern_spearman_heatmap.png"
    grid_dominant_bar_path = preview_dir / "grid_terrain_by_dominant_street_pattern.png"
    grid_multivariate_bar_path = preview_dir / "grid_terrain_by_multivariate_street_pattern.png"

    if (
        (not args.no_cache)
        and manifest_path.exists()
        and blocks_out_path.exists()
        and grid_out_path.exists()
        and corr_out_path.exists()
        and dominant_out_path.exists()
        and multivariate_out_path.exists()
        and grid_corr_out_path.exists()
        and grid_dominant_out_path.exists()
        and grid_multivariate_out_path.exists()
    ):
        _log("Using cached terrain x street-pattern dependency outputs.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(json.dumps(manifest, ensure_ascii=False))
        return

    dem_path = Path(args.dem_path).resolve()
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM path does not exist: {dem_path}")

    blocks_path = _resolve_blocks_path(city_dir, args.blocks_path)
    street_cells_path = _resolve_street_cells_path(city_dir, args.street_pattern_cells)
    _log(
        "Inputs resolved: "
        f"city_dir={city_dir.name}, blocks={blocks_path.name}, street_cells={street_cells_path.name}, dem={dem_path.name}"
    )

    blocks = read_geodata(blocks_path)
    blocks = blocks[blocks.geometry.notna() & ~blocks.geometry.is_empty].copy()
    blocks = blocks[blocks.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if blocks.empty:
        raise ValueError(f"Blocks layer has no polygon features after cleanup: {blocks_path}")
    blocks["block_id"] = blocks.index.astype(str)

    street_cells = read_geodata(street_cells_path)
    street_cells = street_cells[street_cells.geometry.notna() & ~street_cells.geometry.is_empty].copy()
    if street_cells.empty:
        raise ValueError(f"Street-pattern cells are empty after geometry cleanup: {street_cells_path}")

    blocks_enriched = _transfer_street_pattern_to_blocks(blocks=blocks, cells=street_cells)
    _log(f"Street-pattern attributes transferred to blocks: {len(blocks_enriched)} features.")
    grid_enriched = _prepare_street_cells(street_cells)
    _log(f"Street-pattern grid prepared for direct terrain analysis: {len(grid_enriched)} features.")

    slope_path, slope_meta = _build_slope_raster(dem_path=dem_path, band=int(args.dem_band))
    _log(f"Slope raster generated: {slope_path.name}")

    elev_stats = _zonal_stats_table(
        polygons=blocks_enriched,
        raster_path=dem_path,
        raster_band=int(args.dem_band),
        prefix="elevation_m",
    )
    slope_stats = _zonal_stats_table(
        polygons=blocks_enriched,
        raster_path=slope_path,
        raster_band=1,
        prefix="slope_deg",
    )
    terrain = pd.concat([elev_stats, slope_stats], axis=1)
    blocks_enriched = pd.concat([blocks_enriched.reset_index(drop=True), terrain.reset_index(drop=True)], axis=1)
    blocks_enriched["elevation_m_range"] = (
        pd.to_numeric(blocks_enriched["elevation_m_max"], errors="coerce")
        - pd.to_numeric(blocks_enriched["elevation_m_min"], errors="coerce")
    )

    points = _centroid_samples(polygons=blocks_enriched[["block_id", "geometry"]].copy(), dem_path=dem_path, dem_band=int(args.dem_band))
    points = points.rename(columns={"elevation_point_m": "elevation_point_m"})

    grid_elev_stats = _zonal_stats_table(
        polygons=grid_enriched,
        raster_path=dem_path,
        raster_band=int(args.dem_band),
        prefix="elevation_m",
    )
    grid_slope_stats = _zonal_stats_table(
        polygons=grid_enriched,
        raster_path=slope_path,
        raster_band=1,
        prefix="slope_deg",
    )
    grid_terrain = pd.concat([grid_elev_stats, grid_slope_stats], axis=1)
    grid_enriched = pd.concat([grid_enriched.reset_index(drop=True), grid_terrain.reset_index(drop=True)], axis=1)
    grid_enriched["elevation_m_range"] = (
        pd.to_numeric(grid_enriched["elevation_m_max"], errors="coerce")
        - pd.to_numeric(grid_enriched["elevation_m_min"], errors="coerce")
    )
    grid_points = _centroid_samples(polygons=grid_enriched[["grid_id", "geometry"]].copy(), dem_path=dem_path, dem_band=int(args.dem_band))
    grid_points = grid_points.rename(columns={"elevation_point_m": "elevation_point_m"})

    blocks_analysis = blocks_enriched[
        blocks_enriched["street_pattern_dominant_class"].astype(str).str.strip().str.lower() != "unknown"
    ].copy()
    dropped_unknown_blocks = int(len(blocks_enriched) - len(blocks_analysis))
    if dropped_unknown_blocks > 0:
        _log(f"Quarter-level analysis: dropped unknown street-pattern class blocks: {dropped_unknown_blocks}")

    prob_feature_cols = [c for c in blocks_analysis.columns if c.startswith("street_pattern_prob_")]
    if not prob_feature_cols:
        raise RuntimeError("No street-pattern probability feature columns were found after transfer.")

    response_cols = [
        "elevation_m_mean",
        "elevation_m_median",
        "elevation_m_range",
        "slope_deg_mean",
        "slope_deg_median",
    ]
    corr = _spearman_table(blocks_analysis, response_cols=response_cols, feature_cols=prob_feature_cols)
    corr = corr.sort_values(["response", "street_pattern_feature"]).reset_index(drop=True)
    grid_prob_feature_cols = [c for c in grid_enriched.columns if c.startswith("street_pattern_prob_")]
    if not grid_prob_feature_cols:
        raise RuntimeError("No street-pattern probability feature columns were found on street-pattern grid.")
    grid_corr = _spearman_table(grid_enriched, response_cols=response_cols, feature_cols=grid_prob_feature_cols)
    grid_corr = grid_corr.sort_values(["response", "street_pattern_feature"]).reset_index(drop=True)

    dominant = (
        blocks_analysis.groupby("street_pattern_dominant_class", as_index=False)[
            ["elevation_m_mean", "elevation_m_median", "elevation_m_range", "slope_deg_mean", "slope_deg_median"]
        ]
        .mean(numeric_only=True)
    )
    counts = blocks_analysis.groupby("street_pattern_dominant_class").size().rename("block_count").reset_index()
    dominant = dominant.merge(counts, on="street_pattern_dominant_class", how="left")
    class_order = order_street_pattern_classes(set(dominant["street_pattern_dominant_class"].astype(str).tolist()))
    dominant["sort_key"] = dominant["street_pattern_dominant_class"].map({name: idx for idx, name in enumerate(class_order)})
    dominant = dominant.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)
    grid_dominant = (
        grid_enriched.groupby("street_pattern_dominant_class", as_index=False)[
            ["elevation_m_mean", "elevation_m_median", "elevation_m_range", "slope_deg_mean", "slope_deg_median"]
        ]
        .mean(numeric_only=True)
    )
    grid_counts = grid_enriched.groupby("street_pattern_dominant_class").size().rename("cell_count").reset_index()
    grid_dominant = grid_dominant.merge(grid_counts, on="street_pattern_dominant_class", how="left")
    grid_class_order = order_street_pattern_classes(set(grid_dominant["street_pattern_dominant_class"].astype(str).tolist()))
    grid_dominant["sort_key"] = grid_dominant["street_pattern_dominant_class"].map(
        {name: idx for idx, name in enumerate(grid_class_order)}
    )
    grid_dominant = grid_dominant.sort_values(["sort_key", "street_pattern_dominant_class"]).drop(columns=["sort_key"]).reset_index(drop=True)
    multivariate = _weighted_multivariate_summary(blocks_analysis, feature_cols=prob_feature_cols)
    grid_multivariate = _weighted_multivariate_summary(grid_enriched, feature_cols=grid_prob_feature_cols)

    prepare_geodata_for_parquet(blocks_enriched).to_parquet(blocks_out_path)
    prepare_geodata_for_parquet(points).to_parquet(points_out_path)
    prepare_geodata_for_parquet(grid_enriched).to_parquet(grid_out_path)
    prepare_geodata_for_parquet(grid_points).to_parquet(grid_points_out_path)
    corr.to_csv(corr_out_path, index=False)
    dominant.to_csv(dominant_out_path, index=False)
    multivariate.to_csv(multivariate_out_path, index=False)
    grid_corr.to_csv(grid_corr_out_path, index=False)
    grid_dominant.to_csv(grid_dominant_out_path, index=False)
    grid_multivariate.to_csv(grid_multivariate_out_path, index=False)

    boundary_path = city_dir / "analysis_territory" / "buffer.parquet"
    boundary = read_geodata(boundary_path) if boundary_path.exists() else None
    roads_path = city_dir / "derived_layers" / "roads_drive_osmnx.parquet"
    roads = read_geodata(roads_path) if roads_path.exists() else None
    _plot_metric_map(
        blocks=blocks_analysis,
        metric_col="elevation_m_mean",
        title="Mean Elevation By Quarter",
        output_path=elev_map_path,
        boundary=boundary,
        roads=roads,
    )
    _plot_metric_map(
        blocks=blocks_analysis,
        metric_col="slope_deg_mean",
        title="Mean Slope By Quarter",
        output_path=slope_map_path,
        boundary=boundary,
        roads=roads,
    )
    _plot_corr_heatmap(corr, heatmap_path)
    _plot_dominant_class_bars(dominant, dominant_bar_path)
    _plot_multivariate_class_bars(multivariate, multivariate_bar_path)
    _plot_metric_map(
        blocks=grid_enriched,
        metric_col="elevation_m_mean",
        title="Mean Elevation By Street-Pattern Grid Cell",
        output_path=grid_elev_map_path,
        boundary=boundary,
        roads=roads,
    )
    _plot_metric_map(
        blocks=grid_enriched,
        metric_col="slope_deg_mean",
        title="Mean Slope By Street-Pattern Grid Cell",
        output_path=grid_slope_map_path,
        boundary=boundary,
        roads=roads,
    )
    _plot_corr_heatmap(grid_corr, grid_heatmap_path)
    _plot_dominant_class_bars(grid_dominant, grid_dominant_bar_path)
    _plot_multivariate_class_bars(grid_multivariate, grid_multivariate_bar_path)

    counts_payload = {
        "blocks": int(len(blocks_enriched)),
        "blocks_with_elevation_mean": int(pd.to_numeric(blocks_enriched["elevation_m_mean"], errors="coerce").notna().sum()),
        "blocks_with_slope_mean": int(pd.to_numeric(blocks_enriched["slope_deg_mean"], errors="coerce").notna().sum()),
        "quarter_unknown_dropped": int(dropped_unknown_blocks),
        "quarter_analysis_blocks": int(len(blocks_analysis)),
        "street_pattern_classes": int(dominant["street_pattern_dominant_class"].nunique()),
        "grid_cells": int(len(grid_enriched)),
        "grid_cells_with_elevation_mean": int(pd.to_numeric(grid_enriched["elevation_m_mean"], errors="coerce").notna().sum()),
        "grid_cells_with_slope_mean": int(pd.to_numeric(grid_enriched["slope_deg_mean"], errors="coerce").notna().sum()),
        "grid_street_pattern_classes": int(grid_dominant["street_pattern_dominant_class"].nunique()),
    }
    manifest = {
        "experiment": "terrain_x_street_pattern_dependency",
        "city_slug": city_dir.name,
        "counts": counts_payload,
        "inputs": {
            "blocks": str(blocks_path),
            "street_pattern_cells": str(street_cells_path),
            "dem_path": str(dem_path),
            "dem_band": int(args.dem_band),
        },
        "files": {
            "blocks_terrain_street_pattern": str(blocks_out_path),
            "block_points_elevation": str(points_out_path),
            "street_grid_terrain_street_pattern": str(grid_out_path),
            "street_grid_points_elevation": str(grid_points_out_path),
            "terrain_correlations": str(corr_out_path),
            "dominant_class_terrain_summary": str(dominant_out_path),
            "multivariate_class_terrain_summary": str(multivariate_out_path),
            "grid_terrain_correlations": str(grid_corr_out_path),
            "grid_dominant_class_terrain_summary": str(grid_dominant_out_path),
            "grid_multivariate_class_terrain_summary": str(grid_multivariate_out_path),
            "elevation_mean_map_png": str(elev_map_path),
            "slope_mean_map_png": str(slope_map_path),
            "terrain_correlation_heatmap_png": str(heatmap_path),
            "terrain_dominant_class_bar_png": str(dominant_bar_path),
            "terrain_multivariate_class_bar_png": str(multivariate_bar_path),
            "grid_elevation_mean_map_png": str(grid_elev_map_path),
            "grid_slope_mean_map_png": str(grid_slope_map_path),
            "grid_terrain_correlation_heatmap_png": str(grid_heatmap_path),
            "grid_terrain_dominant_class_bar_png": str(grid_dominant_bar_path),
            "grid_terrain_multivariate_class_bar_png": str(grid_multivariate_bar_path),
        },
        "slope": slope_meta,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(
        "Terrain x street-pattern dependency ready: "
        f"blocks={counts_payload['blocks']}, elev_ok={counts_payload['blocks_with_elevation_mean']}, "
        f"slope_ok={counts_payload['blocks_with_slope_mean']}, classes={counts_payload['street_pattern_classes']}; "
        f"grid_cells={counts_payload['grid_cells']}, grid_elev_ok={counts_payload['grid_cells_with_elevation_mean']}, "
        f"grid_slope_ok={counts_payload['grid_cells_with_slope_mean']}"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

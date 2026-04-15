from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from .preview_theme import (
    CANVAS_BACKGROUND,
    CANVAS_BOUNDARY_EDGE,
    CANVAS_BOUNDARY_FILL,
    CANVAS_FRAME_EDGE,
    CANVAS_INK,
    DEFAULT_PREVIEW_DPI,
    LEGEND_EDGE,
    LEGEND_FACE,
)


DEFAULT_TARGET_CRS = "EPSG:3857"


def clip_to_preview_boundary(
    gdf: gpd.GeoDataFrame | None,
    boundary_layer: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty or boundary_layer is None or boundary_layer.empty:
        return gdf
    try:
        work = gdf.copy()
        boundary = boundary_layer.copy()
        if work.crs is not None and boundary.crs is not None and work.crs != boundary.crs:
            boundary = boundary.to_crs(work.crs)
        geom_types = set(work.geom_type.astype(str))
        if any("Point" in geom for geom in geom_types):
            boundary_union = boundary.union_all()
            clipped = work[work.geometry.within(boundary_union) | work.geometry.intersects(boundary_union)].copy()
        else:
            clipped = gpd.clip(work, boundary)
        if clipped is None or clipped.empty:
            return gdf
        clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
        return clipped if not clipped.empty else gdf
    except Exception:
        return gdf


def normalize_preview_gdf(
    gdf: gpd.GeoDataFrame | None,
    boundary_layer: gpd.GeoDataFrame | None = None,
    *,
    target_crs: str = DEFAULT_TARGET_CRS,
) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty:
        return gdf
    work = gdf.copy()
    try:
        if work.crs is not None:
            work = work.to_crs(target_crs)
    except Exception:
        pass
    return clip_to_preview_boundary(work, boundary_layer)


def apply_preview_canvas(
    fig,
    ax,
    boundary_layer: gpd.GeoDataFrame | None,
    *,
    title: str | None = None,
    pad_ratio: float = 0.08,
    min_pad: float = 250.0,
) -> None:
    fig.patch.set_facecolor(CANVAS_BACKGROUND)
    ax.set_facecolor(CANVAS_BACKGROUND)
    if boundary_layer is None or boundary_layer.empty:
        if title:
            ax.set_title(title, fontsize=15, fontweight="bold", color=CANVAS_INK, pad=12)
        return
    try:
        minx, miny, maxx, maxy = boundary_layer.total_bounds
        span_x = maxx - minx
        span_y = maxy - miny
        span = max(span_x, span_y)
        pad = max(span * pad_ratio, min_pad)
        center_x = (minx + maxx) / 2.0
        center_y = (miny + maxy) / 2.0
        half_span = span / 2.0
        frame_minx = center_x - half_span - pad
        frame_maxx = center_x + half_span + pad
        frame_miny = center_y - half_span - pad
        frame_maxy = center_y + half_span + pad
        outer = gpd.GeoDataFrame(
            {"geometry": [box(frame_minx, frame_miny, frame_maxx, frame_maxy)]},
            crs=boundary_layer.crs,
        )
        outer.plot(ax=ax, facecolor=CANVAS_BACKGROUND, edgecolor="none", alpha=1.0, zorder=-20)
        boundary_layer.plot(ax=ax, facecolor=CANVAS_BOUNDARY_FILL, edgecolor="none", linewidth=0.0, alpha=1.0, zorder=-10)
        boundary_layer.boundary.plot(ax=ax, color=CANVAS_BOUNDARY_EDGE, linewidth=1.1, zorder=20)
        outer.boundary.plot(ax=ax, color=CANVAS_FRAME_EDGE, linewidth=0.6, zorder=21)
        ax.set_xlim(frame_minx, frame_maxx)
        ax.set_ylim(frame_miny, frame_maxy)
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", color=CANVAS_INK, pad=12)


def legend_bottom(ax, handles: list, *, max_cols: int = 4, fontsize: int = 8) -> None:
    if not handles:
        return
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(max_cols, len(handles)),
        frameon=True,
        fontsize=fontsize,
        facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE,
        labelcolor=CANVAS_INK,
        framealpha=0.98,
        borderpad=0.35,
        handletextpad=0.5,
        columnspacing=1.0,
        handlelength=1.8,
        borderaxespad=0.2,
    )


def footer_text(
    fig,
    lines: list[str] | None,
    *,
    y: float = 0.02,
    fontsize: int = 8,
    color: str = CANVAS_INK,
    bbox: dict | None = None,
) -> None:
    if not lines:
        return
    text = "\n".join(line for line in lines if line)
    if not text.strip():
        return
    kwargs = {
        "ha": "center",
        "va": "bottom",
        "fontsize": fontsize,
        "color": color,
    }
    if bbox is not None:
        kwargs["bbox"] = bbox
    else:
        kwargs["bbox"] = {
            "facecolor": LEGEND_FACE,
            "edgecolor": LEGEND_EDGE,
            "boxstyle": "round,pad=0.24",
            "alpha": 0.96,
        }
    fig.text(0.5, y, text, **kwargs)


def save_preview_figure(fig, output_path: Path, *, dpi: int = DEFAULT_PREVIEW_DPI) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06, facecolor=fig.get_facecolor())

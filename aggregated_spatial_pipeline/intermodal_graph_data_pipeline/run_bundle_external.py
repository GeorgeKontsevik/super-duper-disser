from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from loguru import logger

from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.intermodal_graph_data_pipeline.pipeline import build_intermodal_graph_bundle
from aggregated_spatial_pipeline.runtime_config import configure_logger
from aggregated_spatial_pipeline.visualization import (
    CANVAS_INK,
    apply_preview_canvas,
    get_palette,
    legend_bottom,
    normalize_preview_gdf,
    save_preview_figure,
)


ROOT = Path(__file__).resolve().parents[2]
def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build intermodal graph bundle in dedicated iduedu runtime.")
    parser.add_argument("--place")
    parser.add_argument("--joint-input-dir")
    parser.add_argument("--boundary-path")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def _configure_logging() -> None:
    configure_logger("[intermodal]")


def _resolve_city_dir(args: argparse.Namespace) -> Path | None:
    if args.joint_input_dir:
        return Path(args.joint_input_dir).resolve()
    if args.place:
        return (ROOT / "aggregated_spatial_pipeline" / "outputs" / "joint_inputs" / _slugify(str(args.place))).resolve()
    return None


def _resolve_run_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    city_dir = _resolve_city_dir(args)
    place = str(args.place) if args.place else (city_dir.name if city_dir is not None else None)
    if not place:
        raise ValueError("Provide --place or --joint-input-dir.")

    if args.boundary_path:
        boundary_path = Path(args.boundary_path).resolve()
    elif city_dir is not None:
        candidates = [
            city_dir / "analysis_territory" / "buffer_collection.parquet",
            city_dir / "analysis_territory" / "buffer.parquet",
        ]
        boundary_path = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if boundary_path is None:
            raise FileNotFoundError(f"Could not resolve intermodal boundary in {city_dir / 'analysis_territory'}")
    else:
        raise ValueError("Provide --boundary-path when city bundle cannot be resolved.")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    elif city_dir is not None:
        output_dir = (city_dir / "intermodal_graph_iduedu").resolve()
    else:
        raise ValueError("Provide --output-dir when city bundle cannot be resolved.")

    return place, boundary_path, output_dir


def _derive_preview_dirs(output_dir: Path) -> tuple[Path, Path | None]:
    local_dir = output_dir / "preview_png"
    shared_dir = None
    if output_dir.name == "intermodal_graph_iduedu":
        shared_dir = output_dir.parent / "preview_png" / "all_together"
    return local_dir, shared_dir


def _save_intermodal_previews(manifest: dict, output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    local_dir, shared_dir = _derive_preview_dirs(output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    if shared_dir is not None:
        shared_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    boundary_path = manifest.get("files", {}).get("boundary")
    nodes_path = manifest.get("files", {}).get("graph_nodes")
    edges_path = manifest.get("files", {}).get("graph_edges")
    if not boundary_path or not nodes_path or not edges_path:
        return outputs

    boundary = normalize_preview_gdf(read_geodata(Path(boundary_path)), target_crs="EPSG:3857")
    nodes = normalize_preview_gdf(read_geodata(Path(nodes_path)), boundary, target_crs="EPSG:3857")
    edges = normalize_preview_gdf(read_geodata(Path(edges_path)), boundary, target_crs="EPSG:3857")
    if boundary is None or boundary.empty:
        return outputs

    def _save(fig, stem: str) -> None:
        out = local_dir / f"{stem}.png"
        save_preview_figure(fig, out)
        plt.close(fig)
        outputs[stem] = str(out)
        if shared_dir is not None:
            try:
                shutil.copy2(out, shared_dir / out.name)
            except Exception:
                pass
        logger.info("Saved preview: {}", out.name)

    if edges is not None and not edges.empty:
        fig, ax = plt.subplots(figsize=(12, 12))
        apply_preview_canvas(fig, ax, boundary, title="Intermodal Graph Modes", min_pad=120.0)
        legend_handles = []
        color_by_mode = get_palette("pt_modes")
        mode_column = next(
            (
                col
                for col in ["transport_type", "route_type", "type", "mode"]
                if col in edges.columns
            ),
            None,
        )
        if mode_column is None:
            edges.plot(ax=ax, color="#0f766e", linewidth=0.6, alpha=0.85)
            legend_handles.append(Line2D([0], [0], color="#0f766e", linewidth=2, label="intermodal edges"))
        else:
            for mode_value, part in edges.groupby(edges[mode_column].astype(str).str.lower()):
                color = color_by_mode.get(mode_value, "#0f766e")
                part.plot(ax=ax, color=color, linewidth=0.6, alpha=0.88)
                legend_handles.append(Line2D([0], [0], color=color, linewidth=2, label=mode_value))
        if nodes is not None and not nodes.empty:
            nodes.plot(ax=ax, color=CANVAS_INK, markersize=5, alpha=0.7)
            legend_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=CANVAS_INK, markersize=6, label="graph nodes"))
        legend_bottom(ax, legend_handles)
        ax.set_axis_off()
        _save(fig, "pt_intermodal_graph_modes")

    if nodes is not None and not nodes.empty:
        fig, ax = plt.subplots(figsize=(12, 12))
        apply_preview_canvas(fig, ax, boundary, title="Intermodal Graph Nodes", min_pad=120.0)
        if edges is not None and not edges.empty:
            edges.plot(ax=ax, color="#cbd5e1", linewidth=0.3, alpha=0.55)
        nodes.plot(ax=ax, color="#be123c", markersize=7, alpha=0.82)
        legend_bottom(
            ax,
            [
                Line2D([0], [0], color="#cbd5e1", linewidth=2, label="intermodal edges"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#be123c", markersize=7, label="graph nodes"),
            ],
        )
        ax.set_axis_off()
        _save(fig, "pt_intermodal_graph_nodes")

    return outputs


def main() -> None:
    _configure_logging()
    args = parse_args()
    place, boundary_path, output_dir = _resolve_run_paths(args)
    manifest = build_intermodal_graph_bundle(
        place=place,
        output_dir=output_dir,
        boundary_path=boundary_path,
        repo_root=ROOT,
    )
    preview_outputs = _save_intermodal_previews(manifest, output_dir)
    if preview_outputs:
        manifest = {**manifest, "preview_outputs": preview_outputs}
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

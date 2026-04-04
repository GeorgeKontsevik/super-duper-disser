from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from loguru import logger

from aggregated_spatial_pipeline.geodata_io import read_geodata
from aggregated_spatial_pipeline.connectpt_data_pipeline.pipeline import build_connectpt_osm_bundle, parse_modalities
from aggregated_spatial_pipeline.visualization import (
    apply_preview_canvas,
    legend_bottom,
    normalize_preview_gdf,
    save_preview_figure,
)


LOG_FORMAT = (
    "<green>{time:DD MMM HH:mm}</green> | "
    "<level>{level: <7}</level> | "
    "<magenta>{extra[tag]}</magenta> "
    "{message}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build connectpt bundle in dedicated connectpt runtime.")
    parser.add_argument("--place", required=True)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speed-kmh", type=float, required=True)
    parser.add_argument("--boundary-path")
    parser.add_argument("--drive-roads-path")
    parser.add_argument("--buildings-path")
    parser.add_argument("--intermodal-nodes-path")
    return parser.parse_args()


def _configure_logging() -> None:
    logger.remove()
    logger.configure(extra={"tag": "[connectpt]"})
    logger.add(
        sys.stderr,
        level="INFO",
        format=LOG_FORMAT,
        colorize=sys.stderr.isatty(),
    )


def _derive_preview_dirs(output_dir: Path) -> tuple[Path, Path | None]:
    local_dir = output_dir / "preview_png"
    shared_dir = None
    if output_dir.name == "connectpt_osm" and output_dir.parent.name not in {"", "."}:
        candidate = output_dir.parent / "preview_png" / "all_together"
        shared_dir = candidate
    return local_dir, shared_dir


def _save_connectpt_previews(manifest: dict, output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    local_dir, shared_dir = _derive_preview_dirs(output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    if shared_dir is not None:
        shared_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}

    boundary_path = manifest.get("boundary")
    if not boundary_path:
        return outputs
    try:
        boundary = read_geodata(Path(boundary_path))
    except Exception:
        return outputs
    if boundary is None or boundary.empty:
        return outputs

    boundary = normalize_preview_gdf(boundary, target_crs="EPSG:3857")

    def _apply_theme(fig, ax, title: str) -> None:
        apply_preview_canvas(fig, ax, boundary, title=title, min_pad=120.0)
        ax.set_axis_off()

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

    for modality in manifest.get("modalities", []):
        modality_name = modality.get("modality", "unknown")
        files = modality.get("files") or {}
        lines = normalize_preview_gdf(read_geodata(Path(files["lines"])), boundary, target_crs="EPSG:3857") if files.get("lines") else None
        projected_lines = normalize_preview_gdf(read_geodata(Path(files["projected_lines"])), boundary, target_crs="EPSG:3857") if files.get("projected_lines") else None
        stops = normalize_preview_gdf(read_geodata(Path(files["aggregated_stops"])), boundary, target_crs="EPSG:3857") if files.get("aggregated_stops") else None
        graph_nodes = normalize_preview_gdf(read_geodata(Path(files["graph_nodes"])), boundary, target_crs="EPSG:3857") if files.get("graph_nodes") else None
        graph_edges = normalize_preview_gdf(read_geodata(Path(files["graph_edges"])), boundary, target_crs="EPSG:3857") if files.get("graph_edges") else None

        if any(g is not None and not g.empty for g in (lines, projected_lines, stops)):
            fig, ax = plt.subplots(figsize=(12, 12))
            legend = []
            _apply_theme(fig, ax, f"ConnectPT {modality_name}")
            if lines is not None and not lines.empty:
                lines.plot(ax=ax, color="#9ca3af", linewidth=0.35, alpha=0.6)
                legend.append(Line2D([0], [0], color="#9ca3af", linewidth=2, label="roads/lines base"))
            if projected_lines is not None and not projected_lines.empty:
                projected_lines.plot(ax=ax, color="#0f766e", linewidth=0.85, alpha=0.95)
                legend.append(Line2D([0], [0], color="#0f766e", linewidth=2, label="routes"))
            if stops is not None and not stops.empty:
                stops.plot(ax=ax, color="#111827", markersize=7, alpha=0.95)
                legend.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markersize=7, label="stops"))
            legend.append(Line2D([0], [0], color="#111111", linewidth=2, label="analysis boundary"))
            legend_bottom(ax, legend)
            _save(fig, f"pt_connectpt_{modality_name}")

        if any(g is not None and not g.empty for g in (graph_nodes, graph_edges)):
            fig, ax = plt.subplots(figsize=(12, 12))
            legend = []
            _apply_theme(fig, ax, f"ConnectPT Graph {modality_name}")
            if graph_edges is not None and not graph_edges.empty:
                graph_edges.plot(ax=ax, color="#0b7285", linewidth=0.4, alpha=0.7)
                legend.append(Line2D([0], [0], color="#0b7285", linewidth=2, label="graph edges"))
            if graph_nodes is not None and not graph_nodes.empty:
                graph_nodes.plot(ax=ax, color="#e03131", markersize=6, alpha=0.9)
                legend.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#e03131", markersize=7, label="graph nodes"))
            legend.append(Line2D([0], [0], color="#111111", linewidth=2, label="analysis boundary"))
            legend_bottom(ax, legend)
            _save(fig, f"pt_connectpt_graph_{modality_name}")

    return outputs


def main() -> None:
    _configure_logging()
    args = parse_args()
    manifest = build_connectpt_osm_bundle(
        place=args.place,
        modalities=parse_modalities(args.modalities),
        output_dir=args.output_dir,
        speed_kmh=float(args.speed_kmh),
        boundary_path=args.boundary_path,
        drive_roads_path=args.drive_roads_path,
        buildings_path=args.buildings_path,
        intermodal_nodes_path=args.intermodal_nodes_path,
    )
    output_dir = Path(args.output_dir).resolve()
    preview_outputs = _save_connectpt_previews(manifest, output_dir)
    if preview_outputs:
        manifest = {**manifest, "preview_outputs": preview_outputs}
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

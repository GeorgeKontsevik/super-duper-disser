from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from aggregated_spatial_pipeline.geodata_io import prepare_geodata_for_parquet, read_geodata

ROOT = Path(__file__).resolve().parents[2]
CONNECTPT_SRC = ROOT / 'connectpt' / 'connectpt'
if str(CONNECTPT_SRC) not in sys.path:
    sys.path.insert(0, str(CONNECTPT_SRC))

from preprocess.stops import aggregate_stops  # noqa: E402

LOG_FORMAT = (
    '<green>{time:DD MMM HH:mm}</green> | '
    '<level>{level: <7}</level> | '
    '<magenta>{extra[tag]}</magenta> '
    '{message}'
)

DEFAULT_DISTANCE_THRESHOLD_M = 30

RAW_TYPE_PREFERENCES = {
    'bus': ['bus', 'platform'],
    'tram': ['tram', 'platform'],
    'trolleybus': ['trolleybus', 'platform'],
    'subway': ['subway', 'subway_platform', 'subway_station', 'subway_entry_exit', 'subway_entry', 'subway_exit', 'platform'],
}


def _configure_logging() -> None:
    logger.remove()
    logger.configure(patcher=lambda record: record['extra'].setdefault('tag', '[log]'))
    logger.add(sys.stderr, level='INFO', format=LOG_FORMAT, colorize=True)


def _log(message: str) -> None:
    logger.bind(tag='[pt-bridge]').info(message)


def _warn(message: str) -> None:
    logger.bind(tag='[pt-bridge]').warning(message)


def slugify_place(place: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', place.lower()).strip('_')
    return slug or 'place'


def _resolve_city_dir(args: argparse.Namespace) -> Path:
    if args.joint_input_dir:
        return Path(args.joint_input_dir).resolve()
    if args.place:
        return (ROOT / 'aggregated_spatial_pipeline' / 'outputs' / 'joint_inputs' / slugify_place(args.place)).resolve()
    raise ValueError('Either --place or --joint-input-dir must be provided.')


def _load_connectpt_manifest(city_dir: Path) -> dict | None:
    path = city_dir / 'connectpt_osm' / 'manifest.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _get_connectpt_modality_entry(manifest: dict | None, modality: str) -> dict | None:
    if not manifest:
        return None
    for item in manifest.get('modalities', []):
        if item.get('modality') == modality:
            return item
    return None


def _extract_iduedu_raw_nodes(city_dir: Path, modality: str) -> tuple[gpd.GeoDataFrame, list[str]]:
    nodes_path = city_dir / 'intermodal_graph_iduedu' / 'graph_nodes.parquet'
    if not nodes_path.exists():
        raise FileNotFoundError(f'Missing iduedu graph nodes: {nodes_path}')
    nodes = read_geodata(nodes_path)
    if nodes.empty:
        raise ValueError(f'iduedu graph nodes are empty: {nodes_path}')
    if 'index' not in nodes.columns:
        nodes = nodes.reset_index(drop=False)
    nodes = nodes.copy()
    nodes['type'] = nodes.get('type', pd.Series([None] * len(nodes), index=nodes.index)).astype('string')
    preferences = RAW_TYPE_PREFERENCES.get(modality, [modality, 'platform'])
    available_types = set(nodes['type'].astype(str))
    selected_types = [node_type for node_type in preferences if node_type in available_types]
    if not selected_types:
        selected_types = [modality]
    primary_types = [selected_types[0]]
    raw = nodes[nodes['type'].astype(str).isin(primary_types)].copy()
    raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
    if raw.empty:
        raise ValueError(f'No candidate iduedu PT nodes found for modality={modality}.')
    raw['raw_iduedu_node_id'] = raw['index'].astype(str)
    raw['modality'] = modality
    if 'name' not in raw.columns:
        raw['name'] = None
    raw = raw[['raw_iduedu_node_id', 'type', 'route', 'name', 'geometry', 'modality']].reset_index(drop=True)
    raw.index = raw['raw_iduedu_node_id']
    return raw, primary_types


def _aggregate_raw_nodes(raw_nodes: gpd.GeoDataFrame, modality: str, distance_threshold: int | None) -> tuple[gpd.GeoDataFrame, pd.DataFrame, int]:
    effective_threshold = int(distance_threshold) if distance_threshold is not None else DEFAULT_DISTANCE_THRESHOLD_M
    agg_input = raw_nodes[['geometry', 'name']].copy()
    simplified = aggregate_stops(
        agg_input,
        distance_threshold=effective_threshold,
        progress_desc=f'iduedu->{modality} bridge aggregation',
    )
    simplified = simplified.reset_index(drop=True)
    simplified['simplified_stop_id'] = [f'{modality}_{i}' for i in range(len(simplified))]
    simplified['modality'] = modality

    records: list[dict] = []
    for _, row in simplified.iterrows():
        simp_id = row['simplified_stop_id']
        centroid = row.geometry
        for raw_id in row['original_ids']:
            raw_geom = raw_nodes.loc[str(raw_id), 'geometry']
            records.append(
                {
                    'raw_iduedu_node_id': str(raw_id),
                    'simplified_stop_id': simp_id,
                    'distance_to_centroid_m': float(raw_geom.distance(centroid)),
                }
            )
    mapping = pd.DataFrame.from_records(records)
    return simplified, mapping, effective_threshold


def _compare_with_connectpt_reference(city_dir: Path, modality: str, simplified: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame | None, dict]:
    ref_path = city_dir / 'connectpt_osm' / modality / 'aggregated_stops.parquet'
    if not ref_path.exists():
        return None, {'reference_available': False}
    reference = read_geodata(ref_path).reset_index(drop=True)
    if reference.empty:
        return reference, {'reference_available': False}
    reference = reference.copy()
    reference['connectpt_stop_id'] = [f'{modality}_ref_{i}' for i in range(len(reference))]
    joined = simplified.sjoin_nearest(
        reference[['connectpt_stop_id', 'name', 'geometry']].rename(columns={'name': 'connectpt_name'}),
        how='left',
        distance_col='distance_to_connectpt_m',
    )
    stats = {
        'reference_available': True,
        'connectpt_reference_count': int(len(reference)),
        'median_distance_to_connectpt_m': float(joined['distance_to_connectpt_m'].median()),
        'mean_distance_to_connectpt_m': float(joined['distance_to_connectpt_m'].mean()),
        'max_distance_to_connectpt_m': float(joined['distance_to_connectpt_m'].max()),
    }
    return joined, stats


def _plot_points(ax, gdf: gpd.GeoDataFrame, *, color: str, size: float, alpha: float, label: str, zorder: int) -> None:
    if gdf is None or gdf.empty:
        return
    gdf.plot(ax=ax, color=color, markersize=size, alpha=alpha, label=label, zorder=zorder)


def _plot_lines(ax, gdf: gpd.GeoDataFrame | None, *, color: str, linewidth: float, alpha: float, label: str | None, zorder: int) -> None:
    if gdf is None or gdf.empty:
        return
    kwargs = {'color': color, 'linewidth': linewidth, 'alpha': alpha, 'zorder': zorder}
    if label is not None:
        kwargs['label'] = label
    gdf.plot(ax=ax, **kwargs)


def _align_to_crs(gdf: gpd.GeoDataFrame | None, target_crs) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty or target_crs is None:
        return gdf
    if gdf.crs is None or str(gdf.crs) == str(target_crs):
        return gdf
    return gdf.to_crs(target_crs)


def _normalize_render_layers(*layers: gpd.GeoDataFrame | None) -> tuple[gpd.GeoDataFrame | None, ...]:
    target_crs = None
    for layer in layers:
        if layer is not None and not layer.empty and layer.crs is not None:
            target_crs = layer.crs
            break
    return tuple(_align_to_crs(layer, target_crs) for layer in layers)


def _plot_boundary_background(ax, boundary: gpd.GeoDataFrame | None, target_crs=None) -> None:
    if boundary is None or boundary.empty:
        return
    boundary = _align_to_crs(boundary, target_crs)
    boundary.plot(ax=ax, color='#f6f0df', edgecolor='white', linewidth=1.2, zorder=-10)


def _save_preview_raw(raw_nodes: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame | None, path: Path) -> None:
    raw_nodes, boundary = _normalize_render_layers(raw_nodes, boundary)
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_boundary_background(ax, boundary, raw_nodes.crs)
    for node_type, color in [('platform', '#1f77b4'), ('bus', '#ff7f0e'), ('tram', '#2ca02c'), ('trolleybus', '#d62728'), ('subway', '#9467bd')]:
        subset = raw_nodes[raw_nodes['type'].astype(str) == node_type]
        if not subset.empty:
            _plot_points(ax, subset, color=color, size=8, alpha=0.75, label=node_type, zorder=5)
    ax.set_title('Raw iduedu PT nodes', fontsize=18, fontweight='bold', color='white')
    ax.set_facecolor('#3a3a3a')
    ax.legend(loc='lower left')
    ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor='#3a3a3a')
    plt.close(fig)


def _save_preview_simplified(raw_nodes: gpd.GeoDataFrame, simplified: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame | None, path: Path) -> None:
    raw_nodes, simplified, boundary = _normalize_render_layers(raw_nodes, simplified, boundary)
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_boundary_background(ax, boundary, raw_nodes.crs)
    _plot_points(ax, raw_nodes, color='#9aa0a6', size=5, alpha=0.35, label='raw iduedu nodes', zorder=3)
    _plot_points(ax, simplified, color='#d62728', size=22, alpha=0.95, label='simplified stops', zorder=6)
    ax.set_title('Simplified PT stops for ConnectPT', fontsize=18, fontweight='bold', color='white')
    ax.set_facecolor('#3a3a3a')
    ax.legend(loc='lower left')
    ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor='#3a3a3a')
    plt.close(fig)


def _save_preview_comparison(simplified_vs_ref: gpd.GeoDataFrame | None, reference: gpd.GeoDataFrame | None, boundary: gpd.GeoDataFrame | None, path: Path) -> None:
    if simplified_vs_ref is None or reference is None or reference.empty:
        return
    simplified_vs_ref, reference, boundary = _normalize_render_layers(simplified_vs_ref, reference, boundary)
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_boundary_background(ax, boundary, reference.crs)
    _plot_points(ax, reference, color='#1f77b4', size=20, alpha=0.8, label='connectpt aggregated stops', zorder=4)
    _plot_points(ax, simplified_vs_ref, color='#d62728', size=18, alpha=0.9, label='iduedu simplified stops', zorder=5)
    if {'geometry', 'index_right'}.issubset(set(simplified_vs_ref.columns)):
        ref_geom = reference.geometry
        for _, row in simplified_vs_ref.iterrows():
            idx_right = row.get('index_right')
            if pd.isna(idx_right):
                continue
            target = ref_geom.iloc[int(idx_right)]
            x1, y1 = row.geometry.x, row.geometry.y
            x2, y2 = target.x, target.y
            ax.plot([x1, x2], [y1, y2], color='#f4d03f', linewidth=0.6, alpha=0.5, zorder=3)
    ax.set_title('Bridge vs ConnectPT reference', fontsize=18, fontweight='bold', color='white')
    ax.set_facecolor('#3a3a3a')
    ax.legend(loc='lower left')
    ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor='#3a3a3a')
    plt.close(fig)


def _save_preview_hist(mapping: pd.DataFrame, simplified: gpd.GeoDataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cluster_sizes = simplified['original_stops'].astype(float)
    axes[0].hist(cluster_sizes, bins=min(20, max(3, int(cluster_sizes.max()))), color='#d62728', alpha=0.85)
    axes[0].set_title('Cluster size')
    axes[1].hist(mapping['distance_to_centroid_m'].astype(float), bins=20, color='#1f77b4', alpha=0.85)
    axes[1].set_title('Raw -> centroid distance (m)')
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle('Bridge aggregation diagnostics', fontsize=16, fontweight='bold')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_preview_network_comparison(
    boundary: gpd.GeoDataFrame | None,
    roads: gpd.GeoDataFrame | None,
    routes: gpd.GeoDataFrame | None,
    iduedu_nodes: gpd.GeoDataFrame,
    connectpt_stops: gpd.GeoDataFrame | None,
    path: Path,
) -> None:
    boundary, roads, routes, iduedu_nodes, connectpt_stops = _normalize_render_layers(
        boundary, roads, routes, iduedu_nodes, connectpt_stops
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_boundary_background(ax, boundary, iduedu_nodes.crs)
    _plot_lines(ax, roads, color='#7d7d7d', linewidth=0.45, alpha=0.35, label='roads', zorder=1)
    _plot_lines(ax, routes, color='#6baed6', linewidth=1.2, alpha=0.45, label='bus routes', zorder=2)
    _plot_points(ax, iduedu_nodes, color='#2b8cbe', size=52, alpha=0.42, label=f'iduedu bus nodes ({len(iduedu_nodes)})', zorder=4)
    if connectpt_stops is not None and not connectpt_stops.empty:
        _plot_points(
            ax,
            connectpt_stops,
            color='#d7301f',
            size=74,
            alpha=0.42,
            label=f'connectpt bus stops ({len(connectpt_stops)})',
            zorder=5,
        )
    ax.set_title('Bus Stops: iduedu vs connectpt', fontsize=22, fontweight='bold', color='white')
    ax.set_facecolor('#3a3a3a')
    ax.legend(loc='lower left', facecolor='white', framealpha=0.95)
    ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor='#3a3a3a')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prototype bridge between iduedu PT nodes and ConnectPT simplified stops.')
    parser.add_argument('--place')
    parser.add_argument('--joint-input-dir')
    parser.add_argument('--modality', default='bus')
    parser.add_argument('--distance-threshold', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = parse_args()
    city_dir = _resolve_city_dir(args)
    if not city_dir.exists():
        raise FileNotFoundError(f'City bundle not found: {city_dir}')
    boundary_path = city_dir / 'analysis_territory' / 'buffer.parquet'
    boundary = read_geodata(boundary_path) if boundary_path.exists() else None
    modality = str(args.modality).strip().lower()

    out_dir = city_dir / 'pt_bridge_prototype' / modality
    prepared_dir = out_dir / 'prepared'
    preview_dir = out_dir / 'previews'
    manifest_path = out_dir / 'manifest.json'
    prepared_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    _log(f'Prototype city bundle: {city_dir.name}')
    _log(f'Prototype modality: {modality}')

    raw_nodes, selected_raw_types = _extract_iduedu_raw_nodes(city_dir, modality)
    _log(f"Raw iduedu candidate nodes: {len(raw_nodes)} (selected types: {', '.join(selected_raw_types)})")

    simplified, mapping, effective_threshold = _aggregate_raw_nodes(raw_nodes, modality, args.distance_threshold)
    _log(f'Simplified bridge stops: {len(simplified)} (distance_threshold={effective_threshold}m)')

    reference_manifest = _load_connectpt_manifest(city_dir)
    modality_entry = _get_connectpt_modality_entry(reference_manifest, modality)
    ref_stats = {}
    reference = None
    simplified_vs_ref = None
    if modality_entry and modality_entry.get('files', {}).get('aggregated_stops'):
        reference = read_geodata(Path(modality_entry['files']['aggregated_stops']))
        simplified_vs_ref, ref_stats = _compare_with_connectpt_reference(city_dir, modality, simplified)
        _log(
            'ConnectPT reference comparison: '
            f"count_ref={ref_stats.get('connectpt_reference_count', 0)}, "
            f"median_distance={ref_stats.get('median_distance_to_connectpt_m', float('nan')):.2f}m"
        )
    else:
        _warn(f'No ConnectPT reference aggregated stops found for modality={modality}.')

    roads = None
    roads_path = city_dir / 'derived_layers' / 'roads_drive_osmnx.parquet'
    if roads_path.exists():
        roads = read_geodata(roads_path)

    routes = None
    if modality_entry:
        projected_lines_path = modality_entry.get('files', {}).get('projected_lines')
        lines_path = modality_entry.get('files', {}).get('lines')
        route_path = Path(projected_lines_path) if projected_lines_path else (Path(lines_path) if lines_path else None)
        if route_path and route_path.exists():
            routes = read_geodata(route_path)

    raw_path = prepared_dir / 'raw_iduedu_pt_nodes.parquet'
    simplified_path = prepared_dir / 'simplified_stops.parquet'
    mapping_path = prepared_dir / 'raw_to_simplified_mapping.parquet'
    ref_compare_path = prepared_dir / 'simplified_to_connectpt_reference.parquet'

    prepare_geodata_for_parquet(raw_nodes.reset_index(drop=True)).to_parquet(raw_path)
    prepare_geodata_for_parquet(simplified).to_parquet(simplified_path)
    mapping.to_parquet(mapping_path, index=False)
    if simplified_vs_ref is not None:
        prepare_geodata_for_parquet(simplified_vs_ref).to_parquet(ref_compare_path)

    _save_preview_raw(raw_nodes, boundary, preview_dir / '01_raw_iduedu_pt_nodes.png')
    _save_preview_simplified(raw_nodes, simplified, boundary, preview_dir / '02_simplified_stops.png')
    _save_preview_comparison(simplified_vs_ref, reference, boundary, preview_dir / '03_bridge_vs_connectpt_reference.png')
    _save_preview_hist(mapping, simplified, preview_dir / '04_bridge_diagnostics.png')
    _save_preview_network_comparison(boundary, roads, routes, raw_nodes, reference, preview_dir / '05_iduedu_vs_connectpt_bus_stops.png')

    manifest = {
        'city_bundle': str(city_dir),
        'modality': modality,
        'distance_threshold_override': args.distance_threshold,
        'effective_distance_threshold_m': effective_threshold,
        'selected_raw_types': selected_raw_types,
        'raw_iduedu_node_count': int(len(raw_nodes)),
        'raw_type_counts': raw_nodes['type'].astype(str).value_counts().to_dict(),
        'simplified_stop_count': int(len(simplified)),
        'compression_ratio': float(len(raw_nodes) / max(1, len(simplified))),
        'cluster_size_stats': {
            'min': int(simplified['original_stops'].min()),
            'median': float(simplified['original_stops'].median()),
            'mean': float(simplified['original_stops'].mean()),
            'max': int(simplified['original_stops'].max()),
        },
        'raw_to_simplified_distance_stats_m': {
            'min': float(mapping['distance_to_centroid_m'].min()),
            'median': float(mapping['distance_to_centroid_m'].median()),
            'mean': float(mapping['distance_to_centroid_m'].mean()),
            'max': float(mapping['distance_to_centroid_m'].max()),
        },
        'connectpt_reference': ref_stats,
        'files': {
            'raw_iduedu_pt_nodes': str(raw_path),
            'simplified_stops': str(simplified_path),
            'raw_to_simplified_mapping': str(mapping_path),
            'simplified_to_connectpt_reference': str(ref_compare_path) if simplified_vs_ref is not None else None,
            'previews_dir': str(preview_dir),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    _log(f'Prototype manifest saved: {manifest_path.name}')


if __name__ == '__main__':
    main()

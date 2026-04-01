"""
Сценарий 1 (OSM): базовый расчёт обеспеченности с OSM-графом вместо arctic transport_df.

Использует bridge.graph_to_arctic_format для конвертации OSMnx-графа в arctic-формат.
"""
import sys
from pathlib import Path

import geopandas as gpd

from pipeline.config import (
    ARCTIC_PATH,
    DATA_PATH,
    SERVICE_NAME,
    SETTL_NAMES,
    RANGE_MONTHS,
    PROVISION_METHOD,
)


def _setup_paths():
    module_path = str(ARCTIC_PATH)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

    data_path_str = str(DATA_PATH / "processed") + "/"
    import scripts.preprocesser.constants as _c
    import scripts.preprocesser.huston as _h
    _c.data_path = data_path_str
    _h.data_path = data_path_str


OSM_BUFFER_M = 500
OSM_CUSTOM_FILTER = (
    '["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|'
    'secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|'
    'living_street|service|pedestrian)$"]["access"!~"^(private|permissive|no)$"]'
    '["service"!~"^(driveway|parking_aisle)$"]'
)


def _get_osm_graph(boundary_wgs84, simplify: bool = True, buffer_m: float = OSM_BUFFER_M):
    import osmnx as ox
    from bridge import ensure_graph_has_time_min

    poly = boundary_wgs84.convex_hull
    if buffer_m and buffer_m > 0:
        gs = gpd.GeoSeries([poly], crs=4326).to_crs(3857)
        poly = gpd.GeoSeries([gs.iloc[0].buffer(buffer_m)], crs=3857).to_crs(4326).iloc[0]
    try:
        G = ox.graph_from_polygon(
            poly, network_type="drive", simplify=simplify, custom_filter=OSM_CUSTOM_FILTER
        )
    except (RuntimeError, IndexError):
        minx, miny, maxx, maxy = gpd.GeoSeries([poly], crs=4326).total_bounds
        bbox = (minx, miny, maxx, maxy)
        G = ox.graph_from_bbox(
            bbox=bbox, network_type="drive", simplify=simplify, custom_filter=OSM_CUSTOM_FILTER
        )
    return ensure_graph_has_time_min(G)


def _get_drive_graph(boundary_wgs84):
    import inspect
    from iduedu import get_drive_graph

    # iduedu API differs by version; keep a strict, signature-based call.
    params = inspect.signature(get_drive_graph).parameters
    if "polygon" in params:
        return get_drive_graph(polygon=boundary_wgs84)
    if "territory" in params:
        return get_drive_graph(territory=boundary_wgs84)
    if "territory_name" in params:
        raise TypeError(
            "iduedu.get_drive_graph in current runtime accepts territory_name/osm_id, "
            "but run_base_osm provides geometry. Update iduedu or pass a compatible runtime."
        )
    return get_drive_graph(boundary_wgs84)


def run(
    settl_names=None,
    service_name=None,
    range_months=None,
):
    import importlib
    import pipeline.config as _cfg
    importlib.reload(_cfg)
    GRAPH_SOURCE = _cfg.GRAPH_SOURCE

    _setup_paths()

    settl_names = settl_names or SETTL_NAMES
    service_name = service_name or SERVICE_NAME
    range_months = range_months or RANGE_MONTHS

    from scripts.calculator.calculator_this_pipeline import make_block_scheme
    from scripts.calculator.calculator_transport_prob import get_transport_probability
    from scripts.calculator.calculator_monthly_mode import create_df_modes_monthly_fixed
    from scripts.calculator.calculator_stat import create_agglomeration_network
    from scripts.preprocesser.preprocesser import get_data
    from scripts.preprocesser.gcreator import make_g, add_temp_to_g
    from scripts.preprocesser.huston import call_nasa
    import scripts.model.provision as provision
    from scripts.preprocesser.constants import (
        CONST_BASE_DEMAND,
        transport_modes,
        transport_mode_name_mapper,
        service_radius_minutes,
        threshold,
        START_YEAR,
        MONTHS_IN_YEAR,
    )
    from scripts.plotter.plotter_transport_mode_prob import plot_transport_probability_legacy
    from scripts.preprocesser.constants import transport_modes_color

    from bridge import graph_to_arctic_format, settl_from_blocks

    threshold_temperatures = plot_transport_probability_legacy(
        transport_modes, transport_modes_color, get_transport_probability,
        threshold, temps=None, font_size=12,
    )

    all_results = {}
    last = {}

    for settl_name in settl_names:
        print("=" * 10, settl_name, f"(OSM/{GRAPH_SOURCE})", "=" * 10)
        all_results[settl_name] = {}

        climate_file = f"df_climate_{settl_name}.csv"
        settl, df_service, _, infr_df = get_data(
            str(DATA_PATH) + "/",
            settl_name,
            transport_mode_name_mapper,
            transport_modes,
            service_name,
        )
        blocks_gdf = make_block_scheme(settl, df_service, service_name=service_name)

        # OSM-граф вместо arctic transport_df (osmnx или iduedu — см. config.GRAPH_SOURCE)
        boundary = settl.geometry.union_all()
        boundary_wgs84 = gpd.GeoSeries([boundary], crs=settl.crs).to_crs(4326).iloc[0]
        use_iduedu = GRAPH_SOURCE == "iduedu"
        if use_iduedu:
            G_drive = _get_drive_graph(boundary_wgs84)
        else:
            G_drive = _get_osm_graph(boundary_wgs84)
        transport_df, _ = graph_to_arctic_format(
            blocks_gdf, G_drive,
            use_iduedu=use_iduedu,
            arctic_compatible=True,
            direct_edges_only=True,
        )
        settl = settl_from_blocks(blocks_gdf)

        G_undirected = make_g(transport_df, transport_modes, blocks_gdf, settl)
        df_monthly_list = call_nasa(blocks_gdf, climate_file)
        G_undirected = add_temp_to_g(G_undirected, df_monthly_list)

        import functools
        _calc = provision.calculate_graph_provision

        @functools.wraps(_calc)
        def _provision_calculator(g, *args, **kwargs):
            kwargs.setdefault("method", PROVISION_METHOD)
            return _calc(g, *args, **kwargs)

        net = create_agglomeration_network(
            graph=G_undirected,
            threshold=threshold,
            probability_function=get_transport_probability,
            provision_calculator=_provision_calculator,
        )
        net.run_all_steps(
            range_months,
            service_radius_minutes=service_radius_minutes[settl_name],
            base_demand=CONST_BASE_DEMAND,
            service_name=service_name,
            return_assignment=True,
        )

        df_stats = net.stats.records
        try:
            df_stats["Month"] = df_stats.index % MONTHS_IN_YEAR + 1
            df_stats["Year"] = START_YEAR + df_stats.index // MONTHS_IN_YEAR
        except Exception:
            pass

        df_modes_monthly = create_df_modes_monthly_fixed(
            G_undirected, transport_modes, threshold_temperatures,
            START_YEAR, MONTHS_IN_YEAR=MONTHS_IN_YEAR,
        )

        all_results[settl_name][service_name] = {
            "net": net,
            "stats": net.stats,
            "graphs": net.stats.graphs,
            "records": net.stats.records,
            "results": net.stats.results,
            "G_undirected": G_undirected,
            "df_modes_monthly": df_modes_monthly,
        }

        last = {
            "settl_name": settl_name,
            "service_name": service_name,
            "blocks_gdf": blocks_gdf,
            "G_undirected": G_undirected,
            "net": net,
            "threshold_temperatures": threshold_temperatures,
        }

    return all_results, last

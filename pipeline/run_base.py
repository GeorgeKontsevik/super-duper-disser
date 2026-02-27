"""
Сценарий 1: базовый расчёт обеспеченности.

Возвращает all_results, blocks_gdf, G_undirected для выбранного settlement + service.
"""
import sys
import importlib
from pathlib import Path

from pipeline.config import (
    ARCTIC_PATH, DATA_PATH, SERVICE_NAME, SETTL_NAMES, RANGE_MONTHS, PROVISION_METHOD,
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


def run(
    settl_names=None,
    service_name=None,
    range_months=None,
):
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
        CONST_BASE_DEMAND, transport_modes, transport_mode_name_mapper,
        service_radius_minutes, threshold, START_YEAR, MONTHS_IN_YEAR,
    )
    from scripts.plotter.plotter_transport_mode_prob import plot_transport_probability_legacy
    from scripts.preprocesser.constants import transport_modes_color

    threshold_temperatures = plot_transport_probability_legacy(
        transport_modes, transport_modes_color, get_transport_probability,
        threshold, temps=None, font_size=12,
    )

    all_results = {}
    last = {}  # возвращаем контекст последнего settlement

    for settl_name in settl_names:
        print("=" * 10, settl_name, "=" * 10)
        all_results[settl_name] = {}

        climate_file = f"df_climate_{settl_name}.csv"
        settl, df_service, transport_df, infr_df = get_data(
            str(DATA_PATH) + "/",
            settl_name,
            transport_mode_name_mapper,
            transport_modes,
            service_name,
        )
        blocks_gdf = make_block_scheme(settl, df_service, service_name=service_name)
        G_undirected = make_g(transport_df, transport_modes, blocks_gdf, settl)
        df_monthly_list = call_nasa(blocks_gdf, climate_file)
        G_undirected = add_temp_to_g(G_undirected, df_monthly_list)

        import functools
        _calc = provision.calculate_graph_provision

        @functools.wraps(_calc)
        def _provision_calculator(g, *args, **kwargs):
            kwargs.setdefault('method', PROVISION_METHOD)
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

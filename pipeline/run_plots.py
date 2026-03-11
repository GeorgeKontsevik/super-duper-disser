"""
Сценарий 4: все визуализации (секции 3–6).

Принимает all_results и параметры из config.
"""
from pipeline.config import (
    SERVICE_NAME,
    MULTILAYER_MONTH,
    SANKEY_MONTH_START, SANKEY_MONTH_END,
    CIRCULAR_MONTH,
    COVERAGE_MONTH_RANGE,
)


def multilayer(all_results, settl_name, service_name=None, month=None):
    from scripts.plotter.plotter_multilayer_service_network import plot_multilayer_network
    service_name = service_name or SERVICE_NAME
    month = month if month is not None else MULTILAYER_MONTH
    return plot_multilayer_network(
        all_results, settl_name, [service_name], month=month, figsize=(15, 30)
    )


def sankey(all_results, settl_name, service_name=None, month_start=None, month_end=None):
    from scripts.plotter.plotter_flow_sankey import create_clean_sankey
    service_name = service_name or SERVICE_NAME
    month_start = month_start if month_start is not None else SANKEY_MONTH_START
    month_end = month_end if month_end is not None else SANKEY_MONTH_END
    graphs = all_results[settl_name][service_name]["stats"].graphs[month_start:month_end]
    return create_clean_sankey(graphs, service_name=service_name, month_start=month_start)


def circular(all_results, settl_name, service_name=None, month=None):
    from scripts.plotter.plotter_circular_network_sankey_style import (
        plot_circular_network_sankey_style,
    )
    from scripts.preprocesser.constants import month_order
    service_name = service_name or SERVICE_NAME
    month = month if month is not None else CIRCULAR_MONTH
    graphs = all_results[settl_name][service_name]["stats"].graphs[month: month + 1]
    figs = []
    for i, g in enumerate(graphs):
        fig = plot_circular_network_sankey_style(
            g, service_name=service_name, month_name=month_order[month + i]
        )
        fig.show(renderer="notebook")
        figs.append(fig)
    return figs


def coverage(all_results, settl_name, service_name=None, month_range=None):
    from scripts.plotter.plotter_multi_temporal_nx_plots import plot_temporal_service_evolution
    service_name = service_name or SERVICE_NAME
    month_range = month_range or COVERAGE_MONTH_RANGE
    # plot_temporal_service_evolution ожидает all_results с ключом service_name
    return plot_temporal_service_evolution(all_results, settl_name, month_range)


def all_plots(all_results, settl_name, service_name=None):
    """Запустить все 4 визуализации подряд."""
    service_name = service_name or SERVICE_NAME
    multilayer(all_results, settl_name, service_name)
    sankey(all_results, settl_name, service_name)
    circular(all_results, settl_name, service_name)
    coverage(all_results, settl_name, service_name)

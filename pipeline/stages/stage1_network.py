"""
Stage 1: Анализ транспортной сети и расчёт доступности.

Использует arctic_access для:
- загрузки данных по поселениям
- построения мультислойного графа с температурными данными
- расчёта провижна (обеспеченности сервисами)

Выход: (G, provision_df, adj_matrix)
  G             — NetworkX граф с атрибутами provision по узлам
  provision_df  — DataFrame с метриками по каждому поселению
  adj_matrix    — матрица кратчайших расстояний (мин.) между узлами
"""

import sys
from pathlib import Path

# Регистрируем arctic_access в пути поиска модулей
ARCTIC_ACCESS_DIR = Path(__file__).parent.parent.parent / "arctic_access"
if str(ARCTIC_ACCESS_DIR) not in sys.path:
    sys.path.insert(0, str(ARCTIC_ACCESS_DIR))

from scripts.preprocesser.preprocesser import get_data
from scripts.preprocesser.gcreator import make_g, add_temp_to_g
from scripts.preprocesser.huston import call_nasa
from scripts.calculator.calculator_this_pipeline import make_block_scheme
from scripts.model.provision import calculate_graph_provision, create_adjacency_matrix
from scripts.preprocesser.constants import (
    CONST_BASE_DEMAND,
    transport_modes,
    transport_mode_name_mapper,
)


def run(config) -> tuple:
    """
    Запускает Stage 1.

    Args:
        config: PipelineConfig

    Returns:
        tuple: (G, provision_df, adj_matrix)
    """
    data_path = str(ARCTIC_ACCESS_DIR / "data") + "/"

    if config.verbose:
        print(f"[Stage 1] Загрузка данных: регион={config.settlement}, "
              f"сервис={config.service}")

    # 1. Загрузка данных
    settl, df_service, transport_df, infr_df = get_data(
        data_path,
        config.settlement,
        transport_mode_name_mapper,
        transport_modes,
        config.service,
    )

    # 2. Формирование blocks_gdf (геометрия + сервисные ёмкости)
    blocks_gdf = make_block_scheme(settl, df_service, service_name=config.service)

    # 3. Построение графа
    G = make_g(transport_df, transport_modes, blocks_gdf, settl)

    # 4. Загрузка климатических данных и добавление температур в граф
    climate_file = f"df_climate_{config.settlement}.csv"
    df_monthly_list = call_nasa(blocks_gdf, climate_file)
    G = add_temp_to_g(G, df_monthly_list)

    # 5. Расчёт провижна
    service_radius = config.resolve_service_radius()

    if config.verbose:
        print(f"[Stage 1] Расчёт провижна (радиус={service_radius} мин, "
              f"сервис={config.service})...")

    G, provision_df = calculate_graph_provision(
        G,
        service_radius=service_radius,
        const_base_demand=CONST_BASE_DEMAND,
        service_name=config.service,
    )

    # 6. Матрица доступности (для Stage 2)
    adj_matrix = create_adjacency_matrix(G)

    if config.verbose:
        n_underserved = (provision_df["provision"] < config.provision_threshold).sum()
        print(f"[Stage 1] Готово. Узлов: {len(provision_df)}, "
              f"неудовлетворённых: {n_underserved} "
              f"(порог провижна={config.provision_threshold})")

    return G, provision_df, adj_matrix

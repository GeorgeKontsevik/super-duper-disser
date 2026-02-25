"""
Stage 2: Оптимизация размещения сервисов (FLP).

Принимает результаты Stage 1:
  - provision_df  — DataFrame с провижном по узлам
  - adj_matrix    — матрица расстояний

Использует solver_flp для:
  - отбора неудовлетворённых блоков (provision < threshold)
  - поиска возможных пар размещения (choose_edges)
  - запуска генетического алгоритма
  - детальной оптимизации через LP (block_coverage)

Выход: (best_solution, fitness_history, coverage_result)
"""

import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

# Регистрируем solver_flp в пути поиска модулей
SOLVER_FLP_DIR = Path(__file__).parent.parent.parent / "solver_flp"
if str(SOLVER_FLP_DIR / "src") not in sys.path:
    sys.path.insert(0, str(SOLVER_FLP_DIR / "src"))

from method import genetic_algorithm_main, choose_edges, block_coverage


def _build_demand_gdf(provision_df: pd.DataFrame, G) -> gpd.GeoDataFrame:
    """
    Строит GeoDataFrame с колонками demand_without и capacity из provision_df и геометрией из G.
    """
    rows = []
    for node, data in G.nodes(data=True):
        name = data.get("name", node)
        row = {
            "id": data.get("id", node),
            "name": name,
            "geometry": data.get("geometry", None),
            "population": data.get("population", 0),
            "capacity": 0,
        }
        if name in provision_df.index:
            row["provision"] = provision_df.loc[name, "provision"]
            row["demand_without"] = provision_df.loc[name, "demand_without"]
        else:
            row["provision"] = 1.0
            row["demand_without"] = 0
        rows.append(row)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry")
    return gdf


def run(G, provision_df: pd.DataFrame, adj_matrix: pd.DataFrame, config) -> tuple:
    """
    Запускает Stage 2.

    Args:
        G:            NetworkX граф (из Stage 1)
        provision_df: DataFrame с провижном (из Stage 1)
        adj_matrix:   Матрица расстояний (из Stage 1)
        config:       PipelineConfig

    Returns:
        tuple: (best_solution, fitness_history, coverage_result)
            best_solution   — список выбранных объектов
            fitness_history — список значений фитнеса по поколениям
            coverage_result — dict с capacities и assignments (из block_coverage)
                              или None если нет неудовлетворённых блоков
    """
    service_radius = config.resolve_service_radius()

    # Фильтруем только неудовлетворённые блоки
    underserved = provision_df[provision_df["provision"] < config.provision_threshold]

    if underserved.empty:
        if config.verbose:
            print("[Stage 2] Все блоки удовлетворены — оптимизация не требуется.")
        return [], [], None

    if config.verbose:
        print(f"[Stage 2] Неудовлетворённых блоков: {len(underserved)}. "
              f"Запуск FLP (поколений={config.generations}, "
              f"популяция={config.population_size})...")

    # Строим GeoDataFrame для solver_flp
    # calculate_fitness внутри GA требует колонки demand_without и capacity
    demand_gdf = _build_demand_gdf(provision_df, G)

    # choose_edges(sim_matrix, service_radius) — ищет пары с distance >= service_radius,
    # которые можно улучшить до <= service_radius при снижении на 40%
    edges = choose_edges(adj_matrix, service_radius)

    if not edges:
        if config.verbose:
            print("[Stage 2] Нет допустимых пар для размещения в радиусе сервиса.")
        return [], [], None

    # Параметры GA
    num_parents = max(2, config.population_size // 5)
    num_offspring = config.population_size - num_parents

    # genetic_algorithm_main(matrix, edges, population_size, num_generations,
    #                        df, service_radius, mutation_rate,
    #                        num_parents, num_offspring, number_res)
    best_solution, fitness_history = genetic_algorithm_main(
        adj_matrix,
        edges,
        config.population_size,
        config.generations,
        demand_gdf,
        service_radius,
        config.mutation_rate,
        num_parents,
        num_offspring,
        "all",
    )

    if config.verbose:
        last_gen = fitness_history[-1]
        best_fitness = min(last_gen) if isinstance(last_gen, list) else last_gen
        print(f"[Stage 2] GA завершён. Лучший фитнес: {best_fitness}")

    # block_coverage(matrix, SERVICE_RADIUS, df, id)
    node_ids = list(adj_matrix.index)
    capacities, res_id = block_coverage(
        adj_matrix.values,
        service_radius,
        demand_gdf.reset_index(drop=True),
        node_ids,
    )
    coverage_result = {"capacities": capacities, "assignments": res_id}

    if config.verbose:
        print("[Stage 2] LP-оптимизация завершена.")

    return best_solution, fitness_history, coverage_result

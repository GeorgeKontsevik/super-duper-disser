"""
Сценарий 2: солвер FLP — поиск мест для новых объектов.

Принимает all_results из run_base.run(), возвращает solver_result.
"""
import sys
import geopandas as gpd
import numpy as np
import pandas as pd

from pipeline.config import (
    SOLVER_PATH, SERVICE_NAME, FLP_MONTH,
    POPULATION_SIZE, NUM_GENERATIONS, NUM_PARENTS, MUTATION_RATE,
)


def _setup_solver_path():
    solver_path = str(SOLVER_PATH)
    if solver_path not in sys.path:
        sys.path.insert(0, solver_path)


def run(all_results, settl_name=None, service_name=None, flp_month=None):
    _setup_solver_path()

    from method import genetic_algorithm_main, choose_edges, block_coverage
    from scripts.model.provision import create_adjacency_matrix
    from scripts.preprocesser.constants import service_radius_minutes

    service_name = service_name or SERVICE_NAME
    flp_month = flp_month if flp_month is not None else FLP_MONTH
    settl_name = settl_name or list(all_results.keys())[0]

    SERVICE_RADIUS = service_radius_minutes[settl_name]
    G_month = all_results[settl_name][service_name]["stats"].graphs[flp_month]

    # Собираем df из атрибутов узлов графа
    node_records = []
    for node, data in G_month.nodes(data=True):
        node_records.append({
            "id": node,
            "name": data.get("name", node),
            "population": data.get("population", 0),
            "demand": data.get("demand", 0),
            "demand_within": data.get("demand_within", 0),
            "demand_without": data.get("demand_without", 0),
            f"capacity_{service_name}": data.get(f"capacity_{service_name}", 0),
            "capacity_left": data.get("capacity_left", 0),
            "provision": data.get("provision", 0),
            "geometry": data.get("geometry", None),
        })

    df_with_demand = gpd.GeoDataFrame(node_records).set_index("id")
    if df_with_demand.geometry.notna().any():
        df_with_demand = df_with_demand.set_geometry("geometry")

    print(f"Всего узлов: {len(df_with_demand)}")
    print(f"Не обеспечены (provision < 1): {(df_with_demand['provision'] < 1).sum()}")

    # Матрица доступности
    adj_matrix = create_adjacency_matrix(G_month)
    uncovered_ids = df_with_demand.index.tolist()

    acc_matrix = adj_matrix.loc[uncovered_ids, uncovered_ids].copy()
    acc_matrix.reset_index(drop=True, inplace=True)
    acc_matrix.columns = acc_matrix.index

    uncovered = df_with_demand.copy()
    uncovered.reset_index(drop=True, inplace=True)
    uncovered.rename(columns={f"capacity_{service_name}": "capacity"}, inplace=True)

    # Генетический алгоритм
    num_offspring = POPULATION_SIZE - NUM_PARENTS
    edges = choose_edges(sim_matrix=acc_matrix, service_radius=SERVICE_RADIUS)
    print(f"Рёбер в графе кандидатов: {len(edges)}")

    best_candidate, fitness_history = genetic_algorithm_main(
        matrix=acc_matrix,
        edges=edges,
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        df=uncovered,
        service_radius=SERVICE_RADIUS,
        mutation_rate=MUTATION_RATE,
        num_parents=NUM_PARENTS,
        num_offspring=num_offspring,
        number_res="all",
    )

    capacities, res_id = block_coverage(
        best_candidate, SERVICE_RADIUS, uncovered, uncovered_ids
    )

    print("Вместимости новых объектов:", [c for c in capacities if c and c > 0])
    print("Привязка:", res_id)

    return {
        "best_candidate": best_candidate,
        "fitness_history": fitness_history,
        "capacities": capacities,
        "res_id": res_id,
        "df_with_demand": df_with_demand,
        "acc_matrix": acc_matrix,
        "uncovered": uncovered,
        "uncovered_ids": uncovered_ids,
        "SERVICE_RADIUS": SERVICE_RADIUS,
        "settl_name": settl_name,
        "service_name": service_name,
    }

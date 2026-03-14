"""
Сценарий 2: солвер FLP — поиск мест для новых объектов.

Принимает all_results из run_base.run(), возвращает solver_result.
"""
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.config import (
    ARCTIC_PATH, SOLVER_PATH, SERVICE_NAME, FLP_MONTH,
    EDGE_IMPROVEMENT_FACTOR, CONNECTIVITY_MODE,
    POPULATION_SIZE, NUM_GENERATIONS, NUM_PARENTS, MUTATION_RATE,
)


def _create_adjacency_matrix_with_modes(G):
    """
    Матрица доступности + path_modes: для каждой пары (i,j) — множество режимов,
    использованных в кратчайшем пути.
    """
    import networkx as nx

    nodes = sorted(G.nodes())
    matrix = pd.DataFrame(float("inf"), index=nodes, columns=nodes)
    np.fill_diagonal(matrix.values, 0)
    path_modes = {}

    def _get_edge_modes(G, u, v):
        """Режимы рёбер между u и v (для MultiGraph — все рёбра, для Graph — одно)."""
        modes = set()
        if G.is_multigraph():
            for key, data in G[u][v].items():
                lbl = data.get("label")
                if lbl:
                    modes.add(lbl)
        else:
            lbl = G[u][v].get("label")
            if lbl:
                modes.add(lbl)
        return modes

    for source in tqdm(nodes, desc="path_modes"):
        try:
            lengths, paths = nx.single_source_dijkstra(G, source, weight="weight")
            for target, dist in lengths.items():
                matrix.loc[source, target] = dist
                if source != target and target in paths:
                    path = paths[target]
                    modes = set()
                    for k in range(len(path) - 1):
                        modes.update(_get_edge_modes(G, path[k], path[k + 1]))
                    path_modes[(source, target)] = modes
                    path_modes[(target, source)] = modes
        except nx.NetworkXNoPath:
            pass

    return matrix, path_modes


def _choose_edges(sim_matrix, service_radius, improvement_factor=0.6):
    """
    Кандидаты рёбер для улучшения: пары (i,j), где при уменьшении времени
    в improvement_factor раз попадаем в service_radius.
    """
    edges = []
    for i in tqdm(sim_matrix.index):
        for j in sim_matrix.columns:
            if sim_matrix.loc[i, j] >= service_radius and i != j:
                val = sim_matrix.loc[i, j]
                if val > service_radius and val * improvement_factor <= service_radius:
                    if [j, i] not in edges:
                        edges.append([i, j])
    return edges


def _setup_paths():
    for path in (str(ARCTIC_PATH), str(SOLVER_PATH)):
        if path not in sys.path:
            sys.path.insert(0, path)


def run(all_results, settl_name=None, service_name=None, flp_month=None):
    _setup_paths()

    from method import genetic_algorithm_main, block_coverage
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

    # Матрица доступности + path_modes (режимы на рёбрах кратчайших путей)
    adj_matrix, path_modes = _create_adjacency_matrix_with_modes(G_month)
    uncovered_ids = df_with_demand.index.tolist()

    acc_matrix = adj_matrix.loc[uncovered_ids, uncovered_ids].copy()
    acc_matrix.reset_index(drop=True, inplace=True)
    acc_matrix.columns = acc_matrix.index

    uncovered = df_with_demand.copy()
    uncovered.reset_index(drop=True, inplace=True)
    uncovered.rename(columns={f"capacity_{service_name}": "capacity"}, inplace=True)

    # Кандидаты рёбер (improvement_factor)
    edges = _choose_edges(
        sim_matrix=acc_matrix,
        service_radius=SERVICE_RADIUS,
        improvement_factor=EDGE_IMPROVEMENT_FACTOR,
    )
    # Фильтр по CONNECTIVITY_MODE: в солвер передаём только пары, где путь использует этот режим
    if CONNECTIVITY_MODE:
        id_to_name = {i: uncovered_ids[i] for i in range(len(uncovered_ids))}
        edges = [
            e
            for e in edges
            if CONNECTIVITY_MODE
            in (
                path_modes.get((id_to_name[e[0]], id_to_name[e[1]]))
                or path_modes.get((id_to_name[e[1]], id_to_name[e[0]]))
                or set()
            )
        ]
    print(f"Рёбер в графе кандидатов: {len(edges)}")

    if not edges:
        print("Нет рёбер-кандидатов — солвер нечего оптимизировать, пропускаем.")
        return {
            "best_candidate": [],
            "fitness_history": [],
            "capacities": [],
            "res_id": [],
            "df_with_demand": df_with_demand,
            "acc_matrix": acc_matrix,
            "uncovered": uncovered,
            "uncovered_ids": uncovered_ids,
            "SERVICE_RADIUS": SERVICE_RADIUS,
            "settl_name": settl_name,
            "service_name": service_name,
        }

    num_offspring = POPULATION_SIZE - NUM_PARENTS
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

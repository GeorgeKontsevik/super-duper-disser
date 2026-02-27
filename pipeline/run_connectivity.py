"""
Сценарий 3: улучшение сервисов и/или связности + пересчёт.

Три функции:
  apply_services(...)     — добавляет новые объекты в граф
  apply_connectivity(...) — подменяет матрицу доступности из best_candidate солвера
  apply_both(...)         — оба улучшения вместе
"""
import copy
import functools
import pandas as pd

from pipeline.config import SERVICE_NAME, RANGE_MONTHS, PROVISION_METHOD


def _rerun(G, settl_name, service_name, range_months, adj_matrix=None, method=None):
    from scripts.calculator.calculator_stat import create_agglomeration_network
    from scripts.calculator.calculator_transport_prob import get_transport_probability
    import scripts.model.provision as provision
    from scripts.preprocesser.constants import CONST_BASE_DEMAND, service_radius_minutes, threshold

    method = method or PROVISION_METHOD

    @functools.wraps(provision.calculate_graph_provision)
    def provision_calculator(g, *args, **kwargs):
        kwargs.setdefault('method', method)
        if adj_matrix is not None:
            kwargs['adj_matrix'] = adj_matrix
        return provision.calculate_graph_provision(g, *args, **kwargs)


    net = create_agglomeration_network(
        graph=G,
        threshold=threshold,
        probability_function=get_transport_probability,
        provision_calculator=provision_calculator,
    )
    net.run_all_steps(
        range_months,
        service_radius_minutes=service_radius_minutes[settl_name],
        base_demand=CONST_BASE_DEMAND,
        service_name=service_name,
        return_assignment=True,
    )
    return net


def _compare(all_results, settl_name, service_name, key_after, flp_month=0):
    G_before = all_results[settl_name][service_name]["stats"].graphs[flp_month]
    G_after = all_results[settl_name][key_after]["stats"].graphs[flp_month]

    df = (
        pd.DataFrame([{"name": n, "provision_before": d.get("provision", 0)}
                      for n, d in G_before.nodes(data=True)]).set_index("name")
        .join(pd.DataFrame([{"name": n, "provision_after": d.get("provision", 0)}
                            for n, d in G_after.nodes(data=True)]).set_index("name"))
    )
    df["delta"] = df["provision_after"] - df["provision_before"]
    return df.sort_values("delta", ascending=False)


def _get_improved_edges(solver_result):
    best_candidate = solver_result["best_candidate"]
    acc_matrix = solver_result["acc_matrix"]
    uncovered_ids = solver_result["uncovered_ids"]

    improved = []
    for num_i, name_i in enumerate(uncovered_ids):
        for num_j, name_j in enumerate(uncovered_ids):
            if num_i >= num_j:
                continue
            orig = acc_matrix.loc[num_i, num_j]
            new_val = best_candidate.loc[num_i, num_j]
            if abs(orig - new_val) > 1e-6 and orig < float("inf"):
                improved.append((name_i, name_j, orig, new_val))
    return improved


def apply_services(all_results, solver_result, base_ctx, range_months=None):
    """
    Добавляет новые объекты из солвера в граф и пересчитывает.
    Результат → all_results[settl_name]['{service}_services'].
    """
    range_months = range_months or RANGE_MONTHS
    settl_name = solver_result["settl_name"]
    service_name = solver_result["service_name"]
    blocks_gdf = base_ctx["blocks_gdf"]
    G_undirected = base_ctx["G_undirected"]

    capacities = solver_result["capacities"]
    res_id = solver_result["res_id"]

    new_capacities = [c for c in capacities if c and c > 0]
    G_new = copy.deepcopy(G_undirected)
    for facility_name, capacity in zip(res_id.keys(), new_capacities):
        mask = blocks_gdf["name"] == facility_name
        blocks_gdf.loc[mask, f"capacity_{service_name}"] += capacity
        if facility_name in G_new.nodes:
            G_new.nodes[facility_name][f"capacity_{service_name}"] = \
                blocks_gdf.loc[mask, f"capacity_{service_name}"].values[0]
        print(f"  {facility_name}: +{capacity:.0f} capacity")

    key = f"{service_name}_services"
    net = _rerun(G_new, settl_name, service_name, range_months)
    all_results[settl_name][key] = {
        "net": net, "stats": net.stats,
        "graphs": net.stats.graphs, "records": net.stats.records,
        "results": net.stats.results, "G_undirected": G_new,
    }

    flp_month = solver_result.get("flp_month", 0)
    df_compare = _compare(all_results, settl_name, service_name, key, flp_month)
    print(f"\nПересчёт с новыми объектами завершён → ключ '{key}'")
    return all_results, {"net": net, "G": G_new, "df_compare": df_compare, "key": key}


def apply_connectivity(all_results, solver_result, base_ctx, range_months=None):
    """
    Подменяет матрицу доступности значениями из best_candidate солвера и пересчитывает.
    Граф не меняется — только матрица расстояний.
    Результат → all_results[settl_name]['{service}_connectivity'].
    """
    range_months = range_months or RANGE_MONTHS
    settl_name = solver_result["settl_name"]
    service_name = solver_result["service_name"]
    G_undirected = base_ctx["G_undirected"]
    best_candidate = solver_result["best_candidate"]
    uncovered_ids = solver_result["uncovered_ids"]

    improved_edges = _get_improved_edges(solver_result)
    print(f"Улучшено пар узлов: {len(improved_edges)}")
    for u, v, old_t, new_t in improved_edges:
        print(f"  {u} ↔ {v}: {old_t:.1f} → {new_t:.1f} мин (−{(1 - new_t/old_t)*100:.0f}%)")

    # Восстанавливаем полную матрицу с именами узлов
    from scripts.model.provision import create_adjacency_matrix
    adj_matrix = create_adjacency_matrix(G_undirected)
    for num_i, name_i in enumerate(uncovered_ids):
        for num_j, name_j in enumerate(uncovered_ids):
            val = best_candidate.loc[num_i, num_j]
            adj_matrix.loc[name_i, name_j] = val
            adj_matrix.loc[name_j, name_i] = val

    key = f"{service_name}_connectivity"
    net = _rerun(G_undirected, settl_name, service_name, range_months, adj_matrix=adj_matrix)
    all_results[settl_name][key] = {
        "net": net, "stats": net.stats,
        "graphs": net.stats.graphs, "records": net.stats.records,
        "results": net.stats.results, "G_undirected": G_undirected,
    }

    flp_month = solver_result.get("flp_month", 0)
    df_compare = _compare(all_results, settl_name, service_name, key, flp_month)
    print(f"\nПересчёт с улучшенной связностью завершён → ключ '{key}'")
    return all_results, {
        "net": net, "G": G_undirected,
        "df_compare": df_compare,
        "improved_edges": improved_edges,
        "key": key,
    }


def apply_both(all_results, solver_result, base_ctx, range_months=None):
    """
    Оба улучшения вместе: новые объекты + подмена матрицы доступности из best_candidate.
    Результат → all_results[settl_name]['{service}_both'].
    """
    range_months = range_months or RANGE_MONTHS
    settl_name = solver_result["settl_name"]
    service_name = solver_result["service_name"]
    G_undirected = base_ctx["G_undirected"]
    best_candidate = solver_result["best_candidate"]
    uncovered_ids = solver_result["uncovered_ids"]

    capacities = solver_result["capacities"]
    res_id = solver_result["res_id"]

    # 1. Новые объекты — только в граф, не мутируем blocks_gdf
    new_capacities = [c for c in capacities if c and c > 0]
    G_new = copy.deepcopy(G_undirected)
    for facility_name, capacity in zip(res_id.keys(), new_capacities):
        if facility_name in G_new.nodes:
            current = G_new.nodes[facility_name].get(f"capacity_{service_name}", 0)
            G_new.nodes[facility_name][f"capacity_{service_name}"] = current + capacity
            print(f"  {facility_name}: +{capacity:.0f} capacity")

    # 2. Подменяем матрицу доступности из best_candidate
    from scripts.model.provision import create_adjacency_matrix
    adj_matrix = create_adjacency_matrix(G_undirected)
    for num_i, name_i in enumerate(uncovered_ids):
        for num_j, name_j in enumerate(uncovered_ids):
            val = best_candidate.loc[num_i, num_j]
            adj_matrix.loc[name_i, name_j] = val
            adj_matrix.loc[name_j, name_i] = val

    key = f"{service_name}_both"
    net = _rerun(G_new, settl_name, service_name, range_months, adj_matrix=adj_matrix)
    all_results[settl_name][key] = {
        "net": net, "stats": net.stats,
        "graphs": net.stats.graphs, "records": net.stats.records,
        "results": net.stats.results, "G_undirected": G_new,
    }

    flp_month = solver_result.get("flp_month", 0)
    df_compare = _compare(all_results, settl_name, service_name, key, flp_month)
    print(f"Пересчёт с обоими улучшениями завершён → ключ '{key}'")
    return all_results, {
        "net": net, "G": G_new,
        "df_compare": df_compare,
        "key": key,
    }

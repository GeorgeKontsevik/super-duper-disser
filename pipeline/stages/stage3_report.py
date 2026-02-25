"""
Stage 3: Формирование итогового отчёта и визуализаций.

Принимает результаты Stage 1 и Stage 2, сохраняет:
  - CSV с метриками провижна
  - plots: fitness convergence, service map, provision heatmap
"""

import sys
from pathlib import Path

import pandas as pd

# Регистрируем solver_flp для плоттеров
SOLVER_FLP_DIR = Path(__file__).parent.parent.parent / "solver_flp"
if str(SOLVER_FLP_DIR / "src") not in sys.path:
    sys.path.insert(0, str(SOLVER_FLP_DIR / "src"))

from method import fitness_plot, services_plot


def run(G, provision_df, best_solution, fitness_history, coverage_result, config):
    """
    Запускает Stage 3.

    Args:
        G:               NetworkX граф с атрибутами provision
        provision_df:    DataFrame с метриками провижна
        best_solution:   Результат GA из Stage 2
        fitness_history: История фитнеса из Stage 2
        coverage_result: Результат LP из Stage 2
        config:          PipelineConfig
    """
    out_dir: Path = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{config.settlement}_{config.service}_{config.year}"

    # 1. Сохранение провижна в CSV
    csv_path = out_dir / f"provision_{tag}.csv"
    provision_df.to_csv(csv_path)
    if config.verbose:
        print(f"[Stage 3] Провижн сохранён: {csv_path}")

    # 2. Сводная статистика
    stats = {
        "settlement": config.settlement,
        "service": config.service,
        "year": config.year,
        "month": config.month,
        "mean_provision": provision_df["provision"].mean(),
        "min_provision": provision_df["provision"].min(),
        "max_provision": provision_df["provision"].max(),
        "n_underserved": (provision_df["provision"] < config.provision_threshold).sum(),
        "n_total": len(provision_df),
    }
    stats_df = pd.DataFrame([stats])
    stats_path = out_dir / f"stats_{tag}.csv"
    stats_df.to_csv(stats_path, index=False)

    if config.verbose:
        print(f"[Stage 3] Статистика:")
        print(f"  Средний провижн:     {stats['mean_provision']:.3f}")
        print(f"  Неудовлетворённых:   {stats['n_underserved']} / {stats['n_total']}")

    if not config.save_plots:
        return stats

    # 3. График сходимости GA
    if fitness_history:
        try:
            fig = fitness_plot(fitness_history)
            fig.savefig(out_dir / f"fitness_{tag}.png", dpi=150, bbox_inches="tight")
            if config.verbose:
                print(f"[Stage 3] График фитнеса сохранён.")
        except Exception as e:
            print(f"[Stage 3] Не удалось сохранить fitness plot: {e}")

    # 4. Карта сервисов
    if best_solution and coverage_result is not None:
        try:
            import geopandas as gpd
            # Геометрия узлов из графа
            rows = [
                {"name": d.get("name", n), "geometry": d.get("geometry")}
                for n, d in G.nodes(data=True)
                if d.get("geometry") is not None
            ]
            if rows:
                gdf = gpd.GeoDataFrame(rows, geometry="geometry")
                fig = services_plot(
                    gdf=gdf,
                    best_solution=best_solution,
                    coverage_result=coverage_result,
                )
                fig.savefig(out_dir / f"services_{tag}.png", dpi=150, bbox_inches="tight")
                if config.verbose:
                    print(f"[Stage 3] Карта сервисов сохранена.")
        except Exception as e:
            print(f"[Stage 3] Не удалось сохранить services plot: {e}")

    return stats

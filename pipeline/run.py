"""
Главный CLI entry point объединённого пайплайна.

Использование:
    python -m pipeline.run [опции]

Примеры:
    python -m pipeline.run
    python -m pipeline.run --settlement yanao_kras --service health --year 2024
    python -m pipeline.run --settlement mezen --service airport --month 1 --no-plots
"""

import argparse
import sys
from pathlib import Path

from pipeline.config import PipelineConfig, OUTPUTS_DIR
from pipeline.stages import stage1_network, stage2_optimize, stage3_report


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Arctic Service Accessibility & Optimization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--settlement",
        choices=["yanao_kras", "mezen", "yakut_chuk", "nao"],
        default="yanao_kras",
        help="Регион/поселение для анализа",
    )
    parser.add_argument(
        "--service",
        choices=["health", "airport", "port", "culture", "post", "marina"],
        default="health",
        help="Тип сервиса",
    )
    parser.add_argument("--year", type=int, default=2024, help="Год анализа")
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        choices=range(1, 13),
        metavar="1-12",
        help="Месяц (не указывать = среднегодовой)",
    )
    parser.add_argument(
        "--service-radius",
        type=int,
        default=None,
        dest="service_radius",
        help="Порог доступности в минутах (по умолчанию — из региона)",
    )
    parser.add_argument(
        "--provision-threshold",
        type=float,
        default=0.55,
        dest="provision_threshold",
        help="Порог провижна (ниже — блок неудовлетворён)",
    )
    parser.add_argument(
        "--generations", type=int, default=20, help="Число поколений GA"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        dest="population_size",
        help="Размер популяции GA",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.7,
        dest="mutation_rate",
        help="Вероятность мутации GA",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR,
        dest="output_dir",
        help="Директория для результатов",
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="save_plots",
        help="Не сохранять графики",
    )
    parser.add_argument(
        "--quiet", action="store_false", dest="verbose", help="Тихий режим"
    )

    args = parser.parse_args()

    config = PipelineConfig(
        settlement=args.settlement,
        service=args.service,
        year=args.year,
        month=args.month,
        service_radius=args.service_radius if args.service_radius else 90,
        provision_threshold=args.provision_threshold,
        generations=args.generations,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        output_dir=args.output_dir,
        save_plots=args.save_plots,
        verbose=args.verbose,
    )

    return config


def main():
    config = parse_args()

    if config.verbose:
        print("=" * 60)
        print("  Arctic Service Accessibility & Optimization Pipeline")
        print("=" * 60)
        print(f"  Регион:  {config.settlement}")
        print(f"  Сервис:  {config.service}")
        print(f"  Год:     {config.year}")
        print(f"  Месяц:   {config.month or 'среднегодовой'}")
        print("=" * 60)

    # Stage 1: сеть + доступность
    G, provision_df, adj_matrix = stage1_network.run(config)

    # Stage 2: оптимизация FLP
    best_solution, fitness_history, coverage_result = stage2_optimize.run(
        G, provision_df, adj_matrix, config
    )

    # Stage 3: отчёт
    stats = stage3_report.run(
        G, provision_df, best_solution, fitness_history, coverage_result, config
    )

    if config.verbose:
        print("=" * 60)
        print("  Готово.")
        print("=" * 60)

    return stats


if __name__ == "__main__":
    main()

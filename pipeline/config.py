"""
Pipeline configuration — параметры запуска объединённого пайплайна.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
ARCTIC_ACCESS_DIR = ROOT_DIR / "arctic_access"
SOLVER_FLP_DIR = ROOT_DIR / "solver_flp"
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"


@dataclass
class PipelineConfig:
    # --- Stage 1: сеть и доступность ---
    settlement: str = "yanao_kras"          # yanao_kras | mezen | yakut_chuk | nao
    service: str = "health"                 # health | airport | port | culture | post | marina
    year: int = 2024
    month: Optional[int] = None            # None = среднегодовой, 1-12 = конкретный месяц

    # --- Stage 2: оптимизация FLP ---
    service_radius: int = 90               # порог доступности в минутах
    population_size: int = 50              # размер популяции генетического алгоритма
    generations: int = 20                  # число поколений
    mutation_rate: float = 0.7             # вероятность мутации
    provision_threshold: float = 0.55     # порог провижна (ниже — блок неудовлетворён)

    # --- Вывод ---
    output_dir: Path = field(default_factory=lambda: OUTPUTS_DIR)
    save_plots: bool = True
    verbose: bool = True

    # Сервисные радиусы по умолчанию для регионов (минуты)
    SERVICE_RADIUS_BY_REGION = {
        "yakut_chuk": 180,
        "yanao_kras": 90,
        "mezen": 60,
        "nao": 180,
    }

    def resolve_service_radius(self) -> int:
        """Вернуть радиус сервиса: явно заданный или дефолтный для региона."""
        if self.service_radius is not None:
            return self.service_radius
        return self.SERVICE_RADIUS_BY_REGION.get(self.settlement, 90)

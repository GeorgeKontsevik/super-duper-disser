"""
Конфигурация пайплайна. Меняй только этот файл перед запуском.
"""
from pathlib import Path

# ─── Пути ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
ARCTIC_PATH = ROOT / "arctic_access"
DATA_PATH = ARCTIC_PATH / "data"
SOLVER_PATH = ROOT / "solver_flp" / "src"

# ─── Сценарий ─────────────────────────────────────────────────────────────────
SETTL_NAMES = ["yanao_kras"]   # yakut_chuk | yanao_kras | mezen | nao
SERVICE_NAME = "health"         # health | post | culture | port | airport | marina
RANGE_MONTHS = range(12)        # диапазон месяцев для расчёта

# ─── Модель провижена ─────────────────────────────────────────────────────────
# "lp_distance"  — LP минимизация суммарного расстояния (текущая arctic модель)
# "lp_coverage"  — LP покрытие спроса по бинарной матрице доступности (солверная логика)
PROVISION_METHOD = "lp_coverage"

# ─── Улучшение связности ──────────────────────────────────────────────────────
# Модальность для новых рёбер: "Aviation" | "Winter road" | "Regular road" | "Water transport"
CONNECTIVITY_MODE = "Aviation"

# ─── Солвер FLP ───────────────────────────────────────────────────────────────
FLP_MONTH = 0          # снапшот какого месяца подаём в солвер

POPULATION_SIZE = 50
NUM_GENERATIONS = 20
NUM_PARENTS = 10
MUTATION_RATE = 0.7

# ─── Визуализация ─────────────────────────────────────────────────────────────
MULTILAYER_MONTH = 0
SANKEY_MONTH_START = 0
SANKEY_MONTH_END = 10
CIRCULAR_MONTH = 0
COVERAGE_MONTH_RANGE = range(4, 8, 3)

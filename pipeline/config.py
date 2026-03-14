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

# ─── Источник графа (для основного ноутбука) ───────────────────────────────────
# False — Arctic (df_time из processed)
# True  — OSM/iduedu с direct_edges_only (для notebook_osm)
USE_OSM_GRAPH = True

# ─── OSM-граф (run_base_osm) ─────────────────────────────────────────────────
# "osmnx"  — OSMnx graph_from_polygon
# "iduedu" — IduEdu get_drive_graph (pip install iduedu)
GRAPH_SOURCE = "iduedu"

# ─── Модель провижена ─────────────────────────────────────────────────────────
# "lp_distance"  — LP минимизация суммарного расстояния (текущая arctic модель)
# "lp_coverage"  — LP покрытие спроса по бинарной матрице доступности (солверная логика)
PROVISION_METHOD = "lp_coverage"

# ─── Улучшение связности ──────────────────────────────────────────────────────
# Модальность для новых рёбер: "Aviation" | "Winter road" | "Regular road" | "Water transport"
CONNECTIVITY_MODE = "Regular road"

# ─── Солвер FLP ───────────────────────────────────────────────────────────────
FLP_MONTH = 0          # снапшот какого месяца подаём в солвер

# Множитель для кандидатов рёбер: пара (i,j) попадает в выбор, если
# sim_matrix[i,j] * EDGE_IMPROVEMENT_FACTOR <= service_radius
# (0.6 = "если уменьшить время на 40%, попадаем в радиус")
EDGE_IMPROVEMENT_FACTOR = 0.9

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

from pathlib import Path
import json
import pandas as pd

ROOT = Path('/Users/gk/Code/super-duper-disser')
JOINT_ROOT = ROOT / 'aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs'
OUT_PATH = ROOT / 'aggregated_spatial_pipeline/outputs/experiments_active19_20260412/README_ACTIVE19_SERVICE_GAPS.md'

CITIES = [
    'bergen_norway',
    'bologna_italy',
    'bristol_united_kingdom',
    'brno_czechia',
    'coimbra_portugal',
    'debrecen_hungary',
    'dresden_germany',
    'freiburg_im_breisgau_germany',
    'gothenburg_sweden',
    'graz_austria',
    'innsbruck_austria',
    'krakow_poland',
    'linz_austria',
    'lyon_france',
    'marseille_france',
    'porto_portugal',
    'turin_italy',
    'turku_finland',
    'zaragoza_spain',
]
SERVICES = ['hospital', 'polyclinic', 'school', 'kindergarten']

STATUS_LABEL = {
    'ok': 'OK',
    'accessibility': 'access',
    'capacity': 'capacity',
    'both': 'both',
    'missing': 'missing',
}
STATUS_BADGE = {
    'ok': '🟢 OK',
    'accessibility': '🔵 access',
    'capacity': '🔴 capacity',
    'both': '🟣 both',
    'missing': '⚪ missing',
}


def fmt_num(v):
    if v is None:
        return '-'
    return f'{v:,.0f}'


def classify(access_gap: float, capacity_gap: float) -> str:
    if access_gap > 1e-9 and capacity_gap > 1e-9:
        return 'both'
    if access_gap > 1e-9:
        return 'accessibility'
    if capacity_gap > 1e-9:
        return 'capacity'
    return 'ok'


def collect_rows():
    rows = []
    counts = {'ok': 0, 'accessibility': 0, 'capacity': 0, 'both': 0, 'missing': 0}
    for city in CITIES:
        city_dir = JOINT_ROOT / city / 'pipeline_2' / 'solver_inputs'
        for service in SERVICES:
            summary_path = city_dir / service / 'summary.json'
            parquet_path = city_dir / service / 'blocks_solver.parquet'
            if not summary_path.exists() or not parquet_path.exists():
                status = 'missing'
                counts[status] += 1
                rows.append({
                    'city': city,
                    'service': service,
                    'status': status,
                    'demand_total': None,
                    'accessibility_gap_total': None,
                    'capacity_gap_total': None,
                })
                continue

            with summary_path.open() as fh:
                summary = json.load(fh)
            df = pd.read_parquet(parquet_path)
            access_gap = float(pd.to_numeric(df.get('demand_without', 0.0), errors='coerce').fillna(0.0).sum())
            capacity_gap = float(pd.to_numeric(df.get('demand_left', 0.0), errors='coerce').fillna(0.0).sum())
            status = classify(access_gap, capacity_gap)
            counts[status] += 1
            rows.append({
                'city': city,
                'service': service,
                'status': status,
                'demand_total': float(summary.get('demand_total', 0.0) or 0.0),
                'accessibility_gap_total': access_gap,
                'capacity_gap_total': capacity_gap,
            })
    return rows, counts


def build_readme(rows, counts) -> str:
    lines = []
    lines.append('# Active19 Service Gap Matrix')
    lines.append('')
    lines.append('Сводка по 19 городам и 4 сервисам из `pipeline_2/solver_inputs`.')
    lines.append('')
    lines.append('Скрипт генерации:')
    lines.append('- [build_active19_service_gaps_readme.py](/Users/gk/Code/super-duper-disser/scripts/build_active19_service_gaps_readme.py)')
    lines.append('')
    lines.append('Команда пересборки:')
    lines.append('```bash')
    lines.append('cd /Users/gk/Code/super-duper-disser')
    lines.append('./.venv/bin/python scripts/build_active19_service_gaps_readme.py')
    lines.append('```')
    lines.append('')
    lines.append('Источник на каждую ячейку:')
    lines.append('- `summary.json` для `demand_total`')
    lines.append('- `blocks_solver.parquet` для сумм `demand_without` и `demand_left`')
    lines.append('')
    lines.append('Правило классификации:')
    lines.append('- `🟢 OK`: `demand_without = 0` и `demand_left = 0`')
    lines.append('- `🔵 access`: необеспеченность из-за доступности, `demand_without > 0`, `demand_left = 0`')
    lines.append('- `🔴 capacity`: необеспеченность из-за capacity, `demand_left > 0`, `demand_without = 0`')
    lines.append('- `🟣 both`: присутствуют оба типа разрыва')
    lines.append('- `⚪ missing`: не найден `summary.json` или `blocks_solver.parquet`')
    lines.append('')
    lines.append('Итоги по всем 76 city-service комбинациям:')
    lines.append(f"- `OK`: {counts['ok']}")
    lines.append(f"- `access`: {counts['accessibility']}")
    lines.append(f"- `capacity`: {counts['capacity']}")
    lines.append(f"- `both`: {counts['both']}")
    lines.append(f"- `missing`: {counts['missing']}")
    lines.append('')
    lines.append('## Matrix')
    lines.append('')
    lines.append('| City | Hospital | Polyclinic | School | Kindergarten |')
    lines.append('| --- | --- | --- | --- | --- |')
    for city in CITIES:
        by_service = {r['service']: r for r in rows if r['city'] == city}
        vals = [STATUS_BADGE[by_service[s]['status']] for s in SERVICES]
        lines.append(f'| {city} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |')
    lines.append('')
    lines.append('## Detail')
    lines.append('')
    lines.append('| City | Service | Status | Demand Total | Accessibility Gap | Capacity Gap |')
    lines.append('| --- | --- | --- | --- | --- | --- |')
    for row in rows:
        lines.append(
            f"| {row['city']} | {row['service']} | {STATUS_BADGE[row['status']]} | {fmt_num(row['demand_total'])} | {fmt_num(row['accessibility_gap_total'])} | {fmt_num(row['capacity_gap_total'])} |"
        )
    lines.append('')
    return '\n'.join(lines)


def main():
    rows, counts = collect_rows()
    OUT_PATH.write_text(build_readme(rows, counts), encoding='utf-8')
    print(OUT_PATH)


if __name__ == '__main__':
    main()

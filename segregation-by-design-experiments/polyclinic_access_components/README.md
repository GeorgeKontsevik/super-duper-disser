# Polyclinic Access Components

Подпроект про компонентную доступность до `polyclinic` и связку этих компонент со `street pattern`.

Базовая исследовательская схема зафиксирована здесь:

- [RESEARCH_SCHEME.md](/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/RESEARCH_SCHEME.md)

Берет готовый diagnostics parquet и оставляет только `polyclinic`.
Для каждого дома сохраняет части пути и бинарные флаги `ok / not ok`:

- `walk_direct_ok`
- `pt_total_ok`
- `access_ok`
- `egress_ok`
- `in_vehicle_ok`
- `transfer_ok`
- `access_egress_sum_ok`

## Запуск

```bash
cd /Users/gk/Code/super-duper-disser

./.venv/bin/python segregation-by-design-experiments/polyclinic_access_components/run_experiments.py
```

## Outputs

- `outputs/polyclinic_home_access_components.parquet`
- `outputs/polyclinic_component_ok_summary_by_city.csv`
- `outputs/polyclinic_component_ok_summary_overall.csv`
- `outputs/polyclinic_requested_summary_overall.csv`
- `outputs/polyclinic_requested_summary_overall.png`
- `outputs/polyclinic_requested_summary_by_city.csv`
- `outputs/polyclinic_requested_summary_by_city.png`
- `outputs/single_component_patterns`

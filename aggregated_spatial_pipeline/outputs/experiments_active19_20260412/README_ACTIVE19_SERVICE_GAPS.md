# Active19 Service Gap Matrix

Сводка по 19 городам и 4 сервисам из `pipeline_2/solver_inputs`.

Скрипт генерации:
- [build_active19_service_gaps_readme.py](/Users/gk/Code/super-duper-disser/scripts/build_active19_service_gaps_readme.py)

Команда пересборки:
```bash
cd /Users/gk/Code/super-duper-disser
./.venv/bin/python scripts/build_active19_service_gaps_readme.py
```

Источник на каждую ячейку:
- `summary.json` для `demand_total`
- `blocks_solver.parquet` для сумм `demand_without` и `demand_left`

Правило классификации:
- `🟢 OK`: `demand_without = 0` и `demand_left = 0`
- `🔵 access`: необеспеченность из-за доступности, `demand_without > 0`, `demand_left = 0`
- `🔴 capacity`: необеспеченность из-за capacity, `demand_left > 0`, `demand_without = 0`
- `🟣 both`: присутствуют оба типа разрыва
- `⚪ missing`: не найден `summary.json` или `blocks_solver.parquet`

Итоги по всем 76 city-service комбинациям:
- `OK`: 4
- `access`: 19
- `capacity`: 30
- `both`: 22
- `missing`: 1

## Matrix

| City | Hospital | Polyclinic | School | Kindergarten |
| --- | --- | --- | --- | --- |
| bergen_norway | 🔴 capacity | 🔵 access | 🟣 both | 🔴 capacity |
| bologna_italy | 🟢 OK | 🔵 access | 🟣 both | 🟣 both |
| bristol_united_kingdom | 🔴 capacity | 🔵 access | 🟣 both | 🔴 capacity |
| brno_czechia | 🟢 OK | 🔵 access | 🟣 both | 🟣 both |
| coimbra_portugal | 🔵 access | 🔵 access | 🟣 both | 🔴 capacity |
| debrecen_hungary | 🔴 capacity | 🟣 both | 🔴 capacity | 🔴 capacity |
| dresden_germany | 🔴 capacity | 🔵 access | 🔴 capacity | 🟣 both |
| freiburg_im_breisgau_germany | 🟢 OK | 🔵 access | 🔴 capacity | 🟣 both |
| gothenburg_sweden | 🔴 capacity | 🔵 access | 🔴 capacity | 🔴 capacity |
| graz_austria | 🔴 capacity | 🔵 access | 🟣 both | 🟣 both |
| innsbruck_austria | 🔴 capacity | 🟣 both | 🔴 capacity | 🟣 both |
| krakow_poland | ⚪ missing | 🔵 access | 🟣 both | 🟣 both |
| linz_austria | 🔴 capacity | 🟣 both | 🔴 capacity | 🔴 capacity |
| lyon_france | 🔴 capacity | 🔵 access | 🔵 access | 🔴 capacity |
| marseille_france | 🔵 access | 🔵 access | 🔵 access | 🔴 capacity |
| porto_portugal | 🟢 OK | 🔵 access | 🔵 access | 🔴 capacity |
| turin_italy | 🔴 capacity | 🟣 both | 🟣 both | 🔴 capacity |
| turku_finland | 🔴 capacity | 🟣 both | 🔴 capacity | 🟣 both |
| zaragoza_spain | 🔴 capacity | 🔵 access | 🟣 both | 🔴 capacity |

## Detail

| City | Service | Status | Demand Total | Accessibility Gap | Capacity Gap |
| --- | --- | --- | --- | --- | --- |
| bergen_norway | hospital | 🔴 capacity | 3,538 | 0 | 1,738 |
| bergen_norway | polyclinic | 🔵 access | 5,062 | 881 | 0 |
| bergen_norway | school | 🟣 both | 45,746 | 605 | 26,605 |
| bergen_norway | kindergarten | 🔴 capacity | 45,746 | 0 | 42,595 |
| bologna_italy | hospital | 🟢 OK | 2,903 | 0 | 0 |
| bologna_italy | polyclinic | 🔵 access | 4,088 | 2,153 | 0 |
| bologna_italy | school | 🟣 both | 35,756 | 12,508 | 356 |
| bologna_italy | kindergarten | 🟣 both | 35,756 | 1,303 | 21,956 |
| bristol_united_kingdom | hospital | 🔴 capacity | 7,977 | 0 | 2,577 |
| bristol_united_kingdom | polyclinic | 🔵 access | 11,325 | 3,887 | 0 |
| bristol_united_kingdom | school | 🟣 both | 100,990 | 158 | 56,463 |
| bristol_united_kingdom | kindergarten | 🔴 capacity | 100,990 | 0 | 82,390 |
| brno_czechia | hospital | 🟢 OK | 4,225 | 0 | 0 |
| brno_czechia | polyclinic | 🔵 access | 6,017 | 2,405 | 0 |
| brno_czechia | school | 🟣 both | 53,822 | 2,007 | 10,159 |
| brno_czechia | kindergarten | 🟣 both | 53,822 | 2,087 | 15,272 |
| coimbra_portugal | hospital | 🔵 access | 5,507 | 2 | 0 |
| coimbra_portugal | polyclinic | 🔵 access | 7,849 | 2,979 | 0 |
| coimbra_portugal | school | 🟣 both | 70,245 | 1,917 | 40,245 |
| coimbra_portugal | kindergarten | 🔴 capacity | 70,245 | 0 | 59,445 |
| debrecen_hungary | hospital | 🔴 capacity | 9,990 | 0 | 7,590 |
| debrecen_hungary | polyclinic | 🟣 both | 14,245 | 27 | 12,445 |
| debrecen_hungary | school | 🔴 capacity | 128,042 | 0 | 89,642 |
| debrecen_hungary | kindergarten | 🔴 capacity | 128,042 | 0 | 110,042 |
| dresden_germany | hospital | 🔴 capacity | 11,120 | 0 | 7,520 |
| dresden_germany | polyclinic | 🔵 access | 15,822 | 6,898 | 0 |
| dresden_germany | school | 🔴 capacity | 141,269 | 0 | 62,869 |
| dresden_germany | kindergarten | 🟣 both | 141,269 | 2,319 | 42,029 |
| freiburg_im_breisgau_germany | hospital | 🟢 OK | 10,017 | 0 | 0 |
| freiburg_im_breisgau_germany | polyclinic | 🔵 access | 14,303 | 10,211 | 0 |
| freiburg_im_breisgau_germany | school | 🔴 capacity | 128,983 | 0 | 71,383 |
| freiburg_im_breisgau_germany | kindergarten | 🟣 both | 128,983 | 2,765 | 53,859 |
| gothenburg_sweden | hospital | 🔴 capacity | 9,562 | 0 | 7,762 |
| gothenburg_sweden | polyclinic | 🔵 access | 13,665 | 3,701 | 0 |
| gothenburg_sweden | school | 🔴 capacity | 123,128 | 0 | 61,928 |
| gothenburg_sweden | kindergarten | 🔴 capacity | 123,128 | 0 | 85,928 |
| graz_austria | hospital | 🔴 capacity | 5,899 | 0 | 3,499 |
| graz_austria | polyclinic | 🔵 access | 8,374 | 5,130 | 0 |
| graz_austria | school | 🟣 both | 74,626 | 7,172 | 13,426 |
| graz_austria | kindergarten | 🟣 both | 74,626 | 5,211 | 11,026 |
| innsbruck_austria | hospital | 🔴 capacity | 9,275 | 0 | 7,475 |
| innsbruck_austria | polyclinic | 🟣 both | 13,277 | 1,778 | 7,277 |
| innsbruck_austria | school | 🔴 capacity | 120,508 | 0 | 66,508 |
| innsbruck_austria | kindergarten | 🟣 both | 120,508 | 552 | 73,108 |
| krakow_poland | hospital | ⚪ missing | - | - | - |
| krakow_poland | polyclinic | 🔵 access | 3,179 | 1,416 | 0 |
| krakow_poland | school | 🟣 both | 28,593 | 2,749 | 14,193 |
| krakow_poland | kindergarten | 🟣 both | 28,593 | 2,397 | 8,568 |
| linz_austria | hospital | 🔴 capacity | 12,285 | 0 | 5,685 |
| linz_austria | polyclinic | 🟣 both | 17,540 | 684 | 13,940 |
| linz_austria | school | 🔴 capacity | 158,538 | 0 | 93,138 |
| linz_austria | kindergarten | 🔴 capacity | 158,538 | 0 | 96,581 |
| lyon_france | hospital | 🔴 capacity | 10,008 | 0 | 2,208 |
| lyon_france | polyclinic | 🔵 access | 14,177 | 1,537 | 0 |
| lyon_france | school | 🔵 access | 125,450 | 75 | 0 |
| lyon_france | kindergarten | 🔴 capacity | 125,450 | 0 | 83,148 |
| marseille_france | hospital | 🔵 access | 585 | 1 | 0 |
| marseille_france | polyclinic | 🔵 access | 813 | 696 | 0 |
| marseille_france | school | 🔵 access | 6,827 | 55 | 0 |
| marseille_france | kindergarten | 🔴 capacity | 6,827 | 0 | 6,004 |
| porto_portugal | hospital | 🟢 OK | 5,692 | 0 | 0 |
| porto_portugal | polyclinic | 🔵 access | 8,025 | 2,947 | 0 |
| porto_portugal | school | 🔵 access | 70,686 | 20,331 | 0 |
| porto_portugal | kindergarten | 🔴 capacity | 70,686 | 0 | 57,486 |
| turin_italy | hospital | 🔴 capacity | 11,185 | 0 | 6,385 |
| turin_italy | polyclinic | 🟣 both | 15,857 | 2,610 | 5,057 |
| turin_italy | school | 🟣 both | 140,614 | 1,037 | 99,214 |
| turin_italy | kindergarten | 🔴 capacity | 140,614 | 0 | 128,014 |
| turku_finland | hospital | 🔴 capacity | 5,910 | 0 | 2,310 |
| turku_finland | polyclinic | 🟣 both | 8,444 | 6,871 | 44 |
| turku_finland | school | 🔴 capacity | 76,112 | 0 | 46,112 |
| turku_finland | kindergarten | 🟣 both | 76,112 | 1,384 | 42,512 |
| zaragoza_spain | hospital | 🔴 capacity | 15,757 | 0 | 7,357 |
| zaragoza_spain | polyclinic | 🔵 access | 22,510 | 4,086 | 0 |
| zaragoza_spain | school | 🟣 both | 202,718 | 9,676 | 63,818 |
| zaragoza_spain | kindergarten | 🔴 capacity | 202,718 | 0 | 170,318 |

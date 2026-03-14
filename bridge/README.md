# Bridge: OSMnx/iduedu → Arctic Format

Перевод графа из OSMnx или iduedu в формат arctic (transport_df, G для make_g / graph_to_city_model).

Кварталы как узлы, drive + transit, выход в минутах.

## Arctic-совместимость (по умолчанию)

При `arctic_compatible=True` (по умолчанию) bridge выдаёт `transport_df` с колонками Arctic:
`Aviation`, `Regular road`, `Winter road`, `Water transport`. Drive-граф маппится в `Regular road`, остальные режимы = 0. Готово для `make_g` и provision.

```python
from bridge import graph_to_arctic_format, settl_from_blocks
from scripts.preprocesser.gcreator import make_g
from scripts.preprocesser.constants import transport_modes

transport_df, G_arctic = graph_to_arctic_format(
    blocks_gdf, G_drive,
    service_name="hospital",
    arctic_compatible=True,  # по умолчанию
)

settl = settl_from_blocks(blocks_gdf)
G_undirected = make_g(transport_df, transport_modes, blocks_gdf, settl)
```

## Классический режим (drive + transit)

При `arctic_compatible=False` — колонки `drive`, `transit`, можно задать `modalities` и `mode_mapping`:

```python
transport_df, G_arctic = graph_to_arctic_format(
    blocks_gdf, G_drive, G_transit=G_transit,
    service_name="hospital",
    arctic_compatible=False,
    modalities=["drive", "transit"],
)
```

## Зависимости

- iduedu (для get_adj_matrix_gdf_to_gdf)
- geopandas, networkx, pandas, numpy
- osmnx (для fallback при OSMnx-графах)

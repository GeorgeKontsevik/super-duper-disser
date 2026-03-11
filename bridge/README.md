# Bridge: OSMnx/iduedu → Arctic Format

Перевод графа из OSMnx или iduedu в формат arctic (transport_df, G для make_g / graph_to_city_model).

Кварталы как узлы, drive + transit, выход в минутах.

## Использование

```python
from bridge import graph_to_arctic_format, settl_from_blocks, ensure_graph_has_time_min
from iduedu import get_drive_graph
import geopandas as gpd

# Блоки (кварталы) — из blocksnet, sm_imputation или своего источника
blocks_gdf = ...  # GeoDataFrame с geometry, population, capacity_{service}

# Граф из iduedu
G_drive = get_drive_graph(polygon=boundary)
# Если граф из OSMnx (только length):
# from bridge import ensure_graph_has_time_min
# G_drive = ensure_graph_has_time_min(ox.graph_from_polygon(...))

# Опционально: transit-граф из connectpt preprocess
G_transit = ...  # или None

transport_df, G_arctic = graph_to_arctic_format(
    blocks_gdf, G_drive, G_transit=G_transit,
    service_name="hospital",
    modalities=["drive", "transit"],
)

# Для arctic make_g
from arctic_access.scripts.preprocesser.gcreator import make_g
settl = settl_from_blocks(blocks_gdf)
G_undirected = make_g(transport_df, ["drive", "transit"], blocks_gdf, settl)
```

## Зависимости

- iduedu (для get_adj_matrix_gdf_to_gdf)
- geopandas, networkx, pandas, numpy
- osmnx (для fallback при OSMnx-графах)

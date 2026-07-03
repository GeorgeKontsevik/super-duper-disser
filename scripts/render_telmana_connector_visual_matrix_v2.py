from __future__ import annotations

import json
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString


plt.rcParams.update({"font.family": "Arial"})

TITLE_FONT_SIZE = 14
CARD_FONT_SIZE = 15
LEGEND_FONT_SIZE = 13
FIGURE_TITLE_FONT_SIZE = 22

ROOT = Path(
    "/Users/gk/Code/super-duper-disser/"
    "aggregated_spatial_pipeline/outputs/experiments_spb_telmana_connector_clean_4x2_20260620"
)
OUT = ROOT / "visual_scenario_maps_square_connector_v2"
PROJECT_REFERENCE_IMAGE = OUT / "telmana_project_reference.png"
CONNECTOR = ROOT.parent / (
    "experiments_spb_telmana_clean_4x2_20260620/"
    "visual_scenario_maps_square_telmana_corrected/"
    "telmana_connector_touching_block29_selected.parquet"
)

SCENARIOS = [
    {
        "id": "01_current",
        "label": "current",
        "project": False,
        "connector": False,
    },
    {
        "id": "02_current_plus_project",
        "label": "current + redevelopment",
        "project": True,
        "connector": False,
    },
    {
        "id": "03_current_plus_project_plus_connector",
        "label": "redevelopment + connector",
        "project": True,
        "connector": True,
    },
    {
        "id": "04_current_plus_connector",
        "label": "current + connector",
        "project": False,
        "connector": True,
    },
]

SCENARIO_LABEL_RU = {
    "01_current": "Текущее состояние",
    "02_current_plus_project": "Квартал добавлен",
    "03_current_plus_project_plus_connector": "Квартал и дорога\nдобавлены",
    "04_current_plus_connector": "Только дорога\nдобавлена",
}

COLUMN_LABELS_RU = [
    "Сценарий\nи результат",
    "Состояние\nквартала и дороги",
    "Маршрут\nОТ",
    "Новые\nполиклиники",
    "Неудовлетворенный\nспрос",
]


def read_gdf(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(32636, allow_override=True)
    elif str(gdf.crs).upper() not in {"EPSG:32636", "32636"}:
        gdf = gdf.to_crs(32636)
    return gdf


def scenario_dir(scenario_id: str) -> Path:
    return ROOT / scenario_id


def route_dir(scenario_id: str) -> Path:
    return ROOT / "one_route_existing_service_search" / scenario_id / "existing_service" / "routes_1"


def orient_line_to_nodes(geometry, start, end):
    if geometry is None or geometry.is_empty or geometry.geom_type != "LineString":
        return geometry
    coords = list(geometry.coords)
    if len(coords) < 2:
        return geometry
    current_distance = start.distance(LineString(coords).boundary.geoms[0]) + end.distance(LineString(coords).boundary.geoms[-1])
    reversed_distance = start.distance(LineString(coords[::-1]).boundary.geoms[0]) + end.distance(LineString(coords[::-1]).boundary.geoms[-1])
    if reversed_distance < current_distance:
        return LineString(coords[::-1])
    return geometry


def route_edges_geometry(scenario_id: str) -> gpd.GeoDataFrame:
    rdir = route_dir(scenario_id)
    generated = pd.read_parquet(rdir / "snapshots/intermodal_replaced/bus_generated_route_edges.parquet")
    nodes = read_gdf(rdir / "snapshots/intermodal_replaced/graph_nodes.parquet")
    nodes_by_id = nodes.set_index("index")
    with (rdir / "snapshots/intermodal_replaced/graph.pkl").open("rb") as fh:
        graph = pickle.load(fh)

    records = []
    for order, row in generated.reset_index(drop=True).iterrows():
        u = int(row["intermodal_u"])
        v = int(row["intermodal_v"])
        data = graph.get_edge_data(u, v) or {}
        geometry = None
        for attrs in data.values():
            if attrs.get("is_generated") and attrs.get("route") == row["route_name"]:
                geometry = attrs.get("geometry")
                break
        if geometry is None and u in nodes_by_id.index and v in nodes_by_id.index:
            geometry = LineString([nodes_by_id.at[u, "geometry"], nodes_by_id.at[v, "geometry"]])
        if geometry is None:
            continue
        if u in nodes_by_id.index and v in nodes_by_id.index:
            geometry = orient_line_to_nodes(geometry, nodes_by_id.at[u, "geometry"], nodes_by_id.at[v, "geometry"])
        records.append({**row.to_dict(), "order": order, "geometry": geometry})

    return gpd.GeoDataFrame(records, geometry="geometry", crs=nodes.crs)


def placement_path(scenario_id: str, with_route: bool) -> Path:
    if with_route:
        return route_dir(scenario_id) / "placement/blocks_solver_after.parquet"
    return scenario_dir(scenario_id) / "pipeline_2/placement_exact/polyclinic/blocks_solver_after.parquet"


def placement_summary_path(scenario_id: str, with_route: bool) -> Path:
    if with_route:
        return route_dir(scenario_id) / "placement/summary_after.json"
    return scenario_dir(scenario_id) / "pipeline_2/placement_exact/polyclinic/summary_after.json"


def placement_summary(scenario_id: str, with_route: bool) -> dict:
    with placement_summary_path(scenario_id, with_route).open() as fh:
        return json.load(fh)


def placement_blocks(scenario_id: str, with_route: bool) -> gpd.GeoDataFrame:
    return read_gdf(placement_path(scenario_id, with_route))


def route_stop_count(scenario_id: str, with_route: bool) -> int | None:
    if not with_route:
        return None
    with (route_dir(scenario_id) / "connectpt_bus_summary.json").open() as fh:
        summary = json.load(fh)
    lengths = summary.get("route_lengths") or []
    if not lengths:
        return None
    return int(lengths[0])


def existing_polyclinics() -> gpd.GeoDataFrame:
    return read_gdf(ROOT / "01_current/pipeline_2/services_raw/polyclinic.parquet")


def water_layer() -> gpd.GeoDataFrame:
    return read_gdf(ROOT / "01_current/blocksnet/water.parquet")


def target_quarter(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "name" in blocks.columns:
        q = blocks[blocks["name"].astype(str) == "29"]
        if len(q):
            return q
    return blocks.iloc[[29]]


def selected_new_services(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return blocks[blocks["placement_status"].astype(str).eq("new")].copy()


def selected_existing_services(blocks: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return blocks[blocks["placement_status"].astype(str).eq("existing")].copy()


def unmet_column(blocks: gpd.GeoDataFrame) -> str:
    for col in ("demand_without_after", "demand_without_after_routes", "demand_without"):
        if col in blocks.columns:
            return col
    raise ValueError("No unmet demand column found")


def plot_base(ax, blocks, roads, title: str | None = None):
    ax.set_facecolor("#fbf5e6")
    blocks.plot(ax=ax, color="#f7f1df", edgecolor="#d8d0bd", linewidth=0.35, zorder=1)
    roads.plot(ax=ax, color="#c7cbd0", linewidth=0.45, alpha=0.75, zorder=2)
    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, loc="center", pad=4)
    style_panel_axis(ax)


def style_panel_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(0.8)


def plot_scenario_card(ax, scenario_id: str, with_route: bool, summary: dict, stops: int | None):
    style_panel_axis(ax)
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    route_label = "с новым маршрутом" if with_route else "без нового маршрута"
    route_value = f"{stops} ост." if stops is not None else "нет"
    unmet = float(summary.get("demand_without_after_total", 0))
    services = int(summary.get("new_count", 0))
    provision = float(summary.get("provision_total_after", 0)) * 100
    ax.text(
        0.06,
        0.88,
        SCENARIO_LABEL_RU[scenario_id],
        ha="left",
        va="top",
        fontsize=CARD_FONT_SIZE,
        fontweight="bold",
        wrap=True,
    )
    ax.text(0.06, 0.66, route_label, ha="left", va="top", fontsize=13.5, color="#333333")
    ax.text(0.06, 0.51, "результат", ha="left", va="top", fontsize=13.5, fontweight="bold")
    ax.text(0.06, 0.38, f"новых сервисов: {services}", ha="left", va="top", fontsize=13)
    ax.text(0.06, 0.27, f"маршрут: {route_value}", ha="left", va="top", fontsize=13)
    ax.text(0.06, 0.20, f"неудовл. спрос: {unmet:.0f}", ha="left", va="top", fontsize=13)
    ax.text(0.06, 0.015, f"удовлетворено: {provision:.1f}%", ha="left", va="bottom", fontsize=13, color="#555555")


def plot_reference_image(ax, image_path: Path, title: str):
    image = plt.imread(image_path)
    ax.imshow(image)
    ax.set_aspect("auto")
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, loc="center", pad=4)
    style_panel_axis(ax)


def plot_water(ax, water):
    if water.empty:
        return
    polygons = water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    lines = water[water.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if len(polygons):
        polygons.plot(ax=ax, facecolor="#a9dff5", edgecolor="#63b7d8", linewidth=0.4, alpha=0.72, zorder=3)
    if len(lines):
        lines.plot(ax=ax, color="#54b8dc", linewidth=1.0, alpha=0.65, zorder=4)


def plot_quarter(ax, quarter, project: bool):
    if project:
        quarter.plot(
            ax=ax,
            facecolor="#e6f4ea",
            edgecolor="#0a8f49",
            linewidth=1.4,
            hatch="///",
            zorder=8,
        )
    else:
        quarter.plot(
            ax=ax,
            facecolor="#fdecec",
            edgecolor="#d62728",
            linewidth=1.2,
            hatch="///",
            zorder=8,
        )


def plot_connector(ax, connector, present: bool):
    connector.plot(
        ax=ax,
        color="#0a8f49" if present else "#d62728",
        linewidth=4.0,
        linestyle="-" if present else "--",
        zorder=10,
    )


def plot_existing_pt_routes(ax, routes):
    if routes.empty:
        return
    styled = routes.reset_index(drop=True).copy()
    styled["_route_color"] = styled.index % 20
    styled.plot(
        ax=ax,
        column="_route_color",
        categorical=True,
        cmap="tab20",
        linewidth=1.25,
        alpha=0.55,
        zorder=9,
        legend=False,
    )


def plot_unmet(ax, blocks, roads, quarter, connector, project, connector_present, norm, title: str | None = None):
    col = unmet_column(blocks)
    blocks.assign(_unmet=blocks[col].fillna(0)).plot(
        ax=ax,
        column="_unmet",
        cmap="RdYlGn_r",
        norm=norm,
        edgecolor="#d0c8b8",
        linewidth=0.25,
        zorder=1,
    )
    roads.plot(ax=ax, color="#d5d8dc", linewidth=0.35, alpha=0.65, zorder=2)
    plot_quarter(ax, quarter, project)
    plot_connector(ax, connector, connector_present)
    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, loc="center", pad=4)
    style_panel_axis(ax)


def plot_services(
    ax,
    blocks,
    roads,
    quarter,
    connector,
    project,
    connector_present,
    title,
    *,
    show_existing: bool = False,
    show_project_quarter_service: bool = False,
):
    plot_base(ax, blocks, roads, title)
    plot_quarter(ax, quarter, project)
    plot_connector(ax, connector, connector_present)
    new = selected_new_services(blocks)
    existing = selected_existing_services(blocks) if show_existing else gpd.GeoDataFrame(geometry=[], crs=blocks.crs)
    if show_existing and len(existing):
        existing.centroid.plot(ax=ax, color="#222831", markersize=11, marker="s", zorder=12)
    if len(new):
        new.centroid.plot(ax=ax, color="#00a88f", markersize=28, marker="*", zorder=13)
    if show_project_quarter_service and len(quarter):
        quarter.centroid.plot(ax=ax, color="#00a88f", markersize=34, marker="*", zorder=14)


def plot_route(ax, blocks, roads, quarter, connector, project, connector_present, route, title):
    plot_base(ax, blocks, roads, title)
    plot_quarter(ax, quarter, project)
    plot_connector(ax, connector, connector_present)
    if route is not None and len(route):
        route.plot(ax=ax, color="#00806f", linewidth=2.8, zorder=12)
        route = route.sort_values("order")
        ends = gpd.GeoSeries([route.geometry.iloc[0].boundary.geoms[0], route.geometry.iloc[-1].boundary.geoms[-1]], crs=route.crs)
        ends.plot(ax=ax, color=["#c2185b", "#009f5d"], markersize=24, zorder=13)


def set_common_extent(axes, bounds):
    xmin, ymin, xmax, ymax = bounds
    side = max(xmax - xmin, ymax - ymin)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    pad = side * 0.035
    for ax in axes:
        ax.set_xlim(cx - side / 2 - pad, cx + side / 2 + pad)
        ax.set_ylim(cy - side / 2 - pad, cy + side / 2 + pad)
        ax.set_aspect("equal", adjustable="box")


def collect_scenario_record(scenario: dict, with_route: bool) -> dict:
    sid = scenario["id"]
    summary = placement_summary(sid, with_route=with_route)
    stops = route_stop_count(sid, with_route=with_route)
    return {
        "scenario": scenario,
        "with_route": with_route,
        "summary": summary,
        "stops": stops,
        "new_services": int(summary.get("new_count", 0)),
        "route_stops": stops,
        "demand_without_after_total": float(summary.get("demand_without_after_total", 0)),
        "provision_total_after": float(summary.get("provision_total_after", 0)),
    }


def render_scenario_group(
    records: list[dict],
    *,
    title: str,
    output_path: Path,
    connector: gpd.GeoDataFrame,
    combined_bounds,
    norm,
    legend_items,
) -> None:
    fig = plt.figure(figsize=(13.33, 13.0), facecolor="white")
    fig.text(
        0.012,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=FIGURE_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    gs = fig.add_gridspec(
        nrows=len(records),
        ncols=5,
        left=0.012,
        right=0.915,
        top=0.885,
        bottom=0.155,
        wspace=0.010,
        hspace=0.065,
    )

    map_axes = []
    first_row_axes = None
    for row, record in enumerate(records):
        scenario = record["scenario"]
        with_route = record["with_route"]
        sid = scenario["id"]
        sdir = scenario_dir(sid)
        blocks = read_gdf(sdir / "derived_layers/blocks_clipped.parquet")
        roads = read_gdf(sdir / "derived_layers/roads_drive_osmnx.parquet")
        q = target_quarter(blocks)
        selected_blocks = placement_blocks(sid, with_route=with_route)
        route = route_edges_geometry(sid) if with_route else None
        row_axes = [fig.add_subplot(gs[row, i]) for i in range(5)]
        if first_row_axes is None:
            first_row_axes = row_axes
            for ax, label in zip(first_row_axes, COLUMN_LABELS_RU, strict=True):
                pos = ax.get_position()
                fig.text(
                    (pos.x0 + pos.x1) / 2,
                    0.902,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=TITLE_FONT_SIZE,
                    fontweight="bold",
                )
        map_axes.extend(row_axes[1:])

        plot_scenario_card(row_axes[0], sid, with_route, record["summary"], record["stops"])

        plot_base(row_axes[1], blocks, roads, None)
        plot_quarter(row_axes[1], q, scenario["project"])
        plot_connector(row_axes[1], connector, scenario["connector"])

        plot_route(
            row_axes[2],
            blocks,
            roads,
            q,
            connector,
            scenario["project"],
            scenario["connector"],
            route,
            None,
        )

        plot_services(
            row_axes[3],
            selected_blocks,
            roads,
            q,
            connector,
            scenario["project"],
            scenario["connector"],
            None,
            show_project_quarter_service=scenario["project"],
        )
        plot_unmet(
            row_axes[4],
            selected_blocks,
            roads,
            q,
            connector,
            scenario["project"],
            scenario["connector"],
            norm,
            None,
        )

    set_common_extent(map_axes, combined_bounds)

    cax = fig.add_axes([0.935, 0.205, 0.012, 0.57])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="RdYlGn_r"), cax=cax)
    cbar.set_label("неудовлетворенный спрос на поликлиники по кварталам (общая шкала)", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        markerscale=1.4,
        handlelength=2.4,
        columnspacing=1.5,
        bbox_to_anchor=(0.49, 0.035),
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    blocks_context = read_gdf(ROOT / "01_current/derived_layers/blocks_clipped.parquet")
    roads_context = read_gdf(ROOT / "01_current/derived_layers/roads_drive_osmnx.parquet")
    connector = read_gdf(CONNECTOR)
    quarter = target_quarter(blocks_context)
    polies = existing_polyclinics()
    existing_routes = read_gdf(ROOT / "01_current/connectpt_osm/bus/projected_lines.parquet")
    water = water_layer()

    baseline = read_gdf(ROOT / "01_current/pipeline_2/solver_inputs/polyclinic/blocks_solver.parquet")
    baseline = baseline.rename(columns={"demand_without": "demand_without_after"})
    all_unmet = [baseline["demand_without_after"].fillna(0)]
    for scenario in SCENARIOS:
        sid = scenario["id"]
        all_unmet.append(placement_blocks(sid, with_route=False)["demand_without_after"].fillna(0))
        all_unmet.append(placement_blocks(sid, with_route=True)["demand_without_after"].fillna(0))
    vmax = max(float(pd.concat(all_unmet).max()), 1.0)
    norm = colors.Normalize(vmin=0, vmax=vmax)

    legend_items = [
        Patch(facecolor="#fdecec", edgecolor="#d62728", hatch="///", label="проектного квартала нет"),
        Patch(facecolor="#e6f4ea", edgecolor="#0a8f49", hatch="///", label="проектный квартал есть"),
        Line2D([0], [0], color="#d62728", lw=4.0, linestyle="--", label="дороги нет"),
        Line2D([0], [0], color="#0a8f49", lw=4.0, linestyle="-", label="дорога есть"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c2185b", markersize=8, label="существующая поликлиника"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#00a88f", markersize=10, label="новая поликлиника"),
        Line2D([0], [0], color="#00806f", lw=2.8, label="сгенерированный маршрут"),
    ]

    context_fig = plt.figure(figsize=(4.2, 15.5), facecolor="white")
    context_fig.text(
        0.012,
        0.985,
        "Исходные слои",
        ha="left",
        va="top",
        fontsize=FIGURE_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    context_gs = context_fig.add_gridspec(
        nrows=5,
        ncols=1,
        left=0.08,
        right=0.92,
        top=0.92,
        bottom=0.02,
        hspace=0.34,
    )
    context_axes = [context_fig.add_subplot(context_gs[i, 0]) for i in range(5)]

    plot_reference_image(context_axes[0], PROJECT_REFERENCE_IMAGE, "проект планировки")

    plot_base(context_axes[1], blocks_context, roads_context, "дороги и кварталы")
    plot_water(context_axes[1], water)
    plot_quarter(context_axes[1], quarter, False)
    plot_connector(context_axes[1], connector, False)

    plot_base(context_axes[2], blocks_context, roads_context, "существующие\nмаршруты ОТ")
    plot_existing_pt_routes(context_axes[2], existing_routes)
    plot_quarter(context_axes[2], quarter, False)
    plot_connector(context_axes[2], connector, False)

    plot_base(context_axes[3], blocks_context, roads_context, "существующие\nполиклиники")
    polies.plot(ax=context_axes[3], color="#c2185b", markersize=18, zorder=10)
    plot_quarter(context_axes[3], quarter, False)
    plot_connector(context_axes[3], connector, False)

    plot_unmet(
        context_axes[4],
        baseline,
        roads_context,
        quarter,
        connector,
        False,
        False,
        norm,
        "базовый неудовлетворенный\nспрос",
    )

    combined_bounds = blocks_context.total_bounds
    set_common_extent(context_axes[1:], combined_bounds)

    context_path = OUT / "telmana_connector_context_row.png"
    context_fig.savefig(context_path, dpi=180)
    plt.close(context_fig)

    scenario_records = []
    for scenario_order, scenario in enumerate(SCENARIOS):
        for with_route in (False, True):
            record = collect_scenario_record(scenario, with_route)
            record["scenario_order"] = scenario_order
            scenario_records.append(record)
    ranked_records = sorted(
        scenario_records,
        key=lambda record: (
            -record["new_services"],
            record["with_route"],
            -record["demand_without_after_total"],
            record["scenario_order"],
        ),
    )
    worst_records = ranked_records[:4]
    best_records = ranked_records[4:]

    worst_path = OUT / "telmana_connector_scenarios_worse_new_services.png"
    best_path = OUT / "telmana_connector_scenarios_better_new_services.png"
    render_scenario_group(
        worst_records,
        title="Сценарии с большим числом новых сервисов",
        output_path=worst_path,
        connector=connector,
        combined_bounds=combined_bounds,
        norm=norm,
        legend_items=legend_items,
    )
    render_scenario_group(
        best_records,
        title="Сценарии с меньшим числом новых сервисов",
        output_path=best_path,
        connector=connector,
        combined_bounds=combined_bounds,
        norm=norm,
        legend_items=legend_items,
    )

    manifest = {
        "context_row": str(context_path),
        "scenario_tables": {
            "worse_new_services": str(worst_path),
            "better_new_services": str(best_path),
        },
        "layout": {
            "context_row": [
                "проект планировки",
                "дороги и кварталы",
                "существующие маршруты ОТ",
                "существующие поликлиники",
                "базовый неудовлетворенный спрос",
            ],
            "scenario_table_columns": COLUMN_LABELS_RU,
            "sort": "descending by new_services; first image contains worse scenarios",
        },
        "scenario_metrics": [
            {
                "scenario": record["scenario"]["id"],
                "with_route": record["with_route"],
                "new_services": record["new_services"],
                "route_stops": record["route_stops"],
                "demand_without_after_total": record["demand_without_after_total"],
                "provision_total_after": record["provision_total_after"],
                "group": "worse_new_services" if record in worst_records else "better_new_services",
            }
            for record in ranked_records
        ],
        "unmet_scale": {
            "column": "demand_without_after",
            "vmin": 0,
            "vmax": vmax,
            "cmap": "RdYlGn_r",
        },
        "scenarios": SCENARIOS,
    }
    (OUT / "telmana_connector_clean_visual_matrix_v2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(context_path)
    print(worst_path)
    print(best_path)


if __name__ == "__main__":
    main()

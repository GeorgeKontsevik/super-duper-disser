from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(
    "/Users/gk/Code/super-duper-disser/"
    "aggregated_spatial_pipeline/outputs/experiments_spb_telmana_connector_clean_4x2_20260620"
)
OUT = ROOT / "visual_scenario_maps_square_connector_v2"
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


def route_edges_geometry(scenario_id: str) -> gpd.GeoDataFrame:
    rdir = route_dir(scenario_id)
    generated = pd.read_parquet(rdir / "snapshots/intermodal_replaced/bus_generated_route_edges.parquet")
    edges = read_gdf(rdir / "snapshots/intermodal_replaced/graph_edges_source.parquet")
    merged = generated.merge(
        edges[["u", "v", "geometry"]],
        left_on=["intermodal_u", "intermodal_v"],
        right_on=["u", "v"],
        how="left",
    )
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=edges.crs).dropna(subset=["geometry"])


def placement_path(scenario_id: str, with_route: bool) -> Path:
    if with_route:
        return route_dir(scenario_id) / "placement/blocks_solver_after.parquet"
    return scenario_dir(scenario_id) / "pipeline_2/placement_exact/polyclinic/blocks_solver_after.parquet"


def placement_blocks(scenario_id: str, with_route: bool) -> gpd.GeoDataFrame:
    return read_gdf(placement_path(scenario_id, with_route))


def existing_polyclinics() -> gpd.GeoDataFrame:
    return read_gdf(ROOT / "01_current/pipeline_2/services_raw/polyclinic.parquet")


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
    blocks.plot(ax=ax, color="#f7f1df", edgecolor="#d8d0bd", linewidth=0.35, zorder=1)
    roads.plot(ax=ax, color="#c7cbd0", linewidth=0.45, alpha=0.75, zorder=2)
    if title:
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.set_axis_off()


def plot_quarter(ax, quarter, project: bool):
    if project:
        quarter.plot(
            ax=ax,
            facecolor="#ffe083",
            edgecolor="#2c3742",
            linewidth=1.4,
            hatch="////",
            zorder=8,
        )
    else:
        quarter.plot(ax=ax, facecolor="#fff3b3", edgecolor="#6f7780", linewidth=1.1, zorder=8)


def plot_connector(ax, connector, present: bool):
    connector.plot(
        ax=ax,
        color="#ff7a00",
        linewidth=2.5,
        linestyle="-" if present else "--",
        zorder=10,
    )


def plot_unmet(ax, blocks, roads, quarter, connector, project, connector_present, norm, title):
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
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.set_axis_off()


def plot_services(ax, blocks, roads, quarter, connector, project, connector_present, title):
    plot_base(ax, blocks, roads, title)
    plot_quarter(ax, quarter, project)
    plot_connector(ax, connector, connector_present)
    new = selected_new_services(blocks)
    existing = selected_existing_services(blocks)
    if len(existing):
        existing.centroid.plot(ax=ax, color="#222831", markersize=11, marker="s", zorder=12)
    if len(new):
        new.centroid.plot(ax=ax, color="#00a88f", markersize=28, marker="*", zorder=13)


def plot_route(ax, blocks, roads, quarter, connector, project, connector_present, route, title):
    plot_base(ax, blocks, roads, title)
    plot_quarter(ax, quarter, project)
    plot_connector(ax, connector, connector_present)
    route.plot(ax=ax, color="#00806f", linewidth=2.8, zorder=12)
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    blocks_context = read_gdf(ROOT / "01_current/derived_layers/blocks_clipped.parquet")
    roads_context = read_gdf(ROOT / "01_current/derived_layers/roads_drive_osmnx.parquet")
    connector = read_gdf(CONNECTOR)
    quarter = target_quarter(blocks_context)
    polies = existing_polyclinics()
    existing_routes = read_gdf(ROOT / "01_current/connectpt_osm/bus/projected_lines.parquet")

    baseline = read_gdf(ROOT / "01_current/pipeline_2/solver_inputs/polyclinic/blocks_solver.parquet")
    baseline = baseline.rename(columns={"demand_without": "demand_without_after"})
    all_unmet = [baseline["demand_without_after"].fillna(0)]
    for scenario in SCENARIOS:
        sid = scenario["id"]
        all_unmet.append(placement_blocks(sid, with_route=False)["demand_without_after"].fillna(0))
        all_unmet.append(placement_blocks(sid, with_route=True)["demand_without_after"].fillna(0))
    vmax = max(float(pd.concat(all_unmet).max()), 1.0)
    norm = colors.Normalize(vmin=0, vmax=vmax)

    fig = plt.figure(figsize=(19, 23), facecolor="#f5f1e8")
    gs = fig.add_gridspec(
        nrows=5,
        ncols=4,
        left=0.035,
        right=0.94,
        top=0.965,
        bottom=0.06,
        wspace=0.08,
        hspace=0.18,
    )
    axes = []
    context_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes.extend(context_axes)

    plot_base(context_axes[0], blocks_context, roads_context, "road + block grid")
    plot_quarter(context_axes[0], quarter, False)
    plot_connector(context_axes[0], connector, False)

    plot_base(context_axes[1], blocks_context, roads_context, "existing PT routes")
    existing_routes.plot(ax=context_axes[1], color="#2c7ecb", linewidth=1.1, alpha=0.88, zorder=9)
    plot_quarter(context_axes[1], quarter, False)
    plot_connector(context_axes[1], connector, False)

    plot_base(context_axes[2], blocks_context, roads_context, "existing polyclinics")
    polies.plot(ax=context_axes[2], color="#c2185b", markersize=18, zorder=10)
    plot_quarter(context_axes[2], quarter, False)
    plot_connector(context_axes[2], connector, False)

    plot_unmet(
        context_axes[3],
        baseline,
        roads_context,
        quarter,
        connector,
        False,
        False,
        norm,
        "baseline unmet demand",
    )

    for row, scenario in enumerate(SCENARIOS, start=1):
        sid = scenario["id"]
        sdir = scenario_dir(sid)
        blocks = read_gdf(sdir / "derived_layers/blocks_clipped.parquet")
        roads = read_gdf(sdir / "derived_layers/roads_drive_osmnx.parquet")
        q = target_quarter(blocks)
        no_route_blocks = placement_blocks(sid, with_route=False)
        with_route_blocks = placement_blocks(sid, with_route=True)
        route = route_edges_geometry(sid)

        row_axes = [fig.add_subplot(gs[row, i]) for i in range(4)]
        axes.extend(row_axes)
        title_prefix = scenario["label"]

        plot_services(
            row_axes[0],
            no_route_blocks,
            roads,
            q,
            connector,
            scenario["project"],
            scenario["connector"],
            f"{title_prefix}\nplacement only: new services",
        )
        plot_route(
            row_axes[1],
            blocks,
            roads,
            q,
            connector,
            scenario["project"],
            scenario["connector"],
            route,
            f"{title_prefix}\none generated route",
        )
        plot_services(
            row_axes[2],
            with_route_blocks,
            roads,
            q,
            connector,
            scenario["project"],
            scenario["connector"],
            f"{title_prefix}\nroute + placement: new services",
        )
        plot_unmet(
            row_axes[3],
            with_route_blocks,
            roads,
            q,
            connector,
            scenario["project"],
            scenario["connector"],
            norm,
            f"{title_prefix}\nfinal unmet demand",
        )

    combined_bounds = blocks_context.total_bounds
    set_common_extent(axes, combined_bounds)

    cax = fig.add_axes([0.955, 0.14, 0.012, 0.64])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="RdYlGn_r"), cax=cax)
    cbar.set_label("unmet polyclinic demand per block (same scale)", fontsize=9)

    legend_items = [
        Patch(facecolor="#fff3b3", edgecolor="#6f7780", label="target quarter"),
        Patch(facecolor="#ffe083", edgecolor="#2c3742", hatch="////", label="redevelopment quarter"),
        Line2D([0], [0], color="#ff7a00", lw=2.5, linestyle="--", label="connector absent"),
        Line2D([0], [0], color="#ff7a00", lw=2.5, linestyle="-", label="connector present"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c2185b", markersize=8, label="existing polyclinic"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#00a88f", markersize=10, label="new service"),
        Line2D([0], [0], color="#00806f", lw=2.8, label="generated route"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.49, 0.024),
    )

    matrix_path = OUT / "telmana_connector_clean_visual_matrix_v2.png"
    fig.savefig(matrix_path, dpi=180)
    plt.close(fig)

    manifest = {
        "matrix": str(matrix_path),
        "layout": {
            "row_0": [
                "road + block grid",
                "existing PT routes",
                "existing polyclinics",
                "baseline unmet demand",
            ],
            "scenario_rows": [
                "placement-only new services",
                "one generated route",
                "route + placement new services",
                "final unmet demand",
            ],
        },
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
    print(matrix_path)


if __name__ == "__main__":
    main()

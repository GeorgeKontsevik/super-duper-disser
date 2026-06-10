#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Callable

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from shapely.geometry import LineString


DEFAULT_INPUT = Path(
    "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/_all_home_to_service_access_diagnostics.parquet"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/gk/Code/super-duper-disser/segregation-by-design-experiments/polyclinic_access_components/outputs"
)
DEFAULT_JOINT_INPUTS_ROOT = Path(
    "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"
)
DEFAULT_PT_WALK_LT_ROOT = Path(
    "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_pt_top1_walk_lt15"
)
DEFAULT_PT_WALK_GE_ROOT = Path(
    "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/residential_to_services_pt_top1_walk15plus"
)
POLYCLINIC = "polyclinic"
COMPONENT_COLUMNS = [
    "walk_direct_ok",
    "pt_total_ok",
    "access_ok",
    "egress_ok",
    "in_vehicle_ok",
    "transfer_ok",
    "access_egress_sum_ok",
]
OK_LABELS = {"ok_walk", "ok_pt_only"}
REQUESTED_COLUMNS = [
    "pct_ok_overall",
    "pct_not_ok_overall",
    "pct_ok_walk_only",
    "pct_ok_walk_plus_pt",
    "pct_not_ok_home_to_stop_overall",
    "pct_not_ok_pt_only_overall",
    "pct_not_ok_stop_to_service_overall",
    "pct_not_ok_both_walks_overall",
    "pct_not_ok_multi_component_overall",
    "pct_not_ok_sum_no_single_overall",
]
REQUESTED_LABELS = {
    "pct_ok_overall": "OK overall",
    "pct_not_ok_overall": "Not OK overall",
    "pct_ok_walk_only": "OK walk only",
    "pct_ok_walk_plus_pt": "OK walk + PT",
    "pct_not_ok_home_to_stop_overall": "Not OK: home -> stop",
    "pct_not_ok_pt_only_overall": "Not OK: PT segment",
    "pct_not_ok_stop_to_service_overall": "Not OK: stop -> service",
    "pct_not_ok_both_walks_overall": "Not OK: both walks > T",
    "pct_not_ok_multi_component_overall": "Not OK: multi-component",
    "pct_not_ok_sum_no_single_overall": "Not OK: sum > T, none > T",
}
REQUESTED_COLORS = [
    "#15803d",
    "#b91c1c",
    "#166534",
    "#1d4ed8",
    "#d97706",
    "#7c3aed",
    "#db2777",
    "#ea580c",
    "#6d28d9",
    "#6b7280",
]
SINGLE_COMPONENT_PATTERN_SPECS = {
    "home_to_stop_not_ok": {
        "label": "failed_access_gt_threshold",
        "pattern_context": "home",
        "pattern_columns": ["home_street_pattern_class"],
    },
    "pt_segment_not_ok": {
        "label": ["failed_in_vehicle_gt_threshold", "failed_transfer_gt_threshold", "failed_no_pt_path"],
        "pattern_context": "pt_route",
        "pattern_columns": [],
    },
    "stop_to_service_not_ok": {
        "label": "failed_egress_gt_threshold",
        "pattern_context": "service",
        "pattern_columns": ["service_street_pattern_class"],
    },
}

PATTERN_RAW_COLUMNS = ["city", "pattern_context", "pattern_value", "weight"]
PT_PATTERN_RAW_COLUMNS = [
    "city",
    "pattern_context",
    "pattern_value",
    "home_graph_node",
    "nearest_service_graph_node",
    "u",
    "v",
    "route_label",
    "edge_type",
    "edge_time_min",
    "usage_count",
    "edge_length_m",
    "intersect_length_m",
    "allocated_time_min",
]


def _add_component_flags(df: pd.DataFrame, threshold_min: float) -> pd.DataFrame:
    out = df.copy()
    out["walk_direct_ok"] = pd.to_numeric(out["walk_time_min"], errors="coerce") <= threshold_min
    out["pt_total_ok"] = pd.to_numeric(out["effective_pt_total_min"], errors="coerce") <= threshold_min
    out["access_ok"] = pd.to_numeric(out["access_walk_time_min"], errors="coerce") <= threshold_min
    out["egress_ok"] = pd.to_numeric(out["egress_walk_time_min"], errors="coerce") <= threshold_min
    out["in_vehicle_ok"] = pd.to_numeric(out["transport_time_min"], errors="coerce") <= threshold_min
    out["transfer_ok"] = pd.to_numeric(out["transfer_time_min"], errors="coerce") <= threshold_min
    out["access_egress_sum_ok"] = (
        pd.to_numeric(out["access_walk_time_min"], errors="coerce").fillna(float("inf"))
        + pd.to_numeric(out["egress_walk_time_min"], errors="coerce").fillna(float("inf"))
    ) <= threshold_min
    return out


def _build_component_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for city, city_df in df.groupby("city", dropna=False):
        for component in COMPONENT_COLUMNS:
            ok = city_df[component].fillna(False).astype(bool)
            rows.append(
                {
                    "city": city,
                    "component": component,
                    "n": int(len(city_df)),
                    "ok_count": int(ok.sum()),
                    "not_ok_count": int((~ok).sum()),
                    "share_ok": float(ok.mean()) if len(ok) else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["city", "component"]).reset_index(drop=True)


def _build_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["overall_ok"] = out["access_diagnosis_label"].isin(OK_LABELS)
    rows: list[dict[str, object]] = []
    for city, city_df in out.groupby("city", dropna=False):
        ok = city_df["overall_ok"].fillna(False).astype(bool)
        rows.append(
            {
                "city": city,
                "n": int(len(city_df)),
                "ok_count": int(ok.sum()),
                "not_ok_count": int((~ok).sum()),
                "share_ok": float(ok.mean()) if len(ok) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["city"]).reset_index(drop=True)


def _build_requested_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for city, city_df in df.groupby("city", dropna=False):
        n = int(len(city_df))
        overall_ok = city_df["access_diagnosis_label"].isin(OK_LABELS)
        overall_not_ok = ~overall_ok
        walk_only_ok = city_df["access_diagnosis_label"].eq("ok_walk")
        walk_plus_pt_ok = city_df["access_diagnosis_label"].eq("ok_pt_only")

        def _share(mask: pd.Series, denom: int) -> float:
            if denom == 0:
                return 0.0
            return float(mask.fillna(False).astype(bool).sum() / denom)

        not_ok_home_to_stop = city_df["access_diagnosis_label"].eq("failed_access_gt_threshold")
        not_ok_pt_only = city_df["access_diagnosis_label"].isin(
            ["failed_in_vehicle_gt_threshold", "failed_transfer_gt_threshold", "failed_no_pt_path"]
        )
        not_ok_stop_to_service = city_df["access_diagnosis_label"].eq("failed_egress_gt_threshold")
        not_ok_both_walks = city_df["access_diagnosis_label"].eq("failed_access_egress_sum_gt_threshold")
        not_ok_multi_component = city_df["access_diagnosis_label"].eq("failed_multi_component_gt_threshold")
        not_ok_sum_no_single = city_df["access_diagnosis_label"].eq(
            "failed_total_gt_threshold_no_single_component_gt_threshold"
        )

        rows.append(
            {
                "city": city,
                "n": n,
                "pct_ok_overall": _share(overall_ok, n),
                "pct_not_ok_overall": _share(overall_not_ok, n),
                "pct_ok_walk_only": _share(walk_only_ok, n),
                "pct_ok_walk_plus_pt": _share(walk_plus_pt_ok, n),
                "pct_not_ok_home_to_stop_overall": _share(not_ok_home_to_stop, n),
                "pct_not_ok_pt_only_overall": _share(not_ok_pt_only, n),
                "pct_not_ok_stop_to_service_overall": _share(not_ok_stop_to_service, n),
                "pct_not_ok_both_walks_overall": _share(not_ok_both_walks, n),
                "pct_not_ok_multi_component_overall": _share(not_ok_multi_component, n),
                "pct_not_ok_sum_no_single_overall": _share(not_ok_sum_no_single, n),
            }
        )
    return pd.DataFrame(rows).sort_values(["city"]).reset_index(drop=True)


def _load_polyclinic(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        input_path,
        columns=[
            "city",
            "service_name",
            "building_idx",
            "nearest_service_name",
            "walk_time_min",
            "effective_pt_total_min",
            "access_walk_time_min",
            "egress_walk_time_min",
            "transport_time_min",
            "transfer_time_min",
            "access_diagnosis_label",
            "home_street_pattern_class",
            "service_street_pattern_class",
        ],
    )
    return df[df["service_name"] == POLYCLINIC].copy().reset_index(drop=True)


def _round_for_export(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number", "float", "int", "bool"]).columns
    out[numeric_cols] = out[numeric_cols].astype(float).round(digits)
    return out


def _sorted_city_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["city"].dropna().astype(str).unique().tolist())


def _build_single_component_pattern_summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    for key, spec in SINGLE_COMPONENT_PATTERN_SPECS.items():
        labels = spec["label"]
        if isinstance(labels, str):
            sub = df[df["access_diagnosis_label"] == labels].copy()
        else:
            sub = df[df["access_diagnosis_label"].isin(labels)].copy()
        if sub.empty:
            summaries[key] = pd.DataFrame(
                columns=["pattern_context", "pattern_value", "n", "share"]
            )
            continue

        if spec["pattern_context"] == "home":
            values = sub["home_street_pattern_class"].fillna("UNKNOWN").astype(str)
        elif spec["pattern_context"] == "service":
            values = sub["service_street_pattern_class"].fillna("UNKNOWN").astype(str)
        elif spec["pattern_context"] == "pt_route":
            summaries[key] = pd.DataFrame(columns=["pattern_context", "pattern_value", "n", "share"])
            continue
        else:
            values = (
                sub["home_street_pattern_class"].fillna("UNKNOWN").astype(str)
                + " -> "
                + sub["service_street_pattern_class"].fillna("UNKNOWN").astype(str)
            )
        counts = values.value_counts(dropna=False).rename_axis("pattern_value").reset_index(name="n")
        counts["share"] = counts["n"] / counts["n"].sum()
        counts.insert(0, "pattern_context", spec["pattern_context"])
        summaries[key] = counts
    return summaries


def _build_single_component_pattern_raw(df: pd.DataFrame, key: str) -> pd.DataFrame:
    spec = SINGLE_COMPONENT_PATTERN_SPECS[key]
    labels = spec["label"]
    if isinstance(labels, str):
        sub = df[df["access_diagnosis_label"] == labels].copy()
    else:
        sub = df[df["access_diagnosis_label"].isin(labels)].copy()
    if sub.empty or spec["pattern_context"] == "pt_route":
        return pd.DataFrame(columns=PATTERN_RAW_COLUMNS)

    if spec["pattern_context"] == "home":
        sub = sub.assign(pattern_value=sub["home_street_pattern_class"].fillna("UNKNOWN").astype(str))
    elif spec["pattern_context"] == "service":
        sub = sub.assign(pattern_value=sub["service_street_pattern_class"].fillna("UNKNOWN").astype(str))
    else:
        sub = sub.assign(
            pattern_value=(
                sub["home_street_pattern_class"].fillna("UNKNOWN").astype(str)
                + " -> "
                + sub["service_street_pattern_class"].fillna("UNKNOWN").astype(str)
            )
        )
    sub = sub.assign(pattern_context=spec["pattern_context"], weight=1.0)
    return sub[PATTERN_RAW_COLUMNS].reset_index(drop=True)


def _aggregate_pattern_raw(
    raw_df: pd.DataFrame,
    *,
    by_city: bool = False,
    weight_col: str = "weight",
) -> pd.DataFrame:
    if raw_df.empty:
        if by_city:
            return pd.DataFrame(columns=["city", "pattern_context", "pattern_value", "n", "share"])
        return pd.DataFrame(columns=["pattern_context", "pattern_value", "n", "share"])

    work = raw_df.copy()
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0)
    if by_city:
        grouped = (
            work.groupby(["city", "pattern_context", "pattern_value"], as_index=False)[weight_col]
            .sum()
            .rename(columns={weight_col: "n"})
            .sort_values(["city", "n", "pattern_value"], ascending=[True, False, True])
            .reset_index(drop=True)
        )
        grouped["share"] = grouped["n"] / grouped.groupby("city")["n"].transform("sum")
        return grouped

    grouped = (
        work.groupby(["pattern_context", "pattern_value"], as_index=False)[weight_col]
        .sum()
        .rename(columns={weight_col: "n"})
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    grouped["share"] = grouped["n"] / grouped["n"].sum()
    return grouped


def _complete_city_pattern_summary(
    df: pd.DataFrame,
    city_order: list[str],
) -> pd.DataFrame:
    if not city_order:
        return df.copy()
    if df.empty:
        return pd.DataFrame(columns=["city", "pattern_context", "pattern_value", "n", "share"])
    pattern_values = sorted(df["pattern_value"].dropna().astype(str).unique().tolist())
    pattern_context = str(df["pattern_context"].iloc[0]) if "pattern_context" in df.columns and not df.empty else "unknown"
    base = pd.MultiIndex.from_product([city_order, pattern_values], names=["city", "pattern_value"]).to_frame(index=False)
    merged = base.merge(
        df[["city", "pattern_context", "pattern_value", "n", "share"]],
        on=["city", "pattern_value"],
        how="left",
    )
    merged["pattern_context"] = merged["pattern_context"].fillna(pattern_context)
    merged["n"] = pd.to_numeric(merged["n"], errors="coerce").fillna(0.0)
    merged["share"] = pd.to_numeric(merged["share"], errors="coerce").fillna(0.0)
    merged["city"] = pd.Categorical(merged["city"], categories=city_order, ordered=True)
    merged = merged.sort_values(["city", "n", "pattern_value"], ascending=[True, False, True]).reset_index(drop=True)
    merged["city"] = merged["city"].astype(str)
    return merged


def _aggregate_city_share_mean(city_df: pd.DataFrame) -> pd.DataFrame:
    if city_df.empty:
        return pd.DataFrame(columns=["pattern_context", "pattern_value", "n", "share"])
    grouped = (
        city_df.groupby(["pattern_context", "pattern_value"], as_index=False)["share"]
        .mean()
        .rename(columns={"share": "n"})
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    grouped["share"] = grouped["n"]
    return grouped


def _build_single_component_pattern_summaries_by_city(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    for key, spec in SINGLE_COMPONENT_PATTERN_SPECS.items():
        labels = spec["label"]
        if isinstance(labels, str):
            sub = df[df["access_diagnosis_label"] == labels].copy()
        else:
            sub = df[df["access_diagnosis_label"].isin(labels)].copy()
        if sub.empty or spec["pattern_context"] == "pt_route":
            summaries[key] = pd.DataFrame(columns=["city", "pattern_context", "pattern_value", "n", "share"])
            continue

        if spec["pattern_context"] == "home":
            sub = sub.assign(pattern_value=sub["home_street_pattern_class"].fillna("UNKNOWN").astype(str))
        elif spec["pattern_context"] == "service":
            sub = sub.assign(pattern_value=sub["service_street_pattern_class"].fillna("UNKNOWN").astype(str))
        else:
            sub = sub.assign(
                pattern_value=(
                    sub["home_street_pattern_class"].fillna("UNKNOWN").astype(str)
                    + " -> "
                    + sub["service_street_pattern_class"].fillna("UNKNOWN").astype(str)
                )
            )

        counts = (
            sub.groupby(["city", "pattern_value"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
            .sort_values(["city", "n", "pattern_value"], ascending=[True, False, True])
            .reset_index(drop=True)
        )
        counts["share"] = counts["n"] / counts.groupby("city")["n"].transform("sum")
        counts.insert(1, "pattern_context", spec["pattern_context"])
        summaries[key] = counts
    return summaries


def _build_pt_route_pattern_summary(route_class_df: pd.DataFrame) -> pd.DataFrame:
    if route_class_df.empty:
        return pd.DataFrame(columns=["pattern_context", "pattern_value", "n", "share"])
    work = route_class_df.copy()
    work["pt_length_m"] = pd.to_numeric(work["pt_length_m"], errors="coerce").fillna(0.0)
    work["street_pattern_class"] = work["street_pattern_class"].fillna("UNKNOWN").astype(str)
    grouped = (
        work.groupby("street_pattern_class", as_index=False)["pt_length_m"]
        .sum()
        .rename(columns={"street_pattern_class": "pattern_value", "pt_length_m": "n"})
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["n"].sum())
    grouped["share"] = grouped["n"] / total if total > 0 else 0.0
    grouped.insert(0, "pattern_context", "pt_route")
    return grouped


def _expand_path_route_patterns(path_routes: pd.DataFrame, route_class: pd.DataFrame) -> pd.DataFrame:
    merged = path_routes.merge(route_class, on=["city", "route_label"], how="left")
    merged["route_class_share"] = pd.to_numeric(merged["route_class_share"], errors="coerce").fillna(0.0)
    merged["route_time_min"] = pd.to_numeric(merged["route_time_min"], errors="coerce").fillna(0.0)
    merged["allocated_route_time_min"] = merged["route_time_min"] * merged["route_class_share"]
    return merged


def _best_edge_data(graph: nx.MultiDiGraph, u: int, v: int) -> dict:
    edge_bundle = graph.get_edge_data(u, v)
    if not edge_bundle:
        raise KeyError(f"missing edge for path step {u}->{v}")
    return min(edge_bundle.values(), key=lambda data: float(data.get("time_min", float("inf"))))


def _load_city_pt_top1(city: str, *, pt_walk_lt_root: Path, pt_walk_ge_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in (pt_walk_lt_root, pt_walk_ge_root):
        path = root / city / "residential_to_services_pt_top1.parquet"
        if not path.exists():
            continue
        frames.append(
            pd.read_parquet(
                path,
                columns=[
                    "building_idx",
                    "service_name",
                    "home_graph_node",
                    "nearest_service_graph_node",
                ],
            )
        )
    if not frames:
        return pd.DataFrame(columns=["building_idx", "service_name", "home_graph_node", "nearest_service_graph_node"])
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["building_idx", "service_name"], keep="first")
        .reset_index(drop=True)
    )


def _load_city_graph(city_dir: Path) -> tuple[gpd.GeoDataFrame, nx.MultiDiGraph]:
    nodes = gpd.read_parquet(city_dir / "intermodal_graph_iduedu" / "graph_nodes.parquet")
    with (city_dir / "intermodal_graph_iduedu" / "graph.pkl").open("rb") as fh:
        graph = pickle.load(fh)
    return nodes, graph


def _multi_source_pt_paths(
    graph: nx.MultiDiGraph,
    service_nodes: list[int],
) -> tuple[dict[int, float], dict[int, int], dict[int, list[int]]]:
    reverse_graph = graph.reverse(copy=False)
    distances, paths = nx.multi_source_dijkstra(reverse_graph, service_nodes, weight="time_min")
    source_map = {int(node): int(path[0]) for node, path in paths.items() if path}
    distance_map = {int(node): float(dist) for node, dist in distances.items()}
    path_map = {int(node): [int(p) for p in reversed(path)] for node, path in paths.items() if path}
    return distance_map, source_map, path_map


def _extract_transport_edges_for_path(
    graph: nx.MultiDiGraph,
    nodes: gpd.GeoDataFrame,
    path_nodes: list[int],
    *,
    city: str,
    home_graph_node: int,
    nearest_service_graph_node: int,
    usage_count: int,
) -> list[dict]:
    node_lookup = nodes.set_index("index")[["x", "y"]]
    rows: list[dict] = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:], strict=False):
        edge_data = _best_edge_data(graph, int(u), int(v))
        edge_type = str(edge_data.get("type", "") or "").lower()
        if edge_type in {"walk", "boarding"}:
            continue
        geom = edge_data.get("geometry")
        if geom is None or getattr(geom, "is_empty", False):
            if int(u) not in node_lookup.index or int(v) not in node_lookup.index:
                continue
            start = node_lookup.loc[int(u)]
            end = node_lookup.loc[int(v)]
            geom = LineString([(float(start["x"]), float(start["y"])), (float(end["x"]), float(end["y"]))])
        rows.append(
            {
                "city": city,
                "home_graph_node": int(home_graph_node),
                "nearest_service_graph_node": int(nearest_service_graph_node),
                "u": int(u),
                "v": int(v),
                "route_label": str(edge_data.get("route", "") or "").strip(),
                "edge_type": edge_type,
                "edge_time_min": float(edge_data.get("time_min", 0.0) or 0.0),
                "usage_count": int(usage_count),
                "geometry": geom,
            }
        )
    return rows


def _load_city_street_cells(city_dir: Path) -> gpd.GeoDataFrame:
    path = city_dir / "street_pattern" / city_dir.name / "predicted_cells.geojson"
    cells = gpd.read_file(path)
    if "top1_class_name" not in cells.columns:
        raise KeyError(f"Missing top1_class_name in {path}")
    return cells[["top1_class_name", "geometry"]].copy()


def _build_city_street_pattern_raw_from_cells(cells: pd.DataFrame, *, city: str) -> pd.DataFrame:
    work = cells.copy()
    work["pattern_context"] = "city_street_pattern"
    work["pattern_value"] = work["top1_class_name"].fillna("UNKNOWN").astype(str)
    work["city"] = str(city)
    work["weight"] = 1.0
    return work[["city", "pattern_context", "pattern_value", "weight"]].reset_index(drop=True)


def _build_all_city_street_pattern_raw(
    *,
    joint_inputs_root: Path,
    city_order: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for city in city_order:
        city_dir = joint_inputs_root / city
        cells = _load_city_street_cells(city_dir)
        parts.append(_build_city_street_pattern_raw_from_cells(cells, city=city))
    if not parts:
        return pd.DataFrame(columns=PATTERN_RAW_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def _filter_polyclinic_diagnostics_by_labels(
    diagnostics_df: pd.DataFrame,
    target_labels: set[str] | None,
) -> pd.DataFrame:
    mask = diagnostics_df["service_name"] == POLYCLINIC
    if target_labels is not None:
        mask &= diagnostics_df["access_diagnosis_label"].isin(target_labels)
    return diagnostics_df[mask][["city", "building_idx"]].drop_duplicates()


def _build_pt_path_pattern_raw(
    diagnostics_df: pd.DataFrame,
    *,
    joint_inputs_root: Path,
    pt_walk_lt_root: Path,
    pt_walk_ge_root: Path,
    target_labels: set[str] | None,
) -> pd.DataFrame:
    failed = _filter_polyclinic_diagnostics_by_labels(diagnostics_df, target_labels)
    if failed.empty:
        return pd.DataFrame(columns=PT_PATTERN_RAW_COLUMNS)

    raw_parts: list[pd.DataFrame] = []
    for city, city_failed in failed.groupby("city", dropna=False):
        city_dir = joint_inputs_root / str(city)
        pt_meta = _load_city_pt_top1(str(city), pt_walk_lt_root=pt_walk_lt_root, pt_walk_ge_root=pt_walk_ge_root)
        if pt_meta.empty:
            continue
        pt_meta = pt_meta[pt_meta["service_name"] == POLYCLINIC].copy()
        pt_meta = pt_meta.merge(city_failed, on=["building_idx"], how="inner")
        if pt_meta.empty:
            continue
        nodes, graph = _load_city_graph(city_dir)
        target_nodes = sorted(pt_meta["nearest_service_graph_node"].dropna().astype(int).unique().tolist())
        if not target_nodes:
            continue
        _, source_map, path_map = _multi_source_pt_paths(graph, target_nodes)
        path_counts = (
            pt_meta.assign(
                resolved_service_graph_node=lambda df_: df_["home_graph_node"].map(source_map)
            )
            .dropna(subset=["resolved_service_graph_node"])
            .query("resolved_service_graph_node == nearest_service_graph_node")
            .groupby(["home_graph_node", "resolved_service_graph_node"], as_index=False)
            .size()
            .rename(columns={"resolved_service_graph_node": "nearest_service_graph_node", "size": "usage_count"})
        )
        if path_counts.empty:
            continue
        transport_rows: list[dict] = []
        for row in path_counts.itertuples(index=False):
            path_nodes = path_map.get(int(row.home_graph_node))
            if not path_nodes:
                continue
            transport_rows.extend(
                _extract_transport_edges_for_path(
                    graph,
                    nodes,
                    path_nodes,
                    city=str(city),
                    home_graph_node=int(row.home_graph_node),
                    nearest_service_graph_node=int(row.nearest_service_graph_node),
                    usage_count=int(row.usage_count),
                )
            )
        if not transport_rows:
            continue
        edges = gpd.GeoDataFrame(transport_rows, geometry="geometry", crs=nodes.crs)
        edges = edges[edges.geometry.notna() & ~edges.geometry.is_empty].copy()
        if edges.empty:
            continue
        edges["edge_length_m"] = edges.geometry.length
        edges = edges[edges["edge_length_m"] > 0].copy()
        cells = _load_city_street_cells(city_dir).to_crs(edges.crs)
        overlay = gpd.overlay(edges, cells, how="intersection")
        overlay = overlay[overlay.geometry.notna() & ~overlay.geometry.is_empty].copy()
        if overlay.empty:
            continue
        overlay["intersect_length_m"] = overlay.geometry.length
        overlay = overlay[overlay["intersect_length_m"] > 0].copy()
        overlay["allocated_time_min"] = (
            overlay["edge_time_min"]
            * (overlay["intersect_length_m"] / overlay["edge_length_m"])
            * overlay["usage_count"]
        )
        overlay["pattern_context"] = "pt_path"
        overlay["pattern_value"] = overlay["top1_class_name"].fillna("UNKNOWN").astype(str)
        raw_parts.append(overlay[PT_PATTERN_RAW_COLUMNS].copy())

    if not raw_parts:
        return pd.DataFrame(columns=PT_PATTERN_RAW_COLUMNS)
    return pd.concat(raw_parts, ignore_index=True)


def _build_pt_path_pattern_summary(
    diagnostics_df: pd.DataFrame,
    *,
    joint_inputs_root: Path,
    pt_walk_lt_root: Path,
    pt_walk_ge_root: Path,
    target_labels: set[str] | None,
    by_city: bool = False,
) -> pd.DataFrame:
    raw_df = _build_pt_path_pattern_raw(
        diagnostics_df,
        joint_inputs_root=joint_inputs_root,
        pt_walk_lt_root=pt_walk_lt_root,
        pt_walk_ge_root=pt_walk_ge_root,
        target_labels=target_labels,
    )
    return _aggregate_pattern_raw(raw_df, by_city=by_city, weight_col="allocated_time_min")


def _load_or_build_pattern_raw(
    raw_path: Path,
    builder: Callable[[], pd.DataFrame],
    *,
    rebuild: bool,
) -> pd.DataFrame:
    if raw_path.exists() and not rebuild:
        return pd.read_parquet(raw_path)
    raw_df = builder()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(raw_path, index=False)
    return raw_df


def _top_pattern_values(df: pd.DataFrame, top_n: int) -> list[str]:
    if df.empty:
        return []
    totals = df.groupby("pattern_value", as_index=False)["n"].sum().sort_values("n", ascending=False)
    return totals.head(top_n)["pattern_value"].tolist()


def _city_pattern_matrix(
    df: pd.DataFrame,
    *,
    selected: list[str],
    city_order: list[str] | None = None,
) -> pd.DataFrame:
    plot_df = (
        df[df["pattern_value"].isin(selected)]
        .pivot(index="city", columns="pattern_value", values="share")
        .fillna(0.0)
    )
    plot_df = plot_df.reindex(columns=selected)
    if city_order is not None:
        plot_df = plot_df.reindex(city_order).fillna(0.0)
    return plot_df


def _plot_pattern_barh(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    color: str = "#2563eb",
    top_n: int = 12,
) -> None:
    plot_df = df.head(top_n).copy()
    bars = ax.barh(plot_df["pattern_value"], plot_df["share"], color=color)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share within case")
    ax.set_title(f"{title} (sum={float(plot_df['share'].sum()):.3f})")
    ax.invert_yaxis()
    for bar, value in zip(bars, plot_df["share"], strict=False):
        ax.text(min(float(value) + 0.01, 0.98), bar.get_y() + bar.get_height() / 2, f"{float(value):.3f}", va="center")


def _plot_pattern_heatmap(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    *,
    title: str,
    vmax: float,
) -> any:
    im = ax.imshow(plot_df.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = float(plot_df.iloc[i, j])
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.5, color="#111827")
    return im


def _render_single_component_pattern_png(df: pd.DataFrame, title: str, out_path: Path, top_n: int = 12) -> None:
    if df.empty:
        return
    plot_df = df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(11, max(4, 0.45 * len(plot_df) + 1.5)), dpi=220)
    _plot_pattern_barh(ax, plot_df, title=title, top_n=top_n)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.04)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _combined_pattern_layout_specs() -> list[dict[str, object]]:
    return [
        {
            "key": "home_to_stop_not_ok",
            "title": "Graph 3a: home -> stop not OK by home street pattern",
            "color": "#2563eb",
            "grid_row": 0,
        },
        {
            "key": "stop_to_service_not_ok",
            "title": "Graph 3b: stop -> service not OK by service street pattern",
            "color": "#2563eb",
            "grid_row": 1,
        },
        {
            "key": "pt_segment_not_ok",
            "title": "Graph 3c: PT-too-long path edges by street pattern class",
            "color": "#ea580c",
            "grid_row": 3,
        },
        {
            "key": "sum_no_single_component_not_ok",
            "title": "Graph 4: total > T, each component <= T, walks sum < T",
            "color": "#ea580c",
            "grid_row": 4,
        },
        {
            "key": "all_polyclinic_pt_paths",
            "title": "Graph 5: all polyclinic PT path edges by street pattern class",
            "color": "#059669",
            "grid_row": 6,
        },
    ]


def _render_combined_single_component_patterns_png(
    summaries: dict[str, pd.DataFrame],
    out_path: Path,
    top_n: int = 10,
) -> None:
    specs = _combined_pattern_layout_specs()
    fig = plt.figure(figsize=(11, 18), dpi=220)
    grid = fig.add_gridspec(
        7,
        1,
        height_ratios=[1.0, 1.0, 0.22, 1.0, 1.0, 0.22, 1.0],
        hspace=0.32,
    )
    global_max = 0.0
    trimmed: dict[str, pd.DataFrame] = {}
    for spec in specs:
        key = str(spec["key"])
        df = summaries.get(key, pd.DataFrame()).head(top_n).copy()
        trimmed[key] = df
        if not df.empty:
            global_max = max(global_max, float(df["share"].max()))
    for spec in specs:
        key = str(spec["key"])
        title = str(spec["title"])
        color = str(spec["color"])
        ax = fig.add_subplot(grid[int(spec["grid_row"]), 0])
        df = trimmed.get(key, pd.DataFrame())
        if df.empty:
            ax.set_axis_off()
            continue
        bars = ax.barh(df["pattern_value"], df["share"], color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share within case")
        ax.set_title(f"{title} (sum={float(df['share'].sum()):.3f})")
        ax.invert_yaxis()
        for bar, value in zip(bars, df["share"], strict=False):
            ax.text(min(float(value) + 0.01, 0.98), bar.get_y() + bar.get_height() / 2, f"{float(value):.3f}", va="center")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.04)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_city_pattern_heatmap_png(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    top_n: int = 10,
    city_order: list[str] | None = None,
) -> None:
    if df.empty:
        return
    selected = _top_pattern_values(df, top_n)
    plot_df = _city_pattern_matrix(df, selected=selected, city_order=city_order)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(plot_df.index) + 2)), dpi=220)
    im = _plot_pattern_heatmap(
        ax,
        plot_df,
        title=title,
        vmax=min(1.0, float(plot_df.values.max()) if plot_df.size else 1.0),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("share within city")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_stacked_city_pattern_heatmaps_png(
    top_df: pd.DataFrame,
    bottom_df: pd.DataFrame,
    *,
    top_title: str,
    bottom_title: str,
    out_path: Path,
    top_n: int = 10,
    city_order: list[str] | None = None,
) -> None:
    if top_df.empty or bottom_df.empty:
        return
    combined = pd.concat([top_df, bottom_df], ignore_index=True)
    selected = _top_pattern_values(combined, top_n)
    top_plot = _city_pattern_matrix(top_df, selected=selected, city_order=city_order)
    bottom_plot = _city_pattern_matrix(bottom_df, selected=selected, city_order=city_order)
    vmax = min(
        1.0,
        max(
            float(top_plot.values.max()) if top_plot.size else 0.0,
            float(bottom_plot.values.max()) if bottom_plot.size else 0.0,
        ),
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, max(10, 0.32 * (len(top_plot.index) + len(bottom_plot.index)) + 4)),
        dpi=220,
    )
    for ax, plot_df, title in zip(axes, [top_plot, bottom_plot], [top_title, bottom_title], strict=False):
        im = _plot_pattern_heatmap(ax, plot_df, title=title, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("share within city")
    fig.subplots_adjust(left=0.2, right=0.92, top=0.96, bottom=0.05, hspace=0.28)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_overall_and_city_pattern_png(
    overall_df: pd.DataFrame,
    city_df: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    top_n: int = 10,
    city_order: list[str] | None = None,
    color: str = "#2563eb",
) -> None:
    if overall_df.empty or city_df.empty:
        return
    selected = _top_pattern_values(overall_df, top_n)
    overall_plot = overall_df[overall_df["pattern_value"].isin(selected)].copy()
    city_plot = _city_pattern_matrix(city_df, selected=selected, city_order=city_order)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, max(10, 0.35 * len(city_plot.index) + 6)),
        dpi=220,
    )
    _plot_pattern_barh(axes[0], overall_plot, title=title, color=color, top_n=top_n)
    im = _plot_pattern_heatmap(
        axes[1],
        city_plot,
        title=f"{title} by city",
        vmax=min(1.0, float(city_plot.values.max()) if city_plot.size else 1.0),
    )
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.025, pad=0.02)
    cbar.set_label("share within city")
    fig.subplots_adjust(left=0.2, right=0.92, top=0.96, bottom=0.05, hspace=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_pt_overall_and_city_pattern_png(
    overall_df: pd.DataFrame,
    case_city_df: pd.DataFrame,
    baseline_city_df: pd.DataFrame,
    *,
    title: str,
    baseline_title: str,
    out_path: Path,
    top_n: int = 10,
    city_order: list[str] | None = None,
    color: str = "#ea580c",
) -> None:
    if overall_df.empty or case_city_df.empty or baseline_city_df.empty:
        return
    selected = _top_pattern_values(pd.concat([overall_df, baseline_city_df], ignore_index=True), top_n)
    overall_plot = overall_df[overall_df["pattern_value"].isin(selected)].copy()
    case_plot = _city_pattern_matrix(case_city_df, selected=selected, city_order=city_order)
    baseline_plot = _city_pattern_matrix(baseline_city_df, selected=selected, city_order=city_order)
    vmax = min(
        1.0,
        max(
            float(case_plot.values.max()) if case_plot.size else 0.0,
            float(baseline_plot.values.max()) if baseline_plot.size else 0.0,
        ),
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, max(14, 0.32 * (len(case_plot.index) + len(baseline_plot.index)) + 8)),
        dpi=220,
    )
    _plot_pattern_barh(axes[0], overall_plot, title=title, color=color, top_n=top_n)
    im = _plot_pattern_heatmap(axes[1], case_plot, title=f"{title} by city", vmax=vmax)
    _plot_pattern_heatmap(axes[2], baseline_plot, title=baseline_title, vmax=vmax)
    cbar = fig.colorbar(im, ax=axes[1:], fraction=0.025, pad=0.02)
    cbar.set_label("share within city")
    fig.subplots_adjust(left=0.2, right=0.92, top=0.97, bottom=0.05, hspace=0.32)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_four_panel_pattern_png(
    top_overall_df: pd.DataFrame,
    upper_city_df: pd.DataFrame,
    lower_overall_df: pd.DataFrame,
    lower_city_df: pd.DataFrame,
    *,
    top_title: str,
    upper_city_title: str,
    lower_overall_title: str,
    lower_city_title: str,
    out_path: Path,
    top_n: int = 10,
    city_order: list[str] | None = None,
    top_color: str = "#059669",
    lower_overall_color: str = "#6b7280",
) -> None:
    if top_overall_df.empty or upper_city_df.empty or lower_overall_df.empty or lower_city_df.empty:
        return
    selected = _top_pattern_values(
        pd.concat([top_overall_df, upper_city_df, lower_overall_df, lower_city_df], ignore_index=True),
        top_n,
    )
    top_overall_plot = top_overall_df[top_overall_df["pattern_value"].isin(selected)].copy()
    upper_city_plot = _city_pattern_matrix(upper_city_df, selected=selected, city_order=city_order)
    lower_overall_plot = lower_overall_df[lower_overall_df["pattern_value"].isin(selected)].copy()
    lower_city_plot = _city_pattern_matrix(lower_city_df, selected=selected, city_order=city_order)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, max(18, 0.22 * (len(upper_city_plot.index) + len(lower_city_plot.index)) + 10)),
        dpi=220,
    )
    _plot_pattern_barh(axes[0], top_overall_plot, title=top_title, color=top_color, top_n=top_n)
    im = _plot_pattern_heatmap(axes[1], upper_city_plot, title=upper_city_title, vmax=1.0)
    _plot_pattern_barh(axes[2], lower_overall_plot, title=lower_overall_title, color=lower_overall_color, top_n=top_n)
    _plot_pattern_heatmap(axes[3], lower_city_plot, title=lower_city_title, vmax=1.0)
    cbar = fig.colorbar(im, ax=[axes[1], axes[3]], fraction=0.025, pad=0.02)
    cbar.set_label("share within city")
    fig.subplots_adjust(left=0.2, right=0.92, top=0.97, bottom=0.05, hspace=0.28)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _requested_overall_sections(row: pd.Series) -> list[tuple[str, list[tuple[str, float, str]]]]:
    return [
        (
            "Overall",
            [
                ("OK overall", float(row["pct_ok_overall"]), "#15803d"),
                ("Not OK overall", float(row["pct_not_ok_overall"]), "#b91c1c"),
            ],
        ),
        (
            "OK breakdown",
            [
                ("OK walk only", float(row["pct_ok_walk_only"]), "#166534"),
                ("OK walk + PT", float(row["pct_ok_walk_plus_pt"]), "#1d4ed8"),
            ],
        ),
        (
            "Single-component not OK",
            [
                ("Home -> stop not OK", float(row["pct_not_ok_home_to_stop_overall"]), "#d97706"),
                ("PT segment not OK", float(row["pct_not_ok_pt_only_overall"]), "#7c3aed"),
                ("Stop -> service not OK", float(row["pct_not_ok_stop_to_service_overall"]), "#db2777"),
            ],
        ),
        (
            "Combined / multi-component not OK",
            [
                ("Walks sum > T (each <= T)", float(row["pct_not_ok_both_walks_overall"]), "#ea580c"),
                ("Multi-component", float(row["pct_not_ok_multi_component_overall"]), "#6d28d9"),
                ("Total sum > T, no single component > T", float(row["pct_not_ok_sum_no_single_overall"]), "#6b7280"),
            ],
        ),
    ]


def _render_requested_summary_overall_png(overall_df: pd.DataFrame, out_path: Path) -> None:
    row = overall_df.iloc[0]
    sections = _requested_overall_sections(row)

    fig, axes = plt.subplots(len(sections), 1, figsize=(11, 11), dpi=220)
    if len(sections) == 1:
        axes = [axes]
    fig.suptitle("Polyclinic accessibility summary", fontsize=14, y=0.98)

    for idx, (ax, (title, items)) in enumerate(zip(axes, sections, strict=False), start=1):
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        colors = [item[2] for item in items]
        bars = ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share")
        ax.set_title(f"Graph {idx}: {title} (sum={sum(values):.3f})", fontsize=11)
        ax.invert_yaxis()
        for bar, value in zip(bars, values, strict=False):
            ax.text(
                min(value + 0.015, 0.98),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_requested_summary_by_city_png(city_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = city_df.set_index("city")[REQUESTED_COLUMNS].copy()
    plot_df.columns = [REQUESTED_LABELS[col] for col in plot_df.columns]

    fig, ax = plt.subplots(figsize=(12, max(8, 0.38 * len(plot_df))), dpi=220)
    im = ax.imshow(plot_df.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Polyclinic accessibility summary by city")
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = float(plot_df.iloc[i, j])
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("share")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--joint-inputs-root", type=Path, default=DEFAULT_JOINT_INPUTS_ROOT)
    parser.add_argument("--pt-walk-lt-root", type=Path, default=DEFAULT_PT_WALK_LT_ROOT)
    parser.add_argument("--pt-walk-ge-root", type=Path, default=DEFAULT_PT_WALK_GE_ROOT)
    parser.add_argument("--threshold-min", type=float, default=15.0)
    parser.add_argument("--rebuild-pattern-raw", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    poly = _load_polyclinic(args.input)
    flagged = _add_component_flags(poly, threshold_min=float(args.threshold_min))
    city_order = _sorted_city_order(flagged)
    overall = _build_overall_summary(flagged)
    requested = _build_requested_summary(flagged)
    summary = _build_component_summary(flagged)

    flagged.to_parquet(args.out_root / "polyclinic_home_access_components.parquet", index=False)
    _round_for_export(overall).to_csv(args.out_root / "polyclinic_overall_ok_summary_by_city.csv", index=False)
    _round_for_export(requested).to_csv(args.out_root / "polyclinic_requested_summary_by_city.csv", index=False)
    _round_for_export(summary).to_csv(args.out_root / "polyclinic_component_ok_summary_by_city.csv", index=False)
    parts_root = args.out_root / "components"
    parts_root.mkdir(parents=True, exist_ok=True)
    for component in COMPONENT_COLUMNS:
        component_df = summary[summary["component"] == component].copy().reset_index(drop=True)
        _round_for_export(component_df).to_csv(parts_root / f"{component}_by_city.csv", index=False)

    overall_totals = {
        "n": int(overall["n"].sum()),
        "ok_count": int(overall["ok_count"].sum()),
        "not_ok_count": int(overall["not_ok_count"].sum()),
    }
    overall_export = pd.DataFrame(
        [
            {
                "scope": "all",
                "n": overall_totals["n"],
                "ok_count": overall_totals["ok_count"],
                "not_ok_count": overall_totals["not_ok_count"],
                "share_ok": float(overall_totals["ok_count"] / overall_totals["n"]),
            }
        ]
    )
    _round_for_export(overall_export).to_csv(args.out_root / "polyclinic_overall_ok_summary_overall.csv", index=False)

    requested_overall = {
        "n": int(requested["n"].sum()),
    }
    requested_overall_df = pd.DataFrame(
        [
            {
                "scope": "all",
                "n": requested_overall["n"],
                "pct_ok_overall": float((flagged["access_diagnosis_label"].isin(OK_LABELS)).mean()),
                "pct_not_ok_overall": float((~flagged["access_diagnosis_label"].isin(OK_LABELS)).mean()),
                "pct_ok_walk_only": float((flagged["access_diagnosis_label"] == "ok_walk").mean()),
                "pct_ok_walk_plus_pt": float((flagged["access_diagnosis_label"] == "ok_pt_only").mean()),
                "pct_not_ok_home_to_stop_overall": float(
                    (flagged["access_diagnosis_label"] == "failed_access_gt_threshold").mean()
                ),
                "pct_not_ok_pt_only_overall": float(
                    flagged["access_diagnosis_label"].isin(
                        ["failed_in_vehicle_gt_threshold", "failed_transfer_gt_threshold", "failed_no_pt_path"]
                    ).mean()
                ),
                "pct_not_ok_stop_to_service_overall": float(
                    (flagged["access_diagnosis_label"] == "failed_egress_gt_threshold").mean()
                ),
                "pct_not_ok_both_walks_overall": float(
                    (flagged["access_diagnosis_label"] == "failed_access_egress_sum_gt_threshold").mean()
                ),
                "pct_not_ok_multi_component_overall": float(
                    (flagged["access_diagnosis_label"] == "failed_multi_component_gt_threshold").mean()
                ),
                "pct_not_ok_sum_no_single_overall": float(
                    (
                        flagged["access_diagnosis_label"]
                        == "failed_total_gt_threshold_no_single_component_gt_threshold"
                    ).mean()
                ),
            }
        ]
    )
    requested_overall_export = _round_for_export(requested_overall_df)
    requested_export = _round_for_export(requested)
    requested_overall_export.to_csv(args.out_root / "polyclinic_requested_summary_overall.csv", index=False)
    _render_requested_summary_overall_png(
        requested_overall_export,
        args.out_root / "polyclinic_requested_summary_overall.png",
    )
    _render_requested_summary_by_city_png(
        requested_export,
        args.out_root / "polyclinic_requested_summary_by_city.png",
    )

    component_overall = (
        summary.groupby("component", as_index=False)
        .agg(
            n=("n", "sum"),
            ok_count=("ok_count", "sum"),
            not_ok_count=("not_ok_count", "sum"),
        )
        .assign(share_ok=lambda df_: df_["ok_count"] / df_["n"])
        .sort_values("component")
        .reset_index(drop=True)
    )
    _round_for_export(component_overall).to_csv(args.out_root / "polyclinic_component_ok_summary_overall.csv", index=False)
    for component in COMPONENT_COLUMNS:
        component_df = component_overall[component_overall["component"] == component].copy().reset_index(drop=True)
        _round_for_export(component_df).to_csv(parts_root / f"{component}_overall.csv", index=False)

    pattern_root = args.out_root / "single_component_patterns"
    pattern_root.mkdir(parents=True, exist_ok=True)
    raw_root = pattern_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    pattern_raws: dict[str, pd.DataFrame] = {}
    for key in ("home_to_stop_not_ok", "stop_to_service_not_ok"):
        pattern_raws[key] = _load_or_build_pattern_raw(
            raw_root / f"{key}_raw.parquet",
            lambda key_=key: _build_single_component_pattern_raw(flagged, key_),
            rebuild=bool(args.rebuild_pattern_raw),
        )
    pattern_raws["pt_segment_not_ok"] = _load_or_build_pattern_raw(
        raw_root / "pt_segment_not_ok_raw.parquet",
        lambda: _build_pt_path_pattern_raw(
            flagged,
            joint_inputs_root=args.joint_inputs_root,
            pt_walk_lt_root=args.pt_walk_lt_root,
            pt_walk_ge_root=args.pt_walk_ge_root,
            target_labels={"failed_in_vehicle_gt_threshold", "failed_transfer_gt_threshold"},
        ),
        rebuild=bool(args.rebuild_pattern_raw),
    )
    pattern_raws["sum_no_single_component_not_ok"] = _load_or_build_pattern_raw(
        raw_root / "sum_no_single_component_not_ok_raw.parquet",
        lambda: _build_pt_path_pattern_raw(
            flagged,
            joint_inputs_root=args.joint_inputs_root,
            pt_walk_lt_root=args.pt_walk_lt_root,
            pt_walk_ge_root=args.pt_walk_ge_root,
            target_labels={"failed_total_gt_threshold_no_single_component_gt_threshold"},
        ),
        rebuild=bool(args.rebuild_pattern_raw),
    )
    pattern_raws["all_polyclinic_pt_paths"] = _load_or_build_pattern_raw(
        raw_root / "all_polyclinic_pt_paths_raw.parquet",
        lambda: _build_pt_path_pattern_raw(
            flagged,
            joint_inputs_root=args.joint_inputs_root,
            pt_walk_lt_root=args.pt_walk_lt_root,
            pt_walk_ge_root=args.pt_walk_ge_root,
            target_labels=None,
        ),
        rebuild=bool(args.rebuild_pattern_raw),
    )
    pattern_raws["all_city_street_pattern"] = _load_or_build_pattern_raw(
        raw_root / "all_city_street_pattern_raw.parquet",
        lambda: _build_all_city_street_pattern_raw(
            joint_inputs_root=args.joint_inputs_root,
            city_order=city_order,
        ),
        rebuild=bool(args.rebuild_pattern_raw),
    )
    pattern_summaries = {
        key: _aggregate_pattern_raw(
            raw_df,
            by_city=False,
            weight_col="allocated_time_min" if key in {"pt_segment_not_ok", "sum_no_single_component_not_ok", "all_polyclinic_pt_paths"} else "weight",
        )
        for key, raw_df in pattern_raws.items()
    }
    pattern_summaries_by_city = {
        key: _complete_city_pattern_summary(
            _aggregate_pattern_raw(
                raw_df,
                by_city=True,
                weight_col="allocated_time_min" if key in {"pt_segment_not_ok", "sum_no_single_component_not_ok", "all_polyclinic_pt_paths"} else "weight",
            ),
            city_order=city_order,
        )
        for key, raw_df in pattern_raws.items()
    }
    pattern_titles = {
        "home_to_stop_not_ok": "Graph 3a: home -> stop not OK by home street pattern",
        "stop_to_service_not_ok": "Graph 3b: stop -> service not OK by service street pattern",
        "pt_segment_not_ok": "Graph 3c: PT-too-long path edges by street pattern class",
    }
    for key, title in pattern_titles.items():
        summary_df = pattern_summaries[key]
        pattern_raws[key].to_parquet(raw_root / f"{key}_raw.parquet", index=False)
        export_df = _round_for_export(summary_df)
        export_df.to_csv(pattern_root / f"{key}.csv", index=False)
        _render_single_component_pattern_png(
            export_df,
            title=title,
            out_path=pattern_root / f"{key}.png",
        )
        city_export_df = _round_for_export(pattern_summaries_by_city.get(key, pd.DataFrame()))
        city_export_df.to_csv(pattern_root / f"{key}_by_city.csv", index=False)
        _render_overall_and_city_pattern_png(
            export_df,
            city_export_df,
            title=title,
            out_path=pattern_root / f"{key}_by_city.png",
            city_order=city_order,
            color="#ea580c" if key == "pt_segment_not_ok" else "#2563eb",
        )

    sum_no_single_df = _round_for_export(pattern_summaries["sum_no_single_component_not_ok"])
    sum_no_single_df.to_csv(pattern_root / "sum_no_single_component_not_ok.csv", index=False)
    _render_single_component_pattern_png(
        sum_no_single_df,
        title="Graph 4: PT path street patterns for cases where total > T, each component <= T, walks sum < T",
        out_path=pattern_root / "sum_no_single_component_not_ok.png",
    )
    sum_no_single_by_city_df = _round_for_export(pattern_summaries_by_city["sum_no_single_component_not_ok"])
    sum_no_single_by_city_df.to_csv(pattern_root / "sum_no_single_component_not_ok_by_city.csv", index=False)
    _render_overall_and_city_pattern_png(
        sum_no_single_df,
        sum_no_single_by_city_df,
        title="Graph 4: PT path street patterns for cases where total > T, each component <= T, walks sum < T",
        out_path=pattern_root / "sum_no_single_component_not_ok_by_city.png",
        city_order=city_order,
        color="#ea580c",
    )

    all_polyclinic_pt_paths_df = _round_for_export(pattern_summaries["all_polyclinic_pt_paths"])
    all_polyclinic_pt_paths_df.to_csv(pattern_root / "all_polyclinic_pt_paths.csv", index=False)
    _render_single_component_pattern_png(
        all_polyclinic_pt_paths_df,
        title="Graph 5: all polyclinic PT path edges by street pattern class",
        out_path=pattern_root / "all_polyclinic_pt_paths.png",
    )
    all_polyclinic_pt_paths_by_city_df = _round_for_export(pattern_summaries_by_city["all_polyclinic_pt_paths"])
    all_polyclinic_pt_paths_by_city_df.to_csv(pattern_root / "all_polyclinic_pt_paths_by_city.csv", index=False)
    _render_overall_and_city_pattern_png(
        all_polyclinic_pt_paths_df,
        all_polyclinic_pt_paths_by_city_df,
        title="Graph 5: all polyclinic PT path edges by street pattern class",
        out_path=pattern_root / "all_polyclinic_pt_paths_by_city.png",
        city_order=city_order,
        color="#059669",
    )
    all_city_street_pattern_by_city_df = _round_for_export(pattern_summaries_by_city["all_city_street_pattern"])
    all_city_street_pattern_by_city_df.to_csv(pattern_root / "all_city_street_pattern_by_city.csv", index=False)
    all_city_street_pattern_df = _round_for_export(
        _aggregate_city_share_mean(pattern_summaries_by_city["all_city_street_pattern"])
    )
    all_city_street_pattern_df.to_csv(pattern_root / "all_city_street_pattern.csv", index=False)
    _render_four_panel_pattern_png(
        all_polyclinic_pt_paths_df,
        all_polyclinic_pt_paths_by_city_df,
        all_city_street_pattern_df,
        all_city_street_pattern_by_city_df,
        top_title="Graph 5: all polyclinic PT path edges by street pattern class",
        upper_city_title="Graph 5: all polyclinic PT path edges by street pattern class by city",
        lower_overall_title="Baseline: all street pattern distribution",
        lower_city_title="Baseline: all street pattern distribution by city",
        out_path=pattern_root / "all_polyclinic_pt_paths_vs_all_city_street_pattern_by_city.png",
        city_order=city_order,
        top_color="#059669",
        lower_overall_color="#6b7280",
    )
    _render_pt_overall_and_city_pattern_png(
        _round_for_export(pattern_summaries["pt_segment_not_ok"]),
        _round_for_export(pattern_summaries_by_city["pt_segment_not_ok"]),
        all_polyclinic_pt_paths_by_city_df,
        title="Graph 3c: PT-too-long path edges by street pattern class",
        baseline_title="Baseline: all polyclinic PT path edges by city",
        out_path=pattern_root / "pt_segment_not_ok_by_city.png",
        city_order=city_order,
    )

    combined_pattern_summaries = {key: _round_for_export(df) for key, df in pattern_summaries.items()}
    combined_pattern_summaries["sum_no_single_component_not_ok"] = sum_no_single_df
    combined_pattern_summaries["all_polyclinic_pt_paths"] = all_polyclinic_pt_paths_df
    _render_combined_single_component_patterns_png(
        combined_pattern_summaries,
        out_path=pattern_root / "single_component_patterns_combined.png",
    )


if __name__ == "__main__":
    main()

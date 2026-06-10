#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_INPUT = Path(
    "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/_all_home_to_service_access_diagnostics.parquet"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Users/gk/Code/super-duper-disser/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/service_access_diagnostics/pattern_tables"
)
SERVICES = ["hospital", "polyclinic", "school", "kindergarten"]
STREET_PATTERN_ORDER = [
    "Regular Grid",
    "Irregular Grid",
    "Warped Parallel",
    "Broken Grid",
    "Sparse",
    "Loops & Lollipops",
]
LABEL_ORDER = [
    "ok_walk",
    "ok_pt_only",
    "failed_no_pt_path",
    "failed_access_gt_threshold",
    "failed_egress_gt_threshold",
    "failed_access_egress_sum_gt_threshold",
    "failed_in_vehicle_gt_threshold",
    "failed_transfer_gt_threshold",
    "failed_multi_component_gt_threshold",
    "failed_total_gt_threshold_no_single_component_gt_threshold",
]
LABEL_DISPLAY = {
    "ok_walk": "Walk OK",
    "ok_pt_only": "PT Only OK",
    "failed_no_pt_path": "No PT Path",
    "failed_access_gt_threshold": "Home->Stop > T",
    "failed_egress_gt_threshold": "Stop->Service > T",
    "failed_access_egress_sum_gt_threshold": "Both Walks > T",
    "failed_in_vehicle_gt_threshold": "In-Vehicle > T",
    "failed_transfer_gt_threshold": "Transfer > T",
    "failed_multi_component_gt_threshold": "Multi > T",
    "failed_total_gt_threshold_no_single_component_gt_threshold": "Sum > T, None > T",
}

sns.set_theme(style="whitegrid", context="notebook")


def _ordered_patterns(values: pd.Series) -> list[str]:
    present = set(values.dropna().astype(str))
    known = [pattern for pattern in STREET_PATTERN_ORDER if pattern in present]
    extra = sorted(pattern for pattern in present if pattern not in STREET_PATTERN_ORDER and pattern.upper() != "UNKNOWN")
    return known + extra


def _service_matrix(
    df: pd.DataFrame,
    service: str,
    pattern_column: str = "home_street_pattern_class",
    label_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if label_order is None:
        label_order = LABEL_ORDER
    sub = df[df["service_name"] == service].copy()
    sub[pattern_column] = sub[pattern_column].fillna("UNKNOWN").astype(str)
    excluded_unknown = int((sub[pattern_column].str.upper() == "UNKNOWN").sum())
    sub = sub[sub[pattern_column].str.upper() != "UNKNOWN"].copy()
    if sub.empty:
        counts = pd.DataFrame(index=[], columns=label_order).fillna(0).astype(int)
        shares = counts.astype(float)
        return counts, shares, excluded_unknown

    pattern_order = _ordered_patterns(sub[pattern_column])
    counts = (
        sub.groupby([pattern_column, "access_diagnosis_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=pattern_order, columns=label_order, fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0)
    return counts, shares, excluded_unknown


def _pair_label(home_pattern: str, service_pattern: str) -> str:
    return f"{home_pattern} -> {service_pattern}"


def _pair_matrix(
    df: pd.DataFrame,
    service: str,
    label_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if label_order is None:
        label_order = LABEL_ORDER
    sub = df[df["service_name"] == service].copy()
    sub["home_street_pattern_class"] = sub["home_street_pattern_class"].fillna("UNKNOWN").astype(str)
    sub["service_street_pattern_class"] = sub["service_street_pattern_class"].fillna("UNKNOWN").astype(str)
    unknown_mask = (
        (sub["home_street_pattern_class"].str.upper() == "UNKNOWN")
        | (sub["service_street_pattern_class"].str.upper() == "UNKNOWN")
    )
    excluded_unknown = int(unknown_mask.sum())
    sub = sub[~unknown_mask].copy()
    if sub.empty:
        counts = pd.DataFrame(index=[], columns=label_order).fillna(0).astype(int)
        shares = counts.astype(float)
        return counts, shares, excluded_unknown

    home_order = {pattern: idx for idx, pattern in enumerate(STREET_PATTERN_ORDER)}
    service_order = {pattern: idx for idx, pattern in enumerate(STREET_PATTERN_ORDER)}
    sub["pair_label"] = sub.apply(
        lambda row: _pair_label(row["home_street_pattern_class"], row["service_street_pattern_class"]),
        axis=1,
    )
    pair_order = (
        sub[["home_street_pattern_class", "service_street_pattern_class", "pair_label"]]
        .drop_duplicates()
        .sort_values(
            by=["home_street_pattern_class", "service_street_pattern_class"],
            key=lambda col: col.map(
                (home_order if col.name == "home_street_pattern_class" else service_order)
            ).fillna(len(STREET_PATTERN_ORDER))
            if col.name in {"home_street_pattern_class", "service_street_pattern_class"}
            else col,
        )["pair_label"]
        .tolist()
    )
    counts = (
        sub.groupby(["pair_label", "access_diagnosis_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=pair_order, columns=label_order, fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0)
    return counts, shares, excluded_unknown


def _render_service_heatmap(
    service: str,
    counts: pd.DataFrame,
    shares: pd.DataFrame,
    excluded_unknown: int,
    out_path: Path,
    title_suffix: str,
    colorbar_label: str,
) -> None:
    if shares.empty:
        return

    annot = shares.apply(lambda col: col.map(lambda x: f"{100 * x:.0f}%"))
    y_labels = [f"{pattern} (n={int(counts.loc[pattern].sum())})" for pattern in shares.index]

    fig, ax = plt.subplots(figsize=(11, max(3.8, 0.9 * len(shares.index) + 1.8)), dpi=220)
    sns.heatmap(
        shares,
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.6,
        linecolor="white",
        annot=annot,
        fmt="",
        cbar_kws={"label": colorbar_label},
        ax=ax,
    )
    ax.set_title(f"{service}: {title_suffix}", fontsize=13, pad=12)
    ax.set_xlabel("diagnosis label")
    ax.set_ylabel("home street pattern")
    ax.set_xticklabels([LABEL_DISPLAY.get(label, label) for label in shares.columns], rotation=25, ha="right")
    ax.set_yticklabels(y_labels, rotation=0)
    if excluded_unknown > 0:
        ax.text(
            0.0,
            -0.16,
            f"excluded homes without street-pattern class: {excluded_unknown}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#475569",
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_combined_service_heatmaps(
    matrices: list[tuple[str, pd.DataFrame, pd.DataFrame, int]],
    out_path: Path,
    title_suffix: str,
    colorbar_label: str,
) -> None:
    if not matrices:
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=220)
    axes = axes.flatten()
    last_mappable = None

    for ax, (service, counts, shares, excluded_unknown) in zip(axes, matrices, strict=False):
        annot = shares.apply(lambda col: col.map(lambda x: f"{100 * x:.0f}%"))
        y_labels = [f"{pattern} (n={int(counts.loc[pattern].sum())})" for pattern in shares.index]
        hm = sns.heatmap(
            shares,
            cmap="Blues",
            vmin=0,
            vmax=1,
            linewidths=0.6,
            linecolor="white",
            annot=annot,
            fmt="",
            cbar=False,
            ax=ax,
        )
        last_mappable = hm.collections[0]
        ax.set_title(service, fontsize=12, pad=10)
        ax.set_xlabel("diagnosis label")
        ax.set_ylabel("home street pattern")
        ax.set_xticklabels([LABEL_DISPLAY.get(label, label) for label in shares.columns], rotation=25, ha="right")
        ax.set_yticklabels(y_labels, rotation=0)
        if excluded_unknown > 0:
            ax.text(
                0.0,
                -0.18,
                f"excluded homes without street-pattern class: {excluded_unknown}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#475569",
            )

    for ax in axes[len(matrices) :]:
        ax.set_axis_off()

    if last_mappable is not None:
        cbar_ax = fig.add_axes([0.93, 0.16, 0.018, 0.68])
        cbar = fig.colorbar(last_mappable, cax=cbar_ax)
        cbar.set_label(colorbar_label)

    fig.suptitle(title_suffix, fontsize=15, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.90, wspace=0.40, hspace=0.36)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--services", nargs="*", default=SERVICES)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(
        args.input,
        columns=[
            "service_name",
            "access_diagnosis_label",
            "home_street_pattern_class",
            "service_street_pattern_class",
        ],
    )

    home_matrices: list[tuple[str, pd.DataFrame, pd.DataFrame, int]] = []
    service_matrices: list[tuple[str, pd.DataFrame, pd.DataFrame, int]] = []
    for service in args.services:
        counts, shares, excluded_unknown = _service_matrix(df, service, pattern_column="home_street_pattern_class")
        if not shares.empty:
            home_matrices.append((service, counts, shares, excluded_unknown))
            base = args.out_root / f"{service}_home_street_pattern_label_share"
            counts.to_csv(base.with_name(base.name + "_counts.csv"))
            shares.to_csv(base.with_name(base.name + "_shares.csv"))
            _render_service_heatmap(
                service=service,
                counts=counts,
                shares=shares,
                excluded_unknown=excluded_unknown,
                out_path=base.with_name(base.name + "_heatmap.png"),
                title_suffix="label share by home street pattern",
                colorbar_label="share within home street pattern",
            )

        counts, shares, excluded_unknown = _service_matrix(df, service, pattern_column="service_street_pattern_class")
        if not shares.empty:
            service_matrices.append((service, counts, shares, excluded_unknown))
            base = args.out_root / f"{service}_service_street_pattern_label_share"
            counts.to_csv(base.with_name(base.name + "_counts.csv"))
            shares.to_csv(base.with_name(base.name + "_shares.csv"))
            _render_service_heatmap(
                service=service,
                counts=counts,
                shares=shares,
                excluded_unknown=excluded_unknown,
                out_path=base.with_name(base.name + "_heatmap.png"),
                title_suffix="label share by service street pattern",
                colorbar_label="share within service street pattern",
            )

        counts, shares, excluded_unknown = _pair_matrix(df, service)
        if not shares.empty:
            base = args.out_root / f"{service}_home_x_service_street_pattern_label_share"
            counts.to_csv(base.with_name(base.name + "_counts.csv"))
            shares.to_csv(base.with_name(base.name + "_shares.csv"))
            _render_service_heatmap(
                service=service,
                counts=counts,
                shares=shares,
                excluded_unknown=excluded_unknown,
                out_path=base.with_name(base.name + "_heatmap.png"),
                title_suffix="label share by home x service street pattern",
                colorbar_label="share within home x service street pattern",
            )

    _render_combined_service_heatmaps(
        home_matrices,
        args.out_root / "all_services_home_street_pattern_label_share_heatmaps.png",
        title_suffix="Label share by home street pattern across services",
        colorbar_label="share within home street pattern",
    )
    _render_combined_service_heatmaps(
        service_matrices,
        args.out_root / "all_services_service_street_pattern_label_share_heatmaps.png",
        title_suffix="Label share by service street pattern across services",
        colorbar_label="share within service street pattern",
    )


if __name__ == "__main__":
    main()

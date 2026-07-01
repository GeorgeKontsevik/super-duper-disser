#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TARGET_SERVICES = ("school", "polyclinic")
TARGET_LABELS = (
    "ok_walk",
    "ok_pt_only",
    "failed_access_gt_threshold",
    "failed_egress_gt_threshold",
    "failed_access_egress_sum_gt_threshold",
    "failed_in_vehicle_gt_threshold",
    "failed_transfer_gt_threshold",
    "failed_multiple_components_gt_threshold",
    "failed_total_gt_threshold_no_single_component_gt_threshold",
    "failed_no_pt_path",
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _read_summary(path: Path) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            city = row["city"]
            service = row["service_name"]
            label = row["access_diagnosis_label"]
            count = int(float(row["count"]))
            out[(city, service)][label] = count
    return out


def _iter_city_dirs(root: Path) -> Iterable[Path]:
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p


def _write_delta_csv(baseline_root: Path, heat_root: Path, out_csv: Path) -> None:
    rows: list[dict[str, object]] = []
    for heat_city_dir in _iter_city_dirs(heat_root):
        city = heat_city_dir.name
        baseline_csv = baseline_root / city / "home_to_service_access_diagnostics_summary.csv"
        heat_csv = heat_city_dir / "home_to_service_access_diagnostics_summary.csv"
        if not baseline_csv.exists() or not heat_csv.exists():
            continue
        baseline = _read_summary(baseline_csv)
        heat = _read_summary(heat_csv)
        for service in TARGET_SERVICES:
            base_counts = baseline.get((city, service), {})
            heat_counts = heat.get((city, service), {})
            total = max(sum(base_counts.values()), sum(heat_counts.values()), 1)
            row: dict[str, object] = {
                "city": city,
                "service_name": service,
                "homes_total": total,
            }
            for label in TARGET_LABELS:
                base_val = base_counts.get(label, 0)
                heat_val = heat_counts.get(label, 0)
                row[f"{label}_baseline"] = base_val
                row[f"{label}_heat"] = heat_val
                row[f"{label}_delta"] = heat_val - base_val
                row[f"{label}_delta_pp"] = ((heat_val - base_val) / total) * 100.0
            rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city",
        "service_name",
        "homes_total",
        *[f"{label}_{suffix}" for label in TARGET_LABELS for suffix in ("baseline", "heat", "delta", "delta_pp")],
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_time_delta_csv(
    baseline_walk_report: Path,
    heat_walk_report: Path,
    baseline_pt_lt_report: Path,
    heat_pt_lt_report: Path,
    baseline_pt_ge_report: Path,
    heat_pt_ge_report: Path,
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def index_by_key(path: Path) -> dict[tuple[str, str], dict[str, str]]:
        return {(r["city"], r["service_name"]): r for r in _read_tsv(path) if r["service_name"] in TARGET_SERVICES}

    base_walk = index_by_key(baseline_walk_report)
    heat_walk = index_by_key(heat_walk_report)
    base_pt_lt = index_by_key(baseline_pt_lt_report)
    heat_pt_lt = index_by_key(heat_pt_lt_report)
    base_pt_ge = index_by_key(baseline_pt_ge_report)
    heat_pt_ge = index_by_key(heat_pt_ge_report)

    keys = sorted(set(base_walk) | set(heat_walk) | set(base_pt_lt) | set(heat_pt_lt) | set(base_pt_ge) | set(heat_pt_ge))
    for city, service in keys:
        row: dict[str, object] = {"city": city, "service_name": service}
        mappings = [
            ("walk", base_walk.get((city, service), {}), heat_walk.get((city, service), {}), ("mean_time_min_reachable", "median_time_min_reachable")),
            ("pt_lt15", base_pt_lt.get((city, service), {}), heat_pt_lt.get((city, service), {}), ("mean_pt_time_min_reachable", "median_pt_time_min_reachable", "mean_access_egress_walk_time_min_reachable", "mean_transport_time_min_reachable")),
            ("pt_ge15", base_pt_ge.get((city, service), {}), heat_pt_ge.get((city, service), {}), ("mean_pt_time_min_reachable", "median_pt_time_min_reachable", "mean_access_egress_walk_time_min_reachable", "mean_transport_time_min_reachable")),
        ]
        for prefix, b, h, metrics in mappings:
            for metric in metrics:
                b_val = float(b.get(metric, "nan")) if b else float("nan")
                h_val = float(h.get(metric, "nan")) if h else float("nan")
                row[f"{prefix}_{metric}_baseline"] = b_val
                row[f"{prefix}_{metric}_heat"] = h_val
                row[f"{prefix}_{metric}_delta"] = h_val - b_val if math.isfinite(b_val) and math.isfinite(h_val) else ""
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["city", "service_name"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>\n'
    )


def _write_gallery_svg(source_dir: Path, out_svg: Path, title: str, columns: int = 4) -> None:
    images = sorted(source_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG files in {source_dir}")
    thumb_w = 520
    thumb_h = 540
    pad = 28
    title_h = 56
    rows = math.ceil(len(images) / columns)
    width = columns * thumb_w + (columns + 1) * pad
    height = title_h + rows * thumb_h + (rows + 1) * pad
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_svg_header(width, height)]
    chunks.append(
        f'<text x="{pad}" y="38" font-family="Helvetica,Arial,sans-serif" '
        f'font-size="28" font-weight="700" fill="#0f172a">{title}</text>\n'
    )
    for idx, img in enumerate(images):
        r = idx // columns
        c = idx % columns
        x = pad + c * thumb_w
        y = title_h + pad + r * thumb_h
        chunks.append(f'<image href="{img.resolve().as_uri()}" x="{x}" y="{y}" width="{thumb_w - pad}" height="{thumb_h - pad}" preserveAspectRatio="xMidYMid meet"/>\n')
        chunks.append(
            f'<text x="{x + 8}" y="{y + 24}" font-family="Helvetica,Arial,sans-serif" '
            f'font-size="18" font-weight="600" fill="#111827">{img.stem.replace("_home_to_service_access_diagnostics", "")}</text>\n'
        )
    chunks.append("</svg>\n")
    out_svg.write_text("".join(chunks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--heat-root", required=True)
    parser.add_argument("--baseline-walk-report", required=True)
    parser.add_argument("--heat-walk-report", required=True)
    parser.add_argument("--baseline-pt-lt-report", required=True)
    parser.add_argument("--heat-pt-lt-report", required=True)
    parser.add_argument("--baseline-pt-ge-report", required=True)
    parser.add_argument("--heat-pt-ge-report", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    heat_root = Path(args.heat_root)
    out_dir = Path(args.out_dir)

    _write_delta_csv(
        baseline_root=baseline_root,
        heat_root=heat_root,
        out_csv=out_dir / "service_access_heat_vs_baseline_school_polyclinic.csv",
    )
    _write_time_delta_csv(
        baseline_walk_report=Path(args.baseline_walk_report),
        heat_walk_report=Path(args.heat_walk_report),
        baseline_pt_lt_report=Path(args.baseline_pt_lt_report),
        heat_pt_lt_report=Path(args.heat_pt_lt_report),
        baseline_pt_ge_report=Path(args.baseline_pt_ge_report),
        heat_pt_ge_report=Path(args.heat_pt_ge_report),
        out_csv=out_dir / "service_access_time_heat_vs_baseline_school_polyclinic.csv",
    )
    _write_gallery_svg(
        source_dir=baseline_root / "maps",
        out_svg=out_dir / "gallery_baseline.svg",
        title="Baseline service accessibility diagnostics",
    )
    _write_gallery_svg(
        source_dir=heat_root / "maps",
        out_svg=out_dir / "gallery_heat.svg",
        title="Heat-on-walk-edges service accessibility diagnostics",
    )


if __name__ == "__main__":
    main()

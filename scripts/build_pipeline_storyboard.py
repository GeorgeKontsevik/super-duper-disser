#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOINT_INPUT_ROOT = (
    ROOT / "aggregated_spatial_pipeline" / "outputs" / "active_19_good_cities_20260412" / "joint_inputs"
)
DEFAULT_ROUTE_PATTERN_CROSS_CITY_PREVIEW_ROOT = (
    ROOT
    / "aggregated_spatial_pipeline"
    / "outputs"
    / "experiments_active19_20260412"
    / "route_pattern_street_pattern"
    / "_cross_city"
    / "preview_png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build slide-friendly HTML storyboards for aggregated_spatial_pipeline city bundles "
            "using existing manifests and preview PNGs."
        )
    )
    parser.add_argument(
        "--joint-input-root",
        type=Path,
        default=DEFAULT_JOINT_INPUT_ROOT,
        help="Root with city bundle directories.",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["bergen_norway"],
        help="City slugs to render, or 'all'.",
    )
    parser.add_argument(
        "--output-dirname",
        default="pipeline_storyboard",
        help="Directory name created inside each city bundle for storyboard outputs.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_city_dirs(root: Path, requested: list[str]) -> list[Path]:
    if requested == ["all"]:
        return sorted(path for path in root.iterdir() if path.is_dir())
    return [root / slug for slug in requested]


def _rel(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")
    except Exception:
        return path.resolve().as_posix()


def _existing(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.exists() else None


def _png(city_dir: Path, name: str) -> Path | None:
    return _existing(city_dir / "preview_png" / "all_together" / name)


def _cross_city_png(name: str) -> Path | None:
    return _existing(DEFAULT_ROUTE_PATTERN_CROSS_CITY_PREVIEW_ROOT / name)


def _metric_pill(label: str, value: object) -> str:
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    return f'<span class="pill"><span class="pill-label">{safe_label}</span>{safe_value}</span>'


def _image_card(title: str, path: Path | None, base: Path) -> str:
    if path is None:
        return (
            '<div class="image-card missing">'
            f"<div class=\"image-head\">{html.escape(title)}</div>"
            '<div class="image-missing">missing preview</div>'
            "</div>"
        )
    src = html.escape(_rel(path, base) or path.name)
    label = html.escape(title)
    return (
        '<div class="image-card">'
        f'<div class="image-head">{label}</div>'
        f'<img src="{src}" alt="{label}">'
        "</div>"
    )


def _slide(title: str, kicker: str, body: str) -> str:
    return (
        '<section class="slide">'
        f'<div class="slide-kicker">{html.escape(kicker)}</div>'
        f'<h2>{html.escape(title)}</h2>'
        f"{body}"
        "</section>"
    )


def _stage_block(name: str, command: str, params: list[tuple[str, object]], images: list[str]) -> str:
    pills = "".join(_metric_pill(label, value) for label, value in params if value not in (None, "", [], {}))
    gallery = '<div class="image-grid">' + "".join(images) + "</div>"
    return (
        '<article class="stage-block">'
        f'<div class="stage-name">{html.escape(name)}</div>'
        f'<div class="stage-command">{html.escape(command)}</div>'
        f'<div class="pill-row">{pills}</div>'
        f"{gallery}"
        "</article>"
    )


def _chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)] or [[]]


def _stage_slides(
    *,
    step_title: str,
    step_kicker: str,
    stage_name: str,
    command: str,
    params: list[tuple[str, object]],
    images: list[str],
    images_per_slide: int = 2,
) -> list[str]:
    chunks = _chunked(images, images_per_slide)
    total_chunks = len(chunks)
    slides: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        suffix = f" (part {idx}/{total_chunks})" if total_chunks > 1 else ""
        body = _stage_block(
            name=f"{stage_name}{suffix}",
            command=command,
            params=params,
            images=chunk,
        )
        slides.append(_slide(title=f"{step_title}{suffix}", kicker=step_kicker, body=body))
    return slides


def _cover_slide(city_dir: Path, prep_manifest: dict | None, street_summary: dict | None, asset_base: Path) -> str:
    title = city_dir.name.replace("_", " ")
    pills: list[str] = []
    if street_summary is not None:
        pills.append(_metric_pill("street cells", street_summary.get("num_predictions")))
        pills.append(_metric_pill("grid step", street_summary.get("grid_step")))
        pills.append(_metric_pill("buffer m", street_summary.get("buffer_m")))
    if prep_manifest is not None:
        accessibility = prep_manifest.get("accessibility_selection", {})
        pills.append(_metric_pill("active blocks", accessibility.get("blocks_included")))
        pills.append(_metric_pill("services", ", ".join(prep_manifest.get("services", []))))
    body = (
        '<div class="cover-grid">'
        '<div class="cover-copy">'
        f'<h1>{html.escape(title.title())}</h1>'
        '<p>Pipeline storyboard: what was available, which step ran, with which parameters, and which preview came out.</p>'
        f'<div class="pill-row">{"".join(pills)}</div>'
        "</div>"
        '<div class="cover-preview">'
        f'{_image_card("prepared blocks + street grid", _png(city_dir, "overview_prepared_blocks_and_street_grid.png"), asset_base)}'
        "</div>"
        "</div>"
    )
    return _slide(title=f"{title.title()} Pipeline", kicker="storyboard", body=body)


def _raw_and_pattern_slide(city_dir: Path, street_summary: dict | None, asset_base: Path) -> str:
    params: list[tuple[str, object]] = []
    if street_summary is not None:
        params.extend(
            [
                ("place", street_summary.get("place")),
                ("road source", street_summary.get("road_source")),
                ("grid step", street_summary.get("grid_step")),
                ("buffer m", street_summary.get("buffer_m")),
                ("predictions", street_summary.get("num_predictions")),
            ]
        )
    block = _stage_block(
        name="Raw -> blocks -> street pattern",
        command="run_joint -> run_pipeline3_street_pattern_to_quarters",
        params=params,
        images=[
            _image_card("raw OSM overview", _png(city_dir, "overview_raw_osm_layers.png"), asset_base),
            _image_card("prepared blocks + street grid", _png(city_dir, "overview_prepared_blocks_and_street_grid.png"), asset_base),
            _image_card("street pattern top1", _png(city_dir, "street_pattern_top1.png"), asset_base),
            _image_card("street pattern multivariate", _png(city_dir, "street_pattern_multivariate.png"), asset_base),
        ],
    )
    return _slide(title="Spatial Base", kicker="input territory and morphology", body=block)


def _pipeline2_slide(city_dir: Path, prep_manifest: dict | None, asset_base: Path) -> str:
    if prep_manifest is None:
        return _slide(
            title="Pipeline 2",
            kicker="services and accessibility",
            body='<div class="empty-note">manifest_prepare_solver_inputs.json is missing.</div>',
        )
    accessibility = prep_manifest.get("accessibility_selection", {})
    block = _stage_block(
        name="Services -> accessibility -> solver inputs",
        command="run_pipeline2_prepare_solver_inputs",
        params=[
            ("services", ", ".join(prep_manifest.get("services", []))),
            ("selection", accessibility.get("selection_policy")),
            ("blocks total", accessibility.get("blocks_total_before_filter")),
            ("active blocks", accessibility.get("blocks_included")),
            ("placement exact", prep_manifest.get("placement_exact_enabled")),
        ],
        images=[
            _image_card("services raw", _png(city_dir, "29_services_raw_all_categories.png"), asset_base),
            _image_card("mean travel time", _png(city_dir, "30_accessibility_mean_time_map.png"), asset_base),
            _image_card("accessibility selection", _png(city_dir, "accessibility_block_selection_status.png"), asset_base),
            _image_card("polyclinic unmet demand", _png(city_dir, "32_lp_polyclinic_provision_unmet.png"), asset_base),
        ],
    )
    service_cards = []
    solver_outputs = prep_manifest.get("solver_outputs", {})
    for service in ("hospital", "polyclinic", "school", "kindergarten"):
        item = solver_outputs.get(service)
        if not isinstance(item, dict):
            continue
        img_selection = _existing(Path(item["selection_preview_png"])) if item.get("selection_preview_png") else None
        img_lp = _existing(Path(item["lp_preview_png"])) if item.get("lp_preview_png") else None
        service_cards.append(
            '<div class="mini-service">'
            f"<h3>{html.escape(service)}</h3>"
            f'<div class="pill-row">{_metric_pill("blocks", item.get("blocks_count"))}</div>'
            '<div class="mini-grid">'
            f"{_image_card('selection', img_selection, asset_base)}"
            f"{_image_card('result', img_lp, asset_base)}"
            "</div>"
            "</div>"
        )
    body = block + '<div class="service-grid">' + "".join(service_cards) + "</div>"
    return _slide(title="Accessibility And Service Gaps", kicker="pipeline_2 baseline", body=body)


def _routing_slide(city_dir: Path, access_manifest: dict | None, route_summary: dict | None, asset_base: Path) -> str:
    if access_manifest is None and route_summary is None:
        return _slide(
            title="Target OD And PT Update",
            kicker="accessibility-first and route generation",
            body='<div class="empty-note">No accessibility-first or route-generator summaries found.</div>',
        )
    route_generation = (access_manifest or {}).get("route_generation", route_summary or {})
    recomputed = route_generation.get("recomputed_accessibility", {})
    comparison = recomputed.get("comparison_previews", {})
    route_files = route_generation.get("files", {})
    block = _stage_block(
        name="Gap-driven PT route generation",
        command="run_pipeline2_accessibility_first -> run_route_generator_external",
        params=[
            ("services", ", ".join((access_manifest or {}).get("services", []))),
            ("modality", route_generation.get("modality")),
            ("n routes", route_generation.get("route_count") or (access_manifest or {}).get("connectpt_n_routes")),
            ("target OD source", (access_manifest or {}).get("service_target_od", {}).get("source_label")),
            ("demand sum", route_generation.get("demand_sum")),
            ("unserved %", route_generation.get("unserved_demand_pct")),
        ],
        images=[
            _image_card("route with network", _existing(Path(route_files["shared_route_with_existing_preview"])) if route_files.get("shared_route_with_existing_preview") else None, asset_base),
            _image_card("generated route only", _existing(Path(route_files["shared_route_generated_only_preview"])) if route_files.get("shared_route_generated_only_preview") else None, asset_base),
            _image_card("before accessibility", _existing(Path(comparison["before_preview_path"])) if comparison.get("before_preview_path") else None, asset_base),
            _image_card("after accessibility", _existing(Path(comparison["after_preview_path"])) if comparison.get("after_preview_path") else None, asset_base),
            _image_card("delta", _existing(Path(comparison["delta_preview_path"])) if comparison.get("delta_preview_path") else None, asset_base),
        ],
    )
    return _slide(title="Targeted PT Intervention", kicker="route generation from service gaps", body=block)


def _overlay_slide(city_dir: Path, pt_manifest: dict | None, asset_base: Path) -> str:
    if pt_manifest is None:
        return _slide(
            title="PT x Street Pattern",
            kicker="dependency audit",
            body='<div class="empty-note">pt_street_pattern_dependency/manifest.json is missing.</div>',
        )
    counts = pt_manifest.get("counts", {})
    files = pt_manifest.get("files", {})
    overlay_shared = (
        _existing(Path(files["overlay_map_shared"])) if files.get("overlay_map_shared") else None
    ) or _png(city_dir, "pt_street_pattern_overlay_map.png")
    shares_shared = (
        _existing(Path(files["modality_shares_shared"])) if files.get("modality_shares_shared") else None
    ) or _png(city_dir, "pt_street_pattern_modality_shares.png")
    block = _stage_block(
        name="PT overlay on street-pattern classes",
        command="run_pt_street_pattern_dependency",
        params=[
            ("pt types", ", ".join(pt_manifest.get("pt_types", []))),
            ("street cells", counts.get("street_pattern_cells")),
            ("overlay segments", counts.get("overlay_segments")),
            ("routes total", counts.get("routes_total")),
        ],
        images=[
            _image_card("overlay map", overlay_shared, asset_base),
            _image_card("bus shares", shares_shared, asset_base),
            _image_card("street pattern top1", _png(city_dir, "street_pattern_top1.png"), asset_base),
        ],
    )
    return _slide(title="PT Meets Morphology", kicker="overlay and class shares", body=block)


def _exact_slide(city_dir: Path, asset_base: Path) -> str | None:
    previews = city_dir / "preview_png" / "all_together"
    exact_candidates = sorted(previews.glob("*exact*placement*.png"))
    if not exact_candidates:
        return None
    images = [_image_card(path.stem.replace("_", " "), path, asset_base) for path in exact_candidates[:6]]
    block = _stage_block(
        name="Exact placement outputs",
        command="solver_flp optimize_placement",
        params=[],
        images=images,
    )
    return _slide(title="Exact Placement", kicker="existing vs new service layout", body=block)


def _flow_slides(
    city_dir: Path,
    prep_manifest: dict | None,
    access_manifest: dict | None,
    route_summary: dict | None,
    pt_manifest: dict | None,
    asset_base: Path,
) -> list[str]:
    solver_services = ", ".join((prep_manifest or {}).get("services", []))
    acc_sel = (prep_manifest or {}).get("accessibility_selection", {})
    route_generation = (access_manifest or {}).get("route_generation", route_summary or {})
    recomputed = route_generation.get("recomputed_accessibility", {})
    comparison = recomputed.get("comparison_previews", {})
    route_files = route_generation.get("files", {})
    counts = (pt_manifest or {}).get("counts", {})
    step_defs = [
        {
            "name": "1) City base + missing building attributes restoration",
            "command": "run_joint -> building restoration (is_living + storey)",
            "params": [
                ("active blocks", acc_sel.get("blocks_included")),
                ("buildings", "restore missing is_living / storey"),
            ],
            "images": [
                _image_card("raw OSM", _png(city_dir, "overview_raw_osm_layers.png"), asset_base),
                _image_card("is_living restored", _png(city_dir, "buildings_is_living_restoration_status.png"), asset_base),
                _image_card("storey restored", _png(city_dir, "buildings_storey_restoration_status.png"), asset_base),
                _image_card("buildings overview", _png(city_dir, "raw_buildings.png"), asset_base),
            ],
        },
        {
            "name": "2) Quarters + morphology (street pattern grid)",
            "command": "run_joint -> run_pipeline3_street_pattern_to_quarters",
            "params": [
                ("street cells", (prep_manifest or {}).get("street_cells")),
                ("morphology", "top-1 + multivariate"),
            ],
            "images": [
                _image_card("prepared blocks + street grid", _png(city_dir, "overview_prepared_blocks_and_street_grid.png"), asset_base),
                _image_card("street pattern top-1", _png(city_dir, "street_pattern_top1.png"), asset_base),
                _image_card("street pattern multivariate", _png(city_dir, "street_pattern_multivariate.png"), asset_base),
                _image_card("blocks (aggregation units)", _png(city_dir, "blocks_clipped.png"), asset_base),
            ],
        },
        {
            "name": "3) Accessibility and provision on intermodal graph",
            "command": "run_pipeline2_prepare_solver_inputs",
            "params": [
                ("services", solver_services),
                ("blocks total", acc_sel.get("blocks_total_before_filter")),
                ("active blocks", acc_sel.get("blocks_included")),
                ("issues", "accessibility gaps + capacity shortages"),
            ],
            "images": [
                _image_card("services baseline", _png(city_dir, "29_services_raw_all_categories.png"), asset_base),
                _image_card("mean travel time", _png(city_dir, "30_accessibility_mean_time_map.png"), asset_base),
                _image_card("unmet demand: accessibility", _png(city_dir, "32_lp_polyclinic_provision_unmet.png"), asset_base),
                _image_card("unmet demand: capacity", _png(city_dir, "31_lp_hospital_provision_unmet.png"), asset_base),
            ],
        },
        {
            "name": "4) Cross-city evidence: these patterns repeat in many cities",
            "command": "README_ACTIVE19_SERVICE_GAPS + cross-city tables",
            "params": [
                ("evidence", "cross-city summary tables"),
            ],
            "images": [
                _image_card(
                    "unmet demand by accessibility (colored)",
                    _cross_city_png("10_unmet_demand_accessibility_by_street_pattern_service_canvas_intermodal_multivariate.png"),
                    asset_base,
                ),
                _image_card(
                    "unmet demand by capacity (colored)",
                    _cross_city_png("11_unmet_demand_capacity_by_street_pattern_service_canvas_intermodal_multivariate.png"),
                    asset_base,
                ),
                _image_card(
                    "route share by street pattern (multivariate)",
                    _cross_city_png("06_city_class_route_length_share_by_modality_canvas_multivariate.png"),
                    asset_base,
                ),
                _image_card(
                    "service locations by street pattern",
                    _cross_city_png("12_service_location_by_street_pattern_service_canvas_multivariate.png"),
                    asset_base,
                ),
            ],
        },
        {
            "name": "5) Joint solution: MCLP services + connectivity recommendation + generated routes",
            "command": "run_pipeline2_accessibility_first -> run_route_generator_external",
            "params": [
                ("modality", route_generation.get("modality")),
                ("routes", route_generation.get("route_count") or (access_manifest or {}).get("connectpt_n_routes")),
                ("unserved %", route_generation.get("unserved_demand_pct")),
                ("optimization", "joint service + route intervention"),
            ],
            "images": [
                _image_card(
                    "generated route with existing network",
                    _existing(Path(route_files["shared_route_with_existing_preview"])) if route_files.get("shared_route_with_existing_preview") else None,
                    asset_base,
                ),
                _image_card(
                    "generated route only",
                    _existing(Path(route_files["shared_route_generated_only_preview"])) if route_files.get("shared_route_generated_only_preview") else None,
                    asset_base,
                ),
                _image_card(
                    "accessibility delta after generated route",
                    _existing(Path(comparison["delta_preview_path"])) if comparison.get("delta_preview_path") else None,
                    asset_base,
                ),
                _image_card("exact service placement status", _png(city_dir, "36_exact_polyclinic_placement_status.png"), asset_base),
            ],
        },
    ]
    slides: list[str] = []
    total = len(step_defs)
    for idx, spec in enumerate(step_defs, start=1):
        slides.extend(
            _stage_slides(
                step_title=f"Pipeline Step {idx}/{total}",
                step_kicker="16:9 step",
                stage_name=str(spec["name"]),
                command=str(spec["command"]),
                params=list(spec["params"]),
                images=list(spec["images"]),
                images_per_slide=2,
            )
        )
    if not slides:
        slides.append(
            _slide(
                title="Pipeline Flow",
                kicker="16:9 step",
                body='<div class="empty-note">No pipeline flow data available.</div>',
            )
        )
    return slides


def _render_storyboard(city_dir: Path, output_dirname: str) -> Path:
    prep_manifest = _load_json(city_dir / "pipeline_2" / "manifest_prepare_solver_inputs.json")
    access_manifest = _load_json(city_dir / "pipeline_2" / "accessibility_first" / "manifest_accessibility_first.json")
    street_summary = _load_json(city_dir / "street_pattern" / city_dir.name / "summary.json")
    pt_manifest = _load_json(city_dir / "pt_street_pattern_dependency" / "manifest.json")
    route_summary = _load_json(city_dir / "connectpt_routes_generator" / "bus" / "summary.json")

    output_dir = city_dir / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    slides = [_cover_slide(city_dir, prep_manifest, street_summary, output_dir)]
    slides.extend(
        _flow_slides(
            city_dir=city_dir,
            prep_manifest=prep_manifest,
            access_manifest=access_manifest,
            route_summary=route_summary,
            pt_manifest=pt_manifest,
            asset_base=output_dir,
        )
    )
    exact_slide = _exact_slide(city_dir, output_dir)
    if exact_slide is not None:
        slides.append(exact_slide)

    title = city_dir.name.replace("_", " ").title()
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} pipeline storyboard</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --panel: #fffdfa;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #ddd4c7;
      --accent: #0f766e;
      --accent-soft: #d8efe9;
      --shadow: 0 18px 36px rgba(36, 30, 18, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.08), transparent 28%),
        linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }}
    main {{
      width: 1600px;
      margin: 0 auto;
      padding: 28px 24px 60px;
    }}
    .slide {{
      width: 1600px;
      height: 900px;
      background: rgba(255, 253, 250, 0.85);
      border: 1px solid rgba(221, 212, 199, 0.8);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px;
      margin-bottom: 28px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      break-after: page;
      overflow: hidden;
    }}
    .slide-kicker {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 12px;
      color: var(--accent);
      font-weight: 700;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 54px; line-height: 1.02; }}
    h2 {{ font-size: 34px; line-height: 1.05; }}
    h3 {{ font-size: 18px; }}
    p {{ font-size: 18px; line-height: 1.45; color: var(--muted); }}
    .cover-grid {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 22px;
      align-items: stretch;
      flex: 1;
    }}
    .cover-copy, .cover-preview, .stage-block, .mini-service {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
    }}
    .cover-copy {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 16px;
    }}
    .stage-block {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .flow-stack {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .flow-arrow {{
      display: grid;
      place-items: center;
      font-size: 44px;
      color: #92400e;
      line-height: 1;
      font-weight: 700;
      user-select: none;
    }}
    .stage-name {{
      font-size: 24px;
      font-weight: 700;
    }}
    .stage-command {{
      font-family: "SF Mono", "Menlo", monospace;
      font-size: 14px;
      color: #7c2d12;
      background: #fff3e8;
      border: 1px solid #f2d5bf;
      border-radius: 999px;
      padding: 8px 12px;
      width: fit-content;
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 999px;
      background: #f8f4ec;
      border: 1px solid var(--line);
      font-size: 13px;
      line-height: 1;
      white-space: nowrap;
    }}
    .pill-label {{
      color: var(--muted);
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.08em;
    }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .image-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: #fffefb;
      display: block;
    }}
    .image-card img {{
      width: 100%;
      height: auto;
      max-height: 420px;
      object-fit: contain;
      object-position: center;
      background: #f2ede2;
      display: block;
    }}
    .image-head {{
      padding: 10px 12px;
      font-size: 13px;
      font-weight: 700;
      color: var(--ink);
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #fffdfa 0%, #f8f4ec 100%);
    }}
    .image-missing {{
      display: grid;
      place-items: center;
      flex: 1;
      color: var(--muted);
      font-size: 14px;
      background: repeating-linear-gradient(
        -45deg,
        #f6f1e8,
        #f6f1e8 10px,
        #fbf8f2 10px,
        #fbf8f2 20px
      );
    }}
    .service-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .mini-service {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .mini-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }}
    .empty-note {{
      display: grid;
      place-items: center;
      min-height: 280px;
      border: 1px dashed var(--line);
      border-radius: 20px;
      color: var(--muted);
      background: rgba(255, 253, 250, 0.7);
    }}
    @media print {{
      body {{ background: white; }}
      main {{ width: auto; margin: 0; padding: 0; }}
      .slide {{ box-shadow: none; margin: 0; border-radius: 0; page-break-after: always; }}
    }}
  </style>
</head>
<body>
  <main>
    {''.join(slides)}
  </main>
</body>
</html>
"""
    output_path = output_dir / "storyboard.html"
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    city_dirs = _discover_city_dirs(args.joint_input_root.resolve(), [str(item) for item in args.cities])
    for city_dir in city_dirs:
        if not city_dir.exists():
            print(f"[skip] missing city bundle: {city_dir}")
            continue
        output_path = _render_storyboard(city_dir.resolve(), args.output_dirname)
        print(f"[ok] storyboard: {output_path}")


if __name__ == "__main__":
    main()

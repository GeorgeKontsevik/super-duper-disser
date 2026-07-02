#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Users/gk/Code/super-duper-disser")
CITY = "debrecen_hungary"
SERVICE = "polyclinic"
OUT_DIR = ROOT / "thermal_access_pilot/outputs/heat_polyclinic_placement_comparison_hottest/debrecen_visuals"

BASE_SUMMARY = ROOT / f"aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/{CITY}/pipeline_2/placement_exact/{SERVICE}/summary_after.json"
HEAT_SUMMARY = ROOT / f"thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/heat_joint_inputs/{CITY}/pipeline_2/placement_exact/{SERVICE}/summary_after.json"
BASE_STATUS = ROOT / f"aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/{CITY}/preview_png/all_together/lp_{SERVICE}_placement_changes.png"
HEAT_STATUS = ROOT / f"thermal_access_pilot/outputs/batch_service_access_hottest_summer2025/heat_joint_inputs/{CITY}/preview_png/all_together/lp_{SERVICE}_placement_changes.png"

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
COLORS = {"baseline": "#7f1734", "heat": "#d9480f"}


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def render_barchart(base: dict, heat: dict, out_path: Path) -> None:
    labels = ["Новые точки", "Добавленная мощность", "Target demand"]
    base_vals = [base["new_count"], base["capacity_added_total"], base["demand_target_total"]]
    heat_vals = [heat["new_count"], heat["capacity_added_total"], heat["demand_target_total"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=220)
    for ax, label, b, h in zip(axes, labels, base_vals, heat_vals, strict=True):
        ax.bar(["baseline", "heat"], [b, h], color=[COLORS["baseline"], COLORS["heat"]], width=0.6)
        ax.set_title(label, fontsize=12)
        ymax = max(b, h) * 1.15 if max(b, h) > 0 else 1
        ax.set_ylim(0, ymax)
        for i, v in enumerate([b, h]):
            ax.text(i, v + ymax * 0.02, f"{int(v):,}".replace(",", " "), ha="center", va="bottom", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Дебрецен, поликлиники — baseline vs heat placement", fontsize=16, y=0.98)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_side_by_side(base_img: Path, heat_img: Path, out_path: Path) -> None:
    base = Image.open(base_img).convert("RGB")
    heat = Image.open(heat_img).convert("RGB")
    title_font = ImageFont.truetype(FONT_PATH, 38)
    label_font = ImageFont.truetype(FONT_PATH, 28)
    pad = 32
    top = 110
    width = base.width + heat.width + pad * 3
    height = max(base.height, heat.height) + top + pad * 2
    canvas = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 28), "Дебрецен, поликлиники — placement status", fill="#111827", font=title_font)
    draw.text((pad, top - 40), "baseline", fill=COLORS["baseline"], font=label_font)
    draw.text((pad * 2 + base.width, top - 40), "heat", fill=COLORS["heat"], font=label_font)
    canvas.paste(base, (pad, top))
    canvas.paste(heat, (pad * 2 + base.width, top))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")


def render_summary_card(base: dict, heat: dict, out_path: Path) -> None:
    title_font = ImageFont.truetype(FONT_PATH, 42)
    big_font = ImageFont.truetype(FONT_PATH, 56)
    label_font = ImageFont.truetype(FONT_PATH, 26)
    small_font = ImageFont.truetype(FONT_PATH, 22)
    img = Image.new("RGB", (1500, 900), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw.text((50, 40), "Дебрецен, поликлиники — эффект жары на placement", fill="#111827", font=title_font)

    cards = [
        ("Новые точки", base["new_count"], heat["new_count"], heat["new_count"] - base["new_count"]),
        ("Добавленная мощность", int(base["capacity_added_total"]), int(heat["capacity_added_total"]), int(heat["capacity_added_total"] - base["capacity_added_total"])),
        ("Target demand", int(base["demand_target_total"]), int(heat["demand_target_total"]), int(heat["demand_target_total"] - base["demand_target_total"])),
    ]
    x_positions = [60, 520, 980]
    for x, (label, b, h, d) in zip(x_positions, cards, strict=True):
        draw.rounded_rectangle((x, 170, x + 400, 620), radius=24, fill="white", outline="#d1d5db", width=2)
        draw.text((x + 24, 210), label, fill="#374151", font=label_font)
        draw.text((x + 24, 310), f"{b:,}".replace(",", " "), fill=COLORS["baseline"], font=big_font)
        draw.text((x + 24, 390), "→", fill="#6b7280", font=big_font)
        draw.text((x + 120, 390), f"{h:,}".replace(",", " "), fill=COLORS["heat"], font=big_font)
        delta_prefix = "+" if d >= 0 else ""
        draw.text((x + 24, 520), f"Δ {delta_prefix}{d:,}".replace(",", " "), fill="#111827", font=label_font)

    notes = [
        f"baseline: new={base['new_count']}, capacity_added={int(base['capacity_added_total'])}, demand_target={int(base['demand_target_total'])}",
        f"heat: new={heat['new_count']}, capacity_added={int(heat['capacity_added_total'])}, demand_target={int(heat['demand_target_total'])}",
        "heat matrix пересчитана на heat graph; placement считался отдельно от baseline.",
    ]
    y = 700
    for line in notes:
        draw.text((60, y), line, fill="#4b5563", font=small_font)
        y += 42
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    base = _load_summary(BASE_SUMMARY)
    heat = _load_summary(HEAT_SUMMARY)
    render_barchart(base, heat, OUT_DIR / "01_debrecen_polyclinic_baseline_vs_heat_bars.png")
    render_side_by_side(BASE_STATUS, HEAT_STATUS, OUT_DIR / "02_debrecen_polyclinic_baseline_vs_heat_status.png")
    render_summary_card(base, heat, OUT_DIR / "03_debrecen_polyclinic_summary_card.png")


if __name__ == "__main__":
    main()

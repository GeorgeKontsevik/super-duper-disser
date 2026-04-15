from __future__ import annotations

CANVAS_BACKGROUND = "#f7f4ee"
CANVAS_BOUNDARY_FILL = "#fcfbf7"
CANVAS_BOUNDARY_EDGE = "#d8d1c7"
CANVAS_FRAME_EDGE = "#ebe5db"
CANVAS_INK = "#1f2937"
CANVAS_MUTED = "#64748b"
CANVAS_GRID = "#d7d7cf"
LEGEND_FACE = "#fffdfa"
LEGEND_EDGE = "#d8d1c7"
WATER_FILL = "#cfe8f7"
WATER_EDGE = "#7bb6d9"
WATER_LINE = "#5aa3cc"
DEFAULT_PREVIEW_DPI = 220
STREET_PATTERN_CLASS_ORDER: tuple[str, ...] = (
    "Irregular Grid",
    "Loops & Lollipops",
    "Regular Grid",
    "Warped Parallel",
    "Broken Grid",
    "Sparse",
    "unknown",
)

PALETTES: dict[str, dict[str, str]] = {
    "services": {
        "hospital": "#b42318",
        "polyclinic": "#0f766e",
        "school": "#4d7c0f",
        "kindergarten": "#c2410c",
    },
    "pt_modes": {
        "bus": "#0f766e",
        "tram": "#0f766e",
        "trolleybus": "#d97706",
        "subway": "#be123c",
        "walk": "#94a3b8",
    },
    "placement_status": {
        "existing": "#78716c",
        "expanded": "#a16207",
        "new": "#c2410c",
        "inactive": "#d6d3d1",
    },
    "street_patterns": {
        "Loops & Lollipops": "#0f766e",
        "Irregular Grid": "#a16207",
        "Regular Grid": "#16a34a",
        "Warped Parallel": "#f97316",
        "Sparse": "#64748b",
        "Broken Grid": "#dc2626",
        "unknown": "#d1d5db",
    },
    "route_generator": {
        "route_1": "#0f766e",
        "route_2": "#4d7c0f",
        "route_3": "#d97706",
        "route_4": "#7c3aed",
        "route_5": "#be123c",
        "route_6": "#854d0e",
        "route_7": "#65a30d",
        "route_8": "#ea580c",
        "start": "#16a34a",
        "end": "#be123c",
    },
}


def get_palette(name: str) -> dict[str, str]:
    return PALETTES.get(str(name), {}).copy()


def palette_color(palette_name: str, key: str, default: str) -> str:
    palette = PALETTES.get(str(palette_name), {})
    return palette.get(str(key), default)


def order_street_pattern_classes(labels: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    values = [str(v) for v in labels if str(v).strip()]
    values_set = set(values)
    ordered = [name for name in STREET_PATTERN_CLASS_ORDER if name in values_set]
    ordered.extend(sorted([name for name in values_set if name not in ordered]))
    return ordered

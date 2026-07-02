from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "itmo-phd-thesis-template-en" / "images" / "ch4"
OUT = IMG / "liberia_yanao_inputs_drivers_slide.png"
TMP = ROOT / "tmp" / "liberia_yanao_inputs_drivers_slide.png"

SOURCES = {
    "liberia_inputs": IMG / "lbr_inputs_panel_matched.png",
    "yanao_inputs": IMG / "arctic" / "yanao_transport_services_panel.png",
    "liberia_driver": IMG / "lbr_precip_weekly_thresholds_ru.png",
    "yanao_driver": IMG / "arctic" / "arctic_temperature_thresholds_yanao_kras.png",
}


def fit(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(box, Image.Resampling.LANCZOS)
    return copy


def pad(image: Image.Image, padding: int = 36) -> Image.Image:
    padded = Image.new("RGB", (image.width + padding * 2, image.height + padding * 2), "white")
    padded.paste(image, (padding, padding))
    return padded


def paste_center(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    resized = fit(image, (w, h))
    px = x + (w - resized.width) // 2
    py = y + (h - resized.height) // 2
    canvas.paste(resized, (px, py))


def main() -> None:
    loaded = {name: Image.open(path).convert("RGB") for name, path in SOURCES.items()}

    width, height = 3840, 2160
    margin = 55
    gap = 70
    top_h = 1260
    bottom_h = 650
    slot_w = (width - margin * 2 - gap) // 2
    top_y = 15
    bottom_y = 1165

    boxes = {
        "liberia_inputs": (margin, top_y, slot_w, top_h),
        "yanao_inputs": (margin + slot_w + gap, top_y, slot_w, top_h),
        "liberia_driver": (margin, bottom_y, slot_w, bottom_h),
        "yanao_driver": (margin + slot_w + gap, bottom_y, slot_w, bottom_h),
    }

    canvas = Image.new("RGB", (width, height), "white")
    for name, box in boxes.items():
        x, y, w, h = box
        paste_center(canvas, pad(loaded[name], padding=20), (x, y, w, h))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    TMP.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, quality=95)
    canvas.save(TMP, quality=95)
    for path in (OUT, TMP):
        assert path.exists() and path.stat().st_size > 100_000, path
        print(f"{path} | {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

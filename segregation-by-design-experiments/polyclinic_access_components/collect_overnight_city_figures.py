"""Collect per-city PNG artifacts from the overnight route batch into one folder."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


DEFAULT_OUT_ROOT = Path(
    "segregation-by-design-experiments/polyclinic_access_components/outputs/"
    "overnight_route_strategy_batch_20260613"
)


def _relative_png_name(city_dir: Path, png_path: Path) -> str:
    rel = png_path.relative_to(city_dir)
    return "__".join(rel.parts)


def collect_once(out_root: Path, figures_dir: Path) -> dict[str, object]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    copied = 0
    for city_dir in sorted(p for p in out_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        city_figures = figures_dir / city_dir.name
        pngs = sorted(city_dir.rglob("*.png"))
        if not pngs:
            rows.append({"city": city_dir.name, "png_count": 0, "figure_dir": str(city_figures)})
            continue
        city_figures.mkdir(parents=True, exist_ok=True)
        for png in pngs:
            target = city_figures / _relative_png_name(city_dir, png)
            if not target.exists() or png.stat().st_mtime > target.stat().st_mtime:
                shutil.copy2(png, target)
                copied += 1
        rows.append({"city": city_dir.name, "png_count": len(pngs), "figure_dir": str(city_figures)})
    manifest = {
        "out_root": str(out_root),
        "figures_dir": str(figures_dir),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "copied_or_updated": copied,
        "cities": rows,
    }
    (figures_dir / "_figures_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    figures_dir = args.figures_dir.resolve() if args.figures_dir else out_root / "_all_city_figures"
    manifest = collect_once(out_root, figures_dir)
    print(
        f"collected {manifest['copied_or_updated']} updated PNGs "
        f"from {len(manifest['cities'])} cities into {figures_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path('/Users/gk/Code/super-duper-disser')
NB_PATH = ROOT / 'corr_transport_street_pattern' / 'pattern_mode_corr.ipynb'
OUT_ROOT = ROOT / 'aggregated_spatial_pipeline' / 'outputs' / 'joint_inputs_pattern_mode_corr_counties_10km'


def _slugify(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')


def _extract_code(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    return '\n\n'.join(
        ''.join(cell.get('source', []))
        for cell in nb.get('cells', [])
        if cell.get('cell_type') == 'code'
    )


def _extract_places_from_county_dict(code: str) -> list[str]:
    matches = re.findall(r'county_osm_dict_cleaned\s*=\s*\{(.*?)\}\s*', code, flags=re.S)
    if not matches:
        return []
    # Notebook may contain both a dict-comprehension assignment and then a literal dump.
    # Prefer the literal one that contains explicit np.int64 values.
    chosen = ""
    for m in matches:
        if "np.int64(" in m:
            chosen = m
            break
    if not chosen:
        chosen = matches[-1]

    body = '{' + chosen + '}'
    # Turn np.int64(123) into 123 so literal_eval works.
    body = re.sub(r'np\.int64\((\d+)\)', r'\1', body)
    data = ast.literal_eval(body)
    return [str(k).strip() for k in data.keys() if str(k).strip()]


def _extract_places_from_geo_places(code: str) -> list[str]:
    m = re.search(r'geo_places\s*=\s*\[(.*?)\]\s*', code, flags=re.S)
    if not m:
        return []
    arr = ast.literal_eval('[' + m.group(1) + ']')
    return [str(x).strip() for x in arr if str(x).strip()]


def load_places(nb_path: Path) -> list[str]:
    code = _extract_code(nb_path)
    places = _extract_places_from_county_dict(code)
    if not places:
        places = _extract_places_from_geo_places(code)
    # keep stable order, remove duplicates
    seen = set()
    uniq = []
    for p in places:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def main() -> None:
    places = load_places(NB_PATH)
    if not places:
        raise RuntimeError(f'No places extracted from {NB_PATH}')

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)
    env['MPLCONFIGDIR'] = '/tmp/mpl-super-duper-disser'

    total = len(places)
    done = 0

    for i, place in enumerate(places, start=1):
        slug = _slugify(place)
        out_dir = OUT_ROOT / slug
        if out_dir.exists():
            print(f'[{i}/{total}] SKIP {place}')
            done += 1
            print(f'progress: {done}/{total}')
            continue

        print(f'[{i}/{total}] RUN  {place}')
        cmd = [
            str(ROOT / '.venv/bin/python'),
            '-m', 'aggregated_spatial_pipeline.pipeline.run_joint',
            '--place', place,
            '--buffer-m', '10000',
            '--street-grid-step', '500',
            '--modalities', 'bus', 'tram', 'trolleybus',
            '--collect-only',
            '--output-dir', str(out_dir),
        ]
        subprocess.run(cmd, env=env, check=True)
        done += 1
        print(f'progress: {done}/{total}')

    print(f'Done -> {OUT_ROOT}')


if __name__ == '__main__':
    main()

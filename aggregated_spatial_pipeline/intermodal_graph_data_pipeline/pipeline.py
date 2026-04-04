from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from aggregated_spatial_pipeline.runtime_paths import intermodal_python
from aggregated_spatial_pipeline.runtime_config import repo_mplconfigdir


def slugify_place(place: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")
    return slug or "place"


def default_intermodal_python(repo_root: Path) -> Path:
    return intermodal_python(repo_root)


def build_intermodal_graph_bundle(
    place: str,
    output_dir: str | Path,
    boundary_path: str | Path,
    *,
    python_executable: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    repo_root_path = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    python_path = Path(python_executable).resolve() if python_executable is not None else default_intermodal_python(repo_root_path)
    if not python_path.exists():
        raise FileNotFoundError(
            f"Intermodal builder python was not found: {python_path}. "
            "Create the dedicated iduedu 1.2.1 environment first."
        )

    script_module = "aggregated_spatial_pipeline.intermodal_graph_data_pipeline.build_bundle_external"
    command = [
        str(python_path),
        "-m",
        script_module,
        "--place",
        str(place),
        "--boundary-path",
        str(Path(boundary_path).resolve()),
        "--output-dir",
        str(output_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root_path)
    )
    env.setdefault("MPLCONFIGDIR", repo_mplconfigdir("mpl-iduedu121", root=repo_root_path))
    subprocess.run(command, check=True, cwd=str(repo_root_path), env=env)

    manifest_path = output_path / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Intermodal graph builder finished, but manifest was not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))

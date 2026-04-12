from __future__ import annotations

import json
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd

from aggregated_spatial_pipeline.runtime_paths import pandana_python, repo_root


def _run_pandana_backend(*, args: list[str], repo_root_path: Path | None = None) -> dict:
    root = repo_root_path or repo_root()
    python_path = pandana_python(root)
    if not python_path.exists():
        raise FileNotFoundError(
            f"Pandana runtime was not found: {python_path}. "
            "Create it first, e.g. `uv venv .venv-pandana` and install pandana dependencies there."
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(root)
    command = [str(python_path), "-m", "aggregated_spatial_pipeline.pandana_backend", *args]
    completed = subprocess.run(
        command,
        check=True,
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    stdout = (completed.stdout or "").strip()
    if not stdout:
        return {}
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return json.loads(candidate)
    return {}


def build_units_matrix_pandana_external(
    *,
    units_path: Path,
    graph_pickle_path: Path,
    output_path: Path,
    weight_key: str,
    repo_root_path: Path | None = None,
) -> pd.DataFrame:
    _run_pandana_backend(
        args=[
            "units-matrix",
            "--units-path",
            str(units_path),
            "--graph-pickle-path",
            str(graph_pickle_path),
            "--output-path",
            str(output_path),
            "--weight-key",
            str(weight_key),
        ],
        repo_root_path=repo_root_path,
    )
    return pd.read_parquet(output_path)


def build_graph_node_matrix_pandana_external(
    *,
    graph_pickle_path: Path,
    output_path: Path,
    weight_key: str,
    repo_root_path: Path | None = None,
) -> pd.DataFrame:
    _run_pandana_backend(
        args=[
            "graph-node-matrix",
            "--graph-pickle-path",
            str(graph_pickle_path),
            "--output-path",
            str(output_path),
            "--weight-key",
            str(weight_key),
        ],
        repo_root_path=repo_root_path,
    )
    return pd.read_parquet(output_path)


def build_pairs_shortest_paths_pandana_external(
    *,
    graph_pickle_path: Path,
    pairs_df: pd.DataFrame,
    weight_key: str,
    repo_root_path: Path | None = None,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="pandana-pairs-", dir="/tmp") as tmp_dir:
        tmp_root = Path(tmp_dir)
        pairs_path = tmp_root / "pairs.pkl"
        output_path = tmp_root / "paths.pkl"
        pairs_df.to_pickle(pairs_path)
        _run_pandana_backend(
            args=[
                "pairs-shortest-paths",
                "--graph-pickle-path",
                str(graph_pickle_path),
                "--pairs-pickle-path",
                str(pairs_path),
                "--output-pickle-path",
                str(output_path),
                "--weight-key",
                str(weight_key),
            ],
            repo_root_path=repo_root_path,
        )
        return pd.read_pickle(output_path)


def stop_complete_then_prune_pandana_external(
    *,
    graph: nx.Graph,
    stop_flag: str = "is_stop",
    weight_attr: str = "mm_len",
    node_x: str = "x",
    node_y: str = "y",
    min_weight: float | None = None,
    max_weight: float | None = None,
    speed_kmh: float | None = None,
    repo_root_path: Path | None = None,
) -> nx.Graph:
    with tempfile.TemporaryDirectory(prefix="pandana-stop-", dir="/tmp") as tmp_dir:
        tmp_root = Path(tmp_dir)
        graph_path = tmp_root / "graph.pkl"
        output_graph_path = tmp_root / "simplified_graph.pkl"
        with graph_path.open("wb") as handle:
            pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _run_pandana_backend(
            args=[
                "stop-complete-then-prune",
                "--graph-pickle-path",
                str(graph_path),
                "--output-graph-path",
                str(output_graph_path),
                "--stop-flag",
                str(stop_flag),
                "--weight-attr",
                str(weight_attr),
                "--node-x",
                str(node_x),
                "--node-y",
                str(node_y),
                *([] if min_weight is None else ["--min-weight", str(float(min_weight))]),
                *([] if max_weight is None else ["--max-weight", str(float(max_weight))]),
                *([] if speed_kmh is None else ["--speed-kmh", str(float(speed_kmh))]),
            ],
            repo_root_path=repo_root_path,
        )
        with output_graph_path.open("rb") as handle:
            return pickle.load(handle)

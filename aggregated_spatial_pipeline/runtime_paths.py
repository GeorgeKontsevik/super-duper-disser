from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def root_python(root: Path | None = None) -> Path:
    root = root or repo_root()
    return _venv_python(root / ".venv")


def blocksnet_python(root: Path | None = None) -> Path:
    root = root or repo_root()
    return _venv_python(root / "blocksnet" / ".venv")


def connectpt_python(root: Path | None = None) -> Path:
    root = root or repo_root()
    return _venv_python(root / "connectpt" / ".venv")


def floor_predictor_python(root: Path | None = None) -> Path:
    root = root or repo_root()
    return _venv_python(root / "floor-predictor" / ".venv")


def street_pattern_python(root: Path | None = None) -> Path:
    root = root or repo_root()
    return _venv_python(root / "segregation-by-design-experiments" / ".venv")


def intermodal_python(root: Path | None = None) -> Path:
    root = root or repo_root()
    sibling_fork = root.parent / "iduedu-fork"
    sibling_python = _venv_python(sibling_fork / ".venv")
    if sibling_python.exists():
        return sibling_python
    legacy_python = _venv_python(root / ".venv-iduedu121")
    return legacy_python

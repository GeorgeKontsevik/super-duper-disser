from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STREET_GRID_STEP_M = 500.0
DEFAULT_BATCH_REGIONS = ("europe", "usa", "australia_oceania", "africa", "asia")
LOG_FORMAT = (
    "<green>{time:DD MMM HH:mm}</green> | "
    "<level>{level: <7}</level> | "
    "<magenta>{extra[tag]}</magenta> "
    "{message}"
)


def repo_cache_dir(*parts: str, root: Path | None = None) -> Path:
    base = (root or ROOT) / ".cache"
    path = base.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_mplconfigdir(name: str, *, root: Path | None = None) -> str:
    return str(repo_cache_dir(name, root=root))


def ensure_repo_mplconfigdir(name: str, *, root: Path | None = None) -> str:
    path = repo_mplconfigdir(name, root=root)
    os.environ.setdefault("MPLCONFIGDIR", path)
    return path


def configure_logger(tag: str, *, level: str = "INFO") -> None:
    logger.remove()
    logger.configure(patcher=lambda record: record["extra"].setdefault("tag", tag))
    logger.add(
        sys.stderr,
        level=level,
        format=LOG_FORMAT,
        colorize=sys.stderr.isatty(),
    )

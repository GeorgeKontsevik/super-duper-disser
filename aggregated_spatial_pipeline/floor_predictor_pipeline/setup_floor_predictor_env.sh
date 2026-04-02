#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${REPO_ROOT}/floor-predictor/.venv"
FLOOR_PREDICTOR_DIR="${FLOOR_PREDICTOR_DIR:-${REPO_ROOT}/floor-predictor}"

if [[ ! -d "${FLOOR_PREDICTOR_DIR}/.git" && ! -f "${FLOOR_PREDICTOR_DIR}/pyproject.toml" ]]; then
  echo "Expected a local floor-predictor checkout at:"
  echo "  ${FLOOR_PREDICTOR_DIR}"
  echo "Clone it first, for example:"
  echo "  git clone https://github.com/GeorgeKontsevik/floor-predictor.git ${FLOOR_PREDICTOR_DIR}"
  exit 1
fi

uv venv "${ENV_DIR}" --python 3.11
uv pip install --python "${ENV_DIR}/bin/python" -e "${FLOOR_PREDICTOR_DIR}"
uv pip install --python "${ENV_DIR}/bin/python" osmnx geopandas pandas scikit-learn shapely

echo "floor-predictor environment is ready:"
echo "  ${ENV_DIR}/bin/python"
echo "Using local checkout:"
echo "  ${FLOOR_PREDICTOR_DIR}"

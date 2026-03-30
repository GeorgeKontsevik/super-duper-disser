#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${REPO_ROOT}/aggregated_spatial_pipeline/.venv-blocksnet-dev"
BLOCKSNET_DIR="${BLOCKSNET_DIR:-${REPO_ROOT}/blocksnet}"

if [[ ! -d "${BLOCKSNET_DIR}/.git" && ! -f "${BLOCKSNET_DIR}/pyproject.toml" ]]; then
  echo "Expected a local BlocksNet checkout at:"
  echo "  ${BLOCKSNET_DIR}"
  echo "Clone it first, for example:"
  echo "  git clone https://github.com/aimclub/blocksnet.git ${BLOCKSNET_DIR}"
  exit 1
fi

uv venv "${ENV_DIR}"
uv pip install --python "${ENV_DIR}/bin/python" -e "${BLOCKSNET_DIR}"
uv pip install --python "${ENV_DIR}/bin/python" osmnx geopandas pandas pyproj shapely

echo "BlocksNet dev environment is ready:"
echo "  ${ENV_DIR}/bin/python"
echo "Using local checkout:"
echo "  ${BLOCKSNET_DIR}"

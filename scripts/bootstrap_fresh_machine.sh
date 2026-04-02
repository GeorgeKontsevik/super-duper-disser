#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-super-duper-disser}"

cd "$ROOT"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Updating git submodules..."
git submodule update --init --recursive

echo "Creating main environment..."
uv python install 3.11
uv venv .venv --python 3.11
uv sync --python .venv/bin/python

echo "Installing local packages into main environment..."
uv pip install --python .venv/bin/python -e ./blocksnet -e ./connectpt -e ./floor-predictor

echo "Installing extra runtime dependencies into main environment..."
uv pip install --python .venv/bin/python \
  osmapi \
  neatnet \
  pygeoops \
  huggingface-hub \
  torch \
  torchvision \
  torch-geometric

echo "Creating dedicated iduedu 1.2.1 environment..."
uv venv .venv-iduedu121 --python 3.11
uv pip install --python .venv-iduedu121/bin/python \
  iduedu==1.2.1 \
  geopandas \
  pyarrow \
  loguru \
  networkx \
  shapely \
  pandas

echo
echo "Bootstrap finished."
echo "Main env: $ROOT/.venv"
echo "Intermodal env: $ROOT/.venv-iduedu121"
echo
echo "Single-city example:"
echo "cd $ROOT && PLACE=\"Saint Petersburg, Russia\" && PYTHONPATH=$ROOT .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_joint --place \"\$PLACE\" --buffer-m 5000 --street-grid-step 500 && PYTHONPATH=$ROOT .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs --place \"\$PLACE\" && PYTHONPATH=$ROOT .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters --place \"\$PLACE\""

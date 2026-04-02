#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-super-duper-disser}"

cd "$ROOT"

detect_pkg_manager() {
  if command -v brew >/dev/null 2>&1; then
    echo "brew"
  elif command -v apt-get >/dev/null 2>&1; then
    echo "apt"
  elif command -v dnf >/dev/null 2>&1; then
    echo "dnf"
  elif command -v yum >/dev/null 2>&1; then
    echo "yum"
  elif command -v pacman >/dev/null 2>&1; then
    echo "pacman"
  elif command -v zypper >/dev/null 2>&1; then
    echo "zypper"
  elif command -v winget >/dev/null 2>&1; then
    echo "winget"
  elif command -v choco >/dev/null 2>&1; then
    echo "choco"
  else
    echo ""
  fi
}

install_homebrew_if_needed() {
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi

  echo "Installing Homebrew..."
  if command -v curl >/dev/null 2>&1; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  elif command -v wget >/dev/null 2>&1; then
    /bin/bash -c "$(wget -qO- https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  elif command -v python3 >/dev/null 2>&1; then
    /bin/bash -c "$(python3 - <<'PY'
import urllib.request
print(urllib.request.urlopen('https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh').read().decode())
PY
)"
  else
    echo "Could not install Homebrew automatically: missing curl, wget and python3"
    exit 1
  fi

  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
}

install_with_pkg_manager() {
  local pkg="$1"
  local manager
  manager="$(detect_pkg_manager)"

  case "$manager" in
    brew)
      brew install "$pkg"
      ;;
    apt)
      sudo apt-get update
      case "$pkg" in
        python@3.11) sudo apt-get install -y python3 python3-venv ;;
        *) sudo apt-get install -y "$pkg" ;;
      esac
      ;;
    dnf)
      case "$pkg" in
        python@3.11) sudo dnf install -y python3 python3-pip ;;
        *) sudo dnf install -y "$pkg" ;;
      esac
      ;;
    yum)
      case "$pkg" in
        python@3.11) sudo yum install -y python3 python3-pip ;;
        *) sudo yum install -y "$pkg" ;;
      esac
      ;;
    pacman)
      case "$pkg" in
        python@3.11) sudo pacman -Sy --noconfirm python ;;
        *) sudo pacman -Sy --noconfirm "$pkg" ;;
      esac
      ;;
    zypper)
      case "$pkg" in
        python@3.11) sudo zypper install -y python311 ;;
        *) sudo zypper install -y "$pkg" ;;
      esac
      ;;
    winget)
      if [[ "$pkg" == "python@3.11" ]]; then
        winget install -e --id Python.Python.3.11
      elif [[ "$pkg" == "curl" ]]; then
        winget install -e --id cURL.cURL
      elif [[ "$pkg" == "git" ]]; then
        winget install -e --id Git.Git
      fi
      ;;
    choco)
      if [[ "$pkg" == "python@3.11" ]]; then
        choco install -y python311
      else
        choco install -y "$pkg"
      fi
      ;;
    "")
      echo "No supported package manager found to install $pkg automatically"
      exit 1
      ;;
  esac
}

ensure_python3() {
  if command -v python3 >/dev/null 2>&1; then
    return 0
  fi
  echo "python3 not found, installing..."
  if [[ -z "$(detect_pkg_manager)" ]]; then
    install_homebrew_if_needed
  fi
  install_with_pkg_manager "python@3.11"
}

ensure_curl() {
  if command -v curl >/dev/null 2>&1; then
    return 0
  fi
  echo "curl not found, installing..."
  if [[ -z "$(detect_pkg_manager)" ]]; then
    if command -v python3 >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; then
      install_homebrew_if_needed
    else
      echo "Could not install curl automatically: missing package manager, python3 and wget"
      exit 1
    fi
  fi
  install_with_pkg_manager "curl"
}

ensure_git() {
  if command -v git >/dev/null 2>&1; then
    return 0
  fi
  echo "git not found, installing..."
  if [[ -z "$(detect_pkg_manager)" ]]; then
    install_homebrew_if_needed
  fi
  install_with_pkg_manager "git"
}

if [[ -x /opt/homebrew/bin/brew ]]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
elif [[ -x /usr/local/bin/brew ]]; then
  eval "$(/usr/local/bin/brew shellenv)"
fi

ensure_python3
ensure_curl
ensure_git

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    python3 - <<'PY' | sh
import urllib.request
print(urllib.request.urlopen("https://astral.sh/uv/install.sh").read().decode())
PY
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Updating git submodules..."
git submodule update --init --recursive

ensure_repo_clone() {
  local repo_url="$1"
  local repo_dir="$2"
  if [[ -d "$repo_dir/.git" || -f "$repo_dir/pyproject.toml" ]]; then
    return 0
  fi
  echo "Cloning missing repository: $repo_url -> $repo_dir"
  git clone "$repo_url" "$repo_dir"
}

create_repo_venv() {
  local repo_dir="$1"
  echo "Creating env: $repo_dir/.venv"
  uv venv "$repo_dir/.venv" --python 3.11
}

echo "Creating main environment..."
uv python install 3.11
uv venv .venv --python 3.11
uv sync --python .venv/bin/python

echo "Installing extra runtime dependencies into main environment..."
uv pip install --python .venv/bin/python \
  osmapi \
  neatnet \
  pygeoops \
  huggingface-hub \
  torch \
  torchvision \
  torch-geometric

echo "Creating dedicated blocksnet environment..."
create_repo_venv "$ROOT/blocksnet"
uv pip install --python "$ROOT/blocksnet/.venv/bin/python" -e "$ROOT/blocksnet"

echo "Creating dedicated connectpt environment..."
create_repo_venv "$ROOT/connectpt"
uv pip install --python "$ROOT/connectpt/.venv/bin/python" -e "$ROOT/connectpt"
uv pip install --python "$ROOT/connectpt/.venv/bin/python" \
  osmnx momepy neatnet pygeoops pandas scipy shapely networkx loguru pyarrow

echo "Creating dedicated floor-predictor environment..."
create_repo_venv "$ROOT/floor-predictor"
uv pip install --python "$ROOT/floor-predictor/.venv/bin/python" -e "$ROOT/floor-predictor"

echo "Creating dedicated street-pattern environment..."
create_repo_venv "$ROOT/segregation-by-design-experiments"
uv pip install --python "$ROOT/segregation-by-design-experiments/.venv/bin/python" \
  geopandas pandas osmnx networkx shapely pyarrow loguru matplotlib \
  torch torchvision torch-geometric huggingface-hub scikit-learn tqdm osmapi momepy

IDUEDU_FORK_DIR="${IDUEDU_FORK_DIR:-$(cd "$ROOT/.." && pwd)/iduedu-fork}"
ensure_repo_clone "https://github.com/GeorgeKontsevik/IduEdu.git" "$IDUEDU_FORK_DIR"
echo "Creating dedicated iduedu-fork environment..."
create_repo_venv "$IDUEDU_FORK_DIR"
uv pip install --python "$IDUEDU_FORK_DIR/.venv/bin/python" -e "$IDUEDU_FORK_DIR"
uv pip install --python "$IDUEDU_FORK_DIR/.venv/bin/python" \
  geopandas pyarrow loguru networkx shapely pandas scipy

echo
echo "Bootstrap finished."
echo "Main env: $ROOT/.venv"
echo "BlocksNet env: $ROOT/blocksnet/.venv"
echo "ConnectPT env: $ROOT/connectpt/.venv"
echo "Floor-predictor env: $ROOT/floor-predictor/.venv"
echo "Street-pattern env: $ROOT/segregation-by-design-experiments/.venv"
echo "Intermodal env: $IDUEDU_FORK_DIR/.venv"
echo
echo "Single-city example:"
echo "cd $ROOT && PLACE=\"Saint Petersburg, Russia\" && PYTHONPATH=$ROOT .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_joint --place \"\$PLACE\" --buffer-m 5000 --street-grid-step 500 && PYTHONPATH=$ROOT .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs --place \"\$PLACE\" && PYTHONPATH=$ROOT .venv/bin/python -m aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters --place \"\$PLACE\""

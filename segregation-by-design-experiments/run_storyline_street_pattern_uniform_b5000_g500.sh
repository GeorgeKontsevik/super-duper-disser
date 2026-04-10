#!/usr/bin/env bash
set -uo pipefail

ROOT="/Users/gk/Code/super-duper-disser"
PYTHON_BIN="$ROOT/.venv/bin/python"
MAP_FILE="$ROOT/segregation-by-design-experiments/storyline_street_pattern_places.tsv"
MPL_DIR="$ROOT/.cache/mpl-street-pattern-b5000-g500"
OUT_ROOT="$ROOT/aggregated_spatial_pipeline/outputs/joint_inputs_storyline_b5000_g500"

BUFFER_M=5000
GRID_STEP=500
TAG="b${BUFFER_M}_g${GRID_STEP}"

mkdir -p "$MPL_DIR"

FAILED=()
SUCCEEDED=()
SKIPPED=()

if [[ ! -f "$MAP_FILE" ]]; then
  echo "City list file not found: $MAP_FILE"
  exit 1
fi

while IFS=$'\t' read -r slug place; do
  [[ -z "${slug:-}" ]] && continue
  [[ -z "${place:-}" ]] && continue

  city_dir="$OUT_ROOT/${slug}/street_pattern_${TAG}"
  summary_path="$city_dir/${slug}_summary.json"
  if [[ -f "$summary_path" ]]; then
    echo "SKIP [$slug] already classified ($TAG)"
    SKIPPED+=("$slug")
    continue
  fi

  mkdir -p "$city_dir"

  echo "=== [$slug] $place | buffer=$BUFFER_M grid=$GRID_STEP ==="
  if PYTHONPATH="$ROOT" MPLCONFIGDIR="$MPL_DIR" \
    "$PYTHON_BIN" "$ROOT/segregation-by-design-experiments/run_street_pattern_city.py" \
      --place "$place" \
      --buffer-m "$BUFFER_M" \
      --grid-step "$GRID_STEP" \
      --road-source auto \
      --map-coloring multivariate \
      --device cpu \
      --output "$summary_path"
  then
    SUCCEEDED+=("$slug")
  else
    echo "FAIL [$slug] $place"
    FAILED+=("$slug")
  fi
done < "$MAP_FILE"

echo "Done: $OUT_ROOT"
echo "Tag: $TAG"
echo "Skipped: ${#SKIPPED[@]}"
echo "Succeeded: ${#SUCCEEDED[@]}"
echo "Failed: ${#FAILED[@]}"

for slug in "${FAILED[@]}"; do
  echo "  FAIL $slug"
done

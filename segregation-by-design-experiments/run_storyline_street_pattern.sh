#!/usr/bin/env bash
set -uo pipefail

ROOT="/Users/gk/Code/super-duper-disser"
PYTHON_BIN="$ROOT/.venv/bin/python"
MAP_FILE="$ROOT/segregation-by-design-experiments/storyline_street_pattern_places.tsv"
MPL_DIR="$ROOT/.cache/mpl-street-pattern"
OUT_ROOT="$ROOT/aggregated_spatial_pipeline/outputs/joint_inputs"

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

  summary_path="$OUT_ROOT/$slug/street_pattern/${slug}_summary.json"
  if [[ -f "$summary_path" ]]; then
    echo "SKIP [$slug] already classified"
    SKIPPED+=("$slug")
    continue
  fi

  echo "=== [$slug] $place ==="
  if PYTHONPATH="$ROOT" MPLCONFIGDIR="$MPL_DIR" \
    "$PYTHON_BIN" "$ROOT/segregation-by-design-experiments/run_street_pattern_city.py" \
      --place "$place" \
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
echo "Skipped: ${#SKIPPED[@]}"
echo "Succeeded: ${#SUCCEEDED[@]}"
echo "Failed: ${#FAILED[@]}"

for slug in "${FAILED[@]}"; do
  echo "  FAIL $slug"
done

#!/usr/bin/env bash
set -u

ROOT="/Users/gk/Code/super-duper-disser"
ROOT_PY="$ROOT/.venv/bin/python"
BLOCKSNET_PY="$ROOT/blocksnet/.venv/bin/python"
export PYTHONPATH="$ROOT"

BUFFER_M=5000
STREET_GRID_STEP=500
CITY_LIST_FILE="$ROOT/cities_small_compare.txt"

FAILED=()
SUCCEEDED=()

cd "$ROOT" || exit 1

if [[ ! -f "$CITY_LIST_FILE" ]]; then
  echo "City list file not found: $CITY_LIST_FILE"
  exit 1
fi

while IFS= read -r PLACE || [[ -n "$PLACE" ]]; do
  [[ -z "$PLACE" ]] && continue
  [[ "$PLACE" =~ ^# ]] && continue

  CITY_STARTED_AT=$(date +%s)

  echo
  echo "============================================================"
  echo "RUNNING: $PLACE"
  echo "buffer=$BUFFER_M m | street_grid_step=$STREET_GRID_STEP m"
  echo "============================================================"

  if "$ROOT_PY" -m aggregated_spatial_pipeline.pipeline.run_joint \
      --place "$PLACE" \
      --buffer-m "$BUFFER_M" \
      --street-grid-step "$STREET_GRID_STEP"
  then
    if "$BLOCKSNET_PY" -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
        --place "$PLACE"
    then
      if "$ROOT_PY" -m aggregated_spatial_pipeline.pipeline.run_pipeline3_street_pattern_to_quarters \
          --place "$PLACE"
      then
        echo "OK: $PLACE"
        SUCCEEDED+=("$PLACE")
      else
      echo "FAILED at pipeline_3: $PLACE"
      FAILED+=("$PLACE :: pipeline_3")
      fi
    else
      echo "FAILED at pipeline_2: $PLACE"
      FAILED+=("$PLACE :: pipeline_2")
    fi
  else
    echo "FAILED at pipeline_1/run_joint: $PLACE"
    echo "  note: if city centre node/place resolution failed, script just skips to the next city"
    FAILED+=("$PLACE :: pipeline_1")
  fi

  CITY_FINISHED_AT=$(date +%s)
  CITY_ELAPSED_S=$((CITY_FINISHED_AT - CITY_STARTED_AT))
  CITY_ELAPSED_H=$((CITY_ELAPSED_S / 3600))
  CITY_ELAPSED_M=$(((CITY_ELAPSED_S % 3600) / 60))
  CITY_ELAPSED_SEC=$((CITY_ELAPSED_S % 60))

  echo
  echo "############################################################"
  echo "### CITY FINISHED: $PLACE"
  printf "### ELAPSED: %02dh %02dm %02ds\n" "$CITY_ELAPSED_H" "$CITY_ELAPSED_M" "$CITY_ELAPSED_SEC"
  echo "############################################################"
done < "$CITY_LIST_FILE"

echo
echo "==================== SUMMARY ===================="
echo "Succeeded: ${#SUCCEEDED[@]}"
for CITY in "${SUCCEEDED[@]}"; do
  echo "  OK   $CITY"
done

echo
echo "Failed: ${#FAILED[@]}"
for CITY in "${FAILED[@]}"; do
  echo "  FAIL $CITY"
done

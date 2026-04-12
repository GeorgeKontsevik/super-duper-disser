#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/gk/Code/super-duper-disser"
PY="$ROOT/.venv/bin/python"
CITY="$ROOT/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs/bergen_norway"
SERVICE="polyclinic"
PLACEMENT_ROOT="placement_exact_genetic"

cd "$ROOT"

echo "[scenario] city=$(basename "$CITY") service=$SERVICE"
echo "[scenario] step 1/2: exact placement with genetic search"

"$PY" -m aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs \
  --joint-input-dir "$CITY" \
  --services "$SERVICE" \
  --placement-exact \
  --placement-genetic \
  --placement-allow-existing-expansion \
  --placement-prefer-existing \
  --placement-capacity-mode fixed_mean

echo "[scenario] step 2/2: ConnectPT with route bounds 6..10 stops"

"$PY" -m aggregated_spatial_pipeline.pipeline.run_pipeline2_accessibility_first \
  --joint-input-dir "$CITY" \
  --services "$SERVICE" \
  --use-placement-outputs \
  --placement-root-name "$PLACEMENT_ROOT" \
  --modality bus \
  --n-routes 1 \
  --min-route-len 6 \
  --max-route-len 10

cat <<'EOF'

[scenario] notes
- Current placement flags approximate this preference order:
  1) prefer existing facilities
  2) allow expanding existing capacities
  3) open new facilities when still needed
- This is not yet a strict lexicographic optimizer with a separate accessibility-first stage inside solver_flp.
EOF

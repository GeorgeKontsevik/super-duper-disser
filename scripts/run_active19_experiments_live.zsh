#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/gk/Code/super-duper-disser"
RUNNER="$ROOT/scripts/run_active19_street_pattern_experiments_20260412.zsh"
PHASE="${1:-service}"          # service | route | ptdep | all
PERM="${SERVICE_PERMUTATIONS:-49}"

if [[ "$PHASE" != "service" && "$PHASE" != "route" && "$PHASE" != "ptdep" && "$PHASE" != "all" ]]; then
  echo "usage: $0 [service|route|ptdep|all]" >&2
  exit 2
fi

LOG_DIR="$ROOT/aggregated_spatial_pipeline/outputs/experiments_active19_20260412/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${PHASE}_$(date +%Y%m%d_%H%M%S).log"

echo "phase=$PHASE"
echo "service_permutations=$PERM"
echo "log=$LOG"
echo

cd "$ROOT"
PYTHONUNBUFFERED=1 RUN_PHASE="$PHASE" SERVICE_PERMUTATIONS="$PERM" \
  zsh "$RUNNER" 2>&1 | tee "$LOG"

echo
echo "done"
echo "log: $LOG"

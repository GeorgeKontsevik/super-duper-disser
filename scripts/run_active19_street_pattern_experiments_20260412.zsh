#!/usr/bin/env zsh
set -u

ROOT="/Users/gk/Code/super-duper-disser"
PY="$ROOT/.venv/bin/python"
JOINT_ROOT="$ROOT/aggregated_spatial_pipeline/outputs/active_19_good_cities_20260412/joint_inputs"
OUT_ROOT="$ROOT/aggregated_spatial_pipeline/outputs/experiments_active19_20260412"
SERVICE_OUT="$OUT_ROOT/service_accessibility_street_pattern"
ROUTE_OUT="$OUT_ROOT/route_pattern_street_pattern"
PT_OUT="$OUT_ROOT/pt_street_pattern_dependency"
REPORT="$OUT_ROOT/_run_report_$(date +%Y%m%d_%H%M%S).tsv"
SERVICE_PERMUTATIONS="${SERVICE_PERMUTATIONS:-49}"
RUN_PHASE="${RUN_PHASE:-all}"  # all | service | route | ptdep

export PYTHONPATH="$ROOT"
export MPLCONFIGDIR="/tmp/mpl-active19-experiments"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$SERVICE_OUT" "$ROUTE_OUT" "$PT_OUT"

if [[ ! -x "$PY" ]]; then
  echo "python not found: $PY" >&2
  exit 1
fi
if [[ ! -d "$JOINT_ROOT" ]]; then
  echo "joint root not found: $JOINT_ROOT" >&2
  exit 1
fi

typeset -a CITIES=(
  bergen_norway
  bologna_italy
  bristol_united_kingdom
  brno_czechia
  coimbra_portugal
  debrecen_hungary
  dresden_germany
  freiburg_im_breisgau_germany
  gothenburg_sweden
  graz_austria
  innsbruck_austria
  krakow_poland
  linz_austria
  lyon_france
  marseille_france
  porto_portugal
  turin_italy
  turku_finland
  zaragoza_spain
)

typeset -a svc_ok svc_fail route_ok route_fail pt_ok pt_fail
svc_ok=(); svc_fail=(); route_ok=(); route_fail=(); pt_ok=(); pt_fail=()

echo "phase\tcity\tstatus\tnote" > "$REPORT"
echo "OUT_ROOT=$OUT_ROOT"
echo "REPORT=$REPORT"
echo "CITIES=${#CITIES[@]}"
echo "SERVICE_PERMUTATIONS=$SERVICE_PERMUTATIONS"
echo "RUN_PHASE=$RUN_PHASE"

if [[ "$RUN_PHASE" != "all" && "$RUN_PHASE" != "service" && "$RUN_PHASE" != "route" && "$RUN_PHASE" != "ptdep" ]]; then
  echo "invalid RUN_PHASE=$RUN_PHASE (expected: all|service|route|ptdep)" >&2
  exit 2
fi

echo
echo "=== Preflight: verify required prepared layers for experiments ==="
for city in "${CITIES[@]}"; do
  d="$JOINT_ROOT/$city/derived_layers"
  q="$d/quarters_clipped.parquet"
  if [[ -f "$q" ]]; then
    printf "preflight\t%s\tok\tquarters_clipped exists\n" "$city" >> "$REPORT"
  else
    echo "[preflight] $city: missing required quarters_clipped.parquet"
    printf "preflight\t%s\tfail\tmissing required quarters_clipped.parquet\n" "$city" >> "$REPORT"
  fi
done

if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "service" ]]; then
  echo
  echo "=== Phase 1/3: service_accessibility_street_pattern (per-city) ==="
  for city in "${CITIES[@]}"; do
    echo "[service] $city"
    if "$PY" "$ROOT/segregation-by-design-experiments/service_accessibility_street_pattern/run_experiments.py" \
        --cities "$city" \
        --joint-input-root "$JOINT_ROOT" \
        --output-root "$SERVICE_OUT" \
        --permutations "$SERVICE_PERMUTATIONS" \
        --local-only \
        --require-ready-data
    then
      svc_ok+=("$city")
      printf "service\t%s\tok\tper-city\n" "$city" >> "$REPORT"
    else
      svc_fail+=("$city")
      printf "service\t%s\tfail\tper-city\n" "$city" >> "$REPORT"
    fi
  done

  if (( ${#svc_ok[@]} >= 2 )); then
    echo
    echo "=== Phase 1b: service_accessibility_street_pattern (cross-city only) ==="
    if "$PY" "$ROOT/segregation-by-design-experiments/service_accessibility_street_pattern/run_experiments.py" \
        --cities "${svc_ok[@]}" \
        --joint-input-root "$JOINT_ROOT" \
        --output-root "$SERVICE_OUT" \
        --permutations "$SERVICE_PERMUTATIONS" \
        --local-only \
        --cross-city-only \
        --require-ready-data
    then
      printf "service\t_cross_city\tok\tcross-city-only on %s cities\n" "${#svc_ok[@]}" >> "$REPORT"
    else
      printf "service\t_cross_city\tfail\tcross-city-only on %s cities\n" "${#svc_ok[@]}" >> "$REPORT"
    fi
  else
    printf "service\t_cross_city\tskip\tneed>=2 succeeded cities (got %s)\n" "${#svc_ok[@]}" >> "$REPORT"
  fi
fi

if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "route" ]]; then
  echo
  echo "=== Phase 2/3: route_pattern_street_pattern (per-city) ==="
  for city in "${CITIES[@]}"; do
    echo "[route] $city"
    if "$PY" "$ROOT/segregation-by-design-experiments/route_pattern_street_pattern/run_experiments.py" \
        --cities "$city" \
        --joint-input-root "$JOINT_ROOT" \
        --output-root "$ROUTE_OUT"
    then
      route_ok+=("$city")
      printf "route\t%s\tok\tper-city\n" "$city" >> "$REPORT"
    else
      route_fail+=("$city")
      printf "route\t%s\tfail\tper-city\n" "$city" >> "$REPORT"
    fi
  done

  if (( ${#route_ok[@]} >= 2 )); then
    echo
    echo "=== Phase 2b: route_pattern_street_pattern (cross-city pooled) ==="
    if "$PY" "$ROOT/segregation-by-design-experiments/route_pattern_street_pattern/run_experiments.py" \
        --cities "${route_ok[@]}" \
        --joint-input-root "$JOINT_ROOT" \
        --output-root "$ROUTE_OUT" \
        --cross-city-only
    then
      printf "route\t_cross_city\tok\tpooled on %s cities\n" "${#route_ok[@]}" >> "$REPORT"
    else
      printf "route\t_cross_city\tfail\tpooled on %s cities\n" "${#route_ok[@]}" >> "$REPORT"
    fi
  else
    printf "route\t_cross_city\tskip\tneed>=2 succeeded cities (got %s)\n" "${#route_ok[@]}" >> "$REPORT"
  fi
fi

if [[ "$RUN_PHASE" == "all" || "$RUN_PHASE" == "ptdep" ]]; then
  echo
  echo "=== Phase 3/3: pt_street_pattern_dependency (per-city) ==="
  for city in "${CITIES[@]}"; do
    echo "[ptdep] $city"
    if "$PY" -m aggregated_spatial_pipeline.pipeline.run_pt_street_pattern_dependency \
        --joint-input-dir "$JOINT_ROOT/$city" \
        --output-dir "$PT_OUT/$city" \
        --street-pattern-cells "$JOINT_ROOT/$city/street_pattern/$city/predicted_cells.geojson" \
        --top-routes 30 \
        --require-ready-data
    then
      pt_ok+=("$city")
      printf "ptdep\t%s\tok\tper-city\n" "$city" >> "$REPORT"
    else
      pt_fail+=("$city")
      printf "ptdep\t%s\tfail\tper-city\n" "$city" >> "$REPORT"
    fi
  done
fi

echo
echo "=== Summary ==="
echo "service: ok=${#svc_ok[@]} fail=${#svc_fail[@]}"
echo "route:   ok=${#route_ok[@]} fail=${#route_fail[@]}"
echo "ptdep:   ok=${#pt_ok[@]} fail=${#pt_fail[@]}"
echo "report:  $REPORT"
echo "out:     $OUT_ROOT"

if (( ${#svc_fail[@]} > 0 )); then
  echo "service failed cities: ${svc_fail[*]}"
fi
if (( ${#route_fail[@]} > 0 )); then
  echo "route failed cities: ${route_fail[*]}"
fi
if (( ${#pt_fail[@]} > 0 )); then
  echo "ptdep failed cities: ${pt_fail[*]}"
fi

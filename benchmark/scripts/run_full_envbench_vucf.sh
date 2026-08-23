
#!/usr/bin/env bash

set -Eeuo pipefail



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SHARD_ROOT="$BENCH_ROOT/workloads/full_shards_9"

RUNNER="$SCRIPT_DIR/run_case_workload.sh"



START_SHARD="${1:-0}"

END_SHARD="${2:-8}"

OUT_OVERRIDE="${3:-}"



[[ "$START_SHARD" =~ ^[0-8]$ ]] || { echo "ERROR: invalid START_SHARD" >&2; exit 2; }

[[ "$END_SHARD" =~ ^[0-8]$ ]] || { echo "ERROR: invalid END_SHARD" >&2; exit 2; }

(( START_SHARD <= END_SHARD )) || { echo "ERROR: START_SHARD > END_SHARD" >&2; exit 2; }



if [[ -n "$OUT_OVERRIDE" ]]; then

  OUT="$OUT_OVERRIDE"

else

  STAMP="$(date +%Y%m%d_%H%M%S)"

  OUT="$BENCH_ROOT/results/full_envbench_vucf_${START_SHARD}_${END_SHARD}_${STAMP}"

fi



mkdir -p "$OUT"



MANIFEST="$OUT/manifest.tsv"

CONFIG="$OUT/campaign_config.txt"



if [[ ! -f "$MANIFEST" ]]; then

  printf 'shard\tcase\tstatus\trun_dir\tsummary\n' > "$MANIFEST"

fi



{

  echo "benchmark=Full Runtime-Eligible EnvBench"

  echo "source=$BENCH_ROOT/workloads/envbench_full_runtime_eligible_max4000.csv"

  echo "shard_root=$SHARD_ROOT"

  echo "start_shard=$START_SHARD"

  echo "end_shard=$END_SHARD"

  echo "concurrency=128"

  echo "cases=V,U,C,F"

  echo "reset_aware_prefix=1"

  echo "max_input_tokens=4000"

  echo "max_model_len=4096"

  echo "selection=whole_trajectory_runtime_eligible"

  echo "pressure_padding=0"

  echo "high_contention_selection=0"

} > "$CONFIG"



export BENCH_RESET_AWARE_PREFIX=1



for i in $(seq "$START_SHARD" "$END_SHARD"); do

  printf -v SHARD "%02d" "$i"

  INPUT="$SHARD_ROOT/shard_${SHARD}.csv"



  [[ -f "$INPUT" ]] || { echo "ERROR: missing $INPUT" >&2; exit 1; }



  echo

  echo "======================================================================"

  echo "FULL ENVBENCH SHARD $SHARD"

  echo "======================================================================"



  for CASE in V U C F; do

    if awk -F '\t' -v s="$SHARD" -v c="$CASE" '$1==s && $2==c && $3=="PASS"{found=1} END{exit !found}' "$MANIFEST"; then

      echo "=== FULL SHARD $SHARD / CASE $CASE: ALREADY PASS, SKIP ==="

      continue

    fi



    echo

    echo "=== FULL SHARD $SHARD / CASE $CASE ==="



    LOG="$OUT/shard_${SHARD}_${CASE}.log"

    LABEL="full_s${SHARD}_${CASE}_c128"



    if "$RUNNER" "$CASE" "$INPUT" 128 "$LABEL" | tee "$LOG"; then

      RUN_DIR="$(grep '^RUN_DIR=' "$LOG" | tail -1 | cut -d= -f2-)"

      SUMMARY="$(grep '^SUMMARY=' "$LOG" | tail -1 | cut -d= -f2-)"

      printf '%s\t%s\tPASS\t%s\t%s\n' "$SHARD" "$CASE" "$RUN_DIR" "$SUMMARY" >> "$MANIFEST"

    else

      RUN_DIR="$(grep '^RUN_DIR=' "$LOG" | tail -1 | cut -d= -f2- || true)"

      SUMMARY="$(grep '^SUMMARY=' "$LOG" | tail -1 | cut -d= -f2- || true)"

      printf '%s\t%s\tFAILED\t%s\t%s\n' "$SHARD" "$CASE" "$RUN_DIR" "$SUMMARY" >> "$MANIFEST"

      echo "FULL_ENVBENCH_VUCF=FAIL shard=$SHARD case=$CASE"

      echo "MANIFEST=$MANIFEST"

      exit 1

    fi

  done



  echo "FULL_SHARD_${SHARD}=PASS"

done



echo

echo "FULL_ENVBENCH_VUCF=PASS"

echo "MANIFEST=$MANIFEST"

echo "CONFIG=$CONFIG"


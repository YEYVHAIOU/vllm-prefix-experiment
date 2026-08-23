#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT="$BENCH_ROOT/workloads/envbench_high_contention_balanced_min3500.csv"
RUNNER="$SCRIPT_DIR/run_case_workload.sh"

[[ -f "$INPUT" ]] || { echo "ERROR: missing $INPUT" >&2; exit 1; }

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$BENCH_ROOT/results/c76_pilot_$STAMP"
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.tsv"
printf 'case\tstatus\trun_dir\tsummary\n' > "$MANIFEST"

for CASE in V U C F; do
  echo "=== C76 PILOT $CASE ==="
  tmp="$OUT/${CASE}.log"
  if "$RUNNER" "$CASE" "$INPUT" 76 "pilot_${CASE}_c76" | tee "$tmp"; then
    run_dir="$(grep '^RUN_DIR=' "$tmp" | tail -1 | cut -d= -f2-)"
    summary="$(grep '^SUMMARY=' "$tmp" | tail -1 | cut -d= -f2-)"
    printf '%s\tPASS\t%s\t%s\n' "$CASE" "$run_dir" "$summary" >> "$MANIFEST"
  else
    run_dir="$(grep '^RUN_DIR=' "$tmp" | tail -1 | cut -d= -f2- || true)"
    summary="$(grep '^SUMMARY=' "$tmp" | tail -1 | cut -d= -f2- || true)"
    printf '%s\tFAILED\t%s\t%s\n' "$CASE" "$run_dir" "$summary" >> "$MANIFEST"
    echo "C76_PILOT=FAIL case=$CASE"
    exit 1
  fi
done

echo "C76_PILOT=PASS"
echo "MANIFEST=$MANIFEST"

#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="$SCRIPT_DIR/run_case_workload.sh"

LEVELS="${CAPACITY_LEVELS:-96 128}"
CASES="${CAPACITY_CASES:-V U C F}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$BENCH_ROOT/results/capacity_vucf_$STAMP"
mkdir -p "$OUT"
MANIFEST="$OUT/manifest.tsv"
printf 'concurrency\tcase\tstatus\trun_dir\tsummary\n' > "$MANIFEST"

for C in $LEVELS; do
  INPUT="$BENCH_ROOT/workloads/envbench_high_contention_balanced_min3500_c${C}.csv"
  [[ -f "$INPUT" ]] || {
    echo "ERROR: missing capacity workload: $INPUT" >&2
    echo "Copy/build the C${C} trajectory workload before running." >&2
    exit 1
  }

  trajectories="$(python - "$INPUT" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1],encoding="utf-8")))
print(len({x["trajectory"] for x in r}))
PY
)"
  [[ "$trajectories" -eq "$C" ]] || {
    echo "ERROR: C${C} workload has trajectories=$trajectories" >&2
    exit 1
  }

  for CASE in $CASES; do
    echo "=== CAPACITY C${C} $CASE ==="
    tmp="$OUT/C${C}_${CASE}.log"
    if "$RUNNER" "$CASE" "$INPUT" "$C" "capacity_${CASE}_c${C}" | tee "$tmp"; then
      run_dir="$(grep '^RUN_DIR=' "$tmp" | tail -1 | cut -d= -f2-)"
      summary="$(grep '^SUMMARY=' "$tmp" | tail -1 | cut -d= -f2-)"
      printf '%s\t%s\tPASS\t%s\t%s\n' "$C" "$CASE" "$run_dir" "$summary" >> "$MANIFEST"
    else
      run_dir="$(grep '^RUN_DIR=' "$tmp" | tail -1 | cut -d= -f2- || true)"
      summary="$(grep '^SUMMARY=' "$tmp" | tail -1 | cut -d= -f2- || true)"
      printf '%s\t%s\tFAILED\t%s\t%s\n' "$C" "$CASE" "$run_dir" "$summary" >> "$MANIFEST"
      echo "CAPACITY_CASE_FAILED=C${C}_${CASE}"
      # A failed case is evidence about capacity; continue with remaining cases.
    fi
  done
done

echo "CAPACITY_CAMPAIGN_COMPLETE=PASS"
echo "MANIFEST=$MANIFEST"

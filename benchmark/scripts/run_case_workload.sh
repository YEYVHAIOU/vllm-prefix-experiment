#!/usr/bin/env bash
set -Eeuo pipefail

CASE="${1:-}"
INPUT="${2:-}"
CONCURRENCY="${3:-}"
LABEL="${4:-}"
RESET_AWARE_OVERRIDE="${BENCH_RESET_AWARE_PREFIX-}"

[[ "$CASE" =~ ^[VUCF]$ ]] || {
  echo "Usage: $0 V|U|C|F INPUT.csv CONCURRENCY LABEL" >&2
  exit 2
}
[[ -n "$INPUT" && -f "$INPUT" ]] || { echo "ERROR: missing input: $INPUT" >&2; exit 2; }
[[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid concurrency" >&2; exit 2; }
[[ -n "$LABEL" ]] || LABEL="${CASE}_c${CONCURRENCY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

set -a
source "$BENCH_ROOT/configs/common.env"
source "$BENCH_ROOT/configs/${CASE}.env"
set +a

if [[ -n "$RESET_AWARE_OVERRIDE" ]]; then
  BENCH_RESET_AWARE_PREFIX="$RESET_AWARE_OVERRIDE"
fi

PY="$VENV_PATH/bin/python"
START="$SCRIPT_DIR/start_case.sh"
STOP="$SCRIPT_DIR/stop_case.sh"
STATE="$BENCH_ROOT/runtime/state"
REPLAY="$BENCH_ROOT/scripts/run_envbench_runtime_httpclient.py"
SUMMARIZER="$BENCH_ROOT/scripts/summarize_case.py"

EVENTS="$("$PY" - "$INPUT" <<'PY'
import csv,sys
print(sum(1 for _ in csv.DictReader(open(sys.argv[1],encoding="utf-8"))))
PY
)"
TRAJECTORIES="$("$PY" - "$INPUT" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1],encoding="utf-8")))
print(len({r["trajectory"] for r in rows}))
PY
)"
EXPECTED=$((EVENTS + TRAJECTORIES))

if [[ "$TRAJECTORIES" -lt "$CONCURRENCY" ]]; then
  echo "ERROR: trajectories=$TRAJECTORIES < requested concurrency=$CONCURRENCY" >&2
  exit 1
fi

cleanup_ssd() {
  if [[ -n "${RUN:-}" && -d "$RUN/ssd_store" ]]; then
    du -sb "$RUN/ssd_store" 2>/dev/null | cut -f1 > "$RUN/ssd_store_bytes.txt" || true
    rm -rf -- "$RUN/ssd_store"
  fi
}

cleanup() {
  "$STOP" >/dev/null 2>&1 || true
  cleanup_ssd
}
trap cleanup EXIT
cleanup

"$START" "$CASE"
RUN="$(cat "$STATE/current_run")"
echo "RUN_DIR=$RUN"

for _ in $(seq 1 240); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
curl -fsS "http://$HOST:$PORT/health" >/dev/null

{
  echo "case=$CASE"
  echo "label=$LABEL"
  echo "input=$INPUT"
  echo "events=$EVENTS"
  echo "trajectories=$TRAJECTORIES"
  echo "expected_requests=$EXPECTED"
  echo "concurrency=$CONCURRENCY"
  echo "duration_scale=$BENCH_DURATION_SCALE"
  echo "max_sleep_seconds=$BENCH_MAX_SLEEP_SECONDS"
  echo "http_mode=$BENCH_HTTP_MODE"
  echo "lane_assignment=$BENCH_LANE_ASSIGNMENT"
  echo "reset_aware_prefix=${BENCH_RESET_AWARE_PREFIX:-0}"
} > "$RUN/workload_config.txt"

RESET_ARGS=()
if [[ "${BENCH_RESET_AWARE_PREFIX:-0}" == "1" ]]; then
  RESET_ARGS+=(--reset-aware-prefix)
elif [[ "${BENCH_RESET_AWARE_PREFIX:-0}" != "0" ]]; then
  echo "BENCH_RESET_AWARE_PREFIX must be 0 or 1" >&2
  exit 2
fi

"$PY" "$REPLAY" \
  --http-connection-mode "$BENCH_HTTP_MODE" \
  --input "$INPUT" \
  --base-url "http://$HOST:$PORT" \
  --model "$MODEL_NAME" \
  --tokenizer "$MODEL_PATH" \
  --run-dir "$RUN" \
  "${RESET_ARGS[@]}" \
  --max-events "$EVENTS" \
  --min-input-tokens 0 \
  --max-input-tokens 4000 \
  --max-output-tokens "$BENCH_MAX_OUTPUT_TOKENS" \
  --duration-scale "$BENCH_DURATION_SCALE" \
  --max-sleep-seconds "$BENCH_MAX_SLEEP_SECONDS" \
  --metrics-interval "$BENCH_METRICS_INTERVAL" \
  --request-timeout "$BENCH_REQUEST_TIMEOUT" \
  --concurrency "$CONCURRENCY" \
  --lane-assignment "$BENCH_LANE_ASSIGNMENT" \
  --log-flush-wait "$BENCH_LOG_FLUSH_WAIT" \
  --experiment-label "$LABEL" \
  2>&1 | tee "$RUN/client.log"

SUMMARY="$("$PY" - "$RUN" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
xs=list(root.rglob("summary.json"))
if not xs:
    raise SystemExit(1)
print(max(xs,key=lambda p:p.stat().st_mtime))
PY
)"
printf '%s\n' "$SUMMARY" > "$RUN/summary_path.txt"

P="$SUMMARY" EXPECTED="$EXPECTED" "$PY" - <<'PY'
import json,os,sys
d=json.load(open(os.environ["P"],encoding="utf-8"))
req=d["requests"]
expected=int(os.environ["EXPECTED"])
ok=(req["total"]==expected and req["success"]==expected and req["failed"]==0)
print(f"REQUESTS={req['total']} SUCCESS={req['success']} FAILED={req['failed']} EXPECTED={expected}")
print("WORKLOAD_RESULT="+("PASS" if ok else "FAIL"))
raise SystemExit(0 if ok else 1)
PY

if grep -Ei 'Traceback|ImportError|GLIBCXX|EngineCore failed|failed to start|CUDA out of memory|OutOfMemoryError' "$RUN/server.log" > "$RUN/runtime_errors.txt"; then
  echo "RUNTIME_ERRORS=FOUND"
  exit 1
else
  : > "$RUN/runtime_errors.txt"
  echo "RUNTIME_ERRORS=NONE"
fi

"$STOP"
cleanup_ssd
"$PY" "$SUMMARIZER" "$RUN" --summary "$SUMMARY"
trap - EXIT
echo "CASE_RUN=PASS"
echo "CASE=$CASE"
echo "CONCURRENCY=$CONCURRENCY"
echo "RUN_DIR=$RUN"
echo "SUMMARY=$SUMMARY"

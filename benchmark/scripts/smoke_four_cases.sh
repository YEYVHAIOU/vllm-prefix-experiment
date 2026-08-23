#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"

set -a
source "$BENCH_ROOT/configs/common.env"
set +a

PY="$VENV_PATH/bin/python"
START="$SCRIPT_DIR/start_case.sh"
STOP="$SCRIPT_DIR/stop_case.sh"
STATE="$BENCH_ROOT/runtime/state"
OUT="$BENCH_ROOT/results/smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf 'case\thealth\thttp\tresponse\tmetrics\tconfig\terrors\tstop\tport\tstatus\trun_dir\n' > "$SUMMARY"

cleanup() {
  "$STOP" >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_health() {
  local url="http://$HOST:$PORT/health"
  for _ in $(seq 1 240); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

port_released() {
  "$PY" - "$HOST" "$PORT" <<'PY'
import socket, sys
s=socket.socket()
s.settimeout(.3)
r=s.connect_ex((sys.argv[1], int(sys.argv[2])))
s.close()
raise SystemExit(0 if r != 0 else 1)
PY
}

for CASE in V U C F; do
  echo "=== SMOKE $CASE ==="
  cleanup
  "$START" "$CASE"
  RUN="$(cat "$STATE/current_run")"

  health=FAIL
  http=FAIL
  response=FAIL
  metrics=FAIL
  config=FAIL
  errors=FAIL
  stop=FAIL
  port=FAIL
  status=FAILED

  if ! wait_health; then
    echo "SMOKE_${CASE}_HEALTH=FAIL"
    tail -120 "$RUN/server.log" || true
  else
    health=PASS
    echo "SMOKE_${CASE}_HEALTH=PASS"

    "$PY" - "$RUN/smoke_request.json" "$MODEL_NAME" <<'PY'
import json,sys
p,model=sys.argv[1],sys.argv[2]
with open(p,"w",encoding="utf-8") as f:
    json.dump({
        "model": model,
        "messages":[{"role":"user","content":"Reply briefly with the word OK."}],
        "temperature":0,
        "max_tokens":32,
    },f)
PY

    code="$(curl -sS -o "$RUN/smoke_response.json" -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      --data-binary @"$RUN/smoke_request.json" \
      "http://$HOST:$PORT/v1/chat/completions" || true)"
    if [[ "$code" == "200" ]]; then
      http=PASS
      echo "SMOKE_${CASE}_HTTP=200"
      if P="$RUN/smoke_response.json" "$PY" -c 'import os,json,sys; d=json.load(open(os.environ["P"],encoding="utf-8")); c=d.get("choices",[]); m=(c[0].get("message",{}) if c else {}); sys.exit(0 if c and m.get("content") is not None else 1)'; then
        response=PASS
        echo "SMOKE_${CASE}_RESPONSE=PASS"
      fi
    else
      echo "SMOKE_${CASE}_HTTP=$code"
    fi

    if curl -fsS "http://$HOST:$PORT/metrics" > "$RUN/smoke_metrics.txt"; then
      if grep -Eq '^http_requests_total\{[^}]*handler="/v1/chat/completions"[^}]*status="2xx"[^}]*\} [1-9][0-9]*(\.[0-9]+)?$' "$RUN/smoke_metrics.txt"; then
        metrics=PASS
        echo "SMOKE_${CASE}_METRICS=PASS"
      fi
    fi

    case "$CASE" in
      V)
        if grep -Fqx 'SCHEDULING_POLICY=fcfs' "$RUN/declared_config.env" \
           && ! grep -q 'UCM offload policy=' "$RUN/server.log"; then
          config=PASS
        fi
        ;;
      U)
        if grep -Fqx 'SCHEDULING_POLICY=fcfs' "$RUN/declared_config.env" \
           && grep -q 'UCM offload policy=eager' "$RUN/server.log" \
           && grep -q 'UCM WHAT policy=full' "$RUN/server.log" \
           && grep -q 'UCM tiered store initialized' "$RUN/server.log"; then
          config=PASS
        fi
        ;;
      C)
        if grep -Fqx 'SCHEDULING_POLICY=continuum' "$RUN/declared_config.env" \
           && ! grep -q 'UCM offload policy=' "$RUN/server.log"; then
          config=PASS
        fi
        ;;
      F)
        if grep -Fqx 'SCHEDULING_POLICY=continuum' "$RUN/declared_config.env" \
           && grep -q 'UCM offload policy=joint' "$RUN/server.log" \
           && grep -q 'UCM joint decision=cost_full' "$RUN/server.log" \
           && grep -q 'UCM WHAT policy=frontier_tail.*retain_blocks=4' "$RUN/server.log" \
           && grep -q 'UCM tiered store initialized' "$RUN/server.log"; then
          config=PASS
        fi
        ;;
    esac
    echo "SMOKE_${CASE}_CONFIG=$config"

    if grep -Ei 'Traceback|ImportError|GLIBCXX|EngineCore failed|failed to start' "$RUN/server.log" > "$RUN/smoke_runtime_errors.txt"; then
      errors=FAIL
      echo "SMOKE_${CASE}_RUNTIME_ERRORS=FOUND"
    else
      errors=NONE
      echo "SMOKE_${CASE}_RUNTIME_ERRORS=NONE"
    fi
  fi

  "$STOP"
  if ! "$SCRIPT_DIR/status_case.sh" >/dev/null 2>&1; then
    stop=PASS
    echo "SMOKE_${CASE}_STOP=PASS"
  fi
  if port_released; then
    port=PASS
    echo "SMOKE_${CASE}_PORT_RELEASE=PASS"
  fi

  if [[ "$health" == PASS && "$http" == PASS && "$response" == PASS \
        && "$metrics" == PASS && "$config" == PASS && "$errors" == NONE \
        && "$stop" == PASS && "$port" == PASS ]]; then
    status=PASS
    echo "SMOKE_${CASE}=PASS"
  else
    echo "SMOKE_${CASE}=FAIL"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$CASE" "$health" "$http" "$response" "$metrics" "$config" "$errors" \
    "$stop" "$port" "$status" "$RUN" >> "$SUMMARY"

  [[ "$status" == PASS ]] || exit 1
done

trap - EXIT
echo "FOUR_CASE_SMOKE=PASS"
echo "SMOKE_SUMMARY=$SUMMARY"

#!/usr/bin/env bash

set -Eeuo pipefail



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPLOY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${FULL_SYSTEM_CONFIG:-$DEPLOY_ROOT/deployment/configs/full_system.env}"

START="$SCRIPT_DIR/start_full.sh"

STOP="$SCRIPT_DIR/stop.sh"

STATUS="$SCRIPT_DIR/status.sh"



[[ -f "$CONFIG" ]] || { echo "SMOKE_CONFIG_MISSING=FAIL"; exit 1; }



set -a

source "$CONFIG"

set +a
VLLM_REPO="${VLLM_REPO:-$DEPLOY_ROOT/../vllm-continuum}"
UCM_REPO="${UCM_REPO:-$DEPLOY_ROOT/../unified-cache-management-continuum}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$DEPLOY_ROOT/deployment/runtime}"



PY="$VENV_PATH/bin/python"

CURL="$(command -v curl || true)"



[[ -x "$PY" ]] || { echo "SMOKE_PYTHON_MISSING=FAIL"; exit 1; }

[[ -n "$CURL" ]] || { echo "SMOKE_CURL_MISSING=FAIL"; exit 1; }



STARTED=0

RUN=""

PID=""



cleanup() {

    if [[ "$STARTED" == "1" ]]; then

        "$STOP" >/dev/null 2>&1 || true

    fi

}

trap cleanup EXIT INT TERM



if "$STATUS" >/dev/null 2>&1; then

    echo "SMOKE_EXISTING_SERVER=FAIL"

    exit 1

fi



"$START"

STARTED=1



RUN="$(cat "$RUNTIME_ROOT/state/current_run")"

PID="$(cat "$RUNTIME_ROOT/state/server.pid")"



HEALTH=0

for _ in $(seq 1 120); do

    if "$CURL" -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then

        HEALTH=1

        break

    fi

    if ! kill -0 "$PID" 2>/dev/null; then

        echo "SMOKE_SERVER_DIED=FAIL"

        tail -120 "$RUN/server.log" || true

        exit 1

    fi

    sleep 1

done



[[ "$HEALTH" == "1" ]] || { echo "SMOKE_HEALTH=FAIL"; tail -120 "$RUN/server.log" || true; exit 1; }

echo "SMOKE_HEALTH=PASS"



tr "\0" "\n" < "/proc/$PID/environ" | grep -E "^(PYTHONPATH|LD_PRELOAD|CUDA_HOME|CONTINUUM_[A-Z0-9_]+|UCM_[A-Z0-9_]+|PYTHONHASHSEED|PYTHONDONTWRITEBYTECODE)=" | sort > "$RUN/runtime_environment_effective.txt" || true



cp "$CONFIG" "$RUN/full_system_config_declared.env"



grep -Fq "$UCM_REPO" "$RUN/runtime_environment_effective.txt"

grep -Fq "$VLLM_REPO" "$RUN/runtime_environment_effective.txt"

grep -q "^CONTINUUM_TTL_CDF_IMPL=optimized$" "$RUN/runtime_environment_effective.txt"

grep -q "^CONTINUUM_HISTORY_THRESHOLD=5$" "$RUN/runtime_environment_effective.txt"

grep -q "^UCM_OFFLOAD_POLICY=joint$" "$RUN/runtime_environment_effective.txt"

grep -q "^UCM_JOINT_DECISION=cost_full$" "$RUN/runtime_environment_effective.txt"

grep -q "^UCM_WHAT_POLICY=frontier_tail$" "$RUN/runtime_environment_effective.txt"

grep -q "^UCM_RETAIN_BLOCKS=4$" "$RUN/runtime_environment_effective.txt"

echo "SMOKE_EFFECTIVE_ENV=PASS"



P="$RUN/smoke_request.json" MODEL="$MODEL_NAME" "$PY" -c 'import os,json,pathlib; pathlib.Path(os.environ["P"]).write_text(json.dumps({"model":os.environ["MODEL"],"messages":[{"role":"user","content":"Reply briefly with: DEPLOYMENT_SMOKE_OK"}],"temperature":0,"max_tokens":32},ensure_ascii=False),encoding="utf-8")'
HTTP_CODE="$("$CURL" -sS -o "$RUN/smoke_response.json" -w "%{http_code}" "http://${HOST}:${PORT}/v1/chat/completions" -H "Content-Type: application/json" --data-binary @"$RUN/smoke_request.json")"



[[ "$HTTP_CODE" == "200" ]] || { echo "SMOKE_HTTP=$HTTP_CODE"; exit 1; }

echo "SMOKE_HTTP=200"



P="$RUN/smoke_response.json" "$PY" -c 'import os,json,sys; d=json.load(open(os.environ["P"],encoding="utf-8")); choices=d.get("choices",[]); content=(choices[0].get("message",{}) if choices else {}).get("content"); sys.exit(0 if choices and content is not None else 1)'

echo "SMOKE_RESPONSE=PASS"



"$CURL" -fsS "http://${HOST}:${PORT}/metrics" > "$RUN/smoke_metrics.txt"

grep -Eq '^http_requests_total\{[^}]*handler="/v1/chat/completions"[^}]*status="2xx"[^}]*\} [1-9][0-9]*(\.[0-9]+)?$' "$RUN/smoke_metrics.txt"

echo "SMOKE_METRICS=PASS"



grep -q "scheduling_policy.*continuum" "$RUN/server.log"

grep -q "UCM offload policy=joint" "$RUN/server.log"

grep -q "UCM WHAT policy=frontier_tail.*retain_blocks=4" "$RUN/server.log"

grep -q "UCM joint decision=cost_full" "$RUN/server.log"

grep -q "UCM tiered store initialized" "$RUN/server.log"

grep -q "Continuum history threshold=5" "$RUN/server.log"

grep -q "Continuum default TTL=2.000000" "$RUN/server.log"

grep -q "Continuum prefill profile scale=1.000" "$RUN/server.log"

grep -q "Set UC::TransferEnable to true" "$RUN/server.log"

grep -q "Set UC::StreamNumber to 2" "$RUN/server.log"

grep -q "Set UC::BufferNumber to 64" "$RUN/server.log"

echo "SMOKE_RUNTIME_CONFIG=PASS"



if grep -Ei "Traceback|ImportError|GLIBCXX|EngineCore failed|failed to start" "$RUN/server.log" > "$RUN/smoke_runtime_errors.txt"; then

    echo "SMOKE_RUNTIME_ERRORS=FAIL"

    cat "$RUN/smoke_runtime_errors.txt"

    exit 1

fi

echo "SMOKE_RUNTIME_ERRORS=NONE"



"$STOP"

STARTED=0



"$STATUS" > "$RUN/post_stop_status.txt" 2>&1 || true

grep -q "STATUS=STOPPED" "$RUN/post_stop_status.txt"

echo "SMOKE_STOP=PASS"



PORT_RESULT="$("$PY" -c 'import socket,sys; s=socket.socket(); s.settimeout(.3); r=s.connect_ex((sys.argv[1],int(sys.argv[2]))); s.close(); print("PASS" if r!=0 else "FAIL")' "$HOST" "$PORT")"

[[ "$PORT_RESULT" == "PASS" ]] || { echo "SMOKE_PORT_RELEASE=FAIL"; exit 1; }

echo "SMOKE_PORT_RELEASE=PASS"



cat > "$RUN/smoke_summary.txt" <<EOF

SMOKE_HEALTH=PASS

SMOKE_EFFECTIVE_ENV=PASS

SMOKE_HTTP=200

SMOKE_RESPONSE=PASS

SMOKE_METRICS=PASS

SMOKE_RUNTIME_CONFIG=PASS

SMOKE_RUNTIME_ERRORS=NONE

SMOKE_STOP=PASS

SMOKE_PORT_RELEASE=PASS

SMOKE_TEST=PASS

EOF



trap - EXIT INT TERM



echo "SMOKE_TEST=PASS"

echo "SMOKE_RUN_DIR=$RUN"


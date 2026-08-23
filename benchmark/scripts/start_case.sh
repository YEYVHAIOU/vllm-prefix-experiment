#!/usr/bin/env bash
set -Eeuo pipefail

CASE="${1:-}"
[[ "$CASE" =~ ^[VUCF]$ ]] || {
  echo "Usage: $0 V|U|C|F" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"

COMMON="$BENCH_ROOT/configs/common.env"
CASE_ENV="$BENCH_ROOT/configs/${CASE}.env"
[[ -f "$COMMON" ]] || { echo "ERROR: missing $COMMON" >&2; exit 1; }
[[ -f "$CASE_ENV" ]] || { echo "ERROR: missing $CASE_ENV" >&2; exit 1; }

set -a
source "$COMMON"
source "$CASE_ENV"
set +a

VLLM_REPO="${VLLM_REPO:-$DEPLOY_ROOT/../vllm-continuum}"
UCM_REPO="${UCM_REPO:-$DEPLOY_ROOT/../unified-cache-management-continuum}"
PY="$VENV_PATH/bin/python"
VLLM_BIN="$VENV_PATH/bin/vllm"

[[ -x "$PY" ]] || { echo "ERROR: missing Python: $PY" >&2; exit 1; }
[[ -x "$VLLM_BIN" ]] || { echo "ERROR: missing vllm executable: $VLLM_BIN" >&2; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "ERROR: missing model: $MODEL_PATH" >&2; exit 1; }
[[ -d "$VLLM_REPO" ]] || { echo "ERROR: missing vLLM source: $VLLM_REPO" >&2; exit 1; }
if [[ "$UCM_ENABLED" == "1" ]]; then
  [[ -d "$UCM_REPO" ]] || { echo "ERROR: missing UCM source: $UCM_REPO" >&2; exit 1; }
fi

export CUDA_HOME LD_PRELOAD
export HF_HUB_OFFLINE TRANSFORMERS_OFFLINE PYTHONDONTWRITEBYTECODE
export PYTHONHASHSEED VLLM_LOGGING_LEVEL

RUNTIME="$BENCH_ROOT/runtime"
STATE="$RUNTIME/state"
mkdir -p "$STATE" "$RUNTIME/runs"

# Never start over an existing benchmark server.
if [[ -f "$STATE/server.pid" ]]; then
  old_pid="$(cat "$STATE/server.pid" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: benchmark server already running PID=$old_pid" >&2
    exit 1
  fi
fi

"$PY" - "$HOST" "$PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(0.3)
r = s.connect_ex((host, port))
s.close()
if r == 0:
    raise SystemExit(f"ERROR: {host}:{port} is already in use")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RUNTIME/runs/${CASE}_${STAMP}"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$STATE/current_run"
printf '%s\n' "$CASE" > "$STATE/current_case"

# Freeze declared benchmark configuration inside the run.
{
  echo "# common.env"
  cat "$COMMON"
  echo
  echo "# ${CASE}.env"
  cat "$CASE_ENV"
} > "$RUN_DIR/declared_config.env"

ARGS=(
  serve "$MODEL_PATH"
  --served-model-name "$MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --scheduling-policy "$SCHEDULING_POLICY"
  --enable-prefix-caching
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --swap-space "$SWAP_SPACE_GIB"
)

if [[ "$UCM_ENABLED" == "1" ]]; then
  SSD_ROOT="$RUN_DIR/ssd_store"
  mkdir -p "$SSD_ROOT"
  DRAM_BYTES=$((UCM_TIER_DRAM_MIB * 1024 * 1024))
  KV_CONFIG="$("$PY" -c 'import json,sys; print(json.dumps({"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"ucm_connectors":[{"ucm_connector_name":"UcmTieredStore","ucm_connector_config":{"dram_max_cache_size":int(sys.argv[1]),"storage_backends":sys.argv[2],"use_direct":sys.argv[3].lower()=="true","stream_number":int(sys.argv[4]),"buffer_number":int(sys.argv[5])}}],"load_only_first_rank":False}},separators=(",",":")))' "$DRAM_BYTES" "$SSD_ROOT" "$UCM_TIER_USE_DIRECT" "$UCM_TIER_STREAM_NUMBER" "$UCM_TIER_BUFFER_NUMBER")"
  printf '%s\n' "$KV_CONFIG" > "$RUN_DIR/kv_transfer_config.json"
  ARGS+=(--kv-transfer-config "$KV_CONFIG")
  RUN_PYTHONPATH="$UCM_REPO:$VLLM_REPO"
else
  RUN_PYTHONPATH="$VLLM_REPO"
fi

echo "Starting benchmark case $CASE"
echo "  case:        $CASE_NAME"
echo "  scheduler:   $SCHEDULING_POLICY"
echo "  UCM:         $UCM_ENABLED"
if [[ "$UCM_ENABLED" == "1" ]]; then
  echo "  WHEN:        $UCM_OFFLOAD_POLICY${UCM_JOINT_DECISION:+/$UCM_JOINT_DECISION}"
  echo "  WHAT:        ${UCM_WHAT_POLICY:-full}${UCM_RETAIN_BLOCKS:+ K=$UCM_RETAIN_BLOCKS}"
  echo "  WHERE:       TieredStore DRAM=${UCM_TIER_DRAM_MIB}MiB"
fi
echo "  GPU util:    $GPU_MEMORY_UTILIZATION"
echo "  eager:       disabled"
echo "  run:         $RUN_DIR"

nohup setsid env \
  PYTHONPATH="$RUN_PYTHONPATH" \
  RUN_OUTPUT_DIR="$RUN_DIR" \
  "$VLLM_BIN" "${ARGS[@]}" \
  > "$RUN_DIR/server.log" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$STATE/server.pid"
printf '%s\n' "$PID" > "$RUN_DIR/server.pid"
echo "PID=$PID"
echo "SERVER_LOG=$RUN_DIR/server.log"

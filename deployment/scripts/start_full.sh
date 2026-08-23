#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPLOY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${FULL_SYSTEM_CONFIG:-$DEPLOY_ROOT/deployment/configs/full_system.env}"

[[ -f "$CONFIG" ]] || { echo "ERROR: missing config: $CONFIG" >&2; exit 1; }

set -a

source "$CONFIG"

set +a

VLLM_REPO="${VLLM_REPO:-$DEPLOY_ROOT/../vllm-continuum}"

UCM_REPO="${UCM_REPO:-$DEPLOY_ROOT/../unified-cache-management-continuum}"

RUNTIME_ROOT="${RUNTIME_ROOT:-$DEPLOY_ROOT/deployment/runtime}"

SSD_ROOT="${SSD_ROOT:-$RUNTIME_ROOT/ssd_store}"

STATE_DIR="$RUNTIME_ROOT/state"

RUN_ROOT="$RUNTIME_ROOT/runs"

PY="$VENV_PATH/bin/python"

VLLM_BIN="$VENV_PATH/bin/vllm"

[[ -x "$PY" ]] || { echo "ERROR: missing python: $PY" >&2; exit 1; }

[[ -x "$VLLM_BIN" ]] || { echo "ERROR: missing vllm executable: $VLLM_BIN" >&2; exit 1; }

[[ -d "$MODEL_PATH" ]] || { echo "ERROR: missing model: $MODEL_PATH" >&2; exit 1; }

[[ -d "$VLLM_REPO" ]] || { echo "ERROR: missing vLLM source: $VLLM_REPO" >&2; exit 1; }

[[ -d "$UCM_REPO" ]] || { echo "ERROR: missing UCM source: $UCM_REPO" >&2; exit 1; }

mkdir -p "$SSD_ROOT" "$STATE_DIR" "$RUN_ROOT"

if "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(.2); r=s.connect_ex((sys.argv[1],int(sys.argv[2]))); s.close(); raise SystemExit(0 if r==0 else 1)" "$HOST" "$PORT"; then echo "ERROR: $HOST:$PORT is already occupied" >&2; exit 1; fi

DRAM_BYTES=$((UCM_TIER_DRAM_MIB * 1024 * 1024))

KV_CONFIG="$("$PY" -c 'import json,sys; print(json.dumps({"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"ucm_connectors":[{"ucm_connector_name":"UcmTieredStore","ucm_connector_config":{"dram_max_cache_size":int(sys.argv[1]),"storage_backends":sys.argv[2],"use_direct":sys.argv[3].lower()=="true","stream_number":int(sys.argv[4]),"buffer_number":int(sys.argv[5])}}],"load_only_first_rank":False}},separators=(",",":")))' "$DRAM_BYTES" "$SSD_ROOT" "$UCM_TIER_USE_DIRECT" "$UCM_TIER_STREAM_NUMBER" "$UCM_TIER_BUFFER_NUMBER")"

STAMP="$(date +%Y%m%d_%H%M%S)"

RUN_DIR="$RUN_ROOT/full_${STAMP}"

mkdir -p "$RUN_DIR"

printf "%s
" "$RUN_DIR" > "$STATE_DIR/current_run"

printf "%s
" "$KV_CONFIG" > "$RUN_DIR/kv_transfer_config.json"

echo "Starting Full System"

echo "  model:       $MODEL_PATH"

echo "  scheduler:   continuum"

echo "  GPU util:    $GPU_MEMORY_UTILIZATION"

echo "  eager:       disabled"

echo "  WHEN:        $UCM_OFFLOAD_POLICY/$UCM_JOINT_DECISION"

echo "  WHAT:        $UCM_WHAT_POLICY K=$UCM_RETAIN_BLOCKS"

echo "  WHERE:       TieredStore DRAM=${UCM_TIER_DRAM_MIB}MiB"

echo "  SSD:         $SSD_ROOT"

echo "  run:         $RUN_DIR"

cd "$VLLM_REPO"

nohup setsid env PYTHONPATH="$UCM_REPO:$VLLM_REPO" RUN_OUTPUT_DIR="$RUN_DIR" "$VLLM_BIN" serve "$MODEL_PATH" --served-model-name "$MODEL_NAME" --host "$HOST" --port "$PORT" --scheduling-policy continuum --enable-prefix-caching --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --swap-space "$SWAP_SPACE_GIB" --kv-transfer-config "$KV_CONFIG" > "$RUN_DIR/server.log" 2>&1 < /dev/null &

PID=$!

printf "%s
" "$PID" > "$STATE_DIR/server.pid"

echo "PID=$PID"

echo "SERVER_LOG=$RUN_DIR/server.log"


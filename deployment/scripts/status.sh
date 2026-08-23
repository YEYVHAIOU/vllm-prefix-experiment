#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPLOY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STATE="$DEPLOY_ROOT/deployment/runtime/state"

if [[ ! -f "$STATE/server.pid" ]]; then echo "STATUS=STOPPED"; exit 1; fi

PID="$(cat "$STATE/server.pid")"

if kill -0 "$PID" 2>/dev/null; then

    echo "STATUS=RUNNING PID=$PID"

    [[ -f "$STATE/current_run" ]] && echo "RUN_DIR=$(cat "$STATE/current_run")"

    exit 0

fi

echo "STATUS=STALE_PID PID=$PID"

exit 1


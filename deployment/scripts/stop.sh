#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPLOY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PID_FILE="$DEPLOY_ROOT/deployment/runtime/state/server.pid"

[[ -f "$PID_FILE" ]] || { echo "SERVER_ALREADY_STOPPED"; exit 0; }

PID="$(cat "$PID_FILE")"

if kill -0 "$PID" 2>/dev/null; then

    echo "Stopping PID=$PID"

    kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true

    for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done

    if kill -0 "$PID" 2>/dev/null; then kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true; fi

fi

rm -f "$PID_FILE"

echo "SERVER_STOPPED"


#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE="$BENCH_ROOT/runtime/state"
PID_FILE="$STATE/server.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "SERVER_STOPPED"
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "SERVER_STOPPED"
  exit 0
fi

echo "Stopping PID=$PID"
kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true

for _ in $(seq 1 40); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "SERVER_STOPPED"
    exit 0
  fi
  sleep 0.5
done

kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
sleep 1
rm -f "$PID_FILE"
echo "SERVER_STOPPED"

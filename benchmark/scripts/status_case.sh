#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE="$BENCH_ROOT/runtime/state"

if [[ ! -f "$STATE/server.pid" ]]; then
  echo "STATUS=STOPPED"
  exit 1
fi

PID="$(cat "$STATE/server.pid" 2>/dev/null || true)"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  echo "STATUS=RUNNING PID=$PID"
  [[ -f "$STATE/current_case" ]] && echo "CASE=$(cat "$STATE/current_case")"
  [[ -f "$STATE/current_run" ]] && echo "RUN_DIR=$(cat "$STATE/current_run")"
  exit 0
fi

echo "STATUS=STOPPED"
exit 1

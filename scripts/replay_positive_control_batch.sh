#!/usr/bin/env bash
# Replay completed single-proxy positive-control images against T1 then T2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROXY="${1:-P2}"
SEED="${2:-42}"
PID=""

cleanup() {
  if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

wait_ready() {
  for _ in $(seq 1 90); do
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "Target server did not become ready" >&2; return 1
}

for target in T1 T2; do
  bash scripts/serve_proxy_selector_target.sh --target "$target" --port 8000 \
    > "outputs/proxy_selector_pilot/logs/${target}_positive_control_batch.log" 2>&1 &
  PID="$!"
  wait_ready
  for split in dev test; do
    PYTHONPATH=src .venv-proxy/bin/python scripts/replay_positive_control.py \
      --split "$split" --proxy "$PROXY" --target "$target" \
      --endpoint http://127.0.0.1:8000 --seed "$SEED"
  done
  cleanup
  PID=""
done

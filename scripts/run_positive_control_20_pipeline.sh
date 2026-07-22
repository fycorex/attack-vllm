#!/usr/bin/env bash
# Resumable 20-image single-proxy positive-control pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAGE="${1:-help}"
DATA="data/proxy_selector_positive_control_20"
OUT="outputs/proxy_selector_positive_control_20"
BEST_PROXY="${BEST_PROXY:-}"
PID=""

cleanup() { if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
wait_ready() { for _ in $(seq 1 120); do curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
serve() { bash scripts/serve_proxy_selector_target.sh --target "$1" --port 8000 > "$OUT/${1}_server.log" 2>&1 & PID="$!"; wait_ready; }
screen_target() {
  local target="$1"
  serve "$target"
  PYTHONPATH=src .venv-proxy/bin/python scripts/screen_vqav2_proxy_selector.py --candidates "$DATA/candidate_manifest.json" --output "$DATA" --target "$target" --endpoint http://127.0.0.1:8000
  cleanup; PID=""
}
finalize_target() {
  local target="$1"
  serve "$target"
  PYTHONPATH=src .venv-proxy/bin/python scripts/finalize_vqav2_proxy_selector.py --pairs "$DATA/screened_pairs.json" --output "$DATA" --target "$target" --endpoint http://127.0.0.1:8000 --dev-count 20 --test-count 0
  cleanup; PID=""
}
attack() {
  local epsilon="$1" proxy="$2"
  PYTHONPATH=src .venv-proxy/bin/python scripts/run_max_strength_positive_control.py \
    --split dev --manifest "$DATA/dev_manifest.json" --candidates "$DATA/candidate_manifest.json" \
    --output "$OUT" --proxy "$proxy" --epsilon "$epsilon" --steps 300 --restarts 5 --eot 8 --seed 42
}
replay() {
  local epsilon="$1" proxy="$2"
  for target in T1 T2; do
    serve "$target"
    PYTHONPATH=src .venv-proxy/bin/python scripts/replay_positive_control.py \
      --split dev --manifest "$DATA/dev_manifest.json" --output "$OUT" --proxy "$proxy" \
      --epsilon "$epsilon" --target "$target" --endpoint http://127.0.0.1:8000 --seed 42
    cleanup; PID=""
  done
}

mkdir -p "$OUT"
case "$STAGE" in
  prepare) bash scripts/prepare_positive_control_20.sh ;;
  screen) screen_target T1; screen_target T2; finalize_target T1; finalize_target T2 ;;
  attack16) for proxy in P2 P3 P1; do attack "$(python3 - <<'PY'
print(16 / 255)
PY
)" "$proxy"; done ;;
  replay16) for proxy in P2 P3 P1; do replay "$(python3 - <<'PY'
print(16 / 255)
PY
)" "$proxy"; done ;;
  attack8)
    [[ -n "$BEST_PROXY" ]] || { echo "Set BEST_PROXY=P1, P2, or P3 after inspecting 16/255 replay." >&2; exit 2; }
    attack "$(python3 - <<'PY'
print(8 / 255)
PY
)" "$BEST_PROXY"
    ;;
  replay8)
    [[ -n "$BEST_PROXY" ]] || { echo "Set BEST_PROXY=P1, P2, or P3." >&2; exit 2; }
    replay "$(python3 - <<'PY'
print(8 / 255)
PY
)" "$BEST_PROXY"
    ;;
  all16) "$0" prepare; "$0" screen; "$0" attack16; "$0" replay16 ;;
  help|--help|-h|*)
    echo "Usage: $0 {prepare|screen|attack16|replay16|attack8|replay8|all16}; set BEST_PROXY for 8/255 stages."
    ;;
esac

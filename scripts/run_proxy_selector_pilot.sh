#!/usr/bin/env bash
# Branch-safe command dispatcher for the cross-family proxy selector pilot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXPECTED_BRANCH="experiment/proxy-selector-pilot"
COMMAND="${1:---help}"
shift || true

usage() {
  cat <<'EOF'
Usage: bash scripts/run_proxy_selector_pilot.sh <command> [args]

Commands: setup, prepare-candidates, screen-data, finalize-data, phase0, smoke, extract-cka,
dev, final, strong, ve-sanity, max-strength, replay-positive-control, serve-target, replay, summarize, estimate, all.

`prepare-candidates` requires local official VQAv2 JSON and COCO val2014 images.
`screen-data` runs one vLLM target at a time; run it once for T1 and once for T2.
Every expensive phase checks the isolated experiment branch.
EOF
}

require_branch() {
  if [[ "${ALLOW_OTHER_BRANCH:-0}" == "1" ]]; then return; fi
  local branch
  branch="$(git branch --show-current)"
  if [[ "$branch" != "$EXPECTED_BRANCH" ]]; then
    echo "Refusing to run on '$branch'; expected '$EXPECTED_BRANCH'." >&2
    echo "Set ALLOW_OTHER_BRANCH=1 only for documented debugging." >&2
    exit 2
  fi
}

case "$COMMAND" in
  --help|-h|help) usage ;;
  setup) bash scripts/setup_proxy_selector_env.sh attack "$@" ;;
  estimate) require_branch; python3 scripts/estimate_proxy_selector_runtime.py "$@" ;;
  prepare-candidates)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/prepare_vqav2_proxy_selector.py "$@"
    ;;
  screen-data)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/screen_vqav2_proxy_selector.py "$@"
    ;;
  finalize-data)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/finalize_vqav2_proxy_selector.py "$@"
    ;;
  phase0)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/run_phase0_proxy_selector.py "$@"
    ;;
  extract-cka)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/extract_proxy_selector_features.py "$@"
    ;;
  dev)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/run_proxy_selector_attacks.py --split dev "$@"
    ;;
  final)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/run_proxy_selector_attacks.py --split test "$@"
    ;;
  strong)
    require_branch
    echo "Strong tracks are separate from C0--C2: S8 is strict, S16 is radius-only, and B16 is the 1000-step/3-restart high-budget track." >&2
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/run_proxy_selector_attacks.py --split test "$@"
    ;;
  ve-sanity)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/run_veattack_sanity.py "$@"
    ;;
  max-strength)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/run_max_strength_positive_control.py "$@"
    ;;
  replay-positive-control)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/replay_positive_control.py "$@"
    ;;
  replay)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/replay_proxy_selector_vllm.py "$@"
    ;;
  select-recipes)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/select_proxy_selector_recipes.py
    ;;
  summarize)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/summarize_proxy_selector.py
    ;;
  plot)
    require_branch
    exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv-proxy/bin/python" scripts/plot_proxy_selector_results.py
    ;;
  smoke|serve-target|all)
    require_branch
    echo "'$COMMAND' is unavailable until its runner implementation lands." >&2
    exit 3
    ;;
  *) usage; exit 2 ;;
esac

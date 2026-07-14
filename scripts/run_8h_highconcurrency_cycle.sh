#!/usr/bin/env bash
# Reproducible host-side launcher. It preserves the prior interrupted output
# and starts a fresh cycle. The 200-image geometry confirmation always follows
# the time-bounded search because it is independent of candidate selection.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT/outputs/transfer_search_cycle_8h_highconcurrency}"
PYTHON="$ROOT/.venv/bin/python"

mkdir -p "$OUTPUT"

set +e
PYTHONPATH="$ROOT/src:$ROOT" TECHUTOPIA_API_KEY="${TECHUTOPIA_API_KEY:-research-cycle}" \
  "$PYTHON" "$ROOT/scripts/run_transfer_search_cycle.py" \
  --hours 8 --gpu-jobs 6 --heldout-batch-size 64 --augmentation-workers 2 --promote 3 --run-api \
  --output "$OUTPUT" \
  >"$OUTPUT/launcher.log" 2>&1
CYCLE_EXIT=$?
printf '{"cycle_exit": %s}\n' "$CYCLE_EXIT" >"$OUTPUT/cycle_exit.json"

set +e
PYTHONPATH="$ROOT/src:$ROOT" \
  "$PYTHON" "$ROOT/scripts/run_caltech200_geometry_confirmation.py" \
  --output-root "$OUTPUT/alignment_confirmation_caltech200" \
  --cycle-root "$OUTPUT" --attack-workers 2 --run-api --api-limit 50 \
  >>"$OUTPUT/confirmation_200.log" 2>&1
CONFIRMATION_EXIT=$?
set -e

printf '{"cycle_exit": %s, "confirmation_exit": %s}\n' \
  "$CYCLE_EXIT" "$CONFIRMATION_EXIT" >"$OUTPUT/overall_exit.json"
exit "$CONFIRMATION_EXIT"

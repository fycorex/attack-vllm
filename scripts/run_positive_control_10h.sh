#!/usr/bin/env bash
# Ten-hour, resumable A6000 scheduler for the paper-inspired single-proxy control.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HOURS="${1:-10}"
OUT="outputs/proxy_selector_pilot/ten_hour_run"
mkdir -p "$OUT"
START="$(date +%s)"
# A restart can preserve the original wall-clock budget by exporting the
# absolute deadline recorded in status.json.  Without it, retain the simple
# user-facing `... 10` interface.
if [[ -n "${PROXY_SELECTOR_DEADLINE_EPOCH:-}" ]]; then
  DEADLINE="$PROXY_SELECTOR_DEADLINE_EPOCH"
else
  DEADLINE=$((START + HOURS * 3600))
fi
if (( DEADLINE <= START + 300 )); then
  echo "Deadline is too close or already expired: $DEADLINE" >&2
  exit 2
fi
ATTACK_DEADLINE=$((DEADLINE - 2 * 3600))

remaining() { local now; now="$(date +%s)"; echo $(( DEADLINE - now )); }
has_time() { (( $(remaining) > 300 )); }
attack_remaining() { local now; now="$(date +%s)"; echo $(( ATTACK_DEADLINE - now )); }
has_attack_time() { (( $(attack_remaining) > 300 )); }
write_status() {
  printf '{"started_epoch":%s,"deadline_epoch":%s,"remaining_seconds":%s,"stage":"%s"}\n' \
    "$START" "$DEADLINE" "$(remaining)" "$1" > "$OUT/status.json.tmp"
  mv "$OUT/status.json.tmp" "$OUT/status.json"
}

run_pair() {
  local proxy="$1" split="$2" pair="$3"
  has_attack_time || return 0
  timeout --signal=TERM --kill-after=30 "$(attack_remaining)" \
    env PYTHONPATH=src .venv-proxy/bin/python scripts/run_max_strength_positive_control.py \
      --split "$split" --proxy "$proxy" --pair-id "$pair" --steps 300 --restarts 5 --eot 8 --seed 42 \
      > "$OUT/${proxy}_${split}_${pair}.log" 2>&1 || true
}

run_pool() {
  # Items use SPLIT:PAIR_ID.  Keeping development and test work in one queue
  # prevents the final development item from leaving two GPU workers idle.
  local proxy="$1"; shift
  local -a items=("$@")
  local next=0 active=0 pid=""
  while (( next < ${#items[@]} || active > 0 )); do
    while (( active < 3 && next < ${#items[@]} )) && has_attack_time; do
      local split="${items[$next]%%:*}"
      local pair="${items[$next]#*:}"
      run_pair "$proxy" "$split" "$pair" &
      next=$((next + 1)); active=$((active + 1))
    done
    wait -n || true
    active=$((active - 1))
    has_attack_time || break
  done
  wait || true
}

write_status "P2_16"
# candidate_000 was already completed; candidate_001 is semantically invalid.
run_pool P2 \
  dev:candidate_004 dev:candidate_005 dev:candidate_013 dev:candidate_016 \
  test:candidate_009 test:candidate_011 test:candidate_018 test:candidate_006 \
  test:candidate_008 test:candidate_025 test:candidate_029 test:candidate_014 test:candidate_017

write_status "P3_16"
run_pool P3 \
  dev:candidate_004 dev:candidate_005 dev:candidate_013 dev:candidate_016 \
  test:candidate_009 test:candidate_011 test:candidate_018 test:candidate_006 \
  test:candidate_008 test:candidate_025 test:candidate_029 test:candidate_014 test:candidate_017

write_status "P1_16"
run_pool P1 \
  dev:candidate_004 dev:candidate_005 dev:candidate_013 dev:candidate_016 \
  test:candidate_009 test:candidate_011 test:candidate_018 test:candidate_006 \
  test:candidate_008 test:candidate_025 test:candidate_029 test:candidate_014 test:candidate_017

write_status "replay_completed_images"
# Attacks are stopped two hours before the deadline.  Replay only generated
# images; per-item caches make this safe to resume after an interruption.
if has_time; then
  for proxy in P2 P3 P1; do
    has_time || break
    bash scripts/replay_positive_control_batch.sh "$proxy" 42 || true
  done
  PYTHONPATH=src .venv-proxy/bin/python scripts/summarize_positive_control.py || true
fi
write_status "complete_or_deadline"

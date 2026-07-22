#!/usr/bin/env bash
# Ten-hour, resumable A6000 scheduler for the paper-inspired single-proxy control.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
HOURS="${1:-10}"
OUT="outputs/proxy_selector_pilot/ten_hour_run"
mkdir -p "$OUT"
START="$(date +%s)"
DEADLINE=$((START + HOURS * 3600))
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
  local proxy="$1" split="$2"; shift 2
  local -a pairs=("$@")
  local next=0 active=0 pid=""
  while (( next < ${#pairs[@]} || active > 0 )); do
    while (( active < 3 && next < ${#pairs[@]} )) && has_attack_time; do
      run_pair "$proxy" "$split" "${pairs[$next]}" &
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
run_pool P2 dev candidate_004 candidate_005 candidate_013 candidate_016
run_pool P2 test candidate_009 candidate_011 candidate_018 candidate_006 candidate_008 candidate_025 candidate_029 candidate_014 candidate_017

write_status "P3_16"
run_pool P3 dev candidate_004 candidate_005 candidate_013 candidate_016

write_status "P1_16"
run_pool P1 dev candidate_004 candidate_005 candidate_013 candidate_016

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

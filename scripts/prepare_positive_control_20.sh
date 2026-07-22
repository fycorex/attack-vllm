#!/usr/bin/env bash
# Build an isolated 20-pair candidate pool; screen it with T1 then T2 before use.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="data/proxy_selector_positive_control_20"

PYTHONPATH=src .venv-proxy/bin/python scripts/prepare_vqav2_proxy_selector.py \
  --questions data/vqav2_official/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations data/vqav2_official/v2_mscoco_val2014_annotations.json \
  --images data/vqav2_official/val2014 --output "$OUT" --candidate-count 120 --seed 314159

printf '%s\n' \
  "Candidate pool: $OUT/candidate_manifest.json" \
  "Screen with each target sequentially, then make exactly 20 screened pairs:" \
  "PYTHONPATH=src .venv-proxy/bin/python scripts/screen_vqav2_proxy_selector.py --candidates $OUT/candidate_manifest.json --output $OUT --target T1 --endpoint http://127.0.0.1:8000" \
  "Restart server as T2, then repeat screen with --target T2." \
  "PYTHONPATH=src .venv-proxy/bin/python scripts/finalize_vqav2_proxy_selector.py --pairs $OUT/screened_pairs.json --output $OUT --target T1 --endpoint http://127.0.0.1:8000 --dev-count 20 --test-count 0" \
  "Restart server as T2, then repeat finalize with --target T2."

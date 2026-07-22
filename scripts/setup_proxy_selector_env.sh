#!/usr/bin/env bash
# Create one dependency-isolated environment without downloading checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
if [[ "$MODE" != "attack" && "$MODE" != "vllm" ]]; then
  echo "Usage: bash scripts/setup_proxy_selector_env.sh {attack|vllm}" >&2
  exit 2
fi

VENV="$ROOT/.venv-proxy"
REQ="$ROOT/requirements-proxy-selector-attack.txt"
if [[ "$MODE" == "vllm" ]]; then
  VENV="$ROOT/.venv-vllm"
  REQ="$ROOT/requirements-proxy-selector-vllm.txt"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REQ"
"$VENV/bin/python" - <<'PY'
import json
import os
import platform
import sys
from pathlib import Path
report = {"python": sys.version, "platform": platform.platform()}
try:
    import torch
    report.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
except ImportError:
    report["torch"] = "not installed"
try:
    import transformers
    report["transformers"] = transformers.__version__
except ImportError:
    report["transformers"] = "not installed"
root = Path(os.environ.get("PROXY_SELECTOR_OUTPUT", "outputs/proxy_selector_pilot"))
path = root / "environment" / f"{os.environ.get('PROXY_SELECTOR_ENV_NAME', 'environment')}.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
print(f"Wrote {path}")
PY

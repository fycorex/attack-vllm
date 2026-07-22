#!/usr/bin/env bash
# Serve one validated target at a time with the CUDA-12.8-compatible vLLM env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET=""
PORT=8000
IMPLEMENTATION="auto"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --implementation) IMPLEMENTATION="$2"; shift 2 ;;
    --help|-h) echo "Usage: $0 --target {T1|T2} [--port 8000]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$TARGET" != "T1" && "$TARGET" != "T2" ]]; then
  echo "--target T1 or T2 is required" >&2; exit 2
fi
VENV="$ROOT/.venv-vllm"
CACHE="${HF_HOME:-$ROOT/model_cache}"
if [[ "$CACHE" == "$ROOT/model_cache" ]]; then
  CACHE="$ROOT/model_cache"
fi
if [[ "$TARGET" == "T1" ]]; then
  REPO="google/gemma-4-E2B-it"
  EXTRA=()
else
  REPO="OpenGVLab/InternVL3_5-2B-HF"
  EXTRA=(--trust-remote-code --mm-processor-kwargs '{"size":{"height":448,"width":448}}')
fi
SNAPSHOT="$(find "$ROOT/model_cache/models--${REPO//\//--}/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)"
if [[ -z "$SNAPSHOT" ]]; then
  echo "Missing local checkpoint for $REPO; download it before serving." >&2; exit 1
fi
export PATH="$VENV/bin:$PATH"
if [[ "$TARGET" == "T1" && "$IMPLEMENTATION" != "vllm" ]]; then
  # vLLM 0.10.2 is retained for CUDA 12.8 / InternVL compatibility. Its
  # Transformers ceiling cannot deserialize model_type=gemma4, so T1 uses the
  # documented native-Transformers fallback.
  exec "$ROOT/.venv-proxy/bin/python" "$ROOT/scripts/serve_gemma4_transformers.py" \
    --model "$SNAPSHOT" --served-model-name T1 --port "$PORT"
fi
exec "$VENV/bin/vllm" serve "$SNAPSHOT" --served-model-name "$TARGET" --dtype bfloat16 \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --port "$PORT" "${EXTRA[@]}"

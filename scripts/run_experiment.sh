#!/bin/bash
# =============================================================================
# Transferable Adversarial Attack on VLLMs - Complete Experiment Script
# Based on arXiv:2505.01050v1
#
# Usage on different machines:
# - RTX 4060 (8GB): Use 5 models, sequential loading
# - A6000 (48GB): Use 8 models, parallel loading
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
VENV="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV/bin/python"
DEFAULT_ATTACK_CONFIG="configs/caption_attack_paper.yaml"
DEFAULT_OUTPUT_DIR="outputs/paper_caltech"
DEFAULT_ITEMS=50
DEFAULT_STEPS=300

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

find_python() {
    if [ -n "${PYTHON_BIN:-}" ]; then
        if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            echo "$PYTHON_BIN"
            return 0
        fi

        log_error "PYTHON_BIN is set to '$PYTHON_BIN', but that command was not found" >&2
        return 1
    fi

    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        echo "python"
        return 0
    fi

    log_error "No Python interpreter found. Install Python 3 or set PYTHON_BIN=/path/to/python" >&2
    return 1
}

create_venv() {
    local python_cmd="$1"
    local clear_flag="${2:-}"
    local venv_args=()

    if [ -n "$clear_flag" ]; then
        venv_args+=("$clear_flag")
    fi

    if ! "$python_cmd" -m venv "${venv_args[@]}" "$VENV"; then
        log_error "Failed to create virtual environment at $VENV"
        log_error "Install the Python venv module for your interpreter, then rerun this script"
        exit 1
    fi
}

# =============================================================================
# Step 1: Environment Setup
# =============================================================================
setup_environment() {
    log_info "Setting up environment..."
    log_info "Project root: $PROJECT_ROOT"
    log_info "Venv path: $VENV"

    local python_cmd
    python_cmd="$(find_python)"
    log_info "Using Python: $(command -v "$python_cmd")"
    log_info "Python version: $($python_cmd --version 2>&1)"

    # Create venv if it does not exist. If a partial/broken venv directory is
    # present, rebuild it in place instead of failing on a missing activate file.
    if [ ! -d "$VENV" ]; then
        log_info "Creating virtual environment..."
        create_venv "$python_cmd"
    elif [ ! -f "$VENV/bin/activate" ] || [ ! -x "$VENV_PYTHON" ]; then
        log_warn "Virtual environment at $VENV is incomplete; rebuilding it..."
        create_venv "$python_cmd" "--clear"
    else
        log_info "Virtual environment exists and is valid"
    fi

    # Check activation
    if [ -f "$VENV/bin/activate" ]; then
        log_info "Activating virtual environment..."
        source "$VENV/bin/activate"
    else
        log_error "Virtual environment not found at $VENV"
        exit 1
    fi

    # Upgrade pip
    log_info "Upgrading pip..."
    "$VENV_PYTHON" -m pip install --upgrade pip

    # Install requirements
    if [ -f "requirements.txt" ]; then
        log_info "Installing requirements from requirements.txt..."
        "$VENV_PYTHON" -m pip install -r requirements.txt
    fi

    # Install scipy for Caltech101 dataset
    log_info "Installing scipy..."
    "$VENV_PYTHON" -m pip install scipy

    # Verify installation
    log_info "Verifying key packages..."
    "$VENV_PYTHON" -c "import torch; print(f'  PyTorch: {torch.__version__}')"
    "$VENV_PYTHON" -c "import open_clip; print(f'  open_clip: OK')"
    "$VENV_PYTHON" -c "import transformers; print(f'  transformers: {transformers.__version__}')"

    log_info "Environment ready"
}

# =============================================================================
# Step 2: Download CLIP Surrogate Models
# =============================================================================
download_models() {
    log_info "Downloading CLIP surrogate models (paper Section 3.2: 4x ViT-H + 3x SigLIP + 1x ConvNeXt)..."

    source "$VENV/bin/activate"

    # Paper Section 3.2: 4x ViT-H + 3x ViT-SigLIP + 1x ConvNeXt XXL
    # These achieve 94.4% ASR on GPT-4o (paper Table 3)
    python3 << 'EOF'
import open_clip
import os

# 8 CLIP models as per paper Section 3.2
models = [
    # ViT-H variants (4 from paper)
    ("ViT-H-14", "laion2b_s32b_b79k"),
    ("ViT-H-14", "metaclip_fullcc"),
    ("ViT-H-14", "metaclip_altogether"),
    ("ViT-H-14-378", "dfn5b"),
    # ViT-SigLIP variants (3 from paper)
    ("ViT-B-16-SigLIP", "webli"),
    ("ViT-B-16-SigLIP-384", "webli"),
    ("ViT-L-16-SigLIP-384", "webli"),
    # ConvNeXt XXL (1 from paper)
    ("convnext_xxlarge", "laion2b_s34b_b82k_augreg"),
]

cache_dir = "models/open_clip"
os.makedirs(cache_dir, exist_ok=True)

def pretrained_uses_quick_gelu(model_name, pretrained):
    pretrained_cfg = open_clip.get_pretrained_cfg(model_name, pretrained)
    return bool(pretrained_cfg and pretrained_cfg.get("quick_gelu", False))

success = 0
failed = 0
available_models = set(open_clip.list_models())
for model_name, pretrained in models:
    try:
        print(f"Loading {model_name}:{pretrained}...")
        if model_name not in available_models:
            raise RuntimeError(f"Model config for '{model_name}' is not available in this open_clip install")
        model = open_clip.create_model(
            model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
            force_quick_gelu=pretrained_uses_quick_gelu(model_name, pretrained),
        )
        print(f"  ✓ {model_name}:{pretrained}")
        success += 1
    except Exception as e:
        print(f"  ✗ {model_name}:{pretrained} - {str(e)[:80]}")
        failed += 1

print(f"\nDownloaded: {success}/{len(models)} models")
if failed > 0:
    print(f"Failed: {failed} models")
    print("Rerun download after fixing network/cache issues, or set HF_TOKEN for Hugging Face rate limits.")
    raise SystemExit(1)
EOF

    log_info "Model download complete"
}

# =============================================================================
# Step 3: Prepare Dataset
# =============================================================================
prepare_dataset() {
    log_info "Preparing dataset..."

    source "$VENV/bin/activate"

    REQUESTED_ITEMS=${1:-$DEFAULT_ITEMS}
    MANIFEST_PATH="data/caltech_large/manifest.json"

    # Option 1: Use existing dataset
    if [ -f "$MANIFEST_PATH" ]; then
        EXISTING_ITEMS=$("$VENV_PYTHON" - "$MANIFEST_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    print(0)
else:
    print(len(payload.get("items", [])))
PY
)
    else
        EXISTING_ITEMS=0
    fi

    if [ "$EXISTING_ITEMS" -ge "$REQUESTED_ITEMS" ]; then
        log_info "Using existing Caltech101 dataset with $EXISTING_ITEMS items"
        export DATASET="caltech_large"
    else
        # Option 2: Prepare requested demo dataset size.
        if [ "$EXISTING_ITEMS" -gt 0 ]; then
            log_warn "Existing Caltech101 manifest has $EXISTING_ITEMS items; regenerating $REQUESTED_ITEMS items"
        else
            log_info "Preparing Caltech101 dataset ($REQUESTED_ITEMS items)..."
        fi
        python scripts/prepare_caltech_demo.py \
            --output_dir data/caltech_large \
            --num_items "$REQUESTED_ITEMS" \
            --num_examples 50
        export DATASET="caltech_large"
    fi

    log_info "Dataset ready: $DATASET"
}

# =============================================================================
# Step 4: Run Adversarial Attack
# =============================================================================
run_attack() {
    log_info "Running adversarial attack..."

    source "$VENV/bin/activate"

    CONFIG=${1:-$DEFAULT_ATTACK_CONFIG}
    ITEMS=${2:-$DEFAULT_ITEMS}
    STEPS=${3:-$DEFAULT_STEPS}

    log_info "Attack config: $CONFIG"
    log_info "Attack items: $ITEMS"
    log_info "Attack steps: $STEPS"

    python scripts/run_caption_attack.py \
        --config "$CONFIG" \
        --attack_limit "$ITEMS" \
        --steps "$STEPS"

    log_info "Attack complete"
}

# =============================================================================
# Step 5: Evaluate with GPT Replay Matrix
# =============================================================================
evaluate_gpt_matrix() {
    log_info "Evaluating replay matrix with TechUtopia GPT models..."

    source "$VENV/bin/activate"

    OUTPUT_DIR=${1:-$DEFAULT_OUTPUT_DIR}
    METRICS_GLOB=${2:-"$OUTPUT_DIR/item_*/metrics.json"}
    export TECHUTOPIA_API_KEY="${TECHUTOPIA_API_KEY:-sk-test}"

    mkdir -p "$OUTPUT_DIR"

    local configs=(
        "gpt4o:configs/techutopia_gpt4o_caption_eval.yaml"
        "gpt5mini:configs/techutopia_gpt5mini_caption_eval.yaml"
    )

    log_info "Eval output: $OUTPUT_DIR"
    log_info "Metrics glob: $METRICS_GLOB"

    for entry in "${configs[@]}"; do
        local name="${entry%%:*}"
        local config="${entry#*:}"
        local result_path="$OUTPUT_DIR/eval_${name}.jsonl"

        log_info "Running $name eval with $config"
        python scripts/replay_gpt_eval.py \
            --config "$config" \
            --glob "$METRICS_GLOB" \
            > "$result_path"
        log_info "Saved $name results to $result_path"
    done

    log_info "GPT replay matrix complete"
}

# =============================================================================
# Step 6: Full Pipeline
# =============================================================================
run_full_experiment() {
    log_info "Starting full experiment pipeline..."

    setup_environment
    download_models
    prepare_dataset "$DEFAULT_ITEMS"
    run_attack "$@"
    evaluate_gpt_matrix

    log_info "Full experiment complete!"
}

run_full_matrix() {
    log_info "Starting 50x300 attack plus GPT eval matrix..."

    prepare_dataset "${2:-$DEFAULT_ITEMS}"
    run_attack "${1:-$DEFAULT_ATTACK_CONFIG}" "${2:-$DEFAULT_ITEMS}" "${3:-$DEFAULT_STEPS}"
    evaluate_gpt_matrix "${4:-$DEFAULT_OUTPUT_DIR}"

    log_info "50x300 attack plus GPT eval matrix complete"
}

# =============================================================================
# Main Entry Point
# =============================================================================
case "${1:-}" in
    setup)
        setup_environment
        ;;
    download)
        download_models
        ;;
    dataset)
        prepare_dataset
        ;;
    attack)
        run_attack "${2:-}" "${3:-}" "${4:-}"
        ;;
    eval-gpt|evaluate-gpt|matrix)
        evaluate_gpt_matrix "${2:-}" "${3:-}"
        ;;
    full-matrix)
        shift
        run_full_matrix "$@"
        ;;
    full)
        shift
        run_full_experiment "$@"
        ;;
    *)
        echo "Usage: $0 {setup|download|dataset|attack|eval-gpt|full-matrix|full}"
        echo ""
        echo "Commands:"
        echo "  setup           - Set up Python environment"
        echo "  download        - Download 8 CLIP models (paper Table 3)"
        echo "  dataset         - Prepare 50-item dataset"
        echo "  attack [cfg] [n] [s] - Run attack (config, items, steps)"
        echo "  eval-gpt [out] [glob] - Replay GPT-4o and GPT-5-mini eval matrix"
        echo "  full-matrix [cfg] [n] [s] [out] - Run 50x300 attack plus GPT eval matrix"
        echo "  full            - Run setup, downloads, attack, and GPT eval matrix"
        echo ""
        echo "Examples:"
        echo "  $0 full-matrix                             # 50 items x 300 steps + GPT eval matrix"
        echo "  $0 attack configs/caption_attack_paper.yaml 50 300"
        echo "  $0 eval-gpt outputs/paper_caltech"
        exit 1
        ;;
esac

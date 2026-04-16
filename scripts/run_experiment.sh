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
DEFAULT_LLAVA_VQA_CONFIG="configs/llava_bench_vqa_eps16.yaml"
DEFAULT_LLAVA_VQA_OUTPUT_DIR="outputs/llava_vqa_eps16"
DEFAULT_RECEIPT_TEXT_CONFIG_16="configs/receipt_text_eps16.yaml"
DEFAULT_RECEIPT_TEXT_CONFIG_32="configs/receipt_text_eps32.yaml"
DEFAULT_RECEIPT_TEXT_OUTPUT_16="outputs/receipt_text_eps16"
DEFAULT_RECEIPT_TEXT_OUTPUT_32="outputs/receipt_text_eps32"
DEFAULT_ITEMS=50
DEFAULT_STEPS=300
DEFAULT_LLAVA_VQA_IMAGES=5
DEFAULT_LLAVA_VQA_STEPS=300
DEFAULT_RECEIPT_IMAGES=20
DEFAULT_RECEIPT_TEXT_STEPS=300

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

evaluate_gpt_vqa_matrix() {
    log_info "Evaluating replay matrix with TechUtopia LLaVA-Bench VQA models..."

    source "$VENV/bin/activate"

    OUTPUT_DIR=${1:-$DEFAULT_LLAVA_VQA_OUTPUT_DIR}
    METRICS_GLOB=${2:-"$OUTPUT_DIR/item_*/metrics.json"}
    export TECHUTOPIA_API_KEY="${TECHUTOPIA_API_KEY:-sk-test}"

    mkdir -p "$OUTPUT_DIR"

    local configs=(
        "gpt4o:configs/techutopia_gpt4o_vqa_eval.yaml"
        "gpt5mini:configs/techutopia_gpt5mini_vqa_eval.yaml"
    )

    log_info "Eval output: $OUTPUT_DIR"
    log_info "Metrics glob: $METRICS_GLOB"

    for entry in "${configs[@]}"; do
        local name="${entry%%:*}"
        local config="${entry#*:}"
        local result_path="$OUTPUT_DIR/eval_${name}.jsonl"

        log_info "Running $name VQA eval with $config"
        python scripts/replay_gpt_eval.py \
            --config "$config" \
            --glob "$METRICS_GLOB" \
            > "$result_path"
        log_info "Saved $name results to $result_path"
    done

    python scripts/analyze_vqa_eval.py \
        "$OUTPUT_DIR/eval_gpt4o.jsonl" \
        "$OUTPUT_DIR/eval_gpt5mini.jsonl" \
        --output "$OUTPUT_DIR/eval_summary.json"

    log_info "LLaVA-Bench VQA GPT replay matrix complete"
}

evaluate_gpt_receipt_text_matrix() {
    log_info "Evaluating replay matrix with TechUtopia receipt text models..."

    source "$VENV/bin/activate"

    OUTPUT_DIR=${1:-$DEFAULT_RECEIPT_TEXT_OUTPUT_16}
    METRICS_GLOB=${2:-"$OUTPUT_DIR/item_*/metrics.json"}
    export TECHUTOPIA_API_KEY="${TECHUTOPIA_API_KEY:-sk-test}"

    mkdir -p "$OUTPUT_DIR"

    local configs=(
        "gpt4o:configs/techutopia_gpt4o_receipt_text_eval.yaml"
        "gpt5mini:configs/techutopia_gpt5mini_receipt_text_eval.yaml"
    )

    log_info "Eval output: $OUTPUT_DIR"
    log_info "Metrics glob: $METRICS_GLOB"

    for entry in "${configs[@]}"; do
        local name="${entry%%:*}"
        local config="${entry#*:}"
        local result_path="$OUTPUT_DIR/eval_${name}.jsonl"

        log_info "Running $name receipt text eval with $config"
        python scripts/replay_gpt_eval.py \
            --config "$config" \
            --glob "$METRICS_GLOB" \
            > "$result_path"
        log_info "Saved $name results to $result_path"
    done

    python scripts/analyze_receipt_text_eval.py \
        "$OUTPUT_DIR/eval_gpt4o.jsonl" \
        "$OUTPUT_DIR/eval_gpt5mini.jsonl" \
        --output "$OUTPUT_DIR/eval_summary.json"

    log_info "Receipt text GPT replay matrix complete"
}

prepare_llava_vqa_dataset() {
    log_info "Preparing LLaVA-Bench COCO VQA demo dataset..."

    source "$VENV/bin/activate"

    local images=${1:-$DEFAULT_LLAVA_VQA_IMAGES}
    local examples=${2:-50}
    python scripts/prepare_llava_bench_coco_vqa.py \
        --cache_dir data/raw/hf_cache \
        --output_dir data/llava_bench_coco_vqa \
        --num_images "$images" \
        --num_examples "$examples"
    log_info "LLaVA-Bench COCO VQA demo dataset ready"
}

prepare_receipt_text_dataset() {
    log_info "Preparing TrainingDataPro receipt text dataset..."

    source "$VENV/bin/activate"

    local images=${1:-$DEFAULT_RECEIPT_IMAGES}
    local examples=${2:-50}
    local repo_id="${RECEIPT_DATASET_REPO:-TrainingDataPro/ocr-receipts-text-detection}"
    python scripts/prepare_trainingdatapro_receipts_text.py \
        --repo_id "$repo_id" \
        --cache_dir data/raw/hf_cache \
        --output_dir data/trainingdatapro_receipts_text \
        --limit_images "$images" \
        --questions_per_image 2 \
        --num_examples "$examples"
    log_info "TrainingDataPro receipt text dataset ready"
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

run_llava_vqa_demo() {
    local images=${1:-$DEFAULT_LLAVA_VQA_IMAGES}
    local steps=${2:-$DEFAULT_LLAVA_VQA_STEPS}
    local items=$((images * 3))

    prepare_llava_vqa_dataset "$images" 50
    run_attack "$DEFAULT_LLAVA_VQA_CONFIG" "$items" "$steps"

    log_info "LLaVA-Bench VQA demo attack complete"
    log_info "Run GPT replay with: bash scripts/run_experiment.sh eval-llava-vqa-gpt $DEFAULT_LLAVA_VQA_OUTPUT_DIR"
}

run_llava_vqa_demo_matrix() {
    local images=${1:-$DEFAULT_LLAVA_VQA_IMAGES}
    local steps=${2:-$DEFAULT_LLAVA_VQA_STEPS}

    run_llava_vqa_demo "$images" "$steps"
    evaluate_gpt_vqa_matrix "$DEFAULT_LLAVA_VQA_OUTPUT_DIR"
}

run_receipt_text_experiment() {
    local epsilon=${1:-16}
    local steps=${2:-$DEFAULT_RECEIPT_TEXT_STEPS}
    local images=${3:-$DEFAULT_RECEIPT_IMAGES}
    local items=$((images * 2))
    local config="$DEFAULT_RECEIPT_TEXT_CONFIG_16"

    if [ "$epsilon" = "32" ]; then
        config="$DEFAULT_RECEIPT_TEXT_CONFIG_32"
    fi

    prepare_receipt_text_dataset "$images" 50
    run_attack "$config" "$items" "$steps"

    log_info "Receipt text epsilon=$epsilon attack complete"
}

run_receipt_text_matrix() {
    local epsilon=${1:-16}
    local steps=${2:-$DEFAULT_RECEIPT_TEXT_STEPS}
    local images=${3:-$DEFAULT_RECEIPT_IMAGES}
    local output="$DEFAULT_RECEIPT_TEXT_OUTPUT_16"

    if [ "$epsilon" = "32" ]; then
        output="$DEFAULT_RECEIPT_TEXT_OUTPUT_32"
    fi

    run_receipt_text_experiment "$epsilon" "$steps" "$images"
    evaluate_gpt_receipt_text_matrix "$output"
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
    prepare-llava-vqa)
        prepare_llava_vqa_dataset "${2:-}" "${3:-}"
        ;;
    prepare-receipt-text)
        prepare_receipt_text_dataset "${2:-}" "${3:-}"
        ;;
    attack)
        run_attack "${2:-}" "${3:-}" "${4:-}"
        ;;
    eval-gpt|evaluate-gpt|matrix)
        evaluate_gpt_matrix "${2:-}" "${3:-}"
        ;;
    eval-llava-vqa-gpt|eval-vqa-gpt)
        evaluate_gpt_vqa_matrix "${2:-$DEFAULT_LLAVA_VQA_OUTPUT_DIR}" "${3:-}"
        ;;
    eval-receipt-text-gpt)
        evaluate_gpt_receipt_text_matrix "${2:-$DEFAULT_RECEIPT_TEXT_OUTPUT_16}" "${3:-}"
        ;;
    llava-vqa-demo|vqa-demo)
        run_llava_vqa_demo "${2:-}" "${3:-}"
        ;;
    llava-vqa-demo-matrix|vqa-demo-matrix)
        run_llava_vqa_demo_matrix "${2:-}" "${3:-}"
        ;;
    receipt-text-16)
        run_receipt_text_experiment 16 "${2:-}" "${3:-}"
        ;;
    receipt-text-32)
        run_receipt_text_experiment 32 "${2:-}" "${3:-}"
        ;;
    receipt-text-16-matrix)
        run_receipt_text_matrix 16 "${2:-}" "${3:-}"
        ;;
    receipt-text-32-matrix)
        run_receipt_text_matrix 32 "${2:-}" "${3:-}"
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
        echo "Usage: $0 {setup|download|dataset|prepare-llava-vqa|prepare-receipt-text|attack|eval-gpt|eval-llava-vqa-gpt|eval-receipt-text-gpt|llava-vqa-demo|llava-vqa-demo-matrix|receipt-text-16|receipt-text-32|receipt-text-16-matrix|receipt-text-32-matrix|full-matrix|full}"
        echo ""
        echo "Commands:"
        echo "  setup           - Set up Python environment"
        echo "  download        - Download 8 CLIP models (paper Table 3)"
        echo "  dataset         - Prepare 50-item dataset"
        echo "  prepare-llava-vqa [images] [examples] - Prepare LLaVA-Bench COCO VQA manifest"
        echo "  prepare-receipt-text [images] [examples] - Prepare TrainingDataPro receipt text manifest"
        echo "  attack [cfg] [n] [s] - Run attack (config, items, steps)"
        echo "  eval-gpt [out] [glob] - Replay GPT-4o and GPT-5-mini eval matrix"
        echo "  eval-llava-vqa-gpt [out] [glob] - Replay GPT VQA eval and summarize categories"
        echo "  eval-receipt-text-gpt [out] [glob] - Replay GPT receipt text eval and summarize question types"
        echo "  llava-vqa-demo [images] [steps] - Prepare and run 5-image LLaVA-Bench VQA attack"
        echo "  llava-vqa-demo-matrix [images] [steps] - LLaVA-Bench VQA attack plus GPT eval"
        echo "  receipt-text-16 [steps] [images] - Receipt text attack with epsilon=16/255"
        echo "  receipt-text-32 [steps] [images] - Receipt text attack with epsilon=32/255"
        echo "  receipt-text-16-matrix [steps] [images] - Receipt text epsilon=16/255 plus GPT eval"
        echo "  receipt-text-32-matrix [steps] [images] - Receipt text epsilon=32/255 plus GPT eval"
        echo "  full-matrix [cfg] [n] [s] [out] - Run 50x300 attack plus GPT eval matrix"
        echo "  full            - Run setup, downloads, attack, and GPT eval matrix"
        echo ""
        echo "Examples:"
        echo "  $0 full-matrix                             # 50 items x 300 steps + GPT eval matrix"
        echo "  $0 llava-vqa-demo 5 300"
        echo "  $0 eval-llava-vqa-gpt outputs/llava_vqa_eps16"
        echo "  $0 receipt-text-16 300 20"
        echo "  $0 receipt-text-32 300 20"
        echo "  $0 attack configs/caption_attack_paper.yaml 50 300"
        echo "  $0 eval-gpt outputs/paper_caltech"
        exit 1
        ;;
esac

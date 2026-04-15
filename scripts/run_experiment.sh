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
export PYTHONPATH=src:$PYTHONPATH
VENV=".venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Step 1: Environment Setup
# =============================================================================
setup_environment() {
    log_info "Setting up environment..."

    # Create venv if not exists
    if [ ! -d "$VENV" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$VENV"
    fi

    # Check activation
    if [ -f "$VENV/bin/activate" ]; then
        source "$VENV/bin/activate"
    else
        log_error "Virtual environment not found at $VENV"
        exit 1
    fi

    # Upgrade pip
    pip install --upgrade pip -q

    # Install requirements
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt -q
    fi

    # Install scipy for Caltech101 dataset
    pip install scipy -q

    log_info "Environment ready"
}

# =============================================================================
# Step 2: Download CLIP Surrogate Models
# =============================================================================
download_models() {
    log_info "Downloading CLIP surrogate models..."

    source "$VENV/bin/activate"

    # Paper Table 3: 8 CLIP models for best transfer (94.4% on GPT-4o)
    # On server with good network: use HuggingFace directly
    python3 << 'EOF'
import open_clip
import os

# 8 CLIP models as per paper (Table 3)
models = [
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-16", "laion2b_s34b_b88k"),
    ("ViT-L-14", "openai"),
    ("ViT-L-14-336", "openai"),
    ("ViT-H-14", "laion2b_s32b_b79k"),
    ("ViT-bigG-14", "laion2b_s39b_b160k"),
    ("RN50", "openai"),
    ("RN101", "openai"),
]

cache_dir = "models/open_clip"
os.makedirs(cache_dir, exist_ok=True)

for model_name, pretrained in models:
    try:
        print(f"Loading {model_name}:{pretrained}...")
        model = open_clip.create_model(model_name, pretrained=pretrained, cache_dir=cache_dir)
        print(f"  ✓ {model_name}:{pretrained}")
    except Exception as e:
        print(f"  ✗ {model_name}:{pretrained} - {str(e)[:60]}")
EOF

    log_info "Model download complete"
}

# =============================================================================
# Step 3: Prepare Dataset
# =============================================================================
prepare_dataset() {
    log_info "Preparing dataset..."

    source "$VENV/bin/activate"

    # Option 1: Use existing dataset
    if [ -d "data/caltech_large" ]; then
        log_info "Using existing Caltech101 dataset"
        export DATASET="caltech_large"
    else
        # Option 2: Prepare 50-item dataset (paper uses 50 images)
        log_info "Preparing Caltech101 dataset (50 items, N=50)..."
        python scripts/prepare_caltech_demo.py --output_dir data/caltech_large --num_items 50 --num_examples 50
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

    CONFIG=${1:-configs/caption_attack_paper.yaml}
    ITEMS=${2:-50}
    STEPS=${3:-300}

    python scripts/run_caption_attack.py \
        --config "$CONFIG" \
        --attack_limit "$ITEMS" \
        --steps "$STEPS"

    log_info "Attack complete"
}

# =============================================================================
# Step 5: Evaluate with SJTU API
# =============================================================================
evaluate_sjtu() {
    log_info "Evaluating with SJTU Qwen3VL API..."

    source "$VENV/bin/activate"

    OUTPUT_DIR=${1:-outputs/paper_caltech}
    API_KEY=${2:-sk-c-EUyeSmz8EfJiqF6ssQVg}

    python scripts/evaluate_sjtu.py \
        --output_dir "$OUTPUT_DIR" \
        --manifest "data/${DATASET:-caltech_large}/manifest.json" \
        --api_key "$API_KEY"

    if [ -f "$OUTPUT_DIR/sjtu_eval.json" ]; then
        log_info "Results saved to $OUTPUT_DIR/sjtu_eval.json"
    fi
}

# =============================================================================
# Step 6: Full Pipeline
# =============================================================================
run_full_experiment() {
    log_info "Starting full experiment pipeline..."

    setup_environment
    download_models
    prepare_dataset
    run_attack "$@"
    evaluate_sjtu

    log_info "Full experiment complete!"
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
    evaluate)
        evaluate_sjtu "${2:-}" "${3:-}"
        ;;
    full)
        shift
        run_full_experiment "$@"
        ;;
    *)
        echo "Usage: $0 {setup|download|dataset|attack|evaluate|full}"
        echo ""
        echo "Commands:"
        echo "  setup           - Set up Python environment"
        echo "  download        - Download 8 CLIP models (paper Table 3)"
        echo "  dataset         - Prepare 50-item dataset"
        echo "  attack [cfg] [n] [s] - Run attack (config, items, steps)"
        echo "  evaluate [out] [key] - Evaluate with SJTU API"
        echo "  full            - Run complete pipeline"
        echo ""
        echo "Examples:"
        echo "  $0 full                                    # Full pipeline"
        echo "  $0 attack configs/caption_attack_paper.yaml 50 300"
        echo "  $0 evaluate outputs/paper_caltech"
        exit 1
        ;;
esac

# Adversarial Attack on VLLMs - Practice Reproduction

A practice reproduction of **arXiv:2505.01050v1** for educational purposes, scaled for consumer GPUs (RTX 4060 8GB / A6000 48GB).

Paper: https://arxiv.org/html/2505.01050v1

## Quick Start

```bash
# 1. Set up environment
bash scripts/run_experiment.sh setup

# 2. Download CLIP surrogate models (HuggingFace allowed on server)
bash scripts/run_experiment.sh download

# 3. Prepare dataset
bash scripts/run_experiment.sh dataset

# 4. Run attack (50 items, 300 steps)
bash scripts/run_experiment.sh attack configs/caption_attack_paper.yaml 50 300

# 5. Evaluate with SJTU API
bash scripts/run_experiment.sh evaluate outputs/paper_caltech
```

## Experiment Settings (Paper-Aligned)

| Parameter | Paper Value |
|-----------|-------------|
| Epsilon | 16/255 ≈ 0.0627 |
| Steps | 300 |
| Temperature | 0.1 |
| Top-K | 10 |
| Positive/Negative Examples (N) | 50 |
| Surrogate Models | 8 CLIP (SigLIP + ViT-H + ConvNeXt) |
| PatchDrop Rate | 20% |
| DropPath Max Rate | 15% |
| Perturbation EMA Decay | 0.99 |

## Dataset

- **caltech_large**: 50 items, 50 positive + 50 negative examples each
- Uses Caltech101 as base (~137MB)

## Evaluation

### SJTU API (Qwen3-VL)
```bash
python scripts/evaluate_sjtu.py \
    --output_dir outputs/paper_caltech \
    --manifest data/caltech_large/manifest.json \
    --api_key YOUR_API_KEY
```

API: `https://models.sjtu.edu.cn/api/v1`

### Local Ollama (Qwen3-VL-4B)
```bash
python scripts/evaluate_ollama.py \
    --output_dir outputs/paper_caltech \
    --model qwen3-vl:4b
```

## Project Structure

```
attack-vllm/
├── src/attack_vlm_repro/
│   ├── attack.py          # Main attack runner
│   ├── config.py          # Configuration
│   ├── losses.py          # Visual contrastive loss
│   ├── surrogates.py      # CLIP surrogate models
│   ├── augmentations.py   # Data augmentation
│   ├── ollama_victim.py   # Local Ollama victim
│   └── eval.py            # Evaluation utilities
├── scripts/
│   ├── run_caption_attack.py    # Main attack script
│   ├── evaluate_ollama.py        # Local evaluation
│   ├── evaluate_sjtu.py          # SJTU API evaluation
│   ├── prepare_caltech_demo.py   # Dataset preparation
│   └── run_experiment.sh         # Full pipeline
├── configs/
│   └── caption_attack_paper.yaml
└── data/
    └── caltech_large/
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- open_clip
- transformers
- pillow
- requests
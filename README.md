# Transferable Adversarial Attacks on Black-Box Vision-Language Models

A method-faithful reproduction of **arXiv:2505.01050v1** for attacking Vision-Language Models (VLLMs), scaled for consumer GPUs (RTX 4060 8GB / A6000 48GB).

**Paper**: https://arxiv.org/html/2505.01050v1

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

# 5. Evaluate with SJTU Qwen3VL API
bash scripts/run_experiment.sh evaluate outputs/paper_caltech
```

Or run the full pipeline:
```bash
bash scripts/run_experiment.sh full
```

## Experiment Settings (Paper-Aligned)

| Parameter | Paper Value | Our Value |
|-----------|-------------|-----------|
| Epsilon | 16/255 ≈ 0.0627 | 0.0627 |
| Steps | 300 | 300 |
| Temperature | 0.1 | 0.1 |
| Top-K | 10 | 10 |
| Positive/Negative Examples (N) | 50 | 50 |
| Surrogate Models | 8 CLIP | 5-8 CLIP |
| PatchDrop Rate | 20% | 20% |
| DropPath Max Rate | 15% | 15% |
| Perturbation EMA Decay | 0.99 | 0.99 |

## Dataset

- **ImageNet-style 50**: 50 items, 50 positive + 50 negative examples each (5050 images)
- Uses Caltech101 as base (ImageNet-like classes, ~137MB vs 150GB ImageNet)
- Or use ImageNet-1K validation set (~6GB)

## Evaluation

### SJTU ModelScope API (Qwen2.5-VL-7B-Instruct)
```bash
python scripts/evaluate_sjtu.py \
    --output_dir outputs/paper_caltech \
    --manifest data/caltech_large/manifest.json \
    --api_key sk-c-EUyeSmz8EfJiqF6ssQVg
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
│   ├── config.py          # Configuration dataclasses
│   ├── losses.py          # Visual contrastive loss
│   ├── surrogates.py      # CLIP surrogate models
│   ├── augmentations.py   # Data augmentation pipeline
│   ├── ollama_victim.py   # Local Ollama victim
│   └── eval.py            # Evaluation utilities
├── scripts/
│   ├── run_caption_attack.py    # Main attack script
│   ├── evaluate_ollama.py        # Local evaluation
│   ├── evaluate_sjtu.py          # SJTU API evaluation
│   ├── prepare_caltech_demo.py   # Dataset preparation
│   └── run_experiment.sh         # Full experiment pipeline
├── configs/
│   └── caption_attack_paper.yaml  # Paper-aligned config
└── data/
    └── imagenet_style_50/        # 50-item dataset
```

## Key Results

| Setting | Proxy ASR | Transfer ASR (Qwen3-VL-4B) |
|---------|-----------|---------------------------|
| 4 surrogates, N=8 | 50% | 0% |
| 5 surrogates, N=50 | 93.75% | 31.2% |
| 8 surrogates, N=50 | TBD | TBD |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- open_clip
- transformers
- pillow
- requests

## Notes

- Server (A6000/better network): Can use HuggingFace for model downloads
- Local: Use ModelScope mirror (`modelscope.cn`) for faster downloads
- SJTU API key required for cloud evaluation
# Adversarial Attack on VLLMs - Practice Reproduction

Practice reproduction of the transferable caption attack in
arXiv:2505.01050v1. The repo contains a runnable Caltech101 engineering
pipeline plus a stricter NIPS 2017/ImageNet manifest path for paper-grade
experiments.

Paper: https://arxiv.org/html/2505.01050v1

## Current Status

Current completed local outputs:

```text
outputs/paper_caltech
outputs/paper_caltech_eps8
```

Current completed full run size:

```text
50 Caltech101 demo items
300 attack steps
8 OpenCLIP surrogates
```

Current full replay result:

| Epsilon | Proxy ASR | GPT-4o ASR | GPT-5-mini ASR |
| --- | ---: | ---: | ---: |
| 16/255 | 50 / 50 = 100% | 48 / 50 = 96% | 48 / 50 = 96% |
| 8/255 | 49 / 50 = 98% | 43 / 50 = 86% | 44 / 50 = 88% |

Full result analysis and uploaded images:

```text
docs/results/full-experiment-analysis.md
docs/results/paper_caltech_eps16_full/
docs/results/paper_caltech_eps8_full/
```

This is a smoke/demo result. It confirms the attack and GPT replay path work
end to end, but it is not a paper-equivalent benchmark result.

Detailed snapshot:

```text
docs/current-reproduction.md
```

Uploaded result images and replay files:

```text
docs/results/paper_caltech_demo/
docs/results/paper_caltech_eps16_full/
docs/results/paper_caltech_eps8_full/
```

Paper reproduction checklist:

```text
docs/paper-reproduction.md
```

## Quick Start

Set up the local environment:

```bash
bash scripts/run_experiment.sh setup
```

Download OpenCLIP surrogate checkpoints:

```bash
bash scripts/run_experiment.sh download
```

Run the full 50 item by 300 step Caltech101 attack and GPT replay matrix with
the default 16/255 epsilon budget:

```bash
bash scripts/run_experiment.sh full-matrix \
  configs/caption_attack_paper.yaml \
  50 \
  300 \
  outputs/paper_caltech
```

This command first checks `data/caltech_large/manifest.json`. If the existing
manifest has fewer than 50 items, it regenerates the Caltech101 demo manifest
with 50 items before launching the attack. That is the path to use when a
previous smoke run only produced 4 items. This dataset regeneration path has
been verified locally to write items `item_00` through `item_49`.

Run the same full experiment with the optional 8/255 epsilon budget:

```bash
bash scripts/run_experiment.sh full-matrix \
  configs/caption_attack_paper_eps8.yaml \
  50 \
  300 \
  outputs/paper_caltech_eps8
```

Replay GPT-4o and GPT-5-mini on existing attack outputs only:

```bash
bash scripts/run_experiment.sh eval-gpt outputs/paper_caltech
```

Run only GPT-5-mini on existing outputs:

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_gpt_eval.py \
  --config configs/techutopia_gpt5mini_caption_eval.yaml \
  --glob 'outputs/paper_caltech/item_*/metrics.json' \
  > outputs/paper_caltech/eval_gpt5mini.jsonl
```

## Architecture

The pipeline is split into five parts:

1. Dataset manifest

   Each item records a source image, source/target labels, label keywords, and
   positive/negative example image paths. The current runnable path uses
   Caltech101. The strict paper path builds a NIPS 2017 captioning manifest
   with ImageNet validation examples.

2. Surrogate ensemble

   The attack uses 8 OpenCLIP CLIP-family surrogates, kept resident on the A6000
   by default with `sequential_surrogates: false`.

3. Attack optimization

   `CaptionAttackRunner` optimizes an L-infinity bounded perturbation with
   projected sign-gradient updates against a visual contrastive loss.

4. Robust augmentations

   The attack applies patch drop, drop path, perturbation EMA, Gaussian noise,
   crop, pad, and JPEG-like transforms. Engineering runs batch stochastic
   augmentation forwards and use a tensor JPEG approximation for speed.

5. Victim replay evaluation

   `scripts/replay_gpt_eval.py` evaluates existing `clean.png` and
   `adversarial.png` pairs. It generates captions for both images and asks the
   configured model to judge target transfer.

## Current Engineering Settings

Config:

```text
configs/caption_attack_paper.yaml
```

Attack:

```text
image_size: 299
epsilon: 0.0627
step_size: 0.004
steps: 300
augmentation_batches: 4
augmentation_forward_batch_size: 4
metrics_interval: 10
temperature: 0.1
top_k: 10
jpeg_prob: 0.2
jpeg_backend: tensor
```

Optional 8/255 epsilon experiment command:

```bash
bash scripts/run_experiment.sh full-matrix \
  configs/caption_attack_paper_eps8.yaml \
  50 \
  300 \
  outputs/paper_caltech_eps8
```

Surrogates:

```text
ViT-H-14:laion2b_s32b_b79k
ViT-H-14:metaclip_fullcc
ViT-H-14:metaclip_altogether
ViT-H-14-378:dfn5b
ViT-B-16-SigLIP:webli
ViT-B-16-SigLIP-384:webli
ViT-L-16-SigLIP-384:webli
convnext_xxlarge:laion2b_s34b_b82k_augreg
```

GPT replay configs:

```text
configs/techutopia_gpt4o_caption_eval.yaml
configs/techutopia_gpt5mini_caption_eval.yaml
```

The TechUtopia endpoint uses:

```text
base_url: https://copilot.techutopia.cn
api_key_env: TECHUTOPIA_API_KEY
```

The local key belongs in `.env`; do not commit secrets. `.env.example` contains
only placeholders.

## Strict Paper Reproduction

The strict config is:

```text
configs/caption_attack_paper_strict_repro.yaml
```

It targets:

```text
1000 NIPS 2017 dev images
50 ImageNet validation positive examples per target class
50 ImageNet validation negative examples per source class
300 attack steps
GPT-4o caption judge evaluation
```

Prepare the manifest with:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_nips2017_caption_manifest.py \
  --nips_csv /path/to/nips2017/dev_dataset.csv \
  --nips_images_dir /path/to/nips2017/images \
  --imagenet_val_dir /path/to/imagenet/val \
  --class_index_json /path/to/imagenet_class_index.json \
  --output_dir data/nips2017_caption_attack \
  --num_examples 50 \
  --limit 1000 \
  --copy_files
```

Then run:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_caption_attack.py \
  --config configs/caption_attack_paper_strict_repro.yaml \
  --verbose
```

## Project Structure

```text
attack-vllm/
├── configs/
│   ├── caption_attack_paper.yaml
│   ├── caption_attack_paper_eps8.yaml
│   ├── caption_attack_paper_strict_repro.yaml
│   ├── techutopia_gpt4o_caption_eval.yaml
│   └── techutopia_gpt5mini_caption_eval.yaml
├── docs/
│   ├── current-reproduction.md
│   ├── paper-reproduction.md
│   └── results/paper_caltech_demo/
├── scripts/
│   ├── prepare_caltech_demo.py
│   ├── prepare_nips2017_caption_manifest.py
│   ├── replay_gpt_eval.py
│   ├── run_caption_attack.py
│   └── run_experiment.sh
└── src/
    ├── attack.py
    ├── augmentations.py
    ├── config.py
    ├── gpt_victim.py
    ├── losses.py
    └── surrogates.py
```

## Requirements

The pinned environment uses CUDA 12.8 PyTorch wheels. Install with:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Core packages:

```text
torch + torchvision cu128
open_clip_torch
transformers
Pillow
httpx
openai
PyYAML
tqdm
```

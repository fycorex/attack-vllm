# Adversarial Attack on VLLMs - Practice Reproduction

Practice reproduction of the transferable VLLM attack in arXiv:2505.01050v1.
The current runnable configs use available surrogate checkpoints from related
model families. They are suitable for engineering validation and smoke/demo
reproduction, but they are not yet an appendix-exact match to the paper’s
surrogate set. Final paper-equivalent claims require checkpoint-by-checkpoint
audit against Table 12.

Paper: https://arxiv.org/html/2505.01050v1

## Current Status

Current completed local outputs:

```text
outputs/paper_caltech
outputs/paper_caltech_eps8
outputs/llava_vqa_eps16
outputs/receipt_text_eps32
```

Current completed caption run size:

```text
50 Caltech101 demo items
300 attack steps
8 OpenCLIP surrogates
```

Current completed VQA/text demo sizes:

```text
LLaVA-Bench COCO VQA: 5 target images, 15 questions, epsilon = 16/255
Receipt text eps32: 5 receipt images, 10 questions, epsilon = 32/255
300 attack steps
8 OpenCLIP surrogates
```

The receipt eps32 output checked here is a demo-scale run with
`attack_limit = 10`. The full receipt-text setting remains 20 receipt images and
40 targeted questions.

Current caption replay result:

| Epsilon | Proxy ASR | GPT-4o ASR | GPT-5-mini ASR |
| --- | ---: | ---: | ---: |
| 16/255 | 50 / 50 = 100% | 48 / 50 = 96% | 48 / 50 = 96% |
| 8/255 | 49 / 50 = 98% | 43 / 50 = 86% | 44 / 50 = 88% |

Current VQA/text replay result:

| Task | Epsilon | Proxy ASR | GPT-4o ASR | GPT-5-mini ASR |
| --- | ---: | ---: | ---: | ---: |
| LLaVA-Bench COCO VQA | 16/255 | 15 / 15 = 100% | 7 / 15 = 46.7% | 10 / 15 = 66.7% |
| Receipt text | 32/255 | 10 / 10 = 100% | 2 / 10 = 20% | 3 / 10 = 30% |

Result analysis and uploaded images:

```text
docs/results/full-experiment-analysis.md
docs/results/vqa-text-demo-analysis.md
docs/results/paper_caltech_eps16_full/
docs/results/paper_caltech_eps8_full/
docs/results/llava_vqa_eps16_demo/
docs/results/receipt_text_eps32_demo/
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
docs/results/llava_vqa_eps16_demo/
docs/results/receipt_text_eps32_demo/
```

Paper reproduction checklist:

```text
docs/paper-reproduction.md
```

Surrogate audit means checking the configured surrogate checkpoint names against
the exact paper appendix/table IDs before making paper-equivalent ASR claims.
The current runnable configs use available OpenCLIP checkpoints from the same
model families, but the strict appendix mapping still needs a final checkpoint
name audit.

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

## LLaVA-Bench COCO VQA Demo

This is the small paper-style VQA benchmark path. It downloads
`lmms-lab/llava-bench-coco` from Hugging Face into `data/raw/hf_cache`, selects
5 target images from the cached full split, and creates 15 attack items:
Conversation, Detail, and Reasoning for each target image. For each item, a
different benchmark image is selected as the source image to perturb.

The strict paper-scale VQA setup is 30 target images / 90 attack items and keeps
the original image size. In this repo, that target setup is documented in
`configs/llava_bench_vqa_paper_strict_repro.yaml`; the commands below remain the
small demo path.

Run the 5-image, 15-item VQA attack with `epsilon = 16/255`:

```bash
bash scripts/run_experiment.sh llava-vqa-demo 5 300
```

Replay GPT-4o and GPT-5-mini VQA evaluation, then summarize Conversation,
Detail, and Reasoning separately:

```bash
bash scripts/run_experiment.sh eval-llava-vqa-gpt outputs/llava_vqa_eps16
```

Run attack and replay evaluation as one command:

```bash
bash scripts/run_experiment.sh llava-vqa-demo-matrix 5 300
```

Generated data:

```text
data/llava_bench_coco_vqa/manifest.json
```

Main output:

```text
outputs/llava_vqa_eps16
```

The judge prompt is the paper's True/False VQA template. The summary file is:

```text
outputs/llava_vqa_eps16/eval_summary.json
```

Latest checked demo result:

| Victim | Overall | Conversation | Detail | Reasoning |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o | 7 / 15 = 46.7% | 2 / 5 = 40% | 1 / 5 = 20% | 4 / 5 = 80% |
| GPT-5-mini | 10 / 15 = 66.7% | 3 / 5 = 60% | 3 / 5 = 60% | 4 / 5 = 80% |

Tracked artifacts:

```text
docs/results/llava_vqa_eps16_demo/
```

## Receipt Text-Recognition

This is an engineering receipt-text demo path built on
`TrainingDataPro/ocr-receipts-text-detection`. It is useful for validating the
crop-and-replace OCR attack pipeline, but it is not the paper's strict text
recognition benchmark. The paper also uses PaddleOCR.

The current demo prep script caches the full Hugging Face dataset under
`data/raw/hf_cache`, reads `annotations.xml`, and builds 40 targeted items from
20 receipt images: two explicit-text questions per receipt. Each target answer
is intentionally incorrect, and positive examples render that incorrect answer
into the annotated receipt region.

Prepare the manifest:

```bash
bash scripts/run_experiment.sh prepare-receipt-text 20 50
```

If the dataset owner namespace resolves differently, set:

```bash
RECEIPT_DATASET_REPO=UniqueData/ocr-receipts-text-detection \
  bash scripts/run_experiment.sh prepare-receipt-text 20 50
```

Run `epsilon = 16/255`:

```bash
bash scripts/run_experiment.sh receipt-text-16 300 20
```

Run `epsilon = 32/255`:

```bash
bash scripts/run_experiment.sh receipt-text-32 300 20
```

Replay GPT-4o and GPT-5-mini on an existing receipt-text output:

```bash
bash scripts/run_experiment.sh eval-receipt-text-gpt outputs/receipt_text_eps16
bash scripts/run_experiment.sh eval-receipt-text-gpt outputs/receipt_text_eps32
```

One-command attack plus replay options:

```bash
bash scripts/run_experiment.sh receipt-text-16-matrix 300 20
bash scripts/run_experiment.sh receipt-text-32-matrix 300 20
```

Generated data:

```text
data/trainingdatapro_receipts_text/manifest.json
```

Outputs:

```text
outputs/receipt_text_eps16
outputs/receipt_text_eps32
```

Latest checked eps32 demo result:

| Victim | Overall | Store | Total | Item |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o | 2 / 10 = 20% | 1 / 5 = 20% | 0 / 4 = 0% | 1 / 1 = 100% |
| GPT-5-mini | 3 / 10 = 30% | 1 / 5 = 20% | 1 / 4 = 25% | 1 / 1 = 100% |

The checked eps32 folder contains 10 items from 5 receipts. Use
`receipt-text-32 300 20` or `receipt-text-32-matrix 300 20` for the full
20-receipt, 40-question setting.

Tracked artifacts:

```text
docs/results/receipt_text_eps32_demo/
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

The engineering caption path currently uses a square resize (`image_size: 299`).
In the paper, the main captioning results are reported at input size 299, while
several captioning ablations are reported at input size 229.

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
│   ├── llava_bench_vqa_eps16.yaml
│   ├── receipt_text_eps16.yaml
│   ├── receipt_text_eps32.yaml
│   ├── techutopia_gpt4o_caption_eval.yaml
│   ├── techutopia_gpt4o_receipt_text_eval.yaml
│   ├── techutopia_gpt4o_vqa_eval.yaml
│   ├── techutopia_gpt5mini_caption_eval.yaml
│   ├── techutopia_gpt5mini_receipt_text_eval.yaml
│   └── techutopia_gpt5mini_vqa_eval.yaml
├── docs/
│   ├── current-reproduction.md
│   ├── paper-reproduction.md
│   └── results/
├── scripts/
│   ├── analyze_receipt_text_eval.py
│   ├── analyze_vqa_eval.py
│   ├── prepare_caltech_demo.py
│   ├── prepare_llava_bench_coco_vqa.py
│   ├── prepare_nips2017_caption_manifest.py
│   ├── prepare_trainingdatapro_receipts_text.py
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

## Citation

This practice reproduction is based on:

```bibtex
@article{hu2025transferable,
  title = {Transferable Adversarial Attacks on Black-Box Vision-Language Models},
  author = {Hu, Kai and Yu, Weichen and Zhang, Li and Robey, Alexander and Zou, Andy and Xu, Chengming and Hu, Haoqi and Fredrikson, Matt},
  journal = {arXiv preprint arXiv:2505.01050},
  year = {2025},
  url = {https://arxiv.org/abs/2505.01050}
}
```

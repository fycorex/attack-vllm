# Current Reproduction Snapshot

Date: 2026-04-16

This document records the current runnable reproduction settings, architecture,
and observed result in this checkout. It is an engineering reproduction snapshot,
not a paper-equivalent benchmark claim.

## Scope

The current completed run uses the local Caltech101 demo manifest:

```text
data/caltech_large/manifest.json
```

The uploaded full result snapshots contain 50 completed attack items for both
16/255 and 8/255 epsilon runs. The wrapper regenerates the Caltech101 demo
manifest when fewer than the requested item count exists, so the 50 item by 300
step run is:

```bash
bash scripts/run_experiment.sh full-matrix \
  configs/caption_attack_paper.yaml \
  50 \
  300 \
  outputs/paper_caltech
```

Use this command for the full 50-item run when a previous smoke run only
produced 4 items. The dataset step has been verified to regenerate
`data/caltech_large/manifest.json` to 50 items before running the attack.

For paper-grade reproduction, use the NIPS 2017/ImageNet manifest flow and
`configs/caption_attack_paper_strict_repro.yaml`; see
`docs/paper-reproduction.md`.

## Architecture

The attack pipeline has five stages:

1. Dataset manifest

   `AttackItem` records source image, source/target labels, keyword aliases,
   and positive/negative visual example paths. The current demo uses Caltech101.
   The strict paper path builds the NIPS 2017 captioning manifest from local
   NIPS dev images plus ImageNet validation examples.

2. Surrogate ensemble

   `CaptionAttackRunner` loads enabled OpenCLIP surrogates from the config.
   Current A6000 settings keep all surrogates resident on CUDA
   (`sequential_surrogates: false`) and use fp16 model weights where supported.

3. Attack optimization

   The adversarial image is optimized with projected sign-gradient updates under
   an L-infinity budget. Each step evaluates the visual contrastive objective
   against target positive examples and source negative examples across the
   surrogate ensemble.

4. Robust augmentations

   The attack applies stochastic patch drop, drop path, perturbation EMA,
   Gaussian noise, crop, pad, and JPEG-like transforms. The engineering config
   batches stochastic augmentations and uses a tensor JPEG approximation to
   reduce CPU round trips.

5. Victim replay evaluation

   `scripts/replay_gpt_eval.py` reuses existing `clean.png`,
   `adversarial.png`, and `metrics.json` files. It evaluates clean and
   adversarial captions, then asks the same model to judge whether the
   adversarial output has shifted toward the target class.

## Current Attack Settings

Primary engineering config:

```text
configs/caption_attack_paper.yaml
```

Runtime:

```text
device: cuda
seed: 42
attack_limit: 50
sequential_surrogates: false
enable_tf32: true
cudnn_benchmark: true
```

Attack hyperparameters:

```text
image_size: 299
epsilon: 16/255 = 0.0627
step_size: 0.004
steps: 300
augmentation_batches: 4
augmentation_forward_batch_size: 4
metrics_interval: 10
temperature: 0.1
top_k: 10
patch_drop_rate: 0.20
drop_path_max_rate: 0.15
perturbation_ema_decay: 0.99
gaussian_prob: 0.5
crop_prob: 0.5
pad_prob: 0.5
jpeg_prob: 0.2
jpeg_backend: tensor
```

This is an engineering convenience resize for the Caltech demo. The strict
paper caption benchmark keeps the original 229 x 299 image size instead; see
`configs/caption_attack_paper_strict_repro.yaml`.

Lower-budget 8/255 epsilon option:

```bash
bash scripts/run_experiment.sh full-matrix \
  configs/caption_attack_paper_eps8.yaml \
  50 \
  300 \
  outputs/paper_caltech_eps8
```

Surrogate ensemble:

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

## GPT Evaluation Matrix

TechUtopia endpoint configs:

```text
configs/techutopia_gpt4o_caption_eval.yaml
configs/techutopia_gpt5mini_caption_eval.yaml
```

Both use:

```text
base_url: https://copilot.techutopia.cn
api_key_env: TECHUTOPIA_API_KEY
task_type: caption
api_mode: chat_completions
success_mode: judge
```

The endpoint requires a browser-like `User-Agent`; this is configured through
`request_user_agent` in the GPT victim config. The local API key is intentionally
kept in `.env` and is not committed.

Run the current matrix:

```bash
bash scripts/run_experiment.sh eval-gpt outputs/paper_caltech
```

Run only GPT-5-mini:

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_gpt_eval.py \
  --config configs/techutopia_gpt5mini_caption_eval.yaml \
  --glob 'outputs/paper_caltech/item_*/metrics.json' \
  > outputs/paper_caltech/eval_gpt5mini.jsonl
```

## Current Result

Current completed attack outputs:

```text
outputs/paper_caltech
outputs/paper_caltech_eps8
```

Tracked uploaded result snapshots:

```text
docs/results/paper_caltech_demo
docs/results/paper_caltech_eps16_full
docs/results/paper_caltech_eps8_full
```

Attack summary:

```text
16/255 proxy success: 50 / 50 = 100%
16/255 average margin gain: 0.6020
8/255 proxy success: 49 / 50 = 98%
8/255 average margin gain: 0.5362
```

Transfer replay results:

| Epsilon | GPT-4o | GPT-5-mini |
| --- | ---: | ---: |
| 16/255 | 48 / 50 = 96% | 48 / 50 = 96% |
| 8/255 | 43 / 50 = 86% | 44 / 50 = 88% |

Detailed analysis:

```text
docs/results/full-experiment-analysis.md
```

Generated replay files are under `outputs/paper_caltech/` and
`outputs/paper_caltech_eps8/`; both output trees remain ignored by Git. The
compact tracked copies are under `docs/results/`.

## Current VQA and Receipt Text Demo Results

Additional paper-style demo outputs are now tracked for the LLaVA-Bench COCO
VQA path and the TrainingDataPro receipt-text path:

```text
docs/results/llava_vqa_eps16_demo
docs/results/receipt_text_eps32_demo
docs/results/vqa-text-demo-analysis.md
```

Replay results:

| Task | Epsilon | Proxy ASR | GPT-4o | GPT-5-mini |
| --- | ---: | ---: | ---: | ---: |
| LLaVA-Bench COCO VQA | 16/255 | 15 / 15 = 100% | 7 / 15 = 46.7% | 10 / 15 = 66.7% |
| Receipt text | 32/255 | 10 / 10 = 100% | 2 / 10 = 20% | 3 / 10 = 30% |

The VQA run uses 5 target images and 15 questions. The receipt eps32 run uses 5
receipt images and 10 questions, so it is still demo-scale rather than the full
20-receipt, 40-question setting.

```text
eval_gpt4o.jsonl
eval_gpt5mini.jsonl
```

## Reproduction Caveats

The current 4-item Caltech result is a smoke/demo result. It demonstrates that
the surrogate attack and GPT replay evaluation path work end to end, but it does
not satisfy the paper's Section 5.1 benchmark requirements.

For paper-grade reporting, the remaining required pieces are:

```text
NIPS 2017 adversarial-learning dev images
ImageNet-1K ground-truth and target labels
50 ImageNet validation positive examples per target class
50 ImageNet validation negative examples per source class
1000 attacked images
GPT-4o judge evaluation on the full completed set
```

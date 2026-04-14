# Small Reproduction: Transferable Adversarial Attacks on Black-Box Vision-Language Models

This project is a **small, method-faithful reproduction** of the paper **Transferable Adversarial Attacks on Black-Box Vision-Language Models** for the **image-captioning setting**, with **local VQA and OCR extensions** added afterward, scaled to fit **1x RTX 4060 8GB**.

Paper reference: https://arxiv.org/html/2505.01050v1

## Scope

### Included in Phase 1
- image-captioning setting only
- delta-only `L_inf` attack
- two small CLIP-like surrogates by default
- visual-only contrastive loss
- multiple positive and negative image examples
- top-k positive selection
- DropPath hook
- PatchDrop hook
- perturbation averaging
- Gaussian noise
- random crop
- random pad
- resize
- differentiable JPEG approximation
- local proxy evaluation

### Added in Phase 2
- local open-source captioning victim evaluation with `Salesforce/blip-image-captioning-base`
- keyword-based caption success check using source/target label aliases from the manifest
- ablation toggles for DropPath, PatchDrop, perturbation averaging, and input-diversity transforms
- ablation suite runner with CSV/JSON aggregation
- local ImageFolder subset builder for ImageNet-style directory trees
- optional third surrogate entry in config, disabled by default for 8GB safety

### Added in Phase 3
- local open-source VQA victim evaluation with `Salesforce/blip-vqa-base`
- manifest support for VQA questions plus source/target answer keywords
- JSON/CSV outputs with VQA success metrics alongside proxy/caption metrics

### Added in Phase 4
- local OCR victim evaluation with Tesseract by default and optional TrOCR support
- manifest support for source/target OCR keywords
- JSON/CSV outputs with OCR success metrics alongside proxy/caption/VQA metrics
- synthetic OCR-word demo builder for local smoke testing

### Explicitly not included
- the paper's full receipt/OCR benchmark
- proprietary API evaluation
- large VLM surrogate ensembles
- full-scale paper numbers

## Hardware-aware defaults
- GPU: single RTX 4060 8GB
- Surrogates: `ViT-B-32` + `ViT-B-16`
- Optional 3rd surrogate: disabled by default
- Input size: `224`
- Attack set: `4` images by default
- Positives / negatives per item: `8 / 8`
- Sequential surrogate loading: enabled by default for memory safety
- Caption victim: loaded sequentially by default for memory safety
- VQA victim: loaded sequentially by default for memory safety
- Tuned default attack schedule: `50` steps, `step_size=0.5/255`, `top_k=1`
- Tuned default transform probabilities: Gaussian/crop/pad/JPEG each at `0.3`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the default OCR victim, also install the local Tesseract binary:

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

## Model download cache

Downloaded models are now saved inside the project-local cache root:
- `models/open_clip/` for surrogate checkpoints
- `models/huggingface/` for caption-victim and VQA-victim checkpoints

The cache root is configured by `paths.model_cache_dir` in the YAML configs, which defaults to `models` in `configs/caption_attack_demo.yaml:2-5` and `configs/caption_attack_phase2.yaml:2-5`.

## Demo data preparation

This builds a tiny local proxy dataset and manifest using CIFAR-10 object categories.

```bash
PYTHONPATH=src python scripts/prepare_cifar10_demo.py \
  --output_dir data/cifar10_caption_attack_demo \
  --num_items 4 \
  --num_examples_per_class 8
```

## Phase 1 demo run

```bash
PYTHONPATH=src python scripts/run_caption_attack.py \
  --config configs/caption_attack_demo.yaml
```

## Phase 2 demo run

```bash
PYTHONPATH=src python scripts/run_caption_attack.py \
  --config configs/caption_attack_phase2.yaml
```

Phase 2 writes both JSON and CSV outputs, including caption-victim fields.

## Phase 3 VQA demo run

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=src python scripts/run_caption_attack.py \
  --config configs/caption_attack_phase2_vqa.yaml
```

Phase 3 adds VQA answers and `vqa_success_rate` to the run summary.

## GPT-4o / GPT-5 reproduction runs

The paper reports `GPT-4o` and `GPT-4o mini`. For the current hosted reproduction path in this repo, the shipped configs target `openai/gpt-4o` and `openai/gpt-5` on GitHub Models, while keeping the same open-ended caption/VQA prompts plus a GPT-4o judge prompt modeled on the paper.

Set a GitHub Models PAT first:

```bash
export OPENAI_API_KEY=...
```

Use the GitHub Models endpoint:
- base URL: `https://models.github.ai/inference`
- model IDs: `openai/gpt-4o`, `openai/gpt-5`

Run all shipped GPT configs:

```bash
PYTHONPATH=src python scripts/run_openai_reproduction.py
```

Or run a single task/model pair:

```bash
PYTHONPATH=src python scripts/run_caption_attack.py \
  --config configs/caption_attack_phase2_gpt4o_caption.yaml

PYTHONPATH=src python scripts/run_caption_attack.py \
  --config configs/caption_attack_phase2_gpt5_vqa.yaml
```

If you are using the official OpenAI API instead of GitHub Models, set `evaluation.gpt_victim.base_url` and `judge_base_url` back to `https://api.openai.com/v1`, and switch model IDs back to the OpenAI-style names you intend to use.

GitHub Models may return `429` on short bursts. The repo now includes retry/backoff controls in the GPT config plus a replay utility for existing adversarial images:

```bash
PYTHONPATH=src python scripts/replay_gpt_eval.py \
  --config configs/caption_attack_phase2_gpt4o_caption.yaml \
  --metrics outputs/caption_attack_phase2_mixed4_localblip/item_03/metrics.json
```

## Phase 4 OCR demo preparation

The OCR path uses a synthetic word-image smoke-test dataset by default.

```bash
PYTHONPATH=src python scripts/prepare_ocr_demo.py \
  --output_dir data/ocr_word_attack_demo \
  --num_items 4 \
  --num_examples_per_class 8
```

## Phase 4 OCR demo run

```bash
PYTHONPATH=src python scripts/run_caption_attack.py \
  --config configs/caption_attack_phase2_ocr.yaml
```

Phase 4 adds OCR text fields and `ocr_success_rate` to the run summary.

## Phase 2 paper-closer run

For a more paper-like local run, prepare an ImageFolder subset with explicit source-target pairs and use the paper-closer config:

```bash
PYTHONPATH=src python scripts/prepare_imagefolder_subset.py \
  --dataset_root /absolute/path/to/imagefolder_root \
  --output_dir data/local_imagefolder_subset \
  --num_examples_per_class 8 \
  --pair_specs_json /absolute/path/to/pairs.json

PYTHONPATH=src python scripts/run_caption_attack.py \
  --config configs/caption_attack_phase2_closer.yaml
```

This profile moves closer to the paper by using:
- `8` attacked items by default
- the optional third surrogate (`RN50`) enabled with sequential loading
- paper-style augmentation probabilities (`0.5` each for Gaussian/crop/pad/JPEG)
- `top_k=4`
- a longer `100`-step attack schedule
- an ImageFolder-style local dataset instead of the CIFAR-10 demo

## Stronger GPU profile

For a more aggressive local run on the RTX 4060 path, use the stronger GPU profile:

```bash
PYTHONPATH=src python scripts/run_caption_attack.py \
  --config configs/caption_attack_phase2_gpu_stronger.yaml
```

This profile uses:
- `5` cached surrogate models (`ViT-B-32`, `ViT-B-16`, `RN50`, `RN101`, `ViT-L-14`)
- `4` stochastic augmentation batches per optimization step
- `120` attack steps with `top_k=4`
- TF32/cuDNN benchmark enabled when CUDA is available

Example `pairs.json` schema:

```json
[
  {
    "source_label": "airplane",
    "target_label": "ship",
    "num_items": 2,
    "source_keywords": ["airplane", "plane", "aircraft"],
    "target_keywords": ["ship", "boat", "vessel"]
  }
]
```

## Phase 2 ablation suite

```bash
PYTHONPATH=src python scripts/run_ablation_suite.py \
  --config configs/caption_attack_phase2.yaml
```

This writes per-variant run folders under `outputs/phase2_ablations/` plus:
- `outputs/phase2_ablations/ablation_summary.json`
- `outputs/phase2_ablations/ablation_summary.csv`

## Local ImageFolder subset builder

For a local dataset arranged like `dataset_root/class_name/image.jpg`:

```bash
PYTHONPATH=src python scripts/prepare_imagefolder_subset.py \
  --dataset_root /absolute/path/to/imagefolder_root \
  --output_dir data/local_imagefolder_subset \
  --num_items 8 \
  --num_examples_per_class 8
```

## Expected outputs

After a run, you should see:
- `outputs/<run_name>/summary.json`
- `outputs/<run_name>/items.csv`
- `outputs/<run_name>/item_XX/clean.png`
- `outputs/<run_name>/item_XX/adversarial.png`
- `outputs/<run_name>/item_XX/delta_vis.png`
- `outputs/<run_name>/item_XX/metrics.json`

If the caption victim is enabled, each `metrics.json` also contains `caption_eval` fields and the run summary includes `caption_success_rate`.

If the VQA victim is enabled, each `metrics.json` also contains `vqa_eval` fields and the run summary includes `vqa_success_rate`.

If the GPT victim is enabled, each `metrics.json` also contains `gpt_eval` fields and the run summary includes `gpt_success_rate`.

## Manifest format

```json
{
  "dataset_name": "demo-dataset",
  "metadata": {
    "note": "optional"
  },
  "items": [
    {
      "id": "item_00",
      "image_path": "/absolute/path/to/source_image.png",
      "source_label": "cat",
      "target_label": "dog",
      "source_keywords": ["cat", "kitten"],
      "target_keywords": ["dog", "puppy"],
      "question": "What is the main object in the image?",
      "source_answer_text": "cat",
      "target_answer_text": "dog",
      "source_answer_keywords": ["cat", "kitten"],
      "target_answer_keywords": ["dog", "puppy"],
      "positive_image_paths": [
        "/absolute/path/to/target_example_0.png"
      ],
      "negative_image_paths": [
        "/absolute/path/to/source_example_0.png"
      ]
    }
  ]
}
```

## What is faithful, scaled down, and approximated

### Faithful-to-paper parts
- delta-only `L_inf` optimization
- multi-surrogate transfer attack
- visual-only contrastive loss
- multiple positive and negative image examples
- top-k positive selection
- DropPath, PatchDrop, perturbation averaging
- Gaussian noise, crop, pad, resize, differentiable JPEG pipeline

### Scaled-down parts
- 2 small CLIP-like surrogates instead of large ensembles
- 224 input size
- 4-image demo attack set
- short optimization schedule
- local demo dataset
- small local captioning victim instead of the paper's broader black-box set

### Approximated parts due to 8GB and local-only constraints
- local embedding-margin proxy evaluation instead of real black-box caption judgments in Phase 1
- local BLIP captioning victim in Phase 2 instead of the paper's proprietary / broader victim set
- CIFAR-10 demo categories instead of the paper's full image-captioning benchmark
- differentiable JPEG implemented as a straight-through approximation
- PatchDrop implemented via patch masking when native support is unavailable
- caption success judged by keyword matching against generated captions, not human evaluation
- GPT runs use small local CIFAR/ImageFolder subsets instead of the paper's larger captioning and LLaVA-Bench benchmarks

## Notes on the Phase 2 caption-victim approximation

The local captioning check is intentionally lightweight:
- victim model: `Salesforce/blip-image-captioning-base`
- evaluation rule: success when the adversarial caption contains a target keyword and, by default, no source keyword
- manifest keyword aliases are used because small captioners often say `car` instead of `automobile`

This is more faithful than the Phase 1 proxy alone, but it is still a local approximation for 8GB hardware.

## Notes on the Phase 3 VQA approximation

The local VQA check is also lightweight:
- victim model: `Salesforce/blip-vqa-base`
- default question: `What is the main object in the image?`
- evaluation rule: success when the adversarial answer contains a target answer keyword and, by default, no source answer keyword
- the shipped smoke validation used `hf-internal-testing/tiny-random-BlipForQuestionAnswering` on CPU to validate the code path quickly

This extends the project toward the paper's broader task scope, but it is still a small local approximation rather than the paper's full VQA benchmark.

## Notes on the GPT reproduction path

The GPT-backed configs are closer to the paper than the local BLIP checks, but they still remain scaled down:
- captioning uses the paper-style caption prompt and a GPT-4o A/B/C/D judge
- VQA uses open-ended answers and a GPT-4o True/False judge prompt matching the paper's evaluation style
- the paper's smaller victim was `GPT-4o mini`; this repo uses `gpt-5-mini` as the current substitute
- the attack side still uses this repo's smaller surrogate ensemble and local manifests, not the paper's full 8-CLIP and benchmark setup

## Plan

See `docs/repro-plan.md` for the phased plan and implementation notes.

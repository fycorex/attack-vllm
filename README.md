# Small Reproduction: Transferable Adversarial Attacks on Black-Box Vision-Language Models

This project is a **small, method-faithful reproduction** of the paper **Transferable Adversarial Attacks on Black-Box Vision-Language Models** for the **image-captioning setting**, with a **local VQA extension** added afterward, scaled to fit **1x RTX 4060 8GB**.

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

### Explicitly not included
- OCR / receipt experiments
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

## Plan

See `docs/repro-plan.md` for the phased plan and implementation notes.

# Paper Reproduction Checklist

Target paper: arXiv:2505.01050v1, Section 5.1 image captioning.

For the current runnable engineering snapshot, including the Caltech demo
settings, architecture, and GPT-4o/GPT-5-mini replay result, see:

```text
docs/current-reproduction.md
```

## Surrogate Audit

A surrogate audit is the checkpoint-name check needed before claiming
paper-equivalent ASR. The current runnable configs use OpenCLIP checkpoints from
the same broad families as the paper's CLIP surrogate ensemble, but a strict
claim requires matching each configured `model_name:pretrained` pair against the
paper appendix/table IDs and confirming the same preprocessing/input size.

## Captioning Dataset Manifest

Prepare these local inputs first:

- NIPS 2017 Adversarial Learning Challenges dev images.
- NIPS dev metadata CSV with image ID, ground-truth label, and target label.
- ImageNet-1K validation images.
- ImageNet class index JSON, such as Keras-style `imagenet_class_index.json`.
- If ImageNet val images are flat, provide either a filename-label CSV or the
  standard validation ground-truth label file.

Example for class-foldered ImageNet validation images:

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

Example for flat ImageNet validation images with the standard ground-truth file:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_nips2017_caption_manifest.py \
  --nips_csv /path/to/nips2017/dev_dataset.csv \
  --nips_images_dir /path/to/nips2017/images \
  --imagenet_val_dir /path/to/imagenet/ILSVRC2012_img_val \
  --imagenet_val_ground_truth /path/to/ILSVRC2012_validation_ground_truth.txt \
  --class_index_json /path/to/imagenet_class_index.json \
  --output_dir data/nips2017_caption_attack \
  --num_examples 50 \
  --limit 1000 \
  --copy_files
```

The generated manifest path is:

```text
data/nips2017_caption_attack/manifest.json
```

## Strict Captioning Attack

Run the strict config after the manifest exists:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_caption_attack.py \
  --config configs/caption_attack_paper_strict_repro.yaml \
  --verbose
```

This config uses:

- `attack.image_size: [229, 299]`
- `attack_limit: 1000`
- 50 positive and 50 negative visual examples per item through the manifest
- 8 OpenCLIP-available CLIP-family surrogates
- GPT-4o caption evaluation with judge mode

## Remaining Audit Items

- Confirm the local NIPS CSV label base. The script defaults to 1-based labels.
- Confirm the ImageNet validation label base. The script defaults to 1-based labels.
- Audit the exact appendix surrogate checkpoint names against available
  OpenCLIP/Hugging Face checkpoints before claiming paper-equivalent ASR.
- Run direct API replay separately for non-GPT-4o victim models.

## LLaVA-Bench COCO VQA

The paper-grade VQA benchmark uses LLaVA-Bench COCO: 30 images, with
Conversation, Detail, and Reasoning questions for each image, for 90 total
attack items. The paper keeps the original image size unless otherwise stated.
For each target data entry, a different benchmark image is selected as the
source image to perturb.

This checkout currently provides:

- A runnable demo path that limits the target image count to 5, creating 15 items.
- A strict config scaffold at `configs/llava_bench_vqa_paper_strict_repro.yaml`
  documenting the paper-scale runtime settings.

The full Hugging Face split is cached under `data/raw/hf_cache` before
selecting target images.

Demo commands:

```bash
bash scripts/run_experiment.sh llava-vqa-demo 5 300
bash scripts/run_experiment.sh eval-llava-vqa-gpt outputs/llava_vqa_eps16
```

Generated data:

```text
data/llava_bench_coco_vqa/manifest.json
```

Output:

```text
outputs/llava_vqa_eps16
```

The category-level replay summary is:

```text
outputs/llava_vqa_eps16/eval_summary.json
```

Latest checked demo result:

| Victim | Overall | Conversation | Detail | Reasoning |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o | 7 / 15 = 46.7% | 2 / 5 = 40% | 1 / 5 = 20% | 4 / 5 = 80% |
| GPT-5-mini | 10 / 15 = 66.7% | 3 / 5 = 60% | 3 / 5 = 60% | 4 / 5 = 80% |

Strict paper config:

```text
configs/llava_bench_vqa_paper_strict_repro.yaml
```

## Text Recognition

The paper's text-recognition benchmark is Omni-OCR, not the local receipt demo.
The strict paper setup is:

- 200 filtered Omni-OCR test cases.
- Keep only cases where both GPT-4o and Claude 3.5 answer the clean image
  correctly.
- Keep the original image size unless otherwise stated.
- Use PaddleOCR to localize text regions.
- Build negative examples from random crops containing the answer bounding box.
- Build positive examples by replacing the relevant text region with the target
  text, then taking random crops from that manipulated image.

This checkout currently provides:

- An engineering receipt-text demo path using
  `TrainingDataPro/ocr-receipts-text-detection`.
- Strict config scaffolds at `configs/omni_ocr_text_paper_eps16.yaml` and
  `configs/omni_ocr_text_paper_eps32.yaml` documenting the paper-scale runtime
  settings and expected manifest path.

Receipt demo manifest prep:


```bash
bash scripts/run_experiment.sh prepare-receipt-text 20 50
```

If the TrainingDataPro namespace is unavailable, use the mirrored owner:

```bash
RECEIPT_DATASET_REPO=UniqueData/ocr-receipts-text-detection \
  bash scripts/run_experiment.sh prepare-receipt-text 20 50
```

Receipt demo `epsilon = 16/255`:

```bash
bash scripts/run_experiment.sh receipt-text-16 300 20
bash scripts/run_experiment.sh eval-receipt-text-gpt outputs/receipt_text_eps16
```

Receipt demo `epsilon = 32/255`:

```bash
bash scripts/run_experiment.sh receipt-text-32 300 20
bash scripts/run_experiment.sh eval-receipt-text-gpt outputs/receipt_text_eps32
```

Receipt demo generated data:

```text
data/trainingdatapro_receipts_text/manifest.json
```

Receipt demo outputs:

```text
outputs/receipt_text_eps16
outputs/receipt_text_eps32
```

Latest checked eps32 result:

| Victim | Overall | Store | Total | Item |
| --- | ---: | ---: | ---: | ---: |
| GPT-4o | 2 / 10 = 20% | 1 / 5 = 20% | 0 / 4 = 0% | 1 / 1 = 100% |
| GPT-5-mini | 3 / 10 = 30% | 1 / 5 = 20% | 1 / 4 = 25% | 1 / 1 = 100% |

This checked eps32 output contains 10 items from 5 receipt images. It is not the
paper's Omni-OCR benchmark and it is not the full 20-receipt engineering
setting either.

The receipt preparation script supports a manual QA override CSV via
`--qa_csv`. Use it when replacing the default annotation-derived store/total
questions with hand-crafted paper-style questions.

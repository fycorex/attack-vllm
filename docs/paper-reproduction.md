# Paper Reproduction Checklist

Target paper: arXiv:2505.01050v1, Section 5.1 image captioning.

For the current runnable engineering snapshot, including the Caltech demo
settings, architecture, and GPT-4o/GPT-5-mini replay result, see:

```text
docs/current-reproduction.md
```

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

- `attack.image_size: 299`
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

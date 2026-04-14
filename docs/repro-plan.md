# Small Reproduction Plan: Transferable Adversarial Attacks on Black-Box Vision-Language Models

Paper: *Transferable Adversarial Attacks on Black-Box Vision-Language Models*  
Source: https://arxiv.org/html/2505.01050v1

## Exact experimental goal for v1
Build a **small, runnable, method-faithful** reproduction of the paper's **image-captioning** attack setting on a **single RTX 4060 8GB**.

The paper's image-captioning setting attacks an image whose main object belongs to source category **A** so that the victim model instead describes target category **B**. The paper evaluates success via caption outputs, but the local reproduction uses scaled local evaluations that fit 8GB VRAM.

## Method components extracted from the paper relevant to image captioning

### Attack formulation
- Optimize only the perturbation **delta** under an **L_inf** bound (`Eq. 2`, `Eq. 4`).
- Use an ensemble of surrogate models and minimize the sum of surrogate losses (`Section 3.1`).
- Use **visual-only** similarity for CLIP-like surrogates (`Eq. 5-7`, `Section 3.3`).

### Loss design
- Use **multiple positive and multiple negative image examples** (`Section 3.3`).
- Use **Top-K positive selection** in the visual contrastive loss (`Eq. 7`).
- The paper motivates visual-only examples because only the surrogate visual embedding needs to align with the victim visual space.

### Model-level transferability techniques
- **DropPath** during optimization (`Section 3.2`).
- **PatchDrop** during optimization, dropping 20% of patches for ViT surrogates (`Section 3.2`).
- **Perturbation averaging** with EMA-style update: `delta_ma <- 0.99 * delta_ma + 0.01 * delta` (`Section 3.2`).

### Data-level transferability techniques
- **Random Gaussian noise**: add Gaussian noise scaled by `epsilon / 4` (`Section 3.4`).
- **Random crop, pad, resize** using a RandomResizedCrop-style crop followed by pad if needed and resize to surrogate input size (`Section 3.4`).
- **Random differentiable JPEG** with random quality in `[0.5, 1.0]` (`Section 3.4`).
- The paper states each augmentation is applied with **50% probability** and separately per surrogate (`Section 3.4`).

### Image-captioning setting details
- The paper uses source and target object categories for image-caption transfer (`Section 4`, `Section 5.1`).
- Full-scale paper experiments use ImageNet/NIPS-style data and VLM caption judging, but that scale is out of scope here.

## Irreducible components to preserve in this reproduction
1. Delta-only optimization under `L_inf`.
2. Multi-surrogate transfer attack.
3. Visual-only contrastive loss.
4. Multiple positive and negative image examples.
5. Top-k positive selection.
6. DropPath hook.
7. PatchDrop hook.
8. Perturbation averaging.
9. Gaussian noise.
10. Random crop.
11. Random pad.
12. Resize.
13. Differentiable JPEG stage.

## What can be safely scaled down
- Number of surrogates: use **2** by default.
- Surrogate size: use **small open_clip CLIP-like models**.
- Attack set size: **4-10 images** initially.
- Positive/negative example count: **8 per side** for demo instead of paper-scale 50.
- Image size: **224** by default.
- Optimization length: **20 steps** by default for a smoke-test-scale run.
- Evaluation: use **local** victims and proxies instead of proprietary VLM APIs.

## Approximations required by 8GB VRAM and local-only scope
- **Victim evaluation approximation**: no proprietary API evaluation; use local embedding-margin proxy and a small local captioning victim instead.
- **Dataset approximation**: use a tiny CIFAR-10-based local demo manifest in Phase 1 and a local ImageFolder subset builder in Phase 2 instead of the paper's larger ImageNet/NIPS benchmark.
- **Surrogate approximation**: use `open_clip` ViT-B-32 and ViT-B-16 rather than the paper's larger and more diverse surrogate pool; an optional third surrogate remains disabled by default.
- **Differentiable JPEG approximation**: use a straight-through compressed forward pass rather than the exact external JPEG implementation from the paper.
- **PatchDrop approximation**: apply patch masking in image/patch space when model-internal patch dropping is unavailable.
- **Caption evaluation approximation**: in Phase 2, use BLIP caption generation plus keyword matching rather than the paper's full victim suite and human/benchmark judging.

## Minimal experiment design for RTX 4060 8GB
- Device: 1x RTX 4060 8GB.
- Default surrogates:
  - `ViT-B-32`, `laion2b_s34b_b79k`
  - `ViT-B-16`, `laion2b_s34b_b88k`
- Optional third surrogate:
  - `RN50`, `openai` (disabled by default)
- Default caption victim:
  - `Salesforce/blip-image-captioning-base`
- Memory-safe mode: sequential surrogate loading and sequential caption-victim loading enabled by default.
- Input size: `224`.
- Attack subset: `4` images.
- Positive examples per item: `8`.
- Negative examples per item: `8`.
- Epsilon: `16/255`.
- Step size: `1/255`.
- Steps: `20`.
- Batch size: `1` attacked image.

## Phases

### Phase 1: Runnable local reproduction
**Status:** implemented.

**Goal:** smallest runnable demo preserving all major attack components.

**Delivered files**
- `docs/repro-plan.md`
- `README.md`
- `requirements.txt`
- `configs/caption_attack_demo.yaml`
- `scripts/prepare_cifar10_demo.py`
- `scripts/run_caption_attack.py`
- `src/attack_vlm_repro/__init__.py`
- `src/attack_vlm_repro/config.py`
- `src/attack_vlm_repro/data.py`
- `src/attack_vlm_repro/surrogates.py`
- `src/attack_vlm_repro/augmentations.py`
- `src/attack_vlm_repro/losses.py`
- `src/attack_vlm_repro/eval.py`
- `src/attack_vlm_repro/attack.py`

### Phase 2: Closer evaluation and ablations
**Status:** implemented as a lightweight 8GB-safe extension.

**Goal:** improve faithfulness while staying within 8GB.

**Delivered work**
- add a small open-source captioning victim for local transfer checks
- add ablation toggles matching paper components
- add optional third small surrogate entry, disabled by default
- add a local ImageFolder subset builder for ImageNet-style directory trees
- add CSV/JSON aggregation for ablation outputs

**Additional files for Phase 2**
- `configs/caption_attack_phase2.yaml`
- `scripts/prepare_imagefolder_subset.py`
- `scripts/run_ablation_suite.py`
- `src/attack_vlm_repro/caption_victim.py`
- `src/attack_vlm_repro/ablations.py`

### Phase 3: Local VQA extension
**Status:** implemented as a post-v1 scope extension.

**Goal:** extend the same attack pipeline to a lightweight local VQA victim without changing the core perturbation optimization method.

**Delivered work**
- add a BLIP VQA victim wrapper for local answer generation
- extend the manifest with question and answer-keyword fields
- add VQA result fields to per-item JSON/CSV outputs and run summaries
- add a Phase 3 YAML config for local VQA runs

**Additional files for Phase 3**
- `configs/caption_attack_phase2_vqa.yaml`
- `src/attack_vlm_repro/vqa_victim.py`

### Phase 4: Local OCR extension
**Status:** implemented as a post-v1 scope extension.

**Goal:** extend the same attack pipeline to a lightweight local OCR victim for text-bearing images without changing the core perturbation optimization method.

**Delivered work**
- add a local OCR victim wrapper supporting Tesseract and optional TrOCR
- extend the manifest with source and target OCR keyword fields
- add OCR result fields to per-item JSON/CSV outputs and run summaries
- add a synthetic OCR-word demo builder and a Phase 4 YAML config for local OCR runs

**Additional files for Phase 4**
- `configs/caption_attack_phase2_ocr.yaml`
- `scripts/prepare_ocr_demo.py`
- `src/attack_vlm_repro/ocr_victim.py`

## Faithful-to-paper vs scaled-down vs approximated

### Faithful-to-paper parts
- delta-only `L_inf` attack
- multi-surrogate transfer objective
- visual-only contrastive loss
- multiple positive/negative image examples
- top-k positive selection
- DropPath, PatchDrop, perturbation averaging hooks
- Gaussian noise, crop, pad, resize, differentiable JPEG pipeline
- source-label to target-label image-caption attack framing

### Scaled-down parts
- 2 small CLIP-like surrogates instead of large ensembles
- 224 input size
- 4-image demo attack set
- 8 positives / 8 negatives per attacked image
- short optimization schedule
- local demo dataset
- one small local captioning victim instead of the paper's broader victim pool

### Approximated parts
- local proxy evaluation instead of real black-box caption judging in Phase 1
- BLIP caption generation with keyword matching instead of the paper's full victim/judging setup in Phase 2
- CIFAR-10-based demo instead of full ImageNet/NIPS setting
- straight-through differentiable JPEG approximation
- patch-space masking approximation for PatchDrop

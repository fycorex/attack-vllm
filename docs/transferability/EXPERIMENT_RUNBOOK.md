# Transferability Experiment Runbook

## Purpose

This runbook records the public, reproducible experiment process used to test
targeted image-embedding transfer. It separates method selection on held-out
open-source encoders from final external API evaluation.

## Fixed rules

- API requests never occur inside image optimization.
- Method selection uses only held-out open-source results.
- Each comparison records clean and adversarial images for the same item.
- Equal-forward comparisons hold total surrogate-sample forwards per item constant.
- API output is retained for audit, but it never changes a selected method.
- Generated images, datasets, model weights, caches, API keys, and private drafts remain local.

## Experiment sequence and decisions

### 1. Establish a common replay and scoring path

The initial repository replayed only one OpenAI-compatible victim. A
provider-neutral replay layer was added so clean/adversarial pairs can be
evaluated after attacks finish, cached by prompt/image/model identity, and
scored with the same task metadata. Conditional success is defined as:

```text
adversarial target success AND NOT clean target success
```

This avoids treating a clean image that already gives the target answer as an
attack success.

### 2. Establish a compute-controlled ensemble baseline

The first completed transfer matrix used caption, VQA, and receipt/OCR; three
seeds; and disjoint OpenCLIP held-out encoders. Single-, two-, and four-proxy
conditions received the same total surrogate-sample forward budget.

Observed outcome: two proxies improved over one across the three tasks; four
proxies did not consistently improve over two. This changed the question from
proxy count to useful representation coverage at a fixed budget.

### 3. Separate global and local alignment diagnostics

Global kernel similarity, neighborhood overlap, neighbor margins, proxy target
distance, and held-out target distance were measured separately. The first
model pool was mostly OpenCLIP-derived, so global CKA had limited dynamic range.
This is treated as a limitation, not a claim that global CKA is generally
irrelevant.

### 4. Test augmentation under equal-forward accounting

The augmentation matrix distinguishes single-sample additive noise,
multi-sample EOT, antithetic sampling, and geometric transformations. Every
condition in the strict screen gets 200 transform samples per item; therefore a
two-sample condition has half as many optimization steps as the single-sample
baseline. This separates a transformation benefit from a benefit caused only by
more model forwards.

### 5. Expand to heterogeneous representation families

The current cycle adds OpenCLIP ViT, ResNet, ConvNeXt, SigLIP, EVA-CLIP, and a
DINOv2 image-only control. DINOv2 is reported separately from the primary
vision-language target score: it tests how far a result extends beyond shared
contrastive vision-language training, rather than serving as a universal target
model.

### 6. Use automatic held-out-only promotion

Screening scores combine primary-target macro ASR, worst-task ASR,
worst-target-family ASR, and a seed-instability penalty. Incomplete trials,
unequal forward counts, and any table containing API-derived fields are
rejected. Promoted conditions are then replicated with seeds 123 and 2026.

### 7. Freeze and externally evaluate

Only after replication are baseline and promoted surrogate/augmentation
candidates frozen. The API replay runs clean and adversarial images with an
identical task prompt and writes tables for target success, conditional ASR,
source suppression, refusals, failures, and paired sample coverage.

### 8. Confirm alignment stability and attack transfer separately

A larger image count is necessary to test whether global alignment rankings
are stable, but it must not be confused with a larger targeted-attack dataset.
The confirmation procedure therefore first measures geometry on 200 unique
caption/natural images. It then runs a separate frozen attack comparison on
the same number of unique images. The attack comparison contains only:

- the single-proxy baseline;
- the best held-out-selected surrogate condition;
- the best held-out-selected augmentation condition;
- their combination.

All four conditions use the same perturbation budget and total
surrogate-sample forward budget. The larger confirmation does not reopen
method selection.

## Current 8-hour full pipeline

| Lane | Work | Selection role |
|---|---|---|
| Cross-family surrogate | 10 proxy-set conditions across three tasks | primary held-out search |
| Augmentation | Nine strict equal-forward conditions across three tasks | primary held-out search |
| Alignment | Joins completed single-proxy outputs with measured geometry | explanation, not selection |
| API replay | Frozen clean/adversarial pairs | external confirmation only |
| 200-image geometry | Unique caption/natural images | alignment robustness only |
| 200-image frozen attack | Four pre-specified conditions | held-out confirmation |
| final API replay | Frozen 200-image attack pairs | external confirmation only |

The screening manifests contain 50 caption images, 20 VQA images, and 19
receipt images. They are sufficient to screen model-family variation, but not
to make a broad global-alignment claim. The later 200-image natural-image
measurement and frozen attack confirmation provide the required larger-sample
check for the caption setting only.

The local live report is generated at:

```text
outputs/transfer_search_cycle_8h_fullpipeline/LIVE_EXPERIMENT_REPORT.md
```

Refresh it with:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/update_live_experiment_report.py \
  --cycle-root outputs/transfer_search_cycle_8h_fullpipeline
```

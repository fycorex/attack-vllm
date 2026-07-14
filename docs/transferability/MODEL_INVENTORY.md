# Model Inventory for the Noise Pilot

## Experimental roles

The noise pilot uses OpenCLIP image encoders for attack and held-out
measurement. These are encoder-level tests; they do not establish end-to-end
MLLM connector or language-decoder behavior. API models are replay-only final
targets and expose no representation measurements.

| Group | Role | OpenCLIP model | Pretrained checkpoint | Explicit family metadata | Input | Notes |
|---|---|---|---|---|---:|---|
| `vit_resnet_diverse` | attack | `ViT-B-32` | `laion2b_s34b_b79k` | ViT, patch 32, LAION-2B contrastive | 224 | Resident in parallel in current search. |
| `vit_resnet_diverse` | attack | `RN50` | `openai` | ResNet-50, OpenAI CLIP contrastive | 224 | Different backbone family from ViT. |
| `vit_resnet_diverse` | held-out | `ViT-B-16` | `openai` | ViT, patch 16, OpenAI CLIP contrastive | 224 | Checkpoint-disjoint from attack set. |
| `vit_resnet_diverse` | held-out | `RN101` | `openai` | ResNet-101, OpenAI CLIP contrastive | 224 | Architecture depth differs from attack RN50. |
| `vit_pretraining_diverse` | attack | `ViT-B-32` | `laion2b_s34b_b79k` | ViT, patch 32, LAION-2B contrastive | 224 | Shared checkpoint across groups, never attack/held-out within a group. |
| `vit_pretraining_diverse` | attack | `ViT-B-16` | `laion2b_s34b_b88k` | ViT, patch 16, LAION-2B contrastive | 224 | Patch/pretraining checkpoint differs. |
| `vit_pretraining_diverse` | held-out | `ViT-L-14` | `openai` | ViT-L, patch 14, OpenAI CLIP contrastive | 224 | Larger held-out ViT. |

The exact checkpoint strings above come from `configs/noise_search.yaml`.
Backbone and training-family labels are explicit experiment metadata, not
inferred during analysis. Parameter counts and measured memory are intentionally
left out until obtained from the loaded model/configuration; they must not be
guessed from names.

## Dataset roles

| ID | Task | Manifest | Pilot / search size | Final maximum |
|---|---|---|---:|---:|
| `caption_caltech` | targeted caption | `data/caltech_large/manifest.json` | 4 / 20 | 50 |
| `llava_vqa` | targeted VQA | `data/llava_bench_coco_vqa/manifest.json` | 20 | 50 |
| `receipt_ocr` | targeted receipt/OCR | `data/trainingdatapro_receipts_text/manifest.json` | 20 | 50 |

## Representation audit fields to add

For every loaded encoder, the measurement manifest must capture exact model and
pretrained IDs, package version, preprocessing transform, input resolution,
output dimension, normalization check, parameter count, measured peak CUDA
memory, role, group, and checkpoint cache identity. Attack and held-out lists
must remain disjoint within every primary comparison.

## API targets

TechUtopia's OpenAI-compatible endpoint is reserved for frozen replay on models
actually returned by its `/v1/models` endpoint (currently expected to include
GPT-4o or GPT-5-mini variants). API keys are environment-only and must never be
written to configs, manifests, logs, or this inventory. Provider model IDs and
resolved IDs must be preserved in replay records.


# Alignment Model Taxonomy

The initial encoder-level study uses the explicit metadata in the surrogate
composition configuration. It must not infer family from checkpoint strings at
analysis time.

| Model role | Checkpoint | Architecture | Pretraining/objective | Input | Representation |
|---|---|---|---|---:|---:|
| proxy candidate | `ViT-B-32:laion2b_s34b_b79k` | ViT-B, patch 32 | LAION-2B / CLIP | 224 | 512 |
| proxy candidate | `ViT-B-16:laion2b_s34b_b88k` | ViT-B, patch 16 | LAION-2B / CLIP | 224 | 512 |
| proxy candidate | `ViT-B-32:openai` | ViT-B, patch 32 | OpenAI / CLIP | 224 | to measure |
| proxy candidate | `RN50:openai` | ResNet-50 | OpenAI / CLIP | 224 | to measure |
| held-out | `ViT-B-16:openai` | ViT-B, patch 16 | OpenAI / CLIP | 224 | 512 |
| held-out | `RN101:openai` | ResNet-101 | OpenAI / CLIP | 224 | 512 |
| held-out | `ViT-L-14:openai` | ViT-L, patch 14 | OpenAI / CLIP | 224 | 768 |

Required negative-control categories:

- same architecture family, different pretraining: LAION versus OpenAI ViT-B;
- different architecture, same CLIP semantic objective: ViT versus ResNet;
- high alignment but weak proxy success, selected after the single-proxy matrix;
- low alignment but strong proxy success, selected after the matrix.

For every loaded model record exact OpenCLIP IDs, package version, parameter
count, preprocessing mean/std, resize/crop behavior, input resolution, output
dimension, normalization, model role, and measured memory. API targets have
only provider/model IDs and observed outputs; no undocumented architecture is
assigned.


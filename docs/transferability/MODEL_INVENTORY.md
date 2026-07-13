# Surrogate Model Inventory

## Audit boundary and sources

This inventory covers every unique OpenCLIP checkpoint currently present in
repository YAML configurations. Architecture fields are taken from the
installed OpenCLIP model registry, and checkpoint/training-family labels are
explicit metadata. Parameter counts and measured memory remain `TBD` until a
checkpoint is loaded and measured; they are not guessed from model names.

The first CPU measurement pilot measured these exact loaded parameter counts:
`ViT-B-32` 151,277,313; `ViT-B-16` 149,620,737; `RN101` 119,688,033; and
`ViT-L-14` 427,616,513. These are counts for the full OpenCLIP model object,
not only its visual tower, and must be labeled accordingly.

Primary local sources:

- repository `configs/*.yaml` for exact checkpoint IDs and configured inputs;
- installed `open_clip.get_model_config()` for image size, patch size, width,
  depth, output dimension, and timm backbone ID;
- checkpoint names for documented pretraining family only; any additional
  dataset claims require a cited upstream model card before publication.

## Currently configured checkpoints

| OpenCLIP model | Pretrained | Backbone / patch | Registry input | Configured input(s) | Objective / training family | Output dim | Current role | Parameters / memory |
|---|---|---|---:|---:|---|---:|---|---|
| `ViT-H-14` | `laion2b_s32b_b79k` | ViT-H, patch 14, 32 layers, width 1280 | 224 | 224 | CLIP contrastive / LAION-2B | 1024 | attack in practical eight | TBD |
| `ViT-H-14` | `metaclip_fullcc` | same ViT-H architecture | 224 | 224 | MetaCLIP / full Common Crawl family | 1024 | attack in practical and strict sets | TBD |
| `ViT-H-14` | `metaclip_altogether` | same ViT-H architecture | 224 | 224 | MetaCLIP / MetaCLIP data family | 1024 | attack in practical eight | TBD |
| `ViT-H-14` | `dfn5b` | ViT-H, patch 14 | 224 | 224 | CLIP-style DFN checkpoint family | 1024 | attack in strict eight | TBD |
| `ViT-H-14-378` | `dfn5b` | ViT-H, patch 14, 32 layers, width 1280 | 378 | 378 | CLIP-style DFN checkpoint family | 1024 | attack in all eight-model sets | TBD |
| `ViT-B-16-SigLIP` | `webli` | timm `vit_base_patch16_siglip_224`, patch 16 | 224 | 224 | SigLIP / WebLI | 768 | attack in practical eight | TBD |
| `ViT-B-16-SigLIP-384` | `webli` | timm `vit_base_patch16_siglip_384`, patch 16 | 384 | 384 | SigLIP / WebLI | 768 | attack in practical eight | TBD |
| `ViT-L-16-SigLIP-384` | `webli` | timm `vit_large_patch16_siglip_384`, patch 16 | 384 | 384 | SigLIP / WebLI | 1024 | attack in practical and strict sets | TBD |
| `ViT-SO400M-14-SigLIP` | `webli` | timm `vit_so400m_patch14_siglip_224`, patch 14 | 224 | 224 | SigLIP / WebLI | 1152 | attack in strict eight | TBD |
| `ViT-SO400M-14-SigLIP-384` | `webli` | timm `vit_so400m_patch14_siglip_384`, patch 14 | 384 | 384 | SigLIP / WebLI | 1152 | attack in strict eight | TBD |
| `ViT-bigG-14` | `laion2b_s39b_b160k` | ViT-bigG, patch 14, 48 layers, width 1664 | 224 | 224 | CLIP contrastive / LAION-2B | 1280 | attack in strict eight | TBD |
| `convnext_xxlarge` | `laion2b_s34b_b82k_augreg` | timm ConvNeXt-XXL | 256 | 224 or 256 | CLIP contrastive / LAION-2B AugReg | 1024 | attack in practical and strict sets | TBD; preprocessing mismatch audit required |

The two configured input sizes for the same ConvNeXt checkpoint are a
reproducibility issue: the registry specifies 256, while several practical
configs force 224. Both may be studied, but they must be separate explicit
conditions rather than silently treated as one model.

## Family metadata

Configuration must store these fields explicitly for experiments:

```text
architecture_family: vit | convnext
objective_family: clip | siglip
pretraining_family: laion2b | metaclip | dfn | webli
checkpoint_id
registry_input_size
attack_input_size
patch_size
```

No runtime code may infer diversity solely by parsing checkpoint strings.

## Held-out pool requirement

The current repository configurations use every listed model as an attack
surrogate and define no disjoint held-out encoder pool. The first pilot must add
explicit, checkpoint-disjoint held-out models before claiming transfer. A
scientifically valid held-out pool should cover at least a ViT CLIP checkpoint
not used for attack and a different-family encoder where cached/hardware-feasible.
Models used to construct an ensemble cannot contribute to its primary held-out
macro ASR.

The new pilot configuration supplies a checkpoint-disjoint held-out pool:
`ViT-B-16:openai`, `RN101:openai`, and `ViT-L-14:openai`. It is valid for the
lightweight proxy sets and must be revalidated if a later attack set adds any
of these checkpoints.

## Memory feasibility gate

Before any eight-model run, measure per-model parameter count, resident fp16
memory, forward/backward peak, load time, and sequential versus parallel peak.
The A6000 has sufficient nominal memory for some combinations but the ongoing
noise experiment currently owns the GPU. Surrogate pilots must not begin until
that run releases the required compute or an explicit resource partition is
chosen.

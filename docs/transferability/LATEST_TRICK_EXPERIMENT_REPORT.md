# Latest Transferability Trick Experiment Report

**Evidence snapshot:** surrogate composition, encoder alignment, and
random-noise/augmentation. Each table distinguishes the attack surrogate from
the held-out evaluation encoder. No primary transfer score is an attack
checkpoint evaluating itself.

## How to read model roles

- **Attack surrogate:** receives gradients during image optimization.
- **Held-out encoder:** not in the optimization loop; evaluates saved
  clean/adversarial pairs afterward.
- **Related-family target:** different checkpoint/model, but related objective
  or architecture. It is transfer, not self-evaluation, but needs its own
  sensitivity analysis.
- **DINOv2 control:** image-only and excluded from primary VLM macro scores.

The attack's white-box proxy success is a diagnostic. The primary transfer
metric is held-out targeted success; API replay is frozen post-selection
confirmation.

## Surrogate-composition

### Attack vs held-out models

| Attack set | Gradient models | Held-out models | Caveat |
|---|---|---|---|
| OpenCLIP ViT baseline | ViT-B/32 LAION-2B | ViT-L/14 OpenAI, RN101 OpenAI, ConvNeXt-B-W, SigLIP2-B/16, EVA02-L/14 | ViT-L is related CLIP/ViT family |
| Single SigLIP | SigLIP-B/16 WebLI | same five models | SigLIP2 is a related-family target |
| Single ConvNeXt | ConvNeXt-B LAION-400M | same five models | ConvNeXt-B-W is related architecture but a different checkpoint/pretraining |
| Two cross-objective | ViT-B/32 LAION-2B + SigLIP-B/16 WebLI | same five models | neither attack checkpoint is held out |

Thus the protocol has no identical proxy/target checkpoint. It does contain
related-family targets, especially SigLIP to SigLIP2, which is why a
leave-SigLIP-family-out macro is still required.

### Completed replicated result

The completed screen/replication cycle used 19 matched items per task, seeds
42/123/2026, and 1,200 surrogate-forward units per item.

| Candidate | Primary held-out macro ASR | Caption | VQA | Receipt/OCR |
|---|---:|---:|---:|---:|
| Single SigLIP-B/16 | 77.2% | 72.6% | 90.0% | 68.9% |
| Single ConvNeXt-B | 65.3% | 64.2% | 74.7% | 56.8% |
| ViT-B/32 + SigLIP-B/16 | 56.1% | 63.7% | 63.7% | 41.1% |
| ViT-B/32 baseline | 48.6% | 53.2% | 56.3% | 36.3% |

The tested equal-loss ensembles do not show monotonic benefit. SigLIP is the
strongest frozen open-source candidate, not yet a universal or closed-model
claim.

A 200-image Caltech101 confirmation is running. Its geometry measurement has
completed: 11 models, 30 proxy-target pairs, 22,000 bootstrap rows. Four
frozen attacks are running: ViT baseline, SigLIP, ViT+translation, and
SigLIP+translation, each at 1,200 forward units per item.

## Encoder-alignment

### Attack vs held-out models

| Role | Models |
|---|---|
| Attack proxy candidates | ViT-B/32 LAION-2B, ViT-B/16 LAION-2B, ViT-B/32 OpenAI, RN50 OpenAI |
| Held-out encoders | ViT-B/16 OpenAI, RN101 OpenAI, ViT-L/14 OpenAI |
| Exact self-evaluation | None |
| Related-family comparisons | ViT-B/16 LAION to ViT-B/16 OpenAI; ViT-B variants to ViT-L/14 |

The completed join has 12 proxy-target pairs, 20 items, three seeds, and 720
item-target rows. It is a small, OpenCLIP-heavy matrix, not a proprietary-MLLM
measurement.

| Predictor of held-out ASR | Spearman | 95% item-clustered interval |
|---|---:|---:|
| Centered linear CKA | -0.025 | [-0.739, 0.594] |
| Uncentered kernel alignment | -0.473 | [-0.742, 0.036] |
| Neighborhood overlap at 1 | 0.685 | [0.132, 0.899] |
| Neighborhood overlap at 5 | -0.025 | [-0.691, 0.578] |
| Neighborhood overlap at 10 | 0.014 | [-0.662, 0.628] |

| Distance relation | Result |
|---|---|
| Delta proxy distance to delta target distance | rho = 0.309, CI [0.168, 0.451] |
| Delta target distance to success | rho = -0.391, CI [-0.527, -0.226] |
| Delta proxy distance to success | rho = -0.184, CI [-0.365, 0.039] |

Global CKA is not established as a predictor in this matrix. NO@1 is promising,
but not yet robust: image deduplication changed receipt NO@1 from approximately
0.925--0.967 to 0.053--0.368, and NO@1 rankings were sample-sensitive.

## Noise-transferability

### Attack vs held-out models

| Surrogate group | Gradient models | Held-out models | Scope |
|---|---|---|---|
| ViT+ResNet diverse | ViT-B/32 LAION-2B + RN50 OpenAI | ViT-B/16 OpenAI + RN101 OpenAI | architecture-diverse CLIP transfer; no identical checkpoint |
| ViT pretraining diverse | ViT-B/32 LAION-2B + ViT-B/16 LAION-2B | ViT-L/14 OpenAI | within ViT/CLIP family; not a cross-family claim |

The saved standalone noise registry contains 17 smoke trials, 126 stage-1
trials, 90 strict equal-forward trials, and 54 final trials.

### Completed noise findings

- **Fixed 100 steps:** Caption with ViT+ResNet, Gaussian EOT with two samples
  and sigma 2/255 improved paired held-out success by **+12.5 pp** over no
  noise (120 pairs; CI [+6.7, +18.3] pp). Antithetic Gaussian and
  variance-matched uniform each gave **+11.7 pp** in that same setting.
- **Strict equal-forward control:** four-sample EOT at 100 steps was compared
  with deterministic no-noise at 400 steps. There is no general positive
  compute-normalized EOT result. On receipt with ViT+ResNet, Gaussian EOT was
  -19.2 pp and antithetic Gaussian -16.7 pp; both intervals were negative.
- **Final 50-item, 300-step study:**

| Dataset / group | Condition | Held-out ASR | Delta vs none | 95% interval |
|---|---|---:|---:|---|
| VQA / ViT+ResNet | Gaussian EOT, 2 samples, sigma 2/255 | 81.0% | +11.3 pp | [+8.0, +15.0] pp |
| VQA / ViT+ResNet | Antithetic Gaussian, 2 samples, sigma 2/255 | 81.0% | +11.3 pp | [+7.7, +15.7] pp |
| Caption / ViT+ResNet | Antithetic Gaussian, 2 samples, sigma 2/255 | 78.3% | +5.7 pp | [+3.3, +8.3] pp |
| Caption / ViT+ResNet | Gaussian EOT, 2 samples, sigma 2/255 | 77.0% | +4.3 pp | [+2.0, +7.0] pp |
| Receipt / ViT+ResNet | Antithetic Gaussian, 2 samples, sigma 2/255 | 62.1% | +2.5 pp | [-1.3, +6.7] pp |

Noise benefits are task- and surrogate-dependent. The result does not support
universal Gaussian improvement, and the strict equal-forward experiment is the
negative control against attributing a fixed-step gain to better
per-forward optimization.

## External API status

Frozen clean/adversarial replay is running for caption, VQA, and receipt/OCR.
The endpoint accepts image inputs on the requested dated GPT-4o route but
returns `gpt-4.1-2025-04-14` as its resolved identifier. The final report will
use the resolved identifier and report conditional ASR, clean false positives,
source suppression, refusals, and failures. There is no aggregate API result
yet.

## Current claims and required work

Supported within this protocol:

- strong single surrogates can beat the tested equal-loss ensembles;
- ensemble count is not monotonic;
- target-conditioned embedding movement is more consistently associated with
  held-out success than global CKA in the small alignment matrix;
- noise effects depend on task, surrogate group, and compute accounting.

Not yet established:

- a universally best surrogate for closed-source MLLMs;
- global CKA is useless or NO@1 is causal;
- Gaussian universally improves transferability;
- a closed-model result before frozen API aggregation.

Next: finish 200-image attacks; report leave-family-out/family-balanced surrogate
macros; extend frozen caption/VQA confirmation; complete API replay without
changing candidates.


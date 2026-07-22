# Experiment Guideline: One-Day Cross-Family Proxy Selector Pilot

## 1. Purpose and Evidence Standard

This document is the binding specification for the new experiment. It turns the
discussion around *Theoretical Analysis on the Transferability of Adversarial
Examples against MLLMs* into a reproducible pilot. The study asks whether
**image-token representation similarity can select one proxy whose targeted VQA
attack transfers well to a held-out, cross-family MLLM**. It is a hypothesis
screening and engineering-validation study, not a statistically powered claim
about transfer laws, scaling, or all MLLMs.

The theoretical motivation is that transfer should improve when proxy and target
share an aligned image representation geometry, and when the perturbation moves
the source image toward target-image representations while moving it away from
the clean-source representation. The experiment therefore separates:

1. **Global model similarity:** centered linear CKA of image-level vectors over
   a common image gallery.
2. **Local image similarity:** token-to-token alignment between an adversarial
   source and target-image token sequences during attack generation.
3. **Behavioral transfer:** targeted answer success on a separately served target
   VLM.

CKA is a selector signal, not an outcome. With six transfer cells, CKA--TASR
correlation, bootstrap intervals, and ranking stability are descriptive only.

## 2. Fixed Experimental Assumptions

- Task: targeted VQAv2 validation VQA. GQA is explicitly deferred.
- Hardware budget: one RTX A6000 (48 GB), no more than 24 hours; the final
  phase may start only if the measured projection is at most 22 hours.
- Seed: `42` is a fixed pseudorandom seed, not a date. The independent CKA
  gallery uses `43`.
- OpenCLIP is not a proxy or selector candidate in this pilot. CLIP means the
  exact Hugging Face checkpoint below.
- All attack, CKA, and screening work uses the experiment branch only. Do not
  modify the existing `main` experiment code or its outputs.
- No model substitution, target gradient, text-fused attack representation,
  text in the attack objective, external LLM judge, Ollama, quantization, or
  silent sample/model failure is permitted.

## 3. Model Bank and Six Required Transfer Cells

| ID | Role | Exact checkpoint | Representation role |
| --- | --- | --- | --- |
| P1 | proxy | `Qwen/Qwen3.5-4B` | visual merger/projector tokens before text fusion |
| P2 | proxy | `openai/clip-vit-large-patch14` | final normalized visual patch tokens, CLS removed |
| P3 | proxy | `google/siglip2-so400m-patch14-384` | final valid visual patch tokens, special/pooled tokens removed |
| T1 | CKA target + vLLM target | `google/gemma-4-E2B-it` | vision-language interface image tokens before text fusion |
| T2 | CKA target + vLLM target | `OpenGVLab/InternVL3_5-2B-HF` | connector-output image tokens before text fusion |

The complete proxy-target matrix is cross-family only:

| Pair | Proxy | Target |
| --- | --- | --- |
| M1 | P1 Qwen3.5-4B | T1 Gemma 4 E2B |
| M2 | P1 Qwen3.5-4B | T2 InternVL3.5-2B |
| M3 | P2 CLIP ViT-L/14 | T1 Gemma 4 E2B |
| M4 | P2 CLIP ViT-L/14 | T2 InternVL3.5-2B |
| M5 | P3 SigLIP2-So400m | T1 Gemma 4 E2B |
| M6 | P3 SigLIP2-So400m | T2 InternVL3.5-2B |

M2 is the mandatory first-contact smoke test. This pilot intentionally excludes
Qwen-scale comparisons, Mistral/Ministral, same-family comparisons, and the
earlier seven-proxy/five-target configuration.

## 4. Adapter Contract and Validation Gate

Every adapter accepts RGB tensors in `[0,1]`, shape `[B, 3, H, W]`, keeps
model-native preprocessing differentiable, freezes all model parameters, and
returns:

```python
ImageTokenOutput(
    global_features: Tensor,  # [B, D], L2-normalized
    local_tokens: Tensor,     # [B, T, D], row-normalized
    token_mask: BoolTensor,   # [B, T]
    metadata: dict,
)
```

Global vectors are masked means of valid image tokens followed by L2
normalization. The original source resolution remains the perturbation domain;
never round-trip an adversarial tensor through PIL/NumPy in the gradient path.
Record module/tensor path, tensor shape, token-mask rule, preprocessing,
checkpoint revision, and processor revision. If a required tap cannot be
validated, write `FAILED_TAP_VALIDATION` diagnostics and stop rather than using
logits, generated answers, arbitrary hidden states, or text tokens.

## 5. Data, Split, and Screening Protocol

Use VQAv2 validation images, questions, and annotations from either official
local JSON + COCO `val2014` images or a complete equivalent Hugging Face source.
Produce disjoint manifests for gallery seed 42 (256 images), gallery seed 43
(256), development (6 image pairs), test (12), candidates, and cached screening
responses. Gallery, development, test, source, and target image IDs must be
globally disjoint.

Each pair stores one source image, one target image, two target-image questions,
their official-normalized canonical answers, category rule, target/source views,
eight hard-negative IDs, and image/artifact hashes. A canonical answer needs at
least 3 of 10 human votes. Categories are `object`, `attribute`, `count`,
`action`, `spatial`, and `relation`: development has one pair/category; test has
two. Classify deterministically from VQAv2 type/text; save the applied rule and
allow manual overrides only as an auditable exception.

Screen a deterministic initial pool of 36 target candidates. Retain a target
only when both T1 and T2 answer both natural target-image questions correctly.
For at most three source candidates per target, retain a source only when
neither target produces either target answer. Stop after 6 dev + 12 test + 4
backups; if insufficient, expand the candidate pool by 12 deterministically.
Reject same IDs/hashes, pHash Hamming distance `<=8`, target answers present in
source VQAv2 answer vocabulary, or target object answers in source COCO labels
when annotations exist. Cache requests and never repeat an identical completed
screening request.

## 6. Proxy-Independent Anchors

For each pair, build exactly nine positive references: target image; translations
`(-4,0)`, `(4,0)`, `(0,-4)`, `(0,4)`; and resize scales `0.95`, `0.975`,
`1.025`, `1.05`. Views return to original size via differentiable center crop or
reflection padding. The local positive pool is identical.

The 17 negative references are clean source, its eight deterministic views, and
eight hard-negative images. Hard negatives share question category, differ in
normalized answer, do not overlap any split, and are ranked without embeddings:
Jaccard overlap of COCO categories and normalized VQA question/answer tokens,
then image ID. Cache only IDs in the manifest and cache per-proxy representations
after selection.

## 7. CKA: Global Representation Similarity

For valid image tokens `E_m(x_i)` and mask `M_i`, define

```text
z_m(x_i) = L2Normalize(sum_r M_ir E_mr(x_i) / sum_r M_ir)
Z_m       = stack_i z_m(x_i)
K_m       = Z_m Z_m^T
H         = I - 11^T/n
K_m^c     = H K_m H
CKA(p,t)  = <K_p^c, K_t^c>_F / (||K_p^c||_F ||K_t^c||_F)
```

Store features FP32, compute CKA CPU FP64, require byte-identical ordered gallery
IDs, and clamp the denominator at `1e-12`. Compute five-model matrices for both
galleries, 100 image-resampling bootstrap repetitions and 95% percentile CIs,
three-proxy rankings per target, and agreement of those rankings across
galleries. Gallery 43 is stability-only; it cannot change the primary selector.
Global CKA may differ across datasets because it measures geometry over the
chosen image distribution; report this rather than treating it as a universal
model constant.

## 8. Attack Objective and Configurations

For proxy `p`, let `g_p` be the global vector and `e_p` the normalized token
sequence. With cosine `s`, global positive/negative pools `P,N`, and `tau=0.10`:

```text
A+(z) = sum_{a in P} exp(s(z, g_p(a))/tau)
A-(z) = sum_{b in N} exp(s(z, g_p(b))/tau)
L_global = -log(A+/(A+ + A-))
M(u,v) = .5[mean_r max_s cos(u_r,v_s) + mean_s max_r cos(u_r,v_s)]
L_local = -mean_{a in P_local} M(e_p(T(x_adv)), e_p(a))
L_src   = cos(g_p(T(x_adv)), g_p(T(x_clean)))
L       = L_global + beta L_local + lambda L_src
```

Average `L` over `R` EOT transforms before one backward pass and minimize it.
`L_src` is source repulsion: minimizing cosine moves the adversarial image away
from its clean source representation. It is not a target-model loss.

| Config | alpha | mu | beta | lambda | R | transforms |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C0 Global | 1/255 | .9 | 0 | 0 | 1 | none |
| C1 +Token | 1/255 | .9 | .25 | .25 | 1 | none |
| C2 +EOT | 1/255 | .9 | .25 | .25 | 2 | integer translation ±4; scale U[.95,1.05] |

All use `epsilon=8/255`, one random start/restart, dev/final steps `15/30`, no
flip, Gaussian/JPEG noise, patch drop, drop path, perturbation EMA, or question
text. C2 uses differentiable reflection-pad translation, antialiased bilinear
resize, and crop/pad back to source shape.

Momentum PGD is fixed as:

```text
x0 = clip_[0,1](x + Uniform[-epsilon, epsilon])
qk = grad_x L(xk); qhat = qk/(mean(abs(qk))+1e-12)
m(k+1) = .9 mk + qhat
x' = xk - (1/255) sign(m(k+1))
x(k+1) = clip_[0,1](x + clip_[-epsilon,epsilon](x' - x))
```

Save lossless 8-bit PNG, reload it, require `L_inf <= 8/255 + 1e-7`, and store
float/reloaded losses and SHA-256. A serialization-budget failure fails the item.

## 9. Development Selection, Replay, and Metrics

For each proxy, run C0--C2 on all six dev pairs (15 steps), replay both targets,
and select the highest mean clean-conditioned pair-macro TASR. Scores within
0.5 percentage points tie; choose lower target-wise standard deviation, then
the simpler order C0, C1, C2. Freeze the recipe before the 12-pair final run.

vLLM is only for target generation/screening, one BF16 server at a time, max
model length 4096, initial GPU utilization .90. Use native implementation first
and record all server/chat-template/revision details. Prompt exactly: `Answer
with a short answer only. Do not explain.` Use temperature 0, top-p 1, maximum
16 tokens, one image/request, and disabled thinking where supported. Evaluate
natural target, clean source, deterministic pair-specific random-noise source,
and adversarial source; preserve raw and official-normalized outputs.

For target correctness `c_tiq` and adversarial answer hit `a_ptiq`:

```text
PairTASR_pti = sum_q c_tiq a_ptiq / sum_q c_tiq
TASR_p->t    = mean_i PairTASR_pti
DeltaTASR    = TASR_attack - TASR_random
```

Report oracle proxy, selector regret, top-2 CKA hit, random selector expectation
over 100 deterministic draws, and a dev-transfer-prior selector. Use 1,000
image-pair cluster bootstrap repetitions; label all inference descriptive.

## 10. Runtime Gate, Outputs, and Required Checks

Phase 0 is M2 only: one valid pair, two questions, C1, ten steps, PNG audit and
T2 replay; it writes `outputs/proxy_selector_pilot/smoke/phase0_report.json` and
must pass before expansion. Complete smoke measures token taps, gradients, PNG
budget, vLLM parsing, model load time, median transform-specific seconds/step,
peak VRAM, and request latency.

If projected total runtime is `<=22 h`, use 12 final pairs; if above, choose a
predeclared category-preserving 8-pair fallback and set `runtime_fallback=true`.
If even that exceeds 22 h, stop after smoke/dev with a partial report. Do not
exceed 24 h or expand scope.

Write all outputs under `outputs/proxy_selector_pilot/`: environment, data,
smoke, representations for both galleries, CKA, dev, final, random_noise, vllm,
summaries, plots, and logs. Required summaries include pair results, the M1--M6
matrix with CKA/TASR/CIs/recipe/runtime/VRAM, selector results, runtime, and a
failures JSONL. Required plots are five-model CKA, CKA-vs-TASR labels, per-target
proxy ranking, gallery-ranking stability, and runtime by proxy/stage.

## 11. Repository Isolation and Delivery Rules

The active branch must be `experiment/proxy-selector-pilot`, created from baseline
`main` commit `28f5b1e60eab8d6030624cc4e0a75e5300fdc181`; the isolated worktree is
`/tmp/attack-vllm-proxy-selector`. Before each expensive phase, fail unless this
branch is active (unless a documented, explicit debug override is passed). Record
baseline branch/commit, experiment branch, worktree path, and dirty status in
`outputs/proxy_selector_pilot/environment/git_baseline.json`, and write active
branch/commit into effective configs and summaries. Never reset, stash, modify,
delete, relocate, or run the pilot from the baseline workspace.

Use separate `.venv-proxy` (PyTorch attack/CKA/data) and `.venv-vllm` (serving)
environments. Respect `HF_TOKEN`, `HF_HOME`, `TRANSFORMERS_CACHE`,
`DATASETS_CACHE`, and `PROXY_SELECTOR_OUTPUT`, but never print or save tokens.
Generated data, model caches, PNGs, outputs, logs, and secrets remain gitignored.
Commit code, configs, tests, templates, and docs in logical conventional commits;
push only the experiment branch and never merge to `main` automatically.

Before claiming readiness, run unit tests for CKA invariances, masked pooling,
local loss symmetry/source-repulsion direction, EOT differentiability, PGD
projection, PNG reload budget, VQAv2 normalization, split disjointness,
hard-negative determinism, synthetic loss reduction, and replay-resume keys.
Also run compileall, proxy-selector dry run and CLI help, runtime planning, and
the pre-existing caption CLI help. GPU/model integration tests are optional and
must be clearly marked; never fabricate their result.

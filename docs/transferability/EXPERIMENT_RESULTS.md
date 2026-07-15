# Transferability Experiment Report

**Report status:** live; completed results are separated from running work.

## Scope and decision rule

This report records a defensive, transferability-focused image-perturbation
study. Candidate selection uses only disjoint open-source held-out encoders.
External API responses are confirmation-only: they are never used to select a
surrogate, augmentation, or hyperparameter.

The primary held-out macro excludes the DINOv2 image-only control. DINOv2 is
reported separately because it is intentionally outside the vision-language
encoder family used by the primary score.

## Completed protocol

| Component | Completed setting |
|---|---|
| Tasks | Caltech caption, LLaVA VQA, receipt/OCR |
| Matched task items | 19 per task |
| Perturbation constraint | L-infinity epsilon = 8/255 |
| Step size | 0.004 |
| Compute comparison | 1,200 surrogate-forward units per item |
| Screening seed | 42 |
| Replication seeds | 123 and 2026 |
| Attack surrogates | OpenCLIP ViT-B/32, RN50, ConvNeXt-B, SigLIP-B/16, EVA02-B/16 and explicit ensembles |
| Primary held-out encoders | OpenCLIP ViT-L/14, RN101, ConvNeXt-B-W, SigLIP2-B/16, EVA02-L/14 |
| Control encoder | DINOv2-B, reported outside the primary macro |

All conditions keep the target construction, positive/negative image examples,
loss, epsilon, step size, and held-out evaluator fixed. Patch-drop, drop-path,
perturbation EMA, and JPEG settings are controlled across the frozen
comparison. Explicit translation replaces the legacy crop path for its own
condition; it is not silently stacked with that path.

## Completed held-out transfer results

The replicated held-out selection ranked the following frozen surrogate
candidates. Values are primary held-out macro ASR across the three tasks and
five primary held-out encoder families.

| Frozen candidate | Primary macro ASR | Caption | VQA | Receipt/OCR |
|---|---:|---:|---:|---:|
| Single SigLIP-B/16 | 77.2% | 72.6% | 90.0% | 68.9% |
| Single ConvNeXt-B | 65.3% | 64.2% | 74.7% | 56.8% |
| ViT-B/32 + SigLIP-B/16 | 56.1% | 63.7% | 63.7% | 41.1% |
| Single OpenCLIP ViT-B/32 baseline | 48.6% | 53.2% | 56.3% | 36.3% |

### Interpretation

- A strong single surrogate currently outperforms the tested equal-loss
  ensembles at equal surrogate-forward budget.
- These results do **not** support a monotonic "more surrogates is better"
  claim.
- SigLIP's advantage is currently an open-source held-out finding, not yet a
  claim about proprietary multimodal models. The held-out set contains a
  SigLIP2 family member, so family-balanced and leave-family-out reporting are
  required before making a broad surrogate claim.

## Completed augmentation results

The augmentation study used the same three tasks, matched items, and
equal-forward accounting. The top frozen augmentation is translation EOT:
two translation samples per optimization step, with the step count reduced so
that total surrogate-forward units remain matched.

| Augmentation condition | Macro ASR | Caption | VQA | Receipt/OCR |
|---|---:|---:|---:|---:|
| Translation EOT | 35.5% | 40.8% | 26.3% | 39.5% |
| Gaussian + translation | 32.5% | 38.2% | 23.7% | 35.5% |
| Single-sample Gaussian | 26.8% | 23.7% | 22.4% | 34.2% |
| No additional noise/geometry | 25.4% | 26.3% | 17.1% | 32.9% |

### Interpretation

- Translation has the clearest equal-forward signal in the completed study.
- Gaussian does not show a universal compute-normalized benefit here.
- The effect is task-dependent: the semantic caption task benefits more than
  OCR should be assumed to benefit. These numbers do not establish a general
  random-noise transfer mechanism.

## Completed representation analysis

The completed alignment branch joins 12 single-proxy transfer trials to
representation metrics. The inferential table contains 720 item-target rows:
four proxies, three seeds, 20 items, and three disjoint held-out targets.

| Predictor of held-out transfer ASR | Spearman | 95% item-clustered bootstrap interval |
|---|---:|---:|
| Centered linear CKA | -0.025 | [-0.739, 0.594] |
| Uncentered normalized alignment | -0.473 | [-0.742, 0.036] |
| NO@1 | 0.685 | [0.132, 0.899] |
| NO@5 | -0.025 | [-0.691, 0.578] |
| NO@10 | 0.014 | [-0.662, 0.628] |

The target-distance path has more consistent support in this matrix:

| Relationship | Spearman / effect | Uncertainty or comparison |
|---|---:|---|
| Delta proxy distance to delta target distance | 0.309 | [0.168, 0.451] bootstrap interval |
| Delta target distance to target success | -0.391 | [-0.527, -0.226] bootstrap interval |
| Mean target-distance change, success vs failure | -0.0502 vs -0.0008 | descriptive item-level comparison |
| Delta proxy distance to target success | -0.184 | [-0.365, 0.039] bootstrap interval |

### Alignment audit and interpretation

- Caltech global centered CKA was high (0.928--0.969) while NO@1 was only
  0.30--0.48, indicating that high global similarity did not imply identical
  nearest-neighbor ordering.
- Alignment rankings vary across data distributions. A hash audit found 50/50
  unique Caltech source images, 15/60 unique LLaVA images, and 19/40 unique
  receipt images before deduplication. Duplicate removal changed receipt NO@1
  from a spuriously high approximately 0.925--0.967 range to 0.053--0.368.
- Centered CKA tracks broader NO@5/NO@10 descriptively, but not NO@1. Only
  NO@1 predicted transfer in this small 12-pair matrix; that signal requires
  the independent 200-image confirmation before it is treated as robust.
- The expanded cross-family analysis has also shown that apparent CKA/NO
  correlations can be driven by the DINOv2 image-only control. Primary-family
  results must therefore be reported both with and without that control.

These are descriptive empirical associations, not a causal proof of a transfer
mechanism and not evidence about proprietary model internals.

## Noise and geometric augmentation status

The standalone `exp/noise-transferability` branch does **not** currently have a
completed saved trial registry from which a defensible result can be reported.
Its tracked result document remains intentionally empty. It would be incorrect
to label the completed geometric-cycle numbers as a completed standalone
Gaussian/random-noise experiment.

The completed augmentation evidence reported above is therefore labeled
precisely:

- **Translation EOT** is a completed geometric-augmentation result.
- **Gaussian** and **Gaussian plus translation** are completed conditions in
  that same equal-forward augmentation matrix.
- No completed result presently establishes a variance-matched
  Gaussian-versus-uniform-versus-Rademacher comparison on the independent noise
  branch.

The missing noise experiment is an explicit gap: run `none`, single-sample
Gaussian, variance-matched uniform, variance-matched Rademacher, and two-sample
Gaussian EOT at equal surrogate-forward budget; then report task-stratified
held-out ASR and the corresponding local-stability diagnostics. API replay must
remain a frozen final confirmation rather than a selector.

## Independent 200-image confirmation

A manually started, resumable 200-image Caltech101 confirmation is independent
of the 19-item screening set. Its representation-measurement phase completed
successfully: 11 models, 30 proxy-target pairs, 55 all-model pairs, 22,000
subsample-bootstrap rows, and 96 ensemble-metric rows. Those measurements have
not yet been joined to completed 200-image attack outcomes, so they are not
interpreted as transfer evidence in this report.

1. **Geometry measurement (complete):** 200 unique natural images, eight
   explicit surrogate sets, five primary held-out encoders, and DINOv2 control.
   It recomputes centered CKA, uncentered alignment, neighborhood metrics,
   kernel discrepancy, and stability across 20/50/100/200-image bootstrap
   subsamples.
2. **Frozen attacks (running):** 200 items, seed 42, 1,200 surrogate-forward units per
   item for each condition:
   - OpenCLIP ViT-B/32 baseline, 300 steps;
   - SigLIP-B/16, 300 steps;
   - OpenCLIP ViT-B/32 plus translation EOT, 150 steps with two geometric samples;
   - SigLIP-B/16 plus translation EOT, 150 steps with two geometric samples.
3. **Held-out evaluation:** the five primary held-out encoder families are
   reported separately and as a macro average; DINOv2 remains a control.

The first attempt stopped before producing attack images because the confirmation
runner emitted explicit EOT fields that the surrogate configuration class did
not yet accept. The configuration and explicit translation implementation were
integrated, compiled, and smoke-tested; the four frozen conditions were then
restarted from the same manifest. No failed attempt output is counted as data.

No result from this stage should be interpreted until all four attack outputs,
constraint checks, forward-count checks, and held-out evaluations are complete.

## External API confirmation

Two frozen replay matrices are currently running in parallel for caption, VQA,
and receipt/OCR:

- surrogate-only candidates: baseline, SigLIP, and ConvNeXt;
- frozen surrogate-plus-augmentation candidates.

The request route is `gpt-4o-2024-05-13` through the configured
OpenAI-compatible endpoint. The capability smoke accepted images, but the
endpoint resolved requests to `gpt-4.1-2025-04-14`; final reporting must use
the resolved model identifier. `gpt-5-mini` remains disabled because the
endpoint did not support it for the required image payload.

The API report will include clean target success, adversarial target success,
conditional ASR, source suppression, refusal rate, failure rate, and raw
auditable outputs. No complete API aggregate exists at this report update.

## Evidence boundaries and next decisions

The completed evidence currently supports the following narrow statements:

- a small, strong surrogate can be preferable to larger equal-loss ensembles
  under matched compute;
- translation is more promising than Gaussian smoothing in the evaluated
  equal-forward setting;
- global CKA alone is insufficiently discriminative in the current primary
  encoder pool, but local NO still needs independent confirmation.

The next experiments required before broader claims are:

1. finish the 200-image confirmation and assess whether frozen rankings hold;
2. run a family-balanced, leave-SigLIP-family-out surrogate comparison;
3. complete the standalone equal-forward Gaussian/uniform/Rademacher noise
   comparison and its local-stability analysis;
4. repeat frozen baseline/SigLIP/translation/combined conditions on larger
   caption and VQA sets; keep receipt/OCR separate;
5. complete the frozen API replay and report its result without revising the
   selected methods.

Generated images, attack outputs, API responses, datasets, checkpoints, keys,
and any private research drafts are intentionally excluded from version
control.

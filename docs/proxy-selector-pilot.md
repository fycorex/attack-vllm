# Proxy-selector pilot

Run only from `experiment/proxy-selector-pilot`. The pilot compares Qwen3.5-4B, CLIP ViT-L/14, and SigLIP2 So400m proxies against Gemma 4 E2B and InternVL3.5-2B targets. All outputs are ignored by Git.

## Setup

```bash
export HF_HOME=/data/hf_cache
export DATASETS_CACHE=/data/datasets_cache
bash scripts/setup_proxy_selector_env.sh attack
bash scripts/setup_proxy_selector_env.sh vllm
```

Checkpoints are read from `model_cache/` in this worktree. Do not store `HF_TOKEN` in configs or output reports.

## Data and services

```bash
bash scripts/run_proxy_selector_pilot.sh prepare-candidates \
  --questions data/vqav2_official/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations data/vqav2_official/v2_mscoco_val2014_annotations.json \
  --images data/vqav2_official/val2014

bash scripts/serve_proxy_selector_target.sh --target T1 --port 8000
bash scripts/run_proxy_selector_pilot.sh screen-data --target T1 --endpoint http://127.0.0.1:8000
# Stop T1, then serve T2 in a separate terminal.
bash scripts/serve_proxy_selector_target.sh --target T2 --port 8000
bash scripts/run_proxy_selector_pilot.sh screen-data --target T2 --endpoint http://127.0.0.1:8000
```

T2 uses vLLM BF16. In the tested CUDA-12.8 environment, vLLM 0.10.2 cannot deserialize Gemma 4; T1 therefore uses the recorded Transformers OpenAI-compatible fallback. It is not quantized.

## M2 first-contact smoke

Generate while no target server occupies the GPU, then replay after restarting T2:

```bash
bash scripts/run_proxy_selector_pilot.sh phase0 --skip-replay
# restart T2
bash scripts/run_proxy_selector_pilot.sh phase0 --replay-only
```

The report is `outputs/proxy_selector_pilot/smoke/phase0_report.json`. It records Qwen token shape, attack speed, loss history, serialized-image L-infinity budget, and four VQA conditions.

## Current scope

The repository provides preparation, screening, Phase 0, CKA extraction, recipe selection, attack generation, replay, plotting, and summarization commands.  Each expensive stage is resumable from its per-item output files.  A successful command only establishes that its stated stage completed; it does not by itself establish transferability.

## Strengthened protocol

CKA is only a selector. It chooses a proxy, after which that proxy is attacked with the same protocol as every other candidate. `attack_S8.yaml` is a strengthened strict 8/255 track; `attack_S16.yaml` changes only the radius. `attack_B16.yaml` is the high-budget attack-strength track: 16/255, 1,000 steps, three deterministic random starts, 4 EOT samples, 9 target views, stronger translation/resize, higher local-token alignment weight, and 8 proxy-independent hard-negative images. Hard negatives are selected before feature extraction by category, a different canonical answer, VQAv2 question-token Jaccard score, and image-ID tie break. B16 is reported separately from the strict selector table.

This remains a strong *single-proxy image-token* attack, not a reproduction of the repository's multi-surrogate, question-conditioned `attack-vllm` method.  A separate question-conditioned likelihood baseline is necessary before attributing a low VQA TASR to the proxy selector rather than to the visual-only objective.

## Literature-grounded single-proxy positive control

Before returning to CKA selection, establish a non-zero targeted-transfer
positive control with one proxy checkpoint per generated image:

```bash
# Untargeted gradient/preprocessing health check; not a targeted result.
bash scripts/run_proxy_selector_pilot.sh ve-sanity --seed 42

# CLIP first, then SigLIP2 if CLIP has no strict replayed hit.
bash scripts/run_proxy_selector_pilot.sh max-strength --proxy P2 --split dev --seed 42
bash scripts/run_proxy_selector_pilot.sh max-strength --proxy P3 --split dev --seed 42
```

`MaxStrengthHierarchicalDirection` is explicitly a paper-inspired combined
positive control, not an exact reproduction. It uses the same proxy's text
path for target-direction guidance, 13 frozen image anchors, 25/50/75/final
visual-token alignment, and optionally reversible 5% (then 2%) pruning of
vision attention/MLP output projections. Its fixed setting is epsilon
16/255, 300 steps, step size 1/255, momentum 1.0, five restarts, and eight
EOT transforms (translation ±8 pixels; resize [0.90, 1.10]). EOT gradients
are accumulated branch-by-branch so SigLIP2-384 fits on one A6000 without
changing the mean-EOT objective.

Replay only after the final PNG exists, with one target server at a time:

```bash
bash scripts/serve_proxy_selector_target.sh --target T1 --port 8000
bash scripts/run_proxy_selector_pilot.sh replay-positive-control \
  --proxy P2 --target T1 --endpoint http://127.0.0.1:8000 --seed 42
```

The replay writes three explicitly separate outcome tiers: **strict targeted**
(the requested target answer under clean/random guards); **controlled
answer-change** (clean and matched random agree, but adversarial differs); and
**raw output-change** (natural target is correct and adversarial differs from
clean even when the target model is intrinsically prompt/noise unstable).
Report each hit count over clean-valid images; never rename either change rate
as targeted TASR. If no method achieves two strict targeted pairs on one target
(or one on each) with a seed-43 reproduction, publish the diagnostic traces
rather than claiming targeted transfer.

# Noise Transferability Experiment Plan

## Claim and primary endpoint

The primary claim is supported only if randomized optimization improves paired
held-out macro ASR over `none` across seeds and retains improvement under equal
surrogate-forward budget. Attack-surrogate proxy success is diagnostic only.

## Search

`configs/noise_search.yaml` defines datasets, disjoint attack/held-out model
groups, stages, seeds, noise distributions, sigma, and EOT samples. The runner
uses stable trial IDs and skips completed trials. Stage 0 validates mechanics;
Stage 1 searches on open-source held-out models. Final candidates must be frozen
from Stage 1 before the final dataset test and API replay.

For equal-forward comparisons, an EOT run with `k` samples is paired with a
no-noise run using `k` times as many optimization steps, or its results are
reported explicitly as `k` times the surrogate forwards. Both views are kept.

## Required outputs

- per-dataset and per-surrogate-group winner;
- global robust default across conditions;
- paired baseline delta and bootstrap interval;
- per-model ASR and seed variation;
- total forwards, wall time, peak CUDA memory, effective noise moments and
  saturation rate;
- frozen GPT-4o and GPT-5-mini confirmation after open-source selection.

No API response may alter the selected configuration.

## Commands

Inspect the initial matrix without running it:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_noise_search.py \
  --spec configs/noise_search.yaml --stage stage0 --plan
```

Run the complete resumable search and validation loop with four concurrent
trials on the A6000:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_adaptive_noise_search.py \
  --spec configs/noise_search.yaml \
  --root outputs/noise_search \
  --until final \
  --workers 4
```

To stop after open-source selection, use `--until stage1`. Re-running the same
command resumes completed attack and held-out phases independently.

After candidates are frozen, caption API replay uses:

```bash
TECHUTOPIA_API_KEY=sk-test PYTHONPATH=src .venv/bin/python \
  scripts/replay_multimodel_eval.py \
  --config configs/techutopia_transferability.yaml \
  --output-dir <frozen-attack-output> \
  --result-dir <api-result-dir> --dry-run
```

VQA and receipt/OCR use `configs/techutopia_transferability_vqa.yaml`. Replace
`--dry-run` with `--allow-real-api --resume --max-requests <reviewed-limit>`
only after inspecting the request estimate.

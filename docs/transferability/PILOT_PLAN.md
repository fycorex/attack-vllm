# Surrogate-Composition Pilot Plan

## Decision objective

Test whether ensembles hedge blind proxy selection and whether useful,
nonredundant target-kernel coverage predicts transfer better than ensemble size
or extra forward compute.

## Stage 0A: measurement only

Use a fixed 20-image natural-image sample and no API calls.

1. Resolve explicit model metadata and reject attack/held-out overlap.
2. Capture normalized representations and preprocessing metadata.
3. Compute centered CKA and uncentered kernel alignment for all model pairs.
4. Compute `NO@1/5/10`, neighbor-margin distributions, kernel norms, Gram rank,
   condition number, inter-proxy alignment, EA, and ensemble discrepancy.
5. Identify candidate high-redundancy, low-redundancy, high-target-alignment,
   and diverse-but-low-target-alignment sets without looking at API outcomes.

Gate: centered and uncentered quantities are separate; EA inversion diagnostics
are present; every ensemble has disjoint held-out targets.

Initial command actually run on CPU:

```bash
PYTHONPATH=src .venv/bin/python scripts/measure_surrogate_geometry.py \
  --config configs/surrogate_composition.yaml \
  --manifest data/caltech_large/manifest.json \
  --sets single_reference two_homogeneous \
  --limit 20 \
  --output outputs/surrogate_geometry/stage0a_caltech_20 \
  --device cpu --batch-size 8 --cache-dir models/open_clip
```

This measurement does not run an attack and does not call an API.

## Stage 0B: compatibility smoke

Use four fixed items, 20 steps, seed 42, one epsilon, and no API. Run single
proxies and the smallest two-model sets. Verify L-infinity constraints,
reproducibility, per-model loss/forward accounting, output schema, and that the
legacy config path is unchanged when composition config is absent.

## Stage 1: single-proxy transfer matrix

Use 20 items, 100 steps, seeds 42/123/2026, and held-out open-source targets.
Run each feasible proxy independently. Produce pairwise proxy-to-target transfer
and theory-metric tables. This establishes blind single-proxy distributions and
prevents an ensemble-only result from hiding weak components.

## Stage 2: controlled ensembles

Construct explicitly documented sets:

```text
single_reference
two_homogeneous
four_homogeneous
four_low_redundancy
four_high_target_alignment
four_diverse_low_target_alignment
eight_current
leave_one_family_out
incremental_addition
random_matched_size
```

Initial attacks retain the existing equal-loss mean. Do not add adaptive
weighting. Compare at equal optimization steps and equal total
surrogate-forward budget.

## Stage 3: claim tests

For each held-out target test:

1. Ensemble versus worst, random, mean, best, and matched-compute single proxy.
2. ASR versus ensemble size, mean target alignment, redundancy, EA, ensemble
   discrepancy, and forward count.
3. Low versus high redundancy at approximately matched target alignment.
4. Incremental/leave-one-out `delta EA`, `delta discrepancy`, and `delta ASR`.
5. Centered versus uncentered alignment rankings and predictive correlations.

Use paired item bootstrap intervals and target-level macro averages. A result
that only beats the worst single proxy demonstrates a worst-case hedge, not a broad
ensemble superiority claim.

## Stage 4: frozen final validation

Select configurations from held-out results only. Run 50 items, 300 steps,
epsilon 8/255, frozen seeds/settings, then replay existing pairs through the
shared API evaluator. API results cannot trigger new ensemble selection.

## Required outputs

```text
proxy_to_target_transfer_matrix.csv
ensemble_theory_metrics.csv
incremental_proxy_addition.csv
leave_one_out_results.csv
equal_compute_results.csv
ensemble_claim_summary.md
```

Each run records Git SHA and dirty state, config/manifest hashes, item IDs,
surrogate/held-out IDs, seed, attack parameters, forwards, runtime, and peak
memory. Outputs, caches, datasets, API keys, and the private paper are never
committed.

## Stop conditions

- Stop if attack and held-out sets overlap.
- Stop if preprocessing or configured input size is ambiguous.
- Do not interpret EA when Gram conditioning/regularization is missing.
- Do not claim diversity from architecture labels without measured kernel
  redundancy.
- Do not launch the full matrix before measurement and four-item smoke gates
  pass.

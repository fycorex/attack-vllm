# Encoder Alignment Pilot Plan

## Stage A: representation measurement

Reuse the fixed natural-image measurement protocol from the surrogate branch
at multiple sample sizes. The current manifests provide 50 unique Caltech
images, 15 unique LLaVA images, and 19 unique receipt images after SHA-256
deduplication. Compute
centered CKA, uncentered alignment, `NO@1/5/10`, gamma-margin distributions,
kernel norms, dimensions, and preprocessing metadata.

Gate: model IDs and item sets match across encoders; both alignment definitions
are explicit; no attack or API is called.

## Stage B: reuse single-proxy transfer outputs

Wait for surrogate Stage 0B and single-proxy Stage 1. Read their output root by
explicit path. Join item-level attack metrics, held-out replay, alignment
measurements, and model metadata without modifying the source files.

Required outputs:

```text
pairwise_alignment_metrics.csv
pairwise_transfer_matrix.csv
cka_transfer_analysis.csv
no_transfer_analysis.csv
dproxy_dtarget_analysis.csv
alignment_claim_summary.md
```

## Stage C: empirical claim checks

1. CKA versus pairwise ASR/margin, with sample-size sensitivity.
2. NO versus CKA and transfer at several k.
3. Per-item `delta D_proxy -> delta D_target`.
4. `delta D_target -> target success`.
5. Family-stratified and attack-effectiveness-controlled comparisons.
6. Empirical gamma distribution and neighbor-rank stability.

Use paired bootstrap intervals. Do not pool targets without per-target results.

## Decision gate

Do not add a full MLLM, connector hook, or DynVLA until
`ENCODER_ALIGNMENT_DECISION.md` is completed. Escalation is justified only if
encoder alignment fails to explain transfer, similar encoders transfer very
differently, API behavior systematically disagrees, or connector/backbone
behavior cannot be identified from encoder measurements.

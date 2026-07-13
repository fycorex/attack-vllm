# Surrogate Ensemble Empirical Status

This document reports public experiment state and observed transfer metrics for
single and ensemble surrogates. It does not depend on a private manuscript.

## Measurement coverage

Four lightweight candidate proxies and three disjoint held-out targets were
measured on matched Caltech caption, LLaVA VQA, and receipt/OCR samples. The
reported geometry includes centered linear CKA, uncentered normalized kernel
alignment, Neighborhood Overlap, inter-proxy redundancy, explained alignment,
kernel discrepancy, kernel norms, and Gram conditioning.

Centered and uncentered alignment produce different rankings. Rankings also
vary across datasets, so no single geometry ordering is treated as universal.

## Equal-step attack result

All nine equal-step trials passed completeness, finite-value, L-infinity,
disjoint-target, and forward-accounting checks. Three-seed held-out macro ASR:

| Surrogate set | Models | Forwards/item | Mean held-out macro ASR |
| --- | ---: | ---: | ---: |
| `single_reference` | 1 | 400 | 0.2111 |
| `two_homogeneous` | 2 | 800 | 0.5222 |
| `four_lightweight_mixed` | 4 | 1600 | 0.5667 |

The ensemble advantage is large under equal optimization steps, but compute is
not matched in this table. Equal-forward trials must complete before attributing
the gain to ensemble composition rather than additional surrogate forwards.

## Evidence boundary

- attack/API separation is preserved;
- held-out open-source results are the only candidate-selection signal;
- API outcomes are not used for tuning;
- equal-step evidence is not presented as an equal-compute conclusion;
- cross-dataset validation uses unique source-image manifests.

No formal API evaluation has been used in the results above.

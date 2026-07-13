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
not matched in this table.

## Equal-forward attack result

All nine equal-forward trials passed the same validation checks, with exactly
1,600 surrogate-sample forwards per item. Three-seed held-out macro ASR:

| Surrogate set | Models | Forwards/item | Mean held-out macro ASR |
| --- | ---: | ---: | ---: |
| `single_reference` | 1 | 1,600 | 0.6889 |
| `two_homogeneous` | 2 | 1,600 | 0.8667 |
| `four_lightweight_mixed` | 4 | 1,600 | 0.5778 |

Paired differences use an item-clustered bootstrap that retains all seed and
held-out-target observations within each of 20 unique source images. Relative
to the single reference, the two-model set improves ASR by 17.8 percentage
points (95% CI [8.3, 28.3]), while the four-model set reduces ASR by 11.1 points
(95% CI [-21.7, -2.8]). The four-model set is 28.9 points below the two-model set
(95% CI [-42.2, -16.7]).

This is positive evidence for composition-sensitive transfer, not for monotonic
scaling with surrogate count. All three methods are frozen for cross-dataset
validation so the underperforming four-model set remains a negative control.

## Evidence boundary

- attack/API separation is preserved;
- held-out open-source results are the only candidate-selection signal;
- API outcomes are not used for tuning;
- equal-step and equal-forward evidence are reported separately;
- cross-dataset validation uses unique source-image manifests.

No formal API evaluation has been used in the results above.

# Noise Smoothing Empirical Status

This document reports post-hoc Gaussian-smoothing measurements on saved attack
outputs. It contains only experiment definitions, quality checks, and observed
results; no private manuscript mapping is required to reproduce the analysis.

## Measurements

For each proxy-target pair, the analysis reports:

- the Jensen gap between the expected noisy embedding distance and the distance
  of the mean noisy embedding;
- centered linear CKA and uncentered normalized kernel alignment;
- kernel discrepancy before and after smoothing;
- normalized residual inner products;
- target-specific held-out ASR and item-paired confidence intervals.

Raw mean embeddings and renormalized mean embeddings are kept as distinct
definitions. Centered and uncentered kernels are never silently mixed.

## Data quality

The preregistered matrix contains 12/12 exact directory IDs: two attack modes,
three seeds, and two held-out targets. It produces 240 item-level Jensen rows,
48 residual/kernel rows, and six target-specific paired attack comparisons.
There are no duplicate rows at their intended grain and no NaN or Inf values.

The gate consumes `configs/noise_theory_caltech12.txt`, excludes implementation
smokes and unrelated output directories, and requires the exact run matrix.
Target-specific ASR is reconstructed from each original
`heldout_summary.json`. The overall interval resamples the 20 item IDs as
clusters across seeds and targets because the same natural images recur.

## Transfer result

For the equal-step comparison between no noise and two-sample Gaussian EOT at
sigma `2/255`, the held-out target ASR delta is positive in all six seed-target
cells. The item-clustered macro effect is `+0.125` (12.5 percentage points),
with a 95% bootstrap interval of `[+0.050, +0.2083]`. Individual cell effects
range from `+0.05` to `+0.20`.

This is evidence of transfer improvement for the tested Caltech caption setup.
It is not yet a general compute-independent result: the broader equal-forward
matrix did not preserve the equal-step advantage.

## Smoothing diagnostics

All 240 item-level Jensen gaps are nonnegative (minimum `0.0010`, mean `0.0040`,
maximum `0.0112`). Across all 12 runs, smoothing increases normalized alignment
and reduces kernel discrepancy under all four reported kernel definitions.

For raw-mean centered kernels, mean alignment delta is `+0.0157` and mean
discrepancy delta is `-0.3001`, with bootstrap intervals excluding zero. The
normalized residual inner products are not close to zero, and their signs
differ between centered and uncentered definitions. These observations show
that the kernel definition materially affects mechanistic interpretation.

With only six target-seed cells, the Spearman correlation between alignment
delta and ASR delta is `0.086`; discrepancy delta versus ASR delta is `-0.486`.
These small-sample correlations are descriptive and not causal evidence.

## Limitations

- one caption dataset;
- one surrogate group;
- two held-out encoders;
- equal-step improvement but no general equal-forward improvement;
- insufficient evidence to attribute the transfer gain to one kernel statistic.

No API outcome was used for method selection or for the results above.

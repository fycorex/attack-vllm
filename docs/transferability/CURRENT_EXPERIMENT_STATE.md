# Current Experiment State

Snapshot date: 2026-07-13 UTC.

- Noise implementation commit: `b50fda262ab3be3dd4df49a57ac99666d280f4c1`.
- Noise trials: Stage 0 17/17, Stage 1 126/126, equal-forward 90/90.
- Equal-forward integrity: 90 unique complete trial IDs, 20 items each, no
  missing attack or held-out summaries, and 3200 forwards per item in every
  condition.
- The original adaptive runner automatically entered final validation after
  equal-forward; eight GPU workers are active. This sidecar remains read-only
  with respect to those outputs.
- Its manifest was promoted from Stage 1 before equal-forward evidence was
  available. The run is exploratory, not a gate-compliant frozen final, and
  cannot determine API candidates.
- Post-hoc smoothing diagnostics are complete for the exact 12/12
  preregistered proxy/target/seed conditions. The corrected target-specific
  aggregate contains six 20-item paired cells and excludes the implementation
  smoke. Gaussian EOT improves equal-step held-out ASR by 12.5 percentage
  points on average, with item-clustered 95% CI `[+5.0, +20.8]`; this does not
  override the negative equal-forward matrix.
- Surrogate Stage 0B completed 4/4 trials and Stage 1 completed 12/12. All
  outputs passed finite-value, L-infinity, completeness, disjoint-target, and
  exact-forward gates.
- Surrogate ensemble equal-step validation completed 9/9 valid trials.
  Three-seed held-out macro ASR is 0.2111 for `single_reference`, 0.5222 for
  `two_homogeneous`, and 0.5667 for `four_lightweight_mixed`. Forward units per
  item are respectively 400, 800, and 1600, so these are strong equal-step
  ensemble signals but not equal-compute claims. Stage 2 equal-forward is now
  running (nine planned trials).
- Alignment taxonomy and the read-only Stage 1 empirical join are complete.
  Across 12 proxy-target pairs, NO@1 was more predictive of transfer than global
  centered CKA; the item-level `delta D_proxy -> delta D_target -> success`
  chain had the expected direction. Confidence intervals remain wide at the
  pair level.
- The OpenAI-compatible gateway has one capability-verified image route,
  `gpt-4o-2024-05-13`. One clean/adversarial interface smoke completed, but it
  is not a frozen-candidate API evaluation and showed no transfer on that one
  pair. `gpt-5-mini` was advertised but not routable at the time of testing.
- API replay now preserves overlapping item IDs across candidates, deduplicates
  identical clean requests, and reports each frozen candidate separately per
  API model. Formal API evaluations completed: none.

Prepared analysis code is not counted as empirical evidence. The post-hoc
smoothing matrix is complete; generality across datasets and compute budgets
remains unresolved.

A four-item smoothing implementation smoke has been measured. It is retained
only as a pipeline check; its residual statistics are not promoted to empirical
claim status.

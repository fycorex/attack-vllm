# Current Experiment State

Live update: 2026-07-13 UTC.

## Completed

- provider-neutral API replay with explicit opt-in, caching, resume, and frozen-candidate verification;
- surrogate geometry measurement on caption, VQA, and receipt/OCR samples;
- 4/4 compatibility smokes;
- 12/12 single-proxy transfer trials;
- 9/9 equal-step ensemble trials;
- 9/9 equal-forward ensemble trials with exact 1,600-forward accounting;
- held-out-only freeze of the single-, two-, and four-model methods;
- read-only encoder-alignment join;
- exact noise-smoothing post-hoc matrix.

Equal-forward three-seed held-out macro ASR is 0.6889 for the reference single
proxy, 0.8667 for the two-proxy set, and 0.5778 for the four-proxy set. At matched
compute, the two-proxy set improves over the reference while the four-proxy set
is worse. Surrogate count alone therefore does not explain transfer.

## Running

A 27-trial, three-dataset validation is running on 19 unique source images per
dataset and three seeds. Generated attacks are evaluated only on disjoint
held-out open-source encoders during selection.

## Prepared next steps

1. Validate all cross-dataset outputs and forward counts.
2. Compare method effects separately for caption, VQA, and receipt/OCR.
3. Freeze one task-specific baseline/candidate manifest per dataset.
4. Produce an API request-count dry run.
5. Replay frozen clean/adversarial pairs only after explicit real-API opt-in.

No private draft, API key, model checkpoint, dataset, generated attack output,
or large cache is intended for Git tracking.

# Current Experiment State

Live update: 2026-07-13 UTC.

## Completed

- provider-neutral API replay with explicit opt-in, caching, resume, and frozen-candidate verification;
- surrogate geometry measurement on caption, VQA, and receipt/OCR samples;
- 4/4 compatibility smokes;
- 12/12 single-proxy transfer trials;
- 9/9 equal-step ensemble trials;
- read-only encoder-alignment join;
- exact noise-smoothing post-hoc matrix.

Equal-step three-seed held-out macro ASR is 0.2111 for the reference single
proxy, 0.5222 for the two-proxy set, and 0.5667 for the four-proxy set. Forward
counts differ, so equal-forward validation is the active decision gate.

## Running

Nine equal-forward surrogate trials are running. Generated attacks are evaluated
only on disjoint held-out open-source encoders during selection.

## Prepared next steps

1. Validate all equal-forward outputs and forward counts.
2. Freeze baseline and leading alternatives using held-out results only.
3. Run a 19-item matched cross-dataset matrix using unique source images.
4. Produce an API request-count dry run.
5. Replay frozen clean/adversarial pairs only after explicit real-API opt-in.

No private draft, API key, model checkpoint, dataset, generated attack output,
or large cache is intended for Git tracking.

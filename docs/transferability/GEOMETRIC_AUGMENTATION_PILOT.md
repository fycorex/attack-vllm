# Geometric Augmentation Pilot

This is a separate follow-up stage. It must not be added to the active additive
noise search or its promotion logic.

## Question

Does stochastic geometric augmentation improve transfer through local
transformation invariance, rather than through the Gaussian-smoothing mechanism
observed for additive-noise smoothing?

## Frozen pilot

- datasets: `caption_caltech` and `receipt_ocr`;
- surrogate group: one already-audited lightweight diverse group;
- conditions: none, existing paper Gaussian, held-out-selected Gaussian EOT,
  translation, resize-and-pad, scale, geometric mixture, and Gaussian plus
  geometric mixture;
- items: 20 matched items;
- seeds: 42, 123, 2026;
- comparisons: equal steps and equal total surrogate-sample forwards;
- no API calls and no API-based selection.

Rotation is deferred to a later small-angle pilot. JPEG is classified as a
compression/photometric transform, not a geometric transform. OCR semantic
preservation must be audited per transform.

## Measurements

For each model and transformation record transformation consistency,
transformation-conditioned centered CKA, uncentered alignment, proxy and target
prototype distances, gradient agreement, clipping/saturation, held-out ASR,
runtime, peak memory, and exact forward count.

Support for a geometric invariance explanation requires a held-out improvement
under matched compute together with improved transformation consistency. An ASR
change without the associated measurement remains descriptive, not mechanistic.

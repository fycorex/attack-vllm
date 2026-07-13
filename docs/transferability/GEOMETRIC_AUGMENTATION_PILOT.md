# Geometric Augmentation Pilot

This branch is isolated from the active additive-noise search. It tests whether
stochastic image geometry improves transfer through transformation invariance,
not necessarily through the same mechanism as additive Gaussian smoothing.

## Conditions

The explicit, non-overlapping geometry modes are `none`, `translation`,
`resize_pad`, `scale`, and `geometric_mixture`. A combined condition uses
two-sample Gaussian EOT and two-sample geometric mixture. Rotation is deferred
because even small angles can change OCR semantics; JPEG remains a compression
augmentation rather than a geometric mode.

Explicit geometry replaces the legacy random crop/pad path and is sampled from
the run seed. It is applied after the bounded pixel perturbation and optional
additive noise, before encoder preprocessing. All outputs are differentiable,
clamped to the valid image range, and resized to each surrogate's input size.

## Gates and budgets

Stage 0 uses four Caltech caption items, 20 steps, one seed, and no API. It must
pass determinism, output completeness, finite-value, L-infinity, held-out
disjointness, and exact-forward checks before a larger pilot.

The equal-step pilot uses 100 optimization steps. The equal-forward pilot
compares 400 one-sample baseline updates with 100 four-sample EOT updates. With
two attack surrogates and the inherited four augmentation batches, every
equal-forward condition uses exactly 3,200 surrogate forwards per item.

Candidate selection uses held-out open-source targets only. API replay is
permitted only after conditions are frozen.

## Dataset independence

The previous 20-item VQA subset contained six unique attacked source images and
the receipt subset contained ten. Those runs remain pilot evidence, but their
item-level bootstrap intervals are not treated as independent-image intervals.

The updated preparation path assigns a unique LLaVA source image to every
target image and removes byte-identical receipt source images. Analysis reports
both item-level and source-image-clustered paired bootstrap intervals. Final
cross-dataset experiments use these regenerated manifests, not the repeated
source-image pilot manifests.

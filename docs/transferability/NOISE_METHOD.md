# Randomized Neighborhood Attack Method

The experiment tests whether optimizing expected loss in a pixel neighborhood
improves transfer to models excluded from optimization. It is not a comparison
whose success depends on beating a paper result.

For adversarial image `x_adv = clamp(x + delta)`, noise is added before model
resize/preprocessing and clamped again. Gaussian, uniform, and Rademacher noise
are zero-mean and variance matched. EOT averages sample losses/gradients. The
antithetic variant uses paired `eta` and `-eta`. The legacy probabilistic
Gaussian path remains available as a compatibility baseline and cannot stack
with explicit EOT.

The attack enforces the original pixel-domain L-infinity projection after every
update. Noise is transient and is not included in the saved adversarial image.
API models never participate in optimization or hyperparameter selection.

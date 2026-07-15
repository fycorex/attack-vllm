# Current Experiment State

**Live status:** the completed screening and replication outputs are frozen;
the 200-image geometry confirmation and frozen API replay are in progress.

## Completed

- 54 cross-family screening/replication trials and held-out summaries;
- frozen held-out-only surrogate selection: SigLIP, ViT+SigLIP, and ConvNeXt;
- 24/24 strict equal-forward augmentation replication trials;
- frozen held-out-only augmentation selection: translation EOT;
- cross-family alignment analysis with DINOv2 reported as a non-primary
  image-only control;
- API image-input capability smoke using the configured OpenAI-compatible
  endpoint.

## Running

- four-condition frozen 200-image attack confirmation on CUDA; its preceding
  geometry measurement is complete (11 models and 22,000 bootstrap rows);
- frozen clean/adversarial API replay matrices for caption, VQA, and
  receipt/OCR.

## Not yet complete

- the 200-image attack and held-out macro results;
- aggregate external API metrics and rubric;
- leave-SigLIP-family-out and family-balanced surrogate analysis;
- larger-caption and larger-VQA frozen confirmation.

See `EXPERIMENT_RESULTS.md` for the fixed protocol, completed numerical
results, interpretation limits, and the next decision gates.

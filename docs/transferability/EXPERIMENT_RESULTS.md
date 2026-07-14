# Transferability Experiment Results

## Status convention

- **Completed**: all planned trials and validation artifacts exist.
- **Running**: a runner has started but no conclusion is drawn from partial artifacts.
- **Planned**: code or a configuration exists, but no empirical claim is made.

## Completed equal-forward surrogate comparison

The completed three-seed, 19-item-per-task comparison used equal total
surrogate-sample forwards and three held-out OpenCLIP targets.

| Task | Two homogeneous proxies vs single | Four mixed proxies vs single | Four mixed vs two |
|---|---:|---:|---:|
| Caption | +25.1 pp, paired CI [10.5, 42.1] | +19.3 pp, paired CI [8.8, 32.7] | -5.8 pp, paired CI [-14.0, -0.6] |
| VQA | +18.1 pp, paired CI [6.4, 32.2] | +17.5 pp, paired CI [6.4, 30.4] | -0.6 pp, paired CI [-1.8, 0.0] |
| Receipt/OCR | +33.3 pp, paired CI [18.7, 50.3] | +25.7 pp, paired CI [14.0, 40.9] | -7.6 pp, paired CI [-13.5, -0.2] |

Interpretation: a small ensemble reduced blind single-proxy risk in this model
pool. Increasing the count from two to four was not reliably beneficial under
the same forward budget.

## Completed alignment analysis

The current clustered analysis contains 720 item-level rows and 12
proxy-target pairs from the earlier OpenCLIP-family matrix.

- Centered CKA to held-out ASR: Spearman -0.025; interval spans zero.
- NO@1 to held-out ASR: Spearman 0.685; interval is positive.
- ΔD_proxy to ΔD_target: Spearman 0.309, interval [0.168, 0.451].
- ΔD_target to target success: Spearman -0.391, interval [-0.527, -0.226].

These are descriptive results from a comparatively homogeneous encoder pool.
They motivate, but do not yet prove, a local-neighborhood explanation. The
running cross-family cycle is the required robustness check. Its current
measurement sample contains 50 caption images, 20 VQA images, and 19 receipt
images; it is a screening run, not the planned larger-sample confirmation.

## Completed augmentation evidence

Some fixed-step augmentation conditions showed positive signals, but
multi-sample EOT did not consistently beat additional deterministic steps under
equal-forward accounting. Translation showed a positive equal-forward signal
for caption and no comparable receipt/OCR gain. These results justify the
current task-stratified augmentation screen; they do not establish a universal
noise benefit.

## Current full-pipeline run

The active time-bounded search uses a separate output root and does not alter
any earlier completed result. It has two external-evaluation gates:

1. after held-out selection and replication in the main search;
2. after a separate 200-unique-image held-out attack confirmation.

Neither API result is used to select a proxy set, augmentation, or
hyperparameter. The 200-image stage has two distinct parts:

- a representation measurement that tests the stability of CKA and
  neighborhood-overlap rankings at a larger natural-image sample;
- a frozen attack comparison of the baseline, the selected surrogate set, the
  selected augmentation, and their combination under an equal-forward budget.

The current task manifests remain small for VQA and receipt/OCR (20 and 19
unique source images). Those tasks are used for screening and replication; the
200-image confirmation is caption/natural-image only and is explicitly not
presented as 200 independent VQA or OCR attacks.

The live status, active trial counts, capability smoke result, and next
milestones are intentionally kept outside version control in:

```text
outputs/transfer_search_cycle_8h_fullpipeline/LIVE_EXPERIMENT_REPORT.md
```

No current-cycle transfer or API conclusion is added to this tracked report
until the relevant trial records are complete, validated, and paired.

## Interpretation boundaries

- Completed results describe the evaluated encoders, tasks, perturbation
  budget, and image counts only. They are not claims about all multimodal
  models.
- Global CKA and local-neighborhood metrics are reported as descriptive
  predictors. Correlation does not establish a causal transfer mechanism.
- A result from a time-bounded screening-only freeze is labeled as such and is
  not conflated with a replicated final selection.

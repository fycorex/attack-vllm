# Full Experiment Result Analysis

Date: 2026-04-16

Runs compared:

```text
16/255: outputs/paper_caltech
8/255:  outputs/paper_caltech_eps8
```

Tracked result folders:

```text
docs/results/paper_caltech_eps16_full
docs/results/paper_caltech_eps8_full
```

## Overall Results

| Epsilon | Proxy ASR | Avg Margin Gain | GPT-4o ASR | GPT-5-mini ASR |
| --- | ---: | ---: | ---: | ---: |
| 16/255 | 50 / 50 = 100% | 0.6020 | 48 / 50 = 96% | 48 / 50 = 96% |
| 8/255 | 49 / 50 = 98% | 0.5362 | 43 / 50 = 86% | 44 / 50 = 88% |

The 16/255 budget is stronger across proxy and transfer metrics. It improves
GPT-4o transfer by 10 percentage points and GPT-5-mini transfer by 8 percentage
points over the 8/255 option on this Caltech101 demo set.

## Per-Pair Transfer

GPT-4o:

| Pair | 16/255 | 8/255 |
| --- | ---: | ---: |
| car -> dog | 10 / 10 | 10 / 10 |
| dog -> watch | 10 / 10 | 10 / 10 |
| laptop -> phone | 10 / 10 | 9 / 10 |
| phone -> car | 9 / 10 | 7 / 10 |
| watch -> laptop | 9 / 10 | 7 / 10 |

GPT-5-mini:

| Pair | 16/255 | 8/255 |
| --- | ---: | ---: |
| car -> dog | 10 / 10 | 9 / 10 |
| dog -> watch | 10 / 10 | 10 / 10 |
| laptop -> phone | 10 / 10 | 10 / 10 |
| phone -> car | 10 / 10 | 8 / 10 |
| watch -> laptop | 8 / 10 | 7 / 10 |

## Failure Concentration

16/255 failures:

```text
GPT-4o:
  item_22 watch -> laptop
  item_24 phone -> car

GPT-5-mini:
  item_22 watch -> laptop
  item_42 watch -> laptop
```

8/255 failures:

```text
GPT-4o:
  item_12 watch -> laptop
  item_22 watch -> laptop
  item_24 phone -> car
  item_34 phone -> car
  item_42 watch -> laptop
  item_43 laptop -> phone
  item_49 phone -> car

GPT-5-mini:
  item_00 car -> dog
  item_12 watch -> laptop
  item_22 watch -> laptop
  item_24 phone -> car
  item_27 watch -> laptop
  item_49 phone -> car
```

The hardest transfer pairs in this run are `watch -> laptop` and `phone -> car`.
The 8/255 run particularly degrades on those pairs.

## Proxy Notes

The 16/255 run succeeded on all 50 proxy evaluations. The 8/255 run had one
proxy failure:

```text
item_49 phone -> car
clean_margin: -0.2316
adversarial_margin: -0.0221
margin_gain: 0.2095
```

Even for that failure, the adversarial margin improved, but it did not cross the
proxy success threshold.

## Artifact Contents

Each full result folder includes:

```text
effective_config.json
summary.json
items.csv
eval_gpt4o.jsonl
eval_gpt5mini.jsonl
item_00 ... item_49
```

Each item folder includes:

```text
clean.png
adversarial.png
delta_vis.png
metrics.json
```

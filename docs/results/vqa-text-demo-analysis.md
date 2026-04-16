# VQA and Receipt Text Demo Result Analysis

Date: 2026-04-16

Runs checked:

```text
VQA:          outputs/llava_vqa_eps16
Receipt text: outputs/receipt_text_eps32
```

Tracked result folders:

```text
docs/results/llava_vqa_eps16_demo
docs/results/receipt_text_eps32_demo
```

Success rates below use the `gpt_success` field written by
`scripts/replay_gpt_eval.py`. For VQA and receipt text, this field is judged
with the paper-style True/False prompt.

## Run Settings

| Task | Dataset | Items | Epsilon | Steps | Surrogates |
| --- | --- | ---: | ---: | ---: | ---: |
| VQA | LLaVA-Bench COCO | 15 questions from 5 target images | 16/255 | 300 | 8 |
| Receipt text | TrainingDataPro receipt OCR | 10 questions from 5 receipt images | 32/255 | 300 | 8 |

The VQA run matches the paper's three question categories for each selected
image: Conversation, Detail, and Reasoning. The paper-scale VQA setting uses 30
images; this checked run uses 5.

The receipt eps32 run is demo-scale. It contains 10 items from 5 receipts, not
the full 20-receipt, 40-question setting.

## Overall Results

| Task | Proxy ASR | GPT-4o ASR | GPT-5-mini ASR |
| --- | ---: | ---: | ---: |
| VQA eps16 | 15 / 15 = 100% | 7 / 15 = 46.7% | 10 / 15 = 66.7% |
| Receipt text eps32 | 10 / 10 = 100% | 2 / 10 = 20% | 3 / 10 = 30% |

## VQA Breakdown

| Category | GPT-4o | GPT-5-mini |
| --- | ---: | ---: |
| Conversation | 2 / 5 = 40% | 3 / 5 = 60% |
| Detail | 1 / 5 = 20% | 3 / 5 = 60% |
| Reasoning | 4 / 5 = 80% | 4 / 5 = 80% |

The VQA transfer is strongest on Reasoning in this demo and weakest on Detail
for GPT-4o.

## Receipt Text Breakdown

| Question Type | GPT-4o | GPT-5-mini |
| --- | ---: | ---: |
| Store | 1 / 5 = 20% | 1 / 5 = 20% |
| Total | 0 / 4 = 0% | 1 / 4 = 25% |
| Item | 1 / 1 = 100% | 1 / 1 = 100% |

The receipt eps32 demo shows strong proxy movement but weak GPT transfer. Store
and total questions remain the limiting cases in this run.

## Artifact Contents

Each tracked result folder includes:

```text
effective_config.json
summary.json
items.csv
eval_gpt4o.jsonl
eval_gpt5mini.jsonl
eval_summary.json
item_* folders
```

Each item folder includes:

```text
clean.png
adversarial.png
delta_vis.png
metrics.json
```

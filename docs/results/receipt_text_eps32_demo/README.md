# Receipt Text Demo Result - 32/255

Config:

```text
configs/receipt_text_eps32.yaml
```

Commands:

```bash
bash scripts/run_experiment.sh receipt-text-32 300 5
bash scripts/run_experiment.sh eval-receipt-text-gpt outputs/receipt_text_eps32
```

Summary:

```text
items: 10
receipt images: 5
questions per image: 2
epsilon: 32/255
steps: 300
surrogates: 8
proxy ASR: 10 / 10 = 100%
GPT-4o replay ASR: 2 / 10 = 20%
GPT-5-mini replay ASR: 3 / 10 = 30%
```

Question-type replay:

| Question Type | GPT-4o | GPT-5-mini |
| --- | ---: | ---: |
| Store | 1 / 5 = 20% | 1 / 5 = 20% |
| Total | 0 / 4 = 0% | 1 / 4 = 25% |
| Item | 1 / 1 = 100% | 1 / 1 = 100% |

This is a demo-scale eps32 result. The full receipt-text setting is:

```bash
bash scripts/run_experiment.sh receipt-text-32 300 20
bash scripts/run_experiment.sh eval-receipt-text-gpt outputs/receipt_text_eps32
```

This folder includes `clean.png`, `adversarial.png`, `delta_vis.png`, and
`metrics.json` for every receipt-text item.

See `docs/results/vqa-text-demo-analysis.md` for the cross-task analysis.

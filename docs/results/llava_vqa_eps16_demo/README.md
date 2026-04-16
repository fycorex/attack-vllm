# LLaVA-Bench COCO VQA Demo Result - 16/255

Config:

```text
configs/llava_bench_vqa_eps16.yaml
```

Commands:

```bash
bash scripts/run_experiment.sh llava-vqa-demo 5 300
bash scripts/run_experiment.sh eval-llava-vqa-gpt outputs/llava_vqa_eps16
```

Summary:

```text
items: 15
target images: 5
questions per image: Conversation, Detail, Reasoning
epsilon: 16/255
steps: 300
surrogates: 8
proxy ASR: 15 / 15 = 100%
GPT-4o replay ASR: 7 / 15 = 46.7%
GPT-5-mini replay ASR: 10 / 15 = 66.7%
```

Category replay:

| Category | GPT-4o | GPT-5-mini |
| --- | ---: | ---: |
| Conversation | 2 / 5 = 40% | 3 / 5 = 60% |
| Detail | 1 / 5 = 20% | 3 / 5 = 60% |
| Reasoning | 4 / 5 = 80% | 4 / 5 = 80% |

This folder includes `clean.png`, `adversarial.png`, `delta_vis.png`, and
`metrics.json` for every VQA item.

See `docs/results/vqa-text-demo-analysis.md` for the cross-task analysis.

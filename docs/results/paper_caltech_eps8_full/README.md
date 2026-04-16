# Paper Caltech Full Result - 8/255

Config:

```text
configs/caption_attack_paper_eps8.yaml
```

Command:

```bash
bash scripts/run_experiment.sh full-matrix \
  configs/caption_attack_paper_eps8.yaml \
  50 \
  300 \
  outputs/paper_caltech_eps8
```

Summary:

```text
items: 50
proxy ASR: 49 / 50 = 98%
average margin gain: 0.5362
GPT-4o replay ASR: 43 / 50 = 86%
GPT-5-mini replay ASR: 44 / 50 = 88%
```

See `docs/results/full-experiment-analysis.md` for the cross-epsilon analysis.

This folder includes `clean.png`, `adversarial.png`, `delta_vis.png`, and
`metrics.json` for every item from `item_00` through `item_49`.

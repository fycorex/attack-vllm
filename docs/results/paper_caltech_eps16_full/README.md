# Paper Caltech Full Result - 16/255

Config:

```text
configs/caption_attack_paper.yaml
```

Command:

```bash
bash scripts/run_experiment.sh full-matrix \
  configs/caption_attack_paper.yaml \
  50 \
  300 \
  outputs/paper_caltech
```

Summary:

```text
items: 50
proxy ASR: 50 / 50 = 100%
average margin gain: 0.6020
GPT-4o replay ASR: 48 / 50 = 96%
GPT-5-mini replay ASR: 48 / 50 = 96%
```

See `docs/results/full-experiment-analysis.md` for the cross-epsilon analysis.

This folder includes `clean.png`, `adversarial.png`, `delta_vis.png`, and
`metrics.json` for every item from `item_00` through `item_49`.

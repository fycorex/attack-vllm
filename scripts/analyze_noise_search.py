#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def percentile(values: list[float], q: float) -> float | None:
    if not values: return None
    values = sorted(values); return values[int(round(q * (len(values) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired held-out noise search results.")
    parser.add_argument("--root", required=True); parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(); stage = Path(args.root) / args.stage
    trials = []
    for result_path in stage.glob("*/trial_result.json"):
        trial = json.loads(result_path.read_text()); heldout = json.loads((result_path.parent / "heldout_summary.json").read_text())
        trial["rows"] = {(r["model"], r["item_id"]): {"success": bool(r["proxy_success"]), "margin_gain": float(r["margin_gain"])} for r in heldout["items"]}
        trial["heldout_macro_margin_gain"] = sum(x["margin_gain"] for x in trial["rows"].values()) / max(1, len(trial["rows"])); trials.append(trial)
    baselines = {(t["dataset"], t["surrogate_group"], t["seed"]): t for t in trials if t["mode"] == "none"}
    grouped = defaultdict(list)
    for t in trials: grouped[(t["dataset"], t["surrogate_group"], t["mode"], t["sigma"], t["samples"])].append(t)
    rng = random.Random(0); output = []
    for key, members in grouped.items():
        deltas = []; margin_deltas = []
        for member in members:
            baseline = baselines.get((member["dataset"], member["surrogate_group"], member["seed"]))
            if baseline is None or member["mode"] == "none": continue
            common = sorted(member["rows"].keys() & baseline["rows"].keys())
            deltas.extend(float(member["rows"][x]["success"]) - float(baseline["rows"][x]["success"]) for x in common)
            margin_deltas.extend(member["rows"][x]["margin_gain"] - baseline["rows"][x]["margin_gain"] for x in common)
        boots = []
        if deltas:
            for _ in range(args.bootstrap_samples): boots.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
        output.append({"dataset": key[0], "surrogate_group": key[1], "mode": key[2], "sigma": key[3], "samples": key[4],
                       "seeds": sorted(t["seed"] for t in members), "mean_heldout_macro_asr": statistics.mean(t["heldout_macro_asr"] for t in members),
                       "mean_heldout_macro_margin_gain": statistics.mean(t["heldout_macro_margin_gain"] for t in members),
                       "seed_std": statistics.stdev(t["heldout_macro_asr"] for t in members) if len(members) > 1 else None,
                       "paired_observations": len(deltas), "paired_delta": statistics.mean(deltas) if deltas else None,
                       "paired_margin_delta": statistics.mean(margin_deltas) if margin_deltas else None,
                       "paired_delta_ci95": [percentile(boots, .025), percentile(boots, .975)] if boots else [None, None]})
    output.sort(key=lambda x: (x["paired_delta"] is not None, x["paired_delta"] or -1, x["paired_margin_delta"] or -1, x["mean_heldout_macro_asr"]), reverse=True)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(output, indent=2))
    print(json.dumps({"conditions": len(output), "top": output[:10]}, indent=2))


if __name__ == "__main__": main()

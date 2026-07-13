#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics


def percentile(values: list[float], q: float) -> float | None:
    if not values: return None
    values = sorted(values); return values[int(round(q * (len(values) - 1)))]


def image_cluster_id(path: Path, fallback: str) -> str:
    if not path.is_file():
        return fallback
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clustered_bootstrap(observations: list[dict], key: str, samples: int, rng: random.Random) -> tuple[float | None, list[float | None], int]:
    clusters = defaultdict(list)
    for row in observations:
        clusters[row["cluster_id"]].append(float(row[key]))
    values = [statistics.mean(group) for _, group in sorted(clusters.items())]
    if not values:
        return None, [None, None], 0
    draws = [statistics.mean(rng.choice(values) for _ in values) for _ in range(samples)]
    return statistics.mean(values), [percentile(draws, .025), percentile(draws, .975)], len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired held-out noise search results.")
    parser.add_argument("--root", required=True); parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(); stage = Path(args.root) / args.stage
    trials = []
    for result_path in stage.glob("*/trial_result.json"):
        trial = json.loads(result_path.read_text()); heldout = json.loads((result_path.parent / "heldout_summary.json").read_text())
        trial["rows"] = {(r["model"], r["item_id"]): {
            "success": bool(r["proxy_success"]), "margin_gain": float(r["margin_gain"]),
            "cluster_id": image_cluster_id(result_path.parent / "attack" / r["item_id"] / "clean.png", r["item_id"]),
        } for r in heldout["items"]}
        trial["heldout_macro_margin_gain"] = sum(x["margin_gain"] for x in trial["rows"].values()) / max(1, len(trial["rows"])); trials.append(trial)
    baselines = {(t["dataset"], t["surrogate_group"], t["seed"]): t for t in trials if t["mode"] == "none"}
    grouped = defaultdict(list)
    for t in trials: grouped[(t["dataset"], t["surrogate_group"], t["mode"], t["sigma"], t["samples"])].append(t)
    rng = random.Random(0); output = []
    for key, members in grouped.items():
        observations = []
        for member in members:
            baseline = baselines.get((member["dataset"], member["surrogate_group"], member["seed"]))
            if baseline is None or member["mode"] == "none": continue
            common = sorted(member["rows"].keys() & baseline["rows"].keys())
            observations.extend({
                "cluster_id": member["rows"][x]["cluster_id"],
                "success_delta": float(member["rows"][x]["success"]) - float(baseline["rows"][x]["success"]),
                "margin_delta": member["rows"][x]["margin_gain"] - baseline["rows"][x]["margin_gain"],
            } for x in common)
        paired_delta, paired_ci, clusters = clustered_bootstrap(observations, "success_delta", args.bootstrap_samples, rng)
        paired_margin, margin_ci, _ = clustered_bootstrap(observations, "margin_delta", args.bootstrap_samples, rng)
        output.append({"dataset": key[0], "surrogate_group": key[1], "mode": key[2], "sigma": key[3], "samples": key[4],
                       "seeds": sorted(t["seed"] for t in members), "mean_heldout_macro_asr": statistics.mean(t["heldout_macro_asr"] for t in members),
                       "mean_heldout_macro_margin_gain": statistics.mean(t["heldout_macro_margin_gain"] for t in members),
                       "seed_std": statistics.stdev(t["heldout_macro_asr"] for t in members) if len(members) > 1 else None,
                       "paired_observations": len(observations), "unique_source_clusters": clusters,
                       "paired_delta": paired_delta, "paired_margin_delta": paired_margin,
                       "paired_delta_ci95": paired_ci, "paired_margin_delta_ci95": margin_ci})
    output.sort(key=lambda x: (x["paired_delta"] is not None, x["paired_delta"] or -1, x["paired_margin_delta"] or -1, x["mean_heldout_macro_asr"]), reverse=True)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(output, indent=2))
    print(json.dumps({"conditions": len(output), "top": output[:10]}, indent=2))


if __name__ == "__main__": main()

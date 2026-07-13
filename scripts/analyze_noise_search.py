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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired held-out noise search results.")
    parser.add_argument("--root", required=True); parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(); stage = Path(args.root) / args.stage
    trials = []
    for result_path in stage.glob("*/trial_result.json"):
        trial = json.loads(result_path.read_text()); heldout = json.loads((result_path.parent / "heldout_summary.json").read_text())
        trial["rows"] = {(r["model"], r["item_id"]): {"success": bool(r["proxy_success"]), "margin_gain": float(r["margin_gain"])} for r in heldout["items"]}
        item_hashes = {}
        for clean_path in (result_path.parent / "attack").glob("item_*/clean.png"):
            item_hashes[clean_path.parent.name] = hashlib.sha256(clean_path.read_bytes()).hexdigest()
        clusters = defaultdict(list)
        for (model, item_id), value in trial["rows"].items():
            clusters[(model, item_hashes.get(item_id, item_id))].append(value)
        trial["clusters"] = {
            key: {"success": statistics.mean(float(row["success"]) for row in values),
                  "margin_gain": statistics.mean(row["margin_gain"] for row in values)}
            for key, values in clusters.items()
        }
        trial["unique_source_images"] = len(set(item_hashes.values()))
        trial["heldout_macro_margin_gain"] = sum(x["margin_gain"] for x in trial["rows"].values()) / max(1, len(trial["rows"])); trials.append(trial)
    baselines = {(t["dataset"], t["surrogate_group"], t["seed"]): t for t in trials
                 if t["mode"] == "none" and t.get("geometry_mode", "none") == "none"}
    grouped = defaultdict(list)
    for t in trials: grouped[(t["dataset"], t["surrogate_group"], t["mode"], t["sigma"], t["samples"],
                              t.get("geometry_mode", "none"), t.get("geometry_samples", 1))].append(t)
    rng = random.Random(0); output = []
    for key, members in grouped.items():
        deltas = []; margin_deltas = []; cluster_deltas = []; cluster_margin_deltas = []
        for member in members:
            baseline = baselines.get((member["dataset"], member["surrogate_group"], member["seed"]))
            is_baseline = member["mode"] == "none" and member.get("geometry_mode", "none") == "none"
            if baseline is None or is_baseline: continue
            common = sorted(member["rows"].keys() & baseline["rows"].keys())
            deltas.extend(float(member["rows"][x]["success"]) - float(baseline["rows"][x]["success"]) for x in common)
            margin_deltas.extend(member["rows"][x]["margin_gain"] - baseline["rows"][x]["margin_gain"] for x in common)
            common_clusters = sorted(member["clusters"].keys() & baseline["clusters"].keys())
            cluster_deltas.extend(member["clusters"][x]["success"] - baseline["clusters"][x]["success"] for x in common_clusters)
            cluster_margin_deltas.extend(member["clusters"][x]["margin_gain"] - baseline["clusters"][x]["margin_gain"] for x in common_clusters)
        boots = []
        if deltas:
            for _ in range(args.bootstrap_samples): boots.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
        cluster_boots = []
        if cluster_deltas:
            for _ in range(args.bootstrap_samples):
                cluster_boots.append(sum(rng.choice(cluster_deltas) for _ in cluster_deltas) / len(cluster_deltas))
        output.append({"dataset": key[0], "surrogate_group": key[1], "mode": key[2], "sigma": key[3], "samples": key[4],
                       "geometry_mode": key[5], "geometry_samples": key[6],
                       "expected_surrogate_forwards_per_item": members[0].get("expected_surrogate_forwards_per_item"),
                       "seeds": sorted(t["seed"] for t in members), "mean_heldout_macro_asr": statistics.mean(t["heldout_macro_asr"] for t in members),
                       "unique_source_images_per_trial": sorted({t["unique_source_images"] for t in members}),
                       "mean_heldout_macro_margin_gain": statistics.mean(t["heldout_macro_margin_gain"] for t in members),
                       "seed_std": statistics.stdev(t["heldout_macro_asr"] for t in members) if len(members) > 1 else None,
                       "paired_observations": len(deltas), "paired_delta": statistics.mean(deltas) if deltas else None,
                       "paired_margin_delta": statistics.mean(margin_deltas) if margin_deltas else None,
                       "paired_delta_ci95": [percentile(boots, .025), percentile(boots, .975)] if boots else [None, None],
                       "paired_source_image_clusters": len(cluster_deltas),
                       "clustered_paired_delta": statistics.mean(cluster_deltas) if cluster_deltas else None,
                       "clustered_paired_margin_delta": statistics.mean(cluster_margin_deltas) if cluster_margin_deltas else None,
                       "clustered_paired_delta_ci95": [percentile(cluster_boots, .025), percentile(cluster_boots, .975)] if cluster_boots else [None, None]})
    output.sort(key=lambda x: (x["paired_delta"] is not None, x["paired_delta"] or -1, x["paired_margin_delta"] or -1, x["mean_heldout_macro_asr"]), reverse=True)
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(output, indent=2))
    print(json.dumps({"conditions": len(output), "top": output[:10]}, indent=2))


if __name__ == "__main__": main()

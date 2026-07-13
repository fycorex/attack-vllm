#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from typing import Any


def condition_id(trial: dict[str, Any]) -> str:
    return (f"noise={trial['mode']}:sigma={float(trial['sigma']):.8g}:n={int(trial['samples'])}"
            f"|geometry={trial.get('geometry_mode', 'none')}:n={int(trial.get('geometry_samples', 1))}")


def is_baseline(trial: dict[str, Any]) -> bool:
    return (trial["mode"] == "none" and float(trial["sigma"]) == 0.0
            and trial.get("geometry_mode", "none") == "none")


def bootstrap_clustered(rows: list[dict[str, Any]], value: str, samples: int = 2000) -> tuple[float | None, list[float | None]]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row["item_id"])].append(float(row[value]))
    values = [sum(group) / len(group) for _, group in sorted(clusters.items())]
    if not values:
        return None, [None, None]
    estimate = sum(values) / len(values)
    rng = random.Random(0)
    draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return estimate, [draws[int(.025 * (len(draws) - 1))], draws[int(.975 * (len(draws) - 1))]]


def load_trials(root: Path) -> list[dict[str, Any]]:
    trials = []
    for path in sorted(root.glob("*/*/trial_result.json")):
        trial = json.loads(path.read_text(encoding="utf-8"))
        heldout_path = path.with_name("heldout_summary.json")
        if trial.get("status") != "complete" or not heldout_path.is_file():
            continue
        heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
        trial["directory"] = str(path.parent.resolve())
        trial["heldout"] = heldout
        trial["condition_id"] = condition_id(trial)
        trials.append(trial)
    return trials


def compare_to_baseline(trials: list[dict[str, Any]], samples: int = 2000) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        groups[(trial["stage"], trial["dataset"], trial["surrogate_group"], int(trial["seed"]))].append(trial)
    observations = []
    for (stage, dataset, surrogate_group, seed), values in groups.items():
        baselines = [trial for trial in values if is_baseline(trial)]
        if len(baselines) != 1:
            continue
        baseline = baselines[0]
        baseline_items = {(row["model"], row["item_id"]): row for row in baseline["heldout"]["items"]}
        for candidate in values:
            if candidate is baseline:
                continue
            candidate_items = {(row["model"], row["item_id"]): row for row in candidate["heldout"]["items"]}
            for model, item_id in sorted(baseline_items.keys() & candidate_items.keys()):
                first, second = baseline_items[(model, item_id)], candidate_items[(model, item_id)]
                observations.append({
                    "stage": stage, "dataset": dataset, "surrogate_group": surrogate_group,
                    "seed": seed, "target": model, "item_id": item_id,
                    "baseline": baseline["condition_id"], "candidate": candidate["condition_id"],
                    "success_delta": float(bool(second.get("proxy_success"))) - float(bool(first.get("proxy_success"))),
                    "margin_gain_delta": float(second.get("margin_gain", 0)) - float(first.get("margin_gain", 0)),
                })
    return observations


def aggregate_comparisons(observations: list[dict[str, Any]], samples: int = 2000) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        key = (row["stage"], row["dataset"], row["surrogate_group"], row["baseline"], row["candidate"])
        groups[key].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        success, success_ci = bootstrap_clustered(rows, "success_delta", samples)
        margin, margin_ci = bootstrap_clustered(rows, "margin_gain_delta", samples)
        stage, dataset, group, baseline, candidate = key
        output.append({
            "stage": stage, "dataset": dataset, "surrogate_group": group,
            "baseline": baseline, "candidate": candidate,
            "asr_delta": success, "asr_ci95_low": success_ci[0], "asr_ci95_high": success_ci[1],
            "margin_gain_delta": margin, "margin_ci95_low": margin_ci[0], "margin_ci95_high": margin_ci[1],
            "seeds": len({row["seed"] for row in rows}),
            "targets": len({row["target"] for row in rows}),
            "unique_item_clusters": len({row["item_id"] for row in rows}),
            "paired_observations": len(rows),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired geometric/noise transfer effects against no augmentation.")
    parser.add_argument("--root", default="outputs/geometric_pilot")
    parser.add_argument("--output", default="outputs/geometric_analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    trials = load_trials(Path(args.root))
    observations = compare_to_baseline(trials, args.bootstrap_samples)
    aggregates = aggregate_comparisons(observations, args.bootstrap_samples)
    output = Path(args.output)
    write_csv(output / "paired_item_effects.csv", observations)
    write_csv(output / "aggregate_effects.csv", aggregates)
    summary = {"trials": len(trials), "paired_observations": len(observations),
               "aggregate_comparisons": len(aggregates), "api_results_used": False}
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path
import random
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_mean(values: list[float], samples: int = 2000) -> tuple[float | None, list[float | None]]:
    if not values:
        return None, [None, None]
    estimate = sum(values) / len(values)
    rng = random.Random(0)
    draws = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return estimate, [draws[int(.025 * (len(draws) - 1))], draws[int(.975 * (len(draws) - 1))]]


def _method_sort_key(name: str) -> tuple[int, str]:
    preferred = {
        "single_reference": 0,
        "two_homogeneous": 1,
        "incremental_three": 2,
        "four_lightweight_mixed": 3,
    }
    return preferred.get(name, 100), name


def paired_method_comparisons(
    trials: list[dict[str, Any]], samples: int = 2000
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare every method pair and bootstrap item clusters across seeds/targets.

    The same item is attacked under multiple seeds and evaluated on multiple
    targets, so those observations are not independent. Aggregate intervals
    resample item IDs and retain all seed/target observations within each item.
    """
    trial_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        trial_groups[(trial["stage"], trial["dataset"], int(trial["seed"]))].append(trial)

    cell_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for (stage, dataset, seed), group in sorted(trial_groups.items()):
        ordered = sorted(group, key=lambda trial: _method_sort_key(trial["surrogate_set"]))
        for baseline, candidate in combinations(ordered, 2):
            baseline_targets = {model for model, _ in baseline["item_success"]}
            candidate_targets = {model for model, _ in candidate["item_success"]}
            for target in sorted(baseline_targets & candidate_targets):
                baseline_ids = {item_id for model, item_id in baseline["item_success"] if model == target}
                candidate_ids = {item_id for model, item_id in candidate["item_success"] if model == target}
                paired_ids = sorted(baseline_ids & candidate_ids)
                deltas = [
                    float(candidate["item_success"][(target, item_id)])
                    - float(baseline["item_success"][(target, item_id)])
                    for item_id in paired_ids
                ]
                delta, interval = _bootstrap_mean(deltas, samples=samples)
                cell_rows.append({
                    "stage": stage,
                    "dataset": dataset,
                    "seed": seed,
                    "target": target,
                    "baseline": baseline["surrogate_set"],
                    "candidate": candidate["surrogate_set"],
                    "baseline_items": len(baseline_ids),
                    "candidate_items": len(candidate_ids),
                    "paired_items": len(paired_ids),
                    "asr_delta": delta,
                    "ci95_low": interval[0],
                    "ci95_high": interval[1],
                })
                observations.extend({
                    "stage": stage,
                    "dataset": dataset,
                    "seed": seed,
                    "target": target,
                    "baseline": baseline["surrogate_set"],
                    "candidate": candidate["surrogate_set"],
                    "item_id": item_id,
                    "delta": item_delta,
                } for item_id, item_delta in zip(paired_ids, deltas))

    aggregate_rows: list[dict[str, Any]] = []
    for scope in ("target", "macro_targets"):
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in observations:
            key = (row["stage"], row["dataset"], row["baseline"], row["candidate"])
            if scope == "target":
                key += (row["target"],)
            grouped[key].append(row)
        for key, group in sorted(grouped.items()):
            item_clusters: dict[str, list[float]] = defaultdict(list)
            for row in group:
                item_clusters[row["item_id"]].append(row["delta"])
            cluster_means = [sum(values) / len(values) for _, values in sorted(item_clusters.items())]
            delta, interval = _bootstrap_mean(cluster_means, samples=samples)
            stage, dataset, baseline, candidate, *target = key
            aggregate_rows.append({
                "scope": scope,
                "stage": stage,
                "dataset": dataset,
                "target": target[0] if target else "__macro__",
                "baseline": baseline,
                "candidate": candidate,
                "asr_delta": delta,
                "ci95_low": interval[0],
                "ci95_high": interval[1],
                "seed_count": len({row["seed"] for row in group}),
                "target_count": len({row["target"] for row in group}),
                "unique_item_clusters": len(item_clusters),
                "paired_observations": len(group),
            })
    return cell_rows, aggregate_rows


def load_trials(root: Path) -> list[dict[str, Any]]:
    trials = []
    for result_path in sorted(root.glob("*/*/trial_result.json")):
        trial = json.loads(result_path.read_text(encoding="utf-8"))
        heldout_path = result_path.with_name("heldout_summary.json")
        if not heldout_path.is_file():
            continue
        heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
        trial["directory"] = str(result_path.parent)
        trial["heldout"] = heldout
        trial["item_success"] = {(row["model"], row["item_id"]): bool(row.get("proxy_success", False)) for row in heldout["items"]}
        trials.append(trial)
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paired surrogate-ensemble transfer tables from completed stages.")
    parser.add_argument("--root", default="outputs/surrogate_experiments")
    parser.add_argument("--output", default="outputs/surrogate_analysis")
    parser.add_argument("--theory-metrics", help="Optional Stage 0A ensemble_theory_metrics.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    trials = load_trials(Path(args.root))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for trial in trials:
        for target, metrics in trial["heldout"]["heldout_models"].items():
            rows.append({key: trial[key] for key in ("stage", "dataset", "surrogate_set", "seed", "items", "steps", "model_count", "budget_mode", "forward_units_per_item")} |
                        {"target": target, **metrics, "directory": trial["directory"]})
    _write_csv(output / "all_transfer_results.csv", rows)
    _write_csv(output / "proxy_to_target_transfer_matrix.csv", [row for row in rows if row["surrogate_set"].startswith("single_")])
    _write_csv(output / "equal_compute_results.csv", [row for row in rows if row["budget_mode"] == "equal_forwards"])
    _write_csv(output / "incremental_proxy_addition.csv", [row for row in rows if row["surrogate_set"] in {"single_reference", "two_homogeneous", "incremental_three", "four_lightweight_mixed"}])
    _write_csv(output / "leave_one_out_results.csv", [row for row in rows if row["surrogate_set"].startswith("leave_out_")])

    comparison_rows, aggregate_comparison_rows = paired_method_comparisons(
        trials, samples=args.bootstrap_samples
    )
    _write_csv(output / "paired_comparisons.csv", comparison_rows)
    _write_csv(output / "aggregate_paired_comparisons.csv", aggregate_comparison_rows)

    theory_rows = []
    if args.theory_metrics:
        with Path(args.theory_metrics).open(newline="", encoding="utf-8") as handle:
            theory_rows = list(csv.DictReader(handle))
        observed = {(row["surrogate_set"], row["target"]): row for row in rows}
        for row in theory_rows:
            result = observed.get((row["surrogate_set"], row["target"]))
            if result:
                row.update({"observed_asr": result["asr"], "observed_margin_gain": result["mean_margin_gain"],
                            "observed_stage": result["stage"], "observed_seed": result["seed"]})
        _write_csv(output / "ensemble_theory_metrics.csv", theory_rows)

    claims = [
        "# Surrogate Ensemble Claim Summary",
        "",
        f"Completed trial records analyzed: {len(trials)}.",
        "",
        "No claim is classified from measurement-only CKA/EA values. Transfer classifications require completed paired attacks.",
        "",
        f"Paired ensemble-to-reference comparisons available: {len(comparison_rows)}.",
        f"Clustered aggregate comparisons available: {len(aggregate_comparison_rows)}.",
        "",
        "API outcomes are not read by this analyzer.",
    ]
    (output / "ensemble_claim_summary.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    print(json.dumps({"trials": len(trials), "result_rows": len(rows), "paired_comparisons": len(comparison_rows),
                      "aggregate_paired_comparisons": len(aggregate_comparison_rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

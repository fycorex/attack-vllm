#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from surrogate_composition import load_composition_spec
from transfer_eval import atomic_write_json


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _group_mean(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(float(row["asr"]))
    return {key: _mean(values) for key, values in sorted(grouped.items())}


def score_candidates(
    rows: list[dict[str, str]],
    composition_path: str | Path,
    stage: str | None = None,
    budget_mode: str = "equal_forwards",
) -> list[dict[str, Any]]:
    """Rank complete, compute-matched candidates using open-source targets only."""
    forbidden = {key for row in rows for key in row if key.startswith("api_") or "gpt" in key.lower()}
    if forbidden:
        raise ValueError(f"API-derived columns are forbidden in candidate selection: {sorted(forbidden)}")
    spec = load_composition_spec(composition_path)
    filtered = [
        row for row in rows
        if (stage is None or row.get("stage") == stage) and row.get("budget_mode") == budget_mode
    ]
    if not filtered:
        raise ValueError("No matching open-source held-out result rows")
    forward_budgets = {int(row["forward_units_per_item"]) for row in filtered}
    if len(forward_budgets) != 1:
        raise ValueError(f"Candidate rows are not compute-matched: {sorted(forward_budgets)}")

    primary_targets = {
        model_id for model_id in spec.heldout_models
        if spec.models[model_id].evaluation_role == "primary"
    }
    control_targets = set(spec.heldout_models) - primary_targets
    if not primary_targets:
        raise ValueError("At least one primary held-out target is required")
    datasets = {row["dataset"] for row in filtered}
    seeds = {int(row["seed"]) for row in filtered}
    expected_primary = len(datasets) * len(seeds) * len(primary_targets)

    by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in filtered:
        if row["surrogate_set"] in spec.sets:
            by_candidate[row["surrogate_set"]].append(row)

    results = []
    for name, candidate_rows in sorted(by_candidate.items()):
        primary = [row for row in candidate_rows if row["target"] in primary_targets]
        controls = [row for row in candidate_rows if row["target"] in control_targets]
        valid = all(int(row.get("missing_items", 0)) == 0 for row in primary)
        complete = len(primary) == expected_primary
        task_scores = _group_mean(primary, "dataset")
        target_scores = _group_mean(primary, "target")
        seed_scores = _group_mean(primary, "seed")
        family_values: dict[str, list[float]] = defaultdict(list)
        for row in primary:
            family_values[spec.models[row["target"]].architecture_family].append(float(row["asr"]))
        family_scores = {key: _mean(value) for key, value in sorted(family_values.items())}
        macro = _mean([float(row["asr"]) for row in primary])
        worst_task = min(task_scores.values(), default=0.0)
        worst_family = min(family_scores.values(), default=0.0)
        seed_std = statistics.pstdev(seed_scores.values()) if len(seed_scores) > 1 else 0.0
        # Emphasize general transfer, then guard against task/family collapse and unstable seeds.
        selection_score = 0.55 * macro + 0.25 * worst_task + 0.20 * worst_family - 0.05 * seed_std
        model_ids = spec.sets[name].models
        objective_families = sorted({spec.models[model_id].objective_family for model_id in model_ids})
        architecture_families = sorted({spec.models[model_id].architecture_family for model_id in model_ids})
        results.append({
            "surrogate_set": name,
            "eligible": bool(valid and complete),
            "selection_score": selection_score if valid and complete else float("-inf"),
            "primary_macro_asr": macro,
            "worst_task_asr": worst_task,
            "worst_target_family_asr": worst_family,
            "seed_score_std": seed_std,
            "task_asr": task_scores,
            "target_asr": target_scores,
            "target_family_asr": family_scores,
            "control_asr": _group_mean(controls, "target"),
            "primary_rows": len(primary),
            "expected_primary_rows": expected_primary,
            "forward_units_per_item": next(iter(forward_budgets)),
            "model_count": len(model_ids),
            "architecture_families": architecture_families,
            "objective_families": objective_families,
        })
    return sorted(results, key=lambda value: value["selection_score"], reverse=True)


def promote(ranking: list[dict[str, Any]], count: int) -> list[str]:
    eligible = [row for row in ranking if row["eligible"]]
    selected: list[str] = []

    def add(row: dict[str, Any] | None) -> None:
        if row and row["surrogate_set"] not in selected and len(selected) < count:
            selected.append(row["surrogate_set"])

    add(eligible[0] if eligible else None)
    add(next((row for row in eligible if row["model_count"] == 2), None))
    add(next((row for row in eligible if len(row["objective_families"]) > 1), None))
    for row in eligible:
        add(row)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select compute-matched transfer candidates using held-out open-source results only."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--composition-config", required=True)
    parser.add_argument("--stage")
    parser.add_argument("--budget-mode", default="equal_forwards")
    parser.add_argument("--promote", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.results).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ranking = score_candidates(rows, args.composition_config, args.stage, args.budget_mode)
    result = {
        "schema_version": 1,
        "selection_source": "heldout_open_source_only",
        "api_results_used_for_selection": False,
        "budget_mode": args.budget_mode,
        "promoted": promote(ranking, args.promote),
        "ranking": ranking,
    }
    atomic_write_json(Path(args.output), result)
    print(json.dumps({"promoted": result["promoted"], "candidates": len(ranking)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


def _bootstrap_delta(pairs: list[tuple[bool, bool]], samples: int = 2000) -> tuple[float | None, list[float | None]]:
    if not pairs:
        return None, [None, None]
    deltas = [float(candidate) - float(baseline) for baseline, candidate in pairs]
    estimate = sum(deltas) / len(deltas)
    rng = random.Random(0)
    draws = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples))
    return estimate, [draws[int(.025 * (len(draws) - 1))], draws[int(.975 * (len(draws) - 1))]]


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

    baseline = {(trial["stage"], trial["dataset"], trial["seed"]): trial for trial in trials if trial["surrogate_set"] == "single_reference"}
    comparison_rows = []
    for trial in trials:
        reference = baseline.get((trial["stage"], trial["dataset"], trial["seed"]))
        if reference is None or trial is reference:
            continue
        for target in trial["heldout"]["heldout_models"]:
            item_ids = sorted({item_id for model, item_id in trial["item_success"] if model == target} |
                              {item_id for model, item_id in reference["item_success"] if model == target})
            pairs = [(reference["item_success"].get((target, item_id), False), trial["item_success"].get((target, item_id), False)) for item_id in item_ids]
            delta, interval = _bootstrap_delta(pairs)
            comparison_rows.append({"stage": trial["stage"], "dataset": trial["dataset"], "seed": trial["seed"],
                                    "target": target, "baseline": "single_reference", "candidate": trial["surrogate_set"],
                                    "paired_items": len(pairs), "asr_delta": delta, "ci95_low": interval[0], "ci95_high": interval[1]})
    _write_csv(output / "paired_comparisons.csv", comparison_rows)

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
        "",
        "API outcomes are not read by this analyzer.",
    ]
    (output / "ensemble_claim_summary.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    print(json.dumps({"trials": len(trials), "result_rows": len(rows), "paired_comparisons": len(comparison_rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

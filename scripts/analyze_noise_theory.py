#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any


def _number(value: Any) -> Any:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def bootstrap_mean(values: list[float], samples: int, seed: int = 0) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    draws = [statistics.mean(rng.choice(values) for _ in values) for _ in range(samples)]
    return [percentile(draws, .025), percentile(draws, .975)]


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 3:
        return None
    x, y = average_ranks(first), average_ranks(second)
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = sum((a - mx) ** 2 for a in x) ** .5 * sum((b - my) ** 2 for b in y) ** .5
    return numerator / denominator if denominator else None


def classify(fraction: float | None, *, contrary: bool = False) -> str:
    if fraction is None:
        return "not identifiable"
    if fraction >= .75:
        return "supported"
    if fraction >= .5:
        return "partially supported"
    return "contradicted" if contrary and fraction <= .25 else "unsupported"


def read_run_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not names or len(names) != len(set(names)):
        raise ValueError("Run list must contain unique non-empty directory names")
    return names


def heldout_target_metrics(row: dict[str, Any]) -> tuple[float | None, float | None, dict[str, bool]]:
    attack_output = row.get("attack_output")
    target = str(row.get("target") or "")
    if attack_output:
        path = Path(str(attack_output)).parent / "heldout_summary.json"
        if path.is_file():
            summary = json.loads(path.read_text(encoding="utf-8"))
            model = (summary.get("heldout_models") or {}).get(target) or {}
            success = {str(item["item_id"]): bool(item.get("proxy_success"))
                       for item in summary.get("items") or [] if str(item.get("model")) == target}
            if model:
                return _number(model.get("asr")), _number(model.get("mean_margin_gain")), success
    return _number(row.get("heldout_macro_asr")), _number(row.get("heldout_macro_margin_gain")), {}


def aggregate(root: Path, *, bootstrap_samples: int = 2000, orthogonality_tolerance: float = .1,
              run_names: list[str] | None = None) -> dict[str, Any]:
    residual, joins, strength, jensen = [], [], [], []
    manifests = [root / name / "run_manifest.json" for name in run_names] if run_names else sorted(root.glob("*/run_manifest.json"))
    missing = [str(path) for path in manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required run manifests: {missing}")
    for manifest in manifests:
        directory = manifest.parent
        run_id = directory.name
        for target, filename in ((residual, "residual_assumption_tests.csv"), (joins, "alignment_asr_join.csv"),
                                 (strength, "strength_tradeoff.csv"), (jensen, "jensen_gap.csv")):
            for row in read_rows(directory / filename):
                if filename == "alignment_asr_join.csv":
                    target_asr, target_margin, _ = heldout_target_metrics(row)
                    row["heldout_target_asr"] = target_asr
                    row["heldout_target_margin_gain"] = target_margin
                target.append({"run_id": run_id, **row})

    by_definition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in residual:
        by_definition[str(row["kernel_definition"])].append(row)
    assumption_summary = []
    for definition, rows in sorted(by_definition.items()):
        orth = [float(row["residual_orthogonality"]) for row in rows]
        relevance = [float(row["residual_target_relevance"]) for row in rows]
        alignment = [float(row["alignment_delta"]) for row in rows]
        discrepancy = [float(row["discrepancy_delta"]) for row in rows]
        fractions = {
            "residual_orthogonality_fraction": sum(abs(value) <= orthogonality_tolerance for value in orth) / len(orth),
            "residual_target_irrelevance_fraction": sum(value <= 0 for value in relevance) / len(relevance),
            "alignment_improvement_fraction": sum(value > 0 for value in alignment) / len(alignment),
            "discrepancy_improvement_fraction": sum(value < 0 for value in discrepancy) / len(discrepancy),
        }
        assumption_summary.append({
            "kernel_definition": definition, "runs": len(rows), **fractions,
            "mean_residual_orthogonality": statistics.mean(orth),
            "mean_residual_orthogonality_ci95": json.dumps(bootstrap_mean(orth, bootstrap_samples)),
            "mean_residual_target_relevance": statistics.mean(relevance),
            "mean_residual_target_relevance_ci95": json.dumps(bootstrap_mean(relevance, bootstrap_samples, 1)),
            "mean_alignment_delta": statistics.mean(alignment),
            "mean_alignment_delta_ci95": json.dumps(bootstrap_mean(alignment, bootstrap_samples, 2)),
            "mean_discrepancy_delta": statistics.mean(discrepancy),
            "mean_discrepancy_delta_ci95": json.dumps(bootstrap_mean(discrepancy, bootstrap_samples, 3)),
            "residual_orthogonality_evidence": classify(fractions["residual_orthogonality_fraction"], contrary=True),
            "residual_target_irrelevance_evidence": classify(fractions["residual_target_irrelevance_fraction"], contrary=True),
            "alignment_improvement_evidence": classify(fractions["alignment_improvement_fraction"]),
            "discrepancy_improvement_evidence": classify(fractions["discrepancy_improvement_fraction"]),
        })

    paired = []
    clustered_item_deltas: dict[str, list[float]] = defaultdict(list)
    grouped: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in joins:
        key = tuple(row.get(name) for name in ("dataset", "surrogate_group", "seed", "proxy", "target"))
        grouped[key][str(row.get("mode"))] = row
    for key, modes in grouped.items():
        if "none" not in modes or "gaussian_eot" not in modes:
            continue
        baseline, noise = modes["none"], modes["gaussian_eot"]
        baseline_asr, _, baseline_items = heldout_target_metrics(baseline)
        noise_asr, _, noise_items = heldout_target_metrics(noise)
        paired_ids = sorted(set(baseline_items) & set(noise_items))
        item_deltas = [float(noise_items[item]) - float(baseline_items[item]) for item in paired_ids]
        for item, delta in zip(paired_ids, item_deltas):
            clustered_item_deltas[item].append(delta)
        paired.append({
            "dataset": key[0], "surrogate_group": key[1], "seed": key[2], "proxy": key[3], "target": key[4],
            "baseline_heldout_asr": baseline_asr, "noise_heldout_asr": noise_asr,
            "delta_heldout_asr": float(noise_asr) - float(baseline_asr),
            "paired_items": len(paired_ids),
            "delta_heldout_asr_ci95": json.dumps(bootstrap_mean(item_deltas, bootstrap_samples, 11)) if item_deltas else None,
            "used_target_specific_heldout": bool(paired_ids),
            "noise_alignment_delta": noise.get("raw_centered_alignment_delta"),
            "noise_discrepancy_delta": noise.get("raw_centered_discrepancy_delta"),
            "noise_residual_orthogonality": noise.get("raw_centered_residual_orthogonality"),
            "noise_residual_target_relevance": noise.get("raw_centered_residual_target_relevance"),
        })
    correlations = []
    for predictor in ("noise_alignment_delta", "noise_discrepancy_delta", "noise_residual_orthogonality", "noise_residual_target_relevance"):
        valid = [row for row in paired if isinstance(row.get(predictor), (int, float))]
        correlations.append({"predictor": predictor, "outcome": "delta_heldout_asr", "pairs": len(valid),
                             "spearman": spearman([float(row[predictor]) for row in valid], [float(row["delta_heldout_asr"]) for row in valid])})
    cluster_effects = [statistics.mean(values) for _, values in sorted(clustered_item_deltas.items())]
    overall_paired_effect = {
        "target_seed_cells": len(paired), "unique_item_clusters": len(cluster_effects),
        "mean_delta_heldout_asr": statistics.mean(cluster_effects) if cluster_effects else None,
        "clustered_item_bootstrap_ci95": bootstrap_mean(cluster_effects, bootstrap_samples, 12),
    }
    return {"residual": residual, "joins": joins, "strength": strength, "jensen": jensen,
            "assumption_summary": assumption_summary, "paired": paired, "correlations": correlations,
            "overall_paired_effect": overall_paired_effect}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate post-hoc noise-smoothing diagnostics.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--orthogonality-tolerance", type=float, default=.1)
    parser.add_argument("--run-list", help="Text file containing the exact run directory names to aggregate.")
    args = parser.parse_args()
    result = aggregate(Path(args.root), bootstrap_samples=args.bootstrap_samples,
                       orthogonality_tolerance=args.orthogonality_tolerance,
                       run_names=read_run_list(Path(args.run_list)) if args.run_list else None)
    output = Path(args.output)
    write_csv(output / "residual_assumption_tests.csv", result["residual"])
    write_csv(output / "smoothed_kernel_alignment.csv", result["residual"])
    write_csv(output / "alignment_asr_join.csv", result["joins"])
    write_csv(output / "strength_tradeoff.csv", result["strength"])
    write_csv(output / "jensen_gap.csv", result["jensen"])
    write_csv(output / "assumption_summary.csv", result["assumption_summary"])
    write_csv(output / "paired_attack_comparisons.csv", result["paired"])
    write_csv(output / "alignment_asr_correlations.csv", result["correlations"])
    summary = {"completed_runs": len({row["run_id"] for row in result["residual"]}),
               "kernel_definitions": len(result["assumption_summary"]), "paired_attack_comparisons": len(result["paired"]),
               "overall_paired_effect": result["overall_paired_effect"],
               "assumptions": result["assumption_summary"], "correlations": result["correlations"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("completed_runs", "kernel_definitions", "paired_attack_comparisons")}, indent=2))


if __name__ == "__main__":
    main()

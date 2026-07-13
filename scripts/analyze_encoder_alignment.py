#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from alignment_analysis import bootstrap_spearman, spearman


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_alignment(paths: list[tuple[str | None, Path]]) -> list[dict[str, Any]]:
    rows = []
    for dataset, path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                converted = {key: value for key, value in row.items()}
                for key, value in list(converted.items()):
                    if key not in {"proxy", "target"} and value not in {"", None}:
                        try: converted[key] = float(value)
                        except ValueError: pass
                converted["alignment_source"] = str(path.resolve())
                converted["alignment_dataset"] = dataset
                rows.append(converted)
    return rows


def _single_set_map(composition_path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    composition = yaml.safe_load(composition_path.read_text(encoding="utf-8"))
    singles = {name: value["models"][0] for name, value in composition["sets"].items() if len(value["models"]) == 1}
    return singles, composition["models"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Join reusable single-proxy outputs with encoder-alignment metrics.")
    parser.add_argument("--surrogate-results-root", required=True)
    parser.add_argument("--composition-config", required=True)
    parser.add_argument("--alignment-metrics", nargs="+", type=Path)
    parser.add_argument(
        "--alignment-metric-map", action="append", default=[], metavar="DATASET=CSV",
        help="Dataset-specific alignment CSV; repeat for cross-dataset analysis.",
    )
    parser.add_argument(
        "--stages", nargs="+", default=["stage1_single"],
        help="Only join these attack stages; defaults to the inferential single-proxy stage.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    result_root = Path(args.surrogate_results_root)
    single_sets, model_metadata = _single_set_map(Path(args.composition_config))
    mapped_paths: list[tuple[str | None, Path]] = [(None, path) for path in (args.alignment_metrics or [])]
    for value in args.alignment_metric_map:
        if "=" not in value:
            parser.error("--alignment-metric-map must use DATASET=CSV")
        dataset, path = value.split("=", 1)
        mapped_paths.append((dataset, Path(path)))
    if not mapped_paths:
        parser.error("provide --alignment-metrics or --alignment-metric-map")
    alignment_rows = _read_alignment(mapped_paths)
    grouped_alignment = defaultdict(list)
    for row in alignment_rows:
        grouped_alignment[(row.get("alignment_dataset"), row["proxy"], row["target"])].append(row)
    preferred_alignment = {
        key: max(values, key=lambda row: int(row["items"])) for key, values in grouped_alignment.items()
    }
    diagnostic_prefixes = ("neighborhood_overlap_at_", "proxy_neighbor_margin_at_", "target_neighbor_margin_at_")

    item_rows = []
    trial_target_rows = []
    for result_path in sorted(result_root.glob("*/*/trial_result.json")):
        trial = json.loads(result_path.read_text(encoding="utf-8"))
        if trial.get("stage") not in set(args.stages):
            continue
        set_name = trial.get("surrogate_set")
        if set_name not in single_sets:
            continue
        proxy_id = single_sets[set_name]
        heldout_path = result_path.with_name("heldout_summary.json")
        if not heldout_path.is_file():
            continue
        heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
        grouped = defaultdict(list)
        for target_row in heldout["items"]:
            target_id, item_id = target_row["model"], target_row["item_id"]
            attack_metrics_path = result_path.parent / "attack" / item_id / "metrics.json"
            if not attack_metrics_path.is_file():
                continue
            attack_metrics = json.loads(attack_metrics_path.read_text(encoding="utf-8"))
            per_proxy = list(attack_metrics.get("proxy_eval", {}).get("per_surrogate", {}).values())
            proxy_metrics = per_proxy[0] if len(per_proxy) == 1 else {}
            row = {
                "stage": trial["stage"], "dataset": trial["dataset"], "seed": trial["seed"], "item_id": item_id,
                "proxy": proxy_id, "target": target_id,
                "proxy_success": bool(proxy_metrics.get("proxy_success", False)),
                "target_success": bool(target_row.get("proxy_success", False)),
                "proxy_margin_gain": proxy_metrics.get("margin_gain"),
                "target_margin_gain": target_row.get("margin_gain"),
                "dproxy_clean": proxy_metrics.get("clean_target_prototype_distance"),
                "dproxy_adversarial": proxy_metrics.get("adversarial_target_prototype_distance"),
                "delta_dproxy": proxy_metrics.get("target_prototype_distance_change"),
                "dtarget_clean": target_row.get("clean_prototype_distance"),
                "dtarget_adversarial": target_row.get("adversarial_prototype_distance"),
                "delta_dtarget": target_row.get("prototype_distance_change"),
            }
            item_rows.append(row)
            grouped[target_id].append(row)
        for target_id, values in grouped.items():
            alignment = preferred_alignment.get((trial["dataset"], proxy_id, target_id), {})
            if not alignment:
                alignment = preferred_alignment.get((None, proxy_id, target_id), {})
            trial_target_rows.append({
                "stage": trial["stage"], "dataset": trial["dataset"], "seed": trial["seed"], "proxy": proxy_id, "target": target_id,
                "items": len(values),
                "proxy_whitebox_asr": sum(row["proxy_success"] for row in values) / len(values),
                "heldout_asr": sum(row["target_success"] for row in values) / len(values),
                "mean_target_margin_gain": sum(float(row["target_margin_gain"] or 0) for row in values) / len(values),
                "centered_linear_cka": alignment.get("centered_linear_cka"),
                "uncentered_kernel_alignment": alignment.get("uncentered_kernel_alignment"),
                **{key: value for key, value in alignment.items() if key.startswith(diagnostic_prefixes)},
            })

    aggregate = defaultdict(list)
    for row in trial_target_rows:
        aggregate[(row["proxy"], row["target"])].append(row)
    pair_rows = []
    for (proxy, target), values in aggregate.items():
        pair_rows.append({
            "proxy": proxy, "target": target, "trials": len(values),
            "mean_proxy_whitebox_asr": sum(row["proxy_whitebox_asr"] for row in values) / len(values),
            "mean_heldout_asr": sum(row["heldout_asr"] for row in values) / len(values),
            "mean_target_margin_gain": sum(row["mean_target_margin_gain"] for row in values) / len(values),
            "centered_linear_cka": values[0].get("centered_linear_cka"),
            "uncentered_kernel_alignment": values[0].get("uncentered_kernel_alignment"),
            **{key: value for key, value in values[0].items() if key.startswith(diagnostic_prefixes)},
            "proxy_architecture_family": model_metadata.get(proxy, {}).get("architecture_family"),
            "target_architecture_family": model_metadata.get(target, {}).get("architecture_family"),
        })

    correlation_rows = []
    theory_predictors = ["centered_linear_cka", "uncentered_kernel_alignment"] + sorted({
        key for row in pair_rows for key in row if key.startswith(diagnostic_prefixes)
    })
    for predictor in theory_predictors:
        valid = [row for row in pair_rows if isinstance(row.get(predictor), (int, float))]
        result = bootstrap_spearman([row[predictor] for row in valid], [row["mean_heldout_asr"] for row in valid], samples=args.bootstrap_samples)
        correlation_rows.append({"predictor": predictor, "outcome": "mean_heldout_asr", **result})

    distance_valid = [row for row in item_rows if isinstance(row.get("delta_dproxy"), (int, float)) and isinstance(row.get("delta_dtarget"), (int, float))]
    distance_correlations = []
    for predictor, outcome in (("delta_dproxy", "delta_dtarget"), ("delta_dproxy", "target_success"),
                               ("delta_dtarget", "target_success")):
        valid = [row for row in item_rows if isinstance(row.get(predictor), (int, float)) and isinstance(row.get(outcome), (int, float, bool))]
        result = bootstrap_spearman([float(row[predictor]) for row in valid], [float(row[outcome]) for row in valid], samples=args.bootstrap_samples)
        distance_correlations.append({"predictor": predictor, "outcome": outcome, **result})
    successes = [row for row in distance_valid if row["target_success"]]
    failures = [row for row in distance_valid if not row["target_success"]]
    distance_summary = {
        "rows": len(distance_valid), "successful_rows": len(successes), "failed_rows": len(failures),
        "mean_delta_dtarget_success": sum(float(row["delta_dtarget"]) for row in successes) / len(successes) if successes else None,
        "mean_delta_dtarget_failure": sum(float(row["delta_dtarget"]) for row in failures) / len(failures) if failures else None,
        "correlations": distance_correlations,
    }

    cka_no_rows = []
    for alignment_name in ("centered_linear_cka", "uncentered_kernel_alignment"):
        for no_name in sorted({key for row in alignment_rows for key in row if key.startswith("neighborhood_overlap_at_")}):
            valid = [row for row in alignment_rows if isinstance(row.get(alignment_name), (int, float)) and isinstance(row.get(no_name), (int, float))]
            result = bootstrap_spearman([float(row[alignment_name]) for row in valid], [float(row[no_name]) for row in valid], samples=args.bootstrap_samples)
            cka_no_rows.append({"predictor": alignment_name, "outcome": no_name, **result})

    family_rows = []
    family_groups = defaultdict(list)
    for row in pair_rows:
        family_groups[(row["proxy_architecture_family"], row["target_architecture_family"])].append(row)
        family_groups[("same_architecture" if row["proxy_architecture_family"] == row["target_architecture_family"] else "different_architecture", "all")].append(row)
    for (proxy_family, target_family), rows in family_groups.items():
        for predictor in ("centered_linear_cka", "uncentered_kernel_alignment", "neighborhood_overlap_at_1"):
            valid = [row for row in rows if isinstance(row.get(predictor), (int, float))]
            result = bootstrap_spearman([float(row[predictor]) for row in valid], [float(row["mean_heldout_asr"]) for row in valid], samples=args.bootstrap_samples)
            family_rows.append({"proxy_family": proxy_family, "target_family": target_family, "predictor": predictor,
                                "outcome": "mean_heldout_asr", **result})

    sensitivity_rows = []
    sensitivity_groups = defaultdict(list)
    for row in alignment_rows:
        sensitivity_groups[(row.get("alignment_dataset"), int(row["items"]))].append(row)
    sample_groups = sorted(sensitivity_groups)
    for index, first_key in enumerate(sample_groups):
        first = {(row["proxy"], row["target"]): row for row in sensitivity_groups[first_key]}
        for second_key in sample_groups[index + 1:]:
            second = {(row["proxy"], row["target"]): row for row in sensitivity_groups[second_key]}
            common = sorted(first.keys() & second.keys())
            for metric in ("centered_linear_cka", "uncentered_kernel_alignment", "neighborhood_overlap_at_1"):
                valid = [key for key in common if isinstance(first[key].get(metric), (int, float)) and isinstance(second[key].get(metric), (int, float))]
                sensitivity_rows.append({"first_dataset": first_key[0], "first_items": first_key[1],
                    "second_dataset": second_key[0], "second_items": second_key[1], "metric": metric, "pairs": len(valid),
                    "ranking_spearman": spearman([float(first[key][metric]) for key in valid], [float(second[key][metric]) for key in valid])})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "pairwise_alignment_metrics.csv", alignment_rows)
    _write_csv(output / "pairwise_transfer_matrix.csv", pair_rows)
    _write_csv(output / "cka_transfer_analysis.csv", correlation_rows)
    _write_csv(output / "no_transfer_analysis.csv", [row for row in correlation_rows if row["predictor"].startswith("neighborhood_overlap")])
    _write_csv(output / "dproxy_dtarget_analysis.csv", item_rows)
    _write_csv(output / "distance_chain_correlations.csv", distance_correlations)
    _write_csv(output / "cka_no_relationship.csv", cka_no_rows)
    _write_csv(output / "family_stratified_transfer.csv", family_rows)
    _write_csv(output / "alignment_sample_sensitivity.csv", sensitivity_rows)
    (output / "distance_summary.json").write_text(json.dumps(distance_summary, indent=2), encoding="utf-8")
    summary = ["# Encoder Alignment Claim Summary", "", f"Joined item rows: {len(item_rows)}.", f"Proxy--target pairs: {len(pair_rows)}.", "",
               "Claims remain not identifiable when the single-proxy matrix is absent or too small.",
               "No proprietary architecture assumptions or API selection are used."]
    (output / "alignment_claim_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"item_rows": len(item_rows), "pair_rows": len(pair_rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

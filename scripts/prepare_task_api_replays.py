#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from freeze_api_candidates import freeze_manifest
except ModuleNotFoundError:  # Imported as scripts.prepare_task_api_replays in tests.
    from scripts.freeze_api_candidates import freeze_manifest
from transfer_eval import atomic_write_json, run_replay


def parse_dataset_configs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset-config must use DATASET=YAML")
        dataset, config = value.split("=", 1)
        if not dataset or not config:
            raise ValueError("--dataset-config must use DATASET=YAML")
        result[dataset] = Path(config)
    return result


def prepare_replays(*, analysis_csv: Path, output: Path, dataset_configs: dict[str, Path],
                    stage: str, budget_mode: str, baseline: str, top_k: int,
                    minimum_seeds: int, minimum_targets: int) -> dict:
    if not dataset_configs:
        raise ValueError("At least one dataset config is required")
    output.mkdir(parents=True, exist_ok=True)
    plans = {}
    for dataset, config in sorted(dataset_configs.items()):
        frozen = output / f"frozen_{dataset}.json"
        manifest = freeze_manifest(
            analysis_csv, frozen, stage=stage, budget_mode=budget_mode,
            top_k=top_k, minimum_seeds=minimum_seeds, minimum_targets=minimum_targets,
            datasets={dataset}, baseline=baseline,
        )
        result_dir = output / f"dry_run_{dataset}"
        estimate = run_replay(
            config, [], result_dir, dry_run=True, allow_real_api=False,
            frozen_candidate_manifest=frozen,
        )
        plans[dataset] = {
            "config": str(config.resolve()),
            "frozen_manifest": str(frozen.resolve()),
            "freeze_hash": manifest["freeze_hash"],
            "candidates": [row["surrogate_set"] for row in manifest["candidates"]],
            **estimate,
        }
    combined = {
        "real_api": False,
        "selection_source": "heldout_open_source_only",
        "api_results_used_for_selection": False,
        "datasets": plans,
        "total_estimated_requests": sum(plan["estimated_requests"] for plan in plans.values()),
        "total_evaluation_records": sum(plan["evaluation_records"] for plan in plans.values()),
    }
    atomic_write_json(output / "api_replay_plan.json", combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one candidate manifest and API dry-run per task dataset.")
    parser.add_argument("--analysis-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-config", action="append", default=[], metavar="DATASET=YAML")
    parser.add_argument("--stage", default="stage3_cross_dataset")
    parser.add_argument("--budget-mode", default="equal_steps")
    parser.add_argument("--baseline", default="single_reference")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--minimum-targets", type=int, default=1)
    args = parser.parse_args()
    try:
        configs = parse_dataset_configs(args.dataset_config)
        result = prepare_replays(
            analysis_csv=Path(args.analysis_csv), output=Path(args.output), dataset_configs=configs,
            stage=args.stage, budget_mode=args.budget_mode, baseline=args.baseline,
            top_k=args.top_k, minimum_seeds=args.minimum_seeds, minimum_targets=args.minimum_targets,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

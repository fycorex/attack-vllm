#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

try:
    from freeze_stage3_surrogate_spec import freeze
    from run_ensemble_gated_pipeline import run_parallel, stage_specs, validate_stage, write_state
except ModuleNotFoundError:  # Imported as scripts.run_cross_dataset_validation in tests.
    from scripts.freeze_stage3_surrogate_spec import freeze
    from scripts.run_ensemble_gated_pipeline import run_parallel, stage_specs, validate_stage, write_state


SELECTION_STAGE = "stage2_equal_forwards"
VALIDATION_STAGE = "stage3_cross_dataset"


def expected_trial_count(spec: dict, stage: str) -> int:
    value = spec["stages"][stage]
    return len(value["datasets"]) * len(value["sets"]) * len(value["seeds"])


def parse_manifest_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset-manifest must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError("--dataset-manifest must use NAME=PATH")
        overrides[name] = Path(path)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze held-out-selected methods and optionally launch unique-image cross-dataset validation."
    )
    parser.add_argument("--base-spec", default="configs/surrogate_experiments.yaml")
    parser.add_argument("--selection-root", default="outputs/surrogate_experiments")
    parser.add_argument("--analysis-output", default="outputs/surrogate_analysis")
    parser.add_argument("--validation-root", default="outputs/surrogate_cross_dataset")
    parser.add_argument("--frozen-spec", default="outputs/surrogate_cross_dataset/frozen_spec.yaml")
    parser.add_argument("--items", type=int, default=19)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--validation-stage", default=VALIDATION_STAGE)
    parser.add_argument("--dataset-manifest", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="models/open_clip")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--launch", action="store_true", help="Run attacks after all validation gates pass.")
    args = parser.parse_args()

    base_path = Path(args.base_spec)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    validation_stage = args.validation_stage
    if validation_stage not in base.get("stages", {}):
        parser.error(f"unknown validation stage: {validation_stage}")
    expected_selection = expected_trial_count(base, SELECTION_STAGE)
    validate_stage(Path(args.selection_root), SELECTION_STAGE, expected_selection)

    analysis_output = Path(args.analysis_output)
    subprocess.run([
        args.python,
        "scripts/analyze_surrogate_experiments.py",
        "--root", args.selection_root,
        "--output", str(analysis_output),
        "--bootstrap-samples", str(args.bootstrap_samples),
    ], check=True)
    analysis_csv = analysis_output / "all_transfer_results.csv"
    overrides = parse_manifest_overrides(args.dataset_manifest)
    required_datasets = set(base["stages"][validation_stage]["datasets"])
    if set(overrides) != required_datasets:
        missing = sorted(required_datasets - set(overrides))
        extra = sorted(set(overrides) - required_datasets)
        parser.error(f"provide one audited manifest for every validation dataset; missing={missing}, extra={extra}")

    frozen_path = Path(args.frozen_spec)
    evidence = freeze(
        base_path,
        analysis_csv,
        frozen_path,
        stage=SELECTION_STAGE,
        budget_mode="equal_forwards",
        selection_dataset="caption_caltech",
        baseline="single_reference",
        top_k=args.top_k,
        minimum_seeds=args.minimum_seeds,
        stage3_items=args.items,
        dataset_manifests=overrides,
        target_stage=validation_stage,
    )
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    selected = frozen["stages"][validation_stage]["sets"]
    expected_validation = expected_trial_count(frozen, validation_stage)
    plan = {
        "selection_trials_validated": expected_selection,
        "selection_source": evidence["selection_source"],
        "api_results_used_for_selection": evidence["api_results_used_for_selection"],
        "selected_methods": selected,
        "validation_stage": validation_stage,
        "budget_mode": frozen["stages"][validation_stage]["budget_mode"],
        "datasets": frozen["stages"][validation_stage]["datasets"],
        "items_per_trial": args.items,
        "seeds": frozen["stages"][VALIDATION_STAGE]["seeds"],
        "expected_validation_trials": expected_validation,
        "freeze_hash": evidence["freeze_hash"],
        "launch": args.launch,
    }
    print(json.dumps(plan, indent=2))
    if not args.launch:
        return

    validation_root = Path(args.validation_root)
    state = validation_root / "cross_dataset_validation_state.json"
    write_state(state, status="running", **plan)
    spec_dir = validation_root / "generated_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    run_parallel(
        stage_specs(frozen, validation_stage, spec_dir),
        validation_stage,
        validation_root,
        args.python,
        args.device,
        args.cache_dir,
    )
    validate_stage(validation_root, validation_stage, expected_validation)
    subprocess.run([
        args.python,
        "scripts/analyze_surrogate_experiments.py",
        "--root", str(validation_root),
        "--output", str(validation_root / "analysis"),
        "--bootstrap-samples", str(args.bootstrap_samples),
    ], check=True)
    write_state(state, status="complete", **plan)


if __name__ == "__main__":
    main()

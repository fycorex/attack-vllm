#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from experiment_validation import validate_attack_output
from surrogate_composition import build_attack_config, load_composition_spec, stage_trials
from transfer_eval import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable single/ensemble surrogate experiments without API access.")
    parser.add_argument("--spec", default="configs/surrogate_experiments.yaml")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--root", default="outputs/surrogate_experiments")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="models/open_clip")
    parser.add_argument("--heldout-batch-size", type=int, default=16)
    parser.add_argument("--defer-heldout", action="store_true",
                        help="Write validated pending trials for the stage-level batched evaluator.")
    args = parser.parse_args()
    experiment = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    composition_path = Path(experiment["composition_config"])
    composition = load_composition_spec(composition_path)
    trials = stage_trials(experiment, composition, args.stage)
    if args.plan:
        print(json.dumps({"stage": args.stage, "trials": trials, "count": len(trials),
                          "total_forward_units_per_item": sum(trial["forward_units_per_item"] for trial in trials)}, indent=2))
        return

    root = Path(args.root) / args.stage
    root.mkdir(parents=True, exist_ok=True)
    for trial in trials:
        directory = root / f"{trial['dataset']}__{trial['surrogate_set']}__seed{trial['seed']}__{trial['trial_id']}"
        result_path = directory / "trial_result.json"
        if result_path.is_file():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        dataset = experiment["datasets"][trial["dataset"]]
        base = yaml.safe_load(Path(dataset["base_attack_config"]).read_text(encoding="utf-8"))
        effective = build_attack_config(base, experiment, composition, trial, directory, device=args.device)
        effective_path = directory / "effective_attack.yaml"
        effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        attack_command = [sys.executable, "scripts/run_caption_attack.py", "--config", str(effective_path)]
        heldout_path = directory / "heldout_summary.json"
        heldout_command = [sys.executable, "scripts/evaluate_surrogate_heldout.py", "--composition-config", str(composition_path),
                           "--manifest", dataset["manifest"], "--attack-output", str(directory / "attack"),
                           "--output", str(heldout_path), "--device", args.device, "--cache-dir", args.cache_dir,
                           "--limit", str(trial["items"]), "--batch-size", str(args.heldout_batch_size)]
        with (directory / "attack.log").open("a", encoding="utf-8") as log:
            subprocess.run(attack_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        validation = validate_attack_output(
            directory / "attack",
            expected_items=trial["items"],
            epsilon=float(effective["attack"]["epsilon"]),
            expected_forward_units_per_item=trial["forward_units_per_item"],
        )
        if args.defer_heldout:
            atomic_write_json(directory / "trial_pending.json", {
                "trial": trial,
                "manifest": dataset["manifest"],
                "items": trial["items"],
                "attack_output": str(directory / "attack"),
                "heldout_output": str(heldout_path),
                "attack_validation": validation,
                "attack_command": attack_command,
            })
            continue
        with (directory / "heldout.log").open("a", encoding="utf-8") as log:
            subprocess.run(heldout_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        heldout = json.loads(heldout_path.read_text(encoding="utf-8"))
        atomic_write_json(result_path, {**trial, "status": "complete", "heldout_macro_asr": heldout["heldout_macro_asr"],
                                        "heldout_macro_margin_gain": heldout["heldout_macro_margin_gain"],
                                        "attack_validation": validation,
                                        "attack_command": attack_command, "heldout_command": heldout_command})
    results = [json.loads(path.read_text()) for path in root.glob("*/trial_result.json")]
    atomic_write_json(root / "stage_summary.json", {"stage": args.stage, "planned": len(trials), "completed": len(results), "results": results})


if __name__ == "__main__":
    main()

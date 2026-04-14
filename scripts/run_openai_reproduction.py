#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attack_vlm_repro.attack import run_attack
from attack_vlm_repro.config import load_config


CONFIG_MATRIX = [
    ("gpt-4o", "caption", Path("configs/caption_attack_phase2_gpt4o_caption.yaml")),
    ("gpt-4o", "vqa", Path("configs/caption_attack_phase2_gpt4o_vqa.yaml")),
    ("gpt-5", "caption", Path("configs/caption_attack_phase2_gpt5_caption.yaml")),
    ("gpt-5", "vqa", Path("configs/caption_attack_phase2_gpt5_vqa.yaml")),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GPT-4o / GPT-5 reproduction configs for captioning and VQA."
    )
    parser.add_argument(
        "--model",
        choices=["all", "gpt-4o", "gpt-5"],
        default="all",
        help="Filter the config matrix by victim model.",
    )
    parser.add_argument(
        "--task",
        choices=["all", "caption", "vqa"],
        default="all",
        help="Filter the config matrix by task.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Optional explicit config path. Can be passed multiple times to bypass the built-in matrix.",
    )
    parser.add_argument("--attack_limit", type=int, default=None, help="Optional override for attacked items.")
    parser.add_argument("--attack_offset", type=int, default=None, help="Optional starting item offset.")
    parser.add_argument("--steps", type=int, default=None, help="Optional override for attack optimization steps.")
    args = parser.parse_args()

    selected_configs: list[Path]
    if args.config:
        selected_configs = [Path(path) for path in args.config]
    else:
        selected_configs = [
            path
            for model_name, task_name, path in CONFIG_MATRIX
            if (args.model == "all" or model_name == args.model) and (args.task == "all" or task_name == args.task)
        ]

    if not selected_configs:
        raise SystemExit("No configs selected.")

    run_summaries = []
    for config_path in selected_configs:
        config = load_config(config_path)
        if args.attack_limit is not None:
            config.runtime.attack_limit = args.attack_limit
        if args.attack_offset is not None:
            config.runtime.attack_offset = args.attack_offset
        if args.steps is not None:
            config.attack.steps = args.steps

        summary = run_attack(config)
        condensed = {
            "config": str(config_path),
            "experiment_name": summary["experiment_name"],
            "dataset_name": summary["dataset_name"],
            "num_items": summary["num_items"],
            "proxy_success_rate": summary["proxy_success_rate"],
            "gpt_success_rate": summary.get("gpt_success_rate"),
            "output_dir": config.paths.output_dir,
        }
        run_summaries.append(condensed)
        print(json.dumps(condensed, indent=2))

    print(json.dumps({"runs": run_summaries}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from attack_vlm_repro.ablations import apply_ablation, list_default_ablations
from attack_vlm_repro.attack import run_attack
from attack_vlm_repro.config import load_config


SUMMARY_FIELDNAMES = [
    "variant",
    "experiment_name",
    "num_items",
    "proxy_success_rate",
    "average_margin_gain",
    "caption_eval_count",
    "caption_success_rate",
    "vqa_eval_count",
    "vqa_success_rate",
    "ocr_eval_count",
    "ocr_success_rate",
    "gpt_eval_count",
    "gpt_success_rate",
    "output_dir",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablations for the small vision-language attack reproduction.")
    parser.add_argument("--config", required=True, help="Base YAML config path.")
    parser.add_argument(
        "--variants",
        default=",".join(list_default_ablations()),
        help="Comma-separated ablation variants. Defaults to the built-in Phase 2 set.",
    )
    parser.add_argument("--output_root", default=None, help="Optional root directory for ablation outputs.")
    parser.add_argument(
        "--attack_limit",
        type=int,
        default=None,
        help="Optional override for the number of attacked items, useful for smoke-test ablations.",
    )
    parser.add_argument(
        "--attack_offset",
        type=int,
        default=None,
        help="Optional starting item offset within the manifest for quicker targeted ablations.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional override for attack optimization steps, useful for faster ablation sweeps.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    output_root = Path(args.output_root) if args.output_root else Path(base_config.paths.output_dir).parent / "phase2_ablations"
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    for variant in variants:
        cfg = apply_ablation(base_config, variant)
        if args.attack_limit is not None:
            cfg.runtime.attack_limit = args.attack_limit
        if args.attack_offset is not None:
            cfg.runtime.attack_offset = args.attack_offset
        if args.steps is not None:
            cfg.attack.steps = args.steps
        cfg.experiment_name = f"{base_config.experiment_name}-{variant}"
        cfg.paths.output_dir = str((output_root / variant).resolve())
        summary = run_attack(cfg)
        aggregate_rows.append(
            {
                "variant": variant,
                "experiment_name": summary["experiment_name"],
                "num_items": summary["num_items"],
                "proxy_success_rate": summary["proxy_success_rate"],
                "average_margin_gain": summary["average_margin_gain"],
                "caption_eval_count": summary.get("caption_eval_count", 0),
                "caption_success_rate": summary.get("caption_success_rate"),
                "vqa_eval_count": summary.get("vqa_eval_count", 0),
                "vqa_success_rate": summary.get("vqa_success_rate"),
                "ocr_eval_count": summary.get("ocr_eval_count", 0),
                "ocr_success_rate": summary.get("ocr_success_rate"),
                "gpt_eval_count": summary.get("gpt_eval_count", 0),
                "gpt_success_rate": summary.get("gpt_success_rate"),
                "per_surrogate_proxy_summary": summary.get("per_surrogate_proxy_summary", {}),
                "output_dir": cfg.paths.output_dir,
            }
        )

    json_path = output_root / "ablation_summary.json"
    json_path.write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")

    csv_path = output_root / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            [
                {key: row[key] for key in SUMMARY_FIELDNAMES}
                for row in aggregate_rows
            ]
        )

    print(json.dumps(aggregate_rows, indent=2))


if __name__ == "__main__":
    main()

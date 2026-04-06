#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from attack_vlm_repro.ablations import apply_ablation, list_default_ablations
from attack_vlm_repro.attack import run_attack
from attack_vlm_repro.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Phase 2 ablation suite for the caption-attack reproduction.")
    parser.add_argument("--config", required=True, help="Base YAML config path.")
    parser.add_argument(
        "--variants",
        default=",".join(list_default_ablations()),
        help="Comma-separated ablation variants. Defaults to the built-in Phase 2 set.",
    )
    parser.add_argument("--output_root", default=None, help="Optional root directory for ablation outputs.")
    args = parser.parse_args()

    base_config = load_config(args.config)
    variants = [variant.strip() for variant in args.variants.split(",") if variant.strip()]
    output_root = Path(args.output_root) if args.output_root else Path(base_config.paths.output_dir).parent / "phase2_ablations"
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows = []
    for variant in variants:
        cfg = apply_ablation(base_config, variant)
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
                "caption_success_rate": summary.get("caption_success_rate"),
                "output_dir": cfg.paths.output_dir,
            }
        )

    json_path = output_root / "ablation_summary.json"
    json_path.write_text(json.dumps(aggregate_rows, indent=2), encoding="utf-8")

    csv_path = output_root / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "experiment_name",
                "num_items",
                "proxy_success_rate",
                "average_margin_gain",
                "caption_success_rate",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(json.dumps(aggregate_rows, indent=2))


if __name__ == "__main__":
    main()

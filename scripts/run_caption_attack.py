#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from attack_vlm_repro.attack import run_attack
from attack_vlm_repro.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the small vision-language attack reproduction.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--attack_limit",
        type=int,
        default=None,
        help="Optional override for the number of attacked items, useful for smoke tests.",
    )
    parser.add_argument(
        "--attack_offset",
        type=int,
        default=None,
        help="Optional starting item offset within the manifest, useful for probing different pairs.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional override for attack optimization steps, useful for faster local debugging.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.attack_limit is not None:
        config.runtime.attack_limit = args.attack_limit
    if args.attack_offset is not None:
        config.runtime.attack_offset = args.attack_offset
    if args.steps is not None:
        config.attack.steps = args.steps

    summary = run_attack(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

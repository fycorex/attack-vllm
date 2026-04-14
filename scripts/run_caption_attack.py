#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys

from attack_vlm_repro.attack import run_attack
from attack_vlm_repro.config import SUPPORTED_PROFILES, apply_profile, enabled_surrogate_names, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the small vision-language attack reproduction.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--profile",
        choices=SUPPORTED_PROFILES,
        default=None,
        help="Optional runtime profile: light for smoke runs, heavy for A6000-class ensembles, api for live API transfer evaluation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print resolved profile and override details before running. Use --profile, not --verbose, to choose light/heavy/api.",
    )
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
    apply_profile(config, args.profile)
    cli_overrides = {}
    if args.attack_limit is not None:
        config.runtime.attack_limit = args.attack_limit
        cli_overrides["runtime.attack_limit"] = args.attack_limit
    if args.attack_offset is not None:
        config.runtime.attack_offset = args.attack_offset
        cli_overrides["runtime.attack_offset"] = args.attack_offset
    if args.steps is not None:
        config.attack.steps = args.steps
        cli_overrides["attack.steps"] = args.steps
    config.profile_metadata["cli_overrides"] = cli_overrides
    config.profile_metadata["enabled_surrogates_after_cli_overrides"] = enabled_surrogate_names(config)

    if args.verbose:
        print(
            json.dumps(
                {
                    "profile": config.run_profile,
                    "profile_metadata": config.profile_metadata,
                    "output_dir": config.paths.output_dir,
                },
                indent=2,
            ),
            file=sys.stderr,
        )

    summary = run_attack(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

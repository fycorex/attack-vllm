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
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Optional override for the base adversarial image size before per-surrogate resizing.",
    )
    parser.add_argument(
        "--augmentation_batches",
        type=int,
        default=None,
        help="Optional override for stochastic augmentation batches per attack step.",
    )
    parser.add_argument(
        "--augmentation_forward_batch_size",
        type=int,
        default=None,
        help="Optional number of stochastic augmentations to evaluate in one surrogate forward.",
    )
    parser.add_argument(
        "--metrics_interval",
        type=int,
        default=None,
        help="Optional attack-history metric interval. Higher values reduce CUDA synchronization overhead.",
    )
    parser.add_argument(
        "--parallel_surrogates",
        action="store_true",
        help="Keep all enabled surrogates resident on GPU instead of moving each model every step.",
    )
    parser.add_argument(
        "--disable_jpeg",
        action="store_true",
        help="Disable JPEG augmentation to avoid CPU/PIL round trips during faster runs.",
    )
    parser.add_argument(
        "--jpeg_backend",
        choices=("tensor", "pil"),
        default=None,
        help="Optional JPEG augmentation backend. tensor is faster; pil preserves the older CPU path.",
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
    if args.image_size is not None:
        config.attack.image_size = args.image_size
        cli_overrides["attack.image_size"] = args.image_size
    if args.augmentation_batches is not None:
        config.attack.augmentation_batches = args.augmentation_batches
        cli_overrides["attack.augmentation_batches"] = args.augmentation_batches
    if args.augmentation_forward_batch_size is not None:
        config.attack.augmentation_forward_batch_size = args.augmentation_forward_batch_size
        cli_overrides["attack.augmentation_forward_batch_size"] = args.augmentation_forward_batch_size
    if args.metrics_interval is not None:
        config.attack.metrics_interval = args.metrics_interval
        cli_overrides["attack.metrics_interval"] = args.metrics_interval
    if args.parallel_surrogates:
        config.runtime.sequential_surrogates = False
        cli_overrides["runtime.sequential_surrogates"] = False
    if args.disable_jpeg:
        config.attack.enable_jpeg = False
        cli_overrides["attack.enable_jpeg"] = False
    if args.jpeg_backend is not None:
        config.attack.jpeg_backend = args.jpeg_backend
        cli_overrides["attack.jpeg_backend"] = args.jpeg_backend
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

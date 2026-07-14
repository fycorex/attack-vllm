#!/usr/bin/env python3
"""Prepare 200 unique Caltech images and re-measure cross-family geometry.

This is deliberately a measurement-only confirmation: it tests the stability of
alignment rankings at 200 unique natural images. It does not present 200 image
representations as 200 targeted-attack trials and does not call an external API.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the independent 200-image cross-family geometry confirmation.")
    parser.add_argument("--output-root", default="outputs/geometry_confirmation_caltech200")
    parser.add_argument("--cycle-root", help="Completed search output used to load frozen candidates for the attack confirmation.")
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", default="models/open_clip")
    parser.add_argument("--attack-workers", type=int, default=2)
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--api-limit", type=int, default=50)
    args = parser.parse_args()
    if args.items < 200:
        parser.error("This confirmation is defined for at least 200 unique images.")

    output_root = (ROOT / args.output_root).resolve()
    data_root = output_root / "dataset"
    manifest = data_root / "manifest.json"
    preparation = [
        sys.executable, "scripts/prepare_caltech_demo.py",
        "--output_dir", str(data_root), "--num_items", str(args.items),
        "--num_examples", str(args.examples),
    ]
    subprocess.run(preparation, cwd=ROOT, check=True)
    measurement = [
        sys.executable, "scripts/measure_surrogate_geometry.py",
        "--config", "configs/research_cycle_cross_family.yaml",
        "--manifest", str(manifest),
        "--sets", "single_openclip_vit", "single_openclip_resnet",
        "single_openclip_convnext", "single_siglip", "single_eva_clip",
        "two_clip_architectures", "two_cross_objective", "four_cross_family",
        "--limit", str(args.items), "--require-limit", "--deduplicate-images",
        "--output", str(output_root / "measurements"),
        "--device", "cuda", "--batch-size", str(args.batch_size),
        "--cache-dir", args.cache_dir,
        "--subsample-sizes", "20", "50", "100", "200",
        "--subsample-repeats", "100",
    ]
    subprocess.run(measurement, cwd=ROOT, check=True)
    if args.cycle_root:
        attack_confirmation = [
            sys.executable, "scripts/run_frozen_200_attack_confirmation.py",
            "--cycle-root", str(Path(args.cycle_root).resolve()), "--manifest", str(manifest),
            "--output-root", str(output_root / "frozen_attack_confirmation"),
            "--items", str(args.items), "--workers", str(args.attack_workers),
            "--cache-dir", args.cache_dir,
        ]
        if args.run_api:
            attack_confirmation.extend(["--run-api", "--api-limit", str(args.api_limit)])
        subprocess.run(attack_confirmation, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

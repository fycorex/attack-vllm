#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from attack_vlm_repro.attack import run_attack_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the small vision-language attack reproduction.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    summary = run_attack_from_config(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

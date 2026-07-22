#!/usr/bin/env python3
"""Estimate remaining pilot time without claiming a measured GPU result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate(seconds_per_step: list[float], test_pairs: int) -> dict[str, float | bool | int]:
    if len(seconds_per_step) != 3 or any(value <= 0 for value in seconds_per_step):
        raise ValueError("Provide exactly three positive proxy seconds-per-step values.")
    development_equivalents = 3 * 3 * 6 * 15 * 2
    final_equivalents = 3 * test_pairs * 30 * 2
    attack_hours = sum(seconds_per_step) * (development_equivalents + final_equivalents) / 3 / 3600
    total_hours = 9.0 + attack_hours
    return {"test_pairs": test_pairs, "attack_hours": round(attack_hours, 3), "projected_total_hours": round(total_hours, 3), "margin_to_24_hours": round(24.0 - total_hours, 3), "run_full_test": total_hours <= 22.0, "runtime_fallback": total_hours > 22.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds-per-step", type=float, nargs=3, metavar=("P1", "P2", "P3"), default=[2.2, 0.25, 0.4])
    parser.add_argument("--test-pairs", type=int, default=12)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--planning-only", action="store_true")
    args = parser.parse_args()
    result = estimate(args.seconds_per_step, args.test_pairs)
    result["planning_only"] = args.planning_only
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

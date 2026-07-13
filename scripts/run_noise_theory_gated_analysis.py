#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.analyze_noise_theory import read_run_list


REQUIRED = ("run_manifest.json", "jensen_gap.csv", "residual_assumption_tests.csv",
            "smoothed_kernel_alignment.csv", "alignment_asr_join.csv", "strength_tradeoff.csv")


def completed_runs(root: Path, run_names: list[str] | None = None) -> list[Path]:
    complete = []
    directories = [root / name for name in run_names] if run_names else sorted(path for path in root.iterdir() if path.is_dir())
    for directory in directories:
        if all((directory / name).is_file() for name in REQUIRED):
            complete.append(directory)
    return complete


def write_state(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(), **values}, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for complete noise-smoothing runs, then aggregate exactly once.")
    parser.add_argument("--root", default="outputs/noise_theory")
    parser.add_argument("--output", default="outputs/noise_theory_aggregate")
    parser.add_argument("--expected-runs", type=int, default=12)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-list", help="Text file containing the exact preregistered run directory names.")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    run_names = read_run_list(Path(args.run_list)) if args.run_list else None
    expected = len(run_names) if run_names is not None else args.expected_runs
    if run_names is not None and args.expected_runs != expected:
        raise ValueError(f"--expected-runs={args.expected_runs} does not match {expected} entries in --run-list")
    state = output / "gate_state.json"
    analysis_command = [args.python, "scripts/analyze_noise_theory.py", "--root", str(root),
                        "--output", str(output), "--bootstrap-samples", str(args.bootstrap_samples)]
    if args.run_list:
        analysis_command.extend(["--run-list", args.run_list])
    if args.plan:
        complete = completed_runs(root, run_names)
        print(json.dumps({"completed": len(complete), "expected": expected,
                          "missing": [name for name in (run_names or []) if root / name not in complete],
                          "command": analysis_command}, indent=2))
        return
    while len(completed_runs(root, run_names)) < expected:
        complete = completed_runs(root, run_names)
        write_state(state, status="waiting", completed=len(complete), expected=expected,
                    missing=[name for name in (run_names or []) if root / name not in complete])
        time.sleep(args.poll_seconds)
    runs = completed_runs(root, run_names)
    if len(runs) != expected:
        raise RuntimeError(f"Expected exactly {expected} complete runs, found {len(runs)}")
    write_state(state, status="aggregating", completed=len(runs), expected=expected)
    subprocess.run(analysis_command, check=True)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if int(summary["completed_runs"]) != expected:
        raise RuntimeError("Aggregate summary run count does not match gate")
    write_state(state, status="complete", completed=len(runs), expected=expected,
                paired_attack_comparisons=summary["paired_attack_comparisons"])


if __name__ == "__main__":
    main()

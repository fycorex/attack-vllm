#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


def completed_trials(root: Path, stage: str) -> list[dict]:
    rows = []
    for path in sorted((root / stage).glob("*/trial_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["result_path"] = str(path)
        rows.append(row)
    return rows


def validate_stage(root: Path, stage: str, expected: int) -> list[dict]:
    rows = completed_trials(root, stage)
    if len(rows) != expected:
        raise RuntimeError(f"{stage}: expected {expected} completed trials, found {len(rows)}")
    ids = [row["trial_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{stage}: duplicate trial IDs")
    failures = []
    for row in rows:
        if row.get("status") != "complete":
            failures.append((row.get("trial_id"), "status", row.get("status")))
        validation = row.get("attack_validation") or {}
        if not validation.get("valid"):
            failures.append((row.get("trial_id"), "attack_validation", validation))
        if int(validation.get("expected_forward_units_per_item", -1)) != int(row["forward_units_per_item"]):
            failures.append((row.get("trial_id"), "forward_units", validation))
    if failures:
        raise RuntimeError(f"{stage}: validation failures: {failures[:5]}")
    return rows


def write_state(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(), **values}, indent=2), encoding="utf-8")
    temporary.replace(path)


def stage_specs(base: dict, stage: str, directory: Path) -> list[Path]:
    paths = []
    for set_name in base["stages"][stage]["sets"]:
        value = copy.deepcopy(base)
        value["stages"] = {stage: copy.deepcopy(base["stages"][stage])}
        value["stages"][stage]["sets"] = [set_name]
        path = directory / f"{stage}__{set_name}.yaml"
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def run_parallel(specs: list[Path], stage: str, root: Path, python: str, device: str, cache_dir: str) -> None:
    processes = []
    for spec in specs:
        command = [python, "scripts/run_surrogate_experiments.py", "--spec", str(spec), "--stage", stage,
                   "--root", str(root), "--device", device, "--cache-dir", cache_dir]
        processes.append((spec, subprocess.Popen(command)))
    failures = []
    for spec, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((str(spec), return_code))
    if failures:
        raise RuntimeError(f"{stage}: runner failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for valid single-proxy evidence, then run gated ensemble stages.")
    parser.add_argument("--spec", default="configs/surrogate_experiments.yaml")
    parser.add_argument("--root", default="outputs/surrogate_experiments")
    parser.add_argument("--analysis-output", default="outputs/surrogate_analysis")
    parser.add_argument("--theory-metrics", default="outputs/surrogate_theory/stage0a_caltech101_balanced_unique_200/ensemble_theory_metrics.csv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="models/open_clip")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    base = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    expected = {stage: len(base["stages"][stage]["datasets"]) * len(base["stages"][stage]["sets"]) * len(base["stages"][stage]["seeds"])
                for stage in ("stage1_single", "stage2_equal_steps", "stage2_equal_forwards")}
    if args.plan:
        print(json.dumps({"expected": expected, "parallel_sets": {
            stage: base["stages"][stage]["sets"] for stage in ("stage2_equal_steps", "stage2_equal_forwards")}}, indent=2))
        return

    state = root / "ensemble_pipeline_state.json"
    while len(completed_trials(root, "stage1_single")) < expected["stage1_single"]:
        write_state(state, status="waiting_for_stage1", completed=len(completed_trials(root, "stage1_single")), expected=expected["stage1_single"])
        time.sleep(args.poll_seconds)
    validate_stage(root, "stage1_single", expected["stage1_single"])
    write_state(state, status="stage1_validated")
    subprocess.run([args.python, "scripts/analyze_surrogate_experiments.py", "--root", str(root),
                    "--output", args.analysis_output, "--theory-metrics", args.theory_metrics], check=True)

    spec_dir = root / "generated_stage_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    for stage in ("stage2_equal_steps", "stage2_equal_forwards"):
        existing = len(completed_trials(root, stage))
        if existing < expected[stage]:
            write_state(state, status=f"running_{stage}", completed=existing, expected=expected[stage])
            run_parallel(stage_specs(base, stage, spec_dir), stage, root, args.python, args.device, args.cache_dir)
        validate_stage(root, stage, expected[stage])
        subprocess.run([args.python, "scripts/analyze_surrogate_experiments.py", "--root", str(root),
                        "--output", args.analysis_output, "--theory-metrics", args.theory_metrics], check=True)
    write_state(state, status="complete", expected=expected)


if __name__ == "__main__":
    main()

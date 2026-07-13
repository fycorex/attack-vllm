#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

try:
    from audit_attack_manifest import audit_manifest
    from run_noise_search import matrix
except ModuleNotFoundError:
    from scripts.audit_attack_manifest import audit_manifest
    from scripts.run_noise_search import matrix


SMOKE_STAGE = "stage0_geometric"
VALIDATION_STAGES = ("equal_steps_geometric", "equal_forward_geometric")


def expected_trials(spec: dict, stage: str) -> int:
    return len(matrix(spec, stage))


def require_valid_smoke(path: Path, expected: int) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing geometric smoke validation: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("all_valid") or int(value.get("trials", -1)) != expected or int(value.get("valid_trials", -1)) != expected:
        raise RuntimeError(f"geometric smoke gate failed: {value}")
    return value


def audit_stage_manifests(spec: dict, stage: str) -> dict[str, dict]:
    reports = {}
    for dataset in spec["stages"][stage]["datasets"]:
        manifest = Path(spec["datasets"][dataset]["manifest"])
        report = audit_manifest(manifest, require_unique_sources=True)
        if not report["valid"]:
            raise RuntimeError(f"manifest audit failed for {dataset}: {report['errors']}")
        if int(report["items"]) < int(spec["stages"][stage]["items"]):
            raise RuntimeError(f"manifest {dataset} has fewer items than the stage requests")
        reports[dataset] = report
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or launch gated geometric transfer experiments.")
    parser.add_argument("--spec", default="configs/geometric_pilot.yaml")
    parser.add_argument("--stage", choices=(SMOKE_STAGE, *VALIDATION_STAGES), required=True)
    parser.add_argument("--root", default="outputs/geometric_pilot")
    parser.add_argument("--validation-output", default="outputs/geometric_analysis")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--launch", action="store_true", help="Explicitly launch attacks; default is a read-only plan.")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    count = expected_trials(spec, args.stage)
    manifests = audit_stage_manifests(spec, args.stage)
    smoke_report_path = Path(args.validation_output) / SMOKE_STAGE / "stage_validation.json"
    if args.stage in VALIDATION_STAGES:
        require_valid_smoke(smoke_report_path, expected_trials(spec, SMOKE_STAGE))
    plan = {
        "stage": args.stage,
        "trials": count,
        "workers": args.workers,
        "manifests": manifests,
        "smoke_gate_required": args.stage in VALIDATION_STAGES,
        "launch": args.launch,
        "api_access": False,
    }
    print(json.dumps(plan, indent=2))
    if not args.launch:
        return

    subprocess.run([
        args.python, "scripts/run_noise_search.py",
        "--spec", str(spec_path), "--stage", args.stage,
        "--root", args.root, "--workers", str(args.workers),
        "--python", args.python,
    ], check=True)
    validation_dir = Path(args.validation_output) / args.stage
    subprocess.run([
        args.python, "scripts/validate_geometric_stage.py",
        "--root", str(Path(args.root) / args.stage),
        "--expected-trials", str(count),
        "--output", str(validation_dir),
    ], check=True)


if __name__ == "__main__":
    main()

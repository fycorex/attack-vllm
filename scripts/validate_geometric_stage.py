#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import yaml


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def image_linf(clean: Path, adversarial: Path) -> float:
    first = np.asarray(Image.open(clean).convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(Image.open(adversarial).convert("RGB"), dtype=np.float32) / 255.0
    if first.shape != second.shape:
        raise ValueError(f"image shape mismatch: {first.shape} != {second.shape}")
    return float(np.abs(first - second).max())


def validate_trial(result_path: Path) -> tuple[dict[str, Any], list[str]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    directory = result_path.parent
    errors = []
    config_path = directory / "effective_attack.yaml"
    if not config_path.is_file():
        return result, ["missing effective_attack.yaml"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    epsilon = float(config["attack"]["epsilon"])
    for name, value in (config.get("evaluation") or {}).items():
        if isinstance(value, dict) and value.get("enabled"):
            errors.append(f"evaluation enabled during attack: {name}")
    metrics_paths = sorted((directory / "attack").glob("item_*/metrics.json"))
    if len(metrics_paths) != int(result["items"]):
        errors.append(f"expected {result['items']} metrics, found {len(metrics_paths)}")
    max_linf = 0.0
    forwards = set()
    for metrics_path in metrics_paths:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not finite_tree(metrics):
            errors.append(f"non-finite metrics: {metrics_path.parent.name}")
        clean, adversarial = metrics_path.with_name("clean.png"), metrics_path.with_name("adversarial.png")
        if not clean.is_file() or not adversarial.is_file():
            errors.append(f"missing image pair: {metrics_path.parent.name}")
            continue
        max_linf = max(max_linf, image_linf(clean, adversarial))
        forwards.add(int((metrics.get("noise_method") or {}).get("total_surrogate_forwards", -1)))
    tolerance = 1.0 / 255.0 + 1e-7
    if max_linf > epsilon + tolerance:
        errors.append(f"L-infinity {max_linf:.8f} exceeds epsilon {epsilon:.8f}")
    expected_forwards = int(result.get("expected_surrogate_forwards_per_item", -1))
    if forwards != {expected_forwards}:
        errors.append(f"actual forwards {sorted(forwards)} != expected {expected_forwards}")
    if not (directory / "heldout_summary.json").is_file():
        errors.append("missing heldout_summary.json")
    summary = {"trial_id": result.get("trial_id"), "geometry_mode": result.get("geometry_mode", "none"),
        "noise_mode": result.get("mode"), "items": len(metrics_paths), "epsilon": epsilon,
        "max_saved_linf": max_linf, "expected_surrogate_forwards_per_item": expected_forwards,
        "actual_surrogate_forwards_per_item": next(iter(forwards)) if len(forwards) == 1 else None,
        "heldout_macro_asr": result.get("heldout_macro_asr"),
        "heldout_macro_margin_gain": result.get("heldout_macro_margin_gain"), "valid": not errors,
        "errors": json.dumps(errors)}
    return summary, errors


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows); temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate geometric attack artifacts and exact forward accounting.")
    parser.add_argument("--root", required=True, help="Stage directory containing trial subdirectories")
    parser.add_argument("--expected-trials", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    paths = sorted(root.glob("*/trial_result.json"))
    rows, failures = [], []
    ids = []
    for path in paths:
        row, errors = validate_trial(path); rows.append(row); ids.append(row["trial_id"])
        if errors: failures.append({"trial_id": row["trial_id"], "errors": errors})
    if len(ids) != len(set(ids)):
        failures.append({"trial_id": None, "errors": ["duplicate trial IDs"]})
    if args.expected_trials is not None and len(rows) != args.expected_trials:
        failures.append({"trial_id": None, "errors": [f"expected {args.expected_trials} trials, found {len(rows)}"]})
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "stage_validation.csv", rows)
    report = {"trials": len(rows), "valid_trials": sum(bool(row["valid"]) for row in rows),
              "failures": failures, "all_valid": not failures}
    (output / "stage_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

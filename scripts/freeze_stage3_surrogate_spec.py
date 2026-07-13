#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_manifest(path: Path, items: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("items") or []
    if len(rows) < items:
        raise ValueError(f"{path} has {len(rows)} items, fewer than requested Stage 3 count {items}")
    selected = rows[:items]
    images = [str(row.get("image_path") or "") for row in selected]
    if any(not image for image in images) or len(set(images)) != items:
        raise ValueError(f"{path} does not provide {items} unique source image paths in manifest order")
    missing = [image for image in images if not Path(image).is_file()]
    if missing:
        raise ValueError(f"{path} references missing source images: {missing[:3]}")
    return {"path": str(path.resolve()), "sha256": sha256(path), "available_items": len(rows),
            "selected_items": items, "unique_selected_source_images": len(set(images))}


def select_methods(rows: list[dict[str, str]], *, stage: str, budget_mode: str,
                   dataset: str, baseline: str, top_k: int, minimum_seeds: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("stage") != stage or row.get("budget_mode") != budget_mode or row.get("dataset") != dataset:
            continue
        if int(float(row.get("missing_items") or 0)) != 0:
            continue
        if int(float(row.get("valid_items") or 0)) != int(float(row.get("items") or -1)):
            continue
        grouped[str(row["surrogate_set"])].append(row)
    summaries = []
    for method, values in grouped.items():
        seed_targets: dict[int, set[str]] = defaultdict(set)
        for row in values:
            seed_targets[int(float(row["seed"]))].add(str(row["target"]))
        seeds = sorted(seed_targets)
        target_sets = list(seed_targets.values())
        if len(seeds) < minimum_seeds or not target_sets or any(targets != target_sets[0] for targets in target_sets):
            continue
        summaries.append({"surrogate_set": method, "heldout_macro_asr": statistics.mean(float(row["asr"]) for row in values),
                          "heldout_macro_margin_gain": statistics.mean(float(row["mean_margin_gain"]) for row in values),
                          "seeds": seeds, "targets": sorted(target_sets[0]),
                          "trial_directories": sorted({str(Path(row["directory"]).resolve()) for row in values})})
    by_name = {row["surrogate_set"]: row for row in summaries}
    if baseline not in by_name:
        raise ValueError(f"Complete baseline {baseline!r} was not found")
    alternatives = sorted((row for row in summaries if row["surrogate_set"] != baseline),
                          key=lambda row: (row["heldout_macro_asr"], row["heldout_macro_margin_gain"]), reverse=True)
    return [by_name[baseline], *alternatives[:top_k]]


def freeze(base_spec: Path, analysis_csv: Path, output: Path, *, stage: str = "stage2_equal_forwards",
           budget_mode: str = "equal_forwards", selection_dataset: str = "caption_caltech",
           baseline: str = "single_reference", top_k: int = 2, minimum_seeds: int = 3,
           stage3_items: int | None = None, dataset_manifests: dict[str, Path] | None = None) -> dict[str, Any]:
    with analysis_csv.open(newline="", encoding="utf-8") as handle:
        selected = select_methods(list(csv.DictReader(handle)), stage=stage, budget_mode=budget_mode,
                                  dataset=selection_dataset, baseline=baseline, top_k=top_k,
                                  minimum_seeds=minimum_seeds)
    spec = yaml.safe_load(base_spec.read_text(encoding="utf-8"))
    if "stage3_cross_dataset" not in spec.get("stages", {}):
        raise ValueError("Base spec has no stage3_cross_dataset")
    if stage3_items is not None:
        spec["stages"]["stage3_cross_dataset"]["items"] = stage3_items
    effective_items = int(spec["stages"]["stage3_cross_dataset"]["items"])
    manifest_audits = {}
    for dataset, path in (dataset_manifests or {}).items():
        if dataset not in spec.get("datasets", {}):
            raise ValueError(f"Unknown dataset manifest override: {dataset}")
        audit = audit_manifest(path, effective_items)
        spec["datasets"][dataset]["manifest"] = audit["path"]
        manifest_audits[dataset] = audit
    for dataset in spec["stages"]["stage3_cross_dataset"]["datasets"]:
        path = Path(spec["datasets"][dataset]["manifest"])
        if dataset not in manifest_audits:
            manifest_audits[dataset] = audit_manifest(path, effective_items)
    spec["stages"]["stage3_cross_dataset"]["sets"] = [row["surrogate_set"] for row in selected]
    evidence = {"created_at": datetime.now(timezone.utc).isoformat(),
                "selection_source": "heldout_open_source_only", "api_results_used_for_selection": False,
                "analysis_csv": str(analysis_csv.resolve()), "analysis_csv_sha256": sha256(analysis_csv),
                "source_spec": str(base_spec.resolve()), "source_spec_sha256": sha256(base_spec),
                "stage": stage, "budget_mode": budget_mode, "selection_dataset": selection_dataset,
                "minimum_seeds": minimum_seeds, "baseline": baseline, "top_k_nonbaseline": top_k,
                "stage3_items": effective_items, "dataset_manifests": manifest_audits, "selected": selected}
    canonical = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    evidence["freeze_hash"] = hashlib.sha256(canonical).hexdigest()
    spec["frozen_stage3_selection"] = evidence
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        yaml.safe_dump(spec, handle, sort_keys=False); temporary = Path(handle.name)
    temporary.replace(output)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Stage 3 methods using complete held-out equal-forward evidence only.")
    parser.add_argument("--base-spec", default="configs/surrogate_experiments.yaml")
    parser.add_argument("--analysis-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", default="stage2_equal_forwards")
    parser.add_argument("--budget-mode", default="equal_forwards")
    parser.add_argument("--selection-dataset", default="caption_caltech")
    parser.add_argument("--baseline", default="single_reference")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--stage3-items", type=int)
    parser.add_argument("--dataset-manifest", action="append", default=[], metavar="NAME=PATH",
                        help="Override and audit a Stage 3 dataset manifest; may be repeated.")
    args = parser.parse_args()
    overrides = {}
    for value in args.dataset_manifest:
        if "=" not in value:
            parser.error("--dataset-manifest must use NAME=PATH")
        name, path = value.split("=", 1); overrides[name] = Path(path)
    result = freeze(Path(args.base_spec), Path(args.analysis_csv), Path(args.output), stage=args.stage,
                    budget_mode=args.budget_mode, selection_dataset=args.selection_dataset,
                    baseline=args.baseline, top_k=args.top_k, minimum_seeds=args.minimum_seeds,
                    stage3_items=args.stage3_items, dataset_manifests=overrides)
    print(json.dumps({"output": args.output, "selected": [row["surrogate_set"] for row in result["selected"]],
                      "freeze_hash": result["freeze_hash"]}, indent=2))


if __name__ == "__main__":
    main()

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("asr", "mean_margin_gain", "seed", "items", "valid_items", "missing_items"):
            if row.get(key) not in {None, ""}:
                row[key] = float(row[key])
    return rows


def select_candidates(rows: list[dict[str, Any]], *, stage: str, budget_mode: str,
                      top_k: int, minimum_seeds: int, minimum_targets: int = 1,
                      datasets: set[str] | None = None, baseline: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("stage") != stage or row.get("budget_mode") != budget_mode:
            continue
        if datasets is not None and str(row.get("dataset")) not in datasets:
            continue
        if int(row.get("missing_items", 0)) or int(row.get("valid_items", 0)) != int(row.get("items", -1)):
            continue
        grouped[(str(row["dataset"]), str(row["surrogate_set"]))].append(row)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (dataset, surrogate_set), values in grouped.items():
        seed_targets: dict[int, set[str]] = defaultdict(set)
        row_keys = set()
        duplicate = False
        for row in values:
            key = (int(row["seed"]), str(row["target"]))
            if key in row_keys:
                duplicate = True
            row_keys.add(key); seed_targets[key[0]].add(key[1])
        seeds = sorted(seed_targets); target_sets = list(seed_targets.values())
        if duplicate or len(seeds) < minimum_seeds or not target_sets or len(target_sets[0]) < minimum_targets or any(targets != target_sets[0] for targets in target_sets):
            continue
        by_dataset[dataset].append({"dataset": dataset, "surrogate_set": surrogate_set,
            "heldout_macro_asr": statistics.mean(float(row["asr"]) for row in values),
            "heldout_macro_margin_gain": statistics.mean(float(row["mean_margin_gain"]) for row in values),
            "seeds": seeds, "targets": sorted({str(row["target"]) for row in values}),
            "attack_output_directories": sorted({str(Path(row["directory"]).resolve() / "attack") for row in values})})
    selected = []
    for dataset, values in sorted(by_dataset.items()):
        values.sort(key=lambda row: (row["heldout_macro_asr"], row["heldout_macro_margin_gain"]), reverse=True)
        if baseline is None:
            chosen = values[:top_k]
        else:
            baseline_rows = [row for row in values if row["surrogate_set"] == baseline]
            if not baseline_rows:
                continue
            alternatives = [row for row in values if row["surrogate_set"] != baseline]
            chosen = [baseline_rows[0], *alternatives[:top_k]]
        for rank, value in enumerate(chosen, 1):
            selected.append({"rank_within_dataset": rank, "selection_role":
                             "baseline" if baseline is not None and value["surrogate_set"] == baseline else "candidate",
                             **value})
    return selected


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2); temporary = Path(handle.name)
    temporary.replace(path)


def freeze_manifest(source: Path, output: Path, *, stage: str, budget_mode: str,
                    top_k: int = 1, minimum_seeds: int = 3, minimum_targets: int = 1,
                    datasets: set[str] | None = None, baseline: str | None = None) -> dict[str, Any]:
    selected = select_candidates(
        read_rows(source), stage=stage, budget_mode=budget_mode, top_k=top_k,
        minimum_seeds=minimum_seeds, minimum_targets=minimum_targets,
        datasets=datasets, baseline=baseline,
    )
    if not selected:
        raise ValueError("No complete held-out candidates satisfy the freeze criteria")
    selected_datasets = {row["dataset"] for row in selected}
    if datasets is not None and selected_datasets != datasets:
        missing = sorted(datasets - selected_datasets)
        raise ValueError(f"No complete held-out candidates for requested datasets: {missing}")
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_source": "heldout_open_source_only", "api_results_used_for_selection": False,
        "analysis_csv": str(source.resolve()), "analysis_csv_sha256": sha256(source),
        "stage": stage, "budget_mode": budget_mode, "top_k_per_dataset": top_k,
        "minimum_seeds": minimum_seeds, "datasets": sorted(selected_datasets),
        "baseline": baseline, "candidates": selected, "minimum_targets": minimum_targets}
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["freeze_hash"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze API candidates selected only by held-out open-source results.")
    parser.add_argument("--analysis-csv", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--budget-mode", required=True)
    parser.add_argument("--top-k", type=int, default=2,
                        help="Candidates per dataset; when --baseline is set, this excludes the baseline.")
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--minimum-targets", type=int, default=1)
    parser.add_argument("--dataset", action="append", default=[],
                        help="Freeze only this dataset; repeat for multiple datasets. Prefer one task per manifest.")
    parser.add_argument("--baseline", help="Always include this complete baseline before ranked candidates.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.analysis_csv)
    try:
        manifest = freeze_manifest(
            source, Path(args.output), stage=args.stage, budget_mode=args.budget_mode,
            top_k=args.top_k, minimum_seeds=args.minimum_seeds,
            minimum_targets=args.minimum_targets,
            datasets=set(args.dataset) if args.dataset else None, baseline=args.baseline,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"candidates": len(manifest["candidates"]), "freeze_hash": manifest["freeze_hash"], "output": args.output}, indent=2))


if __name__ == "__main__":
    main()

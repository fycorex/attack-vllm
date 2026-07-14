#!/usr/bin/env python3
"""GPU-efficient held-out replay for a completed attack stage.

Each target model is loaded once and evaluates batches drawn from every pending
trial.  This preserves the existing per-item metrics while avoiding repeated
model loads and short, CPU-bound per-trial GPU bursts.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import time

import torch

from config import SurrogateConfig
from data import load_manifest
from evaluate_surrogate_heldout import _batches, _score_prepared_batch, prefetched_batches, summarize_rows
from surrogate_composition import load_composition_spec
from surrogates import create_surrogate, unload_surrogate
from transfer_eval import atomic_write_json


def load_pending(stage_root: Path) -> list[dict]:
    pending = []
    for path in sorted(stage_root.glob("*/trial_pending.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["directory"] = path.parent
        pending.append(value)
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all pending trial pairs grouped by held-out model.")
    parser.add_argument("--composition-config", required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="models/open_clip")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--profile-output", help="Write CPU decode and GPU-batch timing for a real replay run.")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    spec = load_composition_spec(args.composition_config)
    pending = load_pending(Path(args.stage_root))
    if not pending:
        print(json.dumps({"pending_trials": 0, "completed_trials": 0}))
        return
    device = args.device if torch.cuda.is_available() else "cpu"
    rows_by_trial: dict[Path, list[dict]] = defaultdict(list)
    timing_rows = []

    records_by_shape: dict[tuple[int, int], list[tuple[dict, object, Path]]] = defaultdict(list)
    for trial in pending:
        manifest = load_manifest(trial["manifest"])
        items = manifest.items[:int(trial["items"])]
        attack_output = Path(trial["attack_output"])
        for item in items:
            item_dir = attack_output / item.item_id
            if not (item_dir / "adversarial.png").is_file():
                for model_id in spec.heldout_models:
                    rows_by_trial[trial["directory"]].append({
                        "item_id": item.item_id, "model": model_id, "missing": True, "proxy_success": False,
                    })
                continue
            records_by_shape[(len(item.positive_image_paths), len(item.negative_image_paths))].append((trial, item, item_dir))

    for model_id in spec.heldout_models:
        metadata = spec.models[model_id]
        wrapper = create_surrogate(SurrogateConfig(**metadata.victim_config()), device, cache_dir=args.cache_dir)
        try:
            for records in records_by_shape.values():
                batches = ([ (item, item_dir) for _, item, item_dir in batch ]
                           for batch in _batches(records, args.batch_size))
                for batch, tensors, cpu_seconds in prefetched_batches(
                    batches, metadata.input_size, args.prefetch_batches, args.loader_workers,
                ):
                    encoded, gpu_wall_seconds = _score_prepared_batch(wrapper, batch, tensors, device, top_k=10)
                    timing_rows.append({"model": model_id, "items": len(batch), "cpu_prepare_seconds": cpu_seconds,
                                        "gpu_batch_wall_seconds": gpu_wall_seconds})
                    for (trial, _, _), row in zip(batch, encoded):
                        rows_by_trial[trial["directory"]].append({"model": model_id, **row})
        finally:
            unload_surrogate(wrapper)

    completed = 0
    for trial in pending:
        directory = trial["directory"]
        summary = summarize_rows(rows_by_trial[directory], spec.heldout_models)
        atomic_write_json(Path(trial["heldout_output"]), summary)
        result = {**trial["trial"], "status": "complete", "heldout_macro_asr": summary["heldout_macro_asr"],
                  "heldout_macro_margin_gain": summary["heldout_macro_margin_gain"],
                  "attack_validation": trial["attack_validation"], "attack_command": trial["attack_command"],
                  "heldout_command": ["scripts/evaluate_heldout_stage.py", "--batch-size", str(args.batch_size)]}
        atomic_write_json(directory / "trial_result.json", result)
        (directory / "trial_pending.json").unlink()
        completed += 1
    print(json.dumps({"pending_trials": len(pending), "completed_trials": completed,
                      "models_loaded_once_each": len(spec.heldout_models), "batch_size": args.batch_size}, indent=2))
    if args.profile_output:
        Path(args.profile_output).write_text(json.dumps({"batches": timing_rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Profile real replay preprocessing and GPU work without changing experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import SurrogateConfig
from data import load_manifest
from evaluate_surrogate_heldout import _prepare_batch, _score_prepared_batch
from surrogate_composition import load_composition_spec
from surrogates import create_surrogate, unload_surrogate


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real clean/adv/reference PNG replay batches.")
    parser.add_argument("--composition-config", default="configs/research_cycle_cross_family.yaml")
    parser.add_argument("--model", default="eva02_l14_heldout")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--attack-output", required=True)
    parser.add_argument("--batch-sizes", default="4,8,12,16")
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="models/open_clip")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = load_composition_spec(args.composition_config)
    metadata = spec.models[args.model]
    manifest = load_manifest(args.manifest)
    source = [(item, Path(args.attack_output) / item.item_id) for item in manifest.items
              if (Path(args.attack_output) / item.item_id / "adversarial.png").is_file()]
    if not source:
        raise RuntimeError("No complete real attack pairs found")
    wrapper = create_surrogate(SurrogateConfig(**metadata.victim_config()), "cuda", cache_dir=args.cache_dir)
    results = []
    try:
        for batch_size in [int(value) for value in args.batch_sizes.split(",") if value.strip()]:
            batch = [source[index % len(source)] for index in range(batch_size)]
            try:
                torch.cuda.reset_peak_memory_stats()
                prepared, tensors, cpu_seconds = _prepare_batch(batch, metadata.input_size, args.loader_workers)
                rows, gpu_wall_seconds = _score_prepared_batch(wrapper, prepared, tensors, "cuda", top_k=10)
                torch.cuda.synchronize()
                results.append({"status": "ok", "batch_size": batch_size, "valid_rows": len(rows),
                                "cpu_prepare_seconds": cpu_seconds, "gpu_batch_wall_seconds": gpu_wall_seconds,
                                "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
                                "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3)})
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                results.append({"status": "oom", "batch_size": batch_size, "error": str(exc).split("\n")[0]})
    finally:
        unload_surrogate(wrapper)
    print(json.dumps({"model": args.model, "reference_counts": {"positive": len(source[0][0].positive_image_paths),
                                                                   "negative": len(source[0][0].negative_image_paths)},
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()

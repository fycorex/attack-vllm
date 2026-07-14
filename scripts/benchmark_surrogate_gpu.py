#!/usr/bin/env python3
"""Measure sustained forward/backward throughput for one real surrogate.

This is a tuning utility only: random image tensors are used, no attack output
or experiment result is created.  CUDA events report device time, while peak
allocated memory gives a safe batch-size boundary for batched attacks.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from config import SurrogateConfig
from surrogate_composition import load_composition_spec
from surrogates import create_surrogate, unload_surrogate


def run_case(wrapper, batch_size: int, size: int, iterations: int, warmup: int) -> dict:
    device = "cuda"
    def step() -> None:
        images = torch.rand(batch_size, 3, size, size, device=device, requires_grad=True)
        embedding = wrapper.encode_image(images)
        embedding.square().mean().backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    device_seconds = start.elapsed_time(end) / 1000.0
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "wall_seconds": wall_seconds,
        "cuda_seconds": device_seconds,
        "cuda_occupancy_proxy": device_seconds / wall_seconds if wall_seconds else 0.0,
        "images_per_wall_second": (batch_size * iterations) / wall_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024 ** 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a real surrogate under sustained image-gradient load.")
    parser.add_argument("--composition-config", default="configs/research_cycle_cross_family.yaml")
    parser.add_argument("--model", default="eva02_l14_heldout")
    parser.add_argument("--batch-sizes", default="16,32,48,64,80,96")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cache-dir", default="models/open_clip")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    spec = load_composition_spec(args.composition_config)
    metadata = spec.models[args.model]
    wrapper = create_surrogate(SurrogateConfig(**metadata.victim_config()), "cuda", cache_dir=args.cache_dir)
    results = []
    try:
        for batch_size in [int(value) for value in args.batch_sizes.split(",") if value.strip()]:
            try:
                results.append({"status": "ok", **run_case(wrapper, batch_size, metadata.input_size, args.iterations, args.warmup)})
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                results.append({"status": "oom", "batch_size": batch_size, "error": str(exc).split("\n")[0]})
    finally:
        unload_surrogate(wrapper)
    print(json.dumps({"model": args.model, "input_size": metadata.input_size, "results": results}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time

import torch

from config import SurrogateConfig
from data import load_image_tensor, load_manifest
from eval import evaluate_proxy
from surrogate_composition import load_composition_spec
from surrogates import create_surrogate, unload_surrogate
from theory_metrics import normalized_kernel_alignment
from transfer_eval import atomic_write_json


def _prototype(examples: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(examples.mean(0, keepdim=True), dim=-1)


def _batches(values: list, batch_size: int):
    for offset in range(0, len(values), batch_size):
        yield values[offset:offset + batch_size]


def _prepare_batch(batch: list, size: int, loader_workers: int = 1) -> tuple[list, tuple[torch.Tensor, ...], float]:
    """Decode/resize a real image batch on CPU, optionally in parallel."""
    started = time.perf_counter()
    requests: list[tuple[str, int, int, Path]] = []
    for item_index, (item, item_dir) in enumerate(batch):
        requests.extend([
            ("clean", item_index, 0, item_dir / "clean.png"),
            ("adversarial", item_index, 0, item_dir / "adversarial.png"),
        ])
        requests.extend(("positive", item_index, index, path)
                        for index, path in enumerate(item.positive_image_paths))
        requests.extend(("negative", item_index, index, path)
                        for index, path in enumerate(item.negative_image_paths))

    def load(request):
        return request[:3], load_image_tensor(request[3], size)

    if loader_workers > 1:
        with ThreadPoolExecutor(max_workers=loader_workers) as pool:
            loaded = pool.map(load, requests)
            tensors = _assemble_preallocated_batch(loaded, batch)
    else:
        tensors = _assemble_preallocated_batch(map(load, requests), batch)
    return batch, tensors, time.perf_counter() - started


def _assemble_preallocated_batch(loaded, batch: list) -> tuple[torch.Tensor, ...]:
    """Write decoded images directly into final pinned tensors.

    Avoid retaining both a Python list of every decoded reference image and a
    second stacked copy.  Large held-out batches may contain thousands of
    positive/negative references, so the old double-buffered representation
    could exhaust host memory before the GPU batch started.
    """
    positive_count = len(batch[0][0].positive_image_paths)
    negative_count = len(batch[0][0].negative_image_paths)
    outputs = None
    for (kind, item_index, reference_index), tensor in loaded:
        if outputs is None:
            options = {"dtype": tensor.dtype, "pin_memory": torch.cuda.is_available()}
            image_shape = tuple(tensor.shape)
            outputs = {
                "clean": torch.empty((len(batch), *image_shape), **options),
                "adversarial": torch.empty((len(batch), *image_shape), **options),
                "positive": torch.empty((len(batch), positive_count, *image_shape), **options),
                "negative": torch.empty((len(batch), negative_count, *image_shape), **options),
            }
        if kind in {"clean", "adversarial"}:
            outputs[kind][item_index].copy_(tensor)
        else:
            outputs[kind][item_index, reference_index].copy_(tensor)
    if outputs is None:
        raise ValueError("Cannot prepare an empty held-out batch")
    return outputs["clean"], outputs["adversarial"], outputs["positive"], outputs["negative"]


def _score_prepared_batch(
    wrapper, batch: list, tensors: tuple[torch.Tensor, ...], device: str, top_k: int,
    reference_chunk_size: int = 512,
) -> tuple[list[dict], float]:
    """Run four real image groups on GPU after CPU-side prefetch has completed."""
    clean, adversarial, positives, negatives = (tensor.to(device, non_blocking=True) for tensor in tensors)
    positive_counts = positives.shape[1]
    negative_counts = negatives.shape[1]
    started = time.perf_counter()
    with torch.inference_mode():
        clean_embeddings = wrapper.encode_image(clean)
        adversarial_embeddings = wrapper.encode_image(adversarial)
        positive_embeddings = _encode_in_chunks(
            wrapper, positives.flatten(0, 1), reference_chunk_size,
        ).reshape(len(batch), positive_counts, -1)
        negative_embeddings = _encode_in_chunks(
            wrapper, negatives.flatten(0, 1), reference_chunk_size,
        ).reshape(len(batch), negative_counts, -1)

    rows = []
    for index, (item, _) in enumerate(batch):
        positive = positive_embeddings[index]
        negative = negative_embeddings[index]
        clean_embedding = clean_embeddings[index:index + 1]
        adversarial_embedding = adversarial_embeddings[index:index + 1]
        prototype = _prototype(positive)
        result = evaluate_proxy(
            clean_embedding, adversarial_embedding, positive, negative,
            top_k=min(top_k, len(item.positive_image_paths)), success_margin_threshold=0.0,
        )
        clean_distance = torch.linalg.vector_norm(clean_embedding - prototype)
        adversarial_distance = torch.linalg.vector_norm(adversarial_embedding - prototype)
        rows.append({
            "item_id": item.item_id,
            "missing": False,
            "clean_prototype_distance": float(clean_distance),
            "adversarial_prototype_distance": float(adversarial_distance),
            "prototype_distance_change": float(adversarial_distance - clean_distance),
            **result,
        })
    return rows, time.perf_counter() - started


def _encode_in_chunks(wrapper, images: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Bound activation memory while preserving one logical replay batch.

    Encoder families have very different activation footprints (notably
    DINOv2 at 518 px).  Retry only the current inference chunk at half size on
    CUDA OOM instead of failing the complete multi-model replay.
    """
    if chunk_size < 1:
        raise ValueError("reference chunk size must be positive")
    outputs = []
    offset = 0
    active_chunk_size = min(chunk_size, len(images))
    while offset < len(images):
        current = min(active_chunk_size, len(images) - offset)
        try:
            outputs.append(wrapper.encode_image(images[offset:offset + current]))
        except torch.cuda.OutOfMemoryError:
            if current == 1:
                raise
            active_chunk_size = max(1, current // 2)
            torch.cuda.empty_cache()
            continue
        offset += current
    return torch.cat(outputs)


def _encode_batch(wrapper, batch: list, size: int, device: str, top_k: int) -> list[dict]:
    """Compatibility wrapper for callers that do not use stage-level prefetch."""
    prepared_batch, tensors, _ = _prepare_batch(batch, size)
    rows, _ = _score_prepared_batch(wrapper, prepared_batch, tensors, device, top_k)
    return rows


def prefetched_batches(batches, size: int, prefetch_batches: int, loader_workers: int):
    """Overlap real PNG decode/resize of future batches with current GPU work."""
    iterator = iter(batches)
    queue = deque()
    with ThreadPoolExecutor(max_workers=max(1, prefetch_batches)) as pool:
        for _ in range(max(1, prefetch_batches)):
            try:
                queue.append(pool.submit(_prepare_batch, next(iterator), size, loader_workers))
            except StopIteration:
                break
        while queue:
            future = queue.popleft()
            yield future.result()
            try:
                queue.append(pool.submit(_prepare_batch, next(iterator), size, loader_workers))
            except StopIteration:
                pass


def summarize_rows(rows: list[dict], model_ids: list[str]) -> dict:
    per_model = {}
    for model_id in model_ids:
        model_rows = [row for row in rows if row["model"] == model_id]
        attempted = len(model_rows)
        valid = [row for row in model_rows if not row["missing"]]
        per_model[model_id] = {
            "attempted_items": attempted,
            "valid_items": len(valid),
            "missing_items": attempted - len(valid),
            "asr": sum(bool(row["proxy_success"]) for row in model_rows) / max(1, attempted),
            "mean_margin_gain": sum(float(row.get("margin_gain", 0.0)) for row in model_rows) / max(1, attempted),
            "mean_prototype_distance_change": sum(float(row.get("prototype_distance_change", 0.0)) for row in model_rows) / max(1, attempted),
        }
    return {
        "attempted_item_model_pairs": len(rows),
        "valid_item_model_pairs": sum(not row["missing"] for row in rows),
        "heldout_models": per_model,
        "heldout_macro_asr": sum(value["asr"] for value in per_model.values()) / max(1, len(per_model)),
        "heldout_macro_margin_gain": sum(value["mean_margin_gain"] for value in per_model.values()) / max(1, len(per_model)),
        "items": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay existing pairs on disjoint held-out OpenCLIP models.")
    parser.add_argument("--composition-config", default="configs/surrogate_composition.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--attack-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="models/open_clip")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Items per held-out encoder forward; metrics remain per-item.")
    args = parser.parse_args()

    spec = load_composition_spec(args.composition_config)
    manifest = load_manifest(args.manifest)
    items = manifest.items[:args.limit] if args.limit else manifest.items
    attack_output = Path(args.attack_output)
    device = args.device if torch.cuda.is_available() else "cpu"
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    rows = []
    for model_id in spec.heldout_models:
        metadata = spec.models[model_id]
        wrapper = create_surrogate(SurrogateConfig(**metadata.victim_config()), device, cache_dir=args.cache_dir)
        try:
            pending_by_shape: dict[tuple[int, int], list[tuple]] = {}
            for item in items:
                item_dir = attack_output / item.item_id
                if not (item_dir / "adversarial.png").is_file():
                    rows.append({"item_id": item.item_id, "model": model_id, "missing": True, "proxy_success": False})
                    continue
                shape = (len(item.positive_image_paths), len(item.negative_image_paths))
                pending_by_shape.setdefault(shape, []).append((item, item_dir))
            for pending in pending_by_shape.values():
                for batch in _batches(pending, args.batch_size):
                    for row in _encode_batch(wrapper, batch, metadata.input_size, device, args.top_k):
                        rows.append({"model": model_id, **row})
        finally:
            unload_surrogate(wrapper)

    summary = summarize_rows(rows, spec.heldout_models)
    atomic_write_json(Path(args.output), summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()

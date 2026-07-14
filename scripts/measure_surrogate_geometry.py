#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import torch

from config import SurrogateConfig
from data import load_image_tensor, load_manifest
from surrogate_composition import load_composition_spec
from surrogates import create_surrogate, unload_surrogate
from theory_metrics import (
    centered_linear_cka,
    ensemble_theory_metrics,
    neighbor_margin,
    neighborhood_overlap,
    uncentered_kernel_alignment,
)
from transfer_eval import atomic_write_json, git_state, sha256_file


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _encode(model, paths: list[Path], size: int, device: str, batch_size: int) -> torch.Tensor:
    rows = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([load_image_tensor(path, size) for path in paths[start:start + batch_size]]).to(device)
        with torch.no_grad():
            rows.append(model.encode_image(batch).cpu())
    return torch.cat(rows)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "q05": float(torch.quantile(values.float(), .05)),
        "median": float(torch.quantile(values.float(), .5)),
    }


def subsample_indices(item_count: int, sizes: list[int], repeats: int, seed: int) -> list[tuple[int, int, list[int]]]:
    if repeats < 1:
        raise ValueError("subsample repeats must be positive")
    rng = random.Random(seed)
    rows = []
    for size in sorted(set(sizes)):
        if size < 3 or size > item_count:
            continue
        for repeat in range(repeats):
            rows.append((size, repeat, sorted(rng.sample(range(item_count), size))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure surrogate kernel geometry without running an attack or API.")
    parser.add_argument("--config", default="configs/surrogate_composition.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sets", nargs="+", default=["single_reference", "two_homogeneous", "four_lightweight_mixed"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--require-limit", action="store_true",
        help="Fail unless exactly --limit usable images remain after optional de-duplication.",
    )
    parser.add_argument(
        "--deduplicate-images", action="store_true",
        help="Select unique source-image SHA-256 values before applying the item limit.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", default="models/open_clip")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--subsample-sizes", nargs="*", type=int, default=[20, 50, 100])
    parser.add_argument("--subsample-repeats", type=int, default=100)
    parser.add_argument("--subsample-seed", type=int, default=20260714)
    args = parser.parse_args()

    config_path, manifest_path = Path(args.config), Path(args.manifest)
    spec = load_composition_spec(config_path)
    selected_sets = {name: spec.sets[name] for name in args.sets}
    proxy_ids = sorted({model_id for value in selected_sets.values() for model_id in value.models})
    required_ids = proxy_ids + list(spec.heldout_models)
    manifest = load_manifest(manifest_path)
    duplicate_items_skipped = 0
    source_hashes: list[str] = []
    items = []
    seen_hashes: set[str] = set()
    for item in manifest.items:
        image_hash = sha256_file(item.image_path)
        if args.deduplicate_images and image_hash in seen_hashes:
            duplicate_items_skipped += 1
            continue
        seen_hashes.add(image_hash)
        items.append(item)
        source_hashes.append(image_hash)
        if len(items) >= args.limit:
            break
    if args.require_limit and len(items) != args.limit:
        raise RuntimeError(
            f"Required {args.limit} usable images after de-duplication, found {len(items)}."
        )
    if len(items) < 3:
        raise RuntimeError("At least three items are required")
    device = args.device if torch.cuda.is_available() else "cpu"
    paths = [item.image_path for item in items]
    representations = {}
    model_rows = []

    for model_id in required_ids:
        metadata = spec.models[model_id]
        wrapper = create_surrogate(SurrogateConfig(**metadata.victim_config()), device, cache_dir=args.cache_dir)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            embeddings = _encode(wrapper, paths, metadata.input_size, device, args.batch_size)
            representations[model_id] = embeddings
            norms = torch.linalg.vector_norm(embeddings, dim=-1)
            model_rows.append({
                "model_id": model_id,
                "backend": metadata.backend,
                "openclip_model": metadata.model_name,
                "checkpoint": metadata.pretrained,
                "pretrained": metadata.pretrained,
                "input_size": metadata.input_size,
                "architecture_family": metadata.architecture_family,
                "objective_family": metadata.objective_family,
                "pretraining_family": metadata.pretraining_family,
                "representation_dimension": embeddings.shape[1],
                "parameter_count": sum(parameter.numel() for parameter in wrapper.model.parameters()),
                "embedding_norm_mean": float(norms.mean()),
                "embedding_norm_max_error": float((norms - 1).abs().max()),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            })
        finally:
            unload_surrogate(wrapper)

    pair_rows = []
    ks = [k for k in (1, 3, 5, 10) if k < len(items)]
    for proxy_id in proxy_ids:
        for target_id in spec.heldout_models:
            proxy, target = representations[proxy_id], representations[target_id]
            row = {
                "proxy": proxy_id,
                "target": target_id,
                "items": len(items),
                "centered_linear_cka": centered_linear_cka(proxy, target),
                "uncentered_kernel_alignment": uncentered_kernel_alignment(proxy, target),
            }
            for k in ks:
                row[f"neighborhood_overlap_at_{k}"] = neighborhood_overlap(proxy, target, k)
                if k < len(items) - 1:
                    for prefix, values in (("proxy", proxy), ("target", target)):
                        for name, result in _quantiles(neighbor_margin(values, k)).items():
                            row[f"{prefix}_neighbor_margin_at_{k}_{name}"] = result
            pair_rows.append(row)

    all_pair_rows = []
    for first_id, second_id in itertools.combinations(required_ids, 2):
        first, second = representations[first_id], representations[second_id]
        row = {
            "first_model": first_id,
            "second_model": second_id,
            "items": len(items),
            "centered_linear_cka": centered_linear_cka(first, second),
            "uncentered_kernel_alignment": uncentered_kernel_alignment(first, second),
        }
        for k in ks:
            row[f"neighborhood_overlap_at_{k}"] = neighborhood_overlap(first, second, k)
        all_pair_rows.append(row)

    subsample_rows = []
    draws = subsample_indices(len(items), args.subsample_sizes, args.subsample_repeats, args.subsample_seed)
    for first_id, second_id in itertools.combinations(required_ids, 2):
        first, second = representations[first_id], representations[second_id]
        for size, repeat, indices in draws:
            first_sample, second_sample = first[indices], second[indices]
            row = {
                "first_model": first_id,
                "second_model": second_id,
                "sample_size": size,
                "repeat": repeat,
                "centered_linear_cka": centered_linear_cka(first_sample, second_sample),
                "uncentered_kernel_alignment": uncentered_kernel_alignment(first_sample, second_sample),
            }
            for k in (1, 3, 5, 10):
                if k < size:
                    row[f"neighborhood_overlap_at_{k}"] = neighborhood_overlap(first_sample, second_sample, k)
            subsample_rows.append(row)

    ensemble_rows = []
    for set_name, surrogate_set in selected_sets.items():
        proxy_values = [representations[model_id] for model_id in surrogate_set.models]
        for target_id in spec.heldout_models:
            for centered in (True, False):
                ensemble_rows.append({
                    "surrogate_set": set_name,
                    "surrogate_ids": json.dumps(surrogate_set.models),
                    "target": target_id,
                    "alignment_definition": "centered_linear_cka" if centered else "uncentered_kernel_alignment",
                    "rationale": surrogate_set.rationale,
                    **ensemble_theory_metrics(proxy_values, representations[target_id], centered=centered, ridge=args.ridge),
                })

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": 1,
        "measurement_only": True,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "item_ids": [item.item_id for item in items],
        "source_image_sha256": source_hashes,
        "deduplicate_images": args.deduplicate_images,
        "duplicate_items_skipped": duplicate_items_skipped,
        "manifest_item_count": len(manifest.items),
        "surrogate_sets": list(selected_sets),
        "heldout_models": list(spec.heldout_models),
        "device": device,
        "ridge": args.ridge,
        "subsample_sizes": args.subsample_sizes,
        "subsample_repeats": args.subsample_repeats,
        "subsample_seed": args.subsample_seed,
        **git_state(Path.cwd()),
    }
    atomic_write_json(output / "run_manifest.json", run_manifest)
    atomic_write_json(output / "model_measurements.json", model_rows)
    _atomic_csv(output / "pairwise_alignment_metrics.csv", pair_rows)
    _atomic_csv(output / "all_model_pair_metrics.csv", all_pair_rows)
    _atomic_csv(output / "alignment_subsample_bootstrap.csv", subsample_rows)
    _atomic_csv(output / "ensemble_theory_metrics.csv", ensemble_rows)
    print(json.dumps({"models": len(model_rows), "proxy_target_pairs": len(pair_rows),
                      "all_model_pairs": len(all_pair_rows), "subsample_rows": len(subsample_rows),
                      "ensemble_rows": len(ensemble_rows),
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

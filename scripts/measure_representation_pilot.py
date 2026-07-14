#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch
import yaml

from config import SurrogateConfig
from data import load_image_tensor, load_manifest
from surrogates import create_surrogate, unload_surrogate
from representation_metrics import (
    centered_linear_cka,
    distance_kernel_audit,
    linear_gram,
    neighbor_margin,
    neighborhood_overlap,
    normalized_prototype,
    prototype_distance,
    uncentered_kernel_alignment,
)
from transfer_eval import atomic_write_json, git_state, sha256_file


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
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


def _model_id(spec: dict[str, Any]) -> str:
    return f"{spec['model_name']}:{spec['pretrained']}"


def _encode_paths(model, paths: list[Path], size: int, device: str, batch_size: int) -> torch.Tensor:
    rows = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([load_image_tensor(path, size) for path in paths[start:start + batch_size]]).to(device)
        with torch.no_grad():
            rows.append(model.encode_image(batch).cpu())
    return torch.cat(rows)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "minimum": float(values.min()),
        "q05": float(torch.quantile(values, 0.05)),
        "median": float(torch.quantile(values, 0.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure representation geometry without changing or rerunning an attack.")
    parser.add_argument("--spec", default="configs/noise_search.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--surrogate-group", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--attack-output", help="Optional existing directory containing clean/adversarial pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dir", default="models/open_clip")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    dataset = spec["datasets"][args.dataset]
    group = spec["surrogate_groups"][args.surrogate_group]
    manifest_path = Path(dataset["manifest"])
    manifest = load_manifest(manifest_path)
    items = manifest.items[:args.limit]
    if len(items) < 3:
        raise RuntimeError("Theory pilot requires at least three items")
    attack_ids = {_model_id(value) for value in group["attack"]}
    heldout_ids = {_model_id(value) for value in group["heldout"]}
    overlap = attack_ids.intersection(heldout_ids)
    if overlap:
        raise RuntimeError(f"Attack and held-out models overlap: {sorted(overlap)}")

    device = args.device if torch.cuda.is_available() else "cpu"
    all_specs = [("attack", value) for value in group["attack"]] + [("heldout", value) for value in group["heldout"]]
    representations: dict[str, torch.Tensor] = {}
    model_rows = []
    distance_rows = []
    source_paths = [item.image_path for item in items]
    attack_output = Path(args.attack_output) if args.attack_output else None

    for role, raw in all_specs:
        config = SurrogateConfig(**raw)
        model = create_surrogate(config, device, cache_dir=args.cache_dir)
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            embeddings = _encode_paths(model, source_paths, config.input_size, device, args.batch_size)
            representations[model.name] = embeddings
            norms = torch.linalg.vector_norm(embeddings, dim=-1)
            model_rows.append({
                "model": model.name,
                "role": role,
                "input_size": config.input_size,
                "patch_size": model.patch_size,
                "representation_dimension": embeddings.shape[1],
                "parameter_count": sum(parameter.numel() for parameter in model.model.parameters()),
                "embedding_norm_mean": float(norms.mean()),
                "embedding_norm_max_error": float((norms - 1).abs().max()),
                "kernel_frobenius_norm": float(torch.linalg.vector_norm(linear_gram(embeddings))),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            })
            if attack_output:
                for item in items:
                    item_dir = attack_output / item.item_id
                    if not (item_dir / "adversarial.png").is_file():
                        continue
                    images = _encode_paths(model, [item_dir / "clean.png", item_dir / "adversarial.png"], config.input_size, device, 2)
                    positives = _encode_paths(model, item.positive_image_paths, config.input_size, device, args.batch_size)
                    prototype = normalized_prototype(positives)
                    distances = prototype_distance(images, prototype)
                    distance_rows.append({
                        "item_id": item.item_id,
                        "model": model.name,
                        "role": role,
                        "clean_prototype_distance": float(distances[0]),
                        "adversarial_prototype_distance": float(distances[1]),
                        "distance_change": float(distances[1] - distances[0]),
                    })
        finally:
            unload_surrogate(model)

    pair_rows = []
    valid_ks = [k for k in (1, 5, 10) if k < len(items)]
    for proxy in sorted(attack_ids):
        for target in sorted(heldout_ids):
            proxy_reps, target_reps = representations[proxy], representations[target]
            row = {
                "proxy": proxy,
                "target": target,
                "items": len(items),
                "centered_linear_cka": centered_linear_cka(proxy_reps, target_reps),
                "uncentered_kernel_alignment": uncentered_kernel_alignment(proxy_reps, target_reps),
                **distance_kernel_audit(proxy_reps, target_reps),
            }
            for k in valid_ks:
                row[f"neighborhood_overlap_at_{k}"] = neighborhood_overlap(proxy_reps, target_reps, k)
                if k < len(items) - 1:
                    for prefix, reps in (("proxy", proxy_reps), ("target", target_reps)):
                        for name, value in _quantiles(neighbor_margin(reps, k)).items():
                            row[f"{prefix}_neighbor_margin_at_{k}_{name}"] = value
            pair_rows.append(row)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": 1,
        "measurement_only": True,
        "dataset": args.dataset,
        "surrogate_group": args.surrogate_group,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "item_ids": [item.item_id for item in items],
        "spec": str(spec_path.resolve()),
        "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "attack_output": str(attack_output.resolve()) if attack_output else None,
        "device": device,
        **git_state(Path.cwd()),
    }
    atomic_write_json(output / "run_manifest.json", run_manifest)
    atomic_write_json(output / "model_measurements.json", model_rows)
    atomic_write_json(output / "pairwise_alignment_metrics.json", pair_rows)
    _atomic_csv(output / "pairwise_alignment_metrics.csv", pair_rows)
    _atomic_csv(output / "prototype_distances.csv", distance_rows)
    print(json.dumps({"models": len(model_rows), "pairs": len(pair_rows), "prototype_rows": len(distance_rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

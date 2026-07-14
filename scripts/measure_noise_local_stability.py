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
import torch.nn.functional as F
import yaml

from augmentations import sample_noise
from config import SurrogateConfig
from data import load_image_tensor, load_manifest
from surrogates import create_surrogate, unload_surrogate
from representation_metrics import (
    centered_linear_cka,
    gradient_stability,
    neighborhood_overlap,
    normalized_prototype,
    pearson_correlation,
    prototype_distance,
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


def _model_id(raw: dict[str, Any]) -> str:
    return f"{raw['model_name']}:{raw['pretrained']}"


def _per_sample_loss(images: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    positive_logits = images @ positives.T
    negative_logits = images @ negatives.T
    log_probs = torch.log_softmax(torch.cat([positive_logits, negative_logits], dim=1) / temperature, dim=1)
    k = max(1, min(top_k, positives.shape[0]))
    positive = torch.topk(log_probs[:, :positives.shape[0]], k=k, dim=1).values.mean(dim=1)
    negative = log_probs[:, positives.shape[0]:].mean(dim=1)
    return -positive + negative


def _encode_examples(model, paths: list[Path], size: int, device: str) -> torch.Tensor:
    batch = torch.stack([load_image_tensor(path, size) for path in paths]).to(device)
    with torch.no_grad():
        return model.encode_image(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe local representation and gradient stability around existing adversarial images.")
    parser.add_argument("--spec", default="configs/noise_search.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--surrogate-group", required=True)
    parser.add_argument("--attack-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--probe-mode", choices=["gaussian_eot", "uniform_eot", "rademacher_eot"], default="gaussian_eot")
    parser.add_argument("--probe-sigma", type=float, default=2.0 / 255.0)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="models/open_clip")
    args = parser.parse_args()
    if args.probes < 3:
        raise ValueError("at least three probes are required")

    spec_path = Path(args.spec)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    dataset = spec["datasets"][args.dataset]
    group = spec["surrogate_groups"][args.surrogate_group]
    attack_ids = {_model_id(raw) for raw in group["attack"]}
    heldout_ids = {_model_id(raw) for raw in group["heldout"]}
    if attack_ids.intersection(heldout_ids):
        raise RuntimeError("attack and held-out models must be disjoint")
    manifest_path = Path(dataset["manifest"])
    manifest = load_manifest(manifest_path)
    attack_output = Path(args.attack_output)
    items = [item for item in manifest.items if (attack_output / item.item_id / "adversarial.png").is_file()][:args.limit]
    if not items:
        raise RuntimeError("no existing adversarial images matched the manifest")
    device = args.device if torch.cuda.is_available() else "cpu"

    common_probes: dict[str, torch.Tensor] = {}
    probe_stats: dict[str, dict[str, float]] = {}
    for item_index, item in enumerate(items):
        adversarial = load_image_tensor(attack_output / item.item_id / "adversarial.png", None).unsqueeze(0)
        generator = torch.Generator().manual_seed(args.seed + item_index * 1009)
        noises = torch.cat([sample_noise(args.probe_mode, adversarial, args.probe_sigma, generator=generator) for _ in range(args.probes)])
        images = (adversarial + noises).clamp(0.0, 1.0)
        common_probes[item.item_id] = images
        effective = images - adversarial
        probe_stats[item.item_id] = {
            "effective_noise_mean": float(effective.mean()),
            "effective_noise_std": float(effective.std(unbiased=False)),
            "saturated_pixel_fraction": float(((adversarial + noises <= 0) | (adversarial + noises >= 1)).float().mean()),
        }

    per_model: dict[tuple[str, str], dict[str, Any]] = {}
    model_rows = []
    model_specs = [("attack", raw) for raw in group["attack"]] + [("heldout", raw) for raw in group["heldout"]]
    for role, raw in model_specs:
        config = SurrogateConfig(**raw)
        model = create_surrogate(config, device, cache_dir=args.cache_dir)
        try:
            for item in items:
                positives = _encode_examples(model, item.positive_image_paths, config.input_size, device)
                prototype = normalized_prototype(positives)
                probes = F.interpolate(common_probes[item.item_id], size=(config.input_size, config.input_size), mode="bilinear", align_corners=False, antialias=True).to(device)
                probes.requires_grad_(role == "attack")
                embeddings = model.encode_image(probes)
                distances = prototype_distance(embeddings, prototype)
                record: dict[str, Any] = {
                    "embeddings": embeddings.detach().cpu(),
                    "distances": distances.detach().cpu(),
                }
                row = {
                    "item_id": item.item_id,
                    "model": model.name,
                    "role": role,
                    "prototype_distance_mean": float(distances.mean()),
                    "prototype_distance_variance": float(distances.var(unbiased=False)),
                    **probe_stats[item.item_id],
                }
                if role == "attack":
                    negatives = _encode_examples(model, item.negative_image_paths, config.input_size, device)
                    losses = _per_sample_loss(embeddings, positives, negatives, args.temperature, args.top_k)
                    gradients = torch.autograd.grad(losses.sum(), probes)[0]
                    row.update({
                        "loss_mean": float(losses.mean()),
                        "loss_variance": float(losses.var(unbiased=False)),
                        **gradient_stability(gradients),
                    })
                per_model[(item.item_id, model.name)] = record
                model_rows.append(row)
        finally:
            unload_surrogate(model)

    pair_rows = []
    for item in items:
        for proxy in sorted(attack_ids):
            for target in sorted(heldout_ids):
                proxy_record = per_model[(item.item_id, proxy)]
                target_record = per_model[(item.item_id, target)]
                proxy_embeddings = proxy_record["embeddings"]
                target_embeddings = target_record["embeddings"]
                row = {
                    "item_id": item.item_id,
                    "proxy": proxy,
                    "target": target,
                    "probe_mode": args.probe_mode,
                    "probe_sigma": args.probe_sigma,
                    "probes": args.probes,
                    "dproxy_dtarget_pearson": pearson_correlation(proxy_record["distances"], target_record["distances"]),
                    "local_centered_linear_cka": centered_linear_cka(proxy_embeddings, target_embeddings),
                    "local_uncentered_kernel_alignment": uncentered_kernel_alignment(proxy_embeddings, target_embeddings),
                }
                for k in (1, 2, 4):
                    if k < args.probes:
                        row[f"local_neighborhood_overlap_at_{k}"] = neighborhood_overlap(proxy_embeddings, target_embeddings, k)
                pair_rows.append(row)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_record = {
        "schema_version": 1,
        "analysis_only": True,
        "standardized_probe": True,
        "probe_mode": args.probe_mode,
        "probe_sigma": args.probe_sigma,
        "probes": args.probes,
        "seed": args.seed,
        "dataset": args.dataset,
        "surrogate_group": args.surrogate_group,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "attack_output": str(attack_output.resolve()),
        "attack_output_id": hashlib.sha256(str(attack_output.resolve()).encode()).hexdigest(),
        "item_ids": [item.item_id for item in items],
        **git_state(Path.cwd()),
    }
    atomic_write_json(output / "run_manifest.json", manifest_record)
    _atomic_csv(output / "noise_local_stability_models.csv", model_rows)
    _atomic_csv(output / "noise_local_stability.csv", pair_rows)
    print(json.dumps({"items": len(items), "model_rows": len(model_rows), "pair_rows": len(pair_rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

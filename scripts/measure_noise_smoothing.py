#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any

import torch
import torch.nn.functional as F

from config import SurrogateConfig
from data import load_image_tensor, load_manifest
from noise_theory_metrics import (
    bootstrap_smoothing_diagnostics,
    clipping_statistics,
    jensen_gap,
    smoothed_embeddings,
    smoothing_diagnostics,
)
from representation_metrics import neighborhood_overlap
from surrogates import create_surrogate, unload_surrogate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sample_noise(shape: torch.Size, mode: str, sigma: float, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    if mode == "gaussian":
        return torch.randn(shape, generator=generator, device=device) * sigma
    if mode == "uniform":
        return (torch.rand(shape, generator=generator, device=device) * 2 - 1) * (math.sqrt(3) * sigma)
    if mode == "rademacher":
        return (torch.randint(0, 2, shape, generator=generator, device=device) * 2 - 1).float() * sigma
    raise ValueError(f"unsupported noise mode: {mode}")


def _encode_condition(
    spec: SurrogateConfig,
    items: list,
    attack_output: Path,
    *,
    device: torch.device,
    cache_dir: str,
    noise_mode: str,
    sigma: float,
    noise_samples: int,
    seed: int,
    sample_noise: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, list[dict[str, float]]]:
    model = create_surrogate(spec, str(device), cache_dir=cache_dir)
    base_rows, sample_rows, anchors, noise_rows = [], [], [], []
    try:
        for index, item in enumerate(items):
            adversarial = load_image_tensor(attack_output / item.item_id / "adversarial.png", spec.input_size).to(device)
            positives = torch.stack([load_image_tensor(path, spec.input_size) for path in item.positive_image_paths]).to(device)
            with torch.no_grad():
                base_rows.append(model.encode_image(adversarial.unsqueeze(0)).squeeze(0).cpu())
                anchors.append(F.normalize(model.encode_image(positives).mean(0), dim=0).cpu())
                if sample_noise:
                    generator = torch.Generator(device=device).manual_seed(seed + index * 1_000_003)
                    noise = _sample_noise(
                        torch.Size((noise_samples, *adversarial.shape)), noise_mode, sigma, generator, device
                    )
                    sampled = (adversarial.unsqueeze(0) + noise).clamp(0, 1)
                    sample_rows.append(model.encode_image(sampled).cpu())
                    noise_rows.append({"item_id": item.item_id, **clipping_statistics(adversarial, sampled)})
    finally:
        unload_surrogate(model)
    return (
        torch.stack(base_rows),
        torch.stack(sample_rows, dim=1) if sample_rows else None,
        torch.stack(anchors),
        noise_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure noise-smoothing diagnostics on saved attacks.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--attack-output", required=True)
    parser.add_argument("--proxy-model", required=True, help="JSON SurrogateConfig")
    parser.add_argument("--target-model", required=True, help="JSON SurrogateConfig")
    parser.add_argument("--noise-mode", choices=["gaussian", "uniform", "rademacher"], default="gaussian")
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--noise-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="models/open_clip")
    parser.add_argument("--trial-result")
    parser.add_argument("--heldout-summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.noise_samples < 2:
        raise ValueError("noise-samples must be at least 2 for smoothing diagnostics")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.manifest)
    attack_output = Path(args.attack_output)
    items = [item for item in manifest.items if (attack_output / item.item_id / "adversarial.png").is_file()][: args.limit]
    if len(items) < 3:
        raise ValueError("at least three completed attack items are required")
    proxy_spec = SurrogateConfig(**json.loads(args.proxy_model))
    target_spec = SurrogateConfig(**json.loads(args.target_model))

    proxy, samples, proxy_anchors, noise_rows = _encode_condition(
        proxy_spec, items, attack_output, device=device, cache_dir=args.cache_dir,
        noise_mode=args.noise_mode, sigma=args.sigma, noise_samples=args.noise_samples,
        seed=args.seed, sample_noise=True,
    )
    target, _, target_anchors, _ = _encode_condition(
        target_spec, items, attack_output, device=device, cache_dir=args.cache_dir,
        noise_mode=args.noise_mode, sigma=args.sigma, noise_samples=args.noise_samples,
        seed=args.seed, sample_noise=False,
    )
    assert samples is not None
    raw_mean, renormalized_mean = smoothed_embeddings(samples)
    gaps = jensen_gap(samples, proxy_anchors)
    diagnostics = smoothing_diagnostics(proxy, samples, target)
    uncertainty = bootstrap_smoothing_diagnostics(
        proxy, samples, target, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )

    metadata = {
        "proxy": f"{proxy_spec.model_name}:{proxy_spec.pretrained}",
        "target": f"{target_spec.model_name}:{target_spec.pretrained}",
        "noise_mode": args.noise_mode, "sigma": args.sigma, "noise_samples": args.noise_samples,
        "seed": args.seed, "items": len(items), "attack_output": str(attack_output.resolve()),
        "normalized_before_averaging": True,
    }
    jensen_rows = []
    for index, item in enumerate(items):
        jensen_rows.append({
            **metadata, "item_id": item.item_id, "jensen_gap": float(gaps[index]),
            "augmented_proxy_objective": float(torch.linalg.vector_norm(samples[:, index] - proxy_anchors[index], dim=-1).mean()),
            "smoothed_proxy_distance_raw": float(torch.linalg.vector_norm(raw_mean[index] - proxy_anchors[index])),
            "smoothed_proxy_distance_renormalized": float(torch.linalg.vector_norm(renormalized_mean[index] - proxy_anchors[index])),
            "ordinary_proxy_distance": float(torch.linalg.vector_norm(proxy[index] - proxy_anchors[index])),
            "ordinary_target_distance": float(torch.linalg.vector_norm(target[index] - target_anchors[index])),
            **noise_rows[index],
        })

    kernel_rows = []
    for definition, values in diagnostics.items():
        intervals = {
            key: json.dumps(value) if isinstance(value, list) else value
            for key, value in uncertainty[definition].items()
        }
        kernel_rows.append({**metadata, "kernel_definition": definition, **values, **intervals})
    no_rows = []
    for k in (1, 5, 10):
        if k < len(items):
            no_rows.append({
                **metadata, "k": k,
                "proxy_target_no_before": neighborhood_overlap(proxy, target, k),
                "proxy_target_no_after_raw": neighborhood_overlap(raw_mean, target, k),
                "proxy_target_no_after_renormalized": neighborhood_overlap(renormalized_mean, target, k),
            })

    output = Path(args.output)
    _write_csv(output / "jensen_gap.csv", jensen_rows)
    _write_csv(output / "residual_assumption_tests.csv", kernel_rows)
    _write_csv(output / "smoothed_kernel_alignment.csv", kernel_rows + no_rows)
    strength = [{
        **metadata,
        "mean_jensen_gap": sum(row["jensen_gap"] for row in jensen_rows) / len(jensen_rows),
        "mean_augmented_proxy_objective": sum(row["augmented_proxy_objective"] for row in jensen_rows) / len(jensen_rows),
        "mean_ordinary_proxy_distance": sum(row["ordinary_proxy_distance"] for row in jensen_rows) / len(jensen_rows),
        "mean_ordinary_target_distance": sum(row["ordinary_target_distance"] for row in jensen_rows) / len(jensen_rows),
    }]
    _write_csv(output / "strength_tradeoff.csv", strength)
    joined = []
    if args.trial_result and args.heldout_summary:
        trial = json.loads(Path(args.trial_result).read_text(encoding="utf-8"))
        heldout = json.loads(Path(args.heldout_summary).read_text(encoding="utf-8"))
        target_heldout = (heldout.get("heldout_models") or {}).get(metadata["target"]) or {}
        joined = [{**metadata, **{key: trial.get(key) for key in ("dataset", "surrogate_group", "mode", "stage")},
                   "heldout_macro_asr": heldout.get("heldout_macro_asr"),
                   "heldout_macro_margin_gain": heldout.get("heldout_macro_margin_gain"),
                   "heldout_target_asr": target_heldout.get("asr"),
                   "heldout_target_margin_gain": target_heldout.get("mean_margin_gain"),
                   **{f"raw_centered_{key}": value for key, value in diagnostics["raw_mean_centered"].items()}}]
    _write_csv(output / "alignment_asr_join.csv", joined)
    (output / "run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "items": len(items), "kernel_definitions": len(kernel_rows)}, indent=2))


if __name__ == "__main__":
    main()

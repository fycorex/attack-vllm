#!/usr/bin/env python3
"""Extract FP32 pooled image-token features and CKA for both galleries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import kendalltau

from proxy_selector.adapters import CLIPHFAdapter, Qwen35TokenAdapter, SigLIP2Adapter, VLMTargetTokenAdapter
from proxy_selector.cka import bootstrap_cka, centered_linear_cka
from proxy_selector.io import atomic_json


def checkpoint(model_id: str) -> Path:
    snapshots = Path("model_cache") / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return next(snapshots.glob("*"))


def adapter(identifier: str):
    if identifier == "P1": return Qwen35TokenAdapter("Qwen/Qwen3.5-4B", checkpoint("Qwen/Qwen3.5-4B"))
    if identifier == "P2": return CLIPHFAdapter(checkpoint("openai/clip-vit-large-patch14"))
    if identifier == "P3": return SigLIP2Adapter(checkpoint("google/siglip2-so400m-patch14-384"))
    if identifier == "T1": return VLMTargetTokenAdapter("google/gemma-4-E2B-it", checkpoint("google/gemma-4-E2B-it"), family="Gemma4", prompt="<|image|> describe this image", image_keys=("pixel_values", "image_position_ids"))
    if identifier == "T2": return VLMTargetTokenAdapter("OpenGVLab/InternVL3_5-2B-HF", checkpoint("OpenGVLab/InternVL3_5-2B-HF"), family="InternVL3.5", prompt="<IMG_CONTEXT> describe this image", image_keys=("pixel_values",))
    raise ValueError(identifier)


def extract_gallery(gallery: Path, output: Path, identifiers: list[str]) -> dict[str, np.ndarray]:
    records = json.loads(gallery.read_text())["images"]
    ids = [str(row["image_id"]) for row in records]
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "metadata.json", {"gallery": str(gallery), "image_ids": ids, "models": identifiers})
    all_features: dict[str, np.ndarray] = {}
    for identifier in identifiers:
        path = output / f"{identifier}.npy"
        if path.exists():
            all_features[identifier] = np.load(path); continue
        model = adapter(identifier); features = []
        for record in records:
            image = Image.open(record["image_path"]).convert("RGB")
            if hasattr(model, "encode_pil"):
                result = model.encode_pil(image)
            else:
                pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float().div(255).to(model.device)
                result = model.encode_image_tokens(pixels, require_grad=False)
            features.append(result.global_features[0].float().cpu().numpy())
        matrix = np.stack(features).astype(np.float32)
        np.save(path, matrix); all_features[identifier] = matrix
        model.unload(); del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return all_features


def write_cka(features: dict[str, np.ndarray], output: Path, seed: int) -> None:
    rows, intervals = [], {}
    keys = list(features)
    for left in keys:
        for right in keys:
            value = bootstrap_cka(features[left], features[right], repetitions=100, seed=seed)
            rows.append({"left": left, "right": right, "cka": value.value, "ci_low": value.low, "ci_high": value.high})
            intervals[f"{left}->{right}"] = {"cka": value.value, "ci_low": value.low, "ci_high": value.high}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(output.with_suffix(".json"), intervals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/proxy_selector_vqav2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_selector_pilot"))
    parser.add_argument("--models", nargs="+", default=["P1", "P2", "P3", "T1", "T2"])
    args = parser.parse_args()
    cka_by_seed: dict[int, dict[str, dict[str, float]]] = {}
    for seed in (42, 43):
        features = extract_gallery(args.data / f"gallery_seed{seed}.json", args.output / "representations" / f"gallery_seed{seed}", args.models)
        write_cka(features, args.output / "cka" / f"cka_seed{seed}.csv", seed)
        rows = list(csv.DictReader((args.output / "cka" / f"cka_seed{seed}.csv").open(encoding="utf-8")))
        cka_by_seed[seed] = {f"{row['left']}->{row['right']}": {key: float(row[key]) for key in ("cka", "ci_low", "ci_high")} for row in rows}
    rankings: dict[str, dict[str, object]] = {}
    for target in ("T1", "T2"):
        primary = sorted((proxy for proxy in ("P1", "P2", "P3") if proxy in args.models), key=lambda proxy: cka_by_seed[42][f"{proxy}->{target}"]["cka"], reverse=True)
        stability = sorted((proxy for proxy in ("P1", "P2", "P3") if proxy in args.models), key=lambda proxy: cka_by_seed[43][f"{proxy}->{target}"]["cka"], reverse=True)
        ranks_primary = [primary.index(proxy) for proxy in ("P1", "P2", "P3") if proxy in primary]
        ranks_stability = [stability.index(proxy) for proxy in ("P1", "P2", "P3") if proxy in stability]
        rankings[target] = {"primary_ranking": primary, "stability_ranking": stability, "kendall_tau": float(kendalltau(ranks_primary, ranks_stability).statistic)}
    atomic_json(args.output / "cka" / "cka_bootstrap.json", {"seed42": cka_by_seed[42], "seed43": cka_by_seed[43], "proxy_rankings": rankings})
    print("CKA extraction complete")


if __name__ == "__main__":
    main()

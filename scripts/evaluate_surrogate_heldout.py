#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    args = parser.parse_args()

    spec = load_composition_spec(args.composition_config)
    manifest = load_manifest(args.manifest)
    items = manifest.items[:args.limit] if args.limit else manifest.items
    attack_output = Path(args.attack_output)
    device = args.device if torch.cuda.is_available() else "cpu"
    rows = []
    for model_id in spec.heldout_models:
        metadata = spec.models[model_id]
        wrapper = create_surrogate(SurrogateConfig(**metadata.victim_config()), device, cache_dir=args.cache_dir)
        try:
            for item in items:
                item_dir = attack_output / item.item_id
                if not (item_dir / "adversarial.png").is_file():
                    rows.append({"item_id": item.item_id, "model": model_id, "missing": True, "proxy_success": False})
                    continue
                size = metadata.input_size
                clean = load_image_tensor(item_dir / "clean.png", size).unsqueeze(0).to(device)
                adversarial = load_image_tensor(item_dir / "adversarial.png", size).unsqueeze(0).to(device)
                positives = torch.stack([load_image_tensor(path, size) for path in item.positive_image_paths]).to(device)
                negatives = torch.stack([load_image_tensor(path, size) for path in item.negative_image_paths]).to(device)
                with torch.no_grad():
                    clean_embedding = wrapper.encode_image(clean)
                    adversarial_embedding = wrapper.encode_image(adversarial)
                    positive_embeddings = wrapper.encode_image(positives)
                    negative_embeddings = wrapper.encode_image(negatives)
                    prototype = _prototype(positive_embeddings)
                    result = evaluate_proxy(clean_embedding, adversarial_embedding, positive_embeddings, negative_embeddings,
                                            top_k=min(args.top_k, len(item.positive_image_paths)), success_margin_threshold=0.0)
                rows.append({
                    "item_id": item.item_id,
                    "model": model_id,
                    "missing": False,
                    "clean_prototype_distance": float(torch.linalg.vector_norm(clean_embedding - prototype)),
                    "adversarial_prototype_distance": float(torch.linalg.vector_norm(adversarial_embedding - prototype)),
                    "prototype_distance_change": float(torch.linalg.vector_norm(adversarial_embedding - prototype) - torch.linalg.vector_norm(clean_embedding - prototype)),
                    **result,
                })
        finally:
            unload_surrogate(wrapper)

    per_model = {}
    for model_id in spec.heldout_models:
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
    summary = {
        "attempted_item_model_pairs": len(rows),
        "valid_item_model_pairs": sum(not row["missing"] for row in rows),
        "heldout_models": per_model,
        "heldout_macro_asr": sum(value["asr"] for value in per_model.values()) / max(1, len(per_model)),
        "heldout_macro_margin_gain": sum(value["mean_margin_gain"] for value in per_model.values()) / max(1, len(per_model)),
        "items": rows,
    }
    atomic_write_json(Path(args.output), summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "items"}, indent=2))


if __name__ == "__main__":
    main()

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


def _batches(values: list, batch_size: int):
    for offset in range(0, len(values), batch_size):
        yield values[offset:offset + batch_size]


def _encode_batch(
    wrapper,
    batch: list,
    size: int,
    device: str,
    top_k: int,
) -> list[dict]:
    """Evaluate equally-shaped image groups with four batched encoder forwards.

    Metric computation remains per item.  Only independent image encodes are
    combined, so this changes throughput rather than evaluation semantics.
    """
    clean = torch.stack([load_image_tensor(entry[1] / "clean.png", size) for entry in batch]).to(device)
    adversarial = torch.stack([load_image_tensor(entry[1] / "adversarial.png", size) for entry in batch]).to(device)
    positive_counts = len(batch[0][0].positive_image_paths)
    negative_counts = len(batch[0][0].negative_image_paths)
    positives = torch.stack([
        torch.stack([load_image_tensor(path, size) for path in item.positive_image_paths])
        for item, _ in batch
    ]).to(device)
    negatives = torch.stack([
        torch.stack([load_image_tensor(path, size) for path in item.negative_image_paths])
        for item, _ in batch
    ]).to(device)
    with torch.inference_mode():
        clean_embeddings = wrapper.encode_image(clean)
        adversarial_embeddings = wrapper.encode_image(adversarial)
        positive_embeddings = wrapper.encode_image(positives.flatten(0, 1)).reshape(len(batch), positive_counts, -1)
        negative_embeddings = wrapper.encode_image(negatives.flatten(0, 1)).reshape(len(batch), negative_counts, -1)

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
    return rows


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

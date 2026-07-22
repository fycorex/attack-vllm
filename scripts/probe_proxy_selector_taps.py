#!/usr/bin/env python3
"""Probe and record validated image-token tap shapes for local checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor


SPECS = {
    "P1": ("Qwen/Qwen3.5-4B", "describe this image", ("pixel_values", "image_grid_thw")),
    "T1": ("google/gemma-4-E2B-it", "<|image|> describe this image", ("pixel_values", "image_position_ids")),
    "T2": ("OpenGVLab/InternVL3_5-2B-HF", "<IMG_CONTEXT> describe this image", ("pixel_values",)),
}


def checkpoint(cache: Path, model_id: str) -> Path:
    snapshots = cache / f"models--{model_id.replace('/', '--')}" / "snapshots"
    matches = sorted(snapshots.iterdir())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one local snapshot for {model_id}: {snapshots}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("model_cache"))
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_selector_pilot/smoke/tap_probe.json"))
    parser.add_argument("--model", choices=tuple(SPECS), nargs="+", default=tuple(SPECS))
    args = parser.parse_args()
    image = Image.new("RGB", (224, 224), (128, 80, 40))
    records: dict[str, dict[str, object]] = {}
    for identifier in args.model:
        model_id, prompt, image_keys = SPECS[identifier]
        path = checkpoint(args.cache_dir, model_id)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=identifier == "T2")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        model = AutoModelForImageTextToText.from_pretrained(
            path, local_files_only=True, trust_remote_code=identifier == "T2", dtype=torch.bfloat16
        ).eval().requires_grad_(False).cuda()
        kwargs = {key: inputs[key].cuda() for key in image_keys if key in inputs}
        with torch.no_grad():
            output = model.get_image_features(**kwargs)
        tokens = output.last_hidden_state
        records[identifier] = {
            "model_id": model_id,
            "revision": path.name,
            "tap_path": "get_image_features(...).last_hidden_state",
            "shape": list(tokens.shape),
            "contains_text_tokens": False,
            "processor_inputs": {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")},
        }
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

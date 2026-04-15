#!/usr/bin/env python3
"""
Prepare demo dataset using Caltech101 (real images, 101 object categories).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import random

from PIL import Image
from torchvision.datasets import Caltech101

# Select diverse classes from Caltech101
SELECTED_CLASSES = [
    "airplane",
    "car_side",
    "dalmatian",
    "dog",
    "cat",
    "deer",
    "watch",
    "camera",
    "laptop",
    "cellphone",
]

SIMPLE_NAMES = {
    "airplane": "airplane",
    "car_side": "car",
    "dalmatian": "dog",
    "dog": "dog",
    "cat": "cat",
    "deer": "deer",
    "watch": "watch",
    "camera": "camera",
    "laptop": "laptop",
    "cellphone": "phone",
}

KEYWORDS = {
    "airplane": ["airplane", "plane", "aircraft"],
    "car": ["car", "automobile", "vehicle"],
    "dog": ["dog", "puppy", "canine"],
    "cat": ["cat", "kitten", "feline"],
    "deer": ["deer", "animal"],
    "watch": ["watch", "timepiece"],
    "camera": ["camera", "photography"],
    "laptop": ["laptop", "computer"],
    "phone": ["phone", "cellphone", "mobile"],
}


def main():
    parser = argparse.ArgumentParser(description="Prepare Caltech101 demo dataset")
    parser.add_argument("--output_dir", default="data/caltech_demo")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Caltech101 dataset...")
    ds = Caltech101(root=str(output_dir / "torchvision"), download=True)

    # Find available classes
    class_to_indices = {}
    for idx, (_, label) in enumerate(ds):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)

    # Map class indices to names
    class_names = ds.categories
    available = []
    for cls_name in SELECTED_CLASSES:
        if cls_name in class_names:
            cls_idx = class_names.index(cls_name)
            if len(class_to_indices.get(cls_idx, [])) >= args.num_examples + 1:
                available.append((cls_idx, cls_name))

    print(f"Found {len(available)} usable classes")

    if len(available) < 2:
        raise RuntimeError("Need at least two usable Caltech101 classes to build source-target pairs.")

    items = []
    for item_idx in range(args.num_items):
        source_position = item_idx % len(available)
        source_idx, source_raw = available[source_position]
        target_idx, target_raw = available[(source_position + 1) % len(available)]
        class_cycle = item_idx // len(available)

        source_name = SIMPLE_NAMES.get(source_raw, source_raw)
        target_name = SIMPLE_NAMES.get(target_raw, target_raw)

        print(f"Preparing item {item_idx}: {source_name} -> {target_name}")

        # Get source image
        src_indices = class_to_indices[source_idx]
        img, _ = ds[src_indices[class_cycle % len(src_indices)]]

        image_path = images_dir / f"item_{item_idx:02d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
        img.save(image_path)

        # Positive examples (target class)
        positive_paths = []
        tgt_indices = class_to_indices[target_idx]
        for ex_idx in range(args.num_examples):
            ex_img, _ = ds[tgt_indices[(class_cycle + ex_idx) % len(tgt_indices)]]
            if ex_img.mode != "RGB":
                ex_img = ex_img.convert("RGB")
            ex_path = examples_dir / target_name / f"positive_{item_idx:02d}_{ex_idx:02d}.png"
            ex_path.parent.mkdir(parents=True, exist_ok=True)
            ex_img = ex_img.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
            ex_img.save(ex_path)
            positive_paths.append(str(ex_path.resolve()))

        # Negative examples (source class)
        negative_paths = []
        for ex_idx in range(args.num_examples):
            ex_img, _ = ds[src_indices[(class_cycle + ex_idx + 1) % len(src_indices)]]
            if ex_img.mode != "RGB":
                ex_img = ex_img.convert("RGB")
            ex_path = examples_dir / source_name / f"negative_{item_idx:02d}_{ex_idx:02d}.png"
            ex_path.parent.mkdir(parents=True, exist_ok=True)
            ex_img = ex_img.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
            ex_img.save(ex_path)
            negative_paths.append(str(ex_path.resolve()))

        items.append({
            "id": f"item_{item_idx:02d}",
            "image_path": str(image_path.resolve()),
            "source_label": source_name,
            "target_label": target_name,
            "source_keywords": KEYWORDS.get(source_name, [source_name]),
            "target_keywords": KEYWORDS.get(target_name, [target_name]),
            "question": "What is the main object in the image? Answer in one word.",
            "source_answer_text": source_name,
            "target_answer_text": target_name,
            "source_answer_keywords": KEYWORDS.get(source_name, [source_name]),
            "target_answer_keywords": KEYWORDS.get(target_name, [target_name]),
            "positive_image_paths": positive_paths,
            "negative_image_paths": negative_paths,
        })

    manifest = {
        "dataset_name": "caltech101-demo",
        "metadata": {
            "note": "Real images from Caltech101 dataset",
            "image_size": args.image_size,
            "num_items": len(items),
        },
        "items": items,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()

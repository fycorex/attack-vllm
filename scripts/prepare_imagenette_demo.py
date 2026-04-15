#!/usr/bin/env python3
"""
Prepare demo dataset using ImageNette (real ImageNet images).

ImageNette is a subset of 10 easily classified classes from ImageNet.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import random

from PIL import Image
from torchvision.datasets import ImageNette

# ImageNette class mapping
IMAGENETTE_CLASSES = {
    0: "tench",  # fish
    1: "English_springer",  # dog
    2: "cassette_player",
    3: "chain_saw",
    4: "church",
    5: "French_horn",
    6: "garbage_truck",
    7: "gas_pump",
    8: "golf_ball",
    9: "parachute",
}

# Simplified class names for VLM
SIMPLE_NAMES = {
    "tench": "fish",
    "English_springer": "dog",
    "cassette_player": "player",
    "chain_saw": "saw",
    "church": "church",
    "French_horn": "horn",
    "garbage_truck": "truck",
    "gas_pump": "pump",
    "golf_ball": "ball",
    "parachute": "parachute",
}

KEYWORDS = {
    "fish": ["fish", "tench"],
    "dog": ["dog", "spaniel"],
    "player": ["player", "cassette"],
    "saw": ["saw", "chain"],
    "church": ["church", "building"],
    "horn": ["horn", "instrument"],
    "truck": ["truck", "vehicle"],
    "pump": ["pump", "gas"],
    "ball": ["ball", "golf"],
    "parachute": ["parachute", "parachuting"],
}


def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNette demo dataset")
    parser.add_argument("--output_dir", default="data/imagenette_demo")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ImageNette dataset...")
    train_ds = ImageNette(root=str(output_dir / "torchvision"), split="train", download=True)
    val_ds = ImageNette(root=str(output_dir / "torchvision"), split="val", download=True)

    # Group by class
    train_by_class = {}
    for idx, (_, label) in enumerate(train_ds):
        if label not in train_by_class:
            train_by_class[label] = []
        train_by_class[label].append(idx)

    val_by_class = {}
    for idx, (_, label) in enumerate(val_ds):
        if label not in val_by_class:
            val_by_class[label] = []
        val_by_class[label].append(idx)

    # Select classes with enough images
    available_classes = [c for c in range(10) if len(train_by_class.get(c, [])) >= args.num_examples]

    items = []
    for item_idx in range(min(args.num_items, len(available_classes) - 1)):
        source_class = available_classes[item_idx]
        target_class = available_classes[(item_idx + 1) % len(available_classes)]

        source_name = SIMPLE_NAMES[IMAGENETTE_CLASSES[source_class]]
        target_name = SIMPLE_NAMES[IMAGENETTE_CLASSES[target_class]]

        print(f"Preparing item {item_idx}: {source_name} -> {target_name}")

        # Get source image from validation set
        val_idx = val_by_class[source_class][0]
        source_img, _ = val_ds[val_idx]

        # Resize and save
        image_path = images_dir / f"item_{item_idx:02d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        source_img = source_img.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
        source_img.save(image_path)

        # Positive examples (target class)
        positive_paths = []
        for ex_idx in range(args.num_examples):
            train_idx = train_by_class[target_class][ex_idx % len(train_by_class[target_class])]
            ex_img, _ = train_ds[train_idx]
            ex_path = examples_dir / target_name / f"positive_{item_idx:02d}_{ex_idx:02d}.png"
            ex_path.parent.mkdir(parents=True, exist_ok=True)
            ex_img = ex_img.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
            ex_img.save(ex_path)
            positive_paths.append(str(ex_path.resolve()))

        # Negative examples (source class)
        negative_paths = []
        for ex_idx in range(args.num_examples):
            train_idx = train_by_class[source_class][(ex_idx + 1) % len(train_by_class[source_class])]
            ex_img, _ = train_ds[train_idx]
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
        "dataset_name": "imagenette-demo",
        "metadata": {
            "note": "Real ImageNet images from ImageNette subset",
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

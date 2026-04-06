#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import json

from PIL import Image
from torchvision.datasets import CIFAR10


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_KEYWORDS = {
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "automobile": ["automobile", "car", "vehicle"],
    "bird": ["bird"],
    "cat": ["cat", "kitten"],
    "deer": ["deer"],
    "dog": ["dog", "puppy"],
    "frog": ["frog", "toad"],
    "horse": ["horse"],
    "ship": ["ship", "boat", "vessel"],
    "truck": ["truck", "lorry"],
}


def save_image(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        image.save(path)
    else:
        Image.fromarray(image).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a tiny CIFAR-10 demo manifest.")
    parser.add_argument("--output_dir", default="data/cifar10_caption_attack_demo")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--num_examples_per_class", type=int, default=8)
    parser.add_argument("--target_offset", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CIFAR10(root=str(output_dir / "torchvision"), train=True, download=True)
    test_ds = CIFAR10(root=str(output_dir / "torchvision"), train=False, download=True)

    train_by_label = {i: [] for i in range(len(CIFAR10_CLASSES))}
    test_by_label = {i: [] for i in range(len(CIFAR10_CLASSES))}
    for idx, (_, label) in enumerate(train_ds):
        train_by_label[label].append(idx)
    for idx, (_, label) in enumerate(test_ds):
        test_by_label[label].append(idx)

    items = []
    for item_idx, source_label in enumerate(range(args.num_items)):
        target_label = (source_label + args.target_offset) % len(CIFAR10_CLASSES)
        test_index = test_by_label[source_label][0]
        image, _ = test_ds[test_index]
        image_path = images_dir / f"item_{item_idx:02d}.png"
        save_image(image, image_path)

        positive_paths = []
        negative_paths = []
        for example_idx in range(args.num_examples_per_class):
            pos_image, _ = train_ds[train_by_label[target_label][example_idx]]
            neg_image, _ = train_ds[train_by_label[source_label][example_idx]]
            pos_path = examples_dir / CIFAR10_CLASSES[target_label] / f"positive_{item_idx:02d}_{example_idx:02d}.png"
            neg_path = examples_dir / CIFAR10_CLASSES[source_label] / f"negative_{item_idx:02d}_{example_idx:02d}.png"
            save_image(pos_image, pos_path)
            save_image(neg_image, neg_path)
            positive_paths.append(str(pos_path.resolve()))
            negative_paths.append(str(neg_path.resolve()))

        source_name = CIFAR10_CLASSES[source_label]
        target_name = CIFAR10_CLASSES[target_label]
        items.append(
            {
                "id": f"item_{item_idx:02d}",
                "image_path": str(image_path.resolve()),
                "source_label": source_name,
                "target_label": target_name,
                "source_keywords": CIFAR10_KEYWORDS[source_name],
                "target_keywords": CIFAR10_KEYWORDS[target_name],
                "positive_image_paths": positive_paths,
                "negative_image_paths": negative_paths,
            }
        )

    manifest = {
        "dataset_name": "cifar10-caption-attack-demo",
        "metadata": {
            "note": "Tiny local proxy dataset for the small caption-attack reproduction. Phase 2 adds label keyword aliases for local caption-victim checks.",
            "num_items": args.num_items,
            "num_examples_per_class": args.num_examples_per_class,
        },
        "items": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()

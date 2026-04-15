#!/usr/bin/env python3
"""
Prepare higher quality demo dataset using ImageNet or web images.

Uses 224x224+ resolution images instead of CIFAR-10's 32x32.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import urllib.request
import hashlib

from PIL import Image

# High-quality example images from Wikimedia Commons (public domain)
# These are actual high-resolution images, not 32x32 CIFAR
EXAMPLE_IMAGES = {
    "airplane": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Boeing_777-300ER_Air_France.jpg/640px-Boeing_777-300ER_Air_France.jpg", "aircraft"),
    ],
    "automobile": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/2015_Chevrolet_Corvette_Stingray%2C_front_6.30.19.jpg/640px-2015_Chevrolet_Corvette_Stingray%2C_front_6.30.19.jpg", "car"),
    ],
    "bird": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/640px-Eopsaltria_australis_-_Mogo_Campground.jpg", "bird"),
    ],
    "cat": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/640px-Cat_November_2010-1a.jpg", "cat"),
    ],
    "deer": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Deer_in_the_forest.jpg/640px-Deer_in_the_forest.jpg", "deer"),
    ],
    "dog": [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/640px-YellowLabradorLooking_new.jpg", "dog"),
    ],
}

IMAGE_SIZE = 224


def download_image(url: str, cache_dir: Path) -> Path:
    """Download image to cache if not exists."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    cache_path = cache_dir / f"{url_hash}.jpg"

    if not cache_path.exists():
        print(f"  Downloading {url[:50]}...")
        urllib.request.urlretrieve(url, cache_path)

    return cache_path


def prepare_image(src_path: Path, dst_path: Path, size: int = IMAGE_SIZE) -> None:
    """Resize and save image."""
    img = Image.open(src_path).convert("RGB")
    # Center crop to square
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare high-quality demo dataset")
    parser.add_argument("--output_dir", default="data/imagenet_demo")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    examples_dir = output_dir / "examples"
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    classes = list(EXAMPLE_IMAGES.keys())
    items = []

    for source_idx, source_class in enumerate(classes[:4]):  # Limit to 4 items
        target_class = classes[(source_idx + 1) % len(classes)]

        print(f"Preparing item: {source_class} -> {target_class}")

        # Download and prepare source image
        source_urls = EXAMPLE_IMAGES[source_class]
        src_path = download_image(source_urls[0][0], cache_dir)

        image_path = images_dir / f"item_{source_idx:02d}.png"
        prepare_image(src_path, image_path, args.image_size)

        # Prepare positive examples (target class)
        positive_paths = []
        target_urls = EXAMPLE_IMAGES[target_class]
        for ex_idx, (url, _) in enumerate(target_urls * 4):  # Repeat to get 8 examples
            ex_cache = download_image(url, cache_dir)
            ex_path = examples_dir / target_class / f"positive_{source_idx:02d}_{ex_idx:02d}.png"
            prepare_image(ex_cache, ex_path, args.image_size)
            positive_paths.append(str(ex_path.resolve()))

        # Prepare negative examples (source class)
        negative_paths = []
        for ex_idx, (url, _) in enumerate(source_urls * 4):
            ex_cache = download_image(url, cache_dir)
            ex_path = examples_dir / source_class / f"negative_{source_idx:02d}_{ex_idx:02d}.png"
            prepare_image(ex_cache, ex_path, args.image_size)
            negative_paths.append(str(ex_path.resolve()))

        items.append({
            "id": f"item_{source_idx:02d}",
            "image_path": str(image_path.resolve()),
            "source_label": source_class,
            "target_label": target_class,
            "source_keywords": [source_class],
            "target_keywords": [target_class],
            "question": "What is the main object in the image? Answer in one word.",
            "source_answer_text": source_class,
            "target_answer_text": target_class,
            "source_answer_keywords": [source_class],
            "target_answer_keywords": [target_class],
            "positive_image_paths": positive_paths[:8],
            "negative_image_paths": negative_paths[:8],
        })

    manifest = {
        "dataset_name": "imagenet-demo-highres",
        "metadata": {
            "note": "High-resolution demo dataset using Wikimedia Commons images",
            "image_size": args.image_size,
            "num_items": len(items),
        },
        "items": items,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest to {manifest_path}")
    print(f"Images saved to {images_dir}")


if __name__ == "__main__":
    main()

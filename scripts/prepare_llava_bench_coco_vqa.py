#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageEnhance, ImageOps


CATEGORIES = ("conv", "detail", "complex")
CATEGORY_NAMES = {
    "conv": "Conversation",
    "detail": "Detail",
    "complex": "Reasoning",
}


def require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The LLaVA-Bench COCO downloader requires the 'datasets' package. "
            "Install requirements first: .venv/bin/python -m pip install -r requirements.txt"
        ) from exc
    return load_dataset


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def image_to_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Expected a PIL image from the dataset, got {type(image)!r}")


def row_image_id(row: dict[str, Any]) -> str:
    return normalize_text(row.get("image_id") or row.get("question_id"))


def group_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        image_id = row_image_id(row)
        category = normalize_text(row.get("category", "")).lower()
        if category in CATEGORIES:
            grouped.setdefault(image_id, {})[category] = row
    return {
        image_id: by_category
        for image_id, by_category in grouped.items()
        if all(category in by_category for category in CATEGORIES)
    }


def save_image_once(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        image.save(path)
    return str(path.resolve())


def save_augmented_crops(
    image: Image.Image,
    output_dir: Path,
    *,
    prefix: str,
    item_id: str,
    count: int,
    rng: random.Random,
) -> list[str]:
    image = image.convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = image.size
    paths: list[str] = []
    for idx in range(count):
        if idx == 0:
            crop = ImageOps.fit(image, (width, height), method=Image.Resampling.BICUBIC)
        else:
            scale = rng.uniform(0.72, 1.0)
            crop_w = max(16, int(width * scale))
            crop_h = max(16, int(height * scale))
            left = rng.randint(0, max(0, width - crop_w))
            top = rng.randint(0, max(0, height - crop_h))
            crop = image.crop((left, top, left + crop_w, top + crop_h)).resize(
                (width, height),
                Image.Resampling.BICUBIC,
            )
        if rng.random() < 0.35:
            crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(0.9, 1.15))
        if rng.random() < 0.35:
            crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.92, 1.08))
        path = output_dir / f"{prefix}_{item_id}_{idx:02d}.png"
        crop.save(path)
        paths.append(str(path.resolve()))
    return paths


def select_target_image_ids(grouped: dict[str, dict[str, dict[str, Any]]], count: int, rng: random.Random) -> list[str]:
    image_ids = sorted(grouped)
    if len(image_ids) < count:
        raise SystemExit(f"Dataset only has {len(image_ids)} complete image groups; requested {count}.")
    rng.shuffle(image_ids)
    return image_ids[:count]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the paper-style LLaVA-Bench COCO VQA attack manifest."
    )
    parser.add_argument("--dataset_name", default="lmms-lab/llava-bench-coco")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--cache_dir",
        default="data/raw/hf_cache",
        help="Hugging Face datasets cache directory. The full split is cached here before selecting demo images.",
    )
    parser.add_argument("--output_dir", default="data/llava_bench_coco_vqa")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow_repeated_source_images", action="store_true",
        help="Use independent random source selection. The default assigns unique source images to avoid pseudo-replication.",
    )
    parser.add_argument(
        "--one_question_per_image", action="store_true",
        help="Emit one cyclically balanced question category per source image for independent-image pilots.",
    )
    args = parser.parse_args()

    load_dataset = require_datasets()
    rng = random.Random(args.seed)
    dataset = load_dataset(args.dataset_name, split=args.split, cache_dir=args.cache_dir)
    rows = [dict(row) for row in dataset]
    grouped = group_rows(rows)
    target_image_ids = select_target_image_ids(grouped, args.num_images, rng)
    all_image_ids = sorted(grouped)
    if args.allow_repeated_source_images:
        source_image_ids = [rng.choice([value for value in all_image_ids if value != target]) for target in target_image_ids]
    elif len(target_image_ids) > 1:
        # A cyclic derangement provides exactly one distinct source image per
        # target without blind random collisions.
        source_image_ids = target_image_ids[1:] + target_image_ids[:1]
    else:
        source_image_ids = [rng.choice([value for value in all_image_ids if value != target_image_ids[0]])]

    output_dir = Path(args.output_dir)
    source_dir = output_dir / "source_images"
    target_dir = output_dir / "target_images"
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for target_index, (target_image_id, source_image_id) in enumerate(zip(target_image_ids, source_image_ids)):
        source_row = grouped[source_image_id]["detail"]
        source_image = image_to_rgb(source_row["image"])
        source_caption = normalize_text(source_row.get("caption") or source_row.get("answer") or source_image_id)
        source_path = save_image_once(source_image, source_dir / source_row["image_id"],)

        categories = (CATEGORIES[target_index % len(CATEGORIES)],) if args.one_question_per_image else CATEGORIES
        for category in categories:
            target_row = grouped[target_image_id][category]
            target_image = image_to_rgb(target_row["image"])
            target_caption = normalize_text(target_row.get("caption") or target_row.get("answer") or target_image_id)
            target_answer = normalize_text(target_row.get("answer", ""))
            question = normalize_text(target_row.get("question", ""))
            item_id = f"item_{target_index:02d}_{category}"
            save_image_once(target_image, target_dir / target_row["image_id"])
            item_rng = random.Random(f"{args.seed}:{item_id}")

            positive_paths = save_augmented_crops(
                target_image,
                examples_dir / item_id / "positive",
                prefix="positive",
                item_id=item_id,
                count=args.num_examples,
                rng=item_rng,
            )
            negative_paths = save_augmented_crops(
                source_image,
                examples_dir / item_id / "negative",
                prefix="negative",
                item_id=item_id,
                count=args.num_examples,
                rng=item_rng,
            )

            items.append(
                {
                    "id": item_id,
                    "image_path": source_path,
                    "source_label": source_image_id,
                    "target_label": target_image_id,
                    "source_keywords": [source_image_id, source_caption],
                    "target_keywords": [target_image_id, target_caption],
                    "question": question,
                    "source_answer_text": source_caption,
                    "target_answer_text": target_answer,
                    "source_answer_keywords": [source_caption],
                    "target_answer_keywords": [target_answer],
                    "positive_image_paths": positive_paths,
                    "negative_image_paths": negative_paths,
                    "metadata": {
                        "dataset_name": args.dataset_name,
                        "split": args.split,
                        "question_category": category,
                        "question_category_name": CATEGORY_NAMES[category],
                        "question_id": normalize_text(target_row.get("question_id", "")),
                        "source_image_id": source_image_id,
                        "target_image_id": target_image_id,
                        "source_caption": source_caption,
                        "target_caption": target_caption,
                        "target_answer": target_answer,
                    },
                }
            )

    manifest = {
        "dataset_name": "llava-bench-coco-vqa-demo",
        "metadata": {
            "paper": "arXiv:2505.01050v1 Section 5.2 Visual Question Answering",
            "source_dataset": args.dataset_name,
            "split": args.split,
            "cache_dir": str(Path(args.cache_dir).resolve()),
            "num_target_images": len(target_image_ids),
            "num_items": len(items),
            "categories": list(CATEGORIES),
            "one_question_per_image": args.one_question_per_image,
            "num_examples_per_item": args.num_examples,
            "seed": args.seed,
            "unique_source_images": not args.allow_repeated_source_images,
            "num_unique_source_images": len(set(source_image_ids)),
            "protocol": (
                "Each target data entry provides the question and ground-truth answer. "
                "A different benchmark image is selected as the source image to perturb. "
                "Positive visual examples are crops of the target image; negative visual examples are crops of the source image."
            ),
        },
        "items": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} VQA items from {len(target_image_ids)} LLaVA-Bench COCO images to {manifest_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from torchvision.datasets import ImageFolder


def build_keywords(class_name: str) -> list[str]:
    return [class_name.replace("_", " ").lower()]


def load_pair_specs(pair_specs_json: str | None) -> list[dict] | None:
    if not pair_specs_json:
        return None
    payload = json.loads(Path(pair_specs_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("--pair_specs_json must point to a JSON array of pair spec objects.")
    return payload


def copy_sample(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small manifest from an ImageFolder-style dataset.")
    parser.add_argument("--dataset_root", required=True, help="Root directory with class subfolders.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--num_examples_per_class", type=int, default=8)
    parser.add_argument("--target_offset", type=int, default=1)
    parser.add_argument("--class_names", default=None, help="Optional comma-separated class names to include.")
    parser.add_argument(
        "--pair_specs_json",
        default=None,
        help=(
            "Optional path to a JSON array describing explicit source/target pairs. "
            "Each object may contain source_label, target_label, optional num_items, "
            "source_keywords, and target_keywords."
        ),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolder(root=str(dataset_root))
    class_to_paths: dict[str, list[Path]] = {class_name: [] for class_name in dataset.classes}
    for sample_path, class_index in dataset.samples:
        class_to_paths[dataset.classes[class_index]].append(Path(sample_path))

    pair_specs = load_pair_specs(args.pair_specs_json)

    items = []
    if pair_specs:
        item_idx = 0
        for pair_spec in pair_specs:
            source_name = pair_spec["source_label"]
            target_name = pair_spec["target_label"]
            if source_name not in class_to_paths:
                raise ValueError(f"Unknown source class name: {source_name}")
            if target_name not in class_to_paths:
                raise ValueError(f"Unknown target class name: {target_name}")

            source_paths = class_to_paths[source_name]
            target_paths = class_to_paths[target_name]
            pair_num_items = int(pair_spec.get("num_items", 1))
            if pair_num_items <= 0:
                raise ValueError("pair spec num_items must be positive.")

            for pair_item_idx in range(pair_num_items):
                source_base = pair_item_idx * (args.num_examples_per_class + 1)
                target_base = pair_item_idx * args.num_examples_per_class
                item_src = source_paths[source_base % len(source_paths)]
                item_ext = item_src.suffix or ".jpg"
                item_path = images_dir / f"item_{item_idx:02d}{item_ext}"
                copied_item_path = copy_sample(item_src, item_path)

                positive_paths = []
                negative_paths = []
                for example_idx in range(args.num_examples_per_class):
                    pos_src = target_paths[(target_base + example_idx) % len(target_paths)]
                    neg_src = source_paths[(source_base + example_idx + 1) % len(source_paths)]
                    pos_dst = examples_dir / target_name / f"positive_{item_idx:02d}_{example_idx:02d}{pos_src.suffix or '.jpg'}"
                    neg_dst = examples_dir / source_name / f"negative_{item_idx:02d}_{example_idx:02d}{neg_src.suffix or '.jpg'}"
                    positive_paths.append(copy_sample(pos_src, pos_dst))
                    negative_paths.append(copy_sample(neg_src, neg_dst))

                items.append(
                    {
                        "id": f"item_{item_idx:02d}",
                        "image_path": copied_item_path,
                        "source_label": source_name,
                        "target_label": target_name,
                        "source_keywords": pair_spec.get("source_keywords", build_keywords(source_name)),
                        "target_keywords": pair_spec.get("target_keywords", build_keywords(target_name)),
                        "positive_image_paths": positive_paths,
                        "negative_image_paths": negative_paths,
                    }
                )
                item_idx += 1
    else:
        if args.class_names:
            selected_classes = [name.strip() for name in args.class_names.split(",") if name.strip()]
        else:
            selected_classes = list(dataset.classes[: args.num_items])
        selected_classes = selected_classes[: args.num_items]

        for item_idx, source_name in enumerate(selected_classes):
            if source_name not in class_to_paths:
                raise ValueError(f"Unknown class name: {source_name}")
            target_name = selected_classes[(item_idx + args.target_offset) % len(selected_classes)]
            source_paths = class_to_paths[source_name]
            target_paths = class_to_paths[target_name]
            if len(source_paths) < args.num_examples_per_class + 1:
                raise ValueError(f"Not enough images in source class '{source_name}' for one item plus negatives.")
            if len(target_paths) < args.num_examples_per_class:
                raise ValueError(f"Not enough images in target class '{target_name}' for positives.")

            item_src = source_paths[0]
            item_ext = item_src.suffix or ".jpg"
            item_path = images_dir / f"item_{item_idx:02d}{item_ext}"
            copied_item_path = copy_sample(item_src, item_path)

            positive_paths = []
            negative_paths = []
            for example_idx in range(args.num_examples_per_class):
                pos_src = target_paths[example_idx]
                neg_src = source_paths[example_idx + 1]
                pos_dst = examples_dir / target_name / f"positive_{item_idx:02d}_{example_idx:02d}{pos_src.suffix or '.jpg'}"
                neg_dst = examples_dir / source_name / f"negative_{item_idx:02d}_{example_idx:02d}{neg_src.suffix or '.jpg'}"
                positive_paths.append(copy_sample(pos_src, pos_dst))
                negative_paths.append(copy_sample(neg_src, neg_dst))

            items.append(
                {
                    "id": f"item_{item_idx:02d}",
                    "image_path": copied_item_path,
                    "source_label": source_name,
                    "target_label": target_name,
                    "source_keywords": build_keywords(source_name),
                    "target_keywords": build_keywords(target_name),
                    "positive_image_paths": positive_paths,
                    "negative_image_paths": negative_paths,
                }
            )

    manifest = {
        "dataset_name": f"imagefolder-subset-{dataset_root.name}",
        "metadata": {
            "note": "Phase 2 local ImageFolder subset builder. This is still a local approximation, not the paper's full ImageNet/NIPS benchmark.",
            "dataset_root": str(dataset_root.resolve()),
            "num_items": len(items),
            "num_examples_per_class": args.num_examples_per_class,
            "selected_classes": selected_classes if not pair_specs else None,
            "pair_specs_json": str(Path(args.pair_specs_json).resolve()) if args.pair_specs_json else None,
        },
        "items": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()

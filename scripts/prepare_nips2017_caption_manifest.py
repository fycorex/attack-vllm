#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
IMAGE_ID_COLUMNS = ("ImageId", "image_id", "id", "filename", "file_name", "image")
SOURCE_LABEL_COLUMNS = ("TrueLabel", "true_label", "ground_truth", "source_label", "label")
TARGET_LABEL_COLUMNS = ("TargetClass", "target_class", "target_label", "target")
FILENAME_COLUMNS = ("filename", "file_name", "image", "ImageId", "image_id", "id")
LABEL_COLUMNS = ("label", "Label", "class", "class_id", "ClassId", "target", "wnid", "synset")


def normalize_name(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split()).strip()


def normalize_lookup(value: str) -> str:
    return normalize_name(value).lower()


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def choose_column(row: dict[str, str], candidates: tuple[str, ...], explicit: str | None, label: str) -> str:
    if explicit:
        if explicit not in row:
            raise KeyError(f"Column '{explicit}' for {label} not found. Available columns: {sorted(row)}")
        return explicit
    for candidate in candidates:
        if candidate in row:
            return candidate
    raise KeyError(f"Could not infer {label} column. Available columns: {sorted(row)}")


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def parse_class_index(path: Path | None, *, index_base: int = 0) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes: dict[int, dict[str, str]] = {}

    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            classes[idx] = {"name": normalize_name(str(value)), "synset": ""}
        return classes

    if not isinstance(payload, dict):
        raise ValueError("--class_index_json must be a JSON list or object.")

    for raw_key, value in payload.items():
        try:
            idx = int(raw_key) - index_base
        except ValueError:
            continue

        if isinstance(value, list):
            synset = str(value[0]) if value else ""
            name = str(value[1]) if len(value) > 1 else synset or str(raw_key)
        elif isinstance(value, dict):
            synset = str(value.get("synset") or value.get("wnid") or value.get("id") or "")
            name = str(value.get("name") or value.get("label") or value.get("class_name") or synset or raw_key)
        else:
            synset = ""
            name = str(value)

        classes[idx] = {"name": normalize_name(name), "synset": synset}
    return classes


def build_name_lookup(class_index: dict[int, dict[str, str]]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, info in class_index.items():
        lookup[str(idx)] = idx
        lookup[str(idx + 1)] = idx
        if info.get("synset"):
            lookup[normalize_lookup(info["synset"])] = idx
        for part in info.get("name", "").split(","):
            normalized = normalize_lookup(part)
            if normalized:
                lookup[normalized] = idx
    return lookup


def label_to_index(value: str, *, label_base: int, class_lookup: dict[str, int]) -> int:
    stripped = str(value).strip()
    try:
        return int(stripped) - label_base
    except ValueError:
        pass
    normalized = normalize_lookup(stripped)
    if normalized in class_lookup:
        return class_lookup[normalized]
    raise ValueError(f"Could not map label '{value}' to an ImageNet class index.")


def class_display_name(class_idx: int, class_index: dict[int, dict[str, str]]) -> str:
    info = class_index.get(class_idx)
    if not info:
        return f"class_{class_idx + 1}"
    primary = info.get("name", "").split(",")[0].strip()
    return normalize_name(primary or info.get("synset", "") or f"class_{class_idx + 1}")


def class_keywords(class_idx: int, class_index: dict[int, dict[str, str]]) -> list[str]:
    info = class_index.get(class_idx)
    if not info:
        return [f"class {class_idx + 1}"]
    values = []
    for part in info.get("name", "").split(","):
        normalized = normalize_name(part).lower()
        if normalized and normalized not in values:
            values.append(normalized)
    if not values and info.get("synset"):
        values.append(info["synset"].lower())
    return values or [f"class {class_idx + 1}"]


def resolve_image_path(images_dir: Path, image_id: str) -> Path:
    candidate = Path(str(image_id).strip())
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    direct = images_dir / candidate
    if direct.is_file():
        return direct
    for ext in IMAGE_EXTENSIONS:
        with_ext = images_dir / f"{candidate}{ext}"
        if with_ext.is_file():
            return with_ext
    matches = sorted(images_dir.glob(f"{candidate}.*"))
    for match in matches:
        if match.suffix.lower() in IMAGE_EXTENSIONS and match.is_file():
            return match
    raise FileNotFoundError(f"Could not resolve image '{image_id}' under {images_dir}")


def index_from_folder_name(folder_name: str, *, label_base: int, class_lookup: dict[str, int]) -> int | None:
    try:
        return int(folder_name) - label_base
    except ValueError:
        pass
    return class_lookup.get(normalize_lookup(folder_name))


def index_imagenet_from_folders(
    imagenet_val_dir: Path,
    *,
    label_base: int,
    class_lookup: dict[str, int],
) -> dict[int, list[Path]]:
    class_to_paths: dict[int, list[Path]] = {}
    subdirs = sorted(path for path in imagenet_val_dir.iterdir() if path.is_dir())
    for subdir in subdirs:
        class_idx = index_from_folder_name(subdir.name, label_base=label_base, class_lookup=class_lookup)
        if class_idx is None:
            continue
        paths = image_files(subdir)
        if paths:
            class_to_paths.setdefault(class_idx, []).extend(paths)
    return class_to_paths


def index_imagenet_from_csv(
    imagenet_val_dir: Path,
    csv_path: Path,
    *,
    label_base: int,
    class_lookup: dict[str, int],
) -> dict[int, list[Path]]:
    rows = load_rows(csv_path)
    filename_col = choose_column(rows[0], FILENAME_COLUMNS, None, "ImageNet filename")
    label_col = choose_column(rows[0], LABEL_COLUMNS, None, "ImageNet label")
    class_to_paths: dict[int, list[Path]] = {}
    for row in rows:
        image_path = resolve_image_path(imagenet_val_dir, row[filename_col])
        class_idx = label_to_index(row[label_col], label_base=label_base, class_lookup=class_lookup)
        class_to_paths.setdefault(class_idx, []).append(image_path)
    return class_to_paths


def index_imagenet_from_ground_truth(
    imagenet_val_dir: Path,
    ground_truth_path: Path,
    *,
    label_base: int,
    class_lookup: dict[str, int],
) -> dict[int, list[Path]]:
    paths = image_files(imagenet_val_dir)
    labels = [line.strip() for line in ground_truth_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths) != len(labels):
        raise ValueError(
            f"ImageNet val image count ({len(paths)}) does not match ground-truth label count ({len(labels)})."
        )
    class_to_paths: dict[int, list[Path]] = {}
    for image_path, label in zip(paths, labels):
        class_idx = label_to_index(label, label_base=label_base, class_lookup=class_lookup)
        class_to_paths.setdefault(class_idx, []).append(image_path)
    return class_to_paths


def copy_or_reference(src: Path, dst: Path, copy_files: bool) -> str:
    if copy_files:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst.resolve())
    return str(src.resolve())


def select_examples(
    class_to_paths: dict[int, list[Path]],
    class_idx: int,
    count: int,
    rng: random.Random,
    label: str,
) -> list[Path]:
    paths = list(class_to_paths.get(class_idx, []))
    if len(paths) < count:
        raise ValueError(f"Class '{label}' has {len(paths)} ImageNet examples, expected at least {count}.")
    rng.shuffle(paths)
    return paths[:count]


def build_imagenet_index(args, class_lookup: dict[str, int]) -> dict[int, list[Path]]:
    imagenet_val_dir = Path(args.imagenet_val_dir)
    if args.imagenet_val_csv:
        return index_imagenet_from_csv(
            imagenet_val_dir,
            Path(args.imagenet_val_csv),
            label_base=args.imagenet_label_base,
            class_lookup=class_lookup,
        )
    if args.imagenet_val_ground_truth:
        return index_imagenet_from_ground_truth(
            imagenet_val_dir,
            Path(args.imagenet_val_ground_truth),
            label_base=args.imagenet_label_base,
            class_lookup=class_lookup,
        )
    class_to_paths = index_imagenet_from_folders(
        imagenet_val_dir,
        label_base=args.imagenet_label_base,
        class_lookup=class_lookup,
    )
    if not class_to_paths:
        raise ValueError(
            "Could not infer ImageNet validation labels from folders. Provide --imagenet_val_csv or "
            "--imagenet_val_ground_truth."
        )
    return class_to_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the paper-style NIPS 2017 captioning attack manifest from local datasets."
    )
    parser.add_argument("--nips_csv", required=True, help="NIPS dev metadata CSV with ImageId, TrueLabel, TargetClass.")
    parser.add_argument("--nips_images_dir", required=True, help="Directory containing NIPS dev images.")
    parser.add_argument("--imagenet_val_dir", required=True, help="ImageNet validation images, flat or class-foldered.")
    parser.add_argument("--output_dir", default="data/nips2017_caption_attack")
    parser.add_argument("--class_index_json", default=None, help="Optional ImageNet class index JSON.")
    parser.add_argument(
        "--class_index_base",
        type=int,
        default=0,
        help="Numeric base for class_index_json object keys. Keras-style imagenet_class_index.json uses 0.",
    )
    parser.add_argument("--imagenet_val_csv", default=None, help="Optional CSV mapping ImageNet val filenames to labels.")
    parser.add_argument(
        "--imagenet_val_ground_truth",
        default=None,
        help="Optional one-label-per-line ImageNet validation ground-truth file, aligned to sorted val images.",
    )
    parser.add_argument("--image_id_col", default=None)
    parser.add_argument("--source_label_col", default=None)
    parser.add_argument("--target_label_col", default=None)
    parser.add_argument("--nips_label_base", type=int, default=1)
    parser.add_argument("--imagenet_label_base", type=int, default=1)
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy_files", action="store_true", help="Copy selected images into output_dir.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    class_index = parse_class_index(
        Path(args.class_index_json) if args.class_index_json else None,
        index_base=args.class_index_base,
    )
    class_lookup = build_name_lookup(class_index)
    imagenet_index = build_imagenet_index(args, class_lookup)
    rows = load_rows(Path(args.nips_csv))
    image_id_col = choose_column(rows[0], IMAGE_ID_COLUMNS, args.image_id_col, "NIPS image ID")
    source_col = choose_column(rows[0], SOURCE_LABEL_COLUMNS, args.source_label_col, "NIPS source label")
    target_col = choose_column(rows[0], TARGET_LABEL_COLUMNS, args.target_label_col, "NIPS target label")

    rng = random.Random(args.seed)
    items = []
    for item_idx, row in enumerate(rows[: args.limit]):
        source_idx = label_to_index(row[source_col], label_base=args.nips_label_base, class_lookup=class_lookup)
        target_idx = label_to_index(row[target_col], label_base=args.nips_label_base, class_lookup=class_lookup)
        source_label = class_display_name(source_idx, class_index)
        target_label = class_display_name(target_idx, class_index)
        nips_image = resolve_image_path(Path(args.nips_images_dir), row[image_id_col])
        item_id = f"item_{item_idx:04d}"
        item_ext = nips_image.suffix or ".png"
        image_path = copy_or_reference(nips_image, images_dir / f"{item_id}{item_ext}", args.copy_files)

        positive_sources = select_examples(imagenet_index, target_idx, args.num_examples, rng, target_label)
        negative_sources = select_examples(imagenet_index, source_idx, args.num_examples, rng, source_label)

        positive_paths = []
        negative_paths = []
        for example_idx, example_path in enumerate(positive_sources):
            dst = examples_dir / target_label.replace(" ", "_") / f"positive_{item_id}_{example_idx:02d}{example_path.suffix or '.jpg'}"
            positive_paths.append(copy_or_reference(example_path, dst, args.copy_files))
        for example_idx, example_path in enumerate(negative_sources):
            dst = examples_dir / source_label.replace(" ", "_") / f"negative_{item_id}_{example_idx:02d}{example_path.suffix or '.jpg'}"
            negative_paths.append(copy_or_reference(example_path, dst, args.copy_files))

        source_keywords = class_keywords(source_idx, class_index)
        target_keywords = class_keywords(target_idx, class_index)
        items.append(
            {
                "id": item_id,
                "image_path": image_path,
                "source_label": source_label,
                "target_label": target_label,
                "source_keywords": source_keywords,
                "target_keywords": target_keywords,
                "question": "What is the main object in the image? Answer in one word.",
                "source_answer_text": source_label,
                "target_answer_text": target_label,
                "source_answer_keywords": source_keywords,
                "target_answer_keywords": target_keywords,
                "positive_image_paths": positive_paths,
                "negative_image_paths": negative_paths,
            }
        )

    manifest = {
        "dataset_name": "nips2017-caption-attack",
        "metadata": {
            "paper": "arXiv:2505.01050v1 Section 5.1 image captioning setup",
            "nips_csv": str(Path(args.nips_csv).resolve()),
            "nips_images_dir": str(Path(args.nips_images_dir).resolve()),
            "imagenet_val_dir": str(Path(args.imagenet_val_dir).resolve()),
            "class_index_json": str(Path(args.class_index_json).resolve()) if args.class_index_json else None,
            "imagenet_val_csv": str(Path(args.imagenet_val_csv).resolve()) if args.imagenet_val_csv else None,
            "imagenet_val_ground_truth": (
                str(Path(args.imagenet_val_ground_truth).resolve()) if args.imagenet_val_ground_truth else None
            ),
            "num_items": len(items),
            "num_examples_per_class": args.num_examples,
            "copy_files": args.copy_files,
        },
        "items": items,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {manifest_path}")


if __name__ == "__main__":
    main()

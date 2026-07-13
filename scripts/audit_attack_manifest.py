#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED = ("id", "image_path", "source_label", "target_label", "positive_image_paths", "negative_image_paths")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_manifest(path: Path, *, require_unique_sources: bool = False,
                   expected_items: int | None = None, expected_unique_sources: int | None = None,
                   max_category_imbalance: int = 1) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    items = manifest.get("items") or []
    errors, warnings = [], []
    ids, hashes, categories = [], [], []
    source_target_overlap = 0
    for index, item in enumerate(items):
        missing = [name for name in REQUIRED if item.get(name) is None or item.get(name) == ""]
        if missing:
            errors.append(f"item {index} missing fields: {missing}")
            continue
        ids.append(str(item["id"]))
        image_path = Path(item["image_path"])
        if not image_path.is_file():
            errors.append(f"missing source image: {image_path}")
        else:
            hashes.append(file_sha256(image_path))
        for group in ("positive_image_paths", "negative_image_paths"):
            missing_examples = [value for value in item[group] if not Path(value).is_file()]
            if missing_examples:
                errors.append(f"{item['id']} has {len(missing_examples)} missing {group}")
        metadata = item.get("metadata") or {}
        source_id, target_id = metadata.get("source_image_id"), metadata.get("target_image_id")
        if source_id and target_id and source_id == target_id:
            source_target_overlap += 1
        category = metadata.get("question_category") or metadata.get("question_type")
        if category:
            categories.append(str(category))
        if item.get("question") and not (item.get("target_answer_text") or item.get("target_answer_keywords")):
            errors.append(f"{item['id']} has a question without target answer semantics")
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate item IDs: {duplicate_ids[:10]}")
    duplicate_source_items = len(hashes) - len(set(hashes))
    if require_unique_sources and duplicate_source_items:
        errors.append(f"source images are repeated by content: {duplicate_source_items} duplicate items")
    if source_target_overlap:
        errors.append(f"source and target image IDs overlap in {source_target_overlap} items")
    if expected_items is not None and len(items) != expected_items:
        errors.append(f"expected {expected_items} items, found {len(items)}")
    if expected_unique_sources is not None and len(set(hashes)) != expected_unique_sources:
        errors.append(f"expected {expected_unique_sources} unique sources, found {len(set(hashes))}")
    category_counts = dict(sorted(Counter(categories).items()))
    if len(category_counts) > 1 and max(category_counts.values()) - min(category_counts.values()) > max_category_imbalance:
        warnings.append(f"category imbalance exceeds {max_category_imbalance}: {category_counts}")
    return {"manifest": str(path.resolve()), "dataset_name": manifest.get("dataset_name"), "items": len(items),
            "unique_item_ids": len(set(ids)), "unique_source_images": len(set(hashes)),
            "duplicate_source_items": duplicate_source_items, "source_target_overlap": source_target_overlap,
            "category_counts": category_counts, "errors": errors, "warnings": warnings, "valid": not errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit attack manifest independence, completeness, and task semantics.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--require-unique-sources", action="store_true")
    parser.add_argument("--expected-items", type=int)
    parser.add_argument("--expected-unique-sources", type=int)
    parser.add_argument("--max-category-imbalance", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = [audit_manifest(Path(path), require_unique_sources=args.require_unique_sources,
        expected_items=args.expected_items, expected_unique_sources=args.expected_unique_sources,
        max_category_imbalance=args.max_category_imbalance) for path in args.manifest]
    result = {"manifests": reports, "valid": all(report["valid"] for report in reports)}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8"); temporary.replace(output)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create deterministic candidate/gallery manifests from local VQAv2 JSON.

The command is deliberately offline after the official files are present; it
does not choose pairs using proxy representations or query every example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from proxy_selector.io import atomic_json
from proxy_selector.vqa_normalization import normalize_answer


def category(question: str, answer: str) -> tuple[str, str]:
    text = question.lower()
    if "how many" in text or answer.isdigit(): return "count", "how_many_or_numeric"
    if any(word in text for word in ("color", "material", "shape", "kind of", "pattern")): return "attribute", "attribute_keyword"
    if any(word in text for word in ("doing", "holding", "riding", "wearing", "playing")): return "action", "action_keyword"
    if any(word in text for word in ("left", "right", "above", "below", "behind", "front", "where")): return "spatial", "spatial_keyword"
    if any(word in text for word in ("next to", "with", "between", "relation")): return "relation", "relation_keyword"
    return "object", "default_object"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/proxy_selector_vqav2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-count", type=int, default=36, help="Initial 36, then deterministic expansions of 12.")
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text())["questions"]
    annotations = {row["question_id"]: row for row in json.loads(args.annotations.read_text())["annotations"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for question in questions:
        annotation = annotations.get(question["question_id"])
        if not annotation: continue
        answers = [normalize_answer(answer["answer"]) for answer in annotation["answers"]]
        answer, votes = Counter(answers).most_common(1)[0]
        if votes < 3: continue
        item = {"question_id": question["question_id"], "question": question["question"], "answer": answer}
        item["category"], item["category_rule"] = category(item["question"], answer)
        by_image[question["image_id"]].append(item)
    candidates = []
    for image_id, items in sorted(by_image.items()):
        path = args.images / f"COCO_val2014_{image_id:012d}.jpg"
        if not path.exists(): continue
        selected = sorted(items, key=lambda row: (row["category"], row["question_id"]))[:2]
        if len(selected) == 2:
            candidates.append({"image_id": image_id, "image_path": str(path.resolve()), "image_sha256": sha256(path), "questions": selected})
    random.Random(args.seed).shuffle(candidates)
    args.output.mkdir(parents=True, exist_ok=True)
    existing_ids: set[int] = set()
    gallery_ids: set[int] = set()
    candidate_path = args.output / "candidate_manifest.json"
    if candidate_path.exists():
        existing_ids = {item["image_id"] for item in json.loads(candidate_path.read_text())["candidates"]}
    for seed in (42, 43):
        gallery_path = args.output / f"gallery_seed{seed}.json"
        if gallery_path.exists():
            gallery_ids.update(item["image_id"] for item in json.loads(gallery_path.read_text())["images"])
    # Existing galleries are frozen once CKA begins; deterministic expansion
    # chooses only non-gallery candidates, preserving split disjointness.
    selectable = [item for item in candidates if item["image_id"] not in gallery_ids]
    candidate_pool = [item for item in candidates if item["image_id"] in existing_ids]
    candidate_pool.extend(item for item in selectable if item["image_id"] not in existing_ids)
    candidate_pool = candidate_pool[:args.candidate_count]
    atomic_json(candidate_path, {"seed": args.seed, "candidates": candidate_pool})
    # Galleries are image-only and disjoint by construction from the bounded
    # candidate pool; screening later partitions that pool into dev/test pairs.
    if not gallery_ids:
        candidate_ids = {item["image_id"] for item in candidate_pool}
        gallery_pool = [item for item in candidates if item["image_id"] not in candidate_ids]
        ordered_primary = gallery_pool.copy()
        random.Random(42).shuffle(ordered_primary)
        primary = ordered_primary[:256]
        atomic_json(args.output / "gallery_seed42.json", {"seed": 42, "images": primary})
        primary_ids = {item["image_id"] for item in primary}
        ordered_stability = [item for item in gallery_pool if item["image_id"] not in primary_ids]
        random.Random(43).shuffle(ordered_stability)
        atomic_json(args.output / "gallery_seed43.json", {"seed": 43, "images": ordered_stability[:256]})
    print(f"Wrote {len(candidate_pool)} candidates to {args.output}")


if __name__ == "__main__":
    main()

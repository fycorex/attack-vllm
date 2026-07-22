#!/usr/bin/env python3
"""Apply clean-source VLM exclusion and write deterministic dev/test manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from proxy_selector.io import atomic_json
from proxy_selector.vllm_client import ask_vqa


CATEGORIES = ("object", "attribute", "count", "action", "spatial", "relation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/proxy_selector_vqav2/screened_pairs.json"))
    parser.add_argument("--target", choices=("T1", "T2"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/proxy_selector_vqav2"))
    parser.add_argument("--dev-count", type=int, default=6)
    parser.add_argument("--test-count", type=int, default=12)
    args = parser.parse_args()
    provisional = json.loads(args.pairs.read_text())["pairs"]
    cache = args.output / "source_screening_cache"
    valid: list[dict] = []
    for pair in provisional:
        source_answers: dict[str, list[dict]] = {}
        keep = True
        for target in ("T1", "T2"):
            rows = []
            for question in pair["questions"]:
                path = cache / target / f"{pair['pair_id']}_{question['question_id']}.json"
                if target == args.target:
                    response = ask_vqa(args.endpoint, target, Path(pair["source"]["image_path"]), question["question"], cache_path=path)
                elif path.exists():
                    response = json.loads(path.read_text())
                else:
                    keep = False; continue
                rows.append(response)
                if response["normalized_answer"] == question["answer"]: keep = False
            source_answers[target] = rows
        pair["source_screen"] = source_answers
        pair["status"] = "READY" if keep else "SOURCE_REJECTED_OR_PENDING"
        if keep: valid.append(pair)
    # Pair category is fixed before splitting and uses the first target question.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for pair in valid:
        pair["category"] = pair["questions"][0]["category"]
        buckets[pair["category"]].append(pair)
    dev, test, leftovers = [], [], []
    for category in CATEGORIES:
        rows = buckets[category]
        if rows: dev.append(rows.pop(0))
        test.extend(rows[:2]); leftovers.extend(rows[2:])
    # Deterministic fill retains all categories where possible if wording-based
    # categorization cannot provide every quota in the initial screen.
    remaining = [pair for category in CATEGORIES for pair in buckets[category] if pair not in dev and pair not in test]
    dev.extend(remaining[: max(0, args.dev_count - len(dev))])
    remaining = [pair for pair in remaining if pair not in dev]
    test.extend(remaining[: max(0, args.test_count - len(test))])
    atomic_json(args.output / "source_screening_results.json", {"target_completed": args.target, "ready_pairs": valid})
    if all((cache / target / f"{pair['pair_id']}_{question['question_id']}.json").exists() for pair in provisional for target in ("T1", "T2") for question in pair["questions"]):
        atomic_json(args.output / "dev_manifest.json", {"pairs": dev[:args.dev_count]})
        atomic_json(args.output / "test_manifest.json", {"pairs": test[:args.test_count]})
        print(f"Ready: {len(valid)}; dev={len(dev[:args.dev_count])}; test={len(test[:args.test_count])}")
    else:
        print(f"Cached {args.target} source checks; run the other target before manifests can be finalized.")


if __name__ == "__main__":
    main()

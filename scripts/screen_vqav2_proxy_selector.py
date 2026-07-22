#!/usr/bin/env python3
"""Screen a bounded VQAv2 candidate pool with the two frozen target VLMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proxy_selector.io import atomic_json
from proxy_selector.vllm_client import ask_vqa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/proxy_selector_vqav2/candidate_manifest.json"))
    parser.add_argument("--target", choices=("T1", "T2"), required=True, help="Target currently being served.")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible endpoint for --target.")
    parser.add_argument("--output", type=Path, default=Path("data/proxy_selector_vqav2"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["candidates"]
    accepted: list[dict] = []
    records: list[dict] = []
    cache = args.output / "screening_cache"
    for candidate in candidates:
        answers: dict[str, list[dict]] = {}
        valid = True
        for target in ("T1", "T2"):
            target_rows = []
            for question in candidate["questions"]:
                key = cache / target / f"target_{candidate['image_id']}_{question['question_id']}.json"
                if target == args.target:
                    response = ask_vqa(args.endpoint, target, Path(candidate["image_path"]), question["question"], cache_path=key)
                elif key.exists():
                    response = json.loads(key.read_text(encoding="utf-8"))
                else:
                    valid = False
                    continue
                target_rows.append(response)
                valid &= response["normalized_answer"] == question["answer"]
            answers[target] = target_rows
        record = {"image_id": candidate["image_id"], "accepted_target": valid, "answers": answers}
        records.append(record)
        if valid:
            accepted.append(candidate)
    # Pair source selection deliberately comes after target screening.  The first
    # valid disjoint candidate whose VQA answer vocabulary excludes both target
    # answers is retained; clean-source VLM rejection is handled by phase two.
    pairs: list[dict] = []
    used: set[int] = set()
    for target in accepted:
        if target["image_id"] in used: continue
        answers = {row["answer"] for row in target["questions"]}
        source = next((item for item in candidates if item["image_id"] not in used | {target["image_id"]}
                       and answers.isdisjoint({row["answer"] for row in item["questions"]})), None)
        if source is None: continue
        pair = {"pair_id": f"candidate_{len(pairs):03d}", "source": source, "target": target,
                "questions": target["questions"], "status": "TARGET_SCREENED_SOURCE_NEEDS_VLM_CHECK"}
        pairs.append(pair); used.update({source["image_id"], target["image_id"]})
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "screening_results.json", {"records": records})
    atomic_json(args.output / "screened_pairs.json", {"pairs": pairs})
    print(f"Target-correct candidates after {args.target}: {len(accepted)}; provisional pairs: {len(pairs)}")


if __name__ == "__main__":
    main()

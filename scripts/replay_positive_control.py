#!/usr/bin/env python3
"""Replay one-proxy positive-control images with clean/random/target guards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from proxy_selector.io import atomic_json, load_png, save_png
from proxy_selector.vllm_client import ask_vqa
from proxy_selector.vqa_normalization import normalize_answer


def allowed(question: dict) -> bool:
    answer = str(question["answer"]).lower().split()
    text = question["question"].lower()
    return question["category"] in {"object", "attribute", "action"} and 1 <= len(answer) <= 3 and answer != ["yes"] and answer != ["no"] and not any(x in text for x in ("how many", "where", "why"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--proxy", choices=("P2", "P3"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target", choices=("T1", "T2"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_selector_pilot"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epsilon", type=float, default=16 / 255, choices=(8 / 255, 16 / 255))
    args = parser.parse_args()
    manifest = args.manifest or Path(f"data/proxy_selector_vqav2/{args.split}_manifest.json")
    pairs = [pair for pair in json.loads(manifest.read_text())["pairs"] if any(allowed(q) for q in pair["questions"])]
    if args.limit:
        pairs = pairs[:args.limit]
    rows = []
    epsilon_root = args.output / "positive_control"
    if abs(args.epsilon - 16 / 255) > 1e-9:
        epsilon_root = epsilon_root / f"eps{round(args.epsilon * 255)}"
    for offset, pair in enumerate(pairs):
        question = next(item for item in pair["questions"] if allowed(item))
        attack = epsilon_root / args.split / args.proxy / f"seed{args.seed}" / pair["pair_id"] / "adversarial.png"
        if not attack.exists():
            continue
        random_path = epsilon_root / f"random_noise_{round(args.epsilon * 255)}" / f"seed{args.seed}" / f"{pair['pair_id']}.png"
        if not random_path.exists():
            clean = load_png(Path(pair["source"]["image_path"]))
            noise = torch.empty_like(clean).uniform_(-args.epsilon, args.epsilon, generator=torch.Generator().manual_seed(args.seed * 10_000 + offset))
            save_png((clean + noise).clamp(0, 1), random_path)
        cache = args.output / "positive_control" / "vllm" / args.target / args.split / args.proxy / f"seed{args.seed}" / pair["pair_id"]
        images = {"target": Path(pair["target"]["image_path"]), "clean": Path(pair["source"]["image_path"]), "random": random_path, "adversarial": attack}
        answers = {name: ask_vqa(args.endpoint, args.target, path, question["question"], cache_path=cache / f"{name}.json") for name, path in images.items()}
        expected = normalize_answer(question["answer"])
        strict_targeted_success = (
            answers["target"]["normalized_answer"] == expected
            and answers["clean"]["normalized_answer"] != expected
            and answers["random"]["normalized_answer"] != expected
            and answers["adversarial"]["normalized_answer"] == expected
        )
        # A broader transfer diagnostic requested for this pilot: the target
        # answers the natural target correctly; clean and matched random noise
        # agree; the adversarial image induces a different normalized answer.
        # This is intentionally reported separately from targeted TASR.
        answer_change_transfer = (
            answers["target"]["normalized_answer"] == expected
            and answers["clean"]["normalized_answer"] == answers["random"]["normalized_answer"]
            and answers["adversarial"]["normalized_answer"] != answers["clean"]["normalized_answer"]
        )
        raw_answer_change = (
            answers["target"]["normalized_answer"] == expected
            and answers["adversarial"]["normalized_answer"] != answers["clean"]["normalized_answer"]
        )
        rows.append({
            "pair_id": pair["pair_id"], "proxy": args.proxy, "target": args.target, "seed": args.seed,
            "answer": expected, "answers": answers, "strict_targeted_success": strict_targeted_success,
            "answer_change_transfer": answer_change_transfer, "raw_answer_change": raw_answer_change,
        })
    output_path = epsilon_root / "vllm" / args.target / f"{args.split}_{args.proxy}_seed{args.seed}.json"
    atomic_json(output_path, {
        "rows": rows,
        "strict_targeted_hit_count": sum(row["strict_targeted_success"] for row in rows),
        "answer_change_hit_count": sum(row["answer_change_transfer"] for row in rows),
        "raw_answer_change_hit_count": sum(row["raw_answer_change"] for row in rows),
        "clean_valid_count": len(rows),
    })
    print(
        f"{args.target}: {sum(row['strict_targeted_success'] for row in rows)}/{len(rows)} strict targeted; "
        f"{sum(row['answer_change_transfer'] for row in rows)}/{len(rows)} controlled answer-change; "
        f"{sum(row['raw_answer_change'] for row in rows)}/{len(rows)} raw output-change"
    )


if __name__ == "__main__":
    main()

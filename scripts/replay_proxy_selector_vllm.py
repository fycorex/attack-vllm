#!/usr/bin/env python3
"""Replay generated attacks against one sequentially served VLM target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from proxy_selector.io import atomic_json
from proxy_selector.io import load_png, save_png
from proxy_selector.vllm_client import ask_vqa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--target", choices=("T1", "T2"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_selector_pilot"))
    parser.add_argument("--recipe", action="append", help="Replay only this recipe; repeat for several recipes.")
    parser.add_argument("--limit", type=int, help="Replay only the first N manifest pairs.")
    args = parser.parse_args()
    manifest = args.manifest or Path(f"data/proxy_selector_vqav2/{'dev_manifest' if args.split == 'dev' else 'test_manifest'}.json")
    pairs = json.loads(manifest.read_text())["pairs"][: args.limit]
    replay_path = args.output / "vllm" / args.target / f"{args.split}_replay.json"
    existing_rows = json.loads(replay_path.read_text()).get("rows", []) if replay_path.exists() else []
    requested_recipes = tuple(args.recipe) if args.recipe else ("C0", "C1", "C2")
    replacement_keys = {
        (pair["pair_id"], proxy, recipe, args.target)
        for pair in pairs
        for proxy in ("P1", "P2", "P3")
        for recipe in requested_recipes
    }
    results = [
        row for row in existing_rows
        if (row["pair_id"], row["proxy"], row["recipe"], row["target"]) not in replacement_keys
    ]
    for pair in pairs:
        random_path = args.output / "random_noise" / f"{pair['pair_id']}.png"
        if not random_path.exists():
            clean_tensor = load_png(Path(pair["source"]["image_path"]))
            seed = 42 + sum(ord(character) for character in pair["pair_id"])
            noise = torch.empty_like(clean_tensor).uniform_(-8 / 255, 8 / 255, generator=torch.Generator().manual_seed(seed))
            save_png((clean_tensor + noise).clamp(0, 1), random_path)
        controls = {}
        for question in pair["questions"]:
            qid = str(question["question_id"])
            controls[qid] = {
                condition: ask_vqa(args.endpoint, args.target, Path(image), question["question"], cache_path=args.output / "vllm" / args.target / args.split / "controls" / pair["pair_id"] / f"{qid}_{condition}.json")
                for condition, image in {"target": pair["target"]["image_path"], "clean": pair["source"]["image_path"], "random": str(random_path)}.items()
            }
        for proxy in ("P1", "P2", "P3"):
            for recipe in requested_recipes:
                directory = args.output / args.split / proxy / recipe / pair["pair_id"]
                image = directory / "adversarial.png"
                if not image.exists(): continue
                answers = {}
                for question in pair["questions"]:
                    qid = str(question["question_id"])
                    answers[qid] = ask_vqa(args.endpoint, args.target, image, question["question"], cache_path=args.output / "vllm" / args.target / args.split / proxy / recipe / pair["pair_id"] / f"{qid}_adversarial.json")
                results.append({"pair_id": pair["pair_id"], "proxy": proxy, "recipe": recipe, "target": args.target, "questions": pair["questions"], "controls": controls, "adversarial": answers})
    atomic_json(replay_path, {"target": args.target, "rows": results})
    print(f"Replayed {len(results)} attack rows against {args.target}")


if __name__ == "__main__":
    main()

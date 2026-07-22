#!/usr/bin/env python3
"""Run the required first-contact M2 smoke on a screened VQAv2 pair."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from proxy_selector.adapters.qwen35 import Qwen35TokenAdapter
from proxy_selector.attack import momentum_pgd_minimize
from proxy_selector.io import atomic_json, load_png, png_linf, save_png
from proxy_selector.losses import global_contrastive_loss, source_repulsion_loss, symmetric_token_score
from proxy_selector.schemas import AttackRecipe
from proxy_selector.vllm_client import ask_vqa


def tensor(path: str, device: torch.device) -> torch.Tensor:
    return load_png(Path(path), device=device)


def translated(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    pad_x, pad_y = abs(dx), abs(dy)
    padded = F.pad(image, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    return padded[:, :, pad_y + dy : pad_y + dy + image.shape[-2], pad_x + dx : pad_x + dx + image.shape[-1]]


def resized(image: torch.Tensor, scale: float) -> torch.Tensor:
    height, width = image.shape[-2:]
    scaled = F.interpolate(image, size=(max(1, round(height * scale)), max(1, round(width * scale))), mode="bilinear", align_corners=False, antialias=True)
    if scale >= 1:
        top, left = (scaled.shape[-2] - height) // 2, (scaled.shape[-1] - width) // 2
        return scaled[:, :, top : top + height, left : left + width]
    pad_h, pad_w = height - scaled.shape[-2], width - scaled.shape[-1]
    return F.pad(scaled, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode="reflect")


def views(image: torch.Tensor) -> list[torch.Tensor]:
    return [image, *(translated(image, dx, dy) for dx, dy in ((-4, 0), (4, 0), (0, -4), (0, 4))), *(resized(image, s) for s in (.95, .975, 1.025, 1.05))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/proxy_selector_vqav2/screened_pairs.json"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_selector_pilot"))
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--skip-replay", action="store_true", help="Generate artifacts, then exit before target replay.")
    parser.add_argument("--replay-only", action="store_true", help="Replay existing Phase 0 PNGs without loading Qwen.")
    args = parser.parse_args()
    pair = json.loads(args.pairs.read_text(encoding="utf-8"))["pairs"][args.pair_index]
    output = args.output / "smoke" / "phase0_M2"
    output.mkdir(parents=True, exist_ok=True)
    if args.replay_only:
        report_path = args.output / "smoke" / "phase0_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        replay: dict[str, dict] = {}
        for question in pair["questions"]:
            qid = str(question["question_id"])
            replay[qid] = {condition: ask_vqa(args.endpoint, "T2", output / f"{condition if condition != 'adversarial' else 'adversarial'}.png", question["question"], cache_path=output / "replay" / f"{qid}_{condition}.json")
                           for condition in ("target", "clean", "random_noise", "adversarial")}
        # Keep paper-facing names stable even though the file is random_noise.png.
        for answers in replay.values(): answers["random"] = answers.pop("random_noise")
        report["replay"] = replay; report["replay_pending"] = False
        atomic_json(report_path, report)
        print(json.dumps({"M2_replay": "PASS", "questions": len(replay)}, indent=2))
        return
    adapter = Qwen35TokenAdapter("Qwen/Qwen3.5-4B", next(Path("model_cache/models--Qwen--Qwen3.5-4B/snapshots").glob("*")))
    clean, target = tensor(pair["source"]["image_path"], adapter.device), tensor(pair["target"]["image_path"], adapter.device)
    with torch.no_grad():
        clean_rep = adapter.encode_image_tokens(clean, require_grad=False)
        positives = [adapter.encode_image_tokens(view, require_grad=False) for view in views(target)]
        negatives = [adapter.encode_image_tokens(view, require_grad=False) for view in views(clean)]
    positive_global = torch.cat([item.global_features for item in positives])
    negative_global = torch.cat([item.global_features for item in negatives])

    def loss(candidate: torch.Tensor) -> torch.Tensor:
        representation = adapter.encode_image_tokens(candidate, require_grad=True)
        global_loss = global_contrastive_loss(representation.global_features, positive_global, negative_global, .10)
        local = torch.stack([symmetric_token_score(representation.local_tokens, item.local_tokens) for item in positives]).mean().neg()
        source = source_repulsion_loss(representation.global_features, clean_rep.global_features)
        return global_loss + .25 * local + .25 * source

    recipe = AttackRecipe(name="C1", local_weight=.25, source_weight=.25)
    generator = torch.Generator(device=adapter.device).manual_seed(42)
    started = time.monotonic()
    adversarial, losses = momentum_pgd_minimize(clean, loss, recipe, steps=10, generator=generator)
    elapsed = time.monotonic() - started
    png = output / "adversarial.png"
    image_hash = save_png(adversarial, png)
    linf = png_linf(clean, png)
    random = (clean + torch.empty_like(clean).uniform_(-recipe.epsilon, recipe.epsilon, generator=generator)).clamp(0, 1)
    random_png = output / "random_noise.png"; save_png(random, random_png)
    clean_png = output / "clean.png"; save_png(clean, clean_png)
    target_png = output / "target.png"; save_png(target, target_png)
    replay: dict[str, dict] = {}
    if not args.skip_replay:
        for question in pair["questions"]:
            qid = str(question["question_id"])
            replay[qid] = {condition: ask_vqa(args.endpoint, "T2", image, question["question"], cache_path=output / "replay" / f"{qid}_{condition}.json")
                           for condition, image in {"target": target_png, "clean": clean_png, "random": random_png, "adversarial": png}.items()}
    report = {"pair_id": pair["pair_id"], "matrix_id": "M2", "proxy": adapter.model_id, "target": "OpenGVLab/InternVL3_5-2B-HF",
              "recipe": "C1", "steps": 10, "elapsed_seconds": elapsed, "seconds_per_step": elapsed / 10,
              "png_sha256": image_hash, "png_linf": linf, "budget_pass": linf <= 8 / 255 + 1e-7,
              "qwen_token_shape": list(clean_rep.local_tokens.shape), "qwen_tap": clean_rep.metadata, "loss_history": losses,
              "replay": replay, "replay_pending": args.skip_replay}
    atomic_json(args.output / "smoke" / "phase0_report.json", report)
    print(json.dumps({"M2": "PASS" if report["budget_pass"] else "FAIL", "seconds_per_step": report["seconds_per_step"], "png_linf": linf, "token_shape": report["qwen_token_shape"]}, indent=2))
    adapter.unload()


if __name__ == "__main__":
    main()

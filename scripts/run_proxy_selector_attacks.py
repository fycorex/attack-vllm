#!/usr/bin/env python3
"""Generate resumable single-proxy image-token attacks for dev or final data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from proxy_selector.adapters import CLIPHFAdapter, Qwen35TokenAdapter, SigLIP2Adapter
from proxy_selector.attack import debiased_momentum_pgd_minimize, momentum_pgd_minimize
from proxy_selector.io import atomic_json, load_png, png_linf, save_png
from proxy_selector.losses import global_contrastive_loss, source_repulsion_loss, symmetric_token_score
from proxy_selector.schemas import AttackRecipe
from proxy_selector.transforms import eot_transform


def checkpoint(model_id: str) -> Path:
    return next((Path("model_cache") / f"models--{model_id.replace('/', '--')}" / "snapshots").glob("*"))


def proxy(identifier: str):
    if identifier == "P1": return Qwen35TokenAdapter("Qwen/Qwen3.5-4B", checkpoint("Qwen/Qwen3.5-4B"))
    if identifier == "P2": return CLIPHFAdapter(checkpoint("openai/clip-vit-large-patch14"))
    if identifier == "P3": return SigLIP2Adapter(checkpoint("google/siglip2-so400m-patch14-384"))
    raise ValueError(f"Unknown proxy {identifier}")


def translation(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    px, py = abs(dx), abs(dy)
    padded = torch.nn.functional.pad(image, (px, px, py, py), mode="reflect")
    return padded[:, :, py + dy : py + dy + image.shape[-2], px + dx : px + dx + image.shape[-1]]


def make_views(image: torch.Tensor) -> list[torch.Tensor]:
    views = [image, *(translation(image, dx, dy) for dx, dy in ((-4, 0), (4, 0), (0, -4), (0, 4)))]
    height, width = image.shape[-2:]
    for scale in (.95, .975, 1.025, 1.05):
        resized = torch.nn.functional.interpolate(image, size=(round(height * scale), round(width * scale)), mode="bilinear", align_corners=False, antialias=True)
        if scale < 1:
            pad_h, pad_w = height - resized.shape[-2], width - resized.shape[-1]
            resized = torch.nn.functional.pad(resized, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode="reflect")
        else:
            top, left = (resized.shape[-2] - height) // 2, (resized.shape[-1] - width) // 2
            resized = resized[:, :, top:top + height, left:left + width]
        views.append(resized)
    return views


def _tokens(questions: list[dict[str, Any]]) -> set[str]:
    """Metadata-only question tokens used to rank hard negatives."""
    return {
        word.strip(".,?!;:'\"()[]{}").lower()
        for question in questions
        for word in question["question"].split()
        if word.strip(".,?!;:'\"()[]{}")
    }


def select_hard_negative_records(
    pair: dict[str, Any],
    candidates: list[dict[str, Any]],
    excluded_image_ids: set[int],
    *,
    count: int = 8,
) -> list[dict[str, Any]]:
    """Choose deterministic metadata-similar negatives without model embeddings.

    Candidate manifests in this pilot do not include COCO category annotations, so
    the deterministic ranking uses VQAv2 question-token Jaccard similarity and
    breaks all ties by image ID.  This is deliberately executed before proxy
    features are computed.
    """
    target_answers = {question["answer"] for question in pair["questions"]}
    category = pair["category"]
    reference_tokens = _tokens(pair["questions"])
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for candidate in candidates:
        image_id = int(candidate["image_id"])
        if image_id in excluded_image_ids:
            continue
        matching = [
            question for question in candidate["questions"]
            if question["category"] == category and question["answer"] not in target_answers
        ]
        if not matching:
            continue
        candidate_tokens = _tokens(matching)
        union = reference_tokens | candidate_tokens
        score = len(reference_tokens & candidate_tokens) / len(union) if union else 0.0
        ranked.append((score, image_id, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen = [candidate for _, _, candidate in ranked[:count]]
    if len(chosen) != count:
        raise RuntimeError(
            f"{pair['pair_id']} has only {len(chosen)} eligible hard negatives; expected {count}."
        )
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--proxy", required=True, choices=("P1", "P2", "P3"))
    parser.add_argument("--recipe", required=True, help="Name of configs/proxy_selector/attack_<name>.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_selector_pilot"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = args.manifest or Path(f"data/proxy_selector_vqav2/{'dev_manifest' if args.split == 'dev' else 'test_manifest'}.json")
    pairs = json.loads(manifest.read_text())["pairs"][: args.limit]
    candidate_manifest = Path("data/proxy_selector_vqav2/candidate_manifest.json")
    candidates = json.loads(candidate_manifest.read_text())["candidates"]
    all_split_image_ids: set[int] = set()
    for split_name in ("dev_manifest.json", "test_manifest.json"):
        split_path = candidate_manifest.parent / split_name
        if split_path.exists():
            for item in json.loads(split_path.read_text())["pairs"]:
                all_split_image_ids.update((int(item["source"]["image_id"]), int(item["target"]["image_id"])))
    config_path = Path("configs/proxy_selector") / f"attack_{args.recipe}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Unknown recipe {args.recipe!r}: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    recipe = AttackRecipe(name=args.recipe, epsilon=float(config.get("epsilon", 8 / 255)), step_size=float(config["step_size"]), momentum=float(config["momentum"]), temperature=float(config["temperature"]), local_weight=float(config["local_weight"]), source_weight=float(config["source_weight"]), eot_samples=int(config["eot_samples"]))
    qa_weight = float(config.get("qa_weight", 0.0))
    steps = args.steps or int(config.get("steps", 15 if args.split == "dev" else 30))
    model = proxy(args.proxy)
    for index, pair in enumerate(pairs):
        directory = args.output / args.split / args.proxy / args.recipe / pair["pair_id"]
        report_path = directory / "metrics.json"
        if report_path.exists(): continue
        clean = load_png(Path(pair["source"]["image_path"]), device=model.device)
        target = load_png(Path(pair["target"]["image_path"]), device=model.device)
        if qa_weight and args.proxy != "P1":
            raise ValueError("Question-conditioned answer NLL currently requires the Qwen P1 proxy.")
        qa_targets = (
            [model.prepare_answer_target(clean, question["question"], question["answer"]) for question in pair["questions"]]
            if qa_weight else []
        )
        with torch.no_grad():
            clean_rep = model.encode_image_tokens(clean, require_grad=False)
            positives = [model.encode_image_tokens(view, require_grad=False) for view in make_views(target)]
            negatives = [model.encode_image_tokens(view, require_grad=False) for view in make_views(clean)]
            hard_negative_records = select_hard_negative_records(
                pair,
                candidates,
                all_split_image_ids | {int(pair["source"]["image_id"]), int(pair["target"]["image_id"])},
                count=int(config.get("hard_negative_count", 8)),
            )
            hard_negative_paths = [item["image_path"] for item in hard_negative_records]
            negatives.extend(
                model.encode_image_tokens(load_png(Path(path), device=model.device), require_grad=False)
                for path in hard_negative_paths
            )
        positive_global = torch.cat([item.global_features for item in positives])
        negative_global = torch.cat([item.global_features for item in negatives])
        eot_generator = torch.Generator(device=model.device).manual_seed(9_000 + index)
        def loss(candidate: torch.Tensor) -> torch.Tensor:
            totals = []
            for _ in range(recipe.eot_samples):
                transformed = candidate if recipe.eot_samples == 1 else eot_transform(candidate, translation_pixels=int(config["translation_pixels"]), resize_min=float(config["resize_min"]), resize_max=float(config["resize_max"]), generator=eot_generator)
                output = model.encode_image_tokens(transformed, require_grad=True)
                total = global_contrastive_loss(output.global_features, positive_global, negative_global, recipe.temperature)
                if recipe.local_weight:
                    local = torch.stack([symmetric_token_score(output.local_tokens, item.local_tokens) for item in positives]).mean().neg()
                    total = total + recipe.local_weight * local
                if recipe.source_weight:
                    total = total + recipe.source_weight * source_repulsion_loss(output.global_features, clean_rep.global_features)
                if qa_targets:
                    answer_nll = torch.stack([model.answer_nll(transformed, item) for item in qa_targets]).mean()
                    total = total + qa_weight * answer_nll
                totals.append(total)
            return torch.stack(totals).mean()
        start = time.monotonic()
        restart_histories: list[list[float]] = []
        restart_projection_coefficients: list[list[float]] = []
        best_generated: torch.Tensor | None = None
        best_terminal_loss = float("inf")
        restart_count = int(config.get("restarts", 1))
        for restart in range(restart_count):
            generator = torch.Generator(device=model.device).manual_seed(42 + index * 10_000 + restart)
            if bool(config.get("debias_enabled", False)):
                generated, history, coefficients = debiased_momentum_pgd_minimize(
                    clean,
                    loss,
                    recipe,
                    dataset_rgb_mean=torch.tensor(config["dataset_rgb_mean"]),
                    reference_noise_std=float(config["reference_noise_std"]),
                    beta=float(config["debias_beta"]),
                    steps=steps,
                    generator=generator,
                )
            else:
                generated, history = momentum_pgd_minimize(clean, loss, recipe, steps=steps, generator=generator)
                coefficients = []
            restart_histories.append(history)
            restart_projection_coefficients.append(coefficients)
            if history[-1] < best_terminal_loss:
                best_generated, best_terminal_loss = generated, history[-1]
        assert best_generated is not None
        generated = best_generated
        adversarial_path = directory / "adversarial.png"
        digest = save_png(generated, adversarial_path)
        save_png(clean, directory / "clean.png"); save_png(target, directory / "target.png")
        linf = png_linf(clean, adversarial_path)
        atomic_json(report_path, {"pair_id": pair["pair_id"], "proxy": args.proxy, "recipe": args.recipe, "steps": steps, "restarts": restart_count, "qa_weight": qa_weight, "debias_enabled": bool(config.get("debias_enabled", False)), "elapsed_seconds": time.monotonic() - start, "png_linf": linf, "budget_pass": linf <= recipe.epsilon + 1e-7, "sha256": digest, "positive_reference_count": len(positives), "negative_reference_count": len(negatives), "hard_negative_image_ids": [item["image_id"] for item in hard_negative_records], "best_terminal_loss": best_terminal_loss, "restart_loss_histories": restart_histories, "restart_projection_coefficients": restart_projection_coefficients})
    model.unload()


if __name__ == "__main__":
    main()

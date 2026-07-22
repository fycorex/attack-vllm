#!/usr/bin/env python3
"""Run the single-proxy MaxStrengthHierarchicalDirection positive control.

This is a paper-inspired combination: same-checkpoint text direction
(UnivIntruder), 13 reference/multi-depth alignment (SGHA), optional reversible
vision-path pruning (RaPA).  It does not query target models while optimizing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from proxy_selector.adapters import CLIPHFAdapter, Qwen35TokenAdapter, SigLIP2Adapter
from proxy_selector.attack import momentum_pgd_minimize_eot
from proxy_selector.io import atomic_json, load_png, png_linf, save_png
from proxy_selector.losses import endpoint_loss, semantic_direction_loss, source_repulsion_loss
from proxy_selector.positive_control import (
    direction_target,
    hierarchical_loss,
    layer_anchor_cache,
    normalized_mean,
    rpa_visual_output_pruning,
)
from proxy_selector.schemas import AttackRecipe
from proxy_selector.transforms import eot_transform
from run_proxy_selector_attacks import make_views, select_hard_negative_records
from proxy_selector.vqa_normalization import normalize_answer


def checkpoint(model_id: str) -> Path:
    return next((Path("model_cache") / f"models--{model_id.replace('/', '--')}" / "snapshots").glob("*"))


def build_proxy(identifier: str):
    if identifier == "P1":
        return Qwen35TokenAdapter("Qwen/Qwen3.5-4B", checkpoint("Qwen/Qwen3.5-4B"))
    if identifier == "P2":
        return CLIPHFAdapter(checkpoint("openai/clip-vit-large-patch14"))
    if identifier == "P3":
        return SigLIP2Adapter(checkpoint("google/siglip2-so400m-patch14-384"))
    raise ValueError(f"Unknown proxy {identifier}")


def layers_and_semantic(model: Any, image: torch.Tensor, *, require_grad: bool):
    """Use the one-pass hierarchy API where a proxy provides it."""
    if hasattr(model, "encode_hierarchical_semantic"):
        return model.encode_hierarchical_semantic(image, require_grad=require_grad)
    return model.encode_image_layers(image, require_grad=require_grad), model.semantic_global(image, require_grad=require_grad)


def allowed_question(question: dict[str, Any]) -> bool:
    answer = str(question["answer"]).strip().lower()
    words = answer.split()
    text = str(question["question"]).lower()
    return (
        question["category"] in {"object", "attribute", "action"}
        and 1 <= len(words) <= 3
        and answer not in {"yes", "no", "unknown", "none", "each other", "same", "different"}
        and not any(marker in text for marker in ("how many", "where", "why", "is there", "are there"))
    )


def additional_same_answer_anchors(
    pair: dict[str, Any], candidates: list[dict[str, Any]], excluded: set[int]
) -> list[dict[str, Any]]:
    question = next(item for item in pair["questions"] if allowed_question(item))
    matches = []
    for candidate in candidates:
        if int(candidate["image_id"]) in excluded:
            continue
        if any(
            item["answer"] == question["answer"] and item["category"] == question["category"]
            for item in candidate["questions"]
        ):
            matches.append(candidate)
    matches.sort(key=lambda item: int(item["image_id"]))
    if len(matches) < 4:
        raise RuntimeError(f"{pair['pair_id']} has {len(matches)} same-answer/category anchors; need four.")
    return matches[:4]


def fallback_official_answer_anchors(pair: dict[str, Any], excluded: set[int]) -> list[dict[str, Any]]:
    """Find four fixed same-answer anchors from official VQAv2, not embeddings.

    The deliberately small 36-image screening pool cannot reliably contain four
    images with a particular answer (e.g. ``zebra``).  This metadata-only scan
    uses the local official annotations; it is never conditioned on target
    outputs, surrogate embeddings, or attack results.
    """
    root = Path("data/vqav2_official")
    annotation_path = root / "v2_mscoco_val2014_annotations.json"
    question_path = root / "v2_OpenEnded_mscoco_val2014_questions.json"
    image_root = root / "val2014"
    if not annotation_path.exists() or not question_path.exists():
        raise RuntimeError("Need local official VQAv2 JSON to expand the four same-answer anchor references.")
    desired = next(item for item in pair["questions"] if allowed_question(item))["answer"]
    questions = {item["question_id"]: item for item in json.loads(question_path.read_text())["questions"]}
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for annotation in json.loads(annotation_path.read_text())["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in excluded or image_id in seen:
            continue
        answers = [normalize_answer(item["answer"]) for item in annotation["answers"]]
        if answers.count(desired) < 3:
            continue
        question = questions.get(annotation["question_id"], {})
        text = str(question.get("question", "")).lower()
        if any(marker in text for marker in ("how many", "where", "why")):
            continue
        image_path = image_root / f"COCO_val2014_{image_id:012d}.jpg"
        if image_path.exists():
            records.append({"image_id": image_id, "image_path": str(image_path.resolve())})
            seen.add(image_id)
            if len(records) == 4:
                return records
    raise RuntimeError(f"Could not find four VQAv2 same-answer anchors for {desired!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--proxy", choices=("P1", "P2", "P3"), default="P2")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--candidates", type=Path, help="Matching screened candidate manifest for hard negatives/anchors.")
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_selector_pilot"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pair-id", action="append", help="Attack only this pair ID; repeat for a small batch shard.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rpa-ratio", type=float, default=0.0, choices=(0.0, 0.02, 0.05))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--eot", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=16 / 255, choices=(8 / 255, 16 / 255))
    args = parser.parse_args()

    manifest_path = args.manifest or Path(f"data/proxy_selector_vqav2/{args.split}_manifest.json")
    pairs = [pair for pair in json.loads(manifest_path.read_text())["pairs"] if any(allowed_question(q) for q in pair["questions"])]
    if args.limit:
        pairs = pairs[: args.limit]
    if args.pair_id:
        requested = set(args.pair_id)
        pairs = [pair for pair in pairs if pair["pair_id"] in requested]
    if not pairs:
        raise RuntimeError("No positive-control eligible pairs in manifest.")
    candidate_path = args.candidates or manifest_path.parent / "candidate_manifest.json"
    candidates = json.loads(candidate_path.read_text())["candidates"]
    model = build_proxy(args.proxy)
    recipe = AttackRecipe(
        name="MaxStrengthHierarchicalDirection",
        epsilon=args.epsilon,
        step_size=1 / 255,
        momentum=1.0,
        temperature=0.1,
        local_weight=0.35,
        source_weight=0.15,
        eot_samples=args.eot,
    )
    for pair_index, pair in enumerate(pairs):
        question = next(item for item in pair["questions"] if allowed_question(item))
        epsilon_root = args.output / "positive_control"
        if abs(args.epsilon - 16 / 255) > 1e-9:
            epsilon_root = epsilon_root / f"eps{round(args.epsilon * 255)}"
        directory = epsilon_root / args.split / args.proxy / f"seed{args.seed}" / pair["pair_id"]
        report_path = directory / "metrics.json"
        if report_path.exists():
            continue
        clean = load_png(Path(pair["source"]["image_path"]), device=model.device)
        target = load_png(Path(pair["target"]["image_path"]), device=model.device)
        excluded = {int(pair["source"]["image_id"]), int(pair["target"]["image_id"])}
        try:
            extra_records = additional_same_answer_anchors(pair, candidates, excluded)
            extra_source = "candidate_manifest"
        except RuntimeError:
            # This is expected for a 36-image candidate pool; retain the same
            # answer constraint through the official-data metadata fallback.
            extra_records = fallback_official_answer_anchors(pair, excluded)
            extra_source = "official_vqav2_metadata"
        hard_records = select_hard_negative_records(pair, candidates, excluded, count=8)
        anchors = make_views(target) + [load_png(Path(item["image_path"]), device=model.device) for item in extra_records]
        if len(anchors) != 13:
            raise AssertionError("Positive control must cache exactly 13 target references.")
        with torch.no_grad():
            clean_layers, clean_semantic = layers_and_semantic(model, clean, require_grad=False)
            target_centroids, target_locals = layer_anchor_cache(model, anchors)
            image_positive = normalized_mean([model.semantic_global(anchor, require_grad=False) for anchor in anchors])
            templates = [
                f"a photo of {question['answer']}",
                f"an image containing {question['answer']}",
                f"the visual answer is {question['answer']}",
                f"Question: {question['question']} Answer: {question['answer']}",
            ]
            text_positive = F.normalize(model.encode_text_concepts(templates).mean(dim=0, keepdim=True), dim=-1)
            positive_semantic = F.normalize(0.5 * image_positive + 0.5 * text_positive, dim=-1)
            negative_images = [clean] + [load_png(Path(item["image_path"]), device=model.device) for item in hard_records]
            negative_semantic = normalized_mean([model.semantic_global(image, require_grad=False) for image in negative_images])
            target_direction = direction_target(positive_semantic, negative_semantic)
        eot_generator = torch.Generator(device=model.device).manual_seed(args.seed * 100_000 + pair_index)
        rpa_generator = torch.Generator(device=model.device).manual_seed(args.seed * 1_000_000 + pair_index)

        def branch_losses(candidate: torch.Tensor):
            for _ in range(args.eot):
                transformed = eot_transform(candidate, translation_pixels=8, resize_min=0.90, resize_max=1.10, generator=eot_generator)
                if args.rpa_ratio:
                    with rpa_visual_output_pruning(model.model, args.rpa_ratio, rpa_generator):
                        layers, semantic = layers_and_semantic(model, transformed, require_grad=True)
                        global_loss, local_loss = hierarchical_loss(layers, target_centroids, target_locals)
                        yield (
                            semantic_direction_loss(semantic, clean_semantic, target_direction)
                            + 0.30 * endpoint_loss(semantic, positive_semantic)
                            + 0.50 * global_loss + 0.35 * local_loss
                            + 0.15 * source_repulsion_loss(semantic, clean_semantic)
                        )
                else:
                    layers, semantic = layers_and_semantic(model, transformed, require_grad=True)
                    global_loss, local_loss = hierarchical_loss(layers, target_centroids, target_locals)
                    yield (
                        semantic_direction_loss(semantic, clean_semantic, target_direction)
                        + 0.30 * endpoint_loss(semantic, positive_semantic)
                        + 0.50 * global_loss + 0.35 * local_loss
                        + 0.15 * source_repulsion_loss(semantic, clean_semantic)
                    )

        started = time.monotonic()
        best: tuple[float, torch.Tensor, list[float]] | None = None
        histories: list[list[float]] = []
        for restart in range(args.restarts):
            generated, history = momentum_pgd_minimize_eot(
                clean, branch_losses, recipe, steps=args.steps,
                generator=torch.Generator(device=model.device).manual_seed(args.seed + pair_index * 1000 + restart),
            )
            histories.append(history)
            temporary = directory / f"restart_{restart}.png"
            save_png(generated, temporary)
            reloaded = load_png(temporary, device=model.device)
            with torch.no_grad():
                post_png_loss = sum(float(item.detach().cpu()) for item in branch_losses(reloaded)) / args.eot
            if best is None or post_png_loss < best[0]:
                best = (post_png_loss, generated, history)
        assert best is not None
        selected_loss, adversarial, _ = best
        digest = save_png(adversarial, directory / "adversarial.png")
        save_png(clean, directory / "clean.png")
        save_png(target, directory / "target.png")
        with torch.no_grad():
            before_direction = float(F.cosine_similarity(F.normalize(clean_semantic - clean_semantic + 1e-12, dim=-1), target_direction, dim=-1).mean())
            adv_semantic = model.semantic_global(adversarial, require_grad=False)
            after_direction = float(F.cosine_similarity(F.normalize(adv_semantic - clean_semantic, dim=-1), target_direction, dim=-1).mean())
        atomic_json(report_path, {
            "method": "MaxStrengthHierarchicalDirection (paper-inspired combined positive control)",
            "pair_id": pair["pair_id"], "question": question, "proxy": args.proxy, "seed": args.seed,
            "epsilon": recipe.epsilon, "steps": args.steps, "restarts": args.restarts, "eot": args.eot,
            "rpa_ratio": args.rpa_ratio, "positive_anchor_count": len(anchors), "hard_negative_count": len(hard_records),
            "additional_anchor_image_ids": [item["image_id"] for item in extra_records],
            "additional_anchor_source": extra_source,
            "hard_negative_image_ids": [item["image_id"] for item in hard_records],
            "elapsed_seconds": time.monotonic() - started, "png_linf": png_linf(clean, directory / "adversarial.png"),
            "post_png_proxy_loss": selected_loss, "restart_loss_histories": histories,
            "direction_alignment_before": before_direction, "direction_alignment_after": after_direction,
            "sha256": digest,
        })
    model.unload()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a VEAttack-style single-proxy visual-token gradient sanity check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from proxy_selector.adapters import CLIPHFAdapter
from proxy_selector.attack import momentum_pgd_minimize
from proxy_selector.io import atomic_json, load_png, png_linf, save_png
from proxy_selector.positive_control import ve_loss_from_outputs
from proxy_selector.schemas import AttackRecipe
from proxy_selector.transforms import eot_transform


def checkpoint() -> Path:
    return next((Path("model_cache/models--openai--clip-vit-large-patch14/snapshots")).glob("*"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/proxy_selector_vqav2/dev_manifest.json"))
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_selector_pilot"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pair = json.loads(args.manifest.read_text())["pairs"][args.pair_index]
    model = CLIPHFAdapter(checkpoint())
    clean = load_png(Path(pair["source"]["image_path"]), device=model.device)
    with torch.no_grad():
        clean_layers = {name: value for name, value in model.encode_image_layers(clean, require_grad=False).items() if name in {"50", "75", "final"}}
    recipe = AttackRecipe("VEAttackSanity", 16 / 255, 1 / 255, 1.0, 0.1, 0.0, 0.0, 4)
    eot_generator = torch.Generator(device=model.device).manual_seed(args.seed)

    def loss(candidate: torch.Tensor) -> torch.Tensor:
        terms = []
        for _ in range(4):
            transformed = eot_transform(candidate, translation_pixels=8, resize_min=.90, resize_max=1.10, generator=eot_generator)
            candidate_layers = {name: value for name, value in model.encode_image_layers(transformed, require_grad=True).items() if name in clean_layers}
            terms.append(ve_loss_from_outputs(candidate_layers, clean_layers))
        return torch.stack(terms).mean()

    adversarial, history = momentum_pgd_minimize(
        clean, loss, recipe, steps=100, generator=torch.Generator(device=model.device).manual_seed(args.seed)
    )
    directory = args.output / "positive_control" / "veattack" / f"seed{args.seed}" / pair["pair_id"]
    save_png(clean, directory / "clean.png")
    digest = save_png(adversarial, directory / "adversarial.png")
    atomic_json(directory / "metrics.json", {
        "method": "VEAttack-style image-token sanity (not targeted)", "proxy": "P2", "pair_id": pair["pair_id"],
        "epsilon": recipe.epsilon, "steps": 100, "momentum": 1.0, "eot": 4,
        "layers": ["50", "75", "final"], "loss_history": history,
        "loss_change": history[0] - history[-1], "png_linf": png_linf(clean, directory / "adversarial.png"), "sha256": digest,
    })
    model.unload()


if __name__ == "__main__":
    main()

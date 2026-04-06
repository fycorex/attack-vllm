from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .augmentations import AttackAugmentationPipeline
from .caption_victim import CaptionVictim
from .config import AttackConfig, load_config
from .data import AttackItem, load_image_tensor, load_manifest, save_tensor_image, tensor_to_pil_image
from .eval import evaluate_proxy, summarize_results, write_item_csv
from .losses import visual_contrastive_loss
from .surrogates import create_surrogate, unload_surrogate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CaptionAttackRunner:
    def __init__(self, config: AttackConfig):
        self.config = config
        self.device = config.runtime.device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(config.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir = Path(config.paths.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        set_seed(config.runtime.seed)
        self.surrogates = []
        self.caption_victim = (
            CaptionVictim(config.evaluation.caption_victim, cache_dir=self.model_cache_dir / "huggingface")
            if config.evaluation.caption_victim.enabled
            else None
        )

    def load_surrogates(self) -> None:
        load_device = self.device if not self.config.runtime.sequential_surrogates else "cpu"
        surrogate_cache_dir = self.model_cache_dir / "open_clip"
        self.surrogates = [
            create_surrogate(spec, load_device, cache_dir=surrogate_cache_dir)
            for spec in self.config.surrogates
            if spec.enabled
        ]

    def _move_surrogate(self, surrogate, device: str) -> None:
        surrogate.to(device)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resize_for_surrogate(self, images: torch.Tensor, input_size: int) -> torch.Tensor:
        if images.shape[-1] == input_size and images.shape[-2] == input_size:
            return images
        return F.interpolate(images, size=(input_size, input_size), mode="bilinear", align_corners=False, antialias=True)

    def _precompute_example_embeddings(self, item: AttackItem) -> dict:
        example_cache = {}
        for surrogate in self.surrogates:
            if self.config.runtime.sequential_surrogates:
                self._move_surrogate(surrogate, self.device)
            size = surrogate.config.input_size
            pos_batch = torch.stack([load_image_tensor(path, size) for path in item.positive_image_paths], dim=0).to(self.device)
            neg_batch = torch.stack([load_image_tensor(path, size) for path in item.negative_image_paths], dim=0).to(self.device)
            with torch.no_grad():
                pos_emb = surrogate.encode_image(pos_batch).cpu()
                neg_emb = surrogate.encode_image(neg_batch).cpu()
            example_cache[surrogate.name] = {
                "positive_embeddings": pos_emb,
                "negative_embeddings": neg_emb,
            }
            if self.config.runtime.sequential_surrogates:
                unload_surrogate(surrogate)
        return example_cache

    def _ensemble_proxy_eval(self, clean: torch.Tensor, adv: torch.Tensor, example_cache: dict) -> dict:
        per_surrogate = {}
        clean_margins = []
        adv_margins = []

        for surrogate in self.surrogates:
            if self.config.runtime.sequential_surrogates:
                self._move_surrogate(surrogate, self.device)
            size = surrogate.config.input_size
            clean_input = self._resize_for_surrogate(clean, size)
            adv_input = self._resize_for_surrogate(adv, size)
            positive_embeddings = example_cache[surrogate.name]["positive_embeddings"].to(self.device)
            negative_embeddings = example_cache[surrogate.name]["negative_embeddings"].to(self.device)
            with torch.no_grad():
                clean_emb = surrogate.encode_image(clean_input)
                adv_emb = surrogate.encode_image(adv_input)
            result = evaluate_proxy(
                clean_emb,
                adv_emb,
                positive_embeddings,
                negative_embeddings,
                top_k=self.config.attack.top_k,
                success_margin_threshold=self.config.evaluation.success_margin_threshold,
            )
            per_surrogate[surrogate.name] = result
            clean_margins.append(result["clean_margin"])
            adv_margins.append(result["adversarial_margin"])
            if self.config.runtime.sequential_surrogates:
                unload_surrogate(surrogate)

        clean_margin = float(sum(clean_margins) / len(clean_margins))
        adversarial_margin = float(sum(adv_margins) / len(adv_margins))
        return {
            "clean_margin": clean_margin,
            "adversarial_margin": adversarial_margin,
            "margin_gain": adversarial_margin - clean_margin,
            "proxy_success": adversarial_margin > self.config.evaluation.success_margin_threshold and adversarial_margin > clean_margin,
            "per_surrogate": per_surrogate,
        }

    def _caption_eval(self, clean: torch.Tensor, adv: torch.Tensor, item: AttackItem) -> dict | None:
        if self.caption_victim is None:
            return None
        clean_image = tensor_to_pil_image(clean[0])
        adv_image = tensor_to_pil_image(adv[0])
        return self.caption_victim.evaluate(clean_image, adv_image, item)

    def attack_item(self, item: AttackItem) -> dict:
        example_cache = self._precompute_example_embeddings(item)
        base_size = next(spec.input_size for spec in self.config.surrogates if spec.enabled)
        clean = load_image_tensor(item.image_path, base_size).unsqueeze(0).to(self.device)
        delta = torch.zeros_like(clean, requires_grad=True)
        delta_ema = torch.zeros_like(clean)
        pipeline = AttackAugmentationPipeline(self.config.attack, base_size)
        history = []
        patch_drop_rate = self.config.attack.patch_drop_rate if self.config.attack.enable_patch_drop else 0.0
        drop_path_max_rate = self.config.attack.drop_path_max_rate if self.config.attack.enable_drop_path else 0.0

        for step in tqdm(range(self.config.attack.steps), desc=f"attack:{item.item_id}", leave=False):
            if delta.grad is not None:
                delta.grad.zero_()

            total_loss_value = 0.0
            step_metrics = {}

            for surrogate in self.surrogates:
                bounded_delta = delta.clamp(-self.config.attack.epsilon, self.config.attack.epsilon)
                adv = (clean + bounded_delta).clamp(0.0, 1.0)
                if self.config.runtime.sequential_surrogates:
                    self._move_surrogate(surrogate, self.device)
                surrogate_input = self._resize_for_surrogate(adv, surrogate.config.input_size)
                surrogate_input = pipeline(surrogate_input, self.config.attack.epsilon)
                positive_embeddings = example_cache[surrogate.name]["positive_embeddings"].to(self.device)
                negative_embeddings = example_cache[surrogate.name]["negative_embeddings"].to(self.device)
                embeddings = surrogate.encode_image(
                    surrogate_input,
                    patch_drop_rate=patch_drop_rate,
                    drop_path_max_rate=drop_path_max_rate,
                )
                loss, metrics = visual_contrastive_loss(
                    embeddings,
                    positive_embeddings,
                    negative_embeddings,
                    temperature=self.config.attack.temperature,
                    top_k=self.config.attack.top_k,
                )
                loss.backward()
                total_loss_value += float(loss.detach().cpu())
                step_metrics[surrogate.name] = metrics
                if self.config.runtime.sequential_surrogates:
                    unload_surrogate(surrogate)

            if delta.grad is None:
                raise RuntimeError("Attack step produced no gradient for delta.")

            with torch.no_grad():
                delta.sub_(self.config.attack.step_size * delta.grad.sign())
                delta.clamp_(-self.config.attack.epsilon, self.config.attack.epsilon)
                delta.copy_((clean + delta).clamp(0.0, 1.0) - clean)
                if self.config.attack.enable_perturbation_ema:
                    delta_ema.mul_(self.config.attack.perturbation_ema_decay).add_(
                        delta.detach() * (1.0 - self.config.attack.perturbation_ema_decay)
                    )
                else:
                    delta_ema.copy_(delta.detach())

            history.append(
                {
                    "step": step,
                    "loss": total_loss_value,
                    "surrogates": step_metrics,
                }
            )

        final_delta = delta_ema.clamp(-self.config.attack.epsilon, self.config.attack.epsilon)
        final_adv = (clean + final_delta).clamp(0.0, 1.0)
        proxy_eval = self._ensemble_proxy_eval(clean, final_adv, example_cache)
        caption_eval = self._caption_eval(clean, final_adv, item)

        item_dir = self.output_dir / item.item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        save_tensor_image(clean[0], item_dir / "clean.png")
        save_tensor_image(final_adv[0], item_dir / "adversarial.png")
        delta_vis = ((final_delta[0] / (2.0 * self.config.attack.epsilon)) + 0.5).clamp(0.0, 1.0)
        save_tensor_image(delta_vis, item_dir / "delta_vis.png")

        result = {
            "item_id": item.item_id,
            "image_path": str(item.image_path),
            "source_label": item.source_label,
            "target_label": item.target_label,
            "source_keywords": item.source_keywords,
            "target_keywords": item.target_keywords,
            "proxy_eval": proxy_eval,
            "caption_eval": caption_eval,
            "history": history,
        }
        (item_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def run(self) -> dict:
        manifest = load_manifest(self.config.paths.manifest)
        self.load_surrogates()
        items = manifest.items if self.config.runtime.attack_limit is None else manifest.items[: self.config.runtime.attack_limit]
        results = [self.attack_item(item) for item in items]
        summary_metrics = summarize_results(results)
        summary = {
            "experiment_name": self.config.experiment_name,
            "dataset_name": manifest.dataset_name,
            **summary_metrics,
            "items": results,
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_item_csv(results, self.output_dir / "items.csv")
        if self.caption_victim is not None:
            self.caption_victim.unload()
        return summary


def run_attack(config: AttackConfig) -> dict:
    runner = CaptionAttackRunner(config)
    return runner.run()


def run_attack_from_config(config_path: str) -> dict:
    config = load_config(config_path)
    return run_attack(config)

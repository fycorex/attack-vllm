from __future__ import annotations

from pathlib import Path
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from augmentations import AttackAugmentationPipeline, effective_sigma, sample_noise
from caption_victim import CaptionVictim
from config import AttackConfig, config_to_dict, enabled_surrogate_names, load_config
from data import AttackItem, load_image_tensor, load_manifest, normalize_image_size, save_tensor_image, tensor_to_pil_image
from eval import evaluate_proxy, summarize_results, write_item_csv
from losses import (batched_relative_proxy_loss, batched_visual_contrastive_loss,
                    relative_proxy_loss, visual_contrastive_loss)
from ocr_victim import OCRVictim
from gpt_victim import GPTVictim
from ollama_victim import OllamaVictim
from qwen_vl_victim import QwenVLVictim
from surrogates import create_surrogate, unload_surrogate
from surrogate_composition import gradient_diagnostics
from vqa_victim import VQAVictim


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _accumulate_step_metrics(target: dict[str, float], metrics: dict[str, float], weight: float = 1.0) -> None:
    for key, value in metrics.items():
        target[key] = target.get(key, 0.0) + (float(value) * weight)


def _average_step_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    if count <= 1:
        return total
    return {key: value / float(count) for key, value in total.items()}


def _should_collect_step_metrics(step: int, total_steps: int, metrics_interval: int) -> bool:
    if metrics_interval <= 0:
        return False
    return step % metrics_interval == 0 or step == total_steps - 1


class CaptionAttackRunner:
    def __init__(self, config: AttackConfig):
        self.config = config
        self.device = config.runtime.device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(config.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir = Path(config.paths.model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        set_seed(config.runtime.seed)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = config.runtime.enable_tf32
            torch.backends.cudnn.allow_tf32 = config.runtime.enable_tf32
            torch.backends.cudnn.benchmark = config.runtime.cudnn_benchmark
        self.surrogates = []
        self.caption_victim = (
            CaptionVictim(config.evaluation.caption_victim, cache_dir=self.model_cache_dir / "huggingface")
            if config.evaluation.caption_victim.enabled
            else None
        )
        self.vqa_victim = (
            VQAVictim(config.evaluation.vqa_victim, cache_dir=self.model_cache_dir / "huggingface")
            if config.evaluation.vqa_victim.enabled
            else None
        )
        self.ocr_victim = (
            OCRVictim(config.evaluation.ocr_victim, cache_dir=self.model_cache_dir / "huggingface")
            if config.evaluation.ocr_victim.enabled
            else None
        )
        self.gpt_victim = GPTVictim(config.evaluation.gpt_victim) if config.evaluation.gpt_victim.enabled else None
        self.ollama_victim = OllamaVictim(config.evaluation.ollama_victim) if config.evaluation.ollama_victim.enabled else None
        self.qwen_vl_victim = (
            QwenVLVictim(config.evaluation.qwen_vl_victim, cache_dir=self.model_cache_dir / "huggingface")
            if config.evaluation.qwen_vl_victim.enabled
            else None
        )

    def _write_effective_config(self) -> None:
        payload = config_to_dict(self.config)
        (self.output_dir / "effective_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _safe_eval(self, label: str, success_key: str, fn) -> dict | None:
        try:
            return fn()
        except Exception as exc:
            return {
                success_key: False,
                "evaluation_failed": True,
                "error": str(exc),
                "evaluation_label": label,
            }

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

    def _base_attack_size(self) -> int | str | list[int] | tuple[int, int]:
        if self.config.attack.image_size is not None:
            size = normalize_image_size(self.config.attack.image_size)
            return self.config.attack.image_size if size is None else size
        for spec in self.config.surrogates:
            if spec.enabled:
                return int(spec.input_size)
        raise RuntimeError("No enabled surrogates are configured.")

    def _augment_for_surrogate(
        self,
        pipeline: AttackAugmentationPipeline,
        surrogate_input: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if batch_size <= 1:
            return pipeline(surrogate_input, self.config.attack.epsilon)
        augmented = [
            pipeline(surrogate_input, self.config.attack.epsilon)
            for _ in range(batch_size)
        ]
        return torch.cat(augmented, dim=0)

    def _precompute_example_embeddings(self, item: AttackItem) -> dict:
        example_cache = {}
        for surrogate in self.surrogates:
            if self.config.runtime.sequential_surrogates:
                self._move_surrogate(surrogate, self.device)
            size = surrogate.config.input_size
            pos_batch = torch.stack([load_image_tensor(path, size) for path in item.positive_image_paths], dim=0).to(self.device)
            neg_batch = torch.stack([load_image_tensor(path, size) for path in item.negative_image_paths], dim=0).to(self.device)
            with torch.no_grad():
                pos_emb = surrogate.encode_image(pos_batch)
                neg_emb = surrogate.encode_image(neg_batch)
                clean_input = load_image_tensor(item.image_path, size).unsqueeze(0).to(self.device)
                clean_emb = surrogate.encode_image(clean_input)
            if self.config.runtime.sequential_surrogates:
                pos_emb = pos_emb.cpu()
                neg_emb = neg_emb.cpu()
                clean_emb = clean_emb.cpu()
            example_cache[surrogate.name] = {
                "positive_embeddings": pos_emb,
                "negative_embeddings": neg_emb,
                "clean_embedding": clean_emb,
            }
            if self.config.runtime.sequential_surrogates:
                unload_surrogate(surrogate)
        return example_cache

    def _precompute_batch_example_embeddings(self, items: list[AttackItem], encode_batch_size: int = 64) -> list[dict]:
        """Precompute item references by model-wide batches instead of 100-image microbatches.

        Each item still receives exactly its own positive, negative, and clean
        embeddings.  Only independent image encodes are packed together.
        """
        caches = [{} for _ in items]
        for surrogate in self.surrogates:
            size = surrogate.config.input_size
            paths = []
            seen = set()
            for item in items:
                for path in [*item.positive_image_paths, *item.negative_image_paths, item.image_path]:
                    key = str(path)
                    if key not in seen:
                        seen.add(key)
                        paths.append(path)
            fingerprint = hashlib.sha256(json.dumps({
                "model": surrogate.name,
                "input_size": size,
                "paths": [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths],
            }, sort_keys=True).encode()).hexdigest()
            cache_dir = self.model_cache_dir / "reference_embeddings"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{fingerprint}.pt"
            if cache_path.is_file():
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                embeddings = list(payload["embeddings"])
            else:
                path_batches = [paths[offset:offset + encode_batch_size]
                                for offset in range(0, len(paths), encode_batch_size)]

                def load_path_batch(path_batch: list[Path]) -> torch.Tensor:
                    with ThreadPoolExecutor(max_workers=8) as loaders:
                        return torch.stack(list(loaders.map(lambda path: load_image_tensor(path, size), path_batch))).pin_memory()

                embeddings = []
                with ThreadPoolExecutor(max_workers=2) as prefetch:
                    futures = [prefetch.submit(load_path_batch, path_batch) for path_batch in path_batches[:2]]
                    next_index = len(futures)
                    with torch.no_grad():
                        for batch_index in range(len(path_batches)):
                            image_batch = futures.pop(0).result().to(self.device, non_blocking=True)
                            if next_index < len(path_batches):
                                futures.append(prefetch.submit(load_path_batch, path_batches[next_index]))
                                next_index += 1
                            embeddings.extend(surrogate.encode_image(image_batch).cpu())
                temporary = cache_path.with_suffix(".tmp")
                torch.save({"embeddings": torch.stack(embeddings)}, temporary)
                temporary.replace(cache_path)
            by_path = {str(path): embedding for path, embedding in zip(paths, embeddings)}
            for cache, item in zip(caches, items):
                cache[surrogate.name] = {
                    "positive_embeddings": torch.stack([by_path[str(path)] for path in item.positive_image_paths]),
                    "negative_embeddings": torch.stack([by_path[str(path)] for path in item.negative_image_paths]),
                    "clean_embedding": by_path[str(item.image_path)].unsqueeze(0),
                }
        return caches

    def _ensemble_proxy_eval(self, clean: torch.Tensor, adv: torch.Tensor, example_cache: dict) -> dict:
        per_surrogate = {}
        clean_margins = []
        adv_margins = []
        clean_prototype_distances = []
        adversarial_prototype_distances = []

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
            positive_prototype = F.normalize(positive_embeddings.mean(dim=0, keepdim=True), dim=-1)
            clean_prototype_distance = float(torch.linalg.vector_norm(clean_emb - positive_prototype).detach().cpu())
            adversarial_prototype_distance = float(torch.linalg.vector_norm(adv_emb - positive_prototype).detach().cpu())
            result.update({
                "clean_target_prototype_distance": clean_prototype_distance,
                "adversarial_target_prototype_distance": adversarial_prototype_distance,
                "target_prototype_distance_change": adversarial_prototype_distance - clean_prototype_distance,
            })
            per_surrogate[surrogate.name] = result
            clean_margins.append(result["clean_margin"])
            adv_margins.append(result["adversarial_margin"])
            clean_prototype_distances.append(clean_prototype_distance)
            adversarial_prototype_distances.append(adversarial_prototype_distance)
            if self.config.runtime.sequential_surrogates:
                unload_surrogate(surrogate)

        clean_margin = float(sum(clean_margins) / len(clean_margins))
        adversarial_margin = float(sum(adv_margins) / len(adv_margins))
        return {
            "clean_margin": clean_margin,
            "adversarial_margin": adversarial_margin,
            "margin_gain": adversarial_margin - clean_margin,
            "proxy_success": adversarial_margin > self.config.evaluation.success_margin_threshold and adversarial_margin > clean_margin,
            "mean_clean_target_prototype_distance": float(sum(clean_prototype_distances) / len(clean_prototype_distances)),
            "mean_adversarial_target_prototype_distance": float(sum(adversarial_prototype_distances) / len(adversarial_prototype_distances)),
            "mean_target_prototype_distance_change": float(
                sum(adversarial_prototype_distances) / len(adversarial_prototype_distances)
                - sum(clean_prototype_distances) / len(clean_prototype_distances)
            ),
            "per_surrogate": per_surrogate,
        }

    def _caption_eval(self, clean: torch.Tensor, adv: torch.Tensor, item: AttackItem) -> dict | None:
        if self.caption_victim is None:
            return None
        clean_image = tensor_to_pil_image(clean[0])
        adv_image = tensor_to_pil_image(adv[0])
        return self.caption_victim.evaluate(clean_image, adv_image, item)

    def _vqa_eval(self, clean: torch.Tensor, adv: torch.Tensor, item: AttackItem) -> dict | None:
        if self.vqa_victim is None:
            return None
        clean_image = tensor_to_pil_image(clean[0])
        adv_image = tensor_to_pil_image(adv[0])
        return self.vqa_victim.evaluate(clean_image, adv_image, item)

    def _ocr_eval(self, clean: torch.Tensor, adv: torch.Tensor, item: AttackItem) -> dict | None:
        if self.ocr_victim is None:
            return None
        clean_image = tensor_to_pil_image(clean[0])
        adv_image = tensor_to_pil_image(adv[0])
        return self.ocr_victim.evaluate(clean_image, adv_image, item)

    def _gpt_eval(self, clean: torch.Tensor, adv: torch.Tensor, item: AttackItem) -> dict | None:
        if self.gpt_victim is None:
            return None
        clean_image = tensor_to_pil_image(clean[0])
        adv_image = tensor_to_pil_image(adv[0])
        return self.gpt_victim.evaluate(clean_image, adv_image, item)

    def _ollama_eval(self, clean: torch.Tensor, adv: torch.Tensor, item: AttackItem) -> dict | None:
        if self.ollama_victim is None:
            return None
        clean_image = tensor_to_pil_image(clean[0])
        adv_image = tensor_to_pil_image(adv[0])
        return self.ollama_victim.evaluate(clean_image, adv_image, item)

    def _qwen_vl_eval(self, clean: torch.Tensor, adv: torch.Tensor, item: AttackItem) -> dict | None:
        if self.qwen_vl_victim is None:
            return None
        clean_image = tensor_to_pil_image(clean[0])
        adv_image = tensor_to_pil_image(adv[0])
        return self.qwen_vl_victim.evaluate(clean_image, adv_image, item)

    def attack_item(self, item: AttackItem) -> dict:
        started_at = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        example_cache = self._precompute_example_embeddings(item)
        base_size = self._base_attack_size()
        clean = load_image_tensor(item.image_path, base_size).unsqueeze(0).to(self.device)
        delta = torch.zeros_like(clean, requires_grad=True)
        delta_ema = torch.zeros_like(clean)
        pipelines = {
            surrogate.config.input_size: AttackAugmentationPipeline(self.config.attack, surrogate.config.input_size)
            for surrogate in self.surrogates
        }
        history = []
        patch_drop_rate = self.config.attack.patch_drop_rate if self.config.attack.enable_patch_drop else 0.0
        drop_path_max_rate = self.config.attack.drop_path_max_rate if self.config.attack.enable_drop_path else 0.0
        explicit_noise_eot = self.config.attack.noise_mode.endswith("_eot")
        explicit_geometry_eot = self.config.attack.geometry_mode not in {"legacy", "none"}
        explicit_eot = explicit_noise_eot or explicit_geometry_eot
        noise_samples = int(self.config.attack.noise_samples) if explicit_noise_eot else 1
        geometry_samples = int(self.config.attack.geometry_samples) if explicit_geometry_eot else 1
        augmentation_batches = max(1, int(self.config.attack.augmentation_batches)) * noise_samples * geometry_samples
        augmentation_forward_batch_size = max(1, int(self.config.attack.augmentation_forward_batch_size))
        metrics_interval = int(self.config.attack.metrics_interval)

        for step in tqdm(range(self.config.attack.steps), desc=f"attack:{item.item_id}", leave=False):
            collect_step_metrics = _should_collect_step_metrics(step, self.config.attack.steps, metrics_interval)
            if delta.grad is not None:
                delta.grad.zero_()

            total_loss_value = 0.0
            step_metrics = {}
            surrogate_gradients = {}
            step_noise_stats = {"sample_count": 0, "sum": 0.0, "sum_sq": 0.0, "saturated_fraction": 0.0}

            for surrogate in self.surrogates:
                gradient_before = delta.grad.detach().clone() if collect_step_metrics and delta.grad is not None else None
                if self.config.runtime.sequential_surrogates:
                    self._move_surrogate(surrogate, self.device)
                positive_embeddings = example_cache[surrogate.name]["positive_embeddings"].to(self.device)
                negative_embeddings = example_cache[surrogate.name]["negative_embeddings"].to(self.device)
                clean_reference_embeddings = example_cache[surrogate.name]["clean_embedding"].to(self.device)
                surrogate_metrics_total: dict[str, float] = {}
                surrogate_metrics_weight = 0

                for aug_start in range(0, augmentation_batches, augmentation_forward_batch_size):
                    current_batch_size = min(augmentation_forward_batch_size, augmentation_batches - aug_start)
                    bounded_delta = delta.clamp(-self.config.attack.epsilon, self.config.attack.epsilon)
                    adv = (clean + bounded_delta).clamp(0.0, 1.0)
                    surrogate_input = self._resize_for_surrogate(adv, surrogate.config.input_size)
                    if explicit_eot:
                        sigma = effective_sigma(
                            self.config.attack,
                            self.config.attack.epsilon,
                            step / max(1, self.config.attack.steps - 1),
                        )
                        generated = []
                        generator = torch.Generator(device=surrogate_input.device)
                        generator.manual_seed(self.config.runtime.seed * 1_000_003 + step * 10_007 + aug_start)
                        anti = None
                        for local_idx in range(current_batch_size):
                            if self.config.attack.noise_mode == "antithetic_gaussian_eot":
                                if local_idx % 2 == 0:
                                    anti = sample_noise("gaussian_eot", surrogate_input, sigma, generator=generator)
                                noise = anti if local_idx % 2 == 0 else -anti
                            elif explicit_noise_eot:
                                noise = sample_noise(self.config.attack.noise_mode, surrogate_input, sigma, generator=generator)
                            else:
                                noise = torch.zeros_like(surrogate_input)
                            augmented = pipelines[surrogate.config.input_size](
                                surrogate_input,
                                self.config.attack.epsilon,
                                noise=noise,
                                generator=generator,
                            )
                            generated.append(augmented)
                            if collect_step_metrics:
                                step_noise_stats["sample_count"] += 1
                                step_noise_stats["sum"] += float(noise.mean().detach().cpu())
                                step_noise_stats["sum_sq"] += float(noise.square().mean().detach().cpu())
                                step_noise_stats["saturated_fraction"] += float(
                                    ((surrogate_input + noise <= 0) | (surrogate_input + noise >= 1)).float().mean().detach().cpu()
                                )
                        surrogate_input = torch.cat(generated, dim=0)
                    else:
                        surrogate_input = self._augment_for_surrogate(
                            pipelines[surrogate.config.input_size],
                            surrogate_input,
                            current_batch_size,
                        )
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
                        collect_metrics=collect_step_metrics,
                    )
                    if self.config.attack.relative_proxy_weight > 0.0:
                        relative_loss, relative_metrics = relative_proxy_loss(
                            clean_reference_embeddings,
                            embeddings,
                            positive_embeddings,
                            negative_embeddings,
                            top_k=self.config.attack.top_k,
                            collect_metrics=collect_step_metrics,
                        )
                        loss = loss + (self.config.attack.relative_proxy_weight * relative_loss)
                        metrics.update(relative_metrics)
                    loss_weight = float(current_batch_size) / float(augmentation_batches)
                    (loss * loss_weight).backward()
                    if collect_step_metrics:
                        total_loss_value += float(loss.detach().cpu()) * loss_weight
                        _accumulate_step_metrics(surrogate_metrics_total, metrics, weight=float(current_batch_size))
                        surrogate_metrics_weight += current_batch_size

                if collect_step_metrics:
                    step_metrics[surrogate.name] = _average_step_metrics(
                        surrogate_metrics_total,
                        surrogate_metrics_weight,
                    )
                    current_gradient = delta.grad.detach()
                    surrogate_gradients[surrogate.name] = (
                        current_gradient.clone() if gradient_before is None else current_gradient - gradient_before
                    )
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

            if collect_step_metrics:
                count = max(1, int(step_noise_stats["sample_count"]))
                history.append(
                    {
                        "step": step,
                        "loss": total_loss_value,
                        "surrogates": step_metrics,
                        "composition": gradient_diagnostics(surrogate_gradients),
                        "noise": {
                            "mode": self.config.attack.noise_mode,
                            "samples": noise_samples,
                            "mean": step_noise_stats["sum"] / count,
                            "second_moment": step_noise_stats["sum_sq"] / count,
                            "saturated_fraction": step_noise_stats["saturated_fraction"] / count,
                        },
                        "surrogate_forwards": len(self.surrogates) * augmentation_batches,
                    }
                )

        final_delta = delta_ema.clamp(-self.config.attack.epsilon, self.config.attack.epsilon)
        final_adv = (clean + final_delta).clamp(0.0, 1.0)
        proxy_eval = self._ensemble_proxy_eval(clean, final_adv, example_cache)
        caption_eval = self._safe_eval(
            "caption", "caption_success", lambda: self._caption_eval(clean, final_adv, item)
        )
        vqa_eval = self._safe_eval("vqa", "vqa_success", lambda: self._vqa_eval(clean, final_adv, item))
        ocr_eval = self._safe_eval("ocr", "ocr_success", lambda: self._ocr_eval(clean, final_adv, item))
        gpt_eval = self._safe_eval("gpt", "gpt_success", lambda: self._gpt_eval(clean, final_adv, item))
        ollama_eval = self._safe_eval("ollama", "ollama_success", lambda: self._ollama_eval(clean, final_adv, item))
        qwen_vl_eval = self._safe_eval("qwen_vl", "qwen_vl_success", lambda: self._qwen_vl_eval(clean, final_adv, item))

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
            "question": item.question,
            "source_answer_text": item.source_answer_text,
            "target_answer_text": item.target_answer_text,
            "source_answer_keywords": item.source_answer_keywords,
            "target_answer_keywords": item.target_answer_keywords,
            "source_text_keywords": item.source_text_keywords,
            "target_text_keywords": item.target_text_keywords,
            "metadata": item.metadata,
            "proxy_eval": proxy_eval,
            "caption_eval": caption_eval,
            "vqa_eval": vqa_eval,
            "ocr_eval": ocr_eval,
            "gpt_eval": gpt_eval,
            "ollama_eval": ollama_eval,
            "qwen_vl_eval": qwen_vl_eval,
            "history": history,
            "composition_diagnostics": {
                "method": "equal_loss_mean",
                "surrogate_count": len(self.surrogates),
                "total_surrogate_forwards": len(self.surrogates) * augmentation_batches * self.config.attack.steps,
                "elapsed_seconds": time.monotonic() - started_at,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            },
            "noise_method": {
                "mode": self.config.attack.noise_mode,
                "sigma": self.config.attack.noise_sigma,
                "samples": noise_samples,
                "geometry_mode": self.config.attack.geometry_mode,
                "geometry_samples": geometry_samples,
                "schedule": self.config.attack.noise_schedule,
                "total_surrogate_forwards": len(self.surrogates) * augmentation_batches * self.config.attack.steps,
            },
        }
        (item_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def attack_items_batch(self, items: list[AttackItem]) -> list[dict]:
        """Optimize independent item perturbations together to fill the GPU.

        This path is limited to replay-style research runs where local/API
        victims are disabled.  Its objective is the mean of the unchanged
        per-item objectives; item perturbations do not share gradients.
        """
        if len(items) == 1:
            return [self.attack_item(items[0])]
        if any((self.caption_victim, self.vqa_victim, self.ocr_victim, self.gpt_victim,
                self.ollama_victim, self.qwen_vl_victim)):
            return [self.attack_item(item) for item in items]
        if len({(len(item.positive_image_paths), len(item.negative_image_paths)) for item in items}) != 1:
            return [self.attack_item(item) for item in items]
        started_at = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        example_caches = self._precompute_batch_example_embeddings(items)
        base_size = self._base_attack_size()
        clean = torch.stack([load_image_tensor(item.image_path, base_size) for item in items]).to(self.device)
        delta = torch.zeros_like(clean, requires_grad=True)
        delta_ema = torch.zeros_like(clean)
        pipelines = {surrogate.config.input_size: AttackAugmentationPipeline(self.config.attack, surrogate.config.input_size)
                     for surrogate in self.surrogates}
        augmentation_batches = max(1, int(self.config.attack.augmentation_batches))
        augmentation_forward_batch_size = max(1, int(self.config.attack.augmentation_forward_batch_size))
        patch_drop_rate = self.config.attack.patch_drop_rate if self.config.attack.enable_patch_drop else 0.0
        drop_path_max_rate = self.config.attack.drop_path_max_rate if self.config.attack.enable_drop_path else 0.0
        history = []
        for step in tqdm(range(self.config.attack.steps), desc=f"attack-batch:{len(items)}", leave=False):
            collect = _should_collect_step_metrics(step, self.config.attack.steps, int(self.config.attack.metrics_interval))
            if delta.grad is not None:
                delta.grad.zero_()
            step_loss = 0.0
            for surrogate in self.surrogates:
                positives = torch.stack([cache[surrogate.name]["positive_embeddings"] for cache in example_caches]).to(self.device)
                negatives = torch.stack([cache[surrogate.name]["negative_embeddings"] for cache in example_caches]).to(self.device)
                clean_reference = torch.cat([cache[surrogate.name]["clean_embedding"] for cache in example_caches]).to(self.device)
                for aug_start in range(0, augmentation_batches, augmentation_forward_batch_size):
                    current = min(augmentation_forward_batch_size, augmentation_batches - aug_start)
                    adv = (clean + delta.clamp(-self.config.attack.epsilon, self.config.attack.epsilon)).clamp(0.0, 1.0)
                    surrogate_input = self._resize_for_surrogate(adv, surrogate.config.input_size)
                    augmented = self._augment_for_surrogate(pipelines[surrogate.config.input_size], surrogate_input, current)
                    embeddings = surrogate.encode_image(augmented, patch_drop_rate=patch_drop_rate,
                                                        drop_path_max_rate=drop_path_max_rate).reshape(current, len(items), -1)
                    loss, _ = batched_visual_contrastive_loss(embeddings, positives, negatives,
                                                              self.config.attack.temperature, self.config.attack.top_k, collect)
                    if self.config.attack.relative_proxy_weight > 0.0:
                        relative_loss, _ = batched_relative_proxy_loss(clean_reference, embeddings, positives, negatives,
                                                                         self.config.attack.top_k, collect)
                        loss = loss + self.config.attack.relative_proxy_weight * relative_loss
                    weight = float(current) / float(augmentation_batches)
                    (loss * weight).backward()
                    if collect:
                        step_loss += float(loss.detach().cpu()) * weight
            if delta.grad is None:
                raise RuntimeError("Batched attack step produced no gradient.")
            with torch.no_grad():
                delta.sub_(self.config.attack.step_size * delta.grad.sign())
                delta.clamp_(-self.config.attack.epsilon, self.config.attack.epsilon)
                delta.copy_((clean + delta).clamp(0.0, 1.0) - clean)
                if self.config.attack.enable_perturbation_ema:
                    delta_ema.mul_(self.config.attack.perturbation_ema_decay).add_(delta.detach() * (1.0 - self.config.attack.perturbation_ema_decay))
                else:
                    delta_ema.copy_(delta.detach())
            if collect:
                history.append({"step": step, "loss": step_loss, "batch_size": len(items),
                                "surrogate_forwards": len(self.surrogates) * augmentation_batches})

        final_adv = (clean + delta_ema.clamp(-self.config.attack.epsilon, self.config.attack.epsilon)).clamp(0.0, 1.0)
        results = []
        for index, item in enumerate(items):
            item_clean, item_adv, item_delta = clean[index:index + 1], final_adv[index:index + 1], delta_ema[index:index + 1]
            proxy_eval = self._ensemble_proxy_eval(item_clean, item_adv, example_caches[index])
            item_dir = self.output_dir / item.item_id
            item_dir.mkdir(parents=True, exist_ok=True)
            save_tensor_image(item_clean[0], item_dir / "clean.png")
            save_tensor_image(item_adv[0], item_dir / "adversarial.png")
            save_tensor_image(((item_delta[0] / (2.0 * self.config.attack.epsilon)) + 0.5).clamp(0.0, 1.0), item_dir / "delta_vis.png")
            result = {
                "item_id": item.item_id, "image_path": str(item.image_path), "source_label": item.source_label,
                "target_label": item.target_label, "source_keywords": item.source_keywords, "target_keywords": item.target_keywords,
                "question": item.question, "source_answer_text": item.source_answer_text, "target_answer_text": item.target_answer_text,
                "source_answer_keywords": item.source_answer_keywords, "target_answer_keywords": item.target_answer_keywords,
                "source_text_keywords": item.source_text_keywords, "target_text_keywords": item.target_text_keywords,
                "metadata": item.metadata, "proxy_eval": proxy_eval,
                "caption_eval": None, "vqa_eval": None, "ocr_eval": None, "gpt_eval": None, "ollama_eval": None, "qwen_vl_eval": None,
                "history": history,
                "composition_diagnostics": {"method": "equal_loss_mean", "surrogate_count": len(self.surrogates),
                    "total_surrogate_forwards": len(self.surrogates) * augmentation_batches * self.config.attack.steps,
                    "elapsed_seconds": time.monotonic() - started_at, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    "attack_batch_size": len(items)},
            }
            (item_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            results.append(result)
        return results

    def run(self) -> dict:
        manifest = load_manifest(self.config.paths.manifest)
        self._write_effective_config()
        self.load_surrogates()
        items = manifest.items[self.config.runtime.attack_offset :]
        if self.config.runtime.attack_limit is not None:
            items = items[: self.config.runtime.attack_limit]
        batch_size = max(1, int(self.config.runtime.attack_batch_size))
        results = [result for offset in range(0, len(items), batch_size)
                   for result in self.attack_items_batch(items[offset:offset + batch_size])]
        summary_metrics = summarize_results(results)
        summary = {
            "experiment_name": self.config.experiment_name,
            "dataset_name": manifest.dataset_name,
            "run_profile": self.config.run_profile,
            "profile_metadata": self.config.profile_metadata,
            "effective_config_path": str(self.output_dir / "effective_config.json"),
            "ensemble": {
                "enabled_surrogate_count": len(enabled_surrogate_names(self.config)),
                "enabled_surrogates": enabled_surrogate_names(self.config),
                "sequential_surrogates": self.config.runtime.sequential_surrogates,
                "attack_batch_size": batch_size,
            },
            **summary_metrics,
            "items": results,
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_item_csv(results, self.output_dir / "items.csv")
        if self.caption_victim is not None:
            self.caption_victim.unload()
        if self.vqa_victim is not None:
            self.vqa_victim.unload()
        if self.ocr_victim is not None:
            self.ocr_victim.unload()
        if self.gpt_victim is not None:
            self.gpt_victim.unload()
        if self.ollama_victim is not None:
            self.ollama_victim.unload()
        if self.qwen_vl_victim is not None:
            self.qwen_vl_victim.unload()
        return summary


def run_attack(config: AttackConfig) -> dict:
    runner = CaptionAttackRunner(config)
    return runner.run()


def run_attack_from_config(config_path: str) -> dict:
    config = load_config(config_path)
    return run_attack(config)

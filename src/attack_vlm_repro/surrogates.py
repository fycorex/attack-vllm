from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import types

import open_clip
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

from .config import SurrogateConfig


class SurrogateWrapper:
    def __init__(self, config: SurrogateConfig, model: torch.nn.Module, mean: tuple[float, float, float], std: tuple[float, float, float], patch_size: int):
        self.config = config
        self.model = model
        self.mean = mean
        self.std = std
        self.patch_size = patch_size

    @property
    def name(self) -> str:
        return f"{self.config.model_name}:{self.config.pretrained}"

    def to(self, device: str) -> None:
        self.model.to(device)

    def cpu(self) -> None:
        self.model.cpu()

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        normalizer = Normalize(self.mean, self.std)
        return normalizer(images)

    def apply_patch_drop(self, images: torch.Tensor, drop_rate: float) -> torch.Tensor:
        if drop_rate <= 0.0 or self.patch_size <= 1:
            return images
        batch, _, height, width = images.shape
        grid_h = max(1, height // self.patch_size)
        grid_w = max(1, width // self.patch_size)
        keep_mask = (torch.rand(batch, 1, grid_h, grid_w, device=images.device) > drop_rate).float()
        keep_mask = F.interpolate(keep_mask, size=(height, width), mode="nearest")
        return images * keep_mask

    @contextmanager
    def drop_path_context(self, max_rate: float):
        blocks = _get_residual_blocks(self.model)
        if not blocks or max_rate <= 0.0:
            yield
            return

        originals: list[tuple[torch.nn.Module, object]] = []
        num_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            drop_prob = max_rate * float(idx + 1) / float(num_blocks)
            original_forward = block.forward

            def wrapped(self, x, *args, _orig=original_forward, _prob=drop_prob, **kwargs):
                if torch.rand((), device=x.device) < _prob:
                    return x
                return _orig(x, *args, **kwargs)

            originals.append((block, original_forward))
            block.forward = types.MethodType(wrapped, block)

        try:
            yield
        finally:
            for block, original_forward in originals:
                block.forward = original_forward

    def encode_image(self, images: torch.Tensor, patch_drop_rate: float = 0.0, drop_path_max_rate: float = 0.0) -> torch.Tensor:
        images = self.apply_patch_drop(images, patch_drop_rate)
        images = self.normalize(images)
        with self.drop_path_context(drop_path_max_rate):
            autocast_enabled = images.device.type == "cuda" and self.config.use_fp16
            with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=autocast_enabled):
                features = self.model.encode_image(images)
        return F.normalize(features.float(), dim=-1)


def _extract_mean_std(preprocess) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    for transform in getattr(preprocess, "transforms", []):
        if isinstance(transform, Normalize):
            return tuple(transform.mean), tuple(transform.std)
    return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def _get_patch_size(model: torch.nn.Module) -> int:
    visual = getattr(model, "visual", None)
    if visual is None:
        return 16
    patch_size = getattr(visual, "patch_size", None)
    if patch_size is not None:
        return int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)
    if hasattr(visual, "conv1") and hasattr(visual.conv1, "kernel_size"):
        kernel_size = visual.conv1.kernel_size
        return int(kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size)
    return 16


def _get_residual_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    visual = getattr(model, "visual", None)
    if visual is None:
        return []
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        return list(visual.transformer.resblocks)
    if hasattr(visual, "trunk") and hasattr(visual.trunk, "blocks"):
        return list(visual.trunk.blocks)
    return []


def create_surrogate(config: SurrogateConfig, device: str, cache_dir: str | Path | None = None) -> SurrogateWrapper:
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else None
    if resolved_cache_dir is not None:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=config.model_name,
        pretrained=config.pretrained,
        device=device,
        cache_dir=str(resolved_cache_dir) if resolved_cache_dir is not None else None,
    )
    model.eval()
    model.requires_grad_(False)
    mean, std = _extract_mean_std(preprocess)
    patch_size = config.patch_size or _get_patch_size(model)
    return SurrogateWrapper(config=config, model=model, mean=mean, std=std, patch_size=patch_size)


def unload_surrogate(wrapper: SurrogateWrapper) -> None:
    wrapper.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

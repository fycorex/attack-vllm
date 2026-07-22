"""Differentiable EOT transforms for the fixed C2 recipe."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def eot_transform(image: torch.Tensor, *, translation_pixels: int, resize_min: float, resize_max: float, generator: torch.Generator) -> torch.Tensor:
    """Reflection translation plus resize/crop-or-pad, preserving `[B,3,H,W]`."""
    height, width = image.shape[-2:]
    dx = int(torch.randint(-translation_pixels, translation_pixels + 1, (), generator=generator, device=image.device))
    dy = int(torch.randint(-translation_pixels, translation_pixels + 1, (), generator=generator, device=image.device))
    padded = F.pad(image, (translation_pixels, translation_pixels, translation_pixels, translation_pixels), mode="reflect")
    moved = padded[:, :, translation_pixels + dy : translation_pixels + dy + height, translation_pixels + dx : translation_pixels + dx + width]
    scale = float(torch.empty((), device=image.device).uniform_(resize_min, resize_max, generator=generator))
    resized = F.interpolate(moved, size=(max(1, round(height * scale)), max(1, round(width * scale))), mode="bilinear", align_corners=False, antialias=True)
    if scale >= 1:
        top, left = (resized.shape[-2] - height) // 2, (resized.shape[-1] - width) // 2
        return resized[:, :, top : top + height, left : left + width]
    pad_h, pad_w = height - resized.shape[-2], width - resized.shape[-1]
    return F.pad(resized, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode="reflect")

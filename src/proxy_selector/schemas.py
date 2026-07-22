"""Typed data shared by the proxy-selector pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class AttackRecipe:
    """Fixed, pre-registered attack configuration; not a tuning search space."""

    name: str
    epsilon: float = 8.0 / 255.0
    step_size: float = 1.0 / 255.0
    momentum: float = 0.9
    temperature: float = 0.10
    local_weight: float = 0.0
    source_weight: float = 0.0
    eot_samples: int = 1
    translation_pixels: int = 0
    resize_min: float = 1.0
    resize_max: float = 1.0


@dataclass
class ImageTokenOutput:
    """A model's image-only representation at a validated vision-language tap."""

    global_features: torch.Tensor
    local_tokens: torch.Tensor
    token_mask: torch.BoolTensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.global_features.ndim != 2:
            raise ValueError("global_features must have shape [B, D].")
        if self.local_tokens.ndim != 3:
            raise ValueError("local_tokens must have shape [B, T, D].")
        if self.token_mask.ndim != 2:
            raise ValueError("token_mask must have shape [B, T].")
        if self.local_tokens.shape[:2] != self.token_mask.shape:
            raise ValueError("token_mask must match the first two local_tokens dimensions.")
        if self.global_features.shape[0] != self.local_tokens.shape[0]:
            raise ValueError("global_features and local_tokens must have the same batch size.")
        if not self.token_mask.any(dim=1).all():
            raise ValueError("Every item must contain at least one valid image token.")

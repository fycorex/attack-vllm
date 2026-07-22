"""Image-token pooling and normalization primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean_pool(tokens: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
    """Pool valid tokens only, preserving gradients and rejecting empty rows."""
    if tokens.ndim != 3 or mask.ndim != 2 or tokens.shape[:2] != mask.shape:
        raise ValueError("tokens must be [B, T, D] and mask must be [B, T].")
    counts = mask.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("Cannot pool an item with no valid tokens.")
    return (tokens * mask.unsqueeze(-1).to(tokens.dtype)).sum(dim=1) / counts.to(tokens.dtype)


def row_normalize(tokens: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize each token vector."""
    return F.normalize(tokens, dim=-1, eps=eps)


def global_features(tokens: torch.Tensor, mask: torch.BoolTensor, eps: float = 1e-12) -> torch.Tensor:
    """Masked token mean followed by L2 normalization."""
    return F.normalize(masked_mean_pool(tokens, mask), dim=-1, eps=eps)

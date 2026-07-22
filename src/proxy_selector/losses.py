"""The fixed image-token objective specified by the pilot guideline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def global_contrastive_loss(
    candidate: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """-log(A+/(A++A-)) for normalized global image features."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    candidate = F.normalize(candidate, dim=-1)
    positives = F.normalize(positives, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    positive_logits = candidate @ positives.T / temperature
    negative_logits = candidate @ negatives.T / temperature
    return -(torch.logsumexp(positive_logits, dim=-1) - torch.logsumexp(
        torch.cat((positive_logits, negative_logits), dim=-1), dim=-1
    )).mean()


def symmetric_token_score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Symmetric max-cosine score M(u,v), returned once per batch element."""
    if left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]:
        raise ValueError("left and right must be [B, T, D] tensors with matching batch sizes.")
    similarity = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).transpose(-1, -2)
    return 0.5 * (similarity.max(dim=-1).values.mean(dim=-1) + similarity.max(dim=-2).values.mean(dim=-1))


def local_alignment_loss(candidate_tokens: torch.Tensor, target_references: torch.Tensor) -> torch.Tensor:
    """Negative mean local score over target references [A, T, D]."""
    if candidate_tokens.shape[0] != 1:
        raise ValueError("The pilot runner attacks one source image at a time.")
    scores = [symmetric_token_score(candidate_tokens, reference.unsqueeze(0)) for reference in target_references]
    return -torch.cat(scores).mean()


def source_repulsion_loss(candidate: torch.Tensor, clean_source: torch.Tensor) -> torch.Tensor:
    """Minimizing this cosine repels the candidate from the clean source."""
    return F.cosine_similarity(candidate, clean_source, dim=-1).mean()


def ve_token_dissimilarity_loss(candidate_layers: dict[str, torch.Tensor], clean_layers: dict[str, torch.Tensor]) -> torch.Tensor:
    """VEAttack-style mean patch-token cosine, minimized to disrupt vision features."""
    common = sorted(set(candidate_layers) & set(clean_layers))
    if not common:
        raise ValueError("VE loss needs at least one common visual layer.")
    return torch.stack([
        F.cosine_similarity(candidate_layers[name], clean_layers[name], dim=-1).mean()
        for name in common
    ]).mean()


def semantic_direction_loss(
    adversarial: torch.Tensor,
    clean: torch.Tensor,
    target_direction: torch.Tensor,
) -> torch.Tensor:
    """UnivIntruder-style semantic displacement direction loss."""
    displacement = F.normalize(adversarial - clean, dim=-1)
    return 1.0 - F.cosine_similarity(displacement, target_direction, dim=-1).mean()


def endpoint_loss(adversarial: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Move a shared image/text embedding to the target semantic endpoint."""
    return 1.0 - F.cosine_similarity(adversarial, target, dim=-1).mean()

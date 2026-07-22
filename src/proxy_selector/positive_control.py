"""Single-checkpoint literature-grounded positive-control utilities.

This module deliberately combines compatible ideas rather than claiming an
exact reimplementation of VEAttack, UnivIntruder, SGHA-Attack, or RaPA.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import torch
import torch.nn.functional as F

from .losses import endpoint_loss, local_alignment_loss, semantic_direction_loss, ve_token_dissimilarity_loss
from .representations import global_features


LAYER_WEIGHTS: dict[str, float] = {"25": 0.10, "50": 0.15, "75": 0.20, "final": 0.25, "interface": 0.30}


def normalized_mean(features: list[torch.Tensor]) -> torch.Tensor:
    if not features:
        raise ValueError("Cannot calculate a centroid from no features.")
    return F.normalize(torch.cat(features, dim=0).mean(dim=0, keepdim=True), dim=-1)


def direction_target(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    return F.normalize(positive - negative, dim=-1)


def hierarchical_loss(
    candidate: dict[str, object],
    target_centroids: dict[str, torch.Tensor],
    target_local: dict[str, list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """SGHA-inspired weighted global and local losses at available depths."""
    available = [name for name in target_centroids if name in candidate]
    if not available:
        raise ValueError("No overlapping hierarchical visual layers.")
    normalizer = sum(LAYER_WEIGHTS.get(name, 0.0) for name in available)
    global_terms: list[torch.Tensor] = []
    local_terms: list[torch.Tensor] = []
    for name in available:
        output = candidate[name]
        weight = LAYER_WEIGHTS.get(name, 0.0) / normalizer
        global_terms.append(weight * (1.0 - F.cosine_similarity(output.global_features, target_centroids[name], dim=-1).mean()))
        local_terms.append(weight * local_alignment_loss(output.local_tokens, torch.cat(target_local[name], dim=0)))
    return torch.stack(global_terms).sum(), torch.stack(local_terms).sum()


def ve_loss_from_outputs(candidate: dict[str, object], clean: dict[str, object]) -> torch.Tensor:
    return ve_token_dissimilarity_loss(
        {name: value.local_tokens for name, value in candidate.items()},
        {name: value.local_tokens for name, value in clean.items()},
    )


@contextmanager
def rpa_visual_output_pruning(model: torch.nn.Module, ratio: float, generator: torch.Generator) -> Iterator[int]:
    """Temporarily prune only vision attention/MLP output projections.

    The context restores every tensor even when autograd raises.  Parameters
    outside the vision model and normalization/embedding/projector parameters
    are never selected.
    """
    if ratio <= 0:
        yield 0
        return
    vision = getattr(model, "vision_model", None)
    if vision is None:
        raise ValueError("RaPA positive control currently supports vision-model proxies only.")
    selected: list[torch.nn.Parameter] = []
    for name, parameter in vision.named_parameters():
        lower = name.lower()
        output_projection = (
            "out_proj.weight" in lower
            or lower.endswith("attention.output.dense.weight")
            or lower.endswith("mlp.fc2.weight")
            or lower.endswith("mlp.fc2.bias")
            or lower.endswith("mlp.wo.weight")
        )
        forbidden = any(term in lower for term in ("norm", "embedding", "position", "patch", "projector", "merger"))
        if output_projection and not forbidden:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("No permitted visual attention/MLP output projections found for RaPA.")
    originals = [parameter.detach().clone() for parameter in selected]
    try:
        with torch.no_grad():
            for parameter in selected:
                keep = torch.rand(parameter.shape, device=parameter.device, generator=generator) >= ratio
                parameter.mul_(keep.to(parameter.dtype))
        yield len(selected)
    finally:
        with torch.no_grad():
            for parameter, original in zip(selected, originals, strict=True):
                parameter.copy_(original)


def layer_anchor_cache(adapter: object, anchors: list[torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]]:
    """Cache fixed 13-anchor global centroids and local token references."""
    layers = [adapter.encode_image_layers(anchor, require_grad=False) for anchor in anchors]
    names = set.intersection(*(set(item) for item in layers))
    centroids: dict[str, torch.Tensor] = {}
    locals_: dict[str, list[torch.Tensor]] = {}
    for name in names:
        centroids[name] = normalized_mean([item[name].global_features for item in layers])
        locals_[name] = [item[name].local_tokens for item in layers]
    return centroids, locals_


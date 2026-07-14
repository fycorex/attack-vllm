from __future__ import annotations

"""Provider-independent representation geometry metrics."""

import math

import torch
import torch.nn.functional as F


def _matrix(value: torch.Tensor, name: str, *, min_samples: int = 2) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if value.shape[0] < min_samples:
        raise ValueError(f"{name} must contain at least {min_samples} samples")
    return value.float()


def linear_gram(representations: torch.Tensor) -> torch.Tensor:
    values = _matrix(representations, "representations")
    return values @ values.T


def center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    values = _matrix(kernel, "kernel")
    if values.shape[0] != values.shape[1]:
        raise ValueError("kernel must be square")
    return values - values.mean(0, keepdim=True) - values.mean(1, keepdim=True) + values.mean()


def normalized_kernel_alignment(first: torch.Tensor, second: torch.Tensor, *, eps: float = 1e-12) -> float:
    first = _matrix(first, "first kernel")
    second = _matrix(second, "second kernel")
    if first.shape != second.shape:
        raise ValueError("kernel matrices must have identical shapes")
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= eps:
        return 0.0
    return float(((first * second).sum() / denominator).item())


def uncentered_kernel_alignment(first: torch.Tensor, second: torch.Tensor) -> float:
    return normalized_kernel_alignment(linear_gram(first), linear_gram(second))


def centered_linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    return normalized_kernel_alignment(center_kernel(linear_gram(first)), center_kernel(linear_gram(second)))


def pairwise_distances(representations: torch.Tensor) -> torch.Tensor:
    return torch.cdist(_matrix(representations, "representations"), _matrix(representations, "representations"))


def neighbor_margin(representations: torch.Tensor, k: int) -> torch.Tensor:
    distances = pairwise_distances(representations)
    n = distances.shape[0]
    if not 1 <= k < n - 1:
        raise ValueError(f"k must satisfy 1 <= k < n-1, got k={k}, n={n}")
    distances.fill_diagonal_(math.inf)
    ordered = distances.sort(dim=1).values
    return ordered[:, k] - ordered[:, k - 1]


def neighborhood_overlap(first: torch.Tensor, second: torch.Tensor, k: int) -> float:
    first_distances = pairwise_distances(first)
    second_distances = pairwise_distances(second)
    if first_distances.shape != second_distances.shape:
        raise ValueError("representation matrices must contain the same number of samples")
    n = first_distances.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < n, got k={k}, n={n}")
    first_distances.fill_diagonal_(math.inf)
    second_distances.fill_diagonal_(math.inf)
    first_neighbors = first_distances.topk(k, largest=False).indices
    second_neighbors = second_distances.topk(k, largest=False).indices
    overlaps = []
    for first_row, second_row in zip(first_neighbors, second_neighbors):
        overlaps.append(len(set(first_row.tolist()).intersection(second_row.tolist())) / k)
    return float(sum(overlaps) / len(overlaps))


def normalized_prototype(examples: torch.Tensor) -> torch.Tensor:
    examples = _matrix(examples, "examples", min_samples=1)
    return F.normalize(examples.mean(dim=0, keepdim=True), dim=-1)


def prototype_distance(representations: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
    representations = _matrix(representations, "representations", min_samples=1)
    if prototype.ndim == 1:
        prototype = prototype.unsqueeze(0)
    if prototype.shape != (1, representations.shape[1]):
        raise ValueError("prototype dimension does not match representations")
    return torch.linalg.vector_norm(representations - F.normalize(prototype.float(), dim=-1), dim=-1)


def explained_alignment(proxy_kernels: list[torch.Tensor], target_kernel: torch.Tensor, *, ridge: float = 1e-8) -> dict[str, float | int]:
    if not proxy_kernels:
        raise ValueError("at least one proxy kernel is required")
    target = _matrix(target_kernel, "target kernel")
    flat = torch.stack([_matrix(kernel, "proxy kernel").flatten() for kernel in proxy_kernels]).double()
    target_flat = target.flatten().double()
    gram = flat @ flat.T
    correlations = flat @ target_flat
    scale = max(float(torch.diagonal(gram).mean()), 1e-12)
    regularization = float(ridge) * scale
    regularized = gram + regularization * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    solution = torch.linalg.solve(regularized, correlations)
    denominator = float(target_flat.square().sum())
    value = float((correlations @ solution).item() / denominator) if denominator else 0.0
    return {
        "explained_alignment": value,
        "gram_condition_number": float(torch.linalg.cond(gram).item()),
        "gram_rank": int(torch.linalg.matrix_rank(gram).item()),
        "regularization": regularization,
    }


def distance_kernel_audit(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    first = F.normalize(_matrix(first, "first representations"), dim=-1)
    second = F.normalize(_matrix(second, "second representations"), dim=-1)
    if first.shape[0] != second.shape[0]:
        raise ValueError("representation matrices must contain the same number of samples")
    first_distances = pairwise_distances(first)
    second_distances = pairwise_distances(second)
    kernel_delta = (linear_gram(first) - linear_gram(second)).abs()
    lhs = (first_distances - second_distances).abs()
    rhs = torch.sqrt(2.0 * kernel_delta)
    violation = (lhs - rhs).clamp_min(0)
    return {
        "maximum_pointwise_violation": float(violation.max().item()),
        "mean_distance_discrepancy": float(lhs.mean().item()),
        "mean_kernel_discrepancy": float(kernel_delta.mean().item()),
    }


def pearson_correlation(first: torch.Tensor, second: torch.Tensor, *, eps: float = 1e-12) -> float:
    first = first.float().flatten()
    second = second.float().flatten()
    if first.shape != second.shape or first.numel() < 2:
        raise ValueError("correlation inputs must have the same shape and at least two values")
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= eps:
        return 0.0
    return float(((first * second).sum() / denominator).item())


def gradient_stability(gradients: torch.Tensor, *, eps: float = 1e-12) -> dict[str, float]:
    if gradients.ndim < 2 or gradients.shape[0] < 2:
        raise ValueError("gradients must contain at least two samples")
    flat = gradients.float().flatten(1)
    normalized = F.normalize(flat, dim=1, eps=eps)
    cosines = normalized @ normalized.T
    mask = ~torch.eye(cosines.shape[0], dtype=torch.bool, device=cosines.device)
    return {
        "gradient_component_variance": float(flat.var(dim=0, unbiased=False).mean().item()),
        "gradient_norm_mean": float(torch.linalg.vector_norm(flat, dim=1).mean().item()),
        "gradient_pairwise_cosine_mean": float(cosines[mask].mean().item()),
    }

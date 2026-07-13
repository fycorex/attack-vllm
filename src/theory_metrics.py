from __future__ import annotations

import math

import torch


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError(f"{name} must be a 2D matrix with at least two rows")
    return value.float()


def linear_gram(representations: torch.Tensor) -> torch.Tensor:
    values = _matrix(representations, "representations")
    return values @ values.T


def center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    values = _matrix(kernel, "kernel")
    if values.shape[0] != values.shape[1]:
        raise ValueError("kernel must be square")
    return values - values.mean(0, keepdim=True) - values.mean(1, keepdim=True) + values.mean()


def normalized_kernel_alignment(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-12) -> float:
    first = _matrix(first, "first kernel")
    second = _matrix(second, "second kernel")
    if first.shape != second.shape:
        raise ValueError("kernels must have identical shapes")
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= eps:
        return 0.0
    return float(((first * second).sum() / denominator).item())


def representation_kernel(representations: torch.Tensor, centered: bool) -> torch.Tensor:
    kernel = linear_gram(representations)
    return center_kernel(kernel) if centered else kernel


def centered_linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    return normalized_kernel_alignment(representation_kernel(first, True), representation_kernel(second, True))


def uncentered_kernel_alignment(first: torch.Tensor, second: torch.Tensor) -> float:
    return normalized_kernel_alignment(representation_kernel(first, False), representation_kernel(second, False))


def neighborhood_overlap(first: torch.Tensor, second: torch.Tensor, k: int) -> float:
    first, second = _matrix(first, "first representations"), _matrix(second, "second representations")
    if first.shape[0] != second.shape[0]:
        raise ValueError("representations must contain the same sample count")
    n = first.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < {n}")
    first_distances, second_distances = torch.cdist(first, first), torch.cdist(second, second)
    first_distances.fill_diagonal_(math.inf)
    second_distances.fill_diagonal_(math.inf)
    first_neighbors = first_distances.topk(k, largest=False).indices
    second_neighbors = second_distances.topk(k, largest=False).indices
    values = [len(set(a.tolist()).intersection(b.tolist())) / k for a, b in zip(first_neighbors, second_neighbors)]
    return float(sum(values) / len(values))


def neighbor_margin(representations: torch.Tensor, k: int) -> torch.Tensor:
    representations = _matrix(representations, "representations")
    if not 1 <= k < representations.shape[0] - 1:
        raise ValueError("neighbor margin requires 1 <= k < n-1")
    distances = torch.cdist(representations, representations)
    distances.fill_diagonal_(math.inf)
    ordered = distances.sort(dim=1).values
    return ordered[:, k] - ordered[:, k - 1]


def explained_alignment(proxy_kernels: list[torch.Tensor], target_kernel: torch.Tensor, ridge: float = 1e-8) -> dict[str, float | int]:
    if not proxy_kernels:
        raise ValueError("at least one proxy kernel is required")
    target = _matrix(target_kernel, "target kernel").double()
    flat = torch.stack([_matrix(value, "proxy kernel").flatten() for value in proxy_kernels]).double()
    target_flat = target.flatten()
    gram = flat @ flat.T
    correlations = flat @ target_flat
    scale = max(float(torch.diagonal(gram).mean()), 1e-12)
    regularization = float(ridge) * scale
    regularized = gram + regularization * torch.eye(gram.shape[0], dtype=gram.dtype)
    coefficients = torch.linalg.solve(regularized, correlations)
    denominator = float(target_flat.square().sum())
    reconstruction = coefficients @ flat
    return {
        "explained_alignment": float((correlations @ coefficients).item() / denominator) if denominator else 0.0,
        "gram_condition_number": float(torch.linalg.cond(gram).item()),
        "gram_rank": int(torch.linalg.matrix_rank(gram).item()),
        "regularization": regularization,
        "relative_reconstruction_error": float(torch.linalg.vector_norm(target_flat - reconstruction).item() / math.sqrt(denominator)) if denominator else 0.0,
    }


def ensemble_theory_metrics(proxy_representations: list[torch.Tensor], target_representations: torch.Tensor, *, centered: bool, ridge: float = 1e-8) -> dict[str, float | int | None]:
    if not proxy_representations:
        raise ValueError("at least one proxy is required")
    target_kernel = representation_kernel(target_representations, centered)
    proxy_kernels = [representation_kernel(value, centered) for value in proxy_representations]
    target_alignments = [normalized_kernel_alignment(value, target_kernel) for value in proxy_kernels]
    inter_proxy = [normalized_kernel_alignment(proxy_kernels[i], proxy_kernels[j]) for i in range(len(proxy_kernels)) for j in range(i + 1, len(proxy_kernels))]
    ensemble_kernel = torch.stack(proxy_kernels).mean(0)
    target_norm = torch.linalg.vector_norm(target_kernel)
    discrepancy = torch.linalg.vector_norm(target_kernel - ensemble_kernel)
    return {
        "ensemble_size": len(proxy_kernels),
        "mean_target_alignment": float(sum(target_alignments) / len(target_alignments)),
        "minimum_target_alignment": float(min(target_alignments)),
        "maximum_target_alignment": float(max(target_alignments)),
        "mean_inter_proxy_alignment": float(sum(inter_proxy) / len(inter_proxy)) if inter_proxy else None,
        "ensemble_kernel_discrepancy": float(discrepancy.item()),
        "relative_ensemble_kernel_discrepancy": float((discrepancy / target_norm).item()) if float(target_norm) else 0.0,
        **explained_alignment(proxy_kernels, target_kernel, ridge=ridge),
    }

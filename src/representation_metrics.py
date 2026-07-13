from __future__ import annotations

import math

import torch


def _representations(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError(f"{name} must be a 2D matrix with at least two samples")
    return value.float()


def linear_gram(representations: torch.Tensor) -> torch.Tensor:
    values = _representations(representations, "representations")
    return values @ values.T


def center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix")
    values = kernel.float()
    return values - values.mean(0, keepdim=True) - values.mean(1, keepdim=True) + values.mean()


def normalized_kernel_alignment(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-12) -> float:
    if first.shape != second.shape:
        raise ValueError("kernels must have identical shapes")
    first, second = first.float(), second.float()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= eps:
        return 0.0
    return float(((first * second).sum() / denominator).item())


def centered_linear_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    return normalized_kernel_alignment(center_kernel(linear_gram(first)), center_kernel(linear_gram(second)))


def uncentered_kernel_alignment(first: torch.Tensor, second: torch.Tensor) -> float:
    return normalized_kernel_alignment(linear_gram(first), linear_gram(second))


def neighborhood_overlap(first: torch.Tensor, second: torch.Tensor, k: int) -> float:
    first = _representations(first, "first representations")
    second = _representations(second, "second representations")
    if first.shape[0] != second.shape[0]:
        raise ValueError("representations must have identical sample counts")
    if not 1 <= k < first.shape[0]:
        raise ValueError("k must satisfy 1 <= k < sample count")
    first_distances, second_distances = torch.cdist(first, first), torch.cdist(second, second)
    first_distances.fill_diagonal_(math.inf)
    second_distances.fill_diagonal_(math.inf)
    first_neighbors = first_distances.topk(k, largest=False).indices
    second_neighbors = second_distances.topk(k, largest=False).indices
    overlap = [len(set(a.tolist()).intersection(b.tolist())) / k for a, b in zip(first_neighbors, second_neighbors)]
    return float(sum(overlap) / len(overlap))


def neighbor_margin(representations: torch.Tensor, k: int) -> torch.Tensor:
    values = _representations(representations, "representations")
    if not 1 <= k < values.shape[0] - 1:
        raise ValueError("neighbor margin requires 1 <= k < sample count - 1")
    distances = torch.cdist(values, values)
    distances.fill_diagonal_(math.inf)
    ordered = distances.sort(dim=1).values
    return ordered[:, k] - ordered[:, k - 1]

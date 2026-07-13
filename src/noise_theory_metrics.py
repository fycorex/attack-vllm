from __future__ import annotations

import torch
import torch.nn.functional as F

from representation_metrics import center_kernel, linear_gram, normalized_kernel_alignment


def normalized_embeddings(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2:
        raise ValueError("embeddings must have a feature dimension")
    return F.normalize(values.float(), dim=-1)


def smoothed_embeddings(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and renormalized means for samples shaped [T, N, D]."""
    if samples.ndim != 3 or samples.shape[0] < 1 or samples.shape[1] < 2:
        raise ValueError("samples must have shape [noise_samples, items, features]")
    raw_mean = normalized_embeddings(samples).mean(dim=0)
    return raw_mean, normalized_embeddings(raw_mean)


def jensen_gap(samples: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """E||phi-a|| - ||E phi-a||, one value per item."""
    normalized = normalized_embeddings(samples)
    if anchors.ndim == 1:
        anchors = anchors.unsqueeze(0).expand(samples.shape[1], -1)
    if anchors.shape != samples.shape[1:]:
        raise ValueError("anchors must have shape [items, features] or [features]")
    anchors = normalized_embeddings(anchors)
    expected_distance = torch.linalg.vector_norm(normalized - anchors.unsqueeze(0), dim=-1).mean(dim=0)
    smoothed_distance = torch.linalg.vector_norm(normalized.mean(dim=0) - anchors, dim=-1)
    return expected_distance - smoothed_distance


def normalized_frobenius_inner(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-12) -> float:
    if first.shape != second.shape:
        raise ValueError("matrices must have identical shapes")
    denominator = torch.linalg.vector_norm(first.float()) * torch.linalg.vector_norm(second.float())
    if float(denominator) <= eps:
        return 0.0
    return float(((first.float() * second.float()).sum() / denominator).item())


def kernel_residual_statistics(
    proxy: torch.Tensor,
    smoothed_proxy: torch.Tensor,
    target: torch.Tensor,
    *,
    centered: bool,
) -> dict[str, float]:
    proxy_kernel = linear_gram(normalized_embeddings(proxy))
    smooth_kernel = linear_gram(smoothed_proxy.float())
    target_kernel = linear_gram(normalized_embeddings(target))
    if centered:
        proxy_kernel = center_kernel(proxy_kernel)
        smooth_kernel = center_kernel(smooth_kernel)
        target_kernel = center_kernel(target_kernel)
    residual = proxy_kernel - smooth_kernel
    return {
        "alignment_before": normalized_kernel_alignment(proxy_kernel, target_kernel),
        "alignment_after": normalized_kernel_alignment(smooth_kernel, target_kernel),
        "alignment_delta": normalized_kernel_alignment(smooth_kernel, target_kernel)
        - normalized_kernel_alignment(proxy_kernel, target_kernel),
        "discrepancy_before": float(torch.linalg.vector_norm(target_kernel - proxy_kernel).item()),
        "discrepancy_after": float(torch.linalg.vector_norm(target_kernel - smooth_kernel).item()),
        "discrepancy_delta": float(
            (torch.linalg.vector_norm(target_kernel - smooth_kernel) - torch.linalg.vector_norm(target_kernel - proxy_kernel)).item()
        ),
        "residual_norm": float(torch.linalg.vector_norm(residual).item()),
        "residual_orthogonality": normalized_frobenius_inner(smooth_kernel, residual),
        "residual_target_relevance": normalized_frobenius_inner(residual, target_kernel),
    }


def smoothing_diagnostics(proxy: torch.Tensor, samples: torch.Tensor, target: torch.Tensor) -> dict[str, dict[str, float]]:
    raw_mean, renormalized_mean = smoothed_embeddings(samples)
    output: dict[str, dict[str, float]] = {}
    for mean_name, smoothed in (("raw_mean", raw_mean), ("renormalized_mean", renormalized_mean)):
        for kernel_name, centered in (("uncentered", False), ("centered", True)):
            output[f"{mean_name}_{kernel_name}"] = kernel_residual_statistics(
                proxy, smoothed, target, centered=centered
            )
    return output


def bootstrap_smoothing_diagnostics(
    proxy: torch.Tensor,
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, list[float] | int]]:
    if proxy.shape[0] != samples.shape[1] or proxy.shape[0] != target.shape[0]:
        raise ValueError("proxy, noise samples, and target must have identical item counts")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    collected: dict[str, dict[str, list[float]]] = {}
    for _ in range(bootstrap_samples):
        indices = torch.randint(0, proxy.shape[0], (proxy.shape[0],), generator=generator)
        draw = smoothing_diagnostics(proxy[indices], samples[:, indices], target[indices])
        for definition, metrics in draw.items():
            for name, value in metrics.items():
                collected.setdefault(definition, {}).setdefault(name, []).append(value)
    output: dict[str, dict[str, list[float] | int]] = {}
    for definition, metrics in collected.items():
        output[definition] = {"bootstrap_samples": bootstrap_samples}
        for name, values in metrics.items():
            ordered = sorted(values)
            output[definition][f"{name}_ci95"] = [
                ordered[int(0.025 * (len(ordered) - 1))],
                ordered[int(0.975 * (len(ordered) - 1))],
            ]
    return output


def clipping_statistics(original: torch.Tensor, sampled: torch.Tensor) -> dict[str, float]:
    if sampled.ndim != original.ndim + 1 or sampled.shape[1:] != original.shape:
        raise ValueError("sampled images must have shape [T, ...original.shape]")
    effective = sampled - original.unsqueeze(0)
    return {
        "effective_noise_mean": float(effective.mean().item()),
        "effective_noise_std": float(effective.std(unbiased=False).item()),
        "saturated_fraction": float(((sampled <= 0) | (sampled >= 1)).float().mean().item()),
    }

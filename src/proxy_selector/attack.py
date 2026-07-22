"""Momentum projected-gradient minimization for a single proxy."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

from .schemas import AttackRecipe


LossFunction = Callable[[torch.Tensor], torch.Tensor]


def project_linf(candidate: torch.Tensor, clean: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Project to the RGB box and L-infinity ball around clean."""
    delta = (candidate - clean).clamp(min=-epsilon, max=epsilon)
    return (clean + delta).clamp(0.0, 1.0)


def momentum_pgd_minimize(
    clean: torch.Tensor,
    loss_fn: LossFunction,
    recipe: AttackRecipe,
    *,
    steps: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Run the exact minus-sign momentum PGD update from the guideline."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    noise = torch.empty_like(clean).uniform_(-recipe.epsilon, recipe.epsilon, generator=generator)
    current = project_linf(clean + noise, clean, recipe.epsilon).detach()
    momentum = torch.zeros_like(clean)
    history: list[float] = []
    for _ in range(steps):
        current.requires_grad_(True)
        loss = loss_fn(current)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("loss_fn must return one finite scalar loss.")
        gradient = torch.autograd.grad(loss, current, only_inputs=True)[0]
        normalized = gradient / (gradient.abs().mean() + 1e-12)
        momentum = recipe.momentum * momentum + normalized
        current = project_linf(current - recipe.step_size * momentum.sign(), clean, recipe.epsilon).detach()
        history.append(float(loss.detach().cpu()))
    return current, history


def momentum_pgd_minimize_eot(
    clean: torch.Tensor,
    branch_losses: Callable[[torch.Tensor], Iterable[torch.Tensor]],
    recipe: AttackRecipe,
    *,
    steps: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Memory-bounded EOT PGD: accumulate one branch gradient at a time.

    This is exactly the gradient of the mean EOT loss, but avoids retaining all
    eight SigLIP-384 autograd graphs simultaneously on the single A6000.
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    current = project_linf(
        clean + torch.empty_like(clean).uniform_(-recipe.epsilon, recipe.epsilon, generator=generator),
        clean,
        recipe.epsilon,
    ).detach()
    momentum = torch.zeros_like(clean)
    history: list[float] = []
    for _ in range(steps):
        current = current.detach().requires_grad_(True)
        accumulated = torch.zeros_like(current)
        values: list[float] = []
        count = 0
        for loss in branch_losses(current):
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("Each EOT branch must return one finite scalar loss.")
            accumulated.add_(torch.autograd.grad(loss, current, only_inputs=True)[0])
            values.append(float(loss.detach().cpu()))
            count += 1
        if count == 0:
            raise ValueError("EOT branch generator returned no losses.")
        gradient = accumulated / count
        normalized = gradient / (gradient.abs().mean() + 1e-12)
        momentum = recipe.momentum * momentum + normalized
        current = project_linf(current - recipe.step_size * momentum.sign(), clean, recipe.epsilon).detach()
        history.append(sum(values) / count)
    return current, history


def debiased_momentum_pgd_minimize(
    clean: torch.Tensor,
    loss_fn: LossFunction,
    recipe: AttackRecipe,
    *,
    dataset_rgb_mean: torch.Tensor,
    reference_noise_std: float,
    beta: float,
    steps: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """Single-surrogate DeBias update adapted from DeBias-Attack.

    The reference branch is centred on the dataset RGB mean with fresh weak
    Gaussian semantics each step.  It is optimized by the *same* surrogate and
    loss as the main branch; its perturbation is then evaluated on the original
    image context before the positive aligned gradient component is removed.
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    mean = dataset_rgb_mean.to(device=clean.device, dtype=clean.dtype).view(1, 3, 1, 1)
    main_delta = torch.empty_like(clean).uniform_(-recipe.epsilon, recipe.epsilon, generator=generator)
    reference_delta = torch.empty_like(clean).uniform_(-recipe.epsilon, recipe.epsilon, generator=generator)
    main_delta = main_delta.clamp(-recipe.epsilon, recipe.epsilon).detach()
    reference_delta = reference_delta.clamp(-recipe.epsilon, recipe.epsilon).detach()
    main_momentum = torch.zeros_like(clean)
    reference_momentum = torch.zeros_like(clean)
    history: list[float] = []
    projection_coefficients: list[float] = []

    for _ in range(steps):
        current = project_linf(clean + main_delta, clean, recipe.epsilon).detach().requires_grad_(True)
        main_loss = loss_fn(current)
        if main_loss.ndim != 0 or not torch.isfinite(main_loss):
            raise ValueError("loss_fn must return one finite scalar loss.")
        main_gradient = torch.autograd.grad(main_loss, current, only_inputs=True)[0]

        weak_base = (mean + torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator) * reference_noise_std).clamp(0.0, 1.0)
        weak_candidate = (weak_base + reference_delta).clamp(0.0, 1.0).detach().requires_grad_(True)
        reference_loss = loss_fn(weak_candidate)
        reference_weak_gradient = torch.autograd.grad(reference_loss, weak_candidate, only_inputs=True)[0]
        normalized_reference = reference_weak_gradient / (reference_weak_gradient.abs().mean() + 1e-12)
        reference_momentum = recipe.momentum * reference_momentum + normalized_reference
        reference_delta = (reference_delta - recipe.step_size * reference_momentum.sign()).clamp(
            -recipe.epsilon, recipe.epsilon
        ).detach()

        reference_on_clean = project_linf(clean + reference_delta, clean, recipe.epsilon).detach().requires_grad_(True)
        reference_context_loss = loss_fn(reference_on_clean)
        reference_gradient = torch.autograd.grad(reference_context_loss, reference_on_clean, only_inputs=True)[0]
        dot = (main_gradient * reference_gradient).flatten(1).sum(dim=1, keepdim=True)
        reference_norm_sq = reference_gradient.square().flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-12)
        coefficient = (dot.clamp_min(0.0) / reference_norm_sq).view(-1, 1, 1, 1)
        corrected = main_gradient - beta * coefficient * reference_gradient
        update_gradient = main_gradient + corrected
        normalized_update = update_gradient / (update_gradient.abs().mean() + 1e-12)
        main_momentum = recipe.momentum * main_momentum + normalized_update
        main_delta = (main_delta - recipe.step_size * main_momentum.sign()).clamp(
            -recipe.epsilon, recipe.epsilon
        ).detach()
        history.append(float(main_loss.detach().cpu()))
        projection_coefficients.append(float(coefficient.detach().mean().cpu()))

    return project_linf(clean + main_delta, clean, recipe.epsilon), history, projection_coefficients

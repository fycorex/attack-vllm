import torch

from proxy_selector.attack import debiased_momentum_pgd_minimize, momentum_pgd_minimize, project_linf
from proxy_selector.schemas import AttackRecipe


def test_projection_obeys_box_and_linf_constraints() -> None:
    clean = torch.full((1, 3, 4, 4), 0.5)
    projected = project_linf(clean + 1.0, clean, epsilon=8.0 / 255.0)
    assert projected.min() >= 0.0 and projected.max() <= 1.0
    assert (projected - clean).abs().max() <= 8.0 / 255.0 + 1e-7


def test_momentum_pgd_minimizes_synthetic_loss() -> None:
    clean = torch.full((1, 1, 2, 2), 0.5)
    target = torch.full_like(clean, 0.45)
    recipe = AttackRecipe(name="test", epsilon=8 / 255, step_size=1 / 255)
    loss = lambda image: ((image - target) ** 2).mean()
    adversarial, history = momentum_pgd_minimize(clean, loss, recipe, steps=6)
    assert history[-1] < history[0]
    assert (adversarial - clean).abs().max() <= recipe.epsilon + 1e-7


def test_debiased_momentum_pgd_is_projected_and_reports_correction() -> None:
    clean = torch.full((1, 3, 2, 2), 0.5)
    target = torch.full_like(clean, 0.45)
    recipe = AttackRecipe(name="debias", epsilon=8 / 255, step_size=1 / 255)
    loss = lambda image: ((image - target) ** 2).mean()
    adversarial, history, coefficients = debiased_momentum_pgd_minimize(
        clean,
        loss,
        recipe,
        dataset_rgb_mean=torch.tensor([0.48, 0.45, 0.41]),
        reference_noise_std=0.01,
        beta=0.5,
        steps=6,
        generator=torch.Generator().manual_seed(7),
    )
    assert len(coefficients) == len(history) == 6
    assert all(value >= 0 for value in coefficients)
    assert torch.isfinite(adversarial).all()
    assert (adversarial - clean).abs().max() <= recipe.epsilon + 1e-7

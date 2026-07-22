import pytest
import torch

from proxy_selector.losses import source_repulsion_loss, symmetric_token_score
from proxy_selector.representations import masked_mean_pool


def test_masked_pool_ignores_invalid_tokens() -> None:
    tokens = torch.tensor([[[1.0, 3.0], [3.0, 7.0], [99.0, 99.0]]])
    pooled = masked_mean_pool(tokens, torch.tensor([[True, True, False]]))
    assert torch.allclose(pooled, torch.tensor([[2.0, 5.0]]))


def test_symmetric_token_score_is_symmetric() -> None:
    left = torch.randn(1, 3, 4)
    right = torch.randn(1, 5, 4)
    assert torch.allclose(symmetric_token_score(left, right), symmetric_token_score(right, left))


def test_source_repulsion_decreases_when_vectors_separate() -> None:
    clean = torch.tensor([[1.0, 0.0]])
    same = torch.tensor([[1.0, 0.0]])
    opposite = torch.tensor([[-1.0, 0.0]])
    assert source_repulsion_loss(opposite, clean) < source_repulsion_loss(same, clean)

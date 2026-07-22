"""Centered linear CKA on common image galleries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CKAInterval:
    value: float
    low: float
    high: float


def _as_features(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("CKA features must be a [n >= 2, d] matrix.")
    if not np.isfinite(value).all():
        raise ValueError("CKA features must be finite.")
    return value


def centered_linear_cka(left: np.ndarray, right: np.ndarray, denominator_eps: float = 1e-12) -> float:
    """Compute CKA through centered Gram matrices; feature widths may differ."""
    x, y = _as_features(left), _as_features(right)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Both feature matrices must have the same ordered gallery length.")
    n = x.shape[0]
    center = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    kx = center @ (x @ x.T) @ center
    ky = center @ (y @ y.T) @ center
    denominator = np.linalg.norm(kx, ord="fro") * np.linalg.norm(ky, ord="fro")
    return float(np.sum(kx * ky) / max(float(denominator), denominator_eps))


def bootstrap_cka(
    left: np.ndarray,
    right: np.ndarray,
    *,
    repetitions: int = 100,
    seed: int = 42,
) -> CKAInterval:
    """Bootstrap images jointly and return a percentile 95% interval."""
    x, y = _as_features(left), _as_features(right)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Both feature matrices must have the same ordered gallery length.")
    generator = np.random.default_rng(seed)
    samples = [
        centered_linear_cka(x[index], y[index])
        for index in (generator.integers(0, x.shape[0], size=x.shape[0]) for _ in range(repetitions))
    ]
    return CKAInterval(
        value=centered_linear_cka(x, y),
        low=float(np.percentile(samples, 2.5)),
        high=float(np.percentile(samples, 97.5)),
    )


def validate_gallery_ids(left: Sequence[str], right: Sequence[str]) -> None:
    """Require exactly the same image identifiers in exactly the same order."""
    if list(left) != list(right):
        raise ValueError("Gallery IDs differ or are not in the same order.")

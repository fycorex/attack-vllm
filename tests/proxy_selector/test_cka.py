import numpy as np
import pytest

from proxy_selector.cka import centered_linear_cka, validate_gallery_ids


def test_cka_invariances_and_different_feature_widths() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(20, 5))
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert centered_linear_cka(features, features) == pytest.approx(1.0)
    assert centered_linear_cka(features, 3.0 * features) == pytest.approx(1.0)
    assert centered_linear_cka(features, features @ rotation) == pytest.approx(1.0)
    assert 0.0 <= centered_linear_cka(features, rng.normal(size=(20, 7))) <= 1.0


def test_cka_requires_ordered_gallery_ids() -> None:
    validate_gallery_ids(["a", "b"], ["a", "b"])
    with pytest.raises(ValueError):
        validate_gallery_ids(["a", "b"], ["b", "a"])
